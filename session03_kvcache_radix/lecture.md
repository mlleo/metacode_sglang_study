# Session 3 이론 — KV Cache 메모리 관리 & RadixAttention

> 50분. §1(메모리 풀의 간접 참조)과 §4(radix tree 손 추적)를 칠판에서 직접 그리며 진행하세요.

| 절 | 내용 | 시간 |
|---|---|---|
| §0 | 지난 회차 연결 | 2분 |
| §1 | KV 메모리를 어떻게 잡을 것인가 — 단편화와 두 단계 풀 | 10분 |
| §2 | page_size라는 다이얼 | 3분 |
| §3 | prefix 공유: 어디에 쓸모가 있나 | 5분 |
| §4 | Radix tree: match / split / insert | 12분 |
| §5 | lock과 eviction | 6분 |
| §6 | Cache-aware 스케줄링 (LPM) | 6분 |
| §7 | 캐시가 깨지는 경우들 | 4분 |
| | 예측 질문 | 2분 |

---

## §0. 지난 회차 연결

Session 1 §6 표에서 오늘 다루는 줄:

| `generate()`의 한계 | 서빙 엔진의 답 |
|---|---|
| KV를 최대 길이만큼 미리 잡는다 → 메모리 낭비 | token 단위 KV pool + 동적 할당 |
| 같은 system prompt를 요청마다 다시 계산 | prefix cache (**RadixAttention**) |

지난 과제에서 "입력 2K일 때 capacity가 줄었다"의 이유 중 하나가 KV 메모리였습니다. 오늘은 그 메모리를 **어떻게 나눠 쓰고, 어떻게 재사용하는지**를 봅니다.

---

## §1. KV 메모리를 어떻게 잡을 것인가

### 순진한 방법: 요청마다 최대 길이만큼 연속 공간

```text
GPU 메모리:  [req A: 4096칸 중 300칸 사용 ··········][req B: 4096칸 중 900칸 ·········][ 빈 공간 ]
```

- **내부 단편화**: 실제 쓰는 건 300칸인데 4096칸을 잡음
- **외부 단편화**: 요청들이 끝나고 생긴 구멍이 제각각이라 큰 요청이 못 들어감
- 출력이 얼마나 길어질지 모르니 "최대 길이"를 예약할 수밖에 없음

### 해결: 작은 조각으로 나누고, 표로 연결한다

운영체제의 가상 메모리와 같은 아이디어입니다. vLLM의 **PagedAttention**(SOSP '23)이 block 단위로 이를 정리했고,
SGLang은 **token 단위(page_size=1 기본)** 로 관리합니다.

```text
① ReqToTokenPool  — "요청 r의 i번째 token은 몇 번 슬롯에 있나?"
   req_to_token[r][i] = slot     (shape: [max_running_requests, context_len], int32)

② TokenToKVPool   — "슬롯 s의 K, V 벡터" (layer마다)
   k_buffer[layer][slot] , v_buffer[layer][slot]      (shape: [num_tokens, kv_heads, head_dim])

③ Allocator       — "지금 비어 있는 슬롯은?"  alloc(n) → 슬롯 n개, free(slots)
```

```text
  요청 A (token 0..4)            요청 B (token 0..3)
  req_to_token[A] = [7, 8, 9, 2, 3]    req_to_token[B] = [7, 8, 5, 6]
                     │  │                                 │  │
                     ▼  ▼                                 ▼  ▼
  k/v_buffer:  slot 0 1 2 3 4 5 6 7 8 9 …
                         ▲ ▲     ▲ ▲ ▲ ▲ ▲
                         A A     B B A,B A,B A          ← 슬롯 7, 8은 A와 B가 공유! (prefix 공유)
```

- 슬롯이 물리적으로 흩어져 있어도 상관없습니다. attention 커널이 `req_to_token` 표를 따라가며 읽습니다 (Session 5에서 커널 쪽을 봅니다).
- 두 요청이 **같은 슬롯 번호를 가리키면** 그게 곧 prefix 공유입니다. 복사가 필요 없습니다.
- 매 iteration 할당량: prefill은 새로 계산할 token 수만큼, decode는 요청당 1칸 (`out_cache_loc`).

서버 시작 로그의 `KV Cache is allocated. #tokens: N`이 ②의 크기이고, 스케줄러는 이 N(`max_total_num_tokens`)을 예산으로 씁니다.
지난 과제의 손계산(112 KiB/token × N ≈ KV size)과 맞는지 오늘 실습에서 확인합니다.

---

## §2. page_size라는 다이얼

| page_size | 장점 | 단점 |
|---|---|---|
| 1 (SGLang 기본) | prefix를 token 단위로 정확히 매칭, 낭비 0 | 표가 크고, 커널이 흩어진 슬롯을 하나씩 따라감 |
| 16, 64 … | 메모리 접근이 덩어리로 모여 빠름, 일부 커널(FA3 등)이 요구 | 매칭이 page 단위로 잘림 (끝자리 버림), 마지막 page 내부 낭비 |

`--page-size 16`이면 공유 prefix가 1000 token이어도 매칭은 992(=16×62)에서 멈춥니다. (과제 Part C 선택 실험)

---

## §3. prefix 공유 — 어디에 쓸모가 있나

```text
system prompt 공유       [SYS ··· 2000 tok][질문 A]      [SYS ··· 2000 tok][질문 B]
few-shot 예시 공유       [예시 8개][문제 1]               [예시 8개][문제 2]
멀티턴 대화              [턴1 질문][턴1 답][턴2 질문]  →  [턴1 질문][턴1 답][턴2 질문][턴2 답][턴3 질문]
에이전트 tool 루프        [도구 설명][관찰1][행동1][관찰2] … 매 스텝 prefix가 길어지기만 함
병렬 샘플링 / ToT         같은 프롬프트에서 N개 분기
```

공통점: **새로 계산해야 할 부분은 짧고, 이미 본 부분은 길다.** prefix 재사용은 TTFT(prefill 시간)를 줄이고, 아낀 GPU 시간만큼 throughput도 늘립니다.
SGLang 논문의 핵심 주장은 이것을 **자동으로**(사용자가 캐시 키를 지정하지 않아도) 해 준다는 것입니다.

---

## §4. Radix tree — match / split / insert

**Radix tree** = 공통 prefix를 한 노드로 압축한 trie. SGLang은 **token 시퀀스를 key**, 그 token들의 **KV 슬롯 번호를 value**로 저장합니다.

```python
class TreeNode:                 # mem_cache/radix_cache.py
    children: dict              # 첫 token(page) → 자식 노드
    parent: TreeNode
    key: RadixKey               # 이 노드가 담당하는 token 조각
    value: Tensor               # 그 token들의 KV 슬롯 번호
    lock_ref: int               # 실행 중인 요청이 쓰고 있으면 > 0 (evict 금지)
    last_access_time: float     # LRU용
```

### 손 추적 예제 (칠판에서 같이)

글자 하나 = token 하나. 요청 4개가 순서대로 끝난다고 합시다.

```text
req1 "ABCDE" 종료 →  root ─ [ABCDE]

req2 "ABCXY"  match("ABCX") : [ABCDE]의 3번째까지 일치 → 노드를 split
              →  root ─ [ABC] ┬ [DE]
                              └ [XY]          cached = 3

req3 "ABCDEZ" match("ABCDE"): [ABC] → [DE] 끝까지 일치 → cached = 5
              →  root ─ [ABC] ┬ [DE] ─ [Z]
                              └ [XY]

req4 "ABQR"   match("ABQ")  : [ABC]의 2번째까지 → 또 split → cached = 2
              →  root ─ [AB] ┬ [C] ┬ [DE] ─ [Z]
                             │     └ [XY]
                             └ [QR]
```

`python session03_kvcache_radix/code/radix_sim.py ABCDE ABCXY ABCDEZ ABQR` 로 그대로 재현됩니다.

### 언제 match하고, 언제 insert하나

| 시점 | 무엇을 | 코드 |
|---|---|---|
| 요청이 prefill 후보가 될 때 | `match_prefix(입력[:-1])` → 재사용할 슬롯 + 마지막 노드 | `Req.init_next_round_input`, `SchedulePolicy` |
| chunked prefill 조각이 끝날 때 | 지금까지 계산한 부분을 insert (긴 프롬프트도 조각마다 공유 가능) | `RadixCache.checkpoint(req, up_to=...)` |
| 요청이 끝났을 때 | `insert(입력 + 출력)` → 다음 요청이 쓸 수 있게 등록, 남는 슬롯은 free, lock 해제 | `checkpoint` → `release_kv_cache` / `on_release` |

> 구버전(~2025) 코드와 블로그 글에서는 같은 역할을 `cache_finished_req` / `cache_unfinished_req`라고 부릅니다. v0.5.23에서는 `checkpoint` 하나로 합쳐졌습니다.

**왜 `입력[:-1]`인가?** 다음 token을 뽑으려면 마지막 위치의 logits가 필요하고, 그러려면 마지막 token은 반드시 forward해야 합니다.
그래서 완전히 똑같은 요청을 다시 보내도 `cached = prompt − 1`입니다 (`_compute_max_prefix_len`: `max_prefix_len = input_len - 1`).

**출력도 tree에 들어간다.** 요청이 끝나면 생성한 token의 KV도 insert되므로, 멀티턴의 "이전 답변"까지 재사용됩니다.

---

## §5. lock과 eviction

메모리가 모자라면 tree에서 지워야 합니다. 단, **지금 실행 중인 요청이 쓰는 KV를 지우면 안 됩니다.**

- 요청이 prefix를 쓰기 시작하면 그 경로의 모든 노드 `lock_ref += 1` (`inc_lock_ref`), 끝나면 `-= 1`
- `lock_ref == 0`인 노드의 token 수 = **evictable**, 아니면 **protected**
- 스케줄러가 보는 "쓸 수 있는 메모리" = `free 슬롯 + evictable` → 캐시는 "공짜로 빌려 쓴 메모리"입니다. 필요하면 언제든 돌려받습니다.
- `evict(n)`: lock이 없는 **leaf**만 후보, 우선순위(기본 LRU = `last_access_time`이 오래된 것)로 heap에서 꺼내 지움. leaf를 지운 뒤 부모가 leaf가 되면 부모도 후보에 들어감

```text
root ─ [SYS 2000] ┬ [Q1 + 답 100]  t=10, lock 0   ← 가장 오래됨 → 먼저 지워짐
                  ├ [Q2 + 답 100]  t=25, lock 0
                  └ [Q3 ···]      t=30, lock 1   ← 실행 중: 보호
```

prefix(SYS)는 자식이 남아 있는 한 leaf가 아니라서 **가장 늦게** 지워집니다. 많이 공유되는 것이 오래 살아남는 구조입니다.
`--radix-eviction-policy`로 LRU 외에 LFU, priority 등도 고를 수 있습니다 (`mem_cache/evict_policy.py`).

---

## §6. Cache-aware 스케줄링 — LPM

대기열에 [A그룹, B그룹, C그룹, A, B, C, …] 순으로 섞여 있고, 메모리에는 그룹 하나의 prefix만 들어간다면?

| 정책 | 동작 | 결과 |
|---|---|---|
| **FCFS** | 도착 순서대로 | A 계산 → B가 A를 밀어냄 → C가 B를 밀어냄 … 매번 cache miss (thrashing) |
| **LPM** (Longest Prefix Match) | 대기열을 "캐시에 이미 있는 prefix가 긴 순서"로 정렬 | 같은 그룹끼리 몰아서 처리 → hit ↑ |

`schedule_policy.py`의 세부 사항:

- **v0.5.23 기본값은 `fcfs`** 입니다 (`--schedule-policy lpm`으로 켬). LPM은 매 iteration 대기열 전체를 tree에 match해야 해서 CPU 비용이 있습니다.
- 대기열이 **128개를 넘으면** LPM도 FCFS로 바뀝니다 (`WAITING_QUEUE_PREFIX_MATCH_MAX = 128`) — 정렬 비용이 이득보다 커지기 때문.
- **In-batch prefix caching**: 같은 batch에 공유 prefix를 가진 새 요청이 여러 개 있으면, 하나만 먼저 계산하고 나머지는 **다음 iteration으로 미룹니다.** 그래야 나머지가 먼저 계산된 KV를 재사용합니다.
- LPM의 대가: cache miss인 요청은 계속 뒤로 밀릴 수 있습니다 (**starvation** → tail TTFT 증가).

---

## §7. 캐시가 깨지는 경우들

| 상황 | 왜 | 대응 |
|---|---|---|
| 프롬프트 앞부분에 매번 바뀌는 값 (현재 시각, 요청 ID, 랜덤 순서의 문서) | prefix가 첫 token부터 달라짐 | **변하는 값은 뒤로** 보내도록 프롬프트 설계 |
| 멀티턴을 텍스트로 이어 붙여 재토크나이즈 | token 경계가 원래 생성과 달라질 수 있음 | token id 그대로 유지, 또는 template이 이전 턴을 똑같이 렌더링하는지 확인 |
| chat template이 이전 턴을 바꿔 렌더링 (예: 추론 블록 제거) | 과거 대화의 token 시퀀스 자체가 바뀜 | 실습/과제에서 직접 확인 |
| `return_logprob`로 입력 logprob 요청 | logprob를 계산할 위치부터는 재사용 불가 (`logprob_start_len`) | 필요한 경우만 |
| LoRA 어댑터가 다름 | 같은 token이라도 KV가 다름 → `RadixKey.extra_key`로 네임스페이스 분리 | 정상 동작 |
| 메모리 부족으로 계속 evict | 재사용 전에 지워짐 | 동시성 조절, LPM, KV 양자화(Session 5), HiCache(Session 6) |

---

## 코드 읽기 (실습 후 과제에서)

[docs/code_map.md](../docs/code_map.md) Session 3 표. 순서:

1. `mem_cache/README.md` — 공식 레이어 설명 (allocator / pool / radix cache)
2. `mem_cache/radix_cache.py` — `match_prefix` → `_match_prefix_helper` → `_split_node` → `insert` → `_insert_helper` → `evict`
3. `mem_cache/registry.py`의 `default_radix_cache_factory` — **v0.5.23에서 실제로 쓰이는 클래스**가 무엇인지 확인 (힌트: `UnifiedRadixCache`)

> 교과서 구현(`radix_cache.py`, 830줄)과 실제 기본 경로(`unified_radix_cache.py` + `unified_cache/`, 4000줄 이상)가 다르다는 것 자체가 오늘의 교훈 중 하나입니다.
> 실제 엔진 코드는 SWA, Mamba, 계층형 캐시까지 한 틀로 다루느라 일반화되어 있습니다. **개념은 작은 구현으로, 사실 확인은 실제 경로로.**

## 예측 질문

1. 공유 prefix가 4096 token, 고유 suffix 64 token일 때 warm 요청의 TTFT는 cold 대비 몇 %일까?
2. 같은 요청을 두 번 보내면 두 번째 `cached_tokens`는 정확히 몇일까?
3. KV pool이 24000 token이고 prefix 2048짜리 그룹 16개가 섞여 들어오면, FCFS의 cache hit rate는 대략 얼마일까? LPM은?
4. 서버 로그 `KV Cache is allocated`의 KV size(GB) ÷ #tokens 는 몇 KiB일까?
