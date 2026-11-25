# Session 3 이론 — KV Cache 메모리 관리 & RadixAttention

> **핵심 질문:** KV 메모리를 단편화 없이 나눠 쓰고, 이미 계산한 prompt를 다시 계산하지 않으려면?

## 이 문서 사용법

- ★ 수업 핵심 / 📖 복습·심화
- 코드 발췌는 **SGLang v0.5.23** 기준이며 간추린 것입니다 (원문 위치 표기). 전체 지도는 [docs/code_map.md](../docs/code_map.md)
**수업 40분**

- 8분 · §1.3에서 두 요청이 공유하는 슬롯을 찾습니다.
- 15분 · §4.4에서 두 번째 요청까지 함께 그립니다. 세 번째와 네 번째 요청은 학생이 그린 뒤 시뮬레이터로 확인합니다.
- 10분 · §4.7과 §5에서 마지막 token의 재계산과 lock이 걸린 경로의 eviction을 설명합니다.
- 7분 · 같은 프롬프트를 다시 보낼 때 `cached_tokens`를 예측합니다. §9의 런타임 경로를 짚고 실습으로 넘어갑니다.

FCFS/LPM과 in-batch prefix caching은 §6에서 읽고 선택 실습의 로그와 함께 봅니다.

## 목차

0. ★ 지난 회차 연결
1. ★ KV 메모리를 어떻게 잡을 것인가 — 단편화와 두 단계 풀
2. 📖 page_size라는 다이얼
3. ★ prefix 공유: 어디에 쓸모가 있나
4. ★ Radix tree: 자료구조와 match / split / insert
5. ★ lock과 eviction — 캐시는 빌린 메모리
6. ★ Cache-aware 스케줄링 (LPM, in-batch prefix caching)
7. ★ 캐시가 깨지는 경우들
8. 📖 다른 설계와의 비교: hash 기반 block prefix cache
9. 📖 v0.5.23의 실제 실행 경로: UnifiedRadixCache
10. 요약 · 자기 점검 · 더 읽을거리

---

## 0. ★ 지난 회차 연결

Session 1 §8 표에서 오늘 다루는 줄:

| `generate()`의 한계 | 서빙 엔진의 답 |
|---|---|
| KV를 최대 길이만큼 미리 잡는다 → 메모리 낭비 | token 단위 KV pool + 동적 할당 |
| 같은 system prompt를 요청마다 다시 계산 | prefix cache (**RadixAttention**) |

Session 2 §4.2에서 "continuous batching을 하려면 KV를 token 단위로 그때그때 할당해야 한다"고 했습니다. 지난 과제의 "입력 2K → capacity 감소"의 한 원인도 KV 메모리였습니다.
오늘은 그 메모리를 **어떻게 나눠 쓰고, 어떻게 재사용하는지**를 봅니다.

---

## 1. ★ KV 메모리를 어떻게 잡을 것인가

### 1.1 순진한 방법: 요청마다 최대 길이만큼 연속 공간

```text
GPU 메모리:  [req A: 4096칸 중 300칸 사용 ··········][req B: 4096칸 중 900칸 ·········][ 빈 공간 ]
```

- **내부 단편화**: 실제 쓰는 건 300칸인데 4096칸을 잡음. 출력이 얼마나 길어질지 모르니 최대 길이를 예약할 수밖에 없음
- **외부 단편화**: 요청들이 끝나고 생긴 구멍의 크기가 제각각이라, 빈 공간을 다 합치면 충분해도 큰 요청이 못 들어감
- **공유 불가**: A와 B가 같은 system prompt를 써도 각자 자기 공간에 따로 저장

vLLM 논문(PagedAttention)은 이런 방식에서 KV 메모리의 대부분이 낭비된다고 측정했습니다.

### 1.2 해결: 작은 조각으로 나누고, 표로 연결한다

운영체제의 가상 메모리(페이지 테이블)와 같은 아이디어입니다. SGLang은 **세 개의 층**으로 나눕니다.

```text
① ReqToTokenPool  — "요청 r의 i번째 token은 몇 번 슬롯에 있나?"   (페이지 테이블)
② TokenToKVPool   — "슬롯 s의 K, V 벡터" (layer마다)               (물리 메모리)
③ Allocator       — "지금 비어 있는 슬롯은?" alloc(n) / free(slots)  (빈 페이지 목록)
```

v0.5.23 코드에서 실제 모습:

```python
# mem_cache/memory_pool.py: ReqToTokenPool  (간추림)
class ReqToTokenPool:
    def __init__(self, size, max_context_len, device, ...):
        # 행 = 요청 슬롯, 열 = 그 요청의 i번째 token → 값 = KV 슬롯 번호
        self.req_to_token = torch.zeros((size + 1, max_context_len), dtype=torch.int32, device=device)
        self.free_slots = list(range(1, size + 1))     # 행 0은 예비
```

```python
# mem_cache/memory_pool.py: MHATokenToKVPool  (간추림 — 표준 NHD 레이아웃)
rows = self.size + self.page_size                  # 슬롯 0(첫 page)은 padding용 더미 쓰기 자리
k_shape = (rows, self.head_num, self.head_dim)     # Qwen3-1.7B: (N+1, 8, 128)
self.k_buffer = [torch.zeros(k_shape, dtype=bf16) for _ in range(self.layer_num)]   # layer 28개
self.v_buffer = [torch.zeros(k_shape, dtype=bf16) for _ in range(self.layer_num)]
```

```python
# mem_cache/allocator/token.py: TokenToKVPoolAllocator  (page_size=1, 간추림)
def clear(self):
    self.free_pages = torch.arange(1, self.size + 1, device=self.device)   # 슬롯 0 제외
def alloc(self, need_size):
    if need_size > len(self.free_pages):
        return None                                  # 모자라면 None → 호출자가 evict/대기 결정
    select_index = self.free_pages[:need_size]
    self.free_pages = self.free_pages[need_size:]
    return select_index
def free(self, free_index):
    self.free_pages = torch.cat((self.free_pages, free_index))
```

관찰:

- 빈 슬롯 목록은 **GPU 텐서**입니다. 할당 결과(슬롯 번호)를 그대로 커널에 넘길 수 있도록.
- `k_buffer[layer]`의 크기 = `#tokens × 8 × 128 × 2 B` = 2 KiB/token/layer. × 28 layers × (K, V) = **112 KiB/token** — Session 1의 공식과 정확히 같은 구조입니다.
- 시작 시 이 버퍼를 **한 번에 전부** 잡습니다. 서버가 GPU 메모리를 거의 다 쓰는 것처럼 보이는 이유입니다 (`nvidia-smi`로 보면 요청이 없어도 꽉 차 있음).

### 1.3 그림으로: 간접 참조와 공유

```text
  요청 A (token 0..4)                    요청 B (token 0..3)
  req_to_token[A] = [7, 8, 9, 2, 3]      req_to_token[B] = [7, 8, 5, 6]
                     │  │                                  │  │
                     ▼  ▼                                  ▼  ▼
  k/v_buffer:  slot 0 1 2 3 4 5 6 7 8 9 …
                         ▲ ▲     ▲ ▲ ▲ ▲ ▲
                         A A     B B A,B A,B A          ← 슬롯 7, 8은 A와 B가 공유! (prefix 공유)
```

- 슬롯이 물리적으로 흩어져 있어도 상관없습니다. attention 커널이 `req_to_token` 표를 따라가며 읽습니다 (Session 5 §2).
- 두 요청이 **같은 슬롯 번호를 가리키면** 그게 곧 prefix 공유입니다. 복사가 필요 없습니다.
- 단편화가 없습니다: 어떤 빈 슬롯이든 어떤 요청에 줄 수 있습니다.

### 1.4 한 iteration의 할당량 (📖)

| batch 종류 | 새로 할당 | `ForwardBatch` 필드 |
|---|---|---|
| prefill (EXTEND) | 요청마다 (입력 길이 − cache hit 길이)개 | `out_cache_loc` (이번에 새 K, V를 써 넣을 슬롯들) |
| decode | 요청마다 1개 | `out_cache_loc` (요청당 1개) |

forward 중 각 layer는 새 K, V를 `out_cache_loc` 위치에 써 넣고(`set_kv_buffer`), attention은 `req_to_token`을 따라 그 요청의 모든 K, V를 읽습니다.

서버 시작 로그의 `KV Cache is allocated. #tokens: N, KV size: X GB`가 ②의 크기이고, 스케줄러는 이 N(`max_total_num_tokens`)을 예산으로 씁니다.

---

## 2. 📖 page_size라는 다이얼

| page_size | 장점 | 단점 |
|---|---|---|
| 1 (SGLang 기본) | prefix를 token 단위로 정확히 매칭, 낭비 0 | 표가 크고, 커널이 흩어진 슬롯을 하나씩 따라감 |
| 16, 64 … | 메모리 접근이 덩어리로 모여 효율↑, 일부 커널(FA3 등)이 요구 | 매칭이 page 단위로 잘림 (끝자리 버림), 마지막 page 내부 낭비 |

- `--page-size 16`이면 공유 prefix가 1000 token이어도 매칭은 992(=16×62)에서 멈춥니다 (`RadixKey.page_aligned`).
- vLLM은 block(=page) 16이 기본입니다. "token 단위 정밀 매칭"과 "덩어리 효율" 사이의 선택입니다.
- page_size가 1보다 크면 radix tree의 자식 키도 "첫 token" 대신 "첫 page"가 됩니다 (`RadixKey.child_key(page_size)`).

---

## 3. ★ prefix 공유 — 어디에 쓸모가 있나

```text
system prompt 공유       [SYS ··· 2000 tok][질문 A]      [SYS ··· 2000 tok][질문 B]
few-shot 예시 공유       [예시 8개][문제 1]               [예시 8개][문제 2]
멀티턴 대화              [턴1 질문][턴1 답][턴2 질문]  →  [턴1 질문][턴1 답][턴2 질문][턴2 답][턴3 질문]
에이전트 tool 루프        [도구 설명][관찰1][행동1][관찰2] … 매 스텝 prefix가 길어지기만 함
병렬 샘플링 / ToT         같은 프롬프트에서 N개 분기
RAG                      [같은 문서 묶음][서로 다른 질문]
```

공통점: **새로 계산해야 할 부분은 짧고, 이미 본 부분은 길다.**

효과를 숫자로 (Qwen3-1.7B, 4090 기준 대략치):

```text
prefix 2000 + 질문 50, 출력 200
  cache 없음: prefill 2050 token  → TTFT ≈ 수십 ms
  cache 있음: prefill   50 token  → TTFT ≈ 한 자릿수~십여 ms   (+ decode는 그대로)
```

- prefix 재사용은 **TTFT(prefill 시간)** 를 줄이고, 아낀 GPU 시간만큼 throughput도 늘립니다.
- 하지만 **decode는 줄지 않습니다.** 출력이 긴 workload에서는 throughput 개선이 TTFT 개선보다 작습니다 (Session 3 과제 Part B의 핵심 관찰).
- 그리고 **KV 메모리도 아낍니다**: 100명이 같은 2000 token prefix를 쓰면 저장은 한 번.

SGLang 논문의 핵심 주장은 이것을 **자동으로** — 사용자가 캐시 키를 지정하지 않아도 — 해 준다는 것입니다.
(OpenAI·Anthropic API의 "prompt caching"도 같은 아이디어의 상용 버전입니다.)

---

## 4. ★ Radix tree — 자료구조와 match / split / insert

### 4.1 자료구조

**Radix tree** = 공통 prefix를 한 노드로 압축한 trie. SGLang은 **token 시퀀스를 key**, 그 token들의 **KV 슬롯 번호를 value**로 저장합니다.

```python
# mem_cache/radix_cache.py  (v0.5.23, 간추림)
class TreeNode:
    def __init__(self):
        self.children = defaultdict(TreeNode)   # 자식 키(첫 token 또는 첫 page) → 자식 노드
        self.parent: TreeNode = None
        self.key: RadixKey = None               # 이 노드가 담당하는 token 조각
        self.value: torch.Tensor = None         # 그 token들의 KV 슬롯 번호 (같은 길이)
        self.lock_ref = 0                       # 실행 중인 요청이 쓰고 있으면 > 0 (evict 금지)
        self.last_access_time = time.monotonic()  # LRU용
        self.hit_count = 0                      # LFU 등에서 사용

    @property
    def evicted(self):
        return self.value is None
```

`RadixKey`는 token id 배열 + `extra_key`(네임스페이스: LoRA ID, cache salt 등)입니다. `extra_key`가 다르면 같은 token이라도 절대 공유하지 않습니다.

루트에서 노드까지 내려가며 `value`를 이어 붙이면 그 경로(prefix)의 KV 슬롯 목록이 됩니다 (`path_device_indices`).

### 4.2 match — 최장 공통 prefix 찾기

```python
# mem_cache/radix_cache.py: _match_prefix_helper  (거의 원문)
def _match_prefix_helper(self, node, key):
    node.last_access_time = now
    child_key = key.child_key(self.page_size)          # 첫 token(page)
    value = []
    while len(key) > 0 and child_key in node.children:
        child = node.children[child_key]
        child.last_access_time = now
        prefix_len = child.key.match(key, page_size=self.page_size)   # child.key와 key의 공통 길이
        if prefix_len < len(child.key):                # 노드 '중간'에서 갈라짐
            new_node = self._split_node(child.key, child, prefix_len)
            value.append(new_node.value)
            node = new_node
            break
        else:                                          # 노드를 통째로 지나감
            value.append(child.value)
            node = child
            key = key[prefix_len:]
            if len(key):
                child_key = key.child_key(self.page_size)
    return value, node                                 # 재사용할 슬롯들, 마지막 노드
```

- 자식을 고르는 데 **첫 token 하나**만 봅니다 (dict 조회, O(1)). radix tree에서는 같은 부모의 자식들은 첫 token이 서로 다르기 때문입니다.
- match 도중 노드 중간에서 갈라지면 **match 자체가 tree를 바꿉니다** (split). "읽기"가 "쓰기"를 하는 셈이라, 이 함수는 스케줄러 프로세스 안에서만 불립니다.
- 지나간 노드들의 `last_access_time`을 갱신 → LRU에서 오래 살아남음.

### 4.3 split — 노드를 둘로 나누기

```python
# mem_cache/radix_cache.py: _split_node  (간추림)
def _split_node(self, key, child, split_len):
    # 전:  parent ─ child[ABCDE]            후:  parent ─ new[ABC] ─ child[DE]
    new_node = TreeNode()
    new_node.children = {key[split_len:].child_key(): child}
    new_node.parent = child.parent
    new_node.lock_ref = child.lock_ref          # ★ child를 쓰던 요청은 new_node도 쓰고 있었던 것
    new_node.key = child.key[:split_len]
    new_node.value = child.value[:split_len].clone()
    child.parent = new_node
    child.key = child.key[split_len:]
    child.value = child.value[split_len:].clone()
    new_node.parent.children[key.child_key()] = new_node
    return new_node
```

KV 자체는 움직이지 않습니다. **슬롯 번호 배열만** 둘로 자릅니다.

### 4.4 손 추적 예제 (칠판에서 같이)

글자 하나 = token 하나. 요청 4개가 순서대로 끝난다고 합시다. 슬롯 번호도 같이 적어 봅니다.

```text
req1 "ABCDE" 종료 →  root ─ [ABCDE] slots 0-4

req2 "ABCXY"  match("ABCX") : [ABCDE]의 3번째까지 일치 → split
              →  root ─ [ABC] 0-2 ┬ [DE] 3-4
                                  └ [XY] 5-6          cached = 3, 새 할당 2

req3 "ABCDEZ" match("ABCDE"): [ABC] → [DE] 끝까지 일치 → cached = 5, 새 할당 1
              →  root ─ [ABC] ┬ [DE] ─ [Z] 7
                              └ [XY]

req4 "ABQR"   match("ABQ")  : [ABC]의 2번째까지 → 또 split → cached = 2, 새 할당 2
              →  root ─ [AB] 0-1 ┬ [C] 2 ┬ [DE] ─ [Z]
                                 │       └ [XY]
                                 └ [QR] 8-9
```

`python3 session03_kvcache_radix/code/radix_sim.py ABCDE ABCXY ABCDEZ ABQR`로 그대로 재현됩니다.

### 4.5 insert — 새 KV를 tree에 등록하기

```python
# mem_cache/radix_cache.py: _insert_helper  (간추림)
def _insert_helper(self, node, key, value):
    child_key = key.child_key()
    total_prefix_length = 0
    while len(key) > 0 and child_key in node.children:     # 이미 있는 부분은 따라 내려가며
        node = node.children[child_key]
        prefix_len = node.key.match(key)
        total_prefix_length += prefix_len
        key, value = key[prefix_len:], value[prefix_len:]
        if prefix_len < len(node.key):
            node = self._split_node(node.key, node, prefix_len)
        if len(key):
            child_key = key.child_key()
    if len(key):                                             # 남은 부분을 새 leaf로
        new_node = TreeNode()
        new_node.parent, new_node.key, new_node.value = node, key, value.clone()
        node.children[child_key] = new_node
        self.evictable_size_ += len(key)
    return total_prefix_length, node     # 이미 tree에 있던 길이 → 호출자가 중복 슬롯을 free
```

`total_prefix_length`가 중요합니다: 두 요청이 **동시에** 같은 prefix를 계산했다면 두 번째 insert에서 그 부분은 이미 tree에 있습니다. 두 번째 요청이 계산한 중복 슬롯은 tree에 들어가지 않으므로 **호출자가 free**해야 합니다.

### 4.6 언제 match하고, 언제 insert하나

| 시점 | 무엇을 | 코드 |
|---|---|---|
| 요청이 prefill 후보가 될 때 | `match_prefix(입력[:-1])` → 재사용할 슬롯 + 마지막 노드 | `Req.init_next_round_input`, `SchedulePolicy` |
| chunked prefill 조각이 끝날 때 | 지금까지 계산한 부분을 insert (긴 프롬프트도 조각마다 공유 가능) | `RadixCache.checkpoint(req, up_to=...)` |
| 요청이 끝났을 때 | `insert(이미 KV를 계산한 입력 + 출력)` → 다음 요청이 쓸 수 있게 등록, 남는 슬롯은 free, lock 해제 | `checkpoint` → `release_kv_cache` / `on_release` |

> 구버전(~2025) 코드와 블로그 글에서는 같은 역할을 `cache_finished_req` / `cache_unfinished_req`라고 부릅니다. v0.5.23에서는 `checkpoint` 하나로 합쳐졌습니다.

```python
# mem_cache/radix_cache.py: checkpoint  (간추림)
def checkpoint(self, req, *, up_to):
    token_ids = req.full_untruncated_fill_ids[:up_to]          # 입력 + (지금까지의) 출력
    radix_key, _, _ = self._insert_cache(req, token_ids, ...)   # tree에 등록 (중복 슬롯은 정리)
    match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
    new_last_node = match_result.last_device_node
    new_indices = self.path_device_indices(new_last_node)
    # 요청의 슬롯 표를 tree가 가진 슬롯으로 다시 연결 (중복으로 계산한 부분이 tree 쪽 슬롯을 가리키도록)
    self.req_to_token_pool.write((req.kv.req_pool_idx, slice(req.kv.cache_protected_len, len(new_indices))),
                                 new_indices[req.kv.cache_protected_len:])
    req.kv.cache_protected_len = len(new_indices)
    self.unlock(req.lock)                     # 옛 경로 lock 해제
    req.lock = self.lock(new_last_node)       # 새 경로 lock
    req.last_node = new_last_node
```

### 4.7 왜 `입력[:-1]`인가

```python
# managers/schedule_batch.py  (간추림)
def _compute_max_prefix_len(self, input_len):
    max_prefix_len = input_len - 1                     # 마지막 token은 반드시 forward
    if self.return_logprob:
        max_prefix_len = min(max_prefix_len, self.logprob_start_len)   # logprob를 낼 위치부터는 재계산
    return max(max_prefix_len, 0)
```

다음 token을 뽑으려면 마지막 위치의 logits가 필요하고, 그러려면 마지막 token은 반드시 forward해야 합니다.
전체 prefix가 캐시에 남아 있고, page size가 1이며 입력 logprob를 요청하지 않았다면 완전히 같은 요청의 `cached_tokens`는 `prompt_tokens − 1`입니다. 실습 `prefix_basics.py`의 3번이 이 조건입니다.

**출력도 tree에 들어간다.** 요청이 끝나면 생성한 token의 KV도 insert되므로, 멀티턴의 "이전 답변"까지 재사용됩니다.
(마지막으로 생성된 token 하나는 아직 forward되지 않아 KV가 없으므로 들어가지 않습니다.)

---

## 5. ★ lock과 eviction — 캐시는 빌린 메모리

### 5.1 lock

메모리가 모자라면 tree에서 지워야 합니다. 단, **지금 실행 중인 요청이 쓰는 KV를 지우면 안 됩니다.**

```python
# mem_cache/radix_cache.py: inc_lock_ref  (간추림)
def inc_lock_ref(self, node):
    while node != self.root_node:           # 마지막 노드부터 루트까지 경로 전체를 잠근다
        if node.lock_ref == 0:
            self.evictable_size_ -= len(node.key)
            self.protected_size_ += len(node.key)
        node.lock_ref += 1
        node = node.parent
```

- `lock_ref == 0`인 노드의 token 수 합 = **evictable**, 아니면 **protected**
- 스케줄러가 보는 "쓸 수 있는 메모리" = `free 슬롯 + evictable` (Session 4 §2의 `rem_total_tokens`)
- 즉 캐시는 **"공짜로 빌려 쓴 메모리"** 입니다. 다른 요청이 메모리가 필요하면 언제든 돌려받습니다. 그래서 radix cache를 켜 두는 비용은 거의 0입니다 (과제 Part B에서 확인).

### 5.2 eviction

```python
# mem_cache/radix_cache.py: evict  (간추림)
def evict(self, params):
    leaves = list(self.evictable_leaves)               # lock 0 이고 자식 없는 노드들
    heap = [(self.eviction_strategy.get_priority(n), n) for n in leaves]
    heapq.heapify(heap)                                # LRU면 last_access_time이 작은(오래된) 것이 먼저
    num_evicted = 0
    while num_evicted < params.num_tokens and heap:
        _, x = heapq.heappop(heap)
        self.token_to_kv_pool_allocator.free_segment(x.value, start_pos=0)   # 슬롯 반납
        num_evicted += len(x.value)
        self._delete_leaf(x)
        if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
            heapq.heappush(heap, (self.eviction_strategy.get_priority(x.parent), x.parent))  # 부모가 새 leaf
    return EvictResult(num_tokens_evicted=num_evicted)
```

```text
root ─ [SYS 2000] ┬ [Q1 + 답 100]  t=10, lock 0   ← 가장 오래됨 → 먼저 지워짐
                  ├ [Q2 + 답 100]  t=25, lock 0
                  └ [Q3 ···]      t=30, lock 1   ← 실행 중: 보호
```

- **leaf부터** 지웁니다. prefix(SYS)는 자식이 남아 있는 한 leaf가 아니라서 **가장 늦게** 지워집니다 → 많이 공유되는 것이 오래 살아남는 구조.
- `--radix-eviction-policy`로 LRU 외에 LFU, FIFO, priority, SLRU, TLRU 등을 고를 수 있습니다 (`mem_cache/evict_policy.py`).
  - LRU: 최근성. 일반적인 대화형 트래픽
  - LFU: 빈도. "몇 개의 인기 system prompt + 많은 일회성 요청"이 섞이면 유리
- eviction은 할당이 실패할 것 같을 때(`alloc` 전에 필요량만큼) 일어납니다.

### 5.3 숫자 예제: 메모리 회계 (📖)

KV pool 10,000 슬롯. 현재 tree에 6,000 token, 그중 실행 중 요청이 lock한 것이 2,500. 빈 슬롯 4,000.

```text
evictable = 6,000 − 2,500 = 3,500
쓸 수 있는 메모리 = free 4,000 + evictable 3,500 = 7,500

새 요청: 입력 5,000 (cache hit 0) → 5,000 ≤ 7,500 → 받을 수 있음
  할당 시: free 4,000을 먼저 쓰고, 부족한 1,000은 LRU leaf를 evict해서 확보
```

---

## 6. ★ Cache-aware 스케줄링

### 6.1 LPM — Longest Prefix Match

대기열에 [A그룹, B그룹, C그룹, A, B, C, …] 순으로 섞여 있고, 메모리에는 그룹 하나의 prefix만 들어간다면?

| 정책 | 동작 | 결과 |
|---|---|---|
| **FCFS** | 도착 순서대로 | A 계산 → B가 A를 밀어냄 → C가 B를 밀어냄 … 매번 cache miss (thrashing) |
| **LPM** | 대기열을 "캐시에 이미 있는 prefix가 긴 순서"로 정렬 | 같은 그룹끼리 몰아서 처리 → hit ↑ |

```python
# managers/schedule_policy.py  (간추림)
def calc_priority(self, waiting_queue, ...):
    policy = self._determine_active_policy(waiting_queue)
    if self.policy == CacheAgnosticPolicy.FCFS:
        pass                                                       # 도착 순서 그대로 (priority 옵션 제외)
    elif isinstance(policy, CacheAwarePolicy):
        deprioritized = self._compute_prefix_matches(waiting_queue, policy)   # 대기 요청 전부를 tree에 match (§6.2)
        if policy == CacheAwarePolicy.LPM:
            waiting_queue.sort(key=lambda r: -r.num_matched_prefix_tokens     # 긴 hit 먼저
                               if r.rid not in deprioritized else float("inf"))  # 미룬 요청은 맨 뒤
    ...

def _determine_active_policy(self, waiting_queue):
    if self.policy in (LPM, HRRN) and len(waiting_queue) > WAITING_QUEUE_PREFIX_MATCH_MAX:   # 128
        return CacheAgnosticPolicy.FCFS     # 대기열이 너무 길면 정렬 비용 > 이득 → FCFS로 전환
    return self.policy
```

- **v0.5.23 기본값은 `fcfs`** 입니다. `--schedule-policy lpm`으로 켭니다. (예전 버전과 블로그 글은 LPM이 기본이라고 적혀 있는 경우가 많으니 주의)
- 다른 정책들: `dfs-weight`(tree를 DFS 순서로 훑어 같은 subtree끼리), `lof`(longest output first), `priority`, `hrrn`(대기 시간 대비 처리 시간), `shortest-prefill-first`, `random` …
- LPM의 대가: cache miss인 요청은 계속 뒤로 밀릴 수 있습니다 (**starvation** → tail TTFT 증가). 실습 5번에서 p99 TTFT로 확인합니다.

### 6.2 In-batch prefix caching

같은 iteration에 **공유 prefix를 가진 새 요청 여러 개**가 들어왔는데, 그 prefix가 아직 tree에 없다면?
전부 한 batch로 prefill하면 **같은 prefix를 batch 안에서 여러 번** 계산합니다 (서로의 결과를 아직 못 보므로).

```python
# managers/schedule_policy.py: _compute_prefix_matches  (간추림)
temporary_deprioritized = set()
self.waiting_queue_radix_tree.reset()                   # 대기열 전용 '가짜' radix tree
for r in waiting_queue:
    match = match_kv_cache(r, self.tree_cache, r.origin_input_ids + r.output_ids)   # 진짜 tree와 match
    if r.prefix_len <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:          # 진짜 tree hit가 짧은 요청만
        m = self.waiting_queue_radix_tree.match_prefix(...)              # 대기열 안의 다른 요청과 공유?
        if m.device_prefix_len >= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD:   # 32 token 이상 공유
            temporary_deprioritized.add(r.rid)                           # → 이번 iteration엔 미룸
        else:
            self.waiting_queue_radix_tree.insert(...)                    # 첫 요청은 대표로 등록
return temporary_deprioritized
```

공유 prefix 후보를 뒤로 미루면 먼저 계산한 요청의 KV가 다음 iteration에 쓰일 기회가 생깁니다. 실제 재사용량은 admission 순서와 그때 남은 슬롯에 따라 달라집니다.

### 6.3 정책 선택 가이드 (📖)

| workload | 추천 |
|---|---|
| prefix 공유가 거의 없음 | `fcfs` (LPM 정렬 비용만 듦) |
| 공유 많음 + 메모리 넉넉 | `fcfs`로도 hit가 잘 남 (안 지워지니까) |
| 공유 많음 + 메모리 빠듯 + 대기열 존재 | `lpm` 또는 `dfs-weight` |
| tail latency가 최우선 | `fcfs` (공정성) |

---

## 7. ★ 캐시가 깨지는 경우들

| 상황 | 왜 | 대응 |
|---|---|---|
| 프롬프트 앞부분에 매번 바뀌는 값 (현재 시각, 요청 ID, 사용자 이름, 랜덤 순서의 문서) | prefix가 첫 token부터 달라짐 | **변하는 값은 뒤로** 보내도록 프롬프트 설계 |
| 멀티턴을 텍스트로 이어 붙여 재토크나이즈 | token 경계가 원래 생성과 달라질 수 있음 | token id 그대로 유지, 또는 template이 이전 턴을 똑같이 렌더링하는지 확인 |
| chat template이 이전 턴을 바꿔 렌더링 (예: 추론 블록 제거) | 과거 대화의 token 시퀀스 자체가 바뀜 | 실습/과제에서 직접 확인 |
| `return_logprob`로 입력 logprob 요청 | logprob를 계산할 위치부터는 재사용 불가 (`logprob_start_len`) | 필요한 경우만 |
| LoRA 어댑터가 다름 | 같은 token이라도 KV가 다름 → `extra_key`로 네임스페이스 분리 | 정상 동작 |
| 메모리 부족으로 계속 evict | 재사용 전에 지워짐 | 동시성 조절, LPM, KV 양자화(Session 5), HiCache(Session 6) |
| DP(서버 여러 개)에서 랜덤 라우팅 | 같은 prefix가 다른 서버로 감 | prefix-aware 라우터 (Session 6 §5) |

**프롬프트 설계 예:**

```text
❌  "오늘 날짜: 2026-11-22 14:03:11\n당신은 친절한 상담원입니다 … (2000 token 규칙) … 질문: …"
✅  "당신은 친절한 상담원입니다 … (2000 token 규칙) …\n오늘 날짜: 2026-11-22\n질문: …"
```

---

## 8. 📖 다른 설계와의 비교: hash 기반 block prefix cache

vLLM 등은 radix tree 대신 **block 단위 hash**로 prefix를 찾습니다.

```text
block 크기 16. 요청의 token을 16개씩 자르고, 각 block의 hash = hash(이전 block hash, 이 block의 token들)
→ hash 표에 있으면 그 block의 KV를 재사용
```

| | Radix tree (SGLang) | Block hash (vLLM 등) |
|---|---|---|
| 매칭 단위 | token (page_size=1) | block (예: 16 token) |
| 찾는 비용 | tree 경로 따라가기 | block 수만큼 hash 조회 |
| 구조 정보 | tree 모양 자체가 공유 관계를 표현 → LPM, DFS 같은 정책에 활용 | 평평한 표 |
| eviction | leaf 우선 (공유 prefix가 오래 생존) | block별 ref count + LRU |

두 방식 모두 "prefix가 같으면 KV가 같다"는 같은 사실을 이용합니다. 성능 차이는 workload와 구현에 따라 달라지며, 최근에는 서로의 장점을 흡수하는 방향으로 수렴하고 있습니다.

---

## 9. 📖 v0.5.23의 실제 실행 경로: UnifiedRadixCache

`mem_cache/registry.py`를 보면:

```python
# mem_cache/registry.py: default_radix_cache_factory  (간추림)
def default_radix_cache_factory(ctx):
    if ctx.disable_radix_cache:
        ...
        return create_unified_radix_cache(ctx)      # 꺼도 객체는 만든다 (disable 모드)
    if get_memory().enable_lmcache: ...
    if is_pure_swa: return PureSWARadixCache(...)
    if get_memory().enable_flexkv: ...
    return create_unified_radix_cache(ctx)          # ← 일반 dense 모델의 기본 경로
```

- 일반 dense 모델(Qwen3-1.7B)도 기본적으로 **`UnifiedRadixCache`** (`unified_radix_cache.py` + `unified_cache/unified_tree_core.py`)를 씁니다.
  Full attention / Sliding-window attention / Mamba(선형 attention) 상태와 HiCache(계층형 캐시)까지 한 틀로 다루기 위한 일반화 버전입니다.
- 알고리즘(최장 prefix 매칭, 노드 split, lock, LRU leaf eviction)은 `radix_cache.py`와 같습니다. 그래서 **개념은 830줄짜리 `radix_cache.py`로 배우고**, "실제로는 어디서 도는가"는 registry로 확인합니다.
- tree 핵심 연산을 Rust로 구현한 경로(`mem_cache/rust_tree_core`)도 있습니다 — 매 iteration 대기열 전체를 match하는 CPU 비용(§6.1)이 실제로 문제가 되기 때문입니다.

교훈: **실제 엔진 코드는 작은 개념 구현보다 수 배 크다.** 개념을 작은 구현으로 먼저 잡고, 큰 코드에서는 같은 이름(`match_prefix`, `insert`, `evict`)을 찾아 대응시키는 식으로 읽으세요.

---


## ★ 실습 전 예측 질문

실습 직전에 노트에 **숫자로** 적으세요. 실습 후 실제 값과 비교하는 것이 이 회차에서 가장 많이 배우는 순간입니다.

1. 공유 prefix가 4096 token, 고유 suffix 64 token일 때 warm 요청의 TTFT는 cold 대비 몇 %일까?
2. 같은 요청을 두 번 보내면 두 번째 `cached_tokens`는 정확히 몇일까?
3. KV pool이 24000 token이고 prefix 2048짜리 그룹 16개가 섞여 들어오면, FCFS의 cache hit rate는 대략 얼마일까? LPM은?
4. 서버 로그 `KV Cache is allocated`의 KV size(GB) ÷ #tokens 는 몇 KiB일까?

---

## 10. 요약 · 자기 점검 · 더 읽을거리

### 요약

1. KV 메모리는 ReqToTokenPool(표) → TokenToKVPool(물리) → Allocator(빈 목록) 세 층. 단편화 없음, 공유는 같은 슬롯 번호.
2. 시작할 때 KV pool을 한 번에 잡는다. 크기 = `max_total_num_tokens`.
3. Radix tree: key = token 시퀀스, value = 슬롯 번호. match는 첫 token으로 자식 선택, 중간에서 갈라지면 split.
4. match는 `입력[:-1]`까지. 같은 요청을 다시 보내도 `prompt − 1`.
5. 요청이 끝나면 입력 + 출력이 insert(`checkpoint`) → 멀티턴 재사용.
6. 실행 중 경로는 lock. evictable = lock 0. 캐시는 언제든 돌려받는 빌린 메모리.
7. eviction은 lock 0 leaf부터 LRU(기본). 공유 prefix가 오래 생존.
8. LPM은 hit를 올리지만 starvation 위험, 대기열 128 초과 시 FCFS로 전환. v0.5.23 기본은 FCFS.
9. 프롬프트 앞부분이 변하면 캐시는 깨진다.

### 자기 점검 문제

1. page_size=1, KV pool 8슬롯. 요청 "ABCD", "ABEF", "XYZ"가 순서대로 실행·종료되면 tree 모양과 evict되는 노드는? (lock은 실행 중에만)
2. 같은 system prompt(3000 token)를 쓰는 요청 32개가 **동시에** 도착했고 tree는 비어 있다. in-batch prefix caching이 없다면 prefill 계산량은 몇 token이고, 있다면 대략 몇 token인가?
3. `--disable-radix-cache`를 켜면 메모리 사용과 throughput은 각각 어떻게 될까? 공유가 전혀 없는 workload라면?
4. LPM에서 starvation이 생기는 시나리오를 하나 만들어 보라.
5. 왜 eviction은 내부 노드가 아니라 leaf부터 하는가?
6. 멀티턴 채팅에서 매 턴 앞에 "현재 시각: …"을 넣는 앱이 있다. 몇 번째 턴부터 cache hit가 0이 되나?
7. 두 요청이 같은 prefix를 동시에 prefill했다. 둘 다 끝난 뒤 tree에는 그 prefix가 몇 벌 있나? 남는 슬롯은 어떻게 되나?

<details>
<summary>답</summary>

1. "ABCD" → [ABCD] (4슬롯, free 4). "ABEF": match("ABE") → [ABCD]의 2번째까지 일치 → split, cached 2, 새 2 → root ─ [AB] ┬ [CD] / └ [EF], free 2. "XYZ": 새 3 필요, free 2 → lock 0 leaf 중 가장 오래된 [CD] (2칸)를 evict → free 4 → 3 할당. 최종: root ┬ [AB] ─ [EF] / └ [XYZ], free 1.
2. 없으면 32 × 3000 ≈ 96,000 token (각자 계산). 있으면 대표 1개 3000 + 나머지 31개는 다음 iteration에 hit → 3000 + 31 × (고유 부분) 정도.
3. 공유가 있으면 KV를 요청마다 따로 저장 → 메모리↑, 재계산 → throughput↓. 공유가 없으면 거의 차이 없음 (radix cache의 비용은 작음).
4. A그룹 요청이 계속 도착해 항상 hit가 길고, 혼자 다른 prefix를 가진 요청 Z는 hit 0이라 매번 뒤로 밀림 → Z의 TTFT가 무한히 늘어날 수 있음 (대기열 128 초과로 FCFS 전환되기 전까지).
5. 내부 노드는 자식들의 prefix다. 내부 노드를 지우면 그 아래 모든 경로가 쓸모없어진다. leaf는 다른 경로에 영향이 없다.
6. 시각이 매 턴 바뀌고 프롬프트 맨 앞에 있으므로 2번째 턴부터 hit ≈ 0 (첫 몇 token "현재 시각: " 정도만).
7. 한 벌. 두 번째 insert에서 `total_prefix_length`만큼은 이미 있으므로 두 번째 요청이 계산한 중복 슬롯은 free되고, 그 요청의 `req_to_token` 행은 tree의 슬롯을 가리키도록 다시 써진다 (`checkpoint`).
</details>

### 더 읽을거리

- Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (NeurIPS 2024) — RadixAttention, cache-aware 스케줄링
- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (SOSP '23)
- SGLang 코드: `mem_cache/README.md`, `mem_cache/radix_cache.py`, `managers/schedule_policy.py`
- SGLang 문서: Radix eviction policy, HiCache 설계 문서 (`docs/advanced_features/`)

### 다음 회차 예고

오늘 "메모리가 모자라면 evict한다"고 했습니다. 그런데 evict할 캐시도 없을 만큼 **실행 중인 요청들이 메모리를 다 쓰면?** 다음 회차는 스케줄러가 요청을 들여보낼지(admission), 쫓아낼지(retraction), 긴 프롬프트를 어떻게 나눌지(chunked prefill)를 봅니다.
