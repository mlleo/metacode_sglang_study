# Session 4 이론 — 스케줄러 심화: Admission · Retraction · Chunked Prefill · Overlap

> **핵심 질문:** 메모리가 모자랄 때, 긴 프롬프트가 끼어들 때, CPU가 병목일 때 스케줄러는 무엇을 하는가?

## 이 문서 사용법

- ★ 수업 핵심 / 📖 복습·심화
- 코드 발췌는 **SGLang v0.5.23** 기준이며 간추린 것입니다. 이 회차의 코드는 특히 많이 간추렸습니다 — 원문에는 SWA, Mamba, DLLM, PD 분리, 우선순위, HiCache 분기가 섞여 있습니다. "일반 dense 모델, GPU 1장" 경로만 남겼습니다.
- 수업용 추천 구성: §0 → §1 → §2.1–2.4 → §2.5(숫자 예제) → §3.1–3.3 → §4.1–4.6 → §5.1 → §7

## 목차

0. ★ 지난 회차 연결
1. ★ 한 iteration을 끝까지 — prefill batch는 어떻게 만들어지나
2. ★ Admission control — 누구를 들여보낼 것인가
3. ★ Retraction — 들여보낸 걸 후회할 때
4. ★ Chunked prefill — 긴 프롬프트가 모두를 멈출 때
5. ★ Overlap scheduling — CPU 시간 숨기기
6. 📖 우선순위 스케줄링과 preemption
7. ★ 손잡이(knob) 정리와 진단 순서
8. 요약 · 자기 점검 · 더 읽을거리

---

## 0. ★ 지난 회차 연결

| `generate()`의 한계 | 서빙 엔진의 답 | 오늘 |
|---|---|---|
| 메모리가 모자라면 그냥 OOM | admission control, retraction | §2, §3 |
| 긴 프롬프트 하나가 모두를 멈춘다 | chunked prefill | §4 |
| Python이 커널을 하나씩 launch (CPU overhead) | overlap scheduling (+ CUDA graph는 Session 5) | §5 |

Session 3에서 "캐시는 언제든 돌려받을 수 있는 빌린 메모리"라고 했습니다. 오늘은 **실행 중인 요청들만으로 메모리가 모자랄 때** 스케줄러가 어떻게 행동하는지 봅니다.
그리고 Session 2 §3.2의 "prefill 우선" 규칙이 낳는 부작용(ITL 스파이크)을 어떻게 다루는지 봅니다.

---

## 1. ★ 한 iteration을 끝까지

### 1.1 전체 흐름

```text
get_next_batch_to_run()
 ├─ 직전 prefill batch를 running_batch에 합침 (Session 2 §3.2)
 ├─ get_new_batch_prefill()                     ← §2 admission
 │    ├─ (조건) running_batch가 꽉 찼거나 대기열이 비었으면 → None
 │    ├─ waiting_queue 정렬 (fcfs / lpm / ...)   ← Session 3 §6
 │    ├─ PrefillAdder 생성: 이번 iteration의 "예산" 계산
 │    ├─ 진행 중인 chunked 요청이 있으면 그 다음 조각을 먼저 넣음   ← §4
 │    └─ for req in waiting_queue:
 │          r = adder.add_one_req(req)          → CONTINUE / NO_TOKEN / OTHER
 │          if r != CONTINUE: break             ← 여기서 잘린 요청들이 #queue-req
 │    → 받아들인 요청이 하나라도 있으면 prefill batch 반환 (prefill 우선)
 └─ (prefill이 없으면) update_running_batch()   ← §3 retraction
      ├─ check_decode_mem(): 다음 decode step에 요청당 1칸씩 들어가나?
      ├─ 안 들어가면 retract_decode()
      └─ decode batch 반환
```

### 1.2 `get_new_batch_prefill` (간추림)

```python
# managers/scheduler.py: get_new_batch_prefill  (v0.5.23, 크게 간추림)
def get_new_batch_prefill(self, running_batch):
    if (running_batch.batch_is_full or len(self.waiting_queue) == 0) and self.chunked_req is None:
        return NextBatchPlan(None, running_batch)
    if self.get_num_allocatable_reqs(running_batch.batch_size()) <= 0 and self.chunked_req is None:
        running_batch.batch_is_full = True                     # 요청 슬롯(max_running_requests)이 꽉 참
        return NextBatchPlan(None, running_batch)

    self.policy.calc_priority(self.waiting_queue)              # 정렬 (Session 3 §6)

    adder = PrefillAdder(
        self.page_size, self.tree_cache, self.token_to_kv_pool_allocator, running_batch,
        self.new_token_ratio_tracker.current,                  # §2.3
        self.max_prefill_tokens,                               # rem_input_tokens
        self.chunked_prefill_size,                             # rem_chunk_tokens
        max_running_requests=self.max_running_requests, ...)

    if self.chunked_req is not None:                           # §4: 진행 중인 긴 프롬프트가 우선
        self.chunked_req.init_next_round_input()
        self.chunked_req = adder.add_chunked_req(self.chunked_req)

    for req in self.waiting_queue:
        if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_batch.batch_size()):
            running_batch.batch_is_full = True
            break
        req.init_next_round_input(self.tree_cache)            # radix cache match (Session 3)
        res = adder.add_one_req(req, has_chunked_req=(self.chunked_req is not None), ...)
        if res != AddReqResult.CONTINUE:
            if res == AddReqResult.NO_TOKEN:
                running_batch.batch_is_full = True             # 메모리 부족: 다음 기회까지 prefill 시도 안 함
            break

    can_run_list = adder.can_run_list
    if adder.new_chunked_req is not None:                      # 이번에 새로 잘린 요청
        self.chunked_req = adder.new_chunked_req
    self.waiting_queue = [r for r in self.waiting_queue if r not in set(can_run_list)]
    new_batch = ScheduleBatch.init_new(can_run_list, ...)
    new_batch.prepare_for_extend()                             # KV 슬롯 할당, ForwardBatch 재료 준비
    ...
    return NextBatchPlan(new_batch, running_batch)
```

관찰:

- **break가 핵심입니다.** 대기열 앞쪽 요청 하나가 예산을 넘으면 그 뒤 요청들은 (더 작더라도) 이번 iteration에 시도조차 안 합니다. 이것이 정렬 정책이 중요한 이유이고, 큰 요청 하나가 앞에 있으면 **대기열 전체가 막히는(head-of-line blocking)** 한 원인입니다.
- `batch_is_full`은 "더 넣어 봐야 소용없다"는 표시입니다. 요청이 끝나거나 메모리가 풀리면 다시 False가 됩니다.

### 1.3 로그와 대응

| 로그 | 의미 |
|---|---|
| `Prefill batch, #new-seq: k` | 이번 iteration에 `add_one_req`가 CONTINUE를 k번 반환 |
| `#queue-req: q` | 잘려서 남은 요청 q개 |
| `#new-token` 이 `chunked_prefill_size`와 같은 줄이 연속 | 긴 요청이 조각으로 들어가는 중 (§4) |
| `KV cache pool is full. Retract requests. #retracted_reqs: n` | `retract_decode()`가 n개를 쫓아냄 (§3) |

---

## 2. ★ Admission control — 누구를 들여보낼 것인가

### 2.1 네 가지 제한

`PrefillAdder`(`managers/schedule_policy.py`)는 iteration마다 다음 제한을 들고 시작합니다.

| 제한 | 무엇을 제한 | 기본값 (4090, v0.5.23) | 플래그 |
|---|---|---|---|
| `rem_total_tokens` | **KV 메모리** (지금 + 앞으로 생성할 몫까지) | pool 크기 ≈ 12만 token | `--mem-fraction-static`, `--max-total-tokens` |
| `rem_input_tokens` | 이번 prefill batch의 계산량 | 16384 | `--max-prefill-tokens` |
| `rem_chunk_tokens` | 요청 하나를 한 번에 prefill할 최대 길이 | 4096 | `--chunked-prefill-size` |
| 요청 슬롯 | 동시에 running일 수 있는 요청 수 | 자동 계산 | `--max-running-requests` |

`rem_input_tokens`와 `rem_chunk_tokens`는 **계산량(시간)** 제한, `rem_total_tokens`는 **메모리** 제한입니다. 셋 중 먼저 바닥나는 것이 이번 iteration을 끝냅니다.

### 2.2 메모리 예산의 계산

```python
# mem_cache/prefill_budget.py: PrefillBudget  (간추림)
@property
def remaining_total(self):
    return (allocator.available_size() + tree_cache.evictable_size()) - self.total_offset

# PrefillAdder.__init__: 실행 중 요청들의 "앞으로 생성할 몫"을 먼저 빼 둔다
self.memory_budget.total_offset += sum(
    self._get_running_request_total_token_offset(r) for r in running_batch.reqs)

def _get_running_request_total_token_offset(self, req):
    ratio = 1.0 if req.sampling_params.ignore_eos else self.new_token_ratio     # ★
    return min(req.sampling_params.max_new_tokens - len(req.output_ids), CLIP_MAX_NEW_TOKENS) * ratio
```

```python
# managers/schedule_policy.py: PrefillAdder.add_one_req → _select_prefill_admission → _commit_prefill_admission
# (v0.5.23, SWA/Mamba/HiCache/DLLM 분기를 지우고 간추림)
def add_one_req(self, req, has_chunked_req, ...):
    max_new = min(max(req.sampling_params.max_new_tokens - len(req.output_ids), 0), CLIP_MAX_NEW_TOKENS)
    extend_len = len(req.full_untruncated_fill_ids) - req.prefix_len      # cache hit를 뺀 나머지
    total_tokens = extend_len + max_new                                   # 새 요청은 max_new 전부로 셈
    with self._lock_node(req.last_node):                                  # 검사 중에 자기 prefix가 evict되지 않게
        adm = self._select_prefill_admission(req, total_tokens=total_tokens, ...)
        if isinstance(adm, AddReqResult):
            return adm                                                    # NO_TOKEN 또는 OTHER
        self._commit_prefill_admission(req, adm, ...)                     # can_run_list에 추가, 예산 차감
    return self.budget_state()                                            # 예산이 남았으면 CONTINUE

def _select_prefill_admission(self, req, *, total_tokens, ...):
    if not total_tokens < self.memory_budget.remaining_total:
        return AddReqResult.NO_TOKEN                                      # 메모리 부족
    if self.rem_chunk_tokens is None and self.can_run_list and extend_len >= self.rem_input_tokens:
        return AddReqResult.OTHER                                         # (청크 끈 경우) 이번 batch 계산량 초과
    max_new_tokens, is_chunked = min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS), False
    if extend_len > available_chunk_tokens:                               # 남은 chunk 예산보다 길면 잘라서 앞부분만 (§4)
        extend_len = available_chunk_tokens // page_size * page_size
        is_chunked, max_new_tokens = True, 0                              # 조각 중에는 생성 몫을 예약하지 않음
    return _PrefillAdmission(prefix_len, extend_len, max_new_tokens, is_chunked)

def budget_state(self):
    if not self.memory_budget.has_capacity():            return AddReqResult.NO_TOKEN
    if self.rem_input_tokens <= 0:                       return AddReqResult.OTHER
    if self.rem_chunk_tokens is not None and self.rem_chunk_tokens <= 0:
                                                         return AddReqResult.OTHER
    return AddReqResult.CONTINUE
```

정리하면:

```text
rem_total_tokens = (빈 슬롯 + evictable 캐시)
                 − Σ_running r  min(max_new_r − 생성한 수, 4096) × ratio_r      ← ratio_r = 1.0 (ignore_eos) 또는 new_token_ratio
                 − Σ_이번에 받은 요청  (prefill token + min(max_new, 4096))

새 요청 하나의 비용 = (새로 prefill할 token 수) + min(max_new_tokens, 4096)
→ 비용 < rem_total_tokens 이면 CONTINUE, 아니면 NO_TOKEN
```

`CLIP_MAX_NEW_TOKENS`(기본 4096, `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION`): 사용자가 `max_new_tokens=32000`을 줘도 예산 계산에는 4096까지만 셉니다. 생성 자체는 32000까지 계속됩니다. 터무니없이 큰 `max_new_tokens`가 admission을 막지 않게 하는 장치입니다.

### 2.3 핵심 질문: 실행 중 요청의 "앞으로 생성할 몫"을 얼마로 잡나

- 전부(×1.0) 잡으면: 절대 메모리가 모자라지 않지만, 대부분의 요청은 `max_new_tokens` 전에 EOS로 끝나므로 **메모리가 남아도는데 새 요청을 못 받음** (throughput↓)
- 조금만 잡으면: 많이 받아들이지만 **실제로 길게 생성하면 모자람** → retraction (§3)

SGLang의 답: **`new_token_ratio`라는 학습하는 낙관도**

```python
# managers/scheduler_components/new_token_ratio_tracker.py  (거의 원문)
@classmethod
def from_config(cls):
    init = min(SGLANG_INIT_NEW_TOKEN_RATIO * schedule_conservativeness, 1.0)   # 0.7 × 1.0
    min_ratio = min(init * SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR, 1.0)             # 0.7 × 0.14 = 0.098
    decay = (init - min_ratio) / SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS            # 600 step에 걸쳐
    return cls(init=init, min=min_ratio, decay=decay, current=init)

def decay_step(self):                    # decode step마다 (retract가 없었을 때)
    self.current = max(self.current - self.decay, self.min)

def reset(self):                         # 서버가 완전히 한가해지면
    self.current = self.init

@staticmethod
def estimate_new_token_ratio_after_retract(reqs):     # retraction 직후의 새 추정
    total_decoded = sum(len(r.output_ids) for r in reqs)
    total_max_new = sum(r.sampling_params.max_new_tokens for r in reqs)
    return min(1.0, (total_decoded + SGLANG_RETRACT_DECODE_STEPS * len(reqs)) / (total_max_new + 1))
```

| 상황 | new_token_ratio |
|---|---|
| 시작 / 서버가 한가해질 때 | 0.7 (× `schedule_conservativeness`) |
| decode step마다 | 조금씩 감소 → 최소 0.098 — "점점 더 과감하게" |
| retraction 발생 시 | "남은 요청들이 실제로 지금까지 생성한 비율 + 여유(20 step)"로 다시 추정 → 대개 **높게** — "혼나고 나서 조심" |

전형적인 AIMD(additive increase, multiplicative decrease)와 비슷한 피드백 제어입니다 — TCP 혼잡 제어처럼 "문제가 없으면 조금씩 더 밀어붙이고, 문제가 생기면 크게 물러선다".

### 2.4 ⚠️ 벤치마크가 스케줄러를 바꾼다

§2.2 코드의 ★ 줄: v0.5.23은 **`ignore_eos=True`인 요청에는 ratio 대신 1.0**을 씁니다. 길이가 확정된 요청은 일찍 끝날 리가 없기 때문입니다.
그런데 우리 벤치마크는 전부 `ignore_eos=True`였습니다!

- 즉 지금까지의 측정은 **"가장 보수적인 admission"** 조건이었습니다.
- 실제 서비스(EOS로 끝나는 요청)에서는 같은 서버가 더 많은 요청을 동시에 받고, 대신 retraction이 생길 수 있습니다.
- 오늘 실습 1에서 `eos` / `fixed` 두 경우를 직접 비교합니다.

> 일반 원칙: **벤치마크 설정이 시스템의 적응 메커니즘을 건드리지 않는지** 의심하세요. 캐시(warm/cold), 적응형 batch, 낙관적 예약, JIT 컴파일 모두 해당됩니다.

### 2.5 숫자 예제 (칠판에서)

pool = 30,000 token. running 요청 10개, 각각 입력 100 + 생성 200 진행 중, `max_new_tokens = 1000`, ratio = 0.5, evictable 캐시 0.

```text
사용 중 KV      = 10 × 300                    =  3,000
빈 슬롯         = 30,000 − 3,000               = 27,000
running 예약    = 10 × (1000 − 200) × 0.5     =  4,000
rem_total       = 27,000 − 4,000              = 23,000

대기 요청: 입력 100, max_new 1000 → 비용 1,100
→ 23,000 / 1,100 ≈ 20개 (rem_input_tokens 16384: 20 × 100 = 2,000 OK, chunk 4096: 각 100 OK)
```

| 조건 | running 예약 | 승인 수 |
|---|---:|---:|
| ratio 0.5 | 4,000 | 약 20 |
| ratio 0.1 (decay 후) | 800 | 약 23 |
| `ignore_eos=True` (ratio 1.0) | 8,000 | 약 17 |

이 차이가 누적되면 동시 batch 크기가 크게 달라집니다 — 그리고 Session 1 §5.4에 따라 batch 크기는 곧 throughput입니다.

### 2.6 `max_running_requests`는 어떻게 정해지나 (📖)

지정하지 않으면 요청 슬롯 수(`ReqToTokenPool` 크기)가 메모리와 context 길이로 자동 결정되고, 시작 로그의 `max_running_requests=`로 확인할 수 있습니다.
진단 팁: `token usage`가 낮은데 `#queue-req`가 쌓이면 메모리가 아니라 이 상한(또는 `max_prefill_tokens`)에 걸린 것입니다.

---

## 3. ★ Retraction — 들여보낸 걸 후회할 때

### 3.1 언제

```python
# managers/scheduler.py: update_running_batch  (간추림)
def update_running_batch(self, batch):
    batch.filter_batch(...)                                   # 끝난 요청 제거
    if batch.is_empty():
        return batch
    if not batch.check_decode_mem():                          # 다음 step에 요청당 1칸 (빈 슬롯 + evictable)
        old_ratio = self.new_token_ratio_tracker.current
        retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode()
        self.new_token_ratio_tracker.current = new_token_ratio
        logger.warning(f"KV cache pool is full. Retract requests. #retracted_reqs: {len(retracted_reqs)}, "
                       f"#new_tokens_gained: ..., #new_token_ratio: {old_ratio:.4f} -> {new_token_ratio:.4f}")
        for req in retracted_reqs:
            self._add_request_to_queue(req, is_retracted=True)   # 대기열로 되돌림
    else:
        self.new_token_ratio_tracker.decay_step()             # 문제없었으면 조금 더 낙관적으로
    batch.prepare_for_decode()                                # 요청마다 KV 1칸 할당 → out_cache_loc
    return batch
```

decode는 step마다 요청당 1칸씩 반드시 필요합니다. 그 1칸을 줄 수 없으면 **누군가는 나가야** 합니다.

### 3.2 누구를?

```python
# managers/schedule_batch.py: _get_decode_retraction_order  (기본 'length' 정책, 거의 원문)
def length_key(req):
    return (len(req.output_ids), -len(req.origin_input_ids))
sorted_indices.sort(key=lambda i: length_key(reqs[i]), reverse=True)   # 남길 순서: 많이 생성한 것, 입력 짧은 것
# retract_decode는 이 목록의 '끝'에서부터 꺼내 쫓아낸다 → 생성을 가장 적게 한 요청, 그중 입력이 긴 요청부터
```

- **생성을 적게 한 요청부터**: 버리는 진행분(= 다시 해야 할 decode)이 가장 적음
- 같으면 **입력이 긴 요청부터**: 돌려받는 메모리가 큼
- `--retraction-policy priority`로 우선순위 기반으로 바꿀 수도 있습니다.
- 마지막 하나까지 쫓아내도 안 되면 그 요청은 abort됩니다 (OOM). 정상 설정에서는 거의 없는 경우입니다.

### 3.3 비용은?

```text
요청 R이 입력 200, 생성 300 진행 중에 retract
  → R의 KV 500칸 반납 (radix tree에 남겨 둘 수도 있지만, 메모리 부족 상황이라 곧 evict되기 쉬움)
  → R은 대기열로. 다시 차례가 오면 입력 200 + 출력 300 = 500 token을 prefill (cache hit가 없다면)
  → 그다음부터 decode 재개
```

- **GPU 비용**: 재-prefill. 로그 요약의 `sum_new_token`이 늘어나는 것으로 보입니다 (실습 1).
- **사용자 비용**: 잘 나오던 스트림이 **멈췄다가** 한참 뒤 다시 나옴 → 그 요청의 max ITL과 E2E가 크게 나빠짐.
- **스케줄러의 반성**: `#new_token_ratio: 0.1043 -> 0.4312` — "너무 낙관적이었다"고 인정하고 조심 모드로 전환.

### 3.4 retraction은 나쁜가? (📖)

반드시 그렇지는 않습니다. 낙관적으로 많이 받아들인 덕분에 batch가 커져 throughput이 오르고, 가끔의 retraction 비용이 그보다 작다면 전체적으로는 이득입니다.
**throughput(낙관)과 tail latency(보수) 사이의 선택**이고, `--schedule-conservativeness`가 그 다이얼입니다. 실습 1의 세 설정 비교가 바로 이 질문입니다.

---

## 4. ★ Chunked prefill — 긴 프롬프트가 모두를 멈출 때

### 4.1 문제: head-of-line blocking

16명이 대화하며 token을 받고 있는데(ITL ≈ 10 ms), 누군가 16,000 token짜리 문서를 붙여 넣습니다.
"prefill 우선" 규칙 때문에 다음 iteration은 그 16K prefill입니다 — Session 1 §5.6 계산으로 RTX 4090에서 1초 안팎.

```text
피해자들의 token 간격:   10ms 10ms 10ms ──────── ~1,000 ms ──────── 10ms 10ms …
                                       ↑ 가해자의 prefill
```

평균 TPOT는 거의 안 변하지만 **ITL p99와 max가 폭발**합니다 (Session 1 §7.2).
게다가 그 iteration에서는 뒤에 도착한 **짧은 새 요청들도** prefill 예산을 못 받아 기다립니다 (TTFT 증가).

### 4.2 Chunked prefill: 잘라서 넣기

`--chunked-prefill-size C`면 긴 prefill을 C token씩 잘라 **여러 iteration에 걸쳐** 처리합니다.

동작 (§1.2 코드와 대응):

1. `add_one_req`에서 `extend_len`이 남은 chunk 예산보다 길면 앞 C token만 이번 batch에 넣고 그 요청을 `new_chunked_req`로 표시
2. 그 조각의 forward가 끝나면 계산한 부분을 radix tree에 `checkpoint` (Session 3 §4.6) — **다음 조각은 이 부분을 cache hit로** 재사용
3. 다음 iteration의 `get_new_batch_prefill`에서 **`chunked_req`가 대기열보다 먼저** `add_chunked_req`로 다음 조각을 넣음
4. 마지막 조각이 끝나면 첫 token이 나오고 running_batch에 합류
5. 한 번에 **chunked 요청은 하나**만 진행됩니다 (`self.chunked_req`)

### 4.3 ⚠️ chunked prefill만으로는 피해자의 멈춤이 줄지 않는다

직관적으로는 이렇게 될 것 같습니다:

```text
기대:  [P 4K][decode][P 4K][decode][P 4K][decode][P 4K][decode] …    ← 조각 사이에 decode
```

하지만 v0.5.23 기본 설정에서 실제로는 이렇습니다:

```text
실제:  [P 4K][P 4K][P 4K][P 4K][decode][decode] …                   ← 조각이 연달아 실행
```

이유는 Session 2 §3.2의 **"prefill 우선"** 규칙입니다. `chunked_req`가 남아 있으면 `get_new_batch_prefill`은 매번 다음 조각으로 batch를 만들고,
`get_next_batch_to_run`은 prefill batch가 있으면 decode를 하지 않습니다. 게다가 chunk 예산(`rem_chunk_tokens`)은 iteration 전체가 공유하므로,
조각 하나가 4096을 다 쓰면 **다른 새 요청도 그 iteration에 못 들어갑니다.**

그래서 chunked prefill 단독의 효과는:

| 효과 | 설명 |
|---|---|
| ✅ activation 메모리 상한 | 한 forward의 token 수가 C로 제한 → 긴 프롬프트도 OOM 없이 처리 (기본값이 GPU 메모리로 정해지는 이유, §4.7) |
| ✅ iteration 하나의 길이 상한 | 스케줄러가 C token마다 한 번씩 "숨을 쉼" → 아래 두 장치가 끼어들 틈이 생김 |
| ✅ 긴 프롬프트의 부분 공유 | 조각마다 tree에 등록 |
| ❌ 피해자의 총 멈춤 | 조각이 연달아 돌면 16K 전체 prefill 시간만큼 그대로 멈춤 |

피해자의 멈춤을 실제로 줄이려면 조각 사이에 decode를 **끼워 넣는 장치**가 필요합니다. 두 가지가 있습니다.

### 4.4 장치 ①: Mixed chunk — 조각과 decode를 한 batch에

`--enable-mixed-chunk`: prefill 조각과 실행 중 요청들의 decode token을 **한 forward에 섞습니다** (`ForwardMode.MIXED`).

```python
# managers/scheduler.py: _get_new_batch_prefill_raw의 끝부분  (간추림)
if self.is_mixed_chunk and not running_batch.is_empty() and not return_logprob:
    if running_batch.check_decode_mem():
        running_batch.prepare_for_decode()          # decode 요청마다 KV 1칸
        new_batch.mix_with_running(running_batch)   # prefill 조각 + decode token들을 ragged로 이어 붙임
        new_batch.decoding_reqs = running_batch.reqs
        running_batch = ScheduleBatch(reqs=[], ...) # 이번 iteration에 running_batch는 따로 안 돔
```

```text
mixed 없음:  [P 1K][P 1K]…[P 1K][D×16][D×16]   피해자: 긴 멈춤 → 진행
mixed 있음:  [P 1K + D×16][P 1K + D×16]…      피해자: 매 iteration 1 token씩 (조각 하나 시간만큼 느리게)
```

- 피해자의 ITL ≈ **조각 하나의 prefill 시간**. 그래서 이제 chunk 크기가 피해자 ITL을 직접 결정합니다 (C↓ → ITL↓).
- 대가: decode step 하나가 prefill 조각만큼 길어지므로 **ITL 평균(p50)은 나빠질 수 있습니다.** mixed batch 안의 decode token만큼 prefill 예산이 줄어듭니다 (`num_mixed_decode_tokens`).
- logprob를 요청한 batch 등 일부 경우엔 섞지 않습니다.
- 이것이 Sarathi-Serve 논문이 말하는 **stall-free scheduling**에 해당합니다 (vLLM은 chunked prefill을 켜면 기본적으로 이렇게 섞습니다).

### 4.5 장치 ②: prefill-decode interval — 조각 사이에 decode를 보장

`--prefill-decode-interval N`: prefill batch를 실행한 뒤 **decode round N번을 먼저** 돌리고 나서 다음 prefill을 받습니다 (`_should_defer_prefill`, `_arm_prefill_decode_interval`).
v0.5.23에서 기본은 꺼져 있습니다(일부 모델·GPU 조합 제외).

```text
interval 1 + chunk 4K:  [P 4K][D][P 4K][D][P 4K][D][P 4K][D] …     ← "기대" 그림이 이제 현실
```

- 피해자의 최악 멈춤 ≈ 조각 하나의 prefill 시간
- 가해자 TTFT는 조각 수 × (조각 + decode round)만큼 늘어남
- mixed chunk와 달리 batch 모양이 단순(순수 prefill / 순수 decode)해서 CUDA graph 등 decode 최적화를 그대로 씁니다

### 4.6 trade-off 정리

| 설정 | 피해자의 최악 멈춤 | 가해자 TTFT | 비고 |
|---|---|---|---|
| chunk 없음 (`-1`) | 전체 prefill 시간 | 가장 짧음 | 아주 긴 입력은 activation OOM 위험 |
| chunk 4096 (기본) | ≈ 전체 prefill 시간 (조각이 연달아 돎) | 약간 늘어남 | 메모리 안전 |
| chunk 1024 | ≈ 전체 prefill 시간 | 더 늘어남 | 작은 GEMM 여러 번 → 효율↓ |
| chunk 1024 + mixed | ≈ 조각 하나 시간 | 늘어남 | ITL p50도 늘 수 있음 |
| chunk 1024 + interval 1 | ≈ 조각 하나 시간 | 늘어남 | 순수 decode step 유지 |

가해자 TTFT가 늘어나는 두 가지 이유:

1. 조각 사이에 decode iteration이 끼어든다 (mixed/interval일 때)
2. 뒤쪽 조각일수록 attention이 **앞 조각들의 KV 전부**를 읽어야 한다 (조각 k의 attention 비용 ∝ C × k·C)

이 trade-off는 근본적으로 **TTFT(가해자) vs ITL(피해자)** 의 교환입니다. 실습 2에서 다섯 설정을 직접 재서, "chunk만 줄여서는 안 줄어든다"는 것을 확인합니다.

### 4.7 기본값은 어떻게 정해지나 (📖)

```python
# arg_groups/memory_hook.py  (간추림)
if gpu_mem < 20 GB:   chunked_prefill_size = 2048;  decode cuda graph max_bs = 8      # T4, 4080
elif gpu_mem < 35 GB: chunked_prefill_size = 4096;  max_bs = 48                       # A10, 4090, 5090
elif gpu_mem < 60 GB: chunked_prefill_size = 4096;  max_bs = 32                       # A100-40G, L40
elif gpu_mem < 90 GB: chunked_prefill_size = 8192;  max_bs = 256                      # H100, A100-80G
...
# 그리고 mem_fraction_static = (GPU 메모리 − 예비) / GPU 메모리
#   예비 ≈ 512 MB + chunked_prefill_size × 1.5 + CUDA graph 몫
```

chunk가 클수록 prefill 중 activation 메모리가 커지므로, 작은 GPU일수록 chunk를 작게 잡습니다. **chunk 크기는 지연시간 손잡이이면서 메모리 손잡이**이기도 합니다.

### 4.8 근본적인 해결: PD disaggregation (📖)

chunked prefill은 "같은 GPU에서 prefill과 decode가 시간을 나눠 쓰는" 문제를 완화할 뿐입니다.
아예 **prefill 전용 GPU와 decode 전용 GPU를 분리**하고 KV를 네트워크로 넘기는 방식(DistServe, Mooncake, SGLang의 `--disaggregation-mode`)이 대규모 서비스의 해법입니다 (Session 6 §5).

---

## 5. ★ Overlap scheduling — CPU 시간 숨기기

### 5.1 아이디어

GPU forward 하나가 5 ms인데, 그 앞뒤로 CPU가 하는 일(다음 batch 고르기, 텐서 준비, 결과 처리, detokenizer로 전송)이 1–2 ms라면?

```text
normal :  CPU▓ GPU████ CPU▓ GPU████ CPU▓ GPU████        GPU가 CPU를 기다리는 구멍
overlap:  CPU▓▓ CPU▓▓ CPU▓▓
               GPU████GPU████GPU████                      batch N 실행 중에 batch N+1 준비
```

`event_loop_overlap` (기본, Session 2 §3.3): batch N을 GPU에 **비동기로** 던져 놓고, 결과를 기다리지 않은 채 batch N+1을 준비합니다.
batch N의 결과 처리는 N+1을 던진 뒤에 합니다 (`result_queue`).

### 5.2 어려운 점 (📖)

batch N+1의 입력은 batch N이 만든 token입니다. 그런데 batch N+1을 준비하는 시점에 **batch N의 결과를 아직 모릅니다.**

- 해법: "다음 입력 token"을 실제 값 대신 **GPU 메모리 위의 자리(placeholder)** 로 연결해 둡니다. GPU는 순서대로 실행하므로, batch N+1의 커널이 돌 때는 batch N이 그 자리에 값을 이미 써 놓았습니다. (`managers/overlap_utils.py`의 `FutureMap`)
- 결과 처리가 한 step 늦기 때문에, **이미 끝난 요청(EOS)이 한 step 더 decode되는** 경우가 생깁니다. 그 token은 버려집니다 (Session 2 §2.4의 `if req.finished() or req.is_retracted: continue`). 작은 낭비를 감수하고 CPU 시간을 숨기는 것입니다.
- 일부 상황(연속 prefill 설정, grammar 동기화가 필요한 speculative decoding 등)에서는 그 batch만 overlap을 끕니다 (`is_disable_overlap_for_batch`).

### 5.3 언제 효과가 큰가

- 작은 모델·작은 batch일수록 GPU 시간이 짧아 CPU overhead 비중이 크고, overlap 효과도 큽니다.
- `--disable-overlap-schedule`로 끌 수 있습니다 (디버깅·비교용). 실습 3에서 0.6B 모델로 비교합니다.

---

## 6. 📖 우선순위 스케줄링과 preemption

- `--enable-priority-scheduling`: 요청에 `priority` 값을 주면 대기열을 우선순위로 정렬합니다 (`_sort_by_priority_and_fcfs`).
- 우선순위가 충분히 높은 요청이 오면 실행 중인 낮은 우선순위 요청을 **preempt**(쫓아내기)할 수 있습니다 (`--priority-scheduling-preemption-threshold`, 기본 차이 10 이상). 메커니즘은 retraction과 같습니다.
- 용도: 유료/무료 사용자 구분, 대화형 요청 vs 배치 작업.
- `--retraction-policy priority`: 메모리 부족 시 낮은 우선순위부터 쫓아냄.

---

## 7. ★ 손잡이 정리와 진단 순서

### 7.1 손잡이

| 플래그 | 늘리면 | 줄이면 | 언제 만지나 |
|---|---|---|---|
| `--mem-fraction-static` | KV pool↑ → 동시 처리↑ | OOM 위험↓ | 시작 시 OOM이면 줄임 |
| `--max-total-tokens` | (디버깅용) pool 직접 지정 | | 실험에서 메모리 압박을 일부러 만들 때 |
| `--max-running-requests` | 동시 decode 수 상한↑ | tail latency 안정 | `token usage`가 낮은데 `#queue-req`가 쌓일 때 확인 |
| `--chunked-prefill-size` | 긴 prefill 효율↑, 가해자 TTFT↓ | activation 메모리↓, (mixed/interval과 함께일 때) 피해자 ITL↓ | 긴 입력이 섞이는 서비스, 시작 시 OOM |
| `--enable-mixed-chunk` | | | ITL 스파이크가 문제일 때 (조각과 decode를 섞음) |
| `--prefill-decode-interval` | prefill 뒤 decode를 N번 보장 | | ITL 스파이크, prefill이 연달아 decode를 굶길 때 |
| `--max-prefill-tokens` | prefill batch 크기↑ | | 짧은 요청이 폭주할 때 |
| `--schedule-conservativeness` | retraction↓, 동시 처리↓ | 동시 처리↑, retraction↑ | 로그에 Retract가 자주 찍힐 때 |
| `--schedule-policy` | `fcfs`(기본) / `lpm` / … | | prefix 공유가 많고 메모리가 빠듯할 때 lpm |
| `--disable-overlap-schedule` | | | 디버깅, overlap 효과 측정 |

### 7.2 진단 순서 (`common/logparse.py` 결과로)

```text
#queue-req가 큼 ─┬─ token usage 높음  → 메모리 부족: pool↑ / 동시성↓ / KV 양자화(Session 5) / max_new_tokens 현실화
                 └─ token usage 낮음  → 설정 상한: max-running-requests, max-prefill-tokens 확인
Retract가 자주 찍힘 → schedule-conservativeness↑ 또는 max_new_tokens를 현실적으로
ITL p99만 나쁨      → 긴 prefill이 끼어드는 중: mixed-chunk 또는 prefill-decode-interval + 작은 chunk
TTFT만 나쁨         → 대기열(#queue-req) 또는 긴 입력. prefix cache hit 확인 (Session 3)
작은 batch에서 ITL이 이론보다 훨씬 김 → CPU overhead: CUDA graph(Session 5), overlap
```

---


## ★ 실습 전 예측 질문

실습 직전에 노트에 **숫자로** 적으세요. 실습 후 실제 값과 비교하는 것이 이 회차에서 가장 많이 배우는 순간입니다.

1. 16명이 decode 중일 때 16K 프롬프트가 들어오면, chunk 없음 / 4096 / 1024 / 1024+mixed / 1024+interval에서 피해자의 **최악 멈춤**은 각각 몇 ms일까? (§4.3의 ⚠️를 기억하세요)
2. pool 30,000 token, 요청 128개, `max_new_tokens=1024`. `ignore_eos=True`일 때 처음에 몇 개를 동시에 받아들일까? `False`라면?
3. retraction이 일어나면 throughput은 늘까 줄까? 어떤 요청의 E2E가 가장 나빠질까?
4. Qwen3-0.6B로 바꾸고 동시성 4로 돌리면 overlap ON/OFF의 ITL 차이는 1.7B보다 클까 작을까?

---

## 8. 요약 · 자기 점검 · 더 읽을거리

### 요약

1. prefill batch는 정렬된 대기열 앞에서부터 `add_one_req`로 채우고, 하나라도 예산을 넘으면 break.
2. 제한은 메모리(`rem_total_tokens`), 계산량(`max_prefill_tokens`, `chunked_prefill_size`), 요청 슬롯.
3. 실행 중 요청의 미래 생성분은 `max_new × new_token_ratio`로 예약. ratio는 0.7에서 시작해 0.098까지 줄고 retraction 때 다시 올라감.
4. `ignore_eos=True`면 ratio = 1.0 — 벤치마크 설정이 스케줄러 동작을 바꾼다.
5. decode에 1칸이 모자라면 retraction: 생성 적게 한 요청, 입력 긴 요청부터. 비용은 재-prefill과 사용자 쪽 멈춤.
6. 긴 prefill은 다른 요청의 ITL을 멈춘다. chunked prefill은 조각을 나누지만 기본 설정에선 조각이 연달아 돌아 멈춤이 그대로다. mixed chunk(조각+decode 섞기)나 prefill-decode interval(조각 사이 decode 보장)이 있어야 멈춤이 조각 하나 크기로 줄어든다. 대가는 가해자 TTFT.
7. overlap scheduling은 batch N 실행 중 N+1을 준비해 CPU 시간을 숨긴다. 대가는 한 step 늦은 결과 처리.

### 자기 점검 문제

1. pool 50,000, 빈 슬롯 30,000, evictable 5,000. running 30개가 각각 `max_new=2048`에 이미 1000 생성, ratio 0.2. 대기 요청은 입력 500, `max_new=2048`. 이번 iteration 승인 수는? (`max_prefill_tokens=16384`도 고려)
2. 위에서 모든 요청이 `ignore_eos=True`라면?
3. 대기열 맨 앞에 입력 20,000짜리 요청이 있고 그 뒤에 입력 100짜리가 50개 있다. chunked prefill이 없다면 어떤 일이 생기나? 있다면?
4. retraction 정렬 키가 (생성 수 많은 것부터 쫓아냄)이라면 무엇이 나빠질까?
5. mixed chunk를 켰더니 ITL p99는 좋아졌는데 ITL p50이 나빠졌다. 왜?
6. overlap scheduling에서 EOS로 끝난 요청이 한 step 더 decode되면 KV 슬롯은 어떻게 되나?
7. `#queue-req`가 계속 30 이상인데 `token usage`는 0.3이다. 가장 먼저 의심할 것은?

<details>
<summary>답</summary>

1. running 예약 = 30 × (2048 − 1000) × 0.2 = 6,288. rem_total = 30,000 + 5,000 − 6,288 = 28,712. 비용 = 500 + 2048 = 2,548 → 메모리 기준 11개 (11 × 2,548 = 28,028). 계산량: 11 × 500 = 5,500 ≤ 16,384 OK → 약 11개.
2. running 예약 = 30 × 1048 × 1.0 = 31,440 > 35,000 − … → rem_total = 3,560 → 1개 (2,548). 낙관 예약이 없으면 거의 못 받는다.
3. chunk 없음: 대기열 첫 요청은 `can_run_list`가 비어 있을 때 계산량 예산을 넘어도 받아들여지므로 20,000 token prefill 한 번이 iteration을 통째로 점유 → 그동안 모든 decode 멈춤, 작은 요청 50개도 대기. chunk 4096: 20,000이 5조각으로 연달아 돌고 조각마다 chunk 예산을 다 쓰므로 작은 요청들은 마지막 조각(3,616 token) iteration에 남는 예산으로 일부 들어가거나 그 뒤에 들어감. decode는 mixed/interval이 없으면 여전히 전체 시간 동안 멈춤.
4. 거의 다 끝나 가는 요청을 쫓아내 가장 많은 진행분을 버림 → 재계산 비용 최대, 그 요청 사용자의 E2E 최악.
5. decode token이 매 iteration 1K prefill 조각과 같은 forward에 섞여 step 하나가 길어짐 → 평소 ITL(p50)이 길어짐. 대신 '멈춤'이 사라져 p99가 개선.
6. 그 step을 위해 1칸이 할당됐지만, 결과 처리에서 버려지고 요청 정리 시 `release_kv_cache`에서 함께 반납된다.
7. 메모리가 아니라 상한: `max_running_requests` (시작 로그 값 확인), 또는 `max_prefill_tokens`/chunk 예산. LPM 정렬 비용(대기열 > 128)도 확인.
</details>

### 더 읽을거리

- Agrawal et al., *Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference* (OSDI '24) — chunked prefill과 stall-free scheduling
- Zhong et al., *DistServe* (OSDI '24), Qin et al., *Mooncake* (FAST '25) — prefill/decode 분리
- SGLang 블로그: overlap scheduler, v0.4 릴리스 노트 (zero-overhead batch scheduler)
- SGLang 코드: `managers/schedule_policy.py` (`PrefillAdder`), `mem_cache/prefill_budget.py`, `managers/scheduler_components/new_token_ratio_tracker.py`

### 다음 회차 예고

지금까지는 "무엇을 GPU에 보낼까"(스케줄러)를 봤습니다. 다음 회차는 **GPU 위에서** 한 step이 어떻게 실행되는지 — ForwardBatch, attention 커널, CUDA graph, 양자화, speculative decoding — 를 봅니다.
