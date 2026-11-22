# Session 5 이론 — Model Execution & 커널 수준 최적화

> 50분. §3(step 시간 분해)이 이 회차의 중심입니다. 모든 최적화를 "step 시간의 어느 조각을 줄이는가"로 분류합니다.

| 절 | 내용 | 시간 |
|---|---|---|
| §0 | 지난 회차 연결 | 2분 |
| §1 | ScheduleBatch → ForwardBatch: GPU가 받는 것 | 7분 |
| §2 | Attention backend: ragged prefill vs paged decode | 8분 |
| §3 | decode step 하나를 분해하기 | 6분 |
| §4 | CUDA graph | 8분 |
| §5 | 양자화: weight FP8, KV FP8 | 8분 |
| §6 | Speculative decoding | 8분 |
| §7 | 최적화를 평가하는 법 + 예측 질문 | 3분 |

---

## §0. 지난 회차 연결

Session 1 과제의 가설 "SGLang이 HF보다 빠른 이유" 중 아직 검증하지 않은 것들 — "커널이 좋아서", "CUDA graph 때문" — 을 오늘 잽니다.
그리고 Session 1 §4의 roofline으로 돌아갑니다: **decode는 memory-bound**이므로, 읽는 bytes를 줄이거나(양자화), 한 번 읽을 때 더 많이 처리하거나(batching, speculative), 읽는 사이의 빈틈을 없애는(CUDA graph) 것이 최적화의 전부입니다.

---

## §1. ScheduleBatch → ForwardBatch: GPU가 받는 것

```text
ScheduleBatch (scheduler, Python 객체: Req 리스트, 캐시 노드, sampling 정보 …)
      │  TpModelWorker.forward_batch_generation  →  ForwardBatch.init_new(batch, model_runner)
      ▼
ForwardBatch (GPU 텐서 묶음)                        model_executor/forward_batch_info.py
  forward_mode       EXTEND | DECODE | MIXED | IDLE
  input_ids          [이번에 계산할 token 총수]           ← 요청들을 padding 없이 이어 붙임 (ragged)
  positions          [같은 길이]                         ← RoPE용 위치
  seq_lens           [batch]                            ← 요청별 전체 길이 (캐시 포함)
  extend_prefix_lens [batch]                            ← (EXTEND) 캐시에서 온 길이
  extend_seq_lens    [batch]                            ← (EXTEND) 이번에 새로 계산할 길이
  req_pool_indices   [batch]                            ← req_to_token 표의 행 번호
  out_cache_loc      [이번에 계산할 token 총수]           ← 새 K, V를 써 넣을 슬롯 번호
      │  ModelRunner.forward → model.forward (models/qwen3.py)
      ▼
logits → Sampler → next token ids
```

예: 요청 3개 — A는 prefix 100 캐시 + 새 20, B는 캐시 없이 새 50, C는 새 10 → EXTEND batch

```text
input_ids          = [A의 20개 | B의 50개 | C의 10개]     (총 80)
extend_prefix_lens = [100, 0, 0]
extend_seq_lens    = [20, 50, 10]
seq_lens           = [120, 50, 10]
out_cache_loc      = 80개의 새 슬롯 번호
```

DECODE batch는 단순합니다: `input_ids`는 요청당 1개, `out_cache_loc`도 요청당 1개.

> v0.5.23 메모: 예전 버전에는 중간 단계 `ModelWorkerBatch`가 있었지만 지금은 `ScheduleBatch`에서 `ForwardBatch`를 바로 만듭니다.

---

## §2. Attention backend: ragged prefill vs paged decode

모델 코드의 attention은 `RadixAttention`(`layers/radix_attention.py`)을 부르는데, 이건 **얇은 껍데기**입니다. 실제 계산은 backend가 합니다.

```python
# layers/attention/base_attn_backend.py (개념)
class AttentionBackend:
    def init_forward_metadata(self, forward_batch): ...   # 이번 batch에 맞는 인덱스/표를 미리 준비 (layer마다 반복하지 않도록)
    def forward_extend(self, q, k, v, layer, forward_batch): ...
    def forward_decode(self, q, k, v, layer, forward_batch): ...
```

| | EXTEND (prefill) | DECODE |
|---|---|---|
| Query | 요청마다 여러 개 (ragged) | 요청마다 1개 |
| Key/Value | 캐시된 prefix(흩어진 슬롯) + 새 token | 전부 캐시 (흩어진 슬롯, `req_to_token`으로 찾아감) |
| 병목 | 연산 (QKᵀ가 큼) | KV 읽기 |
| 커널이 하는 일 | causal mask + prefix/새 token 두 부분 합치기 | 요청마다 길이가 다른 KV를 표를 따라 읽기 |

v0.5.23 backend 선택 (`--attention-backend`):

| backend | 언제 | 비고 |
|---|---|---|
| `flashinfer` | Ampere/Ada(4090 포함) **기본** | JIT 컴파일 → 첫 시작이 느릴 수 있음 |
| `fa3` | Hopper(H100) 기본 | 4090에서는 사용 불가 |
| `fa4`, `trtllm_mha` | Blackwell 계열 | |
| `triton` | 어디서나 | Python으로 쓴 Triton 커널 — 읽기 쉬워 공부용으로 좋음 |
| `torch_native` | 디버깅용 | 느림 |

---

## §3. decode step 하나를 분해하기

```text
decode step 시간 =  CPU 준비 (스케줄링, ForwardBatch 만들기)
                 +  커널 launch overhead (layer 28개 × 커널 수십 개)      ← §4 CUDA graph
                 +  linear layer: weight 읽기 (3.4 GB / 1 TB/s ≈ 3.4 ms)  ← §5 weight 양자화
                 +  attention: KV 읽기 (batch × context × 112 KiB)        ← §5 KV 양자화, GQA
                 +  sampling
             ÷  이 step에서 확정되는 token 수 (보통 요청당 1)              ← §6 speculative decoding
```

모든 최적화가 이 식의 어느 항을 공략하는지 말할 수 있어야 합니다. 오늘 실습의 `onebatch_to_csv.py`는 측정한 step 시간으로 **"달성한 메모리 대역폭 / 최대 대역폭"** 을 계산합니다.
batch=1에서 이 값이 50%라면, 나머지 50%의 시간은 메모리가 아닌 무언가(launch overhead, 작은 커널의 비효율)에 쓰이고 있다는 뜻입니다.

---

## §4. CUDA graph

### 문제

decode step 하나 = 커널 수백 개. Python이 커널을 하나씩 launch할 때마다 수 μs의 CPU 시간이 듭니다.
GPU 커널 하나가 수십 μs밖에 안 걸리는 작은 batch에서는 **GPU가 다음 명령을 기다리며 놉니다.**

### 해결

```text
capture (시작 시 한 번):  batch 크기 1, 2, 4, 8, …, 48 각각에 대해 decode forward 1회를 '녹화'
replay  (매 step):        입력 텐서만 고정 버퍼에 복사 → graph.replay() 한 번 → 커널 수백 개가 CPU 개입 없이 실행
```

- 녹화할 때의 텐서 **주소와 모양이 고정**되어야 합니다. 그래서 batch 크기별로 따로 녹화하고, 실제 batch가 13이면 16짜리 graph에 **padding**해서 실행합니다.
- 4090(24 GB급)의 기본 decode graph 최대 batch는 **48** (`arg_groups/memory_hook.py`). 그보다 큰 batch는 eager로 실행 → 서버 로그의 `cuda graph: False`.
- 왜 decode만? prefill은 매번 길이가 달라 모양이 고정되지 않습니다. v0.5.23은 prefill에도 **"breakable" CUDA graph**(일부 구간만 녹화)를 CUDA에서 기본으로 켭니다 — 시작 로그에서 확인해 보세요.
- 대가: capture 시간(서버 시작이 느려짐), graph 버퍼 메모리(KV pool이 그만큼 작아짐).

플래그 (v0.5.23):

| 하고 싶은 것 | 플래그 |
|---|---|
| decode CUDA graph 끄기 | `--cuda-graph-backend-decode disabled` (또는 `--disable-decode-cuda-graph`) |
| prefill CUDA graph 끄기 | `--cuda-graph-backend-prefill disabled` |
| 둘 다 (구 방식) | `--disable-cuda-graph` (deprecated 경고와 함께 동작) |
| 최대 batch 바꾸기 | `--cuda-graph-max-bs-decode N` |

---

## §5. 양자화 — 읽는 bytes를 줄인다

decode가 memory-bound라면 **같은 정보를 더 적은 bytes로** 읽는 것이 곧 속도입니다.

| 무엇을 | 플래그 | 효과 | 대가 |
|---|---|---|---|
| weight → FP8 | `--quantization fp8` (로딩 시 온라인 양자화) 또는 FP8 체크포인트 | weight 읽기 ½ → 작은 batch decode 최대 ~2배, weight 메모리 ½ → KV pool↑ | 정확도 소폭↓, FP8 tensor core 필요 (Ada/Hopper 이상: 4090 OK, 3090 X) |
| KV → FP8 | `--kv-cache-dtype fp8_e4m3` | KV 메모리 ½ → **담을 수 있는 token 2배**, 긴 context decode의 KV 읽기 ½ | 정확도 소폭↓, backend가 FP8 KV를 지원해야 함 (안 되면 오히려 느려짐) |
| weight → INT4 (AWQ/GPTQ) | 양자화된 체크포인트 | weight ¼ | 정확도↓, 큰 batch에서 dequant 비용 |

**계산 예 (Qwen3-1.7B, 4090):**

```text
BF16: weight 3.4 GB, KV pool ≈ 13.6 GiB ≈ 12만 token
FP8 weight: weight 1.7 GB → KV pool +1.7 GB ≈ +1.5만 token, batch=1 decode 하한 3.4 → 1.7 ms
FP8 KV    : KV 56 KiB/token → 같은 pool에 ≈ 24만 token
```

8B 모델을 4090에 올리는 문제(Session 1 과제 Part A-2)에 대한 답 중 하나가 여기 있습니다.

> 정확도도 측정해야 공정합니다. 이 스터디에서는 "같은 프롬프트 몇 개의 출력을 눈으로 비교"하는 정도만 합니다.
> 실제로는 `sglang/test`나 lm-eval 같은 평가로 확인합니다.

---

## §6. Speculative decoding — 한 번 읽을 때 여러 token 확정하기

decode step 하나에서 요청당 1 token만 나오는 이유는 autoregressive라서입니다. 그런데 **검증은 병렬**로 할 수 있습니다.

```text
1. draft: 싼 방법으로 다음 k개 token을 추측       "the cat sat on"
2. verify: target 모델이 k개를 한 번의 forward로 계산 (prefill처럼, KV 읽기 1회)
3. 앞에서부터 target의 선택과 일치하는 만큼 확정, 처음 틀린 곳에서 target의 token으로 교체
   → 한 step에 평균 (accept length)개의 token
```

- step 하나가 약간 비싸지지만(검증할 token k개), memory-bound 구간에서는 **거의 공짜**입니다 (Session 1 §4: batch 대신 token 수로 AI를 올리는 것).
- draft 방식 (`--speculative-algorithm`): `EAGLE`/`EAGLE3`(작은 draft head 학습 필요), `STANDALONE`(작은 모델), `NGRAM`(학습 불필요 — 이미 본 token열에서 검색), 등.
- **언제 손해인가?** batch가 커서 이미 compute-bound이면 검증용 추가 token이 공짜가 아닙니다. 추측이 잘 안 맞으면(accept length ≈ 1) 순수 손해.
- 로그: speculative를 켜면 `Decode batch` 줄에 `accept len: x.xx, accept rate: …`가 찍힙니다.
- greedy(temperature 0)에서 speculative decoding은 **출력을 바꾸지 않습니다** — 검증 규칙이 target 모델의 선택만 받아들이기 때문입니다 (sampling일 때는 분포를 보존하는 rejection sampling).

---

## §7. 최적화를 평가하는 법

모든 ablation은 이 질문들에 답해야 합니다.

1. **어느 항을 줄이는가?** (§3 식) → 그렇다면 **어떤 workload에서** 효과가 크고, 어디서 사라지는가?
2. **대가는?** 정확도, 시작 시간, 메모리, 특정 GPU 필요
3. **비교 조건이 공정한가?** 같은 GPU, 같은 입출력 길이, 같은 동시성, warm-up, 반복

## 예측 질문

1. CUDA graph를 끄면 batch=1 decode step은 몇 배 느려질까? batch=64에서는?
2. batch=1에서 달성 대역폭은 최대(1 TB/s)의 몇 %일까? batch=128에서는?
3. FP8 weight는 batch=1 decode를 몇 배 빠르게 할까? batch=64에서도 같은 비율일까?
4. NGRAM speculative는 "코드 그대로 다시 쓰기" 작업에서 accept length가 몇일까? 창작 이야기에서는?
