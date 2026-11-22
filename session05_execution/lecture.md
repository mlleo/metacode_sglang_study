# Session 5 이론 — Model Execution & 커널 수준 최적화

> **핵심 질문:** GPU 위에서 decode step 하나는 무엇으로 이루어지고, 각 최적화는 그중 무엇을 줄이는가?

## 이 문서 사용법

- ★ 수업 핵심 / 📖 복습·심화
- 코드 발췌는 **SGLang v0.5.23** 기준이며 간추린 것입니다.
- 수업용 추천 구성: §0 → §1.1–1.3 → §2.1–2.3 → §3 → §4.1–4.3 → §5.1–5.3 → §6.1–6.3 → §8

## 목차

0. ★ 지난 회차 연결
1. ★ ScheduleBatch → ForwardBatch: GPU가 받는 것
2. ★ Attention backend: ragged prefill vs paged decode
3. ★ decode step 하나를 분해하기
4. ★ CUDA graph
5. ★ 양자화: weight FP8, KV FP8
6. ★ Speculative decoding
7. 📖 torch.compile과 커널 퓨전
8. ★ 최적화를 평가하는 법
9. 요약 · 자기 점검 · 더 읽을거리

---

## 0. ★ 지난 회차 연결

Session 1 과제의 가설 "SGLang이 HF보다 빠른 이유" 중 아직 검증하지 않은 것들 — "커널이 좋아서", "CUDA graph 때문" — 을 오늘 잽니다.
그리고 Session 1 §5의 roofline으로 돌아갑니다. **decode는 memory-bound**이므로 최적화는 세 갈래뿐입니다:

1. **읽는 bytes를 줄인다** — 양자화(§5), GQA/MLA 같은 모델 구조
2. **한 번 읽을 때 더 많이 처리한다** — batching(Session 2), speculative decoding(§6)
3. **읽는 사이의 빈틈을 없앤다** — CUDA graph(§4), overlap(Session 4), 커널 퓨전(§7)

---

## 1. ★ ScheduleBatch → ForwardBatch: GPU가 받는 것

### 1.1 변환 경로

```text
ScheduleBatch (scheduler, Python 객체: Req 리스트, 캐시 노드, sampling 정보 …)
      │  TpModelWorker.forward_batch_generation(batch)            managers/tp_worker.py
      │    → ForwardBatch.init_new(batch, model_runner)            model_executor/forward_batch_info.py
      ▼
ForwardBatch (GPU 텐서 묶음)
      │  ModelRunner.forward → _forward_raw                       model_executor/model_runner.py
      │    ├─ CUDA graph로 돌 수 있으면 → decode_cuda_graph_runner.execute()   (§4)
      │    └─ 아니면 → model.forward(input_ids, positions, forward_batch)       models/qwen3.py
      ▼
logits (LogitsProcessor) → next token ids (Sampler)
```

> v0.5.23 메모: 예전 버전에는 중간 단계 `ModelWorkerBatch`가 있었지만 지금은 `ScheduleBatch`에서 `ForwardBatch`를 바로 만듭니다.

### 1.2 ForwardBatch의 주요 필드

```python
# model_executor/forward_batch_info.py  (주요 필드만)
class ForwardMode(IntEnum):
    EXTEND = auto()     # prefill (캐시된 prefix 뒤에 새 token들을 이어 계산하는 일반화된 prefill)
    DECODE = auto()     # 요청당 1 token
    MIXED = auto()      # EXTEND + DECODE 섞음 (Session 4 §4.4)
    IDLE = auto()       # (분산 환경에서) 이 rank는 할 일 없음
    ...                 # TARGET_VERIFY, DRAFT_EXTEND 등 speculative 전용 모드

class ForwardBatch:
    forward_mode: ForwardMode
    batch_size: int
    input_ids: torch.Tensor          # [이번에 계산할 token 총수]  ← padding 없이 이어 붙임 (ragged)
    req_pool_indices: torch.Tensor   # [batch]  ← req_to_token 표의 행 번호 (Session 3)
    seq_lens: torch.Tensor           # [batch]  ← 요청별 전체 길이 (캐시 포함, 이번 token까지)
    out_cache_loc: torch.Tensor      # [이번에 계산할 token 총수]  ← 새 K, V를 써 넣을 슬롯 번호
    positions: torch.Tensor          # [같은 길이]  ← RoPE용 위치
    extend_prefix_lens: ...          # [batch]  (EXTEND) 캐시에서 온 길이
    extend_seq_lens: ...             # [batch]  (EXTEND) 이번에 새로 계산할 길이
    ...
```

### 1.3 예제: EXTEND batch 채우기

요청 3개 — A는 prefix 100 캐시 + 새 20, B는 캐시 없이 새 50, C는 새 10.

```text
input_ids          = [A의 20개 | B의 50개 | C의 10개]                 (총 80)
positions          = [100..119 | 0..49 | 0..9]                       ← A는 100부터!
extend_prefix_lens = [100, 0, 0]
extend_seq_lens    = [20, 50, 10]
seq_lens           = [120, 50, 10]
out_cache_loc      = 80개의 새 슬롯 번호 (allocator가 준 것)
req_pool_indices   = [A의 행, B의 행, C의 행]
```

- **positions**가 prefix 길이부터 시작하는 것에 주의. RoPE는 절대 위치를 쓰므로, cache hit 뒤의 token은 원래 위치를 받아야 캐시된 K와 일관됩니다.
- 이 batch의 logits는 **요청별 마지막 위치 3개만** 필요합니다:

```python
# layers/logits_processor.py: _get_pruned_states  (prefill, logprob 없음 — 원문)
last_index = torch.cumsum(logits_metadata.extend_seq_lens, dim=0) - 1     # [19, 69, 79]
pruned_states = hidden_states[last_index]                                 # 80개 중 3개만 LM head로
```

ragged batch에서 "요청별 마지막 위치"를 cumsum 한 줄로 찾는 것 — 이 스터디에서 가장 짧고 중요한 코드 중 하나입니다.
(Session 1 §1.2의 "LM head는 마지막 위치만"이 구현된 곳)

DECODE batch는 단순합니다: `input_ids`는 요청당 1개, `positions = seq_lens − 1`, `out_cache_loc`도 요청당 1개, logits는 전부 필요.

### 1.4 모델 쪽에서 본 한 layer (📖)

```python
# models/qwen3.py: Qwen3Attention  (간추림)
def forward(self, positions, hidden_states, forward_batch):
    qkv, _ = self.qkv_proj(hidden_states)                    # Q, K, V 투영을 행렬곱 하나로 (퓨전)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm)      # Qwen3는 Q, K에 RMSNorm
    q, k = self.rotary_emb(positions, q, k)                   # RoPE
    attn_output = self.attn(q, k, v, forward_batch)           # ★ RadixAttention → backend
    output, _ = self.o_proj(attn_output)
    return output
```

`self.attn`은 `RadixAttention`(`layers/radix_attention.py`)입니다. 이름과 달리 radix tree 로직은 없고, **attention backend로 위임하는 얇은 껍데기**입니다.
(radix tree가 만든 결과 — 어떤 슬롯을 읽을지 — 는 이미 `req_to_token`과 `ForwardBatch`에 들어 있습니다.)

---

## 2. ★ Attention backend: ragged prefill vs paged decode

### 2.1 backend 계약

```python
# layers/attention/base_attn_backend.py  (간추림)
class AttentionBackend(ABC):
    def init_forward_metadata(self, forward_batch):
        """이번 batch에 맞는 인덱스·표를 한 번 준비 (28개 layer가 공유)"""
    def forward(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        if forward_batch.forward_mode.is_decode():
            return self.forward_decode(q, k, v, layer, forward_batch, save_kv_cache)
        else:
            return self.forward_extend(q, k, v, layer, forward_batch, save_kv_cache)
```

`init_forward_metadata`가 중요합니다. batch마다 "요청 i의 KV는 슬롯 어디어디에 있다"는 표를 만드는 일은 layer 28개에서 똑같으므로 **iteration당 한 번**만 합니다.

### 2.2 KV 위치를 커널에 넘기는 방법: CSR

attention 커널은 요청마다 길이가 다른 KV를 읽어야 합니다. 표준 표현은 **CSR(compressed sparse row)** 형식입니다.

```text
seq_lens   = [5, 3, 4]
kv_indptr  = [0, 5, 8, 12]                ← cumsum: 요청 i의 KV는 kv_indices[indptr[i] : indptr[i+1]]
kv_indices = [7 8 9 2 3 | 7 8 5 | 11 12 13 14]   ← req_to_token에서 각 요청의 앞 seq_len개를 이어 붙임
```

```python
# layers/attention/triton_backend.py: _fill_kv_indptr_and_indices  (간추림)
kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
self.kv_index_translator.pack_read_stream(          # req_to_token[req_pool_indices[i], :seq_lens[i]] 를
    plan, req_pool_indices=req_pool_indices,         # GPU 커널로 kv_indices에 이어 붙임
    seq_lens=seq_lens, indptr=kv_indptr, out=kv_indices)
```

여기서 Session 3의 **두 단계 풀**이 Session 5의 **커널**과 만납니다: radix tree → `req_to_token` → `kv_indices` → 커널이 `k_buffer[layer][kv_indices]`를 읽음.

### 2.3 EXTEND와 DECODE는 다른 커널이다

| | EXTEND (prefill) | DECODE |
|---|---|---|
| Query | 요청마다 여러 개 (ragged) | 요청마다 1개 (GQA면 head 그룹당 몇 개) |
| Key/Value | 캐시된 prefix(흩어진 슬롯) + 새 token(연속) | 전부 캐시 (흩어진 슬롯) |
| 병목 | 연산 (QKᵀ가 큼) | KV 읽기 |
| 커널이 하는 일 | causal mask + prefix/새 token 두 부분 합치기 | 요청마다 길이가 다른 KV를 표를 따라 읽기 |
| 대표 기법 | FlashAttention (타일링 + online softmax) | Flash-decoding (split-KV: 긴 KV를 여러 블록으로 나눠 병렬로 읽고 합침) |

**split-KV가 왜 필요한가 (📖):** batch=1, context 32K decode에서 요청 하나 × head 몇 개만으로는 GPU의 SM(4090: 128개)을 다 못 씁니다. 그래서 KV를 여러 조각으로 나눠 여러 SM이 나눠 읽고 마지막에 softmax를 합칩니다 (`merge_state`). 긴 context·작은 batch에서 decode attention 속도를 좌우합니다.

### 2.4 FlashInfer의 plan / run (📖)

FlashInfer는 "계획(plan)"과 "실행(run)"을 분리합니다.

```text
init_forward_metadata:  wrapper.plan(kv_indptr, kv_indices, head 수, …)   ← CPU 쪽 스케줄 계산 (batch당 1회)
각 layer:              wrapper.forward(q, kv_cache)                         ← GPU 실행 (layer마다)
```

plan은 batch 모양에 따라 "어떤 요청의 어떤 KV 조각을 어느 블록이 맡을지"를 정합니다. 28 layer 동안 재사용되므로 비용이 분산됩니다.

### 2.5 v0.5.23 backend 선택

`--attention-backend`:

| backend | 언제 | 비고 |
|---|---|---|
| `flashinfer` | Ampere/Ada(4090 포함) **기본** | JIT 컴파일 → 첫 시작이 느릴 수 있음 |
| `fa3` | Hopper(H100) 기본 | 4090에서는 사용 불가 |
| `fa4`, `trtllm_mha` | Blackwell 계열 | |
| `triton` | 어디서나 | Python으로 쓴 Triton 커널 — 읽기 쉬워 공부용으로 좋음 (커널 본체: `python/sglang/kernels/ops/attention/decode_attention.py`, `extend_attention.py`) |
| `torch_native`, `flex_attention` | 디버깅·실험용 | 느림 |

`--prefill-attention-backend`와 `--decode-attention-backend`로 단계별로 다르게 고를 수도 있습니다.

---

## 3. ★ decode step 하나를 분해하기

```text
decode step 시간 =  CPU 준비 (스케줄링, ForwardBatch 만들기, plan)              ← overlap(S4)으로 숨김
                 +  커널 launch overhead (layer 28개 × 커널 수십 개)            ← §4 CUDA graph, §7 퓨전
                 +  linear layer: weight 읽기 (3.44 GB / 1 TB/s ≈ 3.4 ms)       ← §5 weight 양자화
                 +  attention: KV 읽기 (Σ context × 112 KiB / 1 TB/s)           ← §5 KV 양자화, GQA/MLA
                 +  sampling (vocab 15만 × batch)
             ÷  이 step에서 확정되는 token 수 (보통 요청당 1)                     ← §6 speculative decoding
```

모든 최적화가 이 식의 어느 항을 공략하는지 말할 수 있어야 합니다.

### 3.1 달성 대역폭 — 얼마나 하한에 가까운가

실습의 `onebatch_to_csv.py`는 측정한 step 시간으로 다음을 계산합니다:

```text
step bytes  ≈ weight bytes + batch × 평균 context × KV bytes/token
달성 대역폭 = step bytes / 측정한 step 시간
BW%         = 달성 대역폭 / GPU 최대 대역폭 (4090 ≈ 1008 GB/s)
```

해석:

| BW% | 의미 |
|---|---|
| 80–90% | 메모리 대역폭을 거의 다 쓰는 중. 더 빠르게 하려면 bytes를 줄여야 함 (양자화) |
| 40–60% | 시간의 절반 가까이가 메모리 외의 것: launch overhead, 작은 커널 비효율, sampling |
| 20% 이하 | overhead 지배. CUDA graph가 꺼져 있거나, batch가 너무 작아 커널이 GPU를 못 채움 |

실제 DRAM 대역폭은 스펙의 90% 정도가 현실적인 상한입니다.

### 3.2 예제 (숫자는 설명용 가정)

```text
batch=1, context 576, 측정 step 4.6 ms
  step bytes ≈ 3.44 GB + 1 × 576 × 114,688 B ≈ 3.51 GB
  달성 ≈ 3.51 / 0.0046 ≈ 763 GB/s → BW ≈ 76%

같은 조건 CUDA graph 끔, 측정 11 ms
  달성 ≈ 319 GB/s → BW ≈ 32%   ← 시간의 절반 이상이 launch overhead
```

---

## 4. ★ CUDA graph

### 4.1 문제

decode step 하나 = 커널 수백 개 (layer당 norm, qkv gemm, rope, attention, o gemm, norm, gate/up gemm, act, down gemm, residual … × 28).
Python + PyTorch가 커널을 하나씩 launch할 때마다 CPU 시간이 수 μs씩 듭니다.
작은 batch에서 커널 하나의 GPU 시간이 수 μs~수십 μs밖에 안 되면 **GPU가 다음 명령을 기다리며 놉니다** (Session 1 §5.7).

### 4.2 해결: 녹화하고 재생하기

```text
capture (시작 시 한 번):  batch 크기 1, 2, 4, 8, …, 48 각각에 대해 decode forward 1회를 '녹화'
replay  (매 step):        입력 텐서만 고정 버퍼에 복사 → graph.replay() 한 번 → 커널 수백 개가 CPU 개입 없이 실행
```

```python
# model_executor/model_runner.py: _forward_raw  (간추림)
can_run_graph = (forward_batch.forward_mode.is_cuda_graph()          # DECODE 등
                 and self.decode_cuda_graph_runner
                 and self.decode_cuda_graph_runner.can_run_graph(forward_batch))
if can_run_graph:
    return self.decode_cuda_graph_runner.execute(forward_batch)      # 고정 버퍼에 복사 → replay
# 아니면 eager: 평소처럼 model.forward를 커널 하나씩 실행
```

```python
# model_executor/runner/decode_cuda_graph_runner.py: can_run_graph  (간추림)
cuda_graph_bs = forward_batch.batch_size
is_bs_supported = cuda_graph_bs <= self.max_bs        # padding 허용(기본)이면 max_bs 이하이기만 하면 됨
```

### 4.3 제약과 대가

- **모양과 주소가 고정**되어야 합니다. 녹화 때의 텐서 주소에 매번 입력을 복사해 넣고 재생합니다. 그래서 batch 크기별로 따로 녹화하고, 실제 batch가 13이면 16짜리 graph에 **padding**해서 실행합니다 (padding된 요청은 슬롯 0 — Session 3 §1.2의 더미 슬롯 — 에 씁니다).
- 4090(24 GB급)의 기본 decode graph 최대 batch는 **48** (`arg_groups/memory_hook.py`). 그보다 큰 batch는 eager로 실행 → 서버 로그의 `cuda graph: False`. 큰 batch에서는 커널 하나가 충분히 길어서 launch overhead 비중이 작으므로 괜찮다는 판단입니다.
- 시작 로그 `Capture target decode CUDA graph end. bs=[...] elapsed=… mem usage=…`에서 녹화한 batch 목록과 메모리를 확인할 수 있습니다.
- **대가**: capture 시간(서버 시작이 느려짐), graph 버퍼 메모리(그만큼 KV pool이 작아짐), 동적인 기능과의 충돌(일부 sampling 옵션, LoRA 조합 등).
- **왜 decode만?** prefill은 매번 길이가 달라 모양이 고정되지 않습니다. v0.5.23은 prefill에도 **"breakable" CUDA graph**(attention처럼 모양이 바뀌는 부분은 eager로 끊고, 나머지 구간만 녹화)를 CUDA에서 기본으로 켭니다 (`model_executor/cuda_graph_config.py: default_prefill_backend`).

### 4.4 플래그 (v0.5.23)

| 하고 싶은 것 | 플래그 |
|---|---|
| decode CUDA graph 끄기 | `--cuda-graph-backend-decode disabled` (또는 `--disable-decode-cuda-graph`) |
| prefill CUDA graph 끄기 | `--cuda-graph-backend-prefill disabled` |
| 둘 다 (구 방식) | `--disable-cuda-graph` (deprecated 경고와 함께 동작) |
| 최대 batch 바꾸기 | `--cuda-graph-max-bs-decode N` |
| 녹화할 batch 목록 직접 지정 | `--cuda-graph-bs-decode 1 2 4 8 …` |

---

## 5. ★ 양자화 — 읽는 bytes를 줄인다

### 5.1 왜 decode에 효과적인가

decode가 memory-bound라면 **같은 정보를 더 적은 bytes로** 읽는 것이 곧 속도입니다.
prefill(compute-bound)에서는 FP8 tensor core가 BF16보다 연산 처리량이 높을 때만 이득입니다 (4090, H100은 해당).

### 5.2 무엇을 양자화하나

| 무엇을 | 플래그 | 효과 | 대가 |
|---|---|---|---|
| weight → FP8 | `--quantization fp8` (로딩 시 온라인 양자화) 또는 FP8 체크포인트 | weight 읽기 ½ → 작은 batch decode 최대 ~2배, weight 메모리 ½ → KV pool↑ | 정확도 소폭↓, FP8 tensor core 필요 (Ada/Hopper 이상: 4090 OK, 3090 X) |
| KV → FP8 | `--kv-cache-dtype fp8_e4m3` | KV 메모리 ½ → **담을 수 있는 token 2배**, 긴 context decode의 KV 읽기 ½ | 정확도 소폭↓, backend가 FP8 KV를 지원해야 함 (안 되면 dequant 비용으로 오히려 느려짐) |
| weight → INT4 (AWQ/GPTQ) | 양자화된 체크포인트 (자동 인식) | weight ¼ | 정확도↓, 큰 batch에서 dequant 비용, 연산은 결국 BF16/FP16 |

**FP8 형식 (📖):** E4M3(지수 4비트, 가수 3비트 — 정밀도↑, 범위↓)과 E5M2(범위↑, 정밀도↓). 값을 FP8로 바꿀 때는 텐서(또는 채널, 블록)마다 **scale**을 곱해 FP8의 표현 범위에 맞춥니다. weight는 로딩 시 한 번, activation은 실행 중 동적으로(또는 미리 정한 scale로) 양자화합니다.

### 5.3 계산 예 (Qwen3-1.7B, 4090)

```text
BF16      : weight 3.44 GB, KV pool ≈ 13.6 GiB ≈ 12.7만 token, batch=1 decode 하한 3.4 ms
FP8 weight: weight 1.72 GB → KV pool +1.6 GiB ≈ +1.5만 token, batch=1 decode 하한 ≈ 1.7 ms
FP8 KV    : KV 56 KiB/token → 같은 pool에 ≈ 25만 token, context 2K × batch 32의 KV 읽기 7.5 → 3.8 GB
```

→ FP8 weight는 **작은 batch**에서 효과가 크고, batch가 커져 KV 읽기가 지배하면 효과가 줄어듭니다 (과제 주제).
→ FP8 KV는 **동시성·긴 context**에서 효과가 큽니다.

8B 모델을 4090에 올리는 문제(Session 1 과제 Part A-2)의 답: 8B × 1 byte = 8 GB → KV pool이 생깁니다.

### 5.4 정확도도 측정해야 공정하다

이 스터디에서는 "같은 프롬프트 몇 개의 출력을 눈으로 비교"하는 정도만 합니다. 실제로는:

- 작은 평가 세트(GSM8K, MMLU 일부)로 정확도 비교 — `sglang/test` 아래의 few-shot 평가 스크립트나 lm-eval
- greedy 출력이 BF16과 몇 token째에서 갈라지는지 (발산 지점)
- 작은 모델일수록, 긴 context일수록, 추론이 복잡할수록 양자화 손실이 커지는 경향

---

## 6. ★ Speculative decoding — 한 번 읽을 때 여러 token 확정하기

### 6.1 아이디어

decode step 하나에서 요청당 1 token만 나오는 이유는 autoregressive라서입니다. 그런데 **검증은 병렬**로 할 수 있습니다.

```text
1. draft : 싼 방법으로 다음 k개 token을 추측                 " sat on the mat"
2. verify: target 모델이 [마지막 확정 token + 추측 k개]를 한 번의 forward로 계산 (prefill처럼, weight 읽기 1회)
3. accept: 앞에서부터 target의 선택과 일치하는 만큼 확정, 처음 틀린 곳은 target의 token으로 교체
           → 한 step에 (accept된 수 + 1)개의 token
```

memory-bound 구간에서는 verify step(k+1 token)의 시간이 일반 decode step(1 token)과 거의 같습니다 — Session 1 §5.4에서 batch로 AI를 올린 것을 **token 수로** 하는 셈입니다.

### 6.2 기대 이득 계산

추측 token 하나가 맞을 확률이 독립적으로 α, 체인으로 k개를 추측하면, verify 한 번에 확정되는 token 수의 기댓값은

```text
E[tokens / step] = 1 + α + α² + … + α^k = (1 − α^(k+1)) / (1 − α)
```

| α | k = 2 | k = 4 | k = 8 |
|---:|---:|---:|---:|
| 0.5 | 1.75 | 1.94 | 2.00 |
| 0.7 | 2.19 | 2.77 | 3.20 |
| 0.9 | 2.71 | 4.10 | 6.13 |

```text
속도 향상 ≈ E[tokens / step] / (verify step 시간 / 일반 step 시간 + draft 시간 / 일반 step 시간)
```

- α가 낮으면 k를 늘려도 소용없습니다 (α=0.5면 2배가 상한).
- verify step이 비싸지면(큰 batch → compute-bound) 분모가 커져 이득이 사라지거나 손해가 납니다.
- 로그의 `accept len`이 E[tokens/step]에 해당합니다.

### 6.3 draft 방식들

| `--speculative-algorithm` | draft | 특징 |
|---|---|---|
| `EAGLE` / `EAGLE3` | target의 hidden state를 입력으로 받는 작은 draft head (학습 필요) | α가 높음. 공개된 draft 가중치가 있는 모델에 사용 |
| `NEXTN` / MTP | 모델에 내장된 multi-token prediction head (DeepSeek 등) | 모델이 지원해야 함 |
| `STANDALONE` | 같은 계열의 작은 모델 | 간단, draft 비용이 큼 |
| **`NGRAM`** | 이미 본 token 열(프롬프트 + 생성분)에서 n-gram을 검색 | **학습 불필요**, draft 비용 ≈ 0, 출력이 입력을 베끼는 작업에서 α가 매우 높음 |

NGRAM(📖): 체인 하나가 아니라 **후보 트리**(여러 갈래)를 한 번에 검증합니다 (`--speculative-ngram-max-bfs-breadth`, `--speculative-num-draft-tokens`). verify 커널은 tree attention mask로 갈래들을 한 forward에서 처리합니다.

### 6.4 성질

- greedy(temperature 0)에서 speculative decoding은 **출력을 바꾸지 않습니다** — target 모델이 고를 token만 받아들이기 때문.
- sampling에서는 **rejection sampling**으로 target 분포를 정확히 보존합니다 (Leviathan et al. 2023).
- verify 중에 KV는 추측 token 수만큼 쓰였다가 거절된 부분은 버려집니다 — KV 할당이 가변적이라 CUDA graph와 메모리 관리가 복잡해집니다 (그래서 문서 예시가 `--cuda-graph-max-bs-decode 8`처럼 작은 batch를 전제).

---

## 7. 📖 torch.compile과 커널 퓨전

- `--enable-torch-compile`: PyTorch 2의 컴파일러가 연속된 작은 연산(norm, activation, residual add …)을 **커널 하나로 합칩니다(퓨전)**. 커널 수가 줄어 launch overhead와 중간 결과의 메모리 왕복이 줄어듭니다.
- CUDA graph와 함께 쓰입니다 (`--torch-compile-max-bs`까지 컴파일).
- 대가: 시작 시 컴파일 시간(수 분), 캐시 관리. 효과는 모델·batch에 따라 다릅니다 → Session 5 과제의 "도전" 주제.
- SGLang은 이미 많은 연산을 손으로 퓨전한 커널을 씁니다 (예: QKV 투영을 행렬곱 하나로, norm+residual 퓨전, `sglang-kernel` 패키지의 커널들). 그래서 torch.compile의 추가 이득은 생각보다 작을 수 있습니다.

---

## 8. ★ 최적화를 평가하는 법

모든 ablation은 이 질문들에 답해야 합니다.

1. **어느 항을 줄이는가?** (§3 식) → 그렇다면 **어떤 workload에서** 효과가 크고, 어디서 사라지는가?
2. **대가는?** 정확도, 시작 시간, 메모리, 특정 GPU 필요, 다른 기능과의 충돌
3. **비교 조건이 공정한가?** 같은 GPU, 같은 입출력 길이, 같은 동시성, warm-up, 반복

| 최적화 | 줄이는 항 | 효과가 큰 곳 | 효과가 사라지는 곳 |
|---|---|---|---|
| CUDA graph | launch overhead | 작은 batch, 작은 모델 | 큰 batch (max_bs 초과, 커널이 길어짐) |
| FP8 weight | weight 읽기 | 작은 batch decode | 큰 batch·긴 context (KV 지배) |
| FP8 KV | KV 읽기·용량 | 긴 context, 높은 동시성 | 짧은 context, 낮은 동시성 |
| attention backend | attention 커널 효율 | 긴 context | 짧은 context |
| speculative | step당 token 수 | 작은 batch, 예측 가능한 출력 | 큰 batch, 창의적 출력 |
| overlap | CPU 준비 시간 | 작은 모델·작은 batch | GPU 시간이 긴 큰 batch |

이 표의 각 칸을 **데이터로 채우는 것**이 이번 과제입니다.

---


## ★ 실습 전 예측 질문

실습 직전에 노트에 **숫자로** 적으세요. 실습 후 실제 값과 비교하는 것이 이 회차에서 가장 많이 배우는 순간입니다.

1. CUDA graph를 끄면 batch=1 decode step은 몇 배 느려질까? batch=64에서는? batch=128에서는?
2. batch=1에서 달성 대역폭은 최대(1 TB/s)의 몇 %일까? batch=128에서는?
3. FP8 weight는 batch=1 decode를 몇 배 빠르게 할까? batch=64에서도 같은 비율일까?
4. NGRAM speculative는 "코드 그대로 다시 쓰기" 작업에서 accept length가 몇일까? 창작 이야기에서는?

---

## 9. 요약 · 자기 점검 · 더 읽을거리

### 요약

1. `ScheduleBatch` → `ForwardBatch`(ragged 텐서: input_ids, positions, seq_lens, out_cache_loc …) → `ModelRunner` → 모델 → logits → sampler.
2. prefill logits는 요청별 마지막 위치만: `cumsum(extend_seq_lens) − 1`.
3. `RadixAttention`은 껍데기, 실제 계산은 backend. KV 위치는 `req_to_token` → CSR(`kv_indptr`, `kv_indices`)로 커널에 전달.
4. EXTEND는 compute-bound(FlashAttention), DECODE는 KV 읽기(flash-decoding, split-KV).
5. decode step = CPU 준비 + launch overhead + weight 읽기 + KV 읽기 + sampling, ÷ step당 token 수.
6. CUDA graph는 launch overhead 제거. 고정 모양, padding, 4090 max_bs 48, 큰 batch는 eager.
7. FP8 weight는 작은 batch, FP8 KV는 긴 context·높은 동시성에 효과적.
8. speculative: E = (1 − α^(k+1)) / (1 − α). 작은 batch·예측 가능한 출력에서 이득.

### 자기 점검 문제

1. EXTEND batch: A(prefix 0, 새 30), B(prefix 200, 새 5). `positions`, `seq_lens`, `last_index`를 쓰라.
2. batch=1 decode가 9 ms, 모델 weight 3.44 GB, 4090. BW%는? 무엇을 먼저 의심하나?
3. CUDA graph가 켜져 있는데 batch 64에서 로그에 `cuda graph: False`가 찍힌다. 왜?
4. FP8 weight로 바꿨더니 batch=1 decode는 1.6배 빨라졌는데 batch=128에서는 1.1배였다. §3 식으로 설명하라.
5. NGRAM speculative에서 accept len이 4인데 throughput이 오히려 떨어졌다. 가능한 원인 두 가지.
6. α = 0.8, k = 3일 때 E[tokens/step]은?
7. 왜 RoPE의 positions를 prefix 길이부터 시작해야 하나? 0부터 시작하면 어떤 일이 생기나?

<details>
<summary>답</summary>

1. positions = [0..29 | 200..204], seq_lens = [30, 205], extend_seq_lens = [30, 5] → last_index = [29, 34].
2. 하한 3.4 ms 대비 약 38%. CUDA graph가 꺼져 있는지(로그 `cuda graph: False`), 첫 step·JIT 포함 여부, 다른 프로세스가 GPU를 쓰는지 확인.
3. 4090 기본 decode graph max_bs = 48. batch 64는 녹화된 크기를 넘어 eager로 실행된다.
4. batch=1에서는 step bytes의 대부분이 weight → 절반이 되면 크게 빨라짐. batch=128에서는 KV 읽기(BF16 그대로)가 step bytes의 대부분이라 weight 절감 효과가 희석됨 (그리고 compute 비중 증가).
5. (a) batch가 커서 verify step이 compute-bound → verify 비용이 일반 step의 여러 배 (b) CUDA graph max_bs가 작아(예: 8) 큰 batch에서 eager로 실행 (c) draft tree 구성·verify 커널 overhead.
6. (1 − 0.8⁴)/(1 − 0.8) = (1 − 0.4096)/0.2 = 2.95
7. 캐시된 prefix의 K는 원래 위치(0..199)로 RoPE가 적용돼 저장돼 있다. 새 token이 위치 0부터 시작하면 모델은 같은 위치에 token이 두 개 있는 것처럼 계산해 결과가 틀린다.
</details>

### 더 읽을거리

- Dao et al., *FlashAttention 1/2/3*; Tri Dao 외, *Flash-Decoding for long-context inference* (블로그)
- FlashInfer 논문/블로그 (plan/run, sorting-free sampling)
- NVIDIA, *CUDA Graphs* (CUDA Programming Guide 해당 장)
- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (2023); Li et al., *EAGLE* 1/2/3
- Micikevicius et al., *FP8 Formats for Deep Learning* (2022)
- SGLang 문서: Attention backend, Quantization, Quantized KV cache, Speculative decoding (`docs/advanced_features/`)

### 다음 회차 예고

이제 엔진의 거의 모든 층을 봤습니다. 마지막 회차는 "그래서 빨라졌다는 걸 어떻게 믿게 만드는가" — 벤치마크 방법론, 프로파일링, 비용 — 와 이 스터디 밖의 큰 그림입니다.
