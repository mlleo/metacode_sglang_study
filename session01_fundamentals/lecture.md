# Session 1 이론 — LLM 추론의 기초

> **핵심 질문:** token 하나는 어떻게 만들어지고, 왜 `model.generate()`만으로는 서빙을 못 하는가?

## 이 문서 사용법

이 문서는 **수업 + 복습용 완결 자료**입니다. 수업 시간에 다 다루지 않습니다.

- ★ 표시 절: 수업에서 다루는 핵심 (강사 추천 40–50분 구성)
- 📖 표시 절: 복습·심화. 수업 후 혼자 읽고 자기 점검 문제로 확인하세요
- 숫자 예제는 모두 **Qwen3-1.7B + RTX 4090** 기준입니다. 실습에서 직접 잰 값과 비교하세요

**수업용 추천 구성:** §1 → §3 → §4 → §5(§5.1–5.4) → §7 → §8 (나머지는 📖)

## 목차

1. ★ token 하나가 만들어지는 과정
2. 📖 Tokenizer — 텍스트가 숫자가 되는 곳
3. ★ Prefill vs Decode
4. ★ KV cache — 메모리를 써서 연산을 산다
5. ★ Compute-bound vs Memory-bound (roofline)
6. 📖 Sampling — logits에서 token으로
7. ★ 지표: TTFT / ITL / TPOT / E2E / Throughput
8. ★ 왜 `generate()`로는 서빙이 안 되는가 → 스터디 지도
9. 📖 SGLang 소개와 이 스터디의 버전 원칙
10. 요약 · 자기 점검 · 더 읽을거리

---

## 1. ★ token 하나가 만들어지는 과정

### 1.1 한 바퀴

```text
"The capital of France is"
        │ tokenizer
        ▼
[785, 6722, 315, 9625, 374]                 ← token id 5개 (예시 — 실습에서 직접 확인)
        │ embedding lookup  (vocab 151,936 × hidden 2,048 표에서 5줄을 꺼냄)
        ▼
x: [5, 2048]
        │ × 28 decoder layers
        │    ├─ RMSNorm
        │    ├─ Attention  : Q,K,V 투영 → RoPE → softmax(QKᵀ/√d)·V → O 투영
        │    ├─ (residual add)
        │    ├─ RMSNorm
        │    ├─ MLP        : gate/up (2048→6144) → SiLU·곱 → down (6144→2048)
        │    └─ (residual add)
        ▼
h: [5, 2048]
        │ final RMSNorm → LM head (hidden 2048 × vocab 151,936)   ← 마지막 위치만 필요!
        ▼
logits: [151936]                             ← vocab 전체에 대한 점수
        │ sampling (greedy / temperature / top-p …)
        ▼
next token = " Paris"   → 시퀀스 끝에 붙이고 처음으로 돌아간다
```

코드로 쓰면 이게 전부입니다.

```python
tokens = tokenize(prompt)
while True:
    logits = model(tokens)            # [seq_len, vocab]
    next_token = sample(logits[-1])   # 마지막 위치만 사용
    tokens.append(next_token)
    if next_token == EOS or len(tokens) == limit:
        break
```

### 1.2 이 그림에서 나오는 성능 직관 세 가지

1. **모델은 "다음 token 하나"만 예측한다.** 출력 512 token = forward 512번(원칙적으로). 그래서 출력 길이는 latency에 거의 **선형**으로 들어갑니다. 프롬프트 길이는 전혀 다른 방식으로 들어갑니다 (§3).
2. **매 step마다 vocab 15만 개의 점수를 만들고 하나만 씁니다.** LM head는 `2048 × 151,936 ≈ 3.1억` 파라미터짜리 행렬곱입니다. 프롬프트 5개 위치 전부에 대해 계산하면 낭비이므로, 서빙 엔진은 **마지막 위치만** 계산합니다. (SGLang: `layers/logits_processor.py`, HF transformers: `logits_to_keep=1` — 실습 코드에서 씁니다)
3. **layer는 28번 반복되는 같은 구조**입니다. 한 layer의 비용을 이해하면 전체를 이해한 것입니다.

### 1.3 Qwen3-1.7B의 파라미터는 어디에 있나 (📖)

`config.json`: hidden 2048, layers 28, q heads 16, kv heads 8, head_dim 128, intermediate 6144, vocab 151,936, tied embedding

| 부분 | 계산 | 파라미터 |
|---|---|---:|
| Q 투영 | 2048 × (16×128) | 4.19 M |
| K, V 투영 | 2 × 2048 × (8×128) | 4.19 M |
| O 투영 | (16×128) × 2048 | 4.19 M |
| MLP (gate, up, down) | 3 × 2048 × 6144 | 37.75 M |
| **layer 1개** | | **≈ 50.3 M** |
| **28 layers** | | **≈ 1.41 B** |
| embedding = LM head (공유) | 151,936 × 2048 | 0.31 B |
| **합계** | | **≈ 1.72 B** |

BF16(2 bytes)이면 weight는 **약 3.44 GB**. 이 숫자가 이 스터디 내내 나옵니다.

관찰: 파라미터의 **75%가 MLP**에 있습니다. "attention이 Transformer의 핵심"이지만, **짧은 context에서 연산과 메모리 읽기의 대부분은 MLP**입니다. 긴 context에서는 attention이 KV를 읽는 비용이 커집니다 (§5.5).

---

## 2. 📖 Tokenizer — 텍스트가 숫자가 되는 곳

- Qwen3는 **byte-level BPE** tokenizer를 씁니다. 자주 나오는 글자 조합을 하나의 token으로 합쳐 둔 사전(vocab 15만 개)을 이용해 텍스트를 쪼갭니다.
- 영어 텍스트는 대략 단어 하나에 1–1.5 token. **한국어는 같은 내용이라도 token 수가 더 많은 경우가 흔합니다** (사전에 한국어 조합이 상대적으로 적기 때문). token 수는 곧 비용이므로, 한국어 서비스라면 꼭 재 봐야 합니다.
  ```python
  from transformers import AutoTokenizer
  tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
  print(len(tok.encode("The capital of France is Paris.")))
  print(len(tok.encode("프랑스의 수도는 파리입니다.")))
  ```
- **토크나이저는 CPU에서 돈다.** 서빙 엔진은 이 일을 GPU를 돌리는 프로세스와 분리합니다 (Session 2).
- **detokenize도 간단하지 않다.** byte-level BPE에서는 한글 한 글자(UTF-8 3 bytes)가 token 두 개에 걸쳐 쪼개질 수 있습니다. token 하나만 보고 텍스트로 바꾸면 깨진 글자(`�`)가 나옵니다. 그래서 스트리밍 서버는 "글자가 완성될 때까지 기다렸다가" 내보냅니다 (Session 2 §2.6).
- 같은 텍스트 → 같은 token 시퀀스지만, **token 시퀀스 → 텍스트 → 다시 token**은 원래와 다를 수 있습니다. 이게 Session 3 멀티턴 캐시 문제의 원인이 됩니다.

---

## 3. ★ Prefill vs Decode

### 3.1 두 단계

```text
┌──────────── PREFILL ────────────┐┌───────────── DECODE ─────────────┐
 프롬프트 N개 token을 한 번에 forward   token 1개 → forward → token 1개
 → 모든 위치의 K, V를 cache에 저장      → 새 K, V 1개분만 cache에 추가
 → 마지막 위치 logits로 첫 token       … 출력 길이만큼 반복
└─────────────────────────────────┘└──────────────────────────────────┘
          ⇒ TTFT를 결정                       ⇒ ITL / TPOT를 결정
```

| | Prefill | Decode |
|---|---|---|
| 한 forward에서 처리하는 token | N (수백~수만) | 요청당 1 |
| forward 횟수 | 1 | 출력 길이만큼 |
| 행렬 연산 모양 | [N × 2048] · [2048 × 6144] — **GEMM** | [1 × 2048] · [2048 × 6144] — 사실상 **GEMV** |
| 병목 | **연산량** (compute-bound) | **메모리 읽기** (memory-bound) |
| 대응 지표 | TTFT | ITL, TPOT |
| GPU 활용률 | 높음 | 낮음 (batch가 작을 때) |

### 3.2 왜 행렬 모양이 중요한가

같은 weight 행렬 W [2048 × 6144]를 쓰더라도:

- prefill: 입력 N=512개 행을 **한 번에** W와 곱한다 → W를 한 번 읽고 512번 재사용
- decode: 입력 1개 행만 W와 곱한다 → W를 한 번 읽고 1번 사용

GPU는 "읽어 온 데이터로 계산을 얼마나 많이 하느냐"에 따라 성능이 완전히 달라집니다. 이게 §5의 주제입니다.

### 3.3 "왜 첫 token은 늦고 이후는 빠른가?"

첫 token은 prompt 전체를 처리하는 prefill을 기다려야 합니다. 이후 token은 token 하나씩만 forward하면 되고, 앞선 계산은 KV cache에 있습니다.
**단, 혼자 쓸 때만.** 다른 요청이 섞이면 대기열 시간(Session 2)과 끼어드는 prefill(Session 4)이 더해집니다.

---

## 4. ★ KV cache — 메모리를 써서 연산을 산다

### 4.1 왜 필요한가

attention에서 위치 t의 Query는 **위치 1…t의 Key/Value 전부**를 봐야 합니다.

```text
cache 없음:  step t마다 forward(t₁ … tₙ₊ₜ)  → 매번 전체를 다시 계산
cache 있음:  step t마다 forward(tₙ₊ₜ) + 저장해 둔 K, V 읽기
```

프롬프트 N, 출력 M일 때 linear layer가 처리하는 token 수:

```text
cache 없음: Σ_{t=0}^{M-1} (N + t) = M·N + M(M−1)/2      ← 출력 길이의 제곱으로 증가
cache 있음: N + (M − 1)                                   ← 선형
```

N=512, M=128이면 cache 없음 ≈ 73,600 token 분량, cache 있음 ≈ 640 → **약 115배**의 linear 연산 차이. (실습 `manual_decode.py --experiment nocache`로 확인)

K, V만 저장하는 이유: 이전 위치의 **Q는 다시 쓰이지 않습니다.** 새 token의 Q가 이전 K, V를 볼 뿐입니다. 그리고 causal mask 때문에 이전 위치의 K, V는 뒤에 오는 token에 의해 바뀌지 않습니다 → 한 번 계산하면 영원히 유효.

### 4.2 크기 공식

```text
KV bytes / token = 2 (K와 V) × num_layers × num_kv_heads × head_dim × dtype_bytes
```

**Qwen3-1.7B (BF16):**

```text
2 × 28 × 8 × 128 × 2 B = 114,688 B = 112 KiB / token
```

| 상황 | KV cache |
|---|---:|
| 1 요청 × 1K token | 112 MiB |
| 1 요청 × 32K token | 3.5 GiB |
| 16 요청 × 2K token | 3.5 GiB |
| 64 요청 × 2K token | **14 GiB** ← weight(3.2 GiB)의 4배 |

`python common/kvcalc.py Qwen/Qwen3-1.7B --params-b 1.72`로 확인할 수 있습니다.

### 4.3 MHA → GQA → MQA → MLA: KV를 줄이는 구조들 (📖)

KV 크기는 `num_kv_heads × head_dim`에 비례합니다. 모델 설계자들이 이걸 줄여 온 역사:

| 방식 | KV head 수 | 예 | KV 크기 (MHA 대비) |
|---|---|---|---|
| MHA (Multi-Head) | = query head 수 | GPT-2, 초기 Llama | 1 |
| **GQA** (Grouped-Query) | query head 몇 개가 KV head 하나를 공유 | Llama-3 (8:1), **Qwen3-1.7B (2:1)** | 1/그룹 크기 |
| MQA (Multi-Query) | 1 | PaLM | 1/head 수 |
| MLA (Multi-head Latent) | K, V를 작은 latent 벡터로 압축해 저장 | DeepSeek-V2/V3 | 훨씬 작음 |

Qwen3-1.7B가 MHA였다면 KV는 224 KiB/token — 같은 GPU에 담을 수 있는 동시 사용자가 절반이 됩니다.
**모델 구조가 서빙 비용을 결정한다**는 것을 기억해 두세요.

### 4.4 GPU에 몇 token이 들어가나

RTX 4090 24 GB에서 SGLang은 activation·CUDA graph 버퍼 몫을 남기고 `mem_fraction_static`을 약 0.7 근처로 자동 설정합니다
(v0.5.23 `arg_groups/memory_hook.py`: 예비 메모리 ≈ 512 MB + chunked_prefill_size × 1.5 + CUDA graph 몫).

```text
24 GiB × 0.7 ≈ 16.8 GiB  (weight + KV pool)
− weight 3.2 GiB
= KV pool ≈ 13.6 GiB ≈ 127,000 token
```

이 숫자가 서버 시작 로그의 `max_total_num_tokens=`와 맞는지 실습에서 확인합니다.
이것이 **동시에 처리할 수 있는 요청의 상한**을 결정합니다: 요청 하나가 평균 2K token이면 ≈ 60명.

---

## 5. ★ Compute-bound vs Memory-bound — 이 스터디의 가장 중요한 그림

### 5.1 두 가지 한계

GPU에서 어떤 연산의 시간은 **둘 중 느린 쪽**이 결정합니다.

```text
time ≈ max( FLOPs / peak_FLOPS ,  bytes_moved / mem_bandwidth )
```

**Arithmetic intensity (AI)** = FLOPs / bytes. GPU마다 "경계선(ridge point)"이 있습니다.

| GPU | BF16 tensor (dense) | 메모리 대역폭 | ridge ≈ FLOPS / BW |
|---|---:|---:|---:|
| RTX 4090 | ≈ 165 TFLOPS | ≈ 1.0 TB/s | **≈ 165 FLOP/byte** |
| L4 | ≈ 121 TFLOPS | ≈ 0.3 TB/s | ≈ 400 FLOP/byte |
| H100 SXM | ≈ 990 TFLOPS | ≈ 3.35 TB/s | ≈ 295 FLOP/byte |

```text
achievable
 FLOPS  │                   ____________________  ← peak compute (165 TFLOPS)
        │                 ╱
        │               ╱   ← 기울기 = 메모리 대역폭
        │             ╱
        │           ╱
        │         ╱
        │       ╱
        └─────┴──────────────────────────────── arithmetic intensity (FLOP/byte)
              ↑ ridge ≈ 165
   decode(B=1)는 여기 (AI≈1)         prefill(N=512)은 여기 (AI≈512)
```

AI가 ridge보다 작으면 → GPU 연산기는 놀고 **메모리를 기다린다** (memory-bound).

### 5.2 Decode의 AI ≈ batch size

decode step 하나에서 linear layer들은

- 읽는 bytes ≈ weight 전체 = 2 B × P
- 하는 연산 ≈ 2 × P × B FLOPs (곱셈 1 + 덧셈 1, B = 이번 step의 요청 수)

```text
AI ≈ (2·P·B) / (2·P) = B
```

**B = 1이면 AI = 1.** ridge 165에 한참 못 미칩니다. batch=1 decode는 "weight를 한 번 훑어 읽는 시간"이 전부입니다.

### 5.3 손으로 계산하는 decode 하한 (화이트보드에서 같이)

```text
Qwen3-1.7B weight ≈ 3.44 GB  (28 layers 2.82 GB + LM head 0.62 GB)
RTX 4090 대역폭   ≈ 1.0 TB/s

decode step 최소 시간 ≈ 3.44 GB / 1.0 TB/s ≈ 3.4 ms
→ batch=1의 이론 최대 속도 ≈ 290 token/s
```

(embedding은 행렬곱이 아니라 "B개 행 꺼내기"라서 decode에서 거의 안 읽습니다. LM head는 weight를 공유하지만 matmul이므로 전부 읽습니다.)

실제로는 커널 launch overhead, attention, sampling 등으로 이보다 느립니다.
**HF eager 구현과 SGLang이 이 하한에 각각 얼마나 가까운지**가 오늘 실습의 질문 중 하나입니다.

### 5.4 그래서 batching이 강력하다

weight는 어차피 한 번 읽습니다. 그 한 번에 요청 B개를 같이 처리하면:

```text
step 시간 ≈ max( 3.4 ms ,  2·P·B / 165 TFLOPS )      (KV 읽기 무시)

B = 1   → 3.4 ms,  throughput ≈ 290 tok/s
B = 32  → ≈ 3.4 ms(+α), throughput ≈ 9,300 tok/s   ← "공짜 구간"
B ≈ 165 → 연산 시간(2·1.72e9·165/165e12 ≈ 3.4 ms)이 메모리 시간을 따라잡음
B > 165 → step 시간이 B에 비례해 늘어남 (compute-bound)
```

이것이 **"batch를 키우면 throughput은 거의 공짜로 늘고 latency는 거의 그대로"인 구간**이 존재하는 이유이며, 서빙 엔진이 batch를 크게 만들려고 애쓰는 이유입니다.

### 5.5 함정: KV 읽기는 batch로 상쇄되지 않는다

weight는 요청끼리 공유하지만 **KV는 요청마다 따로** 읽어야 합니다.

```text
decode step bytes ≈ weight bytes  +  Σ_i (요청 i의 context 길이) × 112 KiB

B = 32, context 2K → KV 읽기 ≈ 32 × 2048 × 112 KiB ≈ 7.5 GB   (weight 3.44 GB의 2배!)
→ step 시간 ≈ (3.44 + 7.5) GB / 1 TB/s ≈ 11 ms   (§5.4의 3.4 ms가 아니라)
```

attention 부분의 AI는 batch와 상관없이 낮습니다 (각 요청의 KV는 그 요청의 Q 하나 — GQA면 몇 개 — 하고만 곱해지므로).
→ 긴 context × 큰 batch에서는 **attention의 KV 읽기가 step 시간을 지배**합니다. 그래서 KV를 줄이는 기술(GQA, KV FP8 양자화, MLA)은 메모리 용량 기술이면서 동시에 **decode 속도 기술**입니다 (Session 5).

### 5.6 Prefill은 반대 — 그리고 attention의 제곱 항

prefill은 N token을 한 번에 처리 → linear의 AI ≈ N. 프롬프트 512 token이면 이미 ridge를 넘습니다 → **compute-bound**.

prefill FLOPs를 세 보면 (📖):

```text
linear      : 2 × 1.41 B × N                         (layer 부분, LM head는 마지막 위치만)
attention   : 28 layers × 2 × (QKᵀ, AV) × N² × (16 heads × 128) × ½ (causal)
            ≈ 28 × 2 × N² × 2048 ≈ 1.15 × 10⁵ × N²

N = 2K  : linear 5.8 TFLOP,  attention 0.48 TFLOP   (attention ≈ 8%)
N = 16K : linear 46 TFLOP,   attention 31 TFLOP     (attention ≈ 40%)
N ≈ 24K 근처에서 두 항이 같아짐
```

→ 짧은 프롬프트의 TTFT는 입력 길이에 **선형**, 아주 긴 프롬프트는 **제곱 항이 보이기 시작**합니다. 4090(실효 100 TFLOPS 정도 가정)에서 16K prefill은 대략 0.8초 — Session 4에서 "긴 프롬프트가 다른 사람을 멈추게 하는" 실험의 근거입니다.

| | 지배 요인 | batch를 키우면 |
|---|---|---|
| Prefill | 연산량 | 이득 작음 (이미 GPU 포화), 다른 요청을 막음 |
| Decode (linear) | weight 읽기 | **이득 큼** (읽기 비용 분산) |
| Decode (attention) | KV 읽기 | 이득 없음 (요청마다 따로 읽음) |

### 5.7 세 번째 병목: overhead (📖)

roofline은 "GPU가 쉬지 않고 일할 때"의 그림입니다. 현실에는 세 번째 시간이 있습니다:

```text
실제 step 시간 = GPU 계산/메모리 시간  +  GPU가 아무것도 안 하고 기다리는 시간
```

기다리는 시간의 주범은 **CPU**입니다. Python이 layer마다 커널 수십 개를 하나씩 GPU에 보내는데(launch), 커널 하나가 수 μs밖에 안 걸리는 작은 batch decode에서는 **launch 속도가 GPU 속도를 못 따라갑니다.**
HF eager 구현이 이론 하한보다 몇 배 느린 주된 이유가 이것이고, Session 5의 CUDA graph가 이걸 없앱니다.
(Horace He의 글 *Making Deep Learning Go Brrrr*의 compute / memory / overhead 세 분류)

---

## 6. 📖 Sampling — logits에서 token으로

### 6.1 방법들

| 방법 | 하는 일 | 성질 |
|---|---|---|
| greedy | `argmax(logits)` | 결정적. **이 스터디의 모든 측정은 temperature=0 (greedy)** |
| temperature T | `softmax(logits / T)`에서 뽑기 | T↓ 날카로움, T↑ 다양성 |
| top-k | 상위 k개만 남기고 뽑기 | |
| top-p (nucleus) | 누적 확률 p까지의 최소 집합에서 뽑기 | 정렬이 필요해 보이지만… |
| min-p | 최댓값의 p배 이상인 것만 | |
| repetition / frequency penalty | 이미 나온 token의 점수를 깎음 | 요청마다 "나온 token 목록" 필요 |

### 6.2 성능 관점

- vocab 15만 개를 매 step 정렬하면 비쌉니다. 서빙 엔진은 정렬 없이 top-k/top-p를 처리하는 GPU 커널(FlashInfer의 sampling 커널 등)을 씁니다.
- batch 안의 요청마다 sampling 설정이 다를 수 있습니다 (A는 greedy, B는 T=0.8). 엔진은 이를 텐서로 묶어 한 번에 처리합니다 (SGLang: `sampling/sampling_batch_info.py`, `layers/sampler.py`).
- **측정 규칙:** sampling은 출력 길이를 바꿉니다. 그래서 측정 때는 `temperature=0` + `ignore_eos=True`(정해진 길이까지 무조건 생성)로 "같은 양의 일"을 시킵니다.
  단, `ignore_eos`는 스케줄러 동작까지 바꾼다는 사실을 Session 4에서 배웁니다.

### 6.3 EOS와 stop

모델이 EOS token(Qwen3: `<|im_end|>` 등)을 내거나, 사용자가 지정한 stop 문자열이 나오거나, `max_new_tokens`에 도달하면 끝납니다.
서빙 엔진은 매 step **요청마다** 이 조건을 검사합니다 (SGLang: `Req.check_finished` 계열) — 그래서 batch 안의 요청들이 제각각 끝나고, 이게 continuous batching이 필요한 이유입니다.

---

## 7. ★ 지표 — 무엇을, 어디서 재는가

### 7.1 정의

```text
t0 요청 전송      t1 첫 token 도착   t2 둘째 token      ...     tₙ 마지막 token
│─────────────────│──────────────────│────── … ───────────────│
   TTFT = t1 − t0    ITL₁ = t2 − t1    ITL₂ …
                                       TPOT = (tₙ − t1) / (n − 1)
                     E2E = tₙ − t0
```

| 지표 | 사용자 체감 | 주로 영향 주는 것 |
|---|---|---|
| TTFT | "답이 시작되기까지 답답함" | 프롬프트 길이, prefill, **대기열 시간** |
| ITL / TPOT | "글자가 흘러나오는 속도" | decode step 시간, batch 크기, 끼어드는 prefill |
| E2E | 전체 완료 시간 | TTFT + (출력 길이 − 1) × TPOT |
| Throughput | 운영 비용 ($/token) | batching, 스케줄링, GPU 활용률 |

참고: 사람이 읽는 속도는 대략 초당 5–10 단어. ITL 50 ms(초당 20 token)면 대부분의 사용자에게 "끊김 없음"입니다. 반면 코드 생성·에이전트처럼 **사람이 아니라 프로그램이 읽는** 경우에는 E2E가 더 중요합니다.

### 7.2 TPOT와 ITL의 차이

- **TPOT = 평균**, **ITL = 분포**입니다.
- 출력 100 token 중 99개는 10 ms 간격이고 1개가 1,000 ms라면: TPOT ≈ 20 ms(괜찮아 보임), ITL p99 = 1,000 ms(사용자는 1초 멈춤을 봄).
- 이런 멈춤은 다른 사용자의 긴 prefill이 끼어들 때 생깁니다. Session 4에서 직접 만듭니다.

### 7.3 평균만 보지 않는다

100개 중 95개가 100 ms, 5개가 5 s이면 평균은 345 ms — 괜찮아 보이지만 20명 중 1명은 5초를 기다립니다.
항상 **p50 / p95 / p99**를 함께 적습니다. (p99 = 100번 중 가장 나쁜 1번 근처)

### 7.4 Throughput의 여러 얼굴

| 이름 | 계산 | 언제 쓰나 |
|---|---|---|
| request throughput | 완료 요청 수 / 시간 | 용량 계획 ("초당 몇 명") |
| output token throughput | 생성 token 수 / 시간 | 비용 계산의 기본 |
| total token throughput | (입력 + 출력) token / 시간 | prefill 비중이 큰 workload 비교 |

입력이 긴 workload에서 total token throughput만 보고 "빠르다"고 하면 오해를 부릅니다. 무엇을 셌는지 항상 적으세요.

### 7.5 어디서 쟀는가

- 우리 `common/client.py`는 **클라이언트에서** 스트림 청크 도착 시각으로 잽니다 → 네트워크·HTTP·detokenize 포함.
- 서버 로그와 `/metrics`(Session 6)는 **서버 내부 시각**입니다.
- 둘이 다르면 그 차이가 서버 밖의 비용입니다. 리포트에는 "어디서 잰 값인지"를 반드시 적습니다.

---

## 8. ★ 왜 `model.generate()`로는 서빙이 안 되는가

`generate()`는 **"이미 모인 batch 하나"를 끝까지 돌리는 함수**입니다. 서비스에는 다음이 필요합니다.

| `generate()`의 한계 | 서빙 엔진의 답 | 회차 |
|---|---|---|
| 요청이 제각각 도착한다 (batch를 미리 모을 수 없다) | 요청 큐 + 매 step마다 batch 재구성 (**continuous batching**) | 2 |
| batch 안에서 짧은 요청이 긴 요청을 기다린다 (padding, 빈 슬롯) | 끝난 요청은 즉시 빼고 새 요청을 넣는다 | 2 |
| KV를 최대 길이만큼 미리 잡는다 → 메모리 낭비 | token 단위 KV pool + 동적 할당 | 3 |
| 같은 system prompt를 요청마다 다시 계산 | prefix cache (**RadixAttention**) | 3 |
| 메모리가 모자라면 그냥 OOM | admission control, retraction | 4 |
| 긴 프롬프트 하나가 모두를 멈춘다 | chunked prefill | 4 |
| Python이 커널을 하나씩 launch (CPU overhead) | CUDA graph, 최적화 커널, overlap scheduling | 4, 5 |
| 메모리 대역폭이 한계 | 양자화, speculative decoding | 5 |
| 측정·관측 수단이 없다 | 로그, metrics, profiler, 벤치마크 도구 | 6 |

이 표가 **스터디 전체 지도**입니다. 매 회차 시작할 때 이 표의 어느 줄을 공부하는지 짚고 갑니다.

### 8.1 "batch를 직접 모으면 되지 않나?" (📖)

"요청을 100 ms 기다렸다가 모아서 `generate()`" 하는 방식(static/dynamic batching)도 있습니다. 문제는:

1. 기다리는 시간이 곧 TTFT 증가
2. 모인 요청들의 출력 길이가 다르면, 가장 긴 요청이 끝날 때까지 batch가 묶여 있음 → 일찍 끝난 자리는 낭비, 새 요청은 다음 batch까지 대기
3. 길이가 다른 프롬프트는 padding해야 함 → 계산 낭비
4. KV를 batch의 최대 길이로 미리 잡음

Session 2에서 이 문제를 iteration 단위 스케줄링으로 어떻게 푸는지 봅니다.

---

## 9. 📖 SGLang 소개와 이 스터디의 버전 원칙

- LMSYS(UC Berkeley Sky Lab 등)에서 시작한 오픈소스 LLM 서빙 엔진. 논문: *SGLang: Efficient Execution of Structured Language Model Programs* (NeurIPS 2024). 핵심 기여가 **RadixAttention**(prefix 캐시 자동 재사용)과 **구조화된 생성(constrained decoding)** 최적화입니다.
- 두 부분: **Frontend language**(여러 번의 생성 호출을 프로그램으로 쓰는 DSL, `sglang.lang`)와 **Runtime(SRT)**. 이 스터디는 **SRT(서버)만** 다룹니다.
- 비슷한 엔진: vLLM, TensorRT-LLM, LMDeploy, llama.cpp. 해결하려는 문제(§8 표)는 모두 같고, **선택(trade-off)이 다릅니다.** SGLang을 끝까지 파 보면 다른 엔진도 같은 질문으로 읽을 수 있습니다.
- **버전 고정:** SGLang은 몇 주마다 릴리스가 나오고 플래그 이름과 파일 구조가 바뀝니다. 이 스터디는 **v0.5.23**(2026-10-28)에 고정합니다. 블로그나 예전 자료의 명령이 안 맞으면 버전부터 의심하세요.
  - 예: `python -m sglang.bench_serving` → v0.5.23에서는 `python -m sglang.benchmark.serving` (구 경로는 deprecated 경고와 함께 동작)
  - 예: `--disable-cuda-graph` → `--cuda-graph-backend-decode disabled` 등으로 세분화
- 처음 실행하는 명령은 두 가지 중 아무거나:
  ```bash
  python -m sglang.launch_server --model-path Qwen/Qwen3-1.7B --port 30000   # 이 스터디의 스크립트가 쓰는 형태
  sglang serve Qwen/Qwen3-1.7B --port 30000                                  # 공식 문서의 새 CLI
  ```

---


## ★ 실습 전 예측 질문

실습 직전에 노트에 **숫자로** 적으세요. 실습 후 실제 값과 비교하는 것이 이 회차에서 가장 많이 배우는 순간입니다.

1. Qwen3-1.7B, 입력 512 / 출력 128, batch 1. HF로 직접 짠 decode loop의 step 시간은 몇 ms일까? 이론 하한 3.4 ms의 몇 배일까?
2. 같은 조건에서 batch를 1 → 8 → 32로 올리면 **step 시간**은 몇 배? **throughput**은 몇 배?
3. KV cache를 끄면(`use_cache=False`) decode step 시간은 몇 배가 될까? 출력이 길수록 더 나빠질까?
4. 같은 요청을 SGLang 서버로 보내면 batch=1 ITL은 몇 ms일까? HF와 차이가 난다면, 같은 GPU·같은 weight인데 **왜** 다를까?

<details>
<summary>강사용: 답의 방향 (숫자는 리허설 결과로 채우기)</summary>

1. HF eager는 Python이 layer마다 수십 개 커널을 launch하므로 하한보다 수 배 느린 경우가 많다 (§5.7 overhead → Session 5 CUDA graph).
2. step 시간은 1.x배 수준, throughput은 batch에 가깝게 증가 (memory-bound 공짜 구간). HF는 overhead가 커서 이 효과가 더 크게 보일 수 있다.
3. cache 없이 t번째 step은 (N+t) token 전체를 다시 계산 → step 시간이 prefill 수준. 출력이 길수록 비율이 나빠진다 (§4.1).
4. SGLang은 CUDA graph + 최적화 커널로 하한에 훨씬 가깝다. "같은 모델인데 엔진이 다르면 속도가 다르다"는 것 자체가 이 스터디의 출발점.
</details>

---

## 10. 요약 · 자기 점검 · 더 읽을거리

### 요약

1. LLM은 token 하나씩 예측한다. 출력 길이 ∝ forward 횟수.
2. prefill(프롬프트 전체, compute-bound, TTFT)과 decode(token 하나씩, memory-bound, ITL)는 다른 종류의 일이다.
3. KV cache는 O(S²) 재계산을 O(S) 저장으로 바꾼다. Qwen3-1.7B는 112 KiB/token.
4. batch=1 decode의 하한 ≈ weight bytes / 대역폭 ≈ 3.4 ms (4090).
5. decode의 AI ≈ batch. batch를 키우면 ridge(≈165)까지는 throughput이 거의 공짜로 는다.
6. 하지만 KV 읽기는 요청마다 따로라 batch로 상쇄되지 않는다. 긴 context에서는 attention이 지배.
7. 실제 시간에는 GPU가 CPU를 기다리는 overhead가 끼어 있다.
8. TTFT / ITL(분포) / TPOT(평균) / E2E / throughput, 그리고 p50/p95/p99.
9. `generate()`에 없는 것 = 이 스터디의 목차.

### 자기 점검 문제

1. Qwen3-8B (36 layers, KV heads 8, head_dim 128)의 BF16 KV bytes/token은?
2. 같은 GPU에서 batch=1 decode를 2배 빠르게 하려면 무엇을 바꿔야 할까? 두 가지 이상.
3. "batch를 키우면 decode가 결국 compute-bound가 된다"는 말이 긴 context에서는 왜 잘 안 맞는가?
4. 프롬프트 100 token, 출력 1,000 token인 요청과 프롬프트 1,000 token, 출력 100 token인 요청 중 E2E가 더 긴 것은? 왜?
5. TPOT가 15 ms인데 사용자가 "가끔 멈춘다"고 불평한다. 어떤 지표를 보면 확인할 수 있나?
6. KV cache에 Q는 왜 저장하지 않는가?
7. RTX 4090 대신 L4를 쓰면 batch=1 decode 하한은 몇 ms가 될까?

<details>
<summary>답</summary>

1. 2 × 36 × 8 × 128 × 2 = 147,456 B = 144 KiB/token
2. (a) 읽는 bytes를 줄인다: weight 양자화(FP8/INT4) (b) 대역폭이 큰 GPU (c) overhead를 줄인다: CUDA graph (d) 한 번 읽을 때 여러 token: speculative decoding
3. KV 읽기는 요청마다 따로라 batch를 키우면 KV bytes도 같이 커진다. attention 부분의 AI는 batch와 무관하게 낮게 머문다 → 연산이 아니라 KV 메모리 대역폭에 막힌다.
4. 출력 1,000 token 쪽. decode step(≈ 수 ms) × 1,000이 prefill 1,000 token(수십 ms)보다 훨씬 크다.
5. ITL의 p99 / max. TPOT는 평균이라 드문 긴 멈춤을 숨긴다.
6. 미래의 token은 과거 위치의 K, V만 참조하고 과거의 Q는 다시 쓰지 않는다.
7. 3.44 GB / 0.3 TB/s ≈ 11.5 ms (약 3.4배 느림)
</details>

### 더 읽을거리

- Kipply, *Transformer Inference Arithmetic* — KV 크기·roofline 계산을 가장 친절하게 설명
- Horace He, *Making Deep Learning Go Brrrr From First Principles* — compute / memory / overhead
- Williams et al., *Roofline: An Insightful Visual Performance Model* (2009) — roofline 원전
- Ainslie et al., *GQA* (2023), DeepSeek-V2 논문의 MLA 절
- SGLang 논문 (arXiv 2312.07104), 문서 <https://docs.sglang.io>

### 다음 회차 예고

오늘은 요청 **하나**를 봤습니다. 다음 회차는 요청이 **여러 개, 제각각** 도착할 때 서버 안에서 무슨 일이 일어나는지 — 프로세스 구조, 스케줄러 루프, continuous batching — 를 봅니다.
