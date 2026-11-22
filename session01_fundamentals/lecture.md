# Session 1 이론 — LLM 추론의 기초

> 50분. §4(roofline)에 가장 많은 시간을 쓰세요. 이후 5회차 전부가 이 그림 위에 서 있습니다.

| 절 | 내용 | 시간 |
|---|---|---|
| §1 | token 하나가 만들어지는 과정 | 6분 |
| §2 | Prefill vs Decode | 6분 |
| §3 | KV cache와 그 크기 | 8분 |
| §4 | Compute-bound vs Memory-bound (roofline) | 14분 |
| §5 | 지표: TTFT / ITL / TPOT / E2E / Throughput | 6분 |
| §6 | 왜 `generate()`로는 서빙이 안 되는가 → 6회차 지도 | 6분 |
| §7 | SGLang 소개 + 예측 질문 | 4분 |

---

## §1. token 하나가 만들어지는 과정

```text
"The capital of France is"
        │ tokenizer
        ▼
[785, 6722, 315, 9625, 374]                 ← token id 5개 (예시 — 실습에서 직접 확인)
        │ embedding  (vocab × hidden)
        ▼
x: [5, 2048]                                 ← Qwen3-1.7B hidden = 2048
        │ × 28 decoder layers
        │    ├─ Attention  (Q, K, V 투영 → softmax(QKᵀ/√d)·V → O 투영)
        │    └─ MLP        (gate/up/down, 2048 → 6144 → 2048)
        ▼
h: [5, 2048]
        │ LM head (hidden × vocab)   ← 마지막 위치만 필요!
        ▼
logits: [151936]                             ← vocab 전체에 대한 점수
        │ sampling (greedy / temperature / top-p)
        ▼
next token = " Paris"   → 시퀀스에 붙이고 처음으로 돌아간다
```

기억할 것 두 가지:

1. **모델은 "다음 token 하나"만 예측한다.** 출력 512 token = forward 512번(원칙적으로). 출력 길이는 latency에 거의 **선형**으로 들어간다.
2. **매 step마다 vocab 15만 개의 점수를 만들고 하나만 쓴다.** 그래서 LogitsProcessor는 마지막 위치만 계산한다 (Session 5에서 코드로 확인).

> 이 스터디의 sampling 규칙: 모든 측정은 `temperature=0`, `ignore_eos=True`.
> 출력 길이가 요청마다 달라지면 "시스템이 빨라서"인지 "덜 생성해서"인지 구분할 수 없기 때문입니다.

---

## §2. Prefill vs Decode

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
| 행렬 연산 모양 | [N × d] · [d × d] — **GEMM** | [1 × d] · [d × d] — 사실상 **GEMV** |
| 병목 | **연산량** (compute-bound) | **메모리 읽기** (memory-bound) |
| 대응 지표 | TTFT | ITL, TPOT |

"왜 첫 token은 늦고 이후는 빠른가?"의 답이 여기 있습니다 — 단, **혼자 쓸 때만**.
다른 요청이 섞이면 이야기가 달라집니다 (Session 2, 4).

---

## §3. KV cache — 메모리를 써서 연산을 산다

attention에서 새 token의 Query는 **이전 모든 token의 Key/Value**를 봐야 합니다.

```text
cache 없음:  step t마다 forward(t₁ … tₙ₊ₜ)  → 매번 전체를 다시 계산 → 총 O(S²)
cache 있음:  step t마다 forward(tₙ₊ₜ) + 저장된 K,V 읽기     → 총 O(S)
```

### 크기 공식

```text
KV bytes / token = 2 (K,V) × num_layers × num_kv_heads × head_dim × dtype_bytes
```

**Qwen3-1.7B (BF16):** 28 layers, KV heads 8, head_dim 128

```text
2 × 28 × 8 × 128 × 2 B = 114,688 B = 112 KiB / token
```

| 상황 | KV cache |
|---|---:|
| 1 요청 × 1K token | 112 MiB |
| 1 요청 × 32K token | 3.5 GiB |
| 16 요청 × 2K token | 3.5 GiB |
| 64 요청 × 2K token | **14 GiB** ← weight(3.2 GiB)의 4배 |

`python common/kvcalc.py Qwen/Qwen3-1.7B --params-b 1.72` 로 직접 확인할 수 있습니다.

**관찰 포인트**

- **GQA**: query head 16개가 KV head 8개를 공유 → MHA 대비 KV가 절반. (Llama-3 계열은 4:1, 8:1까지 줄입니다.)
- 동시 요청이 많아지면 **메모리의 주인공은 weight가 아니라 KV**가 됩니다. 이 스터디의 Session 3, 4는 전부 "이 KV 메모리를 누가, 언제, 얼마나 쓰게 할 것인가"입니다.
- RTX 4090 24 GB에서 SGLang은 activation·CUDA graph 몫을 남기고 `mem_fraction_static`을 약 0.7 근처로 자동 설정합니다 →
  weight 3.2 GiB를 빼면 KV pool은 대략 **13–14 GiB ≈ 12만 token** 입니다. 실습에서 서버 로그의 `max_total_num_tokens=`와 비교해 봅니다.

---

## §4. Compute-bound vs Memory-bound — 이 스터디의 가장 중요한 그림

GPU에서 어떤 연산의 시간은 **둘 중 느린 쪽**이 결정합니다.

```text
time ≈ max( FLOPs / peak_FLOPS ,  bytes_moved / mem_bandwidth )
```

**Arithmetic intensity (AI)** = FLOPs / bytes. GPU마다 "경계선(ridge point)"이 있습니다.

| GPU | BF16 tensor (dense, FP32 acc.) | 메모리 대역폭 | ridge ≈ FLOPS / BW |
|---|---:|---:|---:|
| RTX 4090 | ≈ 165 TFLOPS | ≈ 1.0 TB/s | **≈ 165 FLOP/byte** |
| L4 | ≈ 121 TFLOPS | ≈ 0.3 TB/s | ≈ 400 FLOP/byte |
| H100 SXM | ≈ 990 TFLOPS | ≈ 3.35 TB/s | ≈ 295 FLOP/byte |

AI가 ridge보다 작으면 → GPU 연산기는 놀고 **메모리를 기다린다** (memory-bound).

### Decode의 AI ≈ batch size

decode step 하나에서 linear layer들은

- 읽는 bytes ≈ weight 전체 = 2 B × P (P = 파라미터 수)
- 하는 연산 ≈ 2 × P × B FLOPs (B = 이번 step의 요청 수)

```text
AI ≈ (2·P·B) / (2·P) = B
```

**B = 1이면 AI = 1.** ridge 165에 한참 못 미칩니다. batch=1 decode는 "weight를 한 번 훑어 읽는 시간"이 전부입니다.

### 손으로 계산하는 decode 하한 (화이트보드에서 같이)

```text
Qwen3-1.7B weight ≈ 1.72 B params × 2 B = 3.44 GB
RTX 4090 대역폭   ≈ 1.0 TB/s

decode step 최소 시간 ≈ 3.44 GB / 1.0 TB/s ≈ 3.4 ms
→ batch=1의 이론 최대 속도 ≈ 290 token/s
```

실제로는 커널 launch overhead, attention, sampling 등으로 이보다 느립니다.
**HF eager 구현과 SGLang이 이 하한에 각각 얼마나 가까운지**가 오늘 실습의 질문 중 하나입니다.

### 그래서 batching이 강력하다

weight는 어차피 한 번 읽습니다. 그 한 번에 요청 B개를 같이 처리하면:

```text
step 시간 ≈ max( 3.4 ms ,  2·P·B / 165 TFLOPS )      (KV 읽기 무시)
B = 1   → 3.4 ms,  throughput ≈ 290 tok/s
B = 32  → 여전히 ≈ 3.4 ms(+α), throughput ≈ 9,300 tok/s   ← "공짜 구간"
B ≈ 165 → 연산 시간이 메모리 시간을 따라잡음 → 이후로는 step 시간이 B에 비례
```

### 함정: KV 읽기는 batch로 상쇄되지 않는다

weight는 요청끼리 공유하지만 **KV는 요청마다 따로** 읽어야 합니다.

```text
decode step bytes ≈ weight bytes  +  Σ (요청 i의 context 길이) × 112 KiB

B = 32, context 2K → KV 읽기 ≈ 32 × 2048 × 112 KiB ≈ 7 GiB  (weight 3.2 GiB의 2배!)
```

→ 긴 context × 큰 batch에서는 **attention이 step 시간을 지배**합니다. 그래서 KV를 줄이는 기술(GQA, KV FP8 양자화, MLA)이 decode 성능 기술이기도 합니다 (Session 5).

### Prefill은 반대

prefill은 N token을 한 번에 처리 → AI ≈ N. 프롬프트 512 token이면 이미 ridge를 넘습니다 → **compute-bound**.
prefill을 batch로 묶어도 이득이 작고, 오히려 GPU를 오래 점유해 다른 요청의 decode를 막습니다 (→ Session 4 chunked prefill).

| | 지배 요인 | batch를 키우면 |
|---|---|---|
| Prefill | 연산량 | 이득 작음, 다른 요청을 막음 |
| Decode (linear) | weight 읽기 | **이득 큼** (읽기 비용 분산) |
| Decode (attention) | KV 읽기 | 이득 없음 (요청마다 따로 읽음) |

---

## §5. 지표 — 무엇을, 어디서 재는가

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
| E2E | 전체 완료 시간 | TTFT + 출력 길이 × TPOT |
| Throughput | 운영 비용 ($/token) | batching, 스케줄링, GPU 활용률 |

- **TPOT는 평균, ITL은 분포**입니다. 평균 TPOT가 좋아도 ITL p99에 300 ms짜리 멈춤이 숨어 있을 수 있습니다 (Session 4에서 직접 만듭니다).
- **평균만 보지 않는다.** 100개 중 95개가 100 ms, 5개가 5 s이면 평균은 345 ms — 괜찮아 보이지만 20명 중 1명은 5초를 기다립니다. 항상 p50 / p95 / p99를 함께 적습니다.
- **어디서 쟀는가**를 적는다. 우리 `common/client.py`는 **클라이언트에서** 스트림 청크 도착 시각으로 잽니다 (네트워크·detokenize 포함). 서버 로그/metrics는 서버 내부 시각입니다.

---

## §6. 왜 `model.generate()`로는 서빙이 안 되는가

`generate()`는 "이미 모인 batch 하나"를 끝까지 돌리는 함수입니다. 서비스에는 다음이 필요합니다.

| `generate()`의 한계 | 서빙 엔진의 답 | 회차 |
|---|---|---|
| 요청이 제각각 도착한다 (batch를 미리 모을 수 없다) | 요청 큐 + 매 step마다 batch 재구성 (**continuous batching**) | 2 |
| batch 안에서 짧은 요청이 긴 요청을 기다린다 (padding, 빈 슬롯) | 끝난 요청은 즉시 빼고 새 요청을 넣는다 | 2 |
| KV를 최대 길이만큼 미리 잡는다 → 메모리 낭비 | token 단위 KV pool + 동적 할당 | 3 |
| 같은 system prompt를 요청마다 다시 계산 | prefix cache (**RadixAttention**) | 3 |
| 메모리가 모자라면 그냥 OOM | admission control, retraction | 4 |
| 긴 프롬프트 하나가 모두를 멈춘다 | chunked prefill | 4 |
| Python이 커널을 하나씩 launch (CPU overhead) | CUDA graph, 최적화 커널, overlap scheduling | 4, 5 |
| 측정·관측 수단이 없다 | 로그, metrics, profiler, 벤치마크 도구 | 6 |

이 표가 **스터디 전체 지도**입니다. 매 회차 시작할 때 이 표의 어느 줄을 공부하는지 짚고 갑니다.

---

## §7. SGLang 소개

- LMSYS(UC Berkeley Sky Lab 등)에서 시작한 오픈소스 LLM 서빙 엔진. 논문: *SGLang: Efficient Execution of Structured Language Model Programs* (NeurIPS 2024) — 핵심 기여가 **RadixAttention**(prefix 캐시 자동 재사용)입니다.
- 두 부분: **Frontend language**(프롬프트 프로그래밍 DSL, `sglang.lang`)와 **Runtime(SRT)**. 이 스터디는 **SRT(서버)만** 다룹니다.
- 비슷한 엔진: vLLM, TensorRT-LLM, LMDeploy, llama.cpp. 해결하려는 문제(위 표)는 모두 같고, **선택(trade-off)이 다릅니다.** SGLang을 끝까지 파 보면 다른 엔진도 같은 질문으로 읽을 수 있습니다.
- 이 스터디는 **v0.5.23**에 고정합니다. SGLang은 매달 릴리스가 나오고 플래그 이름도 바뀌기 때문에, 버전을 고정하지 않으면 실습 명령이 깨집니다.

---

## 예측 질문 (실습 전에 노트에 숫자로 적기)

1. Qwen3-1.7B, 입력 512 / 출력 128, batch 1. HF `generate()`의 decode는 token당 몇 ms일까? 이론 하한 3.4 ms의 몇 배일까?
2. 같은 조건에서 batch를 1 → 8로 올리면 **step 시간**은 몇 배? **throughput**은 몇 배?
3. KV cache를 끄면(`use_cache=False`) 출력 128 token 생성 시간은 몇 배가 될까? 출력 길이가 길수록 더 나빠질까?
4. 같은 요청을 SGLang 서버로 보내면 batch=1 decode는 몇 ms일까? HF와 차이가 난다면, 같은 GPU·같은 weight인데 **왜** 다를까?

<details>
<summary>강사용: 답의 방향 (숫자는 리허설 결과로 채우기)</summary>

1. HF eager는 Python이 layer마다 수십 개 커널을 launch하므로 하한보다 수 배 느린 경우가 많다. 이 차이가 "CPU launch overhead"이고, Session 5의 CUDA graph가 줄이는 대상이다.
2. step 시간은 1.x배 수준, throughput은 batch에 가깝게 증가 (memory-bound 공짜 구간). 단, HF는 overhead가 커서 이 효과가 오히려 더 크게 보일 수 있다.
3. cache 없이 t번째 step은 (N+t) token 전체를 다시 계산 → 총 연산이 O(S²). 출력이 길수록 비율이 나빠진다.
4. SGLang은 CUDA graph + 최적화 attention 커널로 하한에 훨씬 가깝다. "같은 모델인데 엔진이 다르면 속도가 다르다"는 것 자체가 이 스터디의 출발점.
</details>

## 더 읽을거리

- Kipply, *Transformer Inference Arithmetic* — KV 크기·roofline 계산을 가장 친절하게 설명
- Horace He, *Making Deep Learning Go Brrrr From First Principles* — compute / memory / overhead 세 가지 병목
- SGLang 논문 (arXiv 2312.07104), SGLang 문서 <https://docs.sglang.io>
