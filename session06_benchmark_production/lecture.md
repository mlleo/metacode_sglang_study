# Session 6 이론 — Benchmarking · Profiling · 비용, 그리고 더 큰 그림

> 50분. §1(벤치마크 함정)은 지난 5회차 동안 우리가 직접 밟은 함정들로 설명하세요. 학생들의 과제가 최고의 예시입니다.

| 절 | 내용 | 시간 |
|---|---|---|
| §0 | 6회차 지도 다시 보기 | 3분 |
| §1 | 좋은 벤치마크, 나쁜 벤치마크 | 12분 |
| §2 | 프로파일링: 어디가 느린지 아는 4단계 | 10분 |
| §3 | 운영 지표 (Prometheus metrics) | 4분 |
| §4 | 비용: $/1M tokens | 7분 |
| §5 | 더 큰 그림: 이 스터디 밖의 기술들 | 10분 |
| §6 | 최종 프로젝트 안내 | 4분 |

---

## §0. 6회차 지도 다시 보기

| `generate()`의 한계 | 서빙 엔진의 답 | 회차 | 우리가 잰 것 |
|---|---|---|---|
| 요청이 제각각 도착 | continuous batching | 2 | latency–throughput 곡선, capacity |
| KV 메모리 낭비 | token 단위 KV pool | 3 | KV pool 크기 손계산 = 실측 |
| 같은 prefix 재계산 | RadixAttention | 3 | 공유 비율 → TTFT/throughput |
| 메모리 부족 시 OOM | admission, retraction | 4 | new_token_ratio, retract 비용 |
| 긴 프롬프트가 모두를 멈춤 | chunked prefill | 4 | TTFT ↔ ITL p99 교환 |
| CPU launch overhead | CUDA graph, overlap | 4, 5 | step 시간 분해, 달성 대역폭 |
| 메모리 대역폭 한계 | 양자화, speculative | 5 | FP8 효과가 사라지는 batch |
| 측정 수단이 없다 | 로그, metrics, profiler, 벤치마크 도구 | **6** | 오늘 |

---

## §1. 좋은 벤치마크, 나쁜 벤치마크

좋은 벤치마크의 세 조건: **재현 가능(reproducible)**, **통제됨(controlled)**, **대표성(representative)**.

### 우리가 이미 밟은 함정들

| 함정 | 어디서 봤나 | 결과가 어떻게 왜곡되나 |
|---|---|---|
| warm-up 없이 측정 | S1 (첫 실행의 JIT, 메모리 할당) | 첫 점만 이상하게 느림 |
| cache가 따뜻한 채로 다음 측정 | S3 (`loadgen` seed 처리, `--flush-cache`) | 뒤 측정일수록 TTFT가 좋아 보임 |
| 입력·출력 길이가 요청마다 다름 | S2 (`--random-range-ratio` 기본 0.0) | 조건 비교가 흐려짐 |
| `ignore_eos=True`로만 측정 | S4 | 스케줄러가 가장 보수적으로 동작 → 실제 서비스와 다른 admission |
| 평균/중앙값만 봄 | S4 (ITL p50은 같고 p99만 다름) | 사용자가 겪는 멈춤이 안 보임 |
| closed-loop 결과로 "용량" 주장 | S2 | 서버가 느려지면 부하도 줄어 latency가 좋아 보임 |
| 한 번만 실행 | 모든 회차 | 노이즈를 효과로 착각 |
| 랜덤 token 프롬프트 | 모든 회차 | 실제 텍스트·출력 길이 분포와 다름 (EOS 시점, prefix 공유 패턴) |
| 클라이언트가 병목 | (가능성) | Python 클라이언트가 CPU를 다 쓰면 서버가 아니라 클라이언트를 잰 것 |
| 다른 GPU·다른 호스트 | RunPod Community Cloud | 같은 4090이라도 호스트마다 CPU·PCIe·클럭이 다름 |

### 체크리스트 (최종 프로젝트에 그대로 적용)

```text
□ 질문이 한 문장이고, 바꾸는 변수가 하나다
□ 고정 조건을 적었다: GPU(+호스트 ID), 모델, dtype, SGLang 버전, 서버 인자 전체, 입출력 길이 분포, 동시성/도착률, sampling
□ warm-up을 버렸다, cache 상태(cold/warm)를 명시했다
□ 3회 이상 반복하고 평균 ± 95% CI를 적었다 (repeat_stats.py)
□ 분포(p50/p95/p99)를 봤다. 최소한 TTFT와 ITL의 p99
□ 부하 모델(closed/open)을 적었다
□ 결과를 이론(어느 항이 줄었나)으로 설명했다
□ 이 실험으로 말할 수 없는 것을 적었다
```

**"A가 B보다 빠르다"** 는 두 신뢰구간이 겹치지 않을 때만 말합니다. 겹치면 "이 측정으로는 차이를 구분할 수 없다"가 정답입니다.

---

## §2. 프로파일링 — 어디가 느린지 아는 4단계

| 단계 | 도구 | 답하는 질문 | 비용 |
|---|---|---|---|
| 1 | 서버 로그, `/metrics` | 지금 batch가 몇이고, 대기열이 있는가? 메모리는? | 0 |
| 2 | `py-spy dump/top` (S2) | CPU(Python)는 어느 함수에서 시간을 쓰나? | 거의 0 |
| 3 | **torch profiler** (`/start_profile`, `sglang.profiler`) | step 하나 안에서 어떤 커널이 얼마나? GPU가 노는 구간은? | 녹화하는 step만 느려짐 |
| 4 | Nsight Systems / Compute (`nsys`, `ncu`) | 커널 하나가 왜 느린가? (메모리 throughput, occupancy) | 큼 — 이 스터디 범위 밖 |

### torch profiler trace 읽기 (Perfetto, <https://ui.perfetto.dev>)

```text
CPU 스레드 (scheduler)   │▓▓ schedule ▓▓│ launch launch launch … │▓ process result ▓│
GPU stream              │      ░░░░ (놀고 있음) ░░░░  ███ gemm ███ ██ attn ██ … │
                                        ↑ 여기가 넓으면 CPU-bound (launch overhead, 스케줄링)
```

보는 순서:

1. **GPU 줄에 빈 칸이 있는가?** → 있다면 CPU가 병목. CUDA graph가 켜져 있나? (`cudaGraphLaunch` 하나로 묶여 보이면 켜진 것)
2. **GPU 시간의 대부분은 어떤 커널인가?** decode에서 작은 batch는 gemm(weight 읽기), 긴 context·큰 batch는 attention(KV 읽기)
3. **memcpy가 눈에 띄는가?** CPU↔GPU 복사가 step마다 있으면 의심

`trace_summary.py`는 이 중 1, 2를 숫자로 뽑아 줍니다 (`gpu_busy_pct`, 카테고리별 비율).

---

## §3. 운영 지표 (Prometheus metrics)

`--enable-metrics`로 띄우면 `/metrics`에서 Prometheus 형식으로 노출합니다 (Grafana 대시보드로 연결하는 것이 표준).

| 지표 | 종류 | 의미 |
|---|---|---|
| `sglang:num_running_reqs` / `num_queue_reqs` | gauge | 지금 batch 크기 / 대기열 |
| `sglang:token_usage` | gauge | KV pool 점유율 |
| `sglang:cache_hit_rate` | gauge | prefix cache hit 비율 |
| `sglang:gen_throughput` | gauge | 최근 decode token/s |
| `sglang:time_to_first_token_seconds` | histogram | TTFT 분포 |
| `sglang:inter_token_latency_seconds` | histogram | ITL 분포 |
| `sglang:prompt_tokens_total` / `generation_tokens_total` | counter | 누적 처리량 |

운영 알람 예: "`num_queue_reqs` > 0 이 5분 지속" → 용량 부족, "TTFT p99 > SLO" → 긴 prefill 간섭 또는 대기열.
벤치마크 숫자(클라이언트 측정)와 metrics(서버 측정)가 다르면, 그 차이가 **네트워크·HTTP·detokenize·클라이언트** 시간입니다.

---

## §4. 비용: $/1M tokens

```text
$/1M output tokens = ($/hour) ÷ (output tokens/s × 3600) × 10⁶ ÷ (평균 활용률)
```

예 (숫자는 예시 — 여러분의 측정값과 Deploy 화면 가격으로 다시 계산하세요):

```text
RTX 4090 Community, $0.59/h, 9,000 output tok/s (동시성 64)
→ 0.59 / (9,000 × 3600) × 10⁶ ≈ $0.018 / 1M output tokens   (100% 활용 가정)
활용률 30%(밤에는 놀고 낮에만 붐빔) → ≈ $0.061
SLO(p95 TTFT ≤ 500 ms) 때문에 동시성 16까지만 쓸 수 있고 그때 4,000 tok/s라면 → ≈ $0.041 (100%) / $0.137 (30%)
```

교훈:

- **가장 빠른 설정 ≠ 가장 싼 설정 ≠ SLO를 지키는 가장 싼 설정.** 셋을 따로 계산합니다.
- 비싼 GPU가 더 쌀 수 있습니다: H100이 4090보다 시간당 몇 배 비싸도 throughput이 그 이상이면 token당 비용은 낮아집니다. (최종 프로젝트 E 주제)
- 이 스터디에서 쓴 최적화 대부분은 결국 **같은 GPU-시간에 더 많은 token**, 즉 비용 최적화였습니다.

---

## §5. 더 큰 그림 — 이 스터디 밖의 기술들

GPU 한 장, 작은 모델로 배운 개념들이 큰 시스템에서 어떻게 확장되는지 지도만 그립니다.

| 기술 | 해결하는 문제 | 이 스터디의 어느 개념과 연결 | SGLang 플래그/키워드 |
|---|---|---|---|
| **Tensor Parallel (TP)** | 모델이 GPU 한 장에 안 들어감 | weight 읽기를 N장으로 나눔 → decode 하한 ÷N (통신 비용 추가) | `--tp-size` |
| **Data Parallel (DP) / router** | 요청이 많음 | 서버 여러 개 + prefix를 고려한 라우팅(같은 prefix는 같은 서버로) | `--dp-size`, 라우터(SMG 등) |
| **DP attention, Expert Parallel (EP)** | MoE 모델(DeepSeek, Qwen-MoE) | attention과 expert를 다르게 나눠 KV 중복 제거 | `--attn-dp-size`, `--ep-size` |
| **PD disaggregation** | prefill과 decode가 서로 방해 (S4) | **prefill 서버와 decode 서버를 분리**, KV를 네트워크로 전송 → S4의 interference를 구조적으로 제거 | `--disaggregation-mode prefill/decode` |
| **HiCache** | GPU KV pool이 작음 (S3) | radix tree를 GPU → CPU 메모리 → 디스크/원격 저장소로 계층화 | `--enable-hierarchical-cache` |
| **Structured output** | JSON 등 형식 강제 | sampling 단계에서 문법에 맞지 않는 token을 mask (S1 §1 sampling) | `json_schema`, `regex`, xgrammar |
| **Multi-LoRA** | 어댑터 여러 개 동시 서빙 | 같은 base weight를 공유, radix cache는 어댑터별 네임스페이스 (S3 §7) | `--lora-paths` |
| **다른 엔진** | — | vLLM, TensorRT-LLM, LMDeploy … 질문은 같고 선택이 다름 | 최종 프로젝트 G 주제 |

새 엔진이나 새 기술을 만나면 항상 이 질문으로 읽으세요:

> batch는 어떻게 만드는가? KV는 어떻게 관리하는가? 이전 계산은 어떻게 재사용하는가?
> GPU에서 병목은 무엇이고, 이 기술은 step 시간 식의 어느 항을 줄이는가? 그리고 그게 정말인지 어떻게 재는가?

---

## §6. 최종 프로젝트 안내

[final_project.md](final_project.md) — 작고 명확한 시스템 질문 하나를 골라 **오늘의 체크리스트대로** 답합니다.
Session 5의 Optimization Note를 확장해도 좋습니다.

## 예측 질문

1. 동시성 1과 32에서 decode trace의 `gpu_busy_pct`는 각각 몇 %일까? CUDA graph를 끄면?
2. 같은 벤치마크를 3번 돌리면 throughput의 95% CI는 평균의 몇 % 폭일까? p99 TTFT는?
3. 지금까지 여러분이 잰 결과 중, SLO를 지키면서 가장 싼 $/1M token은 어느 설정이었나요?
