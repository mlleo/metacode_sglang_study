# Session 6 이론 — Benchmarking · Profiling · 비용, 그리고 더 큰 그림

> **핵심 질문:** "30% 빨라졌다"를 어떻게 믿게 만들고, 1M token에 얼마가 드는가? 그리고 이 스터디 밖에는 무엇이 있는가?

## 이 문서 사용법

- ★ 수업 핵심 / 📖 복습·심화
- §1의 함정 표는 지난 5회차 동안 우리가 직접 밟은 것들입니다. 학생들의 과제가 최고의 예시입니다.
**수업 40분**

- 12분 · §1.2의 반복 결과를 읽고, 팀별로 같은 workload를 짝지어 측정하는 방법을 한 가지 제안합니다.
- 13분 · §2.2에서 trace 한 장을 읽습니다. GPU가 비는 순간을 표시하고, 같은 시각의 CPU와 복사 stream을 확인합니다.
- 10분 · §4.1의 두 부하에서 비용을 구한 다음 Session 2의 SLO를 적용합니다.
- 5분 · 최종 프로젝트의 비교 대상과 결과를 판정할 지표를 한 줄씩 적습니다.

Prometheus와 멀티 GPU 지도는 실습 자료 또는 복습에서 이어서 읽습니다.

## 목차

0. ★ 6회차 지도 다시 보기
1. ★ 좋은 벤치마크, 나쁜 벤치마크
2. ★ 프로파일링: 어디가 느린지 아는 4단계
3. ★ 운영 지표 (Prometheus metrics)
4. ★ 비용: $/1M tokens
5. ★ 더 큰 그림: 이 스터디 밖의 기술들
6. ★ 최종 프로젝트 안내
7. 요약 · 자기 점검 · 더 읽을거리

---

## 0. ★ 6회차 지도 다시 보기

| `generate()`의 한계 | 서빙 엔진의 답 | 회차 | 우리가 잰 것 |
|---|---|---|---|
| 요청이 제각각 도착 | continuous batching | 2 | latency–throughput 곡선, capacity |
| KV 메모리 낭비 | token 단위 KV pool | 3 | KV pool 크기 손계산 = 실측 |
| 같은 prefix 재계산 | RadixAttention | 3 | 공유 비율 → TTFT/throughput |
| 메모리 부족 시 OOM | admission, retraction | 4 | new_token_ratio, retract 비용 |
| 긴 프롬프트가 모두를 멈춤 | chunked prefill + mixed/interval | 4 | TTFT ↔ ITL p99 교환 |
| CPU launch overhead | CUDA graph, overlap | 4, 5 | step 시간 분해, 달성 대역폭 |
| 메모리 대역폭 한계 | 양자화, speculative | 5 | FP8 효과가 사라지는 batch |
| 측정 수단이 없다 | 로그, metrics, profiler, 벤치마크 도구 | **6** | 오늘 |

---

## 1. ★ 좋은 벤치마크, 나쁜 벤치마크

좋은 벤치마크의 세 조건: **재현 가능(reproducible)**, **통제됨(controlled)**, **대표성(representative)**.

### 1.1 우리가 이미 밟은 함정들

| 함정 | 어디서 봤나 | 결과가 어떻게 왜곡되나 |
|---|---|---|
| warm-up 없이 측정 | S1 (첫 실행의 JIT, 메모리 할당) | 첫 점만 이상하게 느림 |
| cache가 따뜻한 채로 다음 측정 | S3 (`loadgen` seed 처리, `--flush-cache`) | 뒤 측정일수록 TTFT가 좋아 보임 |
| 입력·출력 길이가 요청마다 다름 | S2 (`--random-range-ratio` 기본 0.0) | 조건 비교가 흐려짐 |
| `ignore_eos=True`로만 측정 | S4 | 스케줄러가 가장 보수적으로 동작 → 실제 서비스와 다른 admission |
| 평균/중앙값만 봄 | S4 (ITL p50은 같고 p99만 다름) | 사용자가 겪는 멈춤이 안 보임 |
| closed-loop 결과로 "용량" 주장 | S2 | 서버가 느려지면 부하도 줄어 latency가 좋아 보임 (coordinated omission) |
| "chunk를 줄였으니 ITL이 줄었겠지" | S4 | 기본 설정에서는 안 줄어듦 — 측정 전에는 모른다 |
| 한 번만 실행 | 모든 회차 | 노이즈를 효과로 착각 |
| 랜덤 token 프롬프트 | 모든 회차 | 실제 텍스트·출력 길이 분포와 다름 (EOS 시점, prefix 공유 패턴, 언어) |
| 클라이언트가 병목 | (가능성) | Python 클라이언트가 CPU를 다 쓰면 서버가 아니라 클라이언트를 잰 것 |
| 다른 GPU·다른 호스트 | RunPod Community Cloud | 같은 4090이라도 호스트마다 CPU·PCIe·전력 제한·클럭이 다름 |

### 1.2 반복과 신뢰구간

같은 설정을 n번 돌려 평균 x̄, 표준편차 s를 얻었다면, 평균의 95% 신뢰구간은

```text
x̄ ± t(0.975, n−1) × s / √n          t: n=3 → 4.30, n=5 → 2.78, n=10 → 2.26
```

예: throughput 3,000 / 3,100 / 2,950 tok/s → x̄ = 3,017, s ≈ 76 → 95% CI ≈ ±190 (±6.3%)

- A와 B를 같은 호스트에서 같은 seed/workload로 짝지어 실행했다면, 쌍마다 `B − A`를 구하고 그 차이의 95% CI를 계산합니다. 0을 포함하면 이 데이터만으로 차이의 방향을 가리기 어렵습니다. 독립 두 표본은 Welch 방식 등으로 평균 차이의 CI를 구합니다. 설정마다 구한 CI의 겹침은 차이의 CI를 대신할 수 없습니다.
- n=3에서 t값이 큰(4.30) 이유는 표본이 적어 s도 불확실하기 때문입니다. 같은 정도의 변동이라면 반복을 늘릴수록 구간은 좁아집니다.
- **tail 지표(p99)는 평균보다 훨씬 노이즈가 큽니다.** 요청 100개의 p99는 사실상 "가장 나쁜 1–2개"입니다. p99를 비교하려면 요청 수를 늘리세요 (최소 수백 개).
- 반복은 **같은 호스트에서 연속으로** 한 것(측정 노이즈)과 **다른 Pod에서** 한 것(환경 노이즈)을 구분해서 적습니다.

### 1.3 steady state와 ramp 효과 (📖)

closed-loop로 동시성 C, 요청 N개를 보내면:

```text
동시 요청 수
  C │      ┌──────────────────────┐
    │     ╱                        ╲          ← 끝부분: 남은 요청이 C보다 적어져 batch가 작아짐
    │    ╱                          ╲
  0 └───┴──────────────────────────────┴── 시간
       ramp-up      steady state      ramp-down
```

N이 C의 1–2배밖에 안 되면 측정의 상당 부분이 ramp 구간이라 throughput이 실제보다 낮게, latency 분포는 이상하게 나옵니다.
→ **N ≥ 4–10 × C** (우리 스크립트는 4배, 최소 32). 정밀한 측정이라면 앞뒤 구간을 잘라 냅니다.

### 1.4 workload의 대표성 (📖)

| 항목 | random-ids | 실제 서비스 |
|---|---|---|
| 입력 길이 | 고정 또는 균일 | 긴 꼬리 분포 (대부분 짧고 가끔 매우 김) |
| 출력 길이 | 고정 (ignore_eos) | EOS로 제각각, 짧은 답이 많음 |
| prefix 공유 | 없음 | system prompt, 멀티턴, few-shot으로 매우 많음 |
| 언어 | — | 한국어면 token 수 ↑ |
| 도착 패턴 | 즉시 또는 Poisson | 시간대별 출렁임, burst |

`sglang.benchmark.serving`의 `--dataset-name sharegpt`(실제 대화 길이 분포), `generated-shared-prefix`(prefix 공유), `mooncake`/`agentic-trace`(실제 trace 재생) 등을 쓰면 대표성을 높일 수 있습니다.
가능하면 **자기 서비스의 로그에서 길이 분포를 뽑아** 재현하는 것이 가장 좋습니다.

### 1.5 체크리스트 (최종 프로젝트에 그대로 적용)

```text
□ 질문이 한 문장이고, 바꾸는 변수가 하나다 (또는 명시적인 2축 grid)
□ 고정 조건을 적었다: GPU(+호스트 ID), 모델, dtype, SGLang 버전, 서버 인자 전체, 입출력 길이 분포, 동시성/도착률, sampling
□ warm-up을 버렸다, cache 상태(cold/warm)를 명시했다
□ 3회 이상 반복하고 평균 ± 95% CI를 적었다 (repeat_stats.py)
□ 분포(p50/p95/p99)를 봤다. 최소한 TTFT와 ITL의 p99
□ 부하 모델(closed/open)을 적었다, 요청 수 ≥ 4 × 동시성
□ 결과를 이론(어느 항이 줄었나)으로 설명했다
□ 이 실험으로 말할 수 없는 것을 적었다
```

---

## 2. ★ 프로파일링 — 어디가 느린지 아는 4단계

| 단계 | 도구 | 답하는 질문 | 비용 |
|---|---|---|---|
| 1 | 서버 로그, `/metrics` | 지금 batch가 몇이고, 대기열이 있는가? 메모리는? | 0 |
| 2 | `py-spy dump/top` (S2) | CPU(Python)는 어느 함수에서 시간을 쓰나? | 거의 0 |
| 3 | **torch profiler** (`/start_profile`, `sglang.profiler`) | step 하나 안에서 어떤 커널이 얼마나? GPU가 노는 구간은? | 녹화하는 step만 느려짐 |
| 4 | Nsight Systems / Compute (`nsys`, `ncu`) | 커널 하나가 왜 느린가? (메모리 throughput, occupancy) | 큼 |

항상 **1단계부터** 올라갑니다. 대기열이 쌓여서 TTFT가 나쁜 문제를 커널 프로파일로 찾으려 하면 시간만 버립니다.

### 2.1 torch profiler로 녹화하기

```bash
# 서버가 부하를 받고 있는 동안
curl -X POST http://127.0.0.1:30000/start_profile -H "Content-Type: application/json" \
  -d '{"output_dir": "/workspace/profiles/c32", "num_steps": 10, "activities": ["CPU", "GPU"]}'
# num_steps를 주면 10 step 후 자동 정지. 안 주면 /stop_profile 호출까지
```

- `start_step`으로 앞의 몇 step을 건너뛸 수 있습니다 (warm-up 제외).
- `profile_by_stage`로 prefill과 decode를 따로 녹화할 수 있습니다.
- `python3 -m sglang.benchmark.serving ... --profile`도 같은 엔드포인트를 씁니다.
- 결과 `*.trace.json.gz`는 <https://ui.perfetto.dev>에서 엽니다 (브라우저에서만 처리, 업로드 없음).

### 2.2 trace 읽기

```text
CPU 스레드 (scheduler)   │▓▓ schedule ▓▓│ launch launch launch … │▓ process result ▓│
GPU stream              │      ░░░░ (놀고 있음) ░░░░  ███ gemm ███ ██ attn ██ … │
                                        ↑ 여기가 넓으면 CPU-bound (launch overhead, 스케줄링)
```

보는 순서:

1. **GPU 줄에 빈 칸이 있는가?** → 있다면 CPU launch, 동기화, 다른 stream의 복사, 들어온 요청이 없는 구간을 차례로 확인합니다. CUDA graph가 켜져 있나? (켜져 있으면 CPU 쪽에 `cudaGraphLaunch` 하나가 보이고, GPU 쪽에 커널 수백 개가 빈틈없이 붙어 있음)
2. **GPU 시간의 대부분은 어떤 커널인가?** 커널 이름으로 분류합니다 (이름은 라이브러리·버전마다 다름):

   | 이름에 들어 있는 것 | 정체 |
   |---|---|
   | `gemm`, `cutlass`, `xmma`, `sm89_` | linear layer (weight 읽기) |
   | `BatchDecode`, `BatchPrefill`, `flashinfer`, `fwd_kernel` | attention |
   | `rms`, `norm` | RMSNorm |
   | `silu`, `act_and_mul` | MLP activation |
   | `rotary`, `rope` | RoPE |
   | `sampling`, `argmax`, `topk` | sampling |
   | `Memcpy`, `copy`, `index` | 데이터 복사·인덱싱 |

3. **memcpy가 눈에 띄는가?** CPU↔GPU 복사가 step마다 크게 있으면 의심 (예: 매 step 큰 텐서를 CPU로 가져오는 코드)
4. **step 사이의 간격**: 연속한 forward 사이의 GPU 빈칸을 찾고 CPU 쪽에서 같은 시간대에 무엇을 하는지 봅니다.

`session06_benchmark_production/code/trace_summary.py`는 1, 2를 숫자로 뽑아 줍니다 (`gpu_busy_pct`, 카테고리별 비율).

### 2.3 진단 예시

| 관찰 | 해석 | 다음 행동 |
|---|---|---|
| batch=1, GPU busy 40%, 커널 사이 빈칸이 고르게 많음 | launch overhead | CUDA graph 확인, overlap 확인 |
| batch=64, GPU busy 95%, gemm 70% | GEMM 이벤트가 대부분의 시간을 차지 | bandwidth와 연산 지표로 병목 확인 |
| batch=64, context 8K, attention 60% | attention 비중이 큼 | 같은 batch에서 context만 바꿔 확인 |
| forward 사이 간격이 5 ms씩 | 스케줄러 CPU가 느림 (LPM 정렬, 큰 대기열 등) | py-spy로 scheduler 프로세스 확인 |
| GPU 바쁨, 그런데 클라이언트 ITL은 들쭉날쭉 | 서버 밖(Detokenizer, HTTP, 네트워크, 클라이언트) | metrics와 클라이언트 측정 비교 (§3.2) |

### 2.4 Nsight Systems (📖)

```bash
nsys profile -o s05 --force-overwrite=true \
  python3 -m sglang.benchmark.one_batch --model-path Qwen/Qwen3-1.7B --batch-size 1 --input-len 256 \
  --profile --profile-activities CUDA_PROFILER
```

nsys는 CPU 스레드, CUDA API, 커널, 메모리 복사, NVTX 범위를 하나의 타임라인으로 보여 줍니다. torch profiler보다 overhead가 작고 시스템 전체를 보지만, 파일이 크고 GUI가 필요합니다.
커널 하나를 깊게 보려면 Nsight Compute(`ncu`)로 메모리 throughput·occupancy를 봅니다 — 이 스터디 범위 밖입니다.

---

## 3. ★ 운영 지표 (Prometheus metrics)

### 3.1 무엇이 있나

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

- **counter**는 계속 증가만 합니다. 처리량은 두 시점의 차이를 시간으로 나눠 구합니다 (Prometheus의 `rate()`).
- **histogram**은 bucket별 누적 개수입니다. p99는 bucket 경계 사이를 선형 보간한 **근사치**입니다 (`metrics_probe.py`의 `hist_quantile`). bucket이 듬성하면 오차가 큽니다.
- histogram은 서버가 켜진 뒤 **전체 누적**입니다. 특정 구간의 분포는 두 시점 bucket의 차이로 구합니다.

### 3.2 클라이언트 측정과 서버 측정이 다를 때

```text
클라이언트 TTFT  =  네트워크 + HTTP 파싱 + 토큰화 + [대기열 + prefill]  + 첫 token detokenize + 전송
서버 TTFT(metrics) =                토큰화 + [대기열 + prefill] + …     (문서 기준: API 서버에서 요청 생성 ~ 첫 출력)
```

둘의 차이가 크면 서버 밖(네트워크, 프록시, 클라이언트 자체의 CPU)을 의심합니다. 같은 Pod 안의 localhost라면 차이는 작아야 합니다.

### 3.3 운영 알람 예 (📖)

| 조건 | 의미 |
|---|---|
| `num_queue_reqs > 0`이 5분 지속 | 용량 부족 — 증설 또는 부하 제한 |
| TTFT p99 > SLO | 대기열 또는 긴 prefill 간섭 |
| `token_usage > 0.95` 지속 | 메모리 압박 — retraction 증가 예상 |
| `cache_hit_rate` 급락 | 프롬프트 템플릿 변경, 라우팅 변경 |

---

## 4. ★ 비용: $/1M tokens

### 4.1 공식

```text
$/1M output tokens = ($/hour) ÷ (output tokens/s × 3600) × 10⁶ ÷ (해당 부하로 처리하는 시간 비율)
```

예 (숫자는 예시 — 여러분의 측정값과 Deploy 화면 가격으로 다시 계산하세요):

```text
RTX 4090 Community, $0.59/h, 9,000 output tok/s (동시성 64)
→ 0.59 / (9,000 × 3600) × 10⁶ ≈ $0.018 / 1M output tokens   (100% 활용 가정)
활용률 30%(밤에는 놀고 낮에만 붐빔) → ≈ $0.061
SLO(p95 TTFT ≤ 500 ms) 때문에 동시성 16까지만 쓸 수 있고 그때 4,000 tok/s라면 → ≈ $0.041 (100%) / $0.137 (30%)
```

### 4.2 교훈

- **가장 빠른 설정 ≠ 가장 싼 설정 ≠ SLO를 지키는 가장 싼 설정.** 셋을 따로 계산합니다.
- **비싼 GPU가 더 쌀 수 있습니다.** H100이 4090보다 시간당 몇 배 비싸도 throughput이 그 이상이면 token당 비용은 낮아집니다. 특히 메모리가 커서 batch를 크게 키울 수 있으면. (최종 프로젝트 E 주제)
- 여기서 30%는 예시 처리량으로 서비스가 동작한 시간 비율입니다. `nvidia-smi`의 GPU utilization이나 trace의 GPU busy %는 이 계산에 그대로 넣을 수 없습니다. 트래픽이 출렁이는 서비스에서는 autoscaling, 여러 모델의 GPU 공유, 배치 작업으로 빈 시간 채우기가 커널 최적화보다 큰 효과를 내기도 합니다.
- 이 스터디에서 쓴 최적화 대부분(batching, prefix cache, 양자화, CUDA graph)은 결국 **같은 GPU-시간에 더 많은 token**, 즉 비용 최적화였습니다.

### 4.3 입력 token과 출력 token (📖)

API 가격표는 보통 입력과 출력을 따로 매깁니다(출력이 몇 배 비쌈). 이유는 지금까지 배운 것 그대로입니다:

- 입력 token은 prefill에서 **한 번에 병렬로** 처리 (compute-bound, GPU 효율 높음)
- 출력 token은 decode에서 **하나씩** 처리 (memory-bound, step당 weight·KV 전체를 읽음)
- cache hit된 입력은 계산 자체를 안 하므로 더 쌈 (상용 API의 "cached input" 할인)

자기 서버의 비용을 입력/출력으로 나누려면 prefill 시간과 decode 시간을 분리해 측정해야 합니다 (Session 1 `manual_decode.py`의 방식).

---

## 5. ★ 더 큰 그림 — 이 스터디 밖의 기술들

GPU 한 장, 작은 모델로 배운 개념들이 큰 시스템에서 어떻게 확장되는지 지도를 그립니다.

### 5.1 한눈에

| 기술 | 해결하는 문제 | 이 스터디의 어느 개념과 연결 | SGLang 플래그/키워드 |
|---|---|---|---|
| **Tensor Parallel (TP)** | 모델이 GPU 한 장에 안 들어감 | weight 읽기를 N장으로 나눔 → decode 하한 ÷N (통신 비용 추가) | `--tp-size` |
| **Data Parallel (DP) / router** | 요청이 많음 | 서버 여러 개 + prefix를 고려한 라우팅 | `--dp-size`, 라우터(SMG 등) |
| **DP attention, Expert Parallel (EP)** | MoE 모델(DeepSeek, Qwen-MoE) | attention과 expert를 다르게 나눠 KV 중복 제거 | `--attn-dp-size`, `--ep-size` |
| **PD disaggregation** | prefill과 decode가 서로 방해 (S4) | prefill 서버와 decode 서버를 분리, KV를 네트워크로 전송 | `--disaggregation-mode prefill/decode` |
| **HiCache** | GPU KV pool이 작음 (S3) | radix tree를 GPU → CPU 메모리 → 디스크/원격 저장소로 계층화 | `--enable-hierarchical-cache` |
| **Structured output** | JSON 등 형식 강제 | sampling 단계에서 문법에 맞지 않는 token을 mask | `json_schema`, `regex`, xgrammar |
| **Multi-LoRA** | 어댑터 여러 개 동시 서빙 | 같은 base weight 공유, radix cache는 어댑터별 네임스페이스 (S3 §7) | `--lora-paths` |
| **다른 엔진** | — | vLLM, TensorRT-LLM, LMDeploy … 질문은 같고 선택이 다름 | 최종 프로젝트 G 주제 |

### 5.2 Tensor Parallel의 산수 (📖)

layer마다 행렬을 GPU N장에 나눠 계산하고, attention 뒤와 MLP 뒤에 **all-reduce**로 결과를 합칩니다 (layer당 2회).

```text
decode step 하한 ≈ (weight bytes / N) / 대역폭  +  2 × layers × all-reduce 시간
```

- weight 읽기는 N분의 1이지만, all-reduce가 layer마다 2번(Qwen3-1.7B면 56번) 있습니다.
- decode batch가 작으면 all-reduce 데이터는 작지만(32 × 2048 × 2 B = 128 KB) **지연시간**이 문제입니다. NVLink 연결(데이터센터 GPU)에서는 수–수십 μs, PCIe만 있는 소비자 GPU(4090)에서는 훨씬 깁니다.
- 그래서 TP는 "모델이 안 들어갈 때" 또는 "NVLink가 있을 때" 쓰고, 작은 모델을 여러 장에 돌릴 때는 DP가 보통 더 낫습니다.

### 5.3 PD disaggregation의 산수 (📖)

prefill 서버가 계산한 KV를 decode 서버로 보내야 합니다.

```text
2048 token 프롬프트의 KV (Qwen3-1.7B) = 2048 × 112 KiB ≈ 224 MiB
  PCIe 4.0 x16 (~25 GB/s)    → ≈ 9 ms
  100 Gb/s RDMA (~12.5 GB/s) → ≈ 19 ms
  prefill 자체 (4090)        → 수십 ms
```

- KV 전송이 prefill 시간보다 충분히 작아야 이득입니다. 큰 모델(KV가 큰 MHA 모델)일수록 빠른 네트워크(RDMA, NVLink)가 필수입니다.
- 이득: decode 서버는 prefill에 방해받지 않아 ITL이 안정적(Session 4 문제의 구조적 해결), prefill/decode 서버 수를 따로 조절 가능.
- 대가: 시스템 복잡도, 네트워크 비용, KV 전송 지연이 TTFT에 추가.

### 5.4 HiCache의 산수 (📖)

GPU KV pool에서 밀려난 prefix를 CPU 메모리(L2)나 디스크/원격(L3)에 보관했다가 다시 쓸 때 가져옵니다.

```text
같은 2048 token prefix 224 MiB를
  CPU 메모리에서 PCIe로 다시 올리기 ≈ 9 ms
  vs 처음부터 다시 prefill ≈ 수십 ms
→ 가져오는 쪽이 이득 (prefix가 길고 모델이 클수록 더)
```

멀티턴 대화가 많고 사용자가 돌아오는 간격이 긴 서비스(GPU 캐시에서는 이미 지워졌을 시간)에서 효과적입니다. 이전 스터디 자료의 HiCache 세션이 이 주제를 깊게 다룹니다.

### 5.5 Structured output (📖)

"반드시 이 JSON schema에 맞게 출력"을 보장하려면 매 decode step에서 **문법상 올 수 없는 token의 logit을 −∞로** 만들고 sampling합니다.

- 비용: 매 step 문법 상태를 진행시키고 vocab 15만 크기의 mask를 만드는 CPU/GPU 작업. 단순하게 구현하면 decode가 크게 느려집니다.
- SGLang은 xgrammar 등 문법 엔진으로 mask 계산을 빠르게 하고(문법 상태별 mask 캐시 등), 가능한 한 GPU 실행과 겹쳐 비용을 숨깁니다. 겹칠 수 없는 경우가 Session 4 §5.2에서 본 grammar 동기화 예외입니다.
- 형식이 확정된 구간(예: `{"name": "`)은 모델을 돌리지 않고 건너뛰는 최적화(jump-forward)도 있습니다.

### 5.6 새 기술을 읽는 질문

새 엔진이나 새 기술을 만나면 항상 이 질문으로 읽으세요:

> batch는 어떻게 만드는가? KV는 어떻게 관리하는가? 이전 계산은 어떻게 재사용하는가?
> GPU에서 병목은 무엇이고, 이 기술은 step 시간 식(Session 5 §3)의 어느 항을 줄이는가? 그리고 그게 정말인지 어떻게 재는가?

---

## 6. ★ 최종 프로젝트 안내

[final_project.md](final_project.md) — 작고 명확한 시스템 질문 하나를 골라 **§1.5의 체크리스트대로** 답합니다.
Session 5의 Optimization Note를 확장해도 좋습니다.

---


## ★ 실습 전 예측 질문

실습 직전에 노트에 **숫자로** 적으세요. 실습 후 실제 값과 비교하는 것이 이 회차에서 가장 많이 배우는 순간입니다.

1. 동시성 1과 32에서 decode trace의 `gpu_busy_pct`는 각각 몇 %일까? CUDA graph를 끄면?
2. 같은 벤치마크를 3번 돌리면 throughput의 95% CI는 평균의 몇 % 폭일까? p99 TTFT는?
3. 지금까지 여러분이 잰 결과 중, SLO를 지키면서 가장 싼 $/1M token은 어느 설정이었나요?

---

## 7. 요약 · 자기 점검 · 더 읽을거리

### 요약

1. 좋은 벤치마크 = 재현 가능 + 통제됨 + 대표성. 우리가 밟은 함정 목록을 체크리스트로.
2. 반복해서 95% CI를 적고, CI가 겹치면 차이를 주장하지 않는다. p99는 요청 수가 많아야 안정적.
3. 요청 수 ≥ 4–10 × 동시성 (ramp 효과).
4. 프로파일링은 로그 → py-spy → torch profiler → nsys 순으로 올라간다.
5. trace에서 GPU 빈칸(CPU 병목)과 지배 커널(gemm vs attention)을 본다.
6. metrics는 서버 측정, 클라이언트 측정과의 차이가 서버 밖 비용.
7. $/1M tokens = $/h ÷ (tok/s × 3600) × 10⁶ ÷ 활용률. SLO와 활용률이 비용을 지배.
8. TP, DP, PD 분리, HiCache, structured output은 모두 이 스터디의 개념(weight 읽기, KV, prefill/decode 간섭, sampling)의 확장.

### 자기 점검 문제

1. 같은 seed끼리 나열한 A throughput 5회: 4100, 4180, 4050, 4220, 4150. B: 4300, 4250, 4380, 4200, 4310. 각 쌍의 B − A로 95% CI를 구하면? (t=2.78)
2. 동시성 64, 요청 100개로 측정했다. 무엇이 문제이고 어떻게 고치나?
3. batch=1 decode trace에서 GPU busy가 95%인데 ITL이 이론 하한의 1.5배다. 무엇을 의심하나?
4. $0.59/h GPU, SLO 만족 throughput 5,000 tok/s, 평균 활용률 40%. $/1M output tokens는?
5. PD disaggregation에서 프롬프트 8K, Qwen3-1.7B, 25 GB/s 링크. KV 전송 시간은?
6. 클라이언트 TTFT p50 120 ms, 서버 metrics TTFT p50 40 ms (같은 Pod). 무엇을 의심하나?

<details>
<summary>답</summary>

1. 차이는 200, 70, 330, −20, 160 tok/s입니다. 평균 148, 표본 표준편차 약 133, 95% CI는 약 [−17, 313] tok/s입니다. B의 평균이 더 높았지만 이 다섯 쌍으로는 차이의 방향을 가리기 어렵습니다.
2. 요청 수가 동시성의 1.6배 → 대부분이 ramp 구간. 요청 수를 256 이상(4배)으로, 가능하면 앞뒤 구간 제외.
3. GPU는 바쁘지만 하한보다 느림 → 커널 자체 효율(작은 batch gemm의 비효율), attention/sampling 등 weight 외 작업, 실제 대역폭이 스펙보다 낮음(전력 제한, 클럭). 커널별 시간 분포와 달성 대역폭을 확인.
4. 0.59 / (5000 × 3600) × 10⁶ / 0.4 ≈ $0.082
5. 8192 × 112 KiB ≈ 896 MiB ≈ 0.94 GB → / 25 GB/s ≈ 38 ms
6. 80 ms가 서버 밖에서 듦: 클라이언트 자체가 느림(Python 스레드·CPU 포화), 스트림 첫 청크의 detokenize 지연(텍스트가 비어 있는 청크), 클라이언트의 시각 측정 위치. 클라이언트 동시성을 낮춰 차이가 줄어드는지 확인.
</details>

### 더 읽을거리

- Brendan Gregg, *Systems Performance* (2nd ed.) — 측정 방법론 일반
- Gil Tene, "How NOT to Measure Latency" (강연)
- PyTorch Profiler 문서, Perfetto UI 문서, NVIDIA Nsight Systems 문서
- Zhong et al., *DistServe* (OSDI '24); Qin et al., *Mooncake* (FAST '25); Shoeybi et al., *Megatron-LM* (TP)
- Dong et al., *XGrammar* (2024); Sheng et al., *S-LoRA* (2023)
- SGLang 문서: Benchmark and Profiling, Production Metrics, PD Disaggregation, HiCache (`docs/`)

### 스터디를 마치며

이 스터디의 목표는 "SGLang을 써 봤다"가 아니었습니다. 새 엔진, 새 논문, 새 최적화를 만났을 때

> 그건 step 시간 식의 어느 항을 줄이는가? 어떤 workload에서 효과가 사라지는가? 그리고 그걸 어떻게 재는가?

를 묻고 직접 확인할 수 있게 되는 것이었습니다.

> Build less. Measure more. Understand why.
