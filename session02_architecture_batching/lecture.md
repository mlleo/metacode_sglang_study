# Session 2 이론 — SGLang 아키텍처 & Continuous Batching

> 50분. "요청 하나를 끝까지 따라가기"(§2)와 "부하를 거는 두 가지 방법"(§5)이 핵심입니다.

| 절 | 내용 | 시간 |
|---|---|---|
| §0 | 지난 회차 연결 | 2분 |
| §1 | 서버는 프로세스 3개다 | 7분 |
| §2 | 요청 하나의 일생 (코드 이름으로) | 10분 |
| §3 | 스케줄러 루프 — 매 iteration에 일어나는 일 | 8분 |
| §4 | Static vs Continuous batching | 8분 |
| §5 | 부하 모델: closed-loop vs open-loop, Little's law, SLO | 10분 |
| §6 | 로그 읽는 법 + 벤치마크 도구 | 5분 |

---

## §0. 지난 회차 연결

Session 1 §6 표의 첫 두 줄을 오늘 봅니다.

| `generate()`의 한계 | 서빙 엔진의 답 |
|---|---|
| 요청이 제각각 도착한다 | 요청 큐 + 매 step마다 batch 재구성 |
| 짧은 요청이 긴 요청을 기다린다 | 끝난 요청은 즉시 빼고 새 요청을 넣는다 |

지난 과제에서 나온 "SGLang이 HF보다 빠른 이유" 가설들을 칠판에 적어 두고, 오늘 하나씩 지워 봅시다.

---

## §1. 서버는 프로세스 3개다

```text
 HTTP 요청
    │
    ▼
┌────────────────────┐    ZMQ     ┌──────────────────────┐    ZMQ     ┌────────────────────┐
│ TokenizerManager    │ ─────────▶ │ Scheduler             │ ─────────▶ │ DetokenizerManager  │
│ (메인, asyncio+HTTP) │            │ (sglang::scheduler)   │  token ids │ (sglang::detokenizer)│
│ 텍스트 → token ids   │            │ batch 결정 + GPU 실행   │            │ token ids → 텍스트    │
└────────────────────┘            └──────────────────────┘            └────────────────────┘
          ▲                                                                       │
          └───────────────────────────── ZMQ (텍스트 조각) ───────────────────────┘
```

**왜 나눴을까?**

- 토크나이즈·detokenize·HTTP 처리는 **CPU 일**입니다. 스케줄러 루프가 이걸 직접 하면 그동안 GPU가 놉니다.
- Python은 GIL 때문에 한 프로세스 안에서 CPU 일을 진짜 병렬로 못 합니다 → **프로세스를 나누고 메시지(ZMQ)로 통신**합니다.
- 스케줄러 프로세스는 오로지 "다음에 GPU가 무엇을 할지 정하고 시키는 일"에 집중합니다. 이 루프가 빠를수록 GPU가 덜 놉니다.

텐서 병렬(TP, GPU 여러 장)이면 Scheduler가 GPU마다 하나씩 생깁니다 (이 스터디는 GPU 1장).

---

## §2. 요청 하나의 일생

`POST /generate {"text": "...", "sampling_params": {"max_new_tokens": 128}}` 하나를 따라갑니다.
**함수 이름을 기억할 필요는 없습니다.** 다만 과제의 아키텍처 다이어그램에는 이 이름들을 붙여 주세요 ([docs/code_map.md](../docs/code_map.md)).

| # | 어디서 | 무슨 일 | v0.5.23 코드 |
|---|---|---|---|
| 1 | HTTP 서버 | JSON → `GenerateReqInput` | `entrypoints/http_server.py: generate_request` |
| 2 | TokenizerManager | 토크나이즈, 요청 상태 등록, Scheduler로 전송 | `TokenizerManager.generate_request`, `_tokenize_one_request` |
| 3 | Scheduler | 받아서 `Req` 객체로 만들고 **waiting_queue**에 넣음 | `handle_generate_request`, `_add_request_to_queue` |
| 4 | Scheduler | 이번 iteration에 prefill할 요청을 고름 (예산이 허락하는 만큼) | `get_next_batch_to_run` → `get_new_batch_prefill` |
| 5 | GPU | **prefill (EXTEND)** forward — 첫 token 생성 | `run_batch` → `TpModelWorker.forward_batch_generation` |
| 6 | Scheduler | 결과 처리, 첫 token을 Detokenizer로 보냄, 요청을 **running_batch**에 합류 | `process_batch_result` |
| 7 | Detokenizer → Tokenizer | token id → 텍스트 조각 → HTTP 스트림으로 `data: {...}` | `DetokenizerManager.handle_batch_token_id_out` |
| 8 | Scheduler (반복) | running_batch 전체를 **decode** forward 한 번 → 요청마다 token 1개 | `update_running_batch` → `run_batch` |
| 9 | Scheduler | `max_new_tokens` 도달/EOS → 요청 종료, KV를 radix tree에 등록(`checkpoint`)하고 슬롯 정리 | `process_batch_result` (Session 3) |

**TTFT에 들어가는 것:** 1–7 전부 + **4에서 차례를 기다린 시간(queueing)**.
**ITL에 들어가는 것:** 8번 한 바퀴 시간 + 그 사이에 끼어든 다른 요청의 prefill(5번) 시간.

---

## §3. 스케줄러 루프 — 매 iteration에 일어나는 일

`managers/scheduler.py`의 `event_loop_normal`을 단순화하면 이게 전부입니다.

```python
while True:
    recv_reqs = ingest_requests()            # 새로 도착한 요청 → waiting_queue
    batch = get_next_batch_to_run()          # 이번에 GPU에 무엇을 시킬까?
    if batch:
        result = run_batch(batch)            # GPU forward (prefill 또는 decode)
        process_batch_result(batch, result)  # token 기록, 끝난 요청 정리, 결과 전송
    last_batch = batch
```

`get_next_batch_to_run`의 결정 규칙 (단순화):

```text
1. 직전 iteration이 prefill이었다면 → 그 요청들을 running_batch에 합친다
2. waiting_queue에서 새 prefill batch를 만들 수 있으면 → prefill을 실행  (prefill 우선!)
3. 아니면 → running_batch 전체를 decode 한 step
```

**Prefill 우선**이라는 점이 중요합니다.

- 장점: 새 요청이 빨리 running_batch에 합류 → TTFT↓, batch가 빨리 커져 throughput↑
- 대가: prefill이 도는 동안 **이미 decode 중이던 요청들은 멈춥니다** → 그 요청들의 ITL에 스파이크
  → Session 4의 chunked prefill이 이걸 완화합니다.

실제 기본 루프는 `event_loop_overlap`입니다 — GPU가 batch N을 계산하는 동안 CPU가 batch N+1을 준비합니다 (Session 4 §5).

---

## §4. Static vs Continuous batching

요청 4개: A(출력 2), B(출력 6), C(출력 3)가 t=0에, D가 t=2에 도착. 한 칸 = decode step 1번.

```text
Static batching (batch를 한 번 만들면 다 끝날 때까지 고정)
step:  1  2  3  4  5  6  7  8  9 10 11 12
 A    ██ ██ ·· ·· ·· ··                     ← 끝났는데 자리 차지 (빈 슬롯)
 B    ██ ██ ██ ██ ██ ██
 C    ██ ██ ██ ·· ·· ··
 D                      ██ ██ ██ ██ ██ ██   ← t=2에 왔지만 다음 batch까지 대기

Continuous batching (iteration-level scheduling)
step:  1  2  3  4  5  6  7  8
 A    ██ ██
 B    ██ ██ ██ ██ ██ ██
 C    ██ ██ ██
 D          ▓▓ ██ ██ ██ ██ ██               ← 도착 다음 iteration에 prefill(▓) 후 합류
```

- 이 아이디어의 출발점: **Orca** (OSDI '22) — "batch 단위가 아니라 iteration 단위로 스케줄링하자".
- continuous batching이 가능하려면 **요청마다 길이가 다른 batch**를 한 번에 계산할 수 있어야 합니다.
  SGLang은 padding을 하지 않고 모든 요청의 token을 **한 줄로 이어 붙인(ragged)** 텐서 + 요청별 길이(`seq_lens`)로 넘기고,
  attention 커널이 요청 경계를 알아서 처리합니다 (Session 5).
- 그리고 KV 메모리를 요청마다 "최대 길이만큼 미리" 잡지 않고 **token 단위로 그때그때** 할당해야 합니다 (Session 3).

---

## §5. 부하 모델 — 서버를 어떻게 괴롭힐 것인가

### 두 가지 부하

| | Closed-loop (동시성 고정) | Open-loop (도착률 고정) |
|---|---|---|
| 규칙 | 항상 C개 요청이 열려 있음. 하나 끝나면 다음 요청 | 초당 λ개가 (Poisson으로) 도착. 서버 상태와 무관 |
| 벤치마크 인자 | `--max-concurrency C` | `--request-rate λ` |
| 닮은 현실 | 사용자 수가 정해진 내부 도구, 배치 작업 | 공개 서비스 트래픽 |
| 서버가 느려지면 | 요청이 덜 들어온다 (자동 감속) | **대기열이 끝없이 쌓인다** |

같은 서버라도 어느 부하로 재느냐에 따라 "좋아 보이는 숫자"가 다릅니다. 리포트에는 **어느 방식인지 반드시 적습니다.**

### Little's law

```text
L = λ × W        (평균 동시 요청 수 = 도착률 × 평균 체류 시간)
```

- 동시성 32, 요청당 E2E 4초 → 처리량 ≈ 32 / 4 = **8 req/s**.
- 거꾸로, 8 req/s가 도착하는데 E2E가 4초면 서버 안에 평균 32개가 떠 있습니다 → KV 메모리가 32개 분량 필요.

### Latency–Throughput 곡선

```text
latency
  │                                      ╱   ← 용량 초과: 대기열 폭발
  │                                    ╱
  │                                 ╱
  │                       ___..--'
  │ ___________....---''''               ← "공짜 구간": 부하↑ throughput↑, latency 거의 그대로
  └────────────────────────────────────── throughput
```

- 왼쪽 평평한 구간: GPU가 memory-bound라 batch가 커져도 step 시간이 거의 안 늘어남 (Session 1 §4).
- 오른쪽 꺾이는 지점: (a) step 시간이 batch에 비례하기 시작하거나 (b) KV 메모리가 차서 요청을 더 못 받아 **대기열**이 생김.

### SLO와 goodput

서비스는 "최대 throughput"이 아니라 **"약속을 지키면서 낼 수 있는 최대 throughput"** 이 중요합니다.

```text
SLO 예시:  p95 TTFT ≤ 500 ms  그리고  p95 ITL ≤ 50 ms
capacity : SLO를 만족하는 최대 부하
goodput  : SLO를 만족한 요청만 센 throughput
```

오늘 과제가 이 "sweet spot"을 데이터로 찾는 것입니다.

---

## §6. 로그 읽는 법 + 벤치마크 도구

### 스케줄러 로그 두 종류

```text
Prefill batch, #new-seq: 3, #new-token: 1536, #cached-token: 0, token usage: 0.05, #running-req: 13, #queue-req: 0, ...
Decode batch, #running-req: 16, #token: 9216, token usage: 0.07, cuda graph: True, gen throughput (token/s): 3021.55, #queue-req: 0
```

| 필드 | 뜻 |
|---|---|
| `#new-seq` | 이번 prefill에 새로 들어온 요청 수 |
| `#new-token` | 이번 prefill에서 실제로 계산한 token 수 |
| `#cached-token` | prefix cache에서 재사용해 계산을 건너뛴 token 수 (Session 3) |
| `token usage` | KV pool 점유율 (0–1) |
| `#running-req` | decode 중인 요청 수 = 현재 batch 크기 |
| `#queue-req` | 아직 prefill을 못 받고 기다리는 요청 수 |
| `cuda graph` | 이 step이 CUDA graph로 실행됐는가 (Session 5) |
| `gen throughput` | 최근 구간의 decode token/s |

- Decode 로그는 **40 step마다 한 번** 찍힙니다 (`--decode-log-interval`, 기본 40).
- 해석 습관: `#queue-req`가 0보다 크면 "누군가 기다리고 있다". 그 이유가 메모리(`token usage` 높음)인지, 설정(`max_running_requests`)인지 구분합니다 (Session 4).
- `common/logparse.py`가 이 로그를 표로 바꿔 줍니다.

### 공식 벤치마크 도구

```bash
python -m sglang.benchmark.serving --backend sglang --port 30000 \
  --dataset-name random-ids --random-input-len 512 --random-output-len 256 --random-range-ratio 1.0 \
  --num-prompts 128 --max-concurrency 32 --output-file results/x.jsonl
```

- `random-ids`: 랜덤 token id로 정확한 길이의 프롬프트 생성 (데이터셋 다운로드 불필요). `random`은 ShareGPT 텍스트를 받아 씁니다.
- `--random-range-ratio 1.0`: 모든 요청 길이를 정확히 같게. 기본값 0.0이면 길이가 0~지정값 사이에서 랜덤 → 비교가 흐려집니다.
- 출력 길이는 기본적으로 EOS 무시(`ignore_eos`)로 정확히 지정 길이만큼 생성합니다.
- 구 경로 `python -m sglang.bench_serving`도 동작하지만 deprecated 경고가 뜹니다.

---

## 예측 질문 (실습 전에 숫자로)

1. 입력 512 / 출력 256, 동시성을 1 → 64로 올리면 output throughput은 몇 배? median ITL은 몇 배?
2. 동시성 몇에서 p95 TTFT가 급격히 나빠지기 시작할까? 그 이유는 연산 포화일까, 메모리 포화일까? (힌트: KV pool ≈ 12만 token, 요청당 최대 768 token)
3. open-loop로 초당 몇 개까지는 TTFT가 안정적일까? Little's law로 추정해 보세요 (동시성 sweep 결과의 E2E를 이용).
4. `staggered.py` 시나리오에서 B(출력 64)는 A(출력 512)보다 먼저 끝날까? B가 도착했을 때 A의 ITL은 어떻게 될까?
