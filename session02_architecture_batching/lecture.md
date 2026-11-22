# Session 2 이론 — SGLang 아키텍처 & Continuous Batching

> **핵심 질문:** 요청 하나는 서버 안에서 어떤 길을 지나고, 제각각 도착하는 여러 요청은 어떻게 한 batch가 되는가?

## 이 문서 사용법

- ★ 수업 핵심 / 📖 복습·심화
- 코드 발췌는 **SGLang v0.5.23**에서 가져와 **간추린 것**입니다 (로깅·멀티모달·분산·예외 처리 제거). 원문 위치는 각 발췌 위에 적었습니다.
- 수업용 추천 구성: §0 → §1.1–1.2 → §2.1 → §3.1–3.2 → §4.1–4.2 → §5.1–5.4 → §6.1

## 목차

0. ★ 지난 회차 연결
1. ★ 서버는 프로세스 3개다
2. ★ 요청 하나의 일생 — 코드로 따라가기
3. ★ 스케줄러 루프
4. ★ Static vs Continuous batching
5. ★ 부하 모델: closed-loop vs open-loop, Little's law, 대기열, SLO
6. ★ 로그와 벤치마크 도구
7. 📖 OpenAI 호환 API는 어떻게 같은 길로 들어오나
8. 요약 · 자기 점검 · 더 읽을거리

---

## 0. ★ 지난 회차 연결

Session 1 §8 표의 첫 두 줄을 오늘 봅니다.

| `generate()`의 한계 | 서빙 엔진의 답 |
|---|---|
| 요청이 제각각 도착한다 | 요청 큐 + 매 step마다 batch 재구성 |
| 짧은 요청이 긴 요청을 기다린다 | 끝난 요청은 즉시 빼고 새 요청을 넣는다 |

지난 과제에서 나온 "SGLang이 HF보다 빠른 이유" 가설들을 칠판에 적어 두고, 오늘 지울 수 있는 것을 지워 봅시다.
(힌트: 오늘 지울 수 있는 건 "batch를 잘 만들어서" 계열이고, "커널이 빨라서"·"CUDA graph" 계열은 Session 5까지 남습니다.)

---

## 1. ★ 서버는 프로세스 3개다

### 1.1 그림

```text
 HTTP 요청 (FastAPI + uvicorn)
    │
    ▼
┌──────────────────────┐   ZMQ     ┌────────────────────────┐   ZMQ      ┌──────────────────────┐
│ TokenizerManager      │ ────────▶ │ Scheduler               │ ─────────▶ │ DetokenizerManager    │
│ (메인 프로세스, asyncio) │ 토큰화된   │ (프로세스 sglang::scheduler)│ 새 token id │ (sglang::detokenizer)  │
│ - HTTP 처리            │  요청      │ - waiting_queue          │            │ - token id → 텍스트     │
│ - 텍스트 → token ids    │           │ - batch 결정             │            │   (incremental)        │
│ - 요청별 상태(ReqState)  │           │ - GPU 실행 (ModelRunner) │            │                       │
└──────────────────────┘           │ - KV pool, radix cache   │            └──────────────────────┘
          ▲                         └────────────────────────┘                        │
          └───────────────────────────── ZMQ (텍스트 조각) ─────────────────────────────┘
```

| 프로세스 | 하는 일 | v0.5.23 위치 |
|---|---|---|
| TokenizerManager | HTTP 요청을 받아 토큰화하고 Scheduler로 보냄. 결과를 기다려 HTTP로 돌려줌 | `managers/tokenizer_manager.py` |
| Scheduler | 대기열 관리, 매 iteration batch 결정, GPU forward, KV 메모리·prefix cache 소유 | `managers/scheduler.py` (+ `scheduler_components/`) |
| DetokenizerManager | 새 token id를 텍스트로 바꿔 TokenizerManager로 보냄 | `managers/detokenizer_manager.py` |

프로세스 생성은 `entrypoints/engine.py`에서 합니다 (`run_scheduler_process`, `run_detokenizer_process`). 실습에서 `ps`로 이 이름들을 직접 봅니다.

### 1.2 왜 나눴을까?

- 토큰화·detokenize·HTTP·JSON 처리는 **CPU 일**입니다. GPU를 조종하는 루프가 이걸 직접 하면 그동안 GPU에 다음 일이 안 갑니다.
- Python은 GIL 때문에 한 프로세스 안에서 CPU 일을 진짜 병렬로 못 합니다 → **프로세스를 나누고 메시지로 통신**합니다.
- 스케줄러 프로세스는 오로지 "다음에 GPU가 무엇을 할지 정하고 시키는 일"에 집중합니다. 이 루프가 빠를수록 GPU가 덜 놉니다 (Session 1 §5.7 overhead, Session 4 §5 overlap).
- 텐서 병렬(TP, GPU N장)이면 Scheduler가 GPU마다 하나씩 생기고, TokenizerManager·Detokenizer는 하나로 공유됩니다. (이 스터디는 GPU 1장)

### 1.3 ZMQ (📖)

ZeroMQ는 소켓 위의 메시지 큐 라이브러리입니다. SGLang은 같은 머신의 프로세스끼리 `ipc://` 주소의 PUSH/PULL 소켓을 씁니다.
보내는 쪽은 객체를 직렬화(pickle 등)해서 던지고 바로 돌아오고(비동기), 받는 쪽은 자기 루프에서 꺼내 씁니다.
주요 메시지 타입은 `managers/io_struct.py`에 있습니다.

| 방향 | 메시지 | 내용 |
|---|---|---|
| Tokenizer → Scheduler | `TokenizedGenerateReqInput` | rid, input_ids, sampling_params, stream 여부 … |
| Scheduler → Detokenizer | `BatchTokenIDOutput` | 이번 iteration에 token이 나온 요청들의 rid, 새 token id, finished_reason, cached_tokens … |
| Detokenizer → Tokenizer | `BatchStrOutput` | 위 + 새로 확정된 텍스트 조각 |

**batch 단위로 보낸다**는 점이 중요합니다. 요청이 64개면 decode step마다 메시지 64개가 아니라 1개입니다.

---

## 2. ★ 요청 하나의 일생 — 코드로 따라가기

### 2.1 한눈에

`POST /generate {"text": "...", "sampling_params": {"max_new_tokens": 128}, "stream": true}` 하나를 따라갑니다.

| # | 어디서 | 무슨 일 | v0.5.23 코드 |
|---|---|---|---|
| 1 | HTTP 서버 | JSON → `GenerateReqInput`, stream이면 SSE 응답 시작 | `entrypoints/http_server.py: generate_request → serve_generate_request` |
| 2 | TokenizerManager | 요청 상태 등록, 토크나이즈, Scheduler로 전송, 결과 대기 | `TokenizerManager.generate_request`, `_tokenize_one_request`, `_send_one_request`, `_wait_one_response` |
| 3 | Scheduler | 받아서 `Req` 객체로 만들고 **waiting_queue**에 넣음 | `handle_generate_request`, `_add_request_to_queue` |
| 4 | Scheduler | 이번 iteration에 prefill할 요청을 고름 (예산이 허락하는 만큼) | `get_next_batch_to_run` → `get_new_batch_prefill` |
| 5 | GPU | **prefill (EXTEND)** forward — 첫 token 생성 | `run_batch` → `TpModelWorker.forward_batch_generation` |
| 6 | Scheduler | 결과 처리, 첫 token을 Detokenizer로 보냄, 다음 iteration에 **running_batch**에 합류 | `process_batch_result` → `stream_output` |
| 7 | Detokenizer → Tokenizer | token id → 텍스트 조각 → HTTP 스트림으로 `data: {...}` | `DetokenizerManager.handle_batch_token_id_out` |
| 8 | Scheduler (반복) | running_batch 전체를 **decode** forward 한 번 → 요청마다 token 1개 | `update_running_batch` → `run_batch` |
| 9 | Scheduler | `max_new_tokens` 도달/EOS → 요청 종료, KV를 radix tree에 등록(`checkpoint`)하고 슬롯 정리 | `process_batch_result` (Session 3) |

**TTFT에 들어가는 것:** 1–7 전부 + **4에서 차례를 기다린 시간(queueing)**.
**ITL에 들어가는 것:** 8번 한 바퀴 시간 + 그 사이에 끼어든 다른 요청의 prefill(5번) 시간.

### 2.2 HTTP → TokenizerManager (📖)

```python
# entrypoints/http_server.py  (간추림)
async def serve_generate_request(obj: GenerateReqInput, request: Request):
    if obj.stream:
        async def stream_results():
            async for out in tokenizer_manager.generate_request(obj, request):
                yield b"data: " + dumps_json(out) + b"\n\n"      # Server-Sent Events 한 줄
            yield b"data: [DONE]\n\n"
        return StreamingResponse(stream_results(), media_type="text/event-stream")
    else:
        ret = await tokenizer_manager.generate_request(obj, request).__anext__()
        return ORJSONResponse(ret)
```

`generate_request`는 **async generator**입니다. 결과가 생길 때마다 `yield`하고, HTTP 서버는 그걸 바로 클라이언트로 흘려보냅니다.
우리 `common/client.py`가 `data:` 줄을 하나씩 읽으며 시각을 재는 것이 바로 이 스트림입니다.

```python
# managers/tokenizer_manager.py  (간추림)
async def generate_request(self, obj, request=None):
    self.auto_create_handle_loop()            # 결과 수신 루프(백그라운드 태스크)가 없으면 시작
    obj.normalize_batch_and_arguments()
    self._init_req_state(obj, request)        # rid → ReqState(out_list, finished, asyncio.Event) 등록
    tokenized_obj = await self._tokenize_one_request(obj)   # 텍스트 → input_ids
    await self._send_one_request(tokenized_obj)             # ZMQ로 Scheduler에 전송
    async for response in self._wait_one_response(obj, request):
        yield response
```

```python
# _wait_one_response → _stream_one_response  (간추림)
while True:
    await state.event.wait()          # handle_loop가 Detokenizer 결과를 받아 event를 set할 때까지 잠듦
    out_list, state.out_list = state.out_list, []
    state.event.clear()
    out = out_list[-1]                # 그사이 여러 조각이 쌓였으면 마지막(누적) 것
    if state.finished:
        yield out; break
    if is_stream:
        yield out
```

- 요청 하나당 **`ReqState` + `asyncio.Event`** 하나. 수천 개 요청이 동시에 기다려도 스레드가 아니라 이벤트 루프 위의 코루틴이라 가볍습니다.
- 백그라운드 `handle_loop`가 Detokenizer에서 `BatchStrOutput`을 받아 rid별 `ReqState`에 결과를 넣고 event를 깨웁니다.
- 클라이언트가 연결을 끊으면 `abort_request`로 Scheduler에 취소를 알립니다 (GPU 낭비 방지).

### 2.3 Scheduler: `Req`와 waiting_queue

```python
# managers/scheduler.py  (간추림)
def _add_request_to_queue(self, req: Req, is_retracted: bool = False):
    ...
    self.waiting_queue.append(req)
    req.time_stats.set_wait_queue_entry_time()    # 대기열 진입 시각 (TTFT 분해용)
```

`Req`(`managers/schedule_batch.py`)는 요청 하나의 모든 상태를 들고 다닙니다. 꼭 알아야 할 필드:

| 필드 | 뜻 |
|---|---|
| `rid` | 요청 ID |
| `origin_input_ids` | 프롬프트 token id |
| `output_ids` | 지금까지 생성한 token id (decode마다 늘어남) |
| `sampling_params` | `max_new_tokens`, `temperature`, `ignore_eos`, stop 조건 … |
| `prefix_len` / `last_node` | radix cache에서 재사용한 길이와 tree 노드 (Session 3) |
| `req_pool_idx` 계열 (`req.kv`) | 이 요청의 KV 슬롯 표(`req_to_token`)의 행 번호 (Session 3) |
| `finished_reason` | `FINISH_LENGTH` / `FINISH_MATCHED_TOKEN`(EOS) / `FINISH_MATCHED_STR`(stop) / `FINISH_ABORT` |
| `is_retracted` | 메모리 부족으로 쫓겨났다 돌아온 요청인가 (Session 4) |
| `time_stats` | 도착, 대기열 진입, prefill 시작, 첫 token … 시각 기록 |

`ScheduleBatch`는 `Req` 리스트와 그 batch를 GPU에서 돌리는 데 필요한 정보(KV pool 참조, forward mode, sampling 정보 …)를 묶은 것입니다.
Scheduler는 두 종류를 들고 있습니다:

- **waiting_queue**: `List[Req]` — 아직 prefill을 못 받은 요청
- **running_batch**: `ScheduleBatch` — prefill을 마치고 decode 중인 요청들

### 2.4 결과 처리: decode 한 step 뒤 (📖)

```python
# managers/scheduler_components/batch_result_processor.py: process_batch_result_decode  (간추림)
for i, req in enumerate(batch.reqs):
    if req.finished() or req.is_retracted:       # overlap 모드에서 이미 끝난 요청이 섞여 있을 수 있음
        continue
    req.output_ids.extend(next_token_ids[i])     # 보통 1개, speculative면 여러 개
    req.update_finish_state(len(next_token_ids[i]))   # EOS? stop 문자열? max_new_tokens?
    self._handle_finish_state_updated_req(req, ...)   # 끝났으면 KV 정리 (radix tree 등록)
self.output_streamer.stream_output(batch.reqs, ...)   # 이번 step의 새 token들을 batch로 Detokenizer에 전송
```

끝난 요청은 다음 iteration의 `get_next_batch_to_run`에서 running_batch로부터 걸러집니다 (`filter_batch`).

### 2.5 Detokenizer: 왜 "incremental"이 어려운가 (📖)

Session 1 §2에서 말했듯 byte-level BPE에서는 글자 하나가 token 여러 개에 걸칠 수 있습니다.
또 어떤 tokenizer는 앞 token에 따라 공백 처리가 달라서 **token 하나만 따로 decode하면 틀린 텍스트**가 나옵니다.

SGLang의 해법 — 요청마다 두 개의 offset을 유지합니다:

```python
# managers/detokenizer_manager.py  (개념만 간추림)
# s.decode_ids: 지금까지의 token id 전체
# surr_offset : '앞 문맥'으로 쓸 구간의 시작
# read_offset : 이미 텍스트로 확정해 보낸 지점
surr_text = decode(s.decode_ids[surr_offset : read_offset])    # 앞 문맥만
read_text = decode(s.decode_ids[surr_offset : ])               # 앞 문맥 + 새 token
new_text  = read_text[len(surr_text):]                          # 차이 = 새로 생긴 글자

if new_text and not new_text.endswith("�"):      # 글자가 완성됨
    emit(new_text)
    surr_offset, read_offset = read_offset, len(s.decode_ids)
else:                                             # UTF-8이 아직 덜 완성 → 출력 가능한 앞부분만 내보내고
    emit(find_printable_text(new_text))           # offset은 그대로 두어 다음 token과 함께 다시 시도
```

- 그래서 한국어 스트리밍에서 **token은 나왔는데 화면에 글자가 안 나오는 step**이 생길 수 있습니다. 클라이언트에서 ITL을 잴 때 이 효과가 섞입니다.
- 이 작업도 batch 단위로 합니다 (`batch_decode`). Detokenizer가 느려지면 TTFT·ITL이 같이 나빠지는데, GPU는 바쁘지 않습니다 — 프로파일링(Session 6)에서 구분해야 하는 경우입니다.

---

## 3. ★ 스케줄러 루프

### 3.1 normal 루프

```python
# managers/scheduler.py: event_loop_normal  (v0.5.23, 거의 원문)
while True:
    self.ingest_requests()                                   # ZMQ에서 새 요청을 꺼내 waiting_queue로
    plan = self.get_next_batch_to_run(
        running_batch=self.running_batch, last_batch=self.last_batch)
    self.running_batch = plan.running_batch
    batch = plan.batch_to_run
    if batch:
        result = self.run_batch(batch)                       # GPU forward
        self.process_batch_result(batch, result)             # token 기록, 종료 처리, 결과 전송
    else:
        self.on_idle()                                       # 할 일이 없으면 self-check 등
    self.last_batch = batch
```

이 루프 한 바퀴 = **iteration** 하나 = GPU forward 한 번(prefill batch 또는 decode batch).

### 3.2 `get_next_batch_to_run`의 결정 규칙

```python
# (간추림)
def get_next_batch_to_run(self, running_batch, last_batch):
    # 1) 직전 iteration이 prefill이었다면 그 요청들을 running_batch에 합친다
    if last_batch and last_batch.forward_mode.is_extend():
        last_batch.filter_batch(...)              # 끝난 요청, 아직 덜 끝난 chunked 요청 제외
        running_batch.merge_batch(last_batch)     # (running_batch가 비어 있으면 그대로 대체)
    running_batch.filter_batch()                  # 끝난 요청 제거

    # 2) 대기열에서 새 prefill batch를 만들 수 있으면 → prefill 우선
    new_batch = self.get_new_batch_prefill(running_batch).batch_to_run
    if new_batch is not None:
        return new_batch

    # 3) 아니면 running_batch 전체를 decode 한 step
    if not running_batch.is_empty():
        return self.update_running_batch(running_batch)    # 메모리 확인 → 부족하면 retraction (Session 4)
    return None
```

**Prefill 우선**이라는 점이 중요합니다.

- 장점: 새 요청이 빨리 running_batch에 합류 → TTFT↓, batch가 빨리 커져 throughput↑
- 대가: prefill이 도는 iteration 동안 **이미 decode 중이던 요청들은 한 step 멈춥니다** → 그 요청들의 ITL에 스파이크
  → Session 4의 chunked prefill / mixed chunk가 이걸 완화합니다.
- 요청이 계속 들어오면? prefill이 연달아 잡혀 decode가 굶을 수 있을까? — prefill 한 번에 들어가는 양은 예산(`max_prefill_tokens`, `chunked_prefill_size`, KV 메모리)으로 제한되고, 대기열이 비면 바로 decode로 돌아갑니다. 극단적인 경우를 위한 `--prefill-decode-interval` 같은 손잡이도 있습니다.

### 3.3 overlap 루프 (📖, 자세히는 Session 4 §5)

기본 루프는 `event_loop_overlap`입니다. batch N을 GPU에 **비동기로** 던진 뒤 결과를 기다리지 않고 batch N+1을 준비하고, batch N의 결과 처리는 그다음에 합니다.

```python
# (간추림)
while True:
    self.ingest_requests()
    batch = self.get_next_batch_to_run(...).batch_to_run
    if batch:
        batch_result = self.run_batch(batch)              # GPU에 던지고 바로 돌아옴
        self.result_queue.append((batch.copy(), batch_result))
    if self.last_batch:
        tmp_batch, tmp_result = self.result_queue.popleft()
        self.process_batch_result(tmp_batch, tmp_result)  # 한 step 늦게 처리 → CPU 시간을 GPU 시간 뒤에 숨김
    self.last_batch = batch
```

---

## 4. ★ Static vs Continuous batching

### 4.1 타임라인

요청 A(출력 2), B(출력 6), C(출력 3)가 t=0에, D가 t=2에 도착. 한 칸 = decode step 1번.

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

static: 12 step 동안 슬롯 4 × 12 = 48칸 중 실제 일 17칸 (35%). continuous: 8 step, 빈칸 없음에 가까움.
그리고 **D의 TTFT**가 6 step → 1 step으로 줄었습니다.

### 4.2 무엇이 있어야 가능한가

- 이 아이디어의 출발점: **Orca** (OSDI '22) — "batch 단위가 아니라 **iteration 단위**로 스케줄링하자".
- **길이가 제각각인 batch를 한 번에 계산**할 수 있어야 합니다.
  - linear layer(MLP, QKV 투영)는 token마다 독립이므로, 모든 요청의 token을 **한 줄로 이어 붙여(ragged, padding 없음)** 큰 행렬 하나로 곱하면 됩니다.
  - attention만 요청 경계를 알아야 합니다 → 요청별 길이(`seq_lens`)와 KV 위치 표를 같이 넘기고, attention 커널이 요청마다 따로 처리합니다 (Session 5).
- **KV 메모리를 요청마다 "최대 길이만큼 미리" 잡지 않고 token 단위로 그때그때** 할당해야 합니다. 그래야 요청이 언제 들어오고 나가도 메모리를 재활용할 수 있습니다 (Session 3). vLLM의 PagedAttention(SOSP '23)이 이 부분을 정리한 대표 작업입니다.

### 4.3 prefill과 decode를 한 batch에 섞을 수 있나? (📖)

ragged batch라면 원리적으로는 됩니다 — 요청 A의 prefill 500 token과 요청 B, C의 decode 1 token씩을 이어 붙이면 503 token짜리 batch입니다.
SGLang 기본값은 섞지 않고 iteration마다 prefill 또는 decode 중 하나를 하지만, chunked prefill과 함께 `--enable-mixed-chunk`를 켜면 섞습니다 (Session 4 §4).
섞을지 말지는 TTFT와 ITL 사이의 선택입니다.

---

## 5. ★ 부하 모델 — 서버를 어떻게 괴롭힐 것인가

### 5.1 두 가지 부하

| | Closed-loop (동시성 고정) | Open-loop (도착률 고정) |
|---|---|---|
| 규칙 | 항상 C개 요청이 열려 있음. 하나 끝나면 다음 요청 | 초당 λ개가 (Poisson으로) 도착. 서버 상태와 무관 |
| 벤치마크 인자 | `--max-concurrency C` | `--request-rate λ` |
| 닮은 현실 | 사용자 수가 정해진 내부 도구, 배치 작업, 에이전트 워커 풀 | 공개 서비스 트래픽 |
| 서버가 느려지면 | 요청이 덜 들어온다 (**자동 감속**) | **대기열이 끝없이 쌓인다** |

같은 서버라도 어느 부하로 재느냐에 따라 "좋아 보이는 숫자"가 다릅니다. 리포트에는 **어느 방식인지 반드시 적습니다.**

closed-loop의 함정(📖): 서버가 잠깐 멈추면 클라이언트도 새 요청을 안 보내므로 "멈춘 동안 도착했어야 할 요청들"의 대기 시간이 측정에서 빠집니다.
부하 테스트 분야에서 **coordinated omission**이라 부르는 현상입니다. 그래서 "용량(capacity)"은 open-loop로 확인합니다.

### 5.2 Little's law

```text
L = λ × W        (평균 동시 요청 수 = 도착률 × 평균 체류 시간)
```

예제:

1. 동시성 32 (closed-loop), 요청당 E2E 4초 → 처리량 λ ≈ 32 / 4 = **8 req/s**.
2. 8 req/s가 도착(open-loop)하는데 E2E가 4초면 서버 안에 평균 32개가 떠 있습니다 → KV 메모리가 32개 분량 필요.
3. 요청당 평균 768 token(입력 512 + 출력 256), KV pool 12만 token이면 메모리 기준 동시 수용 ≈ 156개. 즉 이 workload에서는 메모리보다 **step 시간**이 먼저 한계에 옵니다 — Session 1 §5.4의 ridge(≈ batch 165)와 비슷한 수준이라는 것도 우연이 아닙니다.

### 5.3 대기열은 왜 갑자기 폭발하나 (📖)

가장 단순한 대기열 모델(M/M/1: Poisson 도착 λ, 처리율 μ)에서 평균 체류 시간은

```text
W = 1 / (μ − λ)
```

μ = 10 req/s일 때:

| λ (도착률) | 활용률 λ/μ | W |
|---:|---:|---:|
| 5 | 50% | 0.2 s |
| 8 | 80% | 0.5 s |
| 9 | 90% | 1.0 s |
| 9.5 | 95% | 2.0 s |
| 9.9 | 99% | 10 s |
| ≥ 10 | ≥ 100% | **무한대** (대기열이 끝없이 증가) |

LLM 서버는 batch가 커지면 처리율 μ 자체가 올라가므로 이 모델보다는 덜 가파르지만, **용량 근처에서 latency가 비선형으로 폭발하는 모양**은 같습니다.
그래서 운영에서는 용량의 70–80% 정도를 목표로 잡습니다.

### 5.4 Latency–Throughput 곡선

```text
latency
  │                                      ╱   ← 용량 초과: 대기열 폭발
  │                                    ╱
  │                                 ╱
  │                       ___..--'
  │ ___________....---''''               ← "공짜 구간": 부하↑ throughput↑, latency 거의 그대로
  └────────────────────────────────────── throughput
```

- 왼쪽 평평한 구간: GPU가 memory-bound라 batch가 커져도 step 시간이 거의 안 늘어남 (Session 1 §5.4).
- 오른쪽 꺾이는 지점: (a) step 시간이 batch에 비례하기 시작하거나 (compute-bound, KV 읽기 증가) (b) KV 메모리가 차서 요청을 더 못 받아 **대기열**이 생김.
- 어느 쪽이 먼저인지는 workload가 결정합니다. 입력이 길면 (b)가, 짧으면 (a)가 먼저 옵니다 (Session 2 과제).

### 5.5 SLO와 goodput

서비스는 "최대 throughput"이 아니라 **"약속을 지키면서 낼 수 있는 최대 throughput"** 이 중요합니다.

```text
SLO 예시:  p95 TTFT ≤ 500 ms  그리고  p95 ITL ≤ 50 ms
capacity : SLO를 만족하는 최대 부하
goodput  : SLO를 만족한 요청만 센 throughput
```

DistServe(OSDI '24) 같은 연구는 "throughput 대신 goodput을 최대화하자"는 관점으로 prefill/decode 분리 설계를 정당화했습니다 (Session 6 §5).
`sglang.benchmark.serving`도 `--goodput` 인자로 SLO 기준 goodput을 계산할 수 있습니다.

---

## 6. ★ 로그와 벤치마크 도구

### 6.1 스케줄러 로그 두 종류

```text
Prefill batch, #new-seq: 3, #new-token: 1536, #cached-token: 0, token usage: 0.05, #running-req: 13, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 15234.56
Decode batch, #running-req: 16, #token: 9216, token usage: 0.07, cuda graph: True, gen throughput (token/s): 3021.55, #queue-req: 0
```

(v0.5.23 `scheduler_components/metrics_reporter.py`의 `report_prefill_stats`, decode 쪽 동일 파일)

| 필드 | 뜻 |
|---|---|
| `#new-seq` | 이번 prefill에 새로 들어온 요청 수 |
| `#new-token` | 이번 prefill에서 실제로 계산한 token 수 |
| `#cached-token` | prefix cache에서 재사용해 계산을 건너뛴 token 수 (Session 3) |
| `token usage` | KV pool 점유율 (0–1) |
| `#running-req` | decode 중인 요청 수 = 현재 batch 크기 |
| `#queue-req` | 아직 prefill을 못 받고 기다리는 요청 수 |
| `#token` (Decode) | running 요청들이 지금 점유한 KV token 수 |
| `cuda graph` | 이 step이 CUDA graph로 실행됐는가 (Session 5) |
| `gen throughput` | 최근 구간의 decode token/s |

- Decode 로그는 **40 step마다 한 번** 찍힙니다 (`--decode-log-interval`, 기본 40).
- 해석 습관: `#queue-req`가 0보다 크면 "누군가 기다리고 있다". 그 이유가 메모리(`token usage` 높음)인지, 설정(`max_running_requests`)인지 구분합니다 (Session 4).
- `common/logparse.py`가 이 로그를 표로 바꿔 줍니다.

### 6.2 공식 벤치마크 도구

```bash
python -m sglang.benchmark.serving --backend sglang --port 30000 --model Qwen/Qwen3-1.7B \
  --dataset-name random-ids --random-input-len 512 --random-output-len 256 --random-range-ratio 1.0 \
  --num-prompts 128 --max-concurrency 32 --output-file results/x.jsonl
```

| 인자 | 주의할 점 |
|---|---|
| `--dataset-name random-ids` | 랜덤 token id로 정확한 길이의 프롬프트 (다운로드 불필요). `random`은 ShareGPT 텍스트를 받아 씀 |
| `--random-range-ratio 1.0` | 모든 요청 길이를 정확히 같게. **기본값 0.0이면 길이가 0~지정값 사이에서 랜덤** → 비교가 흐려짐 |
| `--max-concurrency` / `--request-rate` | closed / open loop. 둘 다 주면 "λ로 도착하되 동시 C개까지만" |
| `--num-prompts` | 동시성의 수 배 이상이어야 정상 상태(steady state) 구간이 생김 |
| `--flush-cache`, `--warmup-requests` | 측정 전 cache 비우기, 워밍업 |
| (기본) ignore_eos | 출력 길이를 정확히 지정값으로 생성. `--disable-ignore-eos`로 끔 |
| `--output-file` | 결과 JSON 한 줄을 **추가(append)** |

출력 지표 정의 (벤치마크 클라이언트 쪽에서 계산):

- TTFT: 요청 전송 → 첫 스트림 청크 도착
- ITL: 연속한 스트림 청크 사이 간격 (청크 하나에 token이 여러 개 오면 그 간격 하나)
- TPOT: (E2E − TTFT) / (출력 token 수 − 1)
- concurrency: 실제로 측정된 평균 동시 요청 수 (= Little's law의 L)

구 경로 `python -m sglang.bench_serving`도 동작하지만 deprecated 경고가 뜹니다.

---

## 7. 📖 OpenAI 호환 API는 어떻게 같은 길로 들어오나

`POST /v1/chat/completions`는 `entrypoints/openai/` 아래의 핸들러가 받습니다.

1. `messages` 리스트를 모델의 **chat template**(tokenizer_config.json의 Jinja 템플릿)으로 렌더링 → 하나의 프롬프트 문자열/ token 시퀀스
2. OpenAI 인자(`max_tokens`, `temperature`, `stop` …)를 SGLang `sampling_params`로 변환
3. **같은 `TokenizerManager.generate_request`** 로 들어감 → 이후 경로는 §2와 동일
4. 결과를 OpenAI 응답 형식(`choices`, `usage`, 스트림이면 `chat.completion.chunk`)으로 포장

그래서 성능 특성은 native `/generate`와 같고, 차이는 template 렌더링과 포장 비용 정도입니다.
`chat_template_kwargs`(예: Qwen3의 `enable_thinking`)는 1번 단계에 영향을 줍니다 — Session 3에서 이게 prefix cache에 어떤 영향을 주는지 봅니다.

---


## ★ 실습 전 예측 질문

실습 직전에 노트에 **숫자로** 적으세요. 실습 후 실제 값과 비교하는 것이 이 회차에서 가장 많이 배우는 순간입니다.

1. 입력 512 / 출력 256, 동시성을 1 → 64로 올리면 output throughput은 몇 배? median ITL은 몇 배?
2. 동시성 몇에서 p95 TTFT가 급격히 나빠지기 시작할까? 그 이유는 연산 포화일까, 메모리 포화일까? (힌트: KV pool ≈ 12만 token, 요청당 최대 768 token)
3. open-loop로 초당 몇 개까지는 TTFT가 안정적일까? Little's law로 추정해 보세요 (동시성 sweep 결과의 E2E를 이용).
4. `staggered.py` 시나리오에서 B(출력 64)는 A(출력 512)보다 먼저 끝날까? B가 도착했을 때 A의 ITL은 어떻게 될까?

---

## 8. 요약 · 자기 점검 · 더 읽을거리

### 요약

1. SGLang 서버 = TokenizerManager(HTTP·토큰화) + Scheduler(batch 결정·GPU) + DetokenizerManager, ZMQ로 batch 단위 메시지.
2. 요청은 waiting_queue → (prefill) → running_batch → (decode 반복) → 종료.
3. 스케줄러 루프 한 바퀴 = iteration = forward 한 번. 규칙은 "prefill 우선, 없으면 decode".
4. Continuous batching = iteration 단위로 batch를 재구성. ragged batch와 token 단위 KV 할당이 전제.
5. 부하는 closed-loop(동시성)와 open-loop(도착률) 두 가지. capacity는 open-loop로, 리포트에 방식 명시.
6. Little's law L = λW. 용량 근처에서 latency는 비선형으로 폭발.
7. 서비스 기준은 SLO를 지키는 throughput(goodput).

### 자기 점검 문제

1. TokenizerManager와 Scheduler를 한 프로세스로 합치면 어떤 지표가 나빠질까? 왜?
2. 동시성 64로 closed-loop 측정했더니 E2E 평균이 8초였다. request throughput은? 같은 throughput이 open-loop로 들어오면 서버 안의 평균 동시 요청 수는?
3. "prefill 우선" 규칙에서 decode 중인 사용자가 겪는 손해는 무엇이고, 어떤 상황에서 가장 커지나?
4. static batching에서 batch 크기 8, 출력 길이가 [10, 10, 10, 10, 10, 10, 10, 200]이면 슬롯 활용률은?
5. 한국어 스트리밍에서 "token은 생성됐는데 클라이언트에 아무것도 안 오는 step"이 생기는 이유는?
6. `--random-range-ratio`를 기본값(0.0)으로 두고 두 설정을 비교하면 무엇이 문제인가?
7. M/M/1에서 μ = 20 req/s. W ≤ 0.5 s를 지키려면 λ는 최대 몇?

<details>
<summary>답</summary>

1. TTFT와 ITL. 토큰화·detokenize·HTTP 처리 동안 GPU에 다음 batch가 안 가서 GPU가 논다 (overhead 증가).
2. 64 / 8 = 8 req/s. 같은 W라면 L = 8 × 8 = 64개 (W가 같다는 가정 하에).
3. 새 요청의 prefill iteration 동안 decode가 멈춰 ITL 스파이크. 긴 프롬프트가 들어올 때, 그리고 요청이 연달아 들어올 때 가장 크다.
4. 일한 칸 = 7×10 + 200 = 270, 전체 = 8×200 = 1600 → 약 17%.
5. 한글 한 글자가 token 여러 개에 걸쳐 있어 글자가 완성될 때까지 Detokenizer가 출력을 보류(`�` 처리)하기 때문.
6. 요청마다 입력/출력 길이가 랜덤이라 두 실행의 workload가 같지 않다 (같은 seed가 아니면 더더욱). 평균 길이도 지정값의 절반 정도가 됨.
7. 1/(20 − λ) ≤ 0.5 → λ ≤ 18 req/s.
</details>

### 더 읽을거리

- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI '22) — iteration-level scheduling
- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (SOSP '23) — vLLM
- Zhong et al., *DistServe* (OSDI '24) — goodput 관점
- Gil Tene, "How NOT to Measure Latency" (강연) — coordinated omission
- SGLang 코드: `managers/tokenizer_manager.py`, `managers/scheduler.py`, `managers/detokenizer_manager.py` ([docs/code_map.md](../docs/code_map.md))

### 다음 회차 예고

오늘 "KV 메모리를 token 단위로 그때그때 할당해야 continuous batching이 된다"고 했습니다. 다음 회차는 그 메모리 관리, 그리고 같은 prefix를 다시 계산하지 않는 RadixAttention을 봅니다.
