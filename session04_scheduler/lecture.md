# Session 4 이론 — 스케줄러 심화: Admission · Retraction · Chunked Prefill · Overlap

> 50분. §2의 예산 계산을 **숫자 예제로 칠판에서** 직접 해 보는 것이 이 회차의 핵심입니다.

| 절 | 내용 | 시간 |
|---|---|---|
| §0 | 지난 회차 연결 | 2분 |
| §1 | 한 iteration을 끝까지 | 6분 |
| §2 | Admission control — 누구를 들여보낼 것인가 | 14분 |
| §3 | Retraction — 들여보낸 걸 후회할 때 | 8분 |
| §4 | Chunked prefill — 긴 프롬프트가 모두를 멈출 때 | 10분 |
| §5 | Overlap scheduling — CPU 시간 숨기기 | 5분 |
| §6 | 손잡이(knob) 정리표 | 5분 |

---

## §0. 지난 회차 연결

| `generate()`의 한계 | 서빙 엔진의 답 | 오늘 |
|---|---|---|
| 메모리가 모자라면 그냥 OOM | admission control, retraction | §2, §3 |
| 긴 프롬프트 하나가 모두를 멈춘다 | chunked prefill | §4 |
| Python이 커널을 하나씩 launch (CPU overhead) | overlap scheduling (+ CUDA graph는 Session 5) | §5 |

Session 3에서 "캐시는 언제든 돌려받을 수 있는 빌린 메모리"라고 했습니다. 오늘은 **진짜로 메모리가 모자랄 때** 스케줄러가 어떻게 행동하는지 봅니다.

---

## §1. 한 iteration을 끝까지

```text
get_next_batch_to_run()
 ├─ 직전 prefill batch를 running_batch에 합침
 ├─ get_new_batch_prefill()                     ← §2 admission
 │    ├─ waiting_queue 정렬 (fcfs / lpm / ...)
 │    ├─ PrefillAdder 생성: 이번 iteration의 "예산" 계산
 │    └─ for req in waiting_queue:
 │          r = adder.add_one_req(req)          → CONTINUE / NO_TOKEN / OTHER
 │          if r != CONTINUE: break             ← 여기서 잘린 요청들이 #queue-req
 │    → 새 요청이 하나라도 있으면 prefill batch 반환 (prefill 우선)
 └─ (prefill이 없으면) update_running_batch()   ← §3 retraction
      ├─ check_decode_mem(): 다음 decode step에 요청당 1칸씩 들어가나?
      ├─ 안 들어가면 retract_decode()
      └─ decode batch 반환
```

로그와 대응:

| 로그 | 의미 |
|---|---|
| `Prefill batch, #new-seq: k` | 이번 iteration에 `add_one_req`가 CONTINUE를 k번 반환 |
| `#queue-req: q` | 잘려서 남은 요청 q개 |
| `KV cache pool is full. Retract requests. #retracted_reqs: n` | `retract_decode()`가 n개를 쫓아냄 |

---

## §2. Admission control — 누구를 들여보낼 것인가

### 세 가지 예산

`PrefillAdder`(`managers/schedule_policy.py`)는 iteration마다 세 개의 예산을 들고 시작합니다.

| 예산 | 무엇을 제한 | 기본값 (4090, v0.5.23) | 플래그 |
|---|---|---|---|
| `rem_total_tokens` | **KV 메모리** (지금 + 앞으로 생성할 몫까지) | pool 크기 ≈ 12만 token | `--mem-fraction-static`, `--max-total-tokens` |
| `rem_input_tokens` | 이번 prefill batch의 계산량 | 16384 | `--max-prefill-tokens` |
| `rem_chunk_tokens` | 요청 하나를 한 번에 prefill할 최대 길이 | 4096 | `--chunked-prefill-size` |

그리고 요청 수 상한 `--max-running-requests` (예산이 남아도 이 숫자를 넘으면 `OTHER`로 멈춤).

### 메모리 예산의 계산

```text
rem_total_tokens = (빈 슬롯 + evictable 캐시)
                 − Σ_running 요청 r  min(max_new_r − 이미 생성한 수, 4096) × new_token_ratio
                 − (이번 iteration에 이미 받아들인 요청들의 몫)

새 요청 하나의 비용 = (새로 prefill할 token 수) + min(max_new_tokens, 4096)
→ 비용 < rem_total_tokens 이면 CONTINUE, 아니면 NO_TOKEN
```

핵심은 **이미 실행 중인 요청의 "앞으로 생성할 token"을 얼마로 잡느냐**입니다.

- 전부(×1.0) 잡으면: 절대 메모리가 모자라지 않지만, 대부분의 요청은 `max_new_tokens` 전에 EOS로 끝나므로 **메모리가 남아도는데 새 요청을 못 받음** (throughput↓)
- 조금만 잡으면: 많이 받아들이지만 **실제로 길게 생성하면 모자람** → retraction

SGLang의 답: **`new_token_ratio`라는 학습하는 낙관도**

| 상황 | new_token_ratio |
|---|---|
| 시작 / 서버가 한가해질 때 | `0.7 × schedule_conservativeness` (= 0.7) |
| decode step마다 | 조금씩 감소 → 최소 `0.7 × 0.14 ≈ 0.098` (600 step에 걸쳐) — "점점 더 과감하게" |
| retraction 발생 시 | 실제 생성 비율로 다시 추정해 **높게 되돌림** — "혼나고 나서 조심" |

(`scheduler_components/new_token_ratio_tracker.py`, 기본값은 `environ.py`의 `SGLANG_INIT_NEW_TOKEN_RATIO=0.7`, `SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR=0.14`, `SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS=600`)

> ⚠️ **벤치마크가 스케줄러를 바꾼다.** v0.5.23은 `ignore_eos=True`인 요청에는 ratio 대신 **1.0**을 씁니다 (`_get_running_request_total_token_offset`).
> 길이가 확정된 요청은 일찍 끝날 리가 없기 때문입니다. 그런데 우리 벤치마크는 전부 `ignore_eos=True`였습니다!
> 즉 지금까지의 측정은 "가장 보수적인 admission" 조건이었습니다. 오늘 실습에서 두 경우를 직접 비교합니다.

### 숫자 예제 (칠판에서)

pool = 30,000 token. running 요청 10개, 각각 입력 100 + 생성 200 진행 중, `max_new_tokens = 1000`, ratio = 0.5.

```text
사용 중 KV      = 10 × 300                    =  3,000
running 예약    = 10 × (1000 − 200) × 0.5     =  4,000
rem_total       = 30,000 − 3,000 − 4,000      = 23,000

대기 요청: 입력 100, max_new 1000 → 비용 1,100
→ 23,000 / 1,100 ≈ 20개를 이번 iteration에 받아들일 수 있다 (rem_input_tokens 16384도 확인: 20 × 100 = 2,000 OK)
```

ratio가 0.1까지 떨어졌다면? running 예약은 800 → 약 23개. 반대로 `ignore_eos=True`면 ratio = 1.0 → 예약 8,000 → 약 17개.

---

## §3. Retraction — 들여보낸 걸 후회할 때

낙관적으로 받아들였는데 요청들이 정말로 길게 생성하면, 어느 decode step에서 **요청당 1칸**을 줄 수 없게 됩니다.

```text
update_running_batch()
  if not batch.check_decode_mem():                       # 이번 step에 필요한 칸 > (빈 칸 + evictable)
      retracted, new_ratio, _ = batch.retract_decode()
      new_token_ratio = new_ratio                         # 낙관도 리셋
      retracted 요청들 → waiting_queue로 되돌림
```

**누구를 쫓아내나?** (`_get_decode_retraction_order`)
정렬 키 = (생성한 token 수, −입력 길이). **생성을 적게 한 요청, 그중 입력이 긴 요청부터** 쫓아냅니다.
→ 가장 적게 진행된 요청을 희생 = 버리는 계산이 가장 적음. 입력이 길면 돌려받는 메모리가 큼.

**비용은?**

- 쫓겨난 요청은 나중에 다시 prefill해야 합니다. 그때 radix cache에 자기 KV가 남아 있으면 재계산이 줄지만, 메모리가 부족한 상황이라 대개 이미 evict되었습니다.
- 사용자 입장: 잘 나오던 스트림이 **멈췄다가** 한참 뒤 다시 나옴 → tail ITL과 E2E가 크게 나빠짐.
- 로그 `#new_token_ratio: 0.1043 -> 0.4312`: 스케줄러가 "너무 낙관적이었다"고 인정하고 조심 모드로 전환.

**손잡이:** `--schedule-conservativeness` (기본 1.0). 크게 하면 초기 ratio가 커져 retraction이 줄지만 동시 처리 수도 줄어듭니다.

---

## §4. Chunked prefill — 긴 프롬프트가 모두를 멈출 때

### 문제: head-of-line blocking

16명이 대화하며 token을 받고 있는데(ITL ≈ 10 ms), 누군가 16,000 token짜리 문서를 붙여 넣습니다.
"prefill 우선" 규칙 때문에 다음 iteration은 그 16K prefill입니다 — RTX 4090에서 1초 안팎.

```text
피해자들의 token 간격:   10ms 10ms 10ms ──────── 1,000+ ms ──────── 10ms 10ms …
                                       ↑ 가해자의 prefill
```

### 해결: 잘라서 나눠 먹기

`--chunked-prefill-size C`면 긴 prefill을 C token씩 잘라 **여러 iteration에 걸쳐** 처리합니다.

```text
C = 4096:  [P 4K][decode][P 4K][decode][P 4K][decode][P 4K][decode] …
             ↑ 멈춤 1회당 시간이 1/4로
```

| chunk 크기 | 피해자의 최악 멈춤 | 가해자 TTFT | 전체 효율 |
|---|---|---|---|
| 없음 (`-1`) | 매우 김 | 가장 짧음 | 큰 GEMM 한 번 — 효율 좋음 |
| 큼 (4096, 기본) | 중간 | 약간 늘어남 | 좋음 |
| 작음 (512–1024) | 짧음 | 꽤 늘어남 | 작은 GEMM 여러 번 + 조각마다 앞부분 KV를 다시 읽음 → 효율↓ |

- 조각 사이에서 계산이 끝난 부분은 radix tree에 등록됩니다 (`checkpoint`). 그래서 긴 문서의 앞부분도 다른 요청과 공유 가능.
- **`--enable-mixed-chunk`**: prefill 조각과 decode token들을 **한 forward에 섞습니다**. 피해자들이 멈추지 않고 조각과 함께 1 token씩 진행 → ITL이 "멈춤" 대신 "조금 느려짐"이 됩니다.
- 기본값은 GPU 메모리로 정해집니다 (`arg_groups/memory_hook.py`): 24 GB급은 4096, 80 GB급은 8192. activation 메모리가 chunk 크기에 비례하기 때문입니다.

이 trade-off는 근본적으로 **TTFT(가해자) vs ITL(피해자)** 의 교환입니다. 오늘 실습에서 4가지 설정으로 직접 잽니다.

---

## §5. Overlap scheduling — CPU 시간 숨기기

GPU forward 하나가 5 ms인데, 그 앞뒤로 CPU가 하는 일(다음 batch 고르기, 텐서 준비, 결과 처리, detokenizer로 전송)이 1–2 ms라면?

```text
normal :  CPU▓ GPU████ CPU▓ GPU████ CPU▓ GPU████        GPU가 CPU를 기다리는 구멍
overlap:  CPU▓▓ CPU▓▓ CPU▓▓
               GPU████GPU████GPU████                      batch N 실행 중에 batch N+1 준비
```

`event_loop_overlap` (기본): batch N을 GPU에 던져 놓고(**비동기**), 결과를 기다리지 않은 채 batch N+1을 준비합니다.
batch N의 결과 처리는 N+1을 던진 뒤에 합니다 (`result_queue`).

- 어려운 점: batch N+1을 만들 때 **batch N이 어떤 token을 냈는지 아직 모릅니다.** 그래서 "다음 입력 token"을 GPU 텐서 참조로 미리 연결해 둡니다.
- 작은 모델·작은 batch일수록 GPU 시간이 짧아 CPU overhead 비중이 크고, overlap 효과도 큽니다.
- `--disable-overlap-schedule`로 끌 수 있습니다 (디버깅·비교용).

---

## §6. 손잡이 정리표

| 플래그 | 늘리면 | 줄이면 | 언제 만지나 |
|---|---|---|---|
| `--mem-fraction-static` | KV pool↑ → 동시 처리↑ | OOM 위험↓ | 시작 시 OOM이면 줄임 |
| `--max-total-tokens` | (디버깅용) pool 직접 지정 | | 실험에서 메모리 압박을 일부러 만들 때 |
| `--max-running-requests` | 동시 decode 수 상한↑ | tail latency 안정 | `token usage`가 낮은데 `#queue-req`가 쌓일 때 확인 |
| `--chunked-prefill-size` | 긴 prefill 효율↑, TTFT↓ | 다른 요청의 ITL 안정 | 긴 입력이 섞이는 서비스 |
| `--enable-mixed-chunk` | | | ITL 스파이크가 문제일 때 |
| `--max-prefill-tokens` | prefill batch 크기↑ | | 짧은 요청이 폭주할 때 |
| `--schedule-conservativeness` | retraction↓, 동시 처리↓ | 동시 처리↑, retraction↑ | 로그에 Retract가 자주 찍힐 때 |
| `--schedule-policy` | `fcfs`(기본) / `lpm` / … | | prefix 공유가 많고 메모리가 빠듯할 때 lpm |
| `--disable-overlap-schedule` | | | 디버깅, overlap 효과 측정 |

**진단 순서** (`common/logparse.py` 결과로):

```text
#queue-req가 큼 ─┬─ token usage 높음  → 메모리 부족: pool↑ / 동시성↓ / KV 양자화(Session 5)
                 └─ token usage 낮음  → 설정 상한: max-running-requests, max-prefill-tokens 확인
Retract가 자주 찍힘 → schedule-conservativeness↑ 또는 max_new_tokens를 현실적으로
ITL p99만 나쁨 → 긴 prefill이 끼어드는 중: chunked-prefill-size↓, mixed-chunk
```

## 예측 질문

1. 16명이 decode 중일 때 16K 프롬프트가 들어오면, chunk 없음 / 4096 / 1024에서 피해자의 **최악 멈춤**은 각각 몇 ms일까?
2. pool 30,000 token, 요청 128개, `max_new_tokens=1024`. `ignore_eos=True`일 때 처음에 몇 개를 동시에 받아들일까? `False`라면?
3. retraction이 일어나면 throughput은 늘까 줄까? 어떤 요청의 E2E가 가장 나빠질까?
4. Qwen3-0.6B로 바꾸고 동시성 4로 돌리면 overlap ON/OFF의 ITL 차이는 1.7B보다 클까 작을까?
