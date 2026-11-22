# Session 4 과제 — Long-Prompt Interference 리포트

> 마감: Session 5 시작 24시간 전 PR · GPU 예산 **$1.2 이내**
> 제출: `submissions/session04/<github-id>/`

## 시나리오

여러분의 서비스에 "긴 문서 요약" 기능이 추가됩니다. 사용자 일부가 8K–16K token 문서를 붙여 넣습니다.
기존 채팅 사용자들의 **p99 ITL ≤ 100 ms** 를 지키면서, 문서 요약 사용자의 TTFT를 최대한 줄이는 설정을 찾아야 합니다.

## Part A. 손 계산 (GPU 불필요)

이론 §2의 공식으로 아래 상황의 이번 iteration 승인 요청 수를 계산하세요. 과정을 보이세요.

- pool 40,000 token, 빈 슬롯 20,000, evictable 캐시 5,000
- running 요청 20개, 각각 `max_new_tokens = 2048`, 이미 500 token 생성, `new_token_ratio = 0.3`
- 대기 요청: 입력 300 token, `max_new_tokens = 2048` 짜리가 50개 (캐시 hit 없음)
- `max_prefill_tokens = 16384`, `chunked_prefill_size = 4096`

1. 메모리 예산 기준 승인 수는?
2. `rem_input_tokens` 기준 승인 수는? 어느 쪽이 먼저 막나요?
3. 같은 대기 요청들이 `ignore_eos=True`(running 요청들도 전부)였다면 승인 수는?

## Part B. 측정 (GPU)

수업의 `interference.py`를 이용해 **chunk 크기 × 가해자 길이** 표를 채웁니다. 피해자는 16명 고정.

| 가해자 입력 \ 설정 | nochunk | 4096 | 1024 | 4096 + mixed | 1024 + mixed | 1024 + interval 1 |
|---|---|---|---|---|---|---|
| 8000 | | | | | | |
| 16000 | | | | | | |

각 칸에 `victim_itl_ms_p99` / `aggressor_ttft_ms_mean` 두 값을 적으세요.

```bash
# 예: chunk 4096 + mixed, 가해자 8000 / 16000
bash common/launch.sh c4096mx --enable-mixed-chunk
python session04_scheduler/code/interference.py --aggressor-in 8000  --tag c4096mx_a8k  --out results/s04_hw.csv
python session04_scheduler/code/interference.py --aggressor-in 16000 --tag c4096mx_a16k --out results/s04_hw.csv
bash common/stop.sh
```

> 예산 팁: 서버 하나 띄울 때마다 두 길이를 모두 돌리세요 (서버 시작이 가장 비싼 부분입니다). 총 6번 launch.
> 수업에서 이미 돌린 16000 결과는 재사용해도 됩니다.

## Part C. 분석

1. **그래프 1**: x = 가해자 TTFT, y = 피해자 p99 ITL. 설정마다 점 하나 (8K, 16K 각각 다른 색). 왼쪽 아래가 좋은 설정입니다.
   SLO(p99 ITL ≤ 100 ms) 선을 그리고, SLO를 만족하는 설정 중 가해자 TTFT가 가장 짧은 것을 고르세요.
2. **그래프 2**: 설정 하나를 골라 `plot_timeline.py`로 ITL 타임라인. 스파이크 하나를 확대해 "이 스파이크의 높이 ≈ chunk 하나의 prefill 시간"인지 확인하세요.
   (chunk 하나의 prefill 시간은 Session 1 과제의 TTFT vs 입력 길이 데이터로 추정 가능)
3. 16K 가해자에서 `1024 + mixed`와 `1024 + interval 1`의 **가해자 TTFT가 nochunk보다 얼마나 늘었는지**, 그 이유를 두 가지 대세요.
   (힌트: ① 조각 사이에 decode가 끼어든다 ② 뒤쪽 조각일수록 attention이 앞 조각들의 KV를 모두 읽는다)
   그리고 chunk 크기만 바꾼 열(nochunk / 4096 / 1024)에서 피해자 p99 ITL이 거의 안 변했다면, 이론 §4.3과 연결해 설명하세요.
4. 팀장에게 추천할 설정과 근거 3문장.

## Part D. 코드 리딩 (GPU 불필요)

1. `managers/schedule_policy.py`의 `PrefillAdder._select_prefill_admission`에서 `NO_TOKEN`을 반환하는 경로를 찾아, 어떤 조건에서 그런지 코드 인용과 함께 설명하세요.
   그리고 `managers/scheduler.py`의 `get_next_batch_to_run` / `_should_defer_prefill`을 읽고, `chunked_req`가 남아 있을 때 decode가 언제 실행되는지 설명하세요.
2. `_get_running_request_total_token_offset`을 읽고, `ignore_eos`가 왜 다르게 취급되는지 자기 말로 설명하세요.
3. `managers/schedule_batch.py`의 `_get_decode_retraction_order`가 정렬하는 키를 쓰고, "생성을 많이 한 요청을 쫓아내면 안 되는 이유"를 설명하세요.

## 채점 기준 (10점)

| 항목 | 점수 |
|---|---:|
| Part A 계산 | 2 |
| Part B 측정 재현성 | 2 |
| Part C 그래프 + SLO 기반 선택 | 3 |
| Part D 코드 리딩 | 2 |
| GPU 기록 | 1 |

## 다음 회차 발표

추천 설정이 서로 다른 2–3명. "같은 데이터를 보고 왜 다른 설정을 골랐는가"를 토론합니다.
