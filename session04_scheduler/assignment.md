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
2. `rem_input_tokens`와 `rem_chunk_tokens` 각각으로 몇 개까지 온전히 처리할 수 있나요? 어느 쪽이 먼저 막히나요?
3. 같은 대기 요청들이 `ignore_eos=True`(running 요청들도 전부)였다면 승인 수는?

## Part B. 측정 (GPU)

수업의 16K 결과를 다시 쓰고, 8K 입력에서 `chunk1024`와 `chunk1024mx` 두 설정만 실행하세요. 수업에서 도착 시각이나 출력 길이를 조정했다면 같은 값을 이 실험에도 씁니다.

| 긴 요청의 입력 길이 | chunk 1024 | chunk 1024 + mixed |
|---|---|---|
| 8000 (과제) | | |
| 16000 (수업) | | |

각 칸에 `victim_itl_ms_p99` / `aggressor_ttft_ms_mean` 두 값을 적으세요.

```bash
RESULT_DIR="$PWD/results/s04_homework" AGGRESSOR_IN=8000 bash session04_scheduler/code/interference_sweep.sh
```

실험은 서버 기동 두 번이면 됩니다. 다른 chunk 크기나 interval은 여유가 있을 때 추가하고, 비교를 확장할 때는 강사가 제공한 실행 데이터도 쓸 수 있습니다.

## Part C. 분석

1. **그래프 1**: x = 가해자 TTFT, y = 피해자 p99 ITL. 설정마다 점 하나 (8K, 16K 각각 다른 색). 왼쪽 아래가 좋은 설정입니다.
   SLO(p99 ITL ≤ 100 ms) 선을 그리고, SLO를 만족하는 설정 중 가해자 TTFT가 가장 짧은 것을 고르세요.
2. **그래프 2**: 설정 하나를 골라 `plot_timeline.py`로 ITL 타임라인. 스파이크 하나를 확대해 "이 스파이크의 높이 ≈ chunk 하나의 prefill 시간"인지 확인하세요.
   (chunk 하나의 prefill 시간은 Session 1 과제의 TTFT vs 입력 길이 데이터로 추정 가능)
3. 두 입력 길이에서 mixed를 켰을 때 TTFT와 기존 요청의 긴 멈춤이 각각 얼마나 바뀌었나요? 관찰한 값이 예측과 다르면 타임라인에서 요청이 겹친 구간부터 확인하세요.
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
