# 4. 긴 입력이 기존 스트림에 미치는 영향

## 실습과 과제의 비교 조건

기존 요청 16개가 스트리밍을 시작한 뒤 0.50초에 긴 요청 하나를 보냈습니다. 네 번 모두 긴 요청이 도착했을 때 16개가 스트리밍 중이었습니다. chunk 크기 1,024와 기존 요청 출력 768은 고정했습니다. 실습의 긴 입력 16,000은 다시 쓰고, 과제에서 입력 8,000을 추가했습니다.

| 긴 입력 (토큰) | mixed | 기존 요청 p99 ITL (ms) | 기존 요청 최대 ITL (ms) | 긴 요청 TTFT (ms) |
|---:|---|---:|---:|---:|
| 8,000 | OFF | 4.14 | 62.58 | 70.88 |
| 8,000 | ON | 6.03 | 10.43 | 71.34 |
| 16,000 | OFF | 3.70 | 151.82 | 161.49 |
| 16,000 | ON | 9.90 | 14.02 | 167.26 |

[16K CSV](data/s04_interference.csv) · [8K CSV](data/s04_homework_interference.csv)

로컬에서 [그래프 스크립트](plot_examples.sh)를 실행하면 `s04_slo.png`와 `s04_itl.png`가 나옵니다. 산점도의 수평선 아래에서 입력 길이별로 가장 왼쪽에 있는 점을 고른 뒤, 타임라인에서 드문 공백도 확인합니다.

## 어떤 값으로 결론을 내렸나

1. 스크립트가 출력한 `스트리밍 중 16/16`으로 실제 overlap부터 확인했습니다. 겹치지 않았다면 값을 표에 추가하기 전에 두 조건의 도착 시각을 같은 값으로 바꿔 다시 실행합니다.
2. `victim_itl_ms_p99`에 100 ms 기준을 적용해 통과한 설정의 `aggressor_ttft_ms_mean`을 비교했습니다. 이 실행에서는 네 조건이 모두 통과했고, 두 입력 길이 모두 OFF에서 긴 요청의 TTFT가 조금 낮았습니다. 8K의 약 0.5 ms 차이가 반복되는지는 이 한 번으로 알 수 없습니다.
3. 타임라인의 수직선은 긴 요청 도착입니다. 직후 기존 요청의 token 도착 사이에서 긴 공백을 찾고 `victim_itl_ms_max`를 봤습니다. 16K는 mixed를 켰을 때 최대 공백이 151.82→14.02 ms로 줄었지만 p99는 올랐습니다. 드문 긴 공백이 여러 짧은 공백으로 바뀌면 두 집계가 반대 방향으로 움직일 수 있습니다.
4. 팀에 보고할 때는 “p99 기준에는 두 설정 모두 통과했다. 16K에서 매우 드문 긴 멈춤까지 줄여야 한다면 mixed를 후보로 두고, 같은 부하를 반복해 긴 요청 TTFT에 드는 비용을 함께 재겠다”라고 씁니다.

타임라인 OFF의 요청 0에서 `0.517426 → 0.668917 s` 사이가 비었습니다. 두 청크 도착 시각을 빼면 약 151.49 ms입니다. 긴 요청은 0.504 s에 도착했고 첫 토큰까지 약 161.49 ms가 걸렸습니다. [Session 1](session01.md)의 단독 요청은 2,048 입력에서도 p50 TTFT가 14.57 ms였습니다. 이번 공백을 1,024-token chunk 한 번의 시간으로 설명하기는 어렵습니다. 개별 prefill을 확인하려면 로그에서 긴 요청 도착 뒤 연속 prefill과 decode 재개를 찾거나, extend trace로 chunk별 GPU 시간을 잽니다. 클라이언트 타임라인으로는 각 chunk의 실행 시간을 구할 수 없습니다.

## 과제 A 손계산

빈 슬롯 20,000과 evictable 5,000을 합친 뒤 실행 중인 요청의 예상 생성 몫 `20 × (2048 − 500) × 0.3 = 9288`을 뺍니다. 다음 요청 하나의 비용은 2,348이므로 남은 15,712에서 여섯 개를 온전히 들일 수 있습니다. 입력 한도만 두면 54개, chunk 한도만 두면 13개입니다. 이 경우 메모리 한도가 먼저 걸립니다. 실행 중인 요청을 전부 고정 길이로 계산하면 예약분이 30,960으로 커져 이번에 새 요청을 받을 여유가 없습니다.

## 재현

[실습 2](../../session04_scheduler/lab.md)와 [과제 B](../../session04_scheduler/assignment.md)를 실행했습니다. [OFF 타임라인 JSON](data/s04_timeline_chunk1024.json)과 [ON 타임라인 JSON](data/s04_timeline_chunk1024mx.json)은 긴 요청 도착 전후 기존 스트림 한 개를 기록한 발췌입니다. 두 파일을 `plot_timeline.py`에 넣으면 같은 요청 번호의 공백을 비교할 수 있습니다. 표의 p99와 최대값은 전체 16개 스트림을 집계했습니다.

## 강사 시연 retraction 로그

30,000-token pool에서 EOS 허용 기본 설정은 실행 중 요청이 최대 53개, retraction이 25회였습니다. 보수성 2.0은 51개와 24회였습니다. 고정 길이에서는 28개와 0회였습니다. 로그에서 `KV cache pool is full. Retract requests`를 찾으면 첫 retraction 직후 예약 비율이 `0.0980 → 0.5439`로 올라간 것을 확인할 수 있습니다.

세 조건에서 출력은 모두 1,024 token이었고, fixed의 `sum_new_token`은 3,843인 데 비해 EOS 두 조건은 22,737과 22,631이었습니다. `sum_new_token`은 prefill한 전체 토큰입니다. 다시 prefill한 요청이 있는지는 `Retract requests` 직후의 `Prefill batch`와 함께 읽습니다. 이번 실행에서 보수성 2.0은 retraction을 거의 줄이지 못했습니다. 초기 예약만 보고 마지막까지 그 상태가 유지된다고 가정하면 놓치는 부분입니다.

[재할당 로그 발췌](data/s04_retraction_excerpt.txt) · [스케줄러 집계](data/s04_retraction_log_summary.txt) · [부하 결과 CSV](data/s04_retract.csv)

## 선택 실습에서 overlap scheduling 읽기

Qwen3-0.6B에서 동시성 16을 비교했습니다. overlap을 켜면 `output_tok_per_s`는 5,971→8,512, `itl_ms_p95`는 3.00→2.18 ms였습니다. `ttft_ms_p95`는 20.04→25.52 ms로 늘었습니다. 먼저 `num_errors=0`과 같은 요청 수·길이를 확인하고, 이 세 열을 나란히 봅니다. 처리량과 출력 간격을 우선하면 ON을 후보로 두고, 첫 토큰이 중요한 서비스라면 TTFT 목표를 적용해 다시 비교합니다. 이 비교는 0.6B 모델과 한 번의 실행에 해당합니다. [overlap CSV](data/s04_overlap.csv)

## 과제 D 코드 리딩

1. `_select_prefill_admission`에서 `_check_prefill_budget`이 요청에 필요한 슬롯을 승인하지 않으면 `NO_TOKEN`으로 돌아갑니다. chunk 예산만 부족한 분기는 `OTHER`일 수 있으므로 두 반환 경로를 구분합니다. `get_next_batch_to_run`은 남은 `chunked_req`를 바로 decode에 합치지 않습니다. 새 prefill 계획이 없거나 `_should_defer_prefill`이 기존 decode를 보호하는 차례라면, 이미 진행 중인 요청으로 decode batch를 만듭니다. [승인 분기](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/managers/schedule_policy.py#L1477-L1510) · [batch 선택](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/managers/scheduler.py#L3642-L3725)
2. `_get_running_request_total_token_offset`에서 `ignore_eos`는 비율 1.0을 사용합니다. 일찍 멈출 EOS가 없으므로 남은 생성 토큰을 `new_token_ratio`만큼 낮춰 잡지 않습니다. [예약 코드](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/managers/schedule_policy.py#L727-L736)
3. 기본 retraction 키는 `(생성 길이, -원래 입력 길이)`입니다. 내림차순으로 정리한 뒤 끝에서 빼므로 이미 적게 생성한 요청을 먼저 빼고, 생성 길이가 같으면 입력이 긴 요청이 먼저입니다. 이미 많이 생성한 요청을 다시 처리하면 앞선 작업을 복구하는 비용과 사용자가 겪는 출력 중단이 커질 수 있습니다. [retraction 정렬](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/managers/schedule_batch.py#L3274-L3308)
