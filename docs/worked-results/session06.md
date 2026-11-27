# 6. 반복·trace·비용과 최종 프로젝트

## 반복에서 읽을 것

동시성마다 네 wave 분량을 세 번씩 요청했습니다. p99 TTFT는 먼저 각 실행의 요청에서 구하고 그 값을 세 번 비교했습니다. 여러 실행의 요청을 한데 섞으면 실행 사이에 얼마나 달라졌는지 보기 어렵습니다.

| 동시성 | output tok/s 평균 ± 반폭 | p99 TTFT 평균 ± 반폭 (ms) |
|---:|---:|---:|
| 16 | 5,299.6 ± 16.3 | 56.9 ± 5.4 |
| 64 | 12,589.1 ± 117.1 | 178.9 ± 17.5 |

동시성 16에서 처리량 반폭은 평균의 0.3%, p99 TTFT는 9.4%였습니다. 지난 과제에서 몇 %의 tail latency 개선을 보고했다면 같은 입력으로 반복한 뒤 차이의 구간부터 볼 만합니다. [실행별 CSV](data/s06_rep.csv) · [repeat_stats 출력](data/s06_repeat_stats.txt)

## trace에서 근거 찾기

| trace | decode batch | CUDA graph | GPU busy | GEMM 비중 | attention 비중 |
|---|---:|---|---:|---:|---:|
| [c1](data/s06_c1-DECODE.trace.json.gz) | 1 | ON | 62.9% | 71.6% | 12.8% |
| [c1_nograph](data/s06_c1_nograph-DECODE.trace.json.gz) | 1 | OFF | 7.4% | 71.1% | 11.8% |
| [c32](data/s06_c32-DECODE.trace.json.gz) | 32 | ON | 43.2% | 58.4% | 24.2% |

1. 표의 `.gz` 파일을 내려받아 [Perfetto](https://ui.perfetto.dev)에 끌어다 놓습니다. `step[DECODE bs=1]`을 찾고 Scheduler가 있는 CPU 트랙에서 step 하나를 선택한 뒤 같은 시간의 GPU stream을 펼칩니다. 다른 batch를 볼 때는 `step[DECODE bs=32]`를 찾습니다.
2. batch 1 두 파일에서 `cudaGraphLaunch`와 `cudaLaunchKernel`을 검색합니다. 발췌에서 ON은 graph launch 1개와 kernel launch 11개, OFF는 kernel launch 75개를 보여 줍니다. 전체 10-step 캡처에서는 ON에 graph launch 10개와 kernel launch 113개가, OFF에 graph launch 0개와 kernel launch 753개가 있었습니다. CPU CUDA 호출이 촘촘한 구간과 GPU 커널이 비는 구간을 시간축에서 나란히 봅니다.
3. batch 1에서 GPU에 기록된 이벤트 시간의 합은 21.2 ms와 21.4 ms로 비슷했지만 관측 범위는 33.5 ms와 290.6 ms였습니다. GPU 이벤트 밖의 시간이 커졌다는 증거입니다. 짧은 프로파일에서 나온 7.4%를 그 서버의 평소 GPU 사용률로 해석할 수는 없습니다.
4. graph ON의 batch 1과 32를 비교하면 attention은 합산 이벤트 시간의 12.8%에서 24.2%로 늘었습니다. 실제 커널 이름도 함께 확인합니다. 이 표의 분류는 커널 이름으로 추정했으며, 합산 비중은 겹친 시각의 이벤트도 각각 셉니다. GPU busy는 겹친 시각을 한 번만 셉니다.

표는 전체 10-step 캡처의 요약입니다. Perfetto 파일은 실제 다섯 번째 decode step 한 개를 읽기 쉽게 발췌했습니다. GPU 이벤트·CUDA 호출·step 표시가 남아 있어 CPU–GPU 시간축을 비교할 수 있습니다. Graph ON 파일을 열면 겹친 이벤트가 별도 줄에 표시될 수 있습니다. 줄의 위치보다 이벤트의 시작과 종료 시각을 비교하세요. Python 호출 스택까지 보고 싶다면 [실습 2](../../session06_benchmark_production/lab.md)를 실행해 전체 trace를 사용합니다. [c1 요약](data/s06_trace_c1.txt) · [c1_nograph 요약](data/s06_trace_c1_nograph.txt) · [c32 요약](data/s06_trace_c32.txt)

Prometheus에서는 `time_to_first_token_seconds_bucket`과 `inter_token_latency_seconds_bucket`이 누적됩니다. 비교할 부하를 보내기 전과 후에 같은 bucket을 읽어 차이를 냅니다. client CSV와 다른 값을 보았을 때 서버·클라이언트의 시작과 끝, bucket 폭, 포함된 요청을 먼저 확인합니다. 예를 들어 이 리허설에서는 여러 실행 뒤에 읽은 서버 histogram에 205개가 잡혔고, 프로젝트 CSV에는 실행별로 측정한 32개씩만 들어갑니다.

## 비용에서 지킬 기준

[512-token 벤치마크 CSV](data/s02_base.csv)에서 `p95_ttft_ms ≤ 500`과 `p95_itl_ms ≤ 50`을 적용한 후 `output_throughput`으로 비용을 계산했습니다. 예시 단가 $0.59/h에서 동시성 128의 16,029.5 output tok/s를 대입하면 `0.59 × 1,000,000 / (3600 × 16029.5) ≈ $0.0102`입니다. 해당 부하로 도는 시간이 30%면 $0.0341입니다. 모든 행에 같은 가동 비율을 적용했으므로 이 표의 순위는 그대로였습니다. 실제 보고서에는 사용한 Pod의 가격과 디스크 비용을 따로 적습니다.

## 최종 프로젝트 예시

**질문.** 같은 GPU에서 입력만 512→2,048로 늘리면 동시성 8의 p95 TTFT와 출력 처리량은 얼마나 달라질까?

**가설.** p95 TTFT는 두 배 이상, 출력 처리량은 10% 이상 떨어질 것으로 예상했습니다. 긴 입력의 prefill이 첫 토큰까지 걸리는 시간에 직접 들어가고, 기존 decode와 GPU 시간을 나눠 쓰기 때문입니다.

**방법.** Qwen3-1.7B BF16, 출력 256, 동시성 8, 요청 32개를 고정했습니다. 각 seed에서 짧은 입력과 긴 입력을 같은 순서로 짝지어 실행했고, 전부 성공했는지 확인했습니다. 작은 리허설에서는 요청 16개로 바꿔 두 설정의 파일이 만들어지는지부터 확인했습니다. [리허설 CSV](data/s06_project_pilot.csv)

| 지표 | 입력 512 평균 | 입력 2,048 평균 | 짝별 차이 평균 (긴 입력 − 짧은 입력) | 차이의 95% CI |
|---|---:|---:|---:|---:|
| p95 TTFT (ms) | 31.49 | 95.56 | +64.07 | [+61.17, +66.98] |
| output tok/s | 2,975.37 | 2,338.48 | −636.89 | [−645.15, −628.63] |

아래 명령으로 p95 TTFT와 출력 처리량 그래프를 각각 그립니다. 한 선이 같은 seed의 두 실행을 잇습니다. 표의 평균을 보기 전에 세 선이 같은 방향인지, 유난히 큰 차이가 한 seed에 몰렸는지 봅니다.

**해석.** 같은 seed의 두 실행에서 `ttft_ms_p95`의 차이를 구하면 +62.91, +64.06, +65.25 ms입니다. 이 세 차이로 평균의 t 구간을 계산했습니다. 지연은 약 3배, 처리량은 약 21% 감소했습니다. Session 2의 `#new-token`과 큐 집계를 함께 보면 긴 입력이 prefill 부하를 늘렸다는 설명과 맞습니다. 이 실험만으로 원인별 시간을 분해할 수는 없습니다. 다음 실험에서는 실행 순서를 교대로 바꾸고 동일 동시성에서 extend trace를 찍어 prefill 시간을 직접 비교합니다.

[본 측정 CSV](data/s06_project.csv) · [실행 스크립트](project/run.sh) · [짝별 차이 계산](project/analyze.py)

```bash
# GPU Pod의 저장소 루트
bash common/setup_pod.sh
bash docs/worked-results/project/run.sh results/my_project.csv

# CSV를 받은 로컬 저장소 (그래프를 그리려면 matplotlib 필요)
python3 docs/worked-results/project/analyze.py results/my_project.csv --figures figures/my_project
```
