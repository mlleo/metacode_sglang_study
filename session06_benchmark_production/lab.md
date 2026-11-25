# Session 6 실습 — 반복·프로파일·지표·비용
> **2시간 수업 기준:** 필수 1·2·4·6 / 선택 3, 5

> 실습 55분 (2시간 수업 기준) / GPU 예산 약 $0.8 / 결과물: `results/s06_*.csv`, `results/s06_trace_*.txt`, Perfetto 스크린샷
> 남은 시간은 최종 프로젝트 첫 실험(작은 규모 리허설)에 씁니다.

| # | 실습 | 시간 |
|---|---|---|
| 0 | 준비 | 실습 직전 |
| 1 | 같은 실험 3번: 노이즈는 얼마나 큰가 ⭐ | 12분 |
| 2 | torch profiler로 decode step 들여다보기 ⭐ | 18분 |
| 3 | Prometheus metrics로 서버 상태 보기 | 8분 |
| 4 | $/1M tokens 계산 | 7분 |
| 5 | 최종 프로젝트 리허설 (작은 설정으로 스크립트가 끝까지 도는지) | 15분 |
| 6 | 정리 & Terminate | 5분 |

## 0. 준비

```bash
cd /workspace && git clone --branch main https://github.com/mlleo/metacode_sglang_study.git && cd metacode_sglang_study
bash common/setup_pod.sh
```

---

## 1. 같은 실험 3번 ⭐ (12분)

```bash
bash common/launch.sh rep
for i in 1 2 3; do
  python3 common/loadgen.py --num 64 --input-len 512 --output-len 256 --concurrency 16 \
    --flush --seed $i --tag c16 --out results/s06_rep.csv
  python3 common/loadgen.py --num 256 --input-len 512 --output-len 256 --concurrency 64 \
    --flush --seed $i --tag c64 --out results/s06_rep.csv
done
python3 session06_benchmark_production/code/repeat_stats.py results/s06_rep.csv --group tag
```

- throughput의 CI 폭은 평균의 몇 %인가요? p99 TTFT는?
- 이번 반복은 각 설정의 변동을 보는 실습입니다. 설정의 효과를 검정하려면 같은 조건끼리 짝지어 차이를 구하거나, 독립 두 표본의 차이 구간을 계산합니다.
- 옆 사람(다른 Pod, 같은 4090)과 c16 평균 throughput을 비교해 보세요. 호스트 간 차이는 반복 내 차이보다 큰가요?

프로파일 실습은 서버를 새로 시작합니다. `bash common/stop.sh`로 반복 측정 서버부터 끄세요.

---

## 2. torch profiler로 decode step 들여다보기 ⭐ (18분)

**예측 먼저:** 동시성 1과 32에서 GPU busy %는? CUDA graph를 끄면?

```bash
bash session06_benchmark_production/code/profile_decode.sh c1 1
python3 session06_benchmark_production/code/trace_summary.py profiles/c1 --stage decode | tee results/s06_trace_c1.txt
```

graph OFF와 동시성 32는 강사가 같은 환경에서 재어 둔 trace를 사용합니다. 직접 재려면 아래 명령을 사용하세요.

```bash
bash session06_benchmark_production/code/profile_decode.sh c1_nograph 1 --cuda-graph-backend-decode disabled
bash session06_benchmark_production/code/profile_decode.sh c32 32
```

| tag | GPU busy % | gemm % | attention % | 상위 커널 1위 |
|---|---:|---:|---:|---|
| c1 | | | | |
| c32 | | | | |
| c1_nograph | | | | |

그다음 이름에 `-DECODE.trace.json.gz`가 붙은 파일 하나를 로컬로 내려받아 <https://ui.perfetto.dev>에 끌어다 놓으세요. 스크립트가 보여 준 `run.*` 폴더에 있습니다. 비교 자료가 있다면 graph OFF trace부터 읽습니다.

- GPU 줄에서 커널 사이의 **빈칸**을 찾으세요. c1_nograph에서 더 넓은가요?
- CPU 줄에서 `cudaLaunchKernel`이 촘촘히 찍힌 구간과, `cudaGraphLaunch` 한 번으로 끝나는 구간(c1)을 비교하세요.
- c32에서 attention 비중이 c1보다 커졌나요? (Session 5의 `KV%`와 같은 이야기)

> trace 파일은 수십 MB일 수 있습니다. 제출에는 요약과 Perfetto 스크린샷만 넣습니다. 스크린샷을 나중에 찍을 계획이면 Pod를 종료하기 전에 decode trace 하나를 따로 내려받으세요.

---

## 3. Prometheus metrics (8분)

```bash
bash common/launch.sh metrics --enable-metrics
python3 session06_benchmark_production/code/metrics_probe.py --watch 2 &
python3 common/loadgen.py --num 96 --input-len 1024 --output-len 256 --concurrency 32 --tag m --out results/s06_metrics.csv
kill %1
python3 session06_benchmark_production/code/metrics_probe.py
curl -s localhost:30000/metrics | grep -E "^sglang:(num_running_reqs|token_usage|cache_hit_rate)" | head
bash common/stop.sh
```

- 부하 중 `num_running_reqs`, `token_usage`가 로그(`Decode batch`)의 값과 같은가요?
- `/metrics`의 TTFT p50과 loadgen이 클라이언트에서 잰 `ttft_ms_p50`(CSV)을 비교하세요. 차이는 어디서 올까요?

---

## 4. $/1M tokens (7분)

RunPod 콘솔에서 **지금 쓰고 있는 Pod의 시간당 가격**을 확인하고, 지난 회차 결과로 계산합니다.

```bash
# Session 2의 동시성 sweep (bench_to_csv 결과)
python3 session06_benchmark_production/code/cost_calc.py results/s02_base.csv --usd-per-hour 0.59 \
  --throughput output_throughput --load max_concurrency --slo p95_ttft_ms=500 --slo p95_itl_ms=50
# 위 처리량으로 동작하는 시간 비율 30% 가정
python3 session06_benchmark_production/code/cost_calc.py results/s02_base.csv --usd-per-hour 0.59 \
  --throughput output_throughput --load max_concurrency --slo p95_ttft_ms=500 --slo p95_itl_ms=50 --utilization 0.3
```

(지난 결과가 Pod에 없다면 로컬에서 실행하면 됩니다 — GPU 불필요)

- 표에서 전체 최저 비용과 두 지연 기준을 모두 만족한 최저 비용을 비교하세요. 처리하는 시간 비율만 30%로 바꾸면 선택도 달라지나요?
- Session 5의 FP8 결과(`s05_quant.csv`)로 같은 계산을 하면 얼마나 싸지나요?

---

## 5. 최종 프로젝트 리허설 (15분)

[final_project.md](final_project.md)에서 주제를 고르고, **가장 작은 설정**(요청 수 ¼, sweep 점 2개)으로 스크립트가 끝까지 도는지만 확인합니다.
본 실험은 집에서 — 예산 $4.0 이내로.

## 6. 정리 & Terminate

```bash
bash common/stop.sh
tar czf s06_$(date +%m%d).tgz results logs
```

다운로드 → **Terminate** → 사용 기록. 남은 크레딧을 확인하고 최종 프로젝트 예산을 정하세요.
