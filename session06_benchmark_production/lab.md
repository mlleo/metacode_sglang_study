# Session 6 실습 — 반복·프로파일·지표·비용
> **2시간 수업 기준:** 필수 1·2·4·6 / 선택 3, 5

> 실습 55분 (2시간 수업 기준) / GPU 예산 약 $0.8 / 결과물: `results/s06_*.csv`, `profiles/*`
> 남은 시간은 최종 프로젝트 첫 실험(작은 규모 리허설)에 씁니다.

| # | 실습 | 시간 |
|---|---|---|
| 0 | 준비 | 수업 시작 시 (발표 중) |
| 1 | 같은 실험 3번: 노이즈는 얼마나 큰가 ⭐ | 12분 |
| 2 | torch profiler로 decode step 들여다보기 ⭐ | 18분 |
| 3 | Prometheus metrics로 서버 상태 보기 | 8분 |
| 4 | $/1M tokens 계산 | 7분 |
| 5 | 최종 프로젝트 리허설 (작은 설정으로 스크립트가 끝까지 도는지) | 15분 |
| 6 | 정리 & Terminate | 5분 |

## 0. 준비

```bash
cd /workspace && git clone https://github.com/<your-id>/metacode_sglang_study.git && cd metacode_sglang_study
bash common/setup_pod.sh
```

---

## 1. 같은 실험 3번 ⭐ (12분)

```bash
bash common/launch.sh rep
for i in 1 2 3; do
  python common/loadgen.py --num 64 --input-len 512 --output-len 256 --concurrency 16 \
    --flush --seed $i --tag c16 --out results/s06_rep.csv
  python common/loadgen.py --num 128 --input-len 512 --output-len 256 --concurrency 64 \
    --flush --seed $i --tag c64 --out results/s06_rep.csv
done
python session06_benchmark_production/code/repeat_stats.py results/s06_rep.csv --group tag
```

- throughput의 CI 폭은 평균의 몇 %인가요? p99 TTFT는?
- 지난 과제들에서 "A가 B보다 5% 빨랐다"고 쓴 적이 있다면, 그 주장은 이 노이즈 수준에서 유효한가요?
- 옆 사람(다른 Pod, 같은 4090)과 c16 평균 throughput을 비교해 보세요. 호스트 간 차이는 반복 내 차이보다 큰가요?

> 서버는 다음 실습에서도 쓰지 않으므로 끕니다: `bash common/stop.sh`

---

## 2. torch profiler로 decode step 들여다보기 ⭐ (18분)

**예측 먼저:** 동시성 1과 32에서 GPU busy %는? CUDA graph를 끄면?

```bash
bash session06_benchmark_production/code/profile_decode.sh c1  1
bash session06_benchmark_production/code/profile_decode.sh c32 32
bash session06_benchmark_production/code/profile_decode.sh c1_nograph 1 --cuda-graph-backend-decode disabled

python session06_benchmark_production/code/trace_summary.py profiles/c1 profiles/c32 profiles/c1_nograph
```

| tag | GPU busy % | gemm % | attention % | 상위 커널 1위 |
|---|---:|---:|---:|---|
| c1 | | | | |
| c32 | | | | |
| c1_nograph | | | | |

그다음 trace 파일 하나를 로컬로 내려받아 <https://ui.perfetto.dev>에 끌어다 놓으세요 (`profiles/c1_nograph/*.trace.json.gz` 추천).

- GPU 줄에서 커널 사이의 **빈칸**을 찾으세요. c1_nograph에서 더 넓은가요?
- CPU 줄에서 `cudaLaunchKernel`이 촘촘히 찍힌 구간과, `cudaGraphLaunch` 한 번으로 끝나는 구간(c1)을 비교하세요.
- c32에서 attention 비중이 c1보다 커졌나요? (Session 5의 `KV%`와 같은 이야기)

> trace 파일은 수십 MB일 수 있습니다. 제출할 때는 올리지 말고 `trace_summary.py` 출력만 붙이세요.

---

## 3. Prometheus metrics (8분)

```bash
bash common/launch.sh metrics --enable-metrics
python session06_benchmark_production/code/metrics_probe.py --watch 2 &
python common/loadgen.py --num 96 --input-len 1024 --output-len 256 --concurrency 32 --tag m --out results/s06_metrics.csv
kill %1
python session06_benchmark_production/code/metrics_probe.py
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
python session06_benchmark_production/code/cost_calc.py results/s02_base.csv --usd-per-hour 0.59 \
  --throughput output_throughput --load max_concurrency --slo p95_ttft_ms=500 --slo p95_itl_ms=50
# 활용률 30% 가정
python session06_benchmark_production/code/cost_calc.py results/s02_base.csv --usd-per-hour 0.59 \
  --throughput output_throughput --load max_concurrency --slo p95_ttft_ms=500 --utilization 0.3
```

(지난 결과가 Pod에 없다면 로컬에서 실행하면 됩니다 — GPU 불필요)

- 가장 싼 설정과 SLO를 지키는 가장 싼 설정은 각각 몇 $/1M인가요?
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
