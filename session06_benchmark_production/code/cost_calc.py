"""throughput 측정 결과 + GPU 시간당 가격 → 1M output token당 비용.

    python session06_benchmark_production/code/cost_calc.py results/s06_cost.csv --usd-per-hour 0.59 \
        --throughput output_tok_per_s --slo ttft_ms_p95=500 --slo itl_ms_p95=50

주의
  - 이 값은 "GPU를 100% 그 부하로 계속 돌릴 때"의 하한입니다. 실제 트래픽은 출렁이므로 평균 활용률로 나눠야 합니다.
  - SLO를 지키는 행만 '팔 수 있는' throughput입니다 (goodput 관점).
"""

from __future__ import annotations

import argparse
import csv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--usd-per-hour", type=float, required=True, help="Deploy 화면에 표시된 Pod 시간당 가격")
    p.add_argument("--throughput", default="output_tok_per_s", help="열 이름 (bench_to_csv 결과면 output_throughput)")
    p.add_argument("--label", default="tag")
    p.add_argument("--load", default="concurrency", help="부하를 나타내는 열")
    p.add_argument("--slo", action="append", default=[])
    p.add_argument("--utilization", type=float, default=1.0, help="평균 GPU 활용률 (0–1). 0.3이면 비용 3.3배")
    args = p.parse_args()

    slos = [(k, float(v)) for k, v in (s.split("=") for s in args.slo)]
    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    print(f"{args.label:<14}{args.load:>12}{'tok/s':>10}{'$/1M tok':>11}  SLO")
    best = None
    for r in rows:
        tps = float(r[args.throughput])
        usd_per_m = args.usd_per_hour / (tps * 3600 * args.utilization) * 1e6 if tps > 0 else float("inf")
        ok = all(float(r[k]) <= lim for k, lim in slos)
        print(f"{r.get(args.label, ''):<14}{r.get(args.load, ''):>12}{tps:>10.0f}{usd_per_m:>11.4f}  {'✅' if ok else '❌'}")
        if ok and (best is None or usd_per_m < best[0]):
            best = (usd_per_m, r)
    if best:
        print(f"\nSLO 만족 최저 비용: {best[0]:.4f} $/1M output tokens  ({args.label}={best[1].get(args.label)}, "
              f"{args.load}={best[1].get(args.load)})")


if __name__ == "__main__":
    main()
