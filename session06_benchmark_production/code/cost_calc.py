"""throughput 측정 결과 + GPU 시간당 가격 → 1M output token당 비용.

    python session06_benchmark_production/code/cost_calc.py results/s06_cost.csv --usd-per-hour 0.59 \
        --throughput output_tok_per_s --slo ttft_ms_p95=500 --slo itl_ms_p95=50

주의
  - 측정한 부하가 유지되는 시간의 비율로 비용을 보정합니다.
  - SLO 조건을 주면 해당 집계 지표를 만족한 행 중 가장 싼 설정을 표시합니다.
"""

from __future__ import annotations

import argparse
import csv
import math


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--usd-per-hour", type=float, required=True, help="Deploy 화면에 표시된 Pod 시간당 가격")
    p.add_argument("--throughput", default="output_tok_per_s", help="열 이름 (bench_to_csv 결과면 output_throughput)")
    p.add_argument("--label", default="tag")
    p.add_argument("--load", default="concurrency", help="부하를 나타내는 열")
    p.add_argument("--slo", action="append", default=[])
    p.add_argument("--utilization", type=float, default=1.0, help="측정한 부하로 동작하는 시간 비율 (0 초과 1 이하)")
    args = p.parse_args()
    if not math.isfinite(args.usd_per_hour) or args.usd_per_hour < 0:
        p.error("--usd-per-hour는 0 이상의 유한한 값이어야 합니다.")
    if not 0 < args.utilization <= 1:
        p.error("--utilization은 0보다 크고 1 이하여야 합니다.")

    slos = [(k, float(v)) for k, v in (s.split("=") for s in args.slo)]
    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    print(f"{args.label:<14}{args.load:>12}{'tok/s':>10}{'$/1M tok':>11}  SLO")
    best = None
    for r in rows:
        tps = float(r[args.throughput])
        valid = math.isfinite(tps) and tps > 0
        usd_per_m = args.usd_per_hour / (tps * 3600 * args.utilization) * 1e6 if valid else float("inf")
        ok = valid and all(math.isfinite(value) and value <= limit for value, limit in (
            (float(r[k]), limit) for k, limit in slos
        ))
        verdict = ("통과" if ok else "실패") if slos else "—"
        print(f"{r.get(args.label, ''):<14}{r.get(args.load, ''):>12}{tps:>10.0f}{usd_per_m:>11.4f}  {verdict}")
        if ok and (best is None or usd_per_m < best[0]):
            best = (usd_per_m, r)
    if best:
        label = "SLO 만족 최저 비용" if slos else "최저 비용"
        print(f"\n{label}: {best[0]:.4f} $/1M output tokens  ({args.label}={best[1].get(args.label)}, "
              f"{args.load}={best[1].get(args.load)})")
    elif slos:
        print("\n지정한 SLO를 만족하는 측정 지점이 없습니다.")


if __name__ == "__main__":
    main()
