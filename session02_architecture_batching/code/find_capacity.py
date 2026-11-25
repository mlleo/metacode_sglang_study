"""측정한 지점에서 SLO를 만족하는 최대 부하와 최대 처리량을 찾는다.

    python session02_architecture_batching/code/find_capacity.py results/s02_base.csv \
        --x max_concurrency --slo p95_ttft_ms=500 --slo p95_itl_ms=50

출력: 각 행의 SLO 만족 여부와 두 지점. 최대 부하와 최대 처리량은 서로 다를 수 있다.
"""

from __future__ import annotations

import argparse
import csv
import math


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--x", default="max_concurrency", help="부하를 나타내는 열 (max_concurrency 또는 request_rate)")
    p.add_argument("--throughput", default="output_throughput")
    p.add_argument("--slo", action="append", default=[], help="열=상한 (예: p95_ttft_ms=500). 여러 번 지정 가능")
    args = p.parse_args()

    slos = [(k, float(v)) for k, v in (s.split("=") for s in args.slo)]
    with open(args.csv) as f:
        rows = list(csv.DictReader(f))

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("inf")

    rows.sort(key=lambda r: num(r[args.x]))
    passing = []
    header = f"{args.x:>16} {args.throughput:>18} " + " ".join(f"{k:>14}" for k, _ in slos) + "   SLO"
    print(header)
    for r in rows:
        load, throughput = num(r[args.x]), num(r[args.throughput])
        ok = math.isfinite(load) and math.isfinite(throughput) and throughput > 0
        ok = ok and all(num(r[k]) <= lim for k, lim in slos)
        print(f"{r[args.x]:>16} {num(r[args.throughput]):>18.1f} "
              + " ".join(f"{num(r[k]):>14.1f}" for k, _ in slos) + ("   ✅" if ok else "   ❌"))
        if ok:
            passing.append(r)
    print()
    if passing:
        capacity = passing[-1]
        best = max(passing, key=lambda r: num(r[args.throughput]))
        print(f"SLO 만족 최대 측정 부하: {args.x}={capacity[args.x]} → {num(capacity[args.throughput]):.1f} {args.throughput}")
        print(f"SLO 만족 최대 throughput: {args.x}={best[args.x]} → {num(best[args.throughput]):.1f} {args.throughput}")
    else:
        print("SLO를 만족하는 지점이 없습니다. SLO를 완화하거나 더 낮은 부하를 측정하세요.")


if __name__ == "__main__":
    main()
