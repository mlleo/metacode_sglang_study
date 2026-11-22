"""SLO를 만족하는 최대 부하(capacity)와 goodput을 찾는다.

    python session02_architecture_batching/code/find_capacity.py results/s02_base.csv \
        --x max_concurrency --slo p95_ttft_ms=500 --slo p95_itl_ms=50

출력: 각 행의 SLO 만족 여부, 만족하는 행 중 throughput이 가장 높은 지점(= sweet spot 후보).
"""

from __future__ import annotations

import argparse
import csv


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
    best = None
    header = f"{args.x:>16} {args.throughput:>18} " + " ".join(f"{k:>14}" for k, _ in slos) + "   SLO"
    print(header)
    for r in rows:
        ok = all(num(r[k]) <= lim for k, lim in slos)
        print(f"{r[args.x]:>16} {num(r[args.throughput]):>18.1f} "
              + " ".join(f"{num(r[k]):>14.1f}" for k, _ in slos) + ("   ✅" if ok else "   ❌"))
        if ok and (best is None or num(r[args.throughput]) > num(best[args.throughput])):
            best = r
    print()
    if best:
        print(f"SLO 만족 최대 throughput: {args.x}={best[args.x]} → {num(best[args.throughput]):.1f} {args.throughput}")
    else:
        print("SLO를 만족하는 지점이 없습니다. SLO를 완화하거나 더 낮은 부하를 측정하세요.")


if __name__ == "__main__":
    main()
