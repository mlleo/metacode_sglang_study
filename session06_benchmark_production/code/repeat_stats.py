"""같은 실험을 N번 반복한 결과(CSV)에서 평균 ± 95% 신뢰구간을 계산한다.

    # 같은 설정을 3번 (tag만 다르게)
    for i in 1 2 3; do python common/loadgen.py --num 64 --concurrency 16 --tag rep --seed $i --out results/s06_rep.csv; done
    python session06_benchmark_production/code/repeat_stats.py results/s06_rep.csv --group tag \
        --metrics output_tok_per_s,ttft_ms_p50,ttft_ms_p99,itl_ms_p99

"A가 B보다 5% 빠르다"는 주장은 두 신뢰구간이 겹치지 않을 때만 하세요.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict

# 양측 95% t 값 (자유도 = n-1)
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--group", default="tag")
    p.add_argument("--metrics", default="output_tok_per_s,ttft_ms_p50,ttft_ms_p99,itl_ms_p50,itl_ms_p99")
    args = p.parse_args()

    groups = defaultdict(list)
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            groups[row[args.group]].append(row)

    metrics = args.metrics.split(",")
    print(f"{'group':<16}{'n':>3}  " + "  ".join(f"{m:>26}" for m in metrics))
    for g, rows in groups.items():
        cells = []
        for m in metrics:
            xs = [float(r[m]) for r in rows if r.get(m) not in (None, "")]
            if len(xs) < 2:
                cells.append(f"{xs[0] if xs else float('nan'):>26.1f}")
                continue
            mean, sd = statistics.mean(xs), statistics.stdev(xs)
            half = T95.get(len(xs) - 1, 1.96) * sd / math.sqrt(len(xs))
            cells.append(f"{mean:>12.1f} ± {half:>7.1f} ({half / mean * 100:4.1f}%)")
        print(f"{g:<16}{len(rows):>3}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
