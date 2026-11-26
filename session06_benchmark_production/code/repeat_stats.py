"""같은 실험을 N번 반복한 결과(CSV)에서 평균 ± 95% 신뢰구간을 계산한다.

    # 같은 설정을 3번 (tag만 다르게)
    for i in 1 2 3; do python common/loadgen.py --num 64 --concurrency 16 --tag rep --seed $i --out results/s06_rep.csv; done
    python session06_benchmark_production/code/repeat_stats.py results/s06_rep.csv --group tag \
        --metrics output_tok_per_s,ttft_ms_p50,ttft_ms_p99,itl_ms_p99

이 출력은 설정별 변동을 보여 준다. 설정 두 개를 비교할 때는 평균 차이의
신뢰구간을 구한다. 같은 workload끼리 짝지었다면 차이를 한 표본으로 쓴다.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict

# 양측 95% t 값 (자유도 = n-1)
T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042, 60: 2.000, 120: 1.980,
}


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
            xs = [x for x in xs if math.isfinite(x)]
            if len(xs) < 2:
                cells.append(f"{xs[0] if xs else float('nan'):>26.1f}")
                continue
            mean, sd = statistics.mean(xs), statistics.stdev(xs)
            # 표에 없는 자유도는 바로 아래 값을 써서 구간을 조금 넓게 잡는다.
            df = max(value for value in T95 if value <= len(xs) - 1)
            half = T95[df] * sd / math.sqrt(len(xs))
            relative = f" ({half / abs(mean) * 100:4.1f}%)" if mean else ""
            cells.append(f"{mean:>12.1f} ± {half:>7.1f}{relative}")
        print(f"{g:<16}{len(rows):>3}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
