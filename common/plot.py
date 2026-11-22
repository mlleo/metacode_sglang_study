"""결과 CSV → 그래프 PNG. 과제 그래프는 대부분 이 스크립트 한 줄로 만들 수 있습니다.

GPU 위에서 그릴 필요 없습니다. CSV를 로컬로 가져와서(또는 git push 후) 로컬에서 그리세요.

예시:
    # 동시성에 따른 throughput과 p95 TTFT (y 지표마다 subplot 하나)
    python common/plot.py results/s02_sweep.csv --x concurrency \
        --y output_tok_per_s,ttft_ms_p95 --group tag --logx --out figures/s02_sweep.png

    # 두 지표를 x-y로: throughput(x) vs p95 latency(y) — "latency-throughput curve"
    python common/plot.py results/s02_sweep.csv --x output_tok_per_s --y e2e_ms_p95 \
        --group tag --annotate concurrency --out figures/s02_curve.png
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="+", help="하나 이상의 CSV (열 이름이 같아야 함)")
    p.add_argument("--x", required=True)
    p.add_argument("--y", required=True, help="쉼표로 여러 개")
    p.add_argument("--group", default=None, help="이 열 값별로 선을 나눔 (예: tag)")
    p.add_argument("--annotate", default=None, help="각 점 옆에 이 열 값을 표시")
    p.add_argument("--logx", action="store_true")
    p.add_argument("--logy", action="store_true")
    p.add_argument("--title", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    df = pd.concat([pd.read_csv(c) for c in args.csv], ignore_index=True)
    ys = args.y.split(",")
    fig, axes = plt.subplots(1, len(ys), figsize=(5.2 * len(ys), 4), squeeze=False)
    groups = defaultdict(lambda: df)
    if args.group:
        groups = {k: g for k, g in df.groupby(args.group, sort=False)}
    else:
        groups = {"": df}

    for ax, y in zip(axes[0], ys):
        for name, g in groups.items():
            g = g.sort_values(args.x)
            ax.plot(g[args.x], g[y], marker="o", label=str(name) if name != "" else None)
            if args.annotate:
                for _, row in g.iterrows():
                    ax.annotate(str(row[args.annotate]), (row[args.x], row[y]),
                                textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_xlabel(args.x)
        ax.set_ylabel(y)
        ax.grid(alpha=0.3)
        if args.logx:
            ax.set_xscale("log", base=2)
        if args.logy:
            ax.set_yscale("log")
        if args.group:
            ax.legend(fontsize=8)
    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()
