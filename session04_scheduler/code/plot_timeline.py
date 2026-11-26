"""interference.py --timeline 결과 → "시간 축 위의 ITL" 그래프 (로컬에서 실행).

    python session04_scheduler/code/plot_timeline.py results/s04_timeline_*.json --out figures/s04_itl_timeline.png

x: 시간(s), y: 피해자들이 겪은 token 간격(ms). 점선: 가해자 도착 시각.
가해자가 도착한 직후 ITL이 수백 ms로 튀는 모습이 head-of-line blocking입니다.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("timelines", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--gap-label", default="ITL (ms)")
    args = p.parse_args()

    fig, axes = plt.subplots(len(args.timelines), 1, figsize=(9, 2.6 * len(args.timelines)),
                             sharex=True, sharey=True, squeeze=False)
    for ax, path in zip(axes[:, 0], args.timelines):
        d = json.load(open(path))
        for times in d["victims"].values():
            xs = times[1:]
            ys = [(b - a) * 1000 for a, b in zip(times, times[1:])]
            ax.scatter(xs, ys, s=2, alpha=0.4)
        for a in d["aggressors"].values():
            ax.axvline(a["arrive_s"], ls="--", c="k", lw=0.8)
        ax.set_yscale("log")
        ax.set_ylabel(args.gap_label)
        ax.set_title(d["tag"], fontsize=10, loc="left")
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()
