"""Session 4 과제 예시의 두 입력 길이와 p99 ITL 기준을 한 그래프에 그립니다."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_8k", type=Path)
    parser.add_argument("csv_16k", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(6.4, 4))
    for length, path, color in [("8K", args.csv_8k, "C0"), ("16K", args.csv_16k, "C1")]:
        with path.open(newline="") as source:
            for row in csv.DictReader(source):
                x = float(row["aggressor_ttft_ms_mean"])
                y = float(row["victim_itl_ms_p99"])
                mixed = row["tag"].endswith("mx")
                ax.scatter(x, y, color=color, marker="o" if mixed else "s")
                offset = (4, 5) if mixed else (-4, 5)
                ax.annotate(f"{length} {'ON' if mixed else 'OFF'}", (x, y),
                            xytext=offset, textcoords="offset points",
                            ha="left" if mixed else "right")

    ax.axhline(100, color="0.4", ls="--", label="p99 ITL SLO (100 ms)")
    ax.set(xlabel="Long-request TTFT (ms)", ylabel="Existing streams p99 ITL (ms)")
    ax.set_ylim(bottom=0)
    ax.margins(x=0.28)
    ax.legend(loc="center")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print("saved", args.out)


if __name__ == "__main__":
    main()
