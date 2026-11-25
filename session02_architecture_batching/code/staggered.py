"""Continuous batching을 눈으로 보기: 요청을 시간차를 두고 보내고, 각 요청의 생애를 타임라인으로 그린다.

시나리오 (기본값):
    t=0.0s  A  출력 512 token  (긴 요청)
    t=0.3s  B  출력  64
    t=0.6s  C  출력  64
    t=0.9s  D  출력 256
    t=1.2s  E  출력  64

static batching이라면 B~E는 A가 끝날 때까지 기다려야 한다.
continuous batching에서는 B~E가 A의 완료를 기다리지 않고 합류할 수 있다. 일찍 들어온 짧은 요청은 A보다 먼저 끝날 수 있다.

    bash common/launch.sh baseline
    python session02_architecture_batching/code/staggered.py --out results/s02_staggered.json --plot figures/s02_timeline.png
    grep -E "Prefill batch|Decode batch" logs/server_baseline.log | tail -20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from client import DEFAULT_URL, flush_cache, generate, random_ids  # noqa: E402

SCENARIO = [  # (이름, 도착 시각 s, 입력 길이, 출력 길이)
    ("A", 0.0, 256, 512),
    ("B", 0.3, 256, 64),
    ("C", 0.6, 256, 64),
    ("D", 0.9, 256, 256),
    ("E", 1.2, 256, 64),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--out", default="results/s02_staggered.json")
    p.add_argument("--plot", default=None, help="PNG 경로 (matplotlib 필요)")
    args = p.parse_args()

    flush_cache(args.url)
    results = {}
    t0 = time.perf_counter()

    def fire(name, at, in_len, out_len):
        time.sleep(max(0.0, at - (time.perf_counter() - t0)))
        r = generate(input_ids=random_ids(in_len, seed=ord(name)), max_new_tokens=out_len, url=args.url)
        start = r["t_start"] - t0
        results[name] = {
            "arrive_s": round(start, 3),
            "first_token_s": round(start + r["ttft"], 3),
            "finish_s": round(start + r["e2e"], 3),
            "ttft_ms": round(r["ttft"] * 1000, 1),
            "out_len": out_len,
            "itl_ms_mean": round(sum(r["itls"]) / max(len(r["itls"]), 1) * 1000, 2),
        }

    threads = [threading.Thread(target=fire, args=s) for s in SCENARIO]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"{'req':<4}{'arrive':>8}{'1st tok':>9}{'finish':>8}{'TTFT ms':>9}{'out':>6}{'ITL ms':>8}")
    for name, *_ in SCENARIO:
        r = results[name]
        print(f"{name:<4}{r['arrive_s']:>8.2f}{r['first_token_s']:>9.2f}{r['finish_s']:>8.2f}"
              f"{r['ttft_ms']:>9.1f}{r['out_len']:>6}{r['itl_ms_mean']:>8.2f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 3))
        for i, (name, *_) in enumerate(SCENARIO):
            r = results[name]
            ax.barh(i, r["first_token_s"] - r["arrive_s"], left=r["arrive_s"], color="#d9a441", label="wait+prefill" if i == 0 else None)
            ax.barh(i, r["finish_s"] - r["first_token_s"], left=r["first_token_s"], color="#3b7dd8", label="decode" if i == 0 else None)
        ax.set_yticks(range(len(SCENARIO)), [s[0] for s in SCENARIO])
        ax.invert_yaxis()
        ax.set_xlabel("time (s)")
        ax.legend(loc="lower right")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        fig.savefig(args.plot, dpi=150)
        print("saved", args.plot)


if __name__ == "__main__":
    main()
