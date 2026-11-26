"""`sglang.benchmark.serving --output-file`이 남긴 JSONL을 그래프용 CSV로 바꾼다.

    python session02_architecture_batching/code/bench_to_csv.py results/s02_base.jsonl --tag base
    → results/s02_base.csv  (common/plot.py로 바로 그릴 수 있음)
"""

from __future__ import annotations

import argparse
import csv
import json
import os

KEEP = [
    "max_concurrency", "request_rate", "random_input_len", "random_output_len", "completed", "duration",
    "request_throughput", "output_throughput", "total_throughput",
    "median_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
    "median_tpot_ms", "p99_tpot_ms",
    "median_itl_ms", "p95_itl_ms", "p99_itl_ms",
    "median_e2e_latency_ms", "p99_e2e_latency_ms",
    "concurrency", "max_concurrent_requests",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", nargs="+")
    p.add_argument("--tag", default=None, help="모든 행에 붙일 tag (기본: JSONL의 tag, 없으면 파일 이름)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rows = []
    for path in args.jsonl:
        file_tag = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                row = {"tag": args.tag or d.get("tag") or file_tag}
                for k in KEEP:
                    v = d.get(k)
                    if k == "request_rate" and (v is None or v == float("inf") or v == "inf"):
                        v = "inf"
                    row[k] = v
                rows.append(row)

    out = args.out or os.path.splitext(args.jsonl[0])[0] + ".csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag"] + KEEP)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
