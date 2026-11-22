"""one_batch 결과(JSONL) → CSV + roofline 대비 '달성 메모리 대역폭' 계산.

decode step 하나가 읽는 bytes ≈ weight bytes + batch × (평균 context 길이) × KV bytes/token
달성 대역폭 = 그 bytes / median decode latency  → GPU 최대 대역폭의 몇 %인가?

    python session05_execution/code/onebatch_to_csv.py results/s05_onebatch.jsonl \
        --weight-gb 3.44 --kv-bytes 114688 --peak-gbs 1008
    # (로컬) python common/plot.py results/s05_onebatch.csv --x batch_size \
    #          --y median_decode_latency_ms,bw_util_pct --group run_name --logx --out figures/s05_decode.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl")
    p.add_argument("--weight-gb", type=float, default=3.44, help="모델 weight 크기 (GB, 10^9). fp8 weight면 절반")
    p.add_argument("--kv-bytes", type=float, default=114688, help="KV bytes/token (Qwen3-1.7B BF16 = 114688)")
    p.add_argument("--peak-gbs", type=float, default=1008, help="GPU 메모리 대역폭 GB/s (4090≈1008)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rows = []
    with open(args.jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            bs, il, ol = d["batch_size"], d["input_len"], d["output_len"]
            lat = d.get("median_decode_latency")
            row = {
                "run_name": d.get("run_name"),
                "batch_size": bs,
                "input_len": il,
                "output_len": ol,
                "prefill_latency_ms": round(d["prefill_latency"] * 1000, 2),
                "prefill_tok_per_s": round(d["prefill_throughput"], 1),
                "median_decode_latency_ms": round(lat * 1000, 3) if lat else None,
                "median_decode_tok_per_s": round(d.get("median_decode_throughput", 0), 1),
            }
            if lat:
                ctx = il + ol / 2  # decode 동안의 평균 context 길이
                step_bytes = args.weight_gb * 1e9 + bs * ctx * args.kv_bytes
                bw = step_bytes / lat / 1e9
                row["est_step_gb"] = round(step_bytes / 1e9, 3)
                row["achieved_gbs"] = round(bw, 1)
                row["bw_util_pct"] = round(bw / args.peak_gbs * 100, 1)
                row["kv_share_pct"] = round(bs * ctx * args.kv_bytes / step_bytes * 100, 1)
            rows.append(row)

    out = args.out or os.path.splitext(args.jsonl)[0] + ".csv"
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{'run':<12}{'bs':>5}{'decode ms':>11}{'tok/s':>10}{'GB/s':>8}{'BW%':>7}{'KV%':>6}")
    for r in rows:
        print(f"{str(r['run_name']):<12}{r['batch_size']:>5}{r['median_decode_latency_ms'] or 0:>11.3f}"
              f"{r['median_decode_tok_per_s']:>10.0f}{r.get('achieved_gbs', 0):>8.0f}"
              f"{r.get('bw_util_pct', 0):>7.1f}{r.get('kv_share_pct', 0):>6.1f}")
    print("saved", out)


if __name__ == "__main__":
    main()
