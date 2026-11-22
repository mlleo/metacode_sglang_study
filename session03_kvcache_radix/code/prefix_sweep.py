"""공유 prefix 길이에 따른 TTFT: cache가 차갑을 때(cold) vs 따뜻할 때(warm).

각 prefix 길이 L에 대해
    cold : 매 요청 전에 flush → prefix를 매번 새로 계산
    warm : prefix를 한 번 먼저 보내 tree에 올려 둔 뒤, suffix만 다른 요청들을 보냄
요청 = [공유 prefix L token] + [고유 suffix 64 token], 출력 8 token

    python session03_kvcache_radix/code/prefix_sweep.py --out results/s03_prefix_sweep.csv
    # (로컬) python common/plot.py results/s03_prefix_sweep.csv --x prefix_len --y ttft_ms_p50 --group mode \
    #          --out figures/s03_prefix_sweep.png
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from client import DEFAULT_URL, flush_cache, generate, random_ids  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--prefix-lens", default="0,256,1024,2048,4096,8192")
    p.add_argument("--suffix-len", type=int, default=64)
    p.add_argument("--output-len", type=int, default=8)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--out", default="results/s03_prefix_sweep.csv")
    args = p.parse_args()

    generate(input_ids=random_ids(64, seed=0), max_new_tokens=4, url=args.url)  # warm-up
    rows = []
    for L in [int(x) for x in args.prefix_lens.split(",")]:
        prefix = random_ids(L, seed=1000 + L)
        for mode in ("cold", "warm"):
            flush_cache(args.url)
            if mode == "warm" and L > 0:
                generate(input_ids=prefix + random_ids(args.suffix_len, seed=-1), max_new_tokens=1, url=args.url)
            ttfts, cached = [], []
            for i in range(args.repeat):
                if mode == "cold":
                    flush_cache(args.url)
                ids = prefix + random_ids(args.suffix_len, seed=L * 100 + i)
                r = generate(input_ids=ids, max_new_tokens=args.output_len, url=args.url)
                ttfts.append(r["ttft"] * 1000)
                cached.append(r["cached_tokens"])
            row = {
                "prefix_len": L,
                "mode": mode,
                "prompt_len": L + args.suffix_len,
                "ttft_ms_p50": round(statistics.median(ttfts), 2),
                "ttft_ms_min": round(min(ttfts), 2),
                "cached_mean": round(statistics.mean(cached), 1),
            }
            print(row)
            rows.append(row)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved", args.out)


if __name__ == "__main__":
    main()
