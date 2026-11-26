"""`hf_generate_baseline.py`와 **같은 workload**를 SGLang 서버에 보낸다.

HF에서 "batch = B"는 B개 요청이 동시에 같은 길이로 들어온 상황이므로,
여기서는 B개 요청을 한꺼번에 보내고 전부 끝날 때까지의 시간을 잰다. CSV 열 이름이 같아서
두 파일을 합쳐 바로 그래프를 그릴 수 있다.

    bash common/launch.sh baseline
    python session01_fundamentals/code/sglang_same_workload.py --out results/s01_sglang.csv
    # (로컬) python common/plot.py results/s01_hf.csv results/s01_sglang.csv \
    #          --x batch --y output_tok_per_s,e2e_s --group engine --logx --out figures/s01_hf_vs_sglang.png
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from client import DEFAULT_URL, flush_cache  # noqa: E402
from loadgen import make_random_requests, run_requests  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--in-len", type=int, default=512)
    p.add_argument("--out-len", type=int, default=128)
    p.add_argument("--batches", default="1,8,32")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--out", default="results/s01_sglang.csv")
    args = p.parse_args()

    run_requests(make_random_requests(2, 32, 8, seed=999), concurrency=2, url=args.url)  # warm-up

    rows = []
    for b in [int(x) for x in args.batches.split(",")]:
        walls, ttfts, itls = [], [], []
        for rep in range(args.repeat):
            flush_cache(args.url)  # HF에는 prefix cache가 없으니 공정하게 비우고 시작
            reqs = make_random_requests(b, args.in_len, args.out_len, seed=b * 100 + rep)
            res, wall = run_requests(reqs, concurrency=b, url=args.url)
            if len(res) != b or any(r.get("error") or r.get("completion_tokens") != args.out_len for r in res):
                raise SystemExit(f"batch={b}, 반복={rep + 1}: 모든 요청이 지정한 길이로 완료되어야 비교할 수 있습니다. 서버 로그를 확인하세요.")
            walls.append(wall)
            ttfts += [r["ttft"] for r in res if r.get("ttft")]
            itls += [x for r in res for x in r.get("itls", [])]
        wall = statistics.median(walls)
        row = {
            "engine": "sglang",
            "batch": b,
            "in_len": args.in_len,
            "out_len": args.out_len,
            "e2e_s": round(wall, 3),
            "output_tok_per_s": round(b * args.out_len / wall, 1),
            "peak_mem_gib": "",  # 서버는 KV pool을 미리 잡아 두므로 요청별 peak 개념이 없다 (Session 3)
            "ttft_ms_p50": round(statistics.median(ttfts) * 1000, 2),
            "itl_ms_p50": round(statistics.median(itls) * 1000, 3),
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
