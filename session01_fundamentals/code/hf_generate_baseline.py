"""Hugging Face `model.generate()` 기준선 — "inference engine 없이"의 숫자.

모든 요청이 동시에, 같은 길이로 도착했다는 (현실에 없는) 가장 유리한 조건에서 잰다.
그런데도 Session 2에서 SGLang과 비교하면 차이가 크게 난다.

    python session01_fundamentals/code/hf_generate_baseline.py --batches 1,8,32 --out results/s01_hf.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time

import torch
from transformers import AutoModelForCausalLM


@torch.inference_mode()
def run(model, batch: int, in_len: int, out_len: int) -> float:
    ids = torch.randint(1000, 100_000, (batch, in_len), device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=out_len,
        min_new_tokens=out_len,  # EOS로 일찍 끝나지 않게 → 모든 실행이 정확히 같은 양의 일
        do_sample=False,         # greedy
        use_cache=True,
    )
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen3-1.7B"))
    p.add_argument("--in-len", type=int, default=512)
    p.add_argument("--out-len", type=int, default=128)
    p.add_argument("--batches", default="1,8,32")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--out", default="results/s01_hf.csv")
    args = p.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    run(model, 1, 32, 8)  # warm-up

    rows = []
    for b in [int(x) for x in args.batches.split(",")]:
        torch.cuda.reset_peak_memory_stats()
        lat = statistics.median(run(model, b, args.in_len, args.out_len) for _ in range(args.repeat))
        row = {
            "engine": "hf_generate",
            "batch": b,
            "in_len": args.in_len,
            "out_len": args.out_len,
            "e2e_s": round(lat, 3),                         # batch 전체가 끝나는 시간 = 각 요청의 E2E
            "output_tok_per_s": round(b * args.out_len / lat, 1),
            "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
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
