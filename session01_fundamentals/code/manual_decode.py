"""직접 짠 autoregressive loop로 prefill과 decode를 분리해서 잰다 (Hugging Face transformers).

`model.generate()`가 내부에서 하는 일을 손으로 풀어 쓴 것입니다:
    prefill: model(prompt 전체)      → KV cache 생성 + 첫 token
    decode : model(직전 token 1개, past_key_values=KV) → 다음 token   ← 반복

실험 (--experiment):
    batch    : 입력 512 고정, batch 1/8/32 → decode step 시간이 batch에 따라 얼마나 늘어나는가
    inlen    : batch 1 고정, 입력 128/512/2048 → prefill 시간은 늘고 decode step은 거의 그대로인가
    nocache  : KV cache 없이 매 step 전체를 다시 계산하면 얼마나 느려지는가
    all      : 위 셋 모두 (기본값, RTX 4090에서 약 3–5분)

예:
    python session01_fundamentals/code/manual_decode.py --out results/s01_manual.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time

import torch
from transformers import AutoModelForCausalLM


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()  # CUDA는 비동기 — 이게 없으면 "커널을 큐에 넣는 시간"만 잰다
    return out, time.perf_counter() - t0


@torch.inference_mode()
def run(model, batch: int, in_len: int, out_len: int, use_cache: bool) -> dict:
    ids = torch.randint(1000, 100_000, (batch, in_len), device="cuda")
    torch.cuda.reset_peak_memory_stats()

    # ---- prefill: 프롬프트 전체를 한 번에. logits는 마지막 위치만 계산 (logits_to_keep=1)
    out, t_prefill = timed(lambda: model(input_ids=ids, use_cache=use_cache, logits_to_keep=1))
    past = out.past_key_values if use_cache else None
    next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)  # greedy
    seq = torch.cat([ids, next_tok], dim=1)

    # ---- decode: token 1개씩
    steps = []
    for _ in range(out_len - 1):
        if use_cache:
            out, dt = timed(lambda: model(input_ids=next_tok, past_key_values=past,
                                          use_cache=True, logits_to_keep=1))
            past = out.past_key_values
        else:
            # cache가 없으면 지금까지의 시퀀스 전체를 다시 넣어야 한다
            out, dt = timed(lambda: model(input_ids=seq, use_cache=False, logits_to_keep=1))
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
        seq = torch.cat([seq, next_tok], dim=1)
        steps.append(dt)

    steps_ms = sorted(s * 1000 for s in steps)
    total = t_prefill + sum(steps)
    return {
        "kv_cache": use_cache,
        "batch": batch,
        "in_len": in_len,
        "out_len": out_len,
        "prefill_ms": round(t_prefill * 1000, 2),
        "decode_step_ms_p50": round(statistics.median(steps_ms), 3),
        "decode_step_ms_p90": round(steps_ms[int(len(steps_ms) * 0.9)], 3),
        "total_s": round(total, 3),
        "decode_tok_per_s": round(batch * len(steps) / sum(steps), 1),
        "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen3-1.7B"))
    p.add_argument("--experiment", default="all", choices=["batch", "inlen", "nocache", "all"])
    p.add_argument("--out-len", type=int, default=64)
    p.add_argument("--bw-gbs", type=float, default=1008.0, help="GPU 메모리 대역폭 GB/s (4090≈1008, L4≈300)")
    p.add_argument("--out", default="results/s01_manual.csv")
    args = p.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    weight_bytes = sum(t.numel() * t.element_size() for t in model.parameters())
    lower_ms = weight_bytes / (args.bw_gbs * 1e9) * 1000
    print(f"weights = {weight_bytes / 1e9:.2f} GB → batch=1 decode 이론 하한 ≈ {lower_ms:.2f} ms/step "
          f"(대역폭 {args.bw_gbs:.0f} GB/s 가정)")

    plan = []
    if args.experiment in ("batch", "all"):
        plan += [("batch", True, b, 512) for b in (1, 8, 32)]
    if args.experiment in ("inlen", "all"):
        plan += [("inlen", True, 1, n) for n in (128, 512, 2048)]
    if args.experiment in ("nocache", "all"):
        plan += [("nocache", False, 1, 512)]

    run(model, 1, 32, 4, True)  # warm-up: 첫 실행에는 커널 선택·메모리 할당 비용이 섞인다

    rows = []
    for exp, cache, b, n in plan:
        r = run(model, b, n, args.out_len, cache)
        r = {"experiment": exp, **r, "theory_lower_ms": round(lower_ms, 2)}
        rows.append(r)
        print(r)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved", args.out)


if __name__ == "__main__":
    main()
