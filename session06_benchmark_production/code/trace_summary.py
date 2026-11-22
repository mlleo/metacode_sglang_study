"""torch profiler trace(.json / .json.gz)를 요약한다: GPU가 바빴던 비율, 커널 종류별 시간.

Perfetto로 눈으로 보는 것이 먼저고, 이 스크립트는 숫자로 확인하는 용도입니다. GPU 불필요 (로컬에서 실행).

    python session06_benchmark_production/code/trace_summary.py profiles/c1
    python session06_benchmark_production/code/trace_summary.py profiles/c1 profiles/c32    # 비교

출력
    gpu_busy_pct : (GPU 커널이 하나라도 돌고 있던 시간) / (첫 커널 시작 ~ 마지막 커널 끝)
                   → 낮으면 GPU가 CPU를 기다리고 있다 (launch overhead, 스케줄링)
    카테고리별 GPU 시간 비율 (gemm / attention / norm·act / sampling / memcpy / other)
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
from collections import defaultdict

CATEGORIES = [  # 위에서부터 먼저 걸리는 것으로 분류 (근사치입니다)
    ("sampling", ("sampling", "argmax", "topk", "top_k", "softmax", "sample", "logits")),
    ("attention", ("flashinfer", "attention", "fmha", "batchdecode", "batchprefill", "decode_attention", "extend_attention", "fwd_kernel")),
    ("gemm", ("gemm", "cutlass", "cublas", "matmul", "xmma", "sm80_", "sm89_", "sm90_", "ampere_", "fp8")),
    ("norm/act/rope", ("rms", "norm", "silu", "act_and_mul", "rotary", "rope", "gelu")),
    ("memcpy/index", ("memcpy", "memset", "copy", "index", "gather", "scatter", "cat")),
]


def category(name: str) -> str:
    n = name.lower()
    for cat, keys in CATEGORIES:
        if any(k in n for k in keys):
            return cat
    return "other"


def load(path: str) -> list:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        d = json.load(f)
    return d.get("traceEvents", d if isinstance(d, list) else [])


def summarize(paths: list) -> dict:
    kernels = []
    for p in paths:
        for e in load(p):
            if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
                kernels.append((e["ts"], e["ts"] + e.get("dur", 0), e.get("name", "?")))
    if not kernels:
        return {}
    kernels.sort()
    span = kernels[-1][1] - kernels[0][0]
    busy, cur_s, cur_e = 0.0, *kernels[0][:2]
    for s, e, _ in kernels[1:]:
        if s > cur_e:
            busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    by_cat = defaultdict(float)
    by_name = defaultdict(float)
    for s, e, n in kernels:
        by_cat[category(n)] += e - s
        by_name[n] += e - s
    total = sum(by_cat.values())
    return {
        "num_kernels": len(kernels),
        "span_ms": span / 1000,
        "gpu_busy_pct": busy / span * 100 if span else 0,
        "by_cat": {k: v / total * 100 for k, v in sorted(by_cat.items(), key=lambda x: -x[1])},
        "top": sorted(by_name.items(), key=lambda x: -x[1])[:8],
        "total_kernel_ms": total / 1000,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+", help="trace 파일이 든 폴더 또는 파일")
    args = p.parse_args()
    for d in args.dirs:
        paths = [d] if os.path.isfile(d) else sorted(glob.glob(os.path.join(d, "**", "*.json*"), recursive=True))
        if not paths:
            print(f"[{d}] trace 파일 없음")
            continue
        s = summarize(paths)
        if not s:
            print(f"[{d}] GPU 커널 이벤트를 찾지 못했습니다 (activities에 GPU가 포함됐나요?)")
            continue
        print(f"=== {d}  ({len(paths)} files)")
        print(f"  kernels {s['num_kernels']:,}  span {s['span_ms']:.1f} ms  kernel time {s['total_kernel_ms']:.1f} ms  "
              f"GPU busy {s['gpu_busy_pct']:.1f}%")
        print("  GPU 시간 비율: " + "  ".join(f"{k} {v:.1f}%" for k, v in s["by_cat"].items()))
        print("  상위 커널:")
        for name, us in s["top"]:
            print(f"    {us / 1000:8.2f} ms  [{category(name):>13}]  {name[:90]}")


if __name__ == "__main__":
    main()
