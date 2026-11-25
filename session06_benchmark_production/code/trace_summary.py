"""torch profiler trace(.json / .json.gz)를 요약한다: GPU가 바빴던 비율, 커널 종류별 시간.

Perfetto로 눈으로 보는 것이 먼저고, 이 스크립트는 숫자로 확인하는 용도입니다. GPU 불필요 (로컬에서 실행).

    python session06_benchmark_production/code/trace_summary.py profiles/c1
    python session06_benchmark_production/code/trace_summary.py profiles/c1 profiles/c32    # 비교

출력
    gpu_busy_pct : GPU 이벤트(커널, 복사, 초기화)가 겹친 시간을 한 번만 센 비율
                   관측 구간은 첫 GPU 이벤트 시작부터 가장 늦게 끝난 GPU 이벤트까지
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
    return d if isinstance(d, list) else d.get("traceEvents", [])


def summarize(paths: list) -> dict:
    kernels = []
    for p in paths:
        for e in load(p):
            if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("dur", 0) > 0:
                kernels.append((e["ts"], e["ts"] + e["dur"], e.get("name", "?")))
    if not kernels:
        return {}
    kernels.sort()
    span = max(end for _, end, _ in kernels) - kernels[0][0]
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
        for path in paths:
            s = summarize([path])
            if not s:
                print(f"[{path}] GPU 이벤트가 없습니다. GPU activity가 포함됐는지 확인하세요.")
                continue
            print(f"=== {path}")
            print(f"  events {s['num_kernels']:,}  span {s['span_ms']:.1f} ms  "
                  f"summed GPU time {s['total_kernel_ms']:.1f} ms  GPU busy {s['gpu_busy_pct']:.1f}%")
            print("  GPU 이벤트 시간 비율: " + "  ".join(f"{k} {v:.1f}%" for k, v in s["by_cat"].items()))
            print("  시간이 긴 이벤트")
            for name, us in s["top"]:
                print(f"    {us / 1000:8.2f} ms  [{category(name):>13}]  {name[:90]}")


if __name__ == "__main__":
    main()
