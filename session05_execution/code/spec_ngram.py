"""Speculative decoding(NGRAM)이 '언제' 이득인가: 반복적인 출력 vs 창의적인 출력, 동시성 1 vs 8.

NGRAM은 별도의 draft 모델 없이, 이미 본 token 시퀀스(프롬프트+생성분)에서 "다음에 올 법한 token 열"을 찾아
target 모델이 한 번에 검증한다. 출력이 입력을 많이 베끼는 작업(코드 수정, 문서 교정, 요약에서 인용)에 강하다.

    # 1) 기준선
    bash common/launch.sh spec_off --mem-fraction-static 0.7 --cuda-graph-max-bs-decode 8
    python session05_execution/code/spec_ngram.py --tag off --out results/s05_spec.csv
    bash common/stop.sh
    # 2) NGRAM (SGLang 문서 예시 설정)
    bash common/launch.sh spec_ngram --speculative-algorithm NGRAM --speculative-num-draft-tokens 16 \
        --speculative-ngram-max-bfs-breadth 10 --mem-fraction-static 0.7 --cuda-graph-max-bs-decode 8
    python session05_execution/code/spec_ngram.py --tag ngram --out results/s05_spec.csv
    grep "accept len" logs/server_spec_ngram.log | tail -5
    bash common/stop.sh
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from client import DEFAULT_URL, generate  # noqa: E402
from metrics import append_csv  # noqa: E402

DOC = (
    "def binary_search(arr, target):\n"
    "    lo, hi = 0, len(arr) - 1\n"
    "    while lo <= hi:\n"
    "        mid = (lo + hi) // 2\n"
    "        if arr[mid] == target:\n"
    "            return mid\n"
    "        elif arr[mid] < target:\n"
    "            lo = mid + 1\n"
    "        else:\n"
    "            hi = mid - 1\n"
    "    return -1\n"
)
WORKLOADS = {
    # 출력이 입력을 거의 그대로 베낀다 → n-gram 추측이 잘 맞는다
    "repetitive": lambda i: (f"Here is a Python file (version {i}):\n\n{DOC * 4}\n"
                             "Rewrite the file exactly, but rename the function to `bsearch`. Output only code:\n\n"),
    # 출력이 입력과 무관하다 → 추측이 거의 안 맞는다
    "creative": lambda i: f"Write an imaginative short story (#{i}) about a dragon who loves astronomy.\n\n",
}


def run(prompt_fn, concurrency: int, n: int, max_new: int, url: str):
    def one(i):
        r = generate(prompt=prompt_fn(i), max_new_tokens=max_new, ignore_eos=False, url=url)
        decode_s = r["e2e"] - (r["ttft"] or 0)
        toks = (r["completion_tokens"] or 1) - 1
        return {"tok_per_s": toks / decode_s if decode_s > 0 else 0, "ttft": r["ttft"],
                "itl_p50": statistics.median(r["itls"]) if r["itls"] else 0, "n": toks + 1}
    with ThreadPoolExecutor(concurrency) as ex:
        return list(ex.map(one, range(n)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--tag", required=True)
    p.add_argument("--concurrency", default="1,8")
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--out", default="results/s05_spec.csv")
    args = p.parse_args()

    run(WORKLOADS["creative"], 1, 1, 16, args.url)  # warm-up
    for name, fn in WORKLOADS.items():
        for c in [int(x) for x in args.concurrency.split(",")]:
            res = run(fn, c, max(c * 2, 4), args.max_new_tokens, args.url)
            row = {
                "tag": args.tag, "workload": name, "concurrency": c,
                "per_req_decode_tok_per_s": statistics.median(r["tok_per_s"] for r in res),
                "itl_ms_p50": statistics.median(r["itl_p50"] for r in res) * 1000,
                "ttft_ms_p50": statistics.median(r["ttft"] for r in res) * 1000,
                "completion_len_mean": statistics.mean(r["n"] for r in res),
            }
            print("  ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
            append_csv(args.out, row)


if __name__ == "__main__":
    main()
