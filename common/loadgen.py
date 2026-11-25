"""간단한 부하 생성기: 정해진 동시성(closed-loop) 또는 도착률(open-loop)로 요청을 보낸다.

공식 도구(`python -m sglang.benchmark.serving`)가 표준 벤치마크라면, 이 파일은
"내가 원하는 모양의 workload"를 직접 만들 때 쓰는 도구입니다. (prefix 공유, 긴 프롬프트 섞기 등)

예시:
    # 동시성 1,4,16에서 입력 512 / 출력 128 token 요청 64개씩
    python common/loadgen.py --num 64 --input-len 512 --output-len 128 \
        --concurrency 1,4,16 --tag baseline --out results/s02_sweep.csv

    # 모든 요청이 앞 2048 token을 공유 (system prompt 시나리오)
    python common/loadgen.py --num 32 --shared-prefix-len 2048 --input-len 64 \
        --output-len 32 --concurrency 8 --tag prefix2k --out results/s03.csv

    # open-loop: 초당 4개 도착 (Poisson)
    python common/loadgen.py --num 100 --request-rate 4 --concurrency 256 ...

라이브러리로 사용:
    from loadgen import run_requests, make_random_requests
    results, wall = run_requests(reqs, concurrency=8)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import DEFAULT_URL, flush_cache, generate, random_ids  # noqa: E402
from metrics import append_csv, print_row, save_json, summarize  # noqa: E402


def make_random_requests(
    num: int,
    input_len: int,
    output_len: int,
    shared_prefix_len: int = 0,
    num_prefix_groups: int = 1,
    seed: int = 0,
) -> List[Dict]:
    """요청 명세 리스트를 만든다.

    - 각 요청의 프롬프트 = [공유 prefix (shared_prefix_len)] + [고유 suffix (input_len)]
    - num_prefix_groups > 1 이면 prefix 종류가 여러 개이고 요청들이 A,B,C,A,B,C... 순으로 섞인다.
    """
    prefixes = [random_ids(shared_prefix_len, seed=10_000 + g) for g in range(num_prefix_groups)]
    reqs = []
    for i in range(num):
        g = i % num_prefix_groups
        suffix = random_ids(input_len, seed=seed * 1_000_003 + i)
        reqs.append({
            "input_ids": prefixes[g] + suffix,
            "max_new_tokens": output_len,
            "group": g,
            "idx": i,
        })
    return reqs


def run_requests(
    reqs: List[Dict],
    concurrency: int,
    request_rate: Optional[float] = None,
    url: str = DEFAULT_URL,
    seed: int = 0,
) -> Tuple[List[Dict], float]:
    """요청들을 보내고 (결과 리스트, 총 소요 시간)을 돌려준다.

    concurrency    : 동시에 보낼 수 있는 HTTP 요청 수
    request_rate   : None이면 빈 worker에 바로 투입. 숫자면 worker와 별도로 Poisson 도착을 예약
    t_arrival_rel  : 유한 도착률에서 클라이언트에 도착할 예정이었던 시각 (초)
                     HTTP TTFT는 t_start부터 재므로 클라이언트 대기 시간은 따로 계산한다.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be at least one")
    if request_rate is not None and request_rate <= 0:
        raise ValueError("request_rate must be positive")
    rng = random.Random(seed)
    results: List[Optional[Dict]] = [None] * len(reqs)

    def worker(i: int, spec: Dict, arrival: Optional[float]) -> None:
        try:
            r = generate(
                input_ids=spec.get("input_ids"),
                prompt=spec.get("prompt"),
                max_new_tokens=spec["max_new_tokens"],
                url=url,
            )
            r.pop("text", None)
            r.pop("token_times", None)  # 요약에는 불필요 (크기만 큼)
            r.update({k: v for k, v in spec.items() if k not in ("input_ids", "prompt")})
            if arrival is not None:
                r["t_arrival_rel"] = arrival
            results[i] = r
        except Exception as e:  # 실패도 기록해야 분포가 왜곡되지 않는다
            results[i] = {"error": repr(e), "idx": i}

    t0 = time.perf_counter()
    elapsed_until_arrival = 0.0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for i, spec in enumerate(reqs):
            if request_rate is None:
                pool.submit(worker, i, spec, None)
            else:
                elapsed_until_arrival += rng.expovariate(request_rate)
                time.sleep(max(0.0, t0 + elapsed_until_arrival - time.perf_counter()))
                pool.submit(worker, i, spec, elapsed_until_arrival)
    wall = time.perf_counter() - t0
    for r in results:
        if r and "t_start" in r:
            r["t_start_rel"] = r["t_start"] - t0
    return [r for r in results if r is not None], wall


def main() -> None:
    p = argparse.ArgumentParser(description="SGLang 부하 생성기 (closed/open loop)")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--num", type=int, default=64, help="동시성 값마다 보낼 요청 수")
    p.add_argument("--input-len", type=int, default=512, help="요청별 고유 부분 길이 (token)")
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--shared-prefix-len", type=int, default=0)
    p.add_argument("--num-prefix-groups", type=int, default=1)
    p.add_argument("--concurrency", default="8", help="쉼표로 여러 값: 1,2,4,8")
    p.add_argument("--request-rate", type=float, default=None, help="open-loop 도착률 (req/s)")
    p.add_argument("--warmup", type=int, default=2, help="측정 전 버리는 요청 수")
    p.add_argument("--flush", action="store_true", help="각 측정 전에 radix cache 비우기")
    p.add_argument("--tag", default="run")
    p.add_argument("--out", default=None, help="요약 CSV 경로 (행 추가)")
    p.add_argument("--raw-dir", default=None, help="요청별 원시 결과 JSON 저장 폴더")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.warmup:
        warm = make_random_requests(args.warmup, 32, 8, seed=999)
        run_requests(warm, concurrency=args.warmup, url=args.url)

    for c in [int(x) for x in args.concurrency.split(",")]:
        if args.flush and not flush_cache(args.url):
            p.exit(1, "cache를 비우지 못해 측정을 중단했습니다. 서버 로그를 확인하세요.\n")
        # 동시성 값마다 고유 부분의 seed를 바꾼다. 같은 seed를 쓰면 두 번째 측정부터
        # 이전 측정의 KV가 radix cache에 남아 있어 '공짜' cache hit이 섞인다.
        # (공유 prefix는 seed와 무관하게 항상 같다 → prefix 실험에는 영향 없음)
        reqs = make_random_requests(
            args.num, args.input_len, args.output_len,
            args.shared_prefix_len, args.num_prefix_groups, seed=args.seed * 1000 + c,
        )
        results, wall = run_requests(reqs, c, args.request_rate, args.url, args.seed)
        row = {
            "tag": args.tag, "concurrency": c, "request_rate": args.request_rate or "inf",
            "input_len": args.input_len, "shared_prefix_len": args.shared_prefix_len,
            "output_len": args.output_len,
        }
        row.update(summarize(results, wall))
        print_row(row)
        if args.raw_dir:
            save_json(os.path.join(args.raw_dir, f"{args.tag}_c{c}.json"), results)
        if row["num_errors"]:
            error = next(r["error"] for r in results if r.get("error"))
            p.exit(1, f"동시성 {c}: 요청 {len(results)}개 중 {row['num_errors']}개 실패. 첫 오류: {error}\n")
        if args.out:
            append_csv(args.out, row)


if __name__ == "__main__":
    main()
