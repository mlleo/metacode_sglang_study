"""긴 프롬프트 하나가 다른 사용자들의 token 흐름을 얼마나 멈추게 하는가 (head-of-line blocking).

시나리오:
    t=0      '피해자' 16명이 streaming으로 대화 중 (입력 256, 출력 600)
    t=2s     '가해자' 긴 프롬프트 도착 (입력 16000, 출력 8)
    t=4.5s   가해자 하나 더
피해자들의 token 도착 시각을 전부 기록해서, 가해자 prefill 동안 ITL이 얼마나 튀는지 본다.

    bash common/launch.sh chunk4096                                   # 기본 (4090: chunked_prefill_size=4096)
    python session04_scheduler/code/interference.py --tag chunk4096 --out results/s04_interference.csv \
        --timeline results/s04_timeline_chunk4096.json

서버 설정만 바꿔 가며 반복하는 스크립트: session04_scheduler/code/interference_sweep.sh
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from client import DEFAULT_URL, flush_cache, generate, random_ids  # noqa: E402
from metrics import append_csv, percentile  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--victims", type=int, default=16)
    p.add_argument("--victim-in", type=int, default=256)
    p.add_argument("--victim-out", type=int, default=600)
    p.add_argument("--aggressor-in", type=int, default=16000)
    p.add_argument("--aggressor-at", default="2.0,4.5", help="가해자 도착 시각들 (s)")
    p.add_argument("--tag", default="run")
    p.add_argument("--out", default="results/s04_interference.csv")
    p.add_argument("--timeline", default=None, help="피해자 token 도착 시각 JSON (그래프용)")
    args = p.parse_args()

    flush_cache(args.url)
    generate(input_ids=random_ids(64, seed=0), max_new_tokens=4, url=args.url)  # warm-up
    t0 = time.perf_counter()
    victims, aggressors = {}, {}

    def victim(i):
        r = generate(input_ids=random_ids(args.victim_in, seed=500 + i), max_new_tokens=args.victim_out, url=args.url)
        victims[i] = r

    def aggressor(j, at):
        time.sleep(max(0.0, at - (time.perf_counter() - t0)))
        r = generate(input_ids=random_ids(args.aggressor_in, seed=900 + j), max_new_tokens=8, url=args.url)
        aggressors[j] = {"arrive_s": r["t_start"] - t0, "ttft_ms": r["ttft"] * 1000}

    threads = [threading.Thread(target=victim, args=(i,)) for i in range(args.victims)]
    threads += [threading.Thread(target=aggressor, args=(j, float(a)))
                for j, a in enumerate(args.aggressor_at.split(","))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    all_itl = [x * 1000 for r in victims.values() for x in r["itls"]]
    worst = [max(r["itls"]) * 1000 for r in victims.values() if r["itls"]]
    row = {
        "tag": args.tag,
        "victim_itl_ms_p50": percentile(all_itl, 50),
        "victim_itl_ms_p99": percentile(all_itl, 99),
        "victim_itl_ms_max": max(all_itl),
        "victim_worst_gap_ms_mean": sum(worst) / len(worst),  # 피해자 1명이 겪은 '가장 긴 멈춤'의 평균
        "victim_e2e_s_mean": sum(r["e2e"] for r in victims.values()) / len(victims),
        "aggressor_ttft_ms_mean": sum(a["ttft_ms"] for a in aggressors.values()) / len(aggressors),
    }
    for k, v in row.items():
        print(f"{k:<28} {v:.1f}" if isinstance(v, float) else f"{k:<28} {v}")
    append_csv(args.out, row)

    if args.timeline:
        os.makedirs(os.path.dirname(args.timeline) or ".", exist_ok=True)
        with open(args.timeline, "w") as f:
            json.dump({
                "tag": args.tag,
                "aggressors": aggressors,
                "victims": {i: [t - t0 for t in r["token_times"]] for i, r in victims.items()},
            }, f)
        print("timeline →", args.timeline)


if __name__ == "__main__":
    main()
