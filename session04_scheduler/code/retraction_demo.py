"""작은 KV pool에서 '낙관적 예약 → 메모리 부족 → retraction'을 일으키고, 그 비용을 잰다.

두 가지 모드로 같은 요청들을 보낸다:
    --mode eos     : ignore_eos=False. 스케줄러는 "다 생성하지 않을 것"이라 보고 max_new_tokens × new_token_ratio 만큼만 예약
                     → 많이 받아들이지만, 실제로 길게 생성하면 KV가 모자라 retract
    --mode fixed   : ignore_eos=True. 길이가 확정이므로 스케줄러가 max_new_tokens 전부를 예약 (ratio=1.0)
                     → 적게 받아들이지만 retract는 거의 없다
                     (v0.5.23 schedule_policy.py `_get_running_request_total_token_offset`)

    bash common/launch.sh tight --max-total-tokens 30000
    python session04_scheduler/code/retraction_demo.py --mode eos   --tag eos   --out results/s04_retract.csv
    python session04_scheduler/code/retraction_demo.py --mode fixed --tag fixed --out results/s04_retract.csv
    python common/logparse.py logs/server_tight.log
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from client import DEFAULT_URL, flush_cache, generate  # noqa: E402
from metrics import append_csv, summarize  # noqa: E402

TOPICS = ["a lighthouse keeper", "a robot learning to paint", "the history of tea", "a city under the sea",
          "a chess tournament", "an old map", "a volcano observatory", "a forgotten library"]


def prompt(i: int) -> str:
    return (f"Story #{i}. Write a very long, detailed story about {TOPICS[i % len(TOPICS)]}. "
            f"Describe every scene, every character and every conversation in depth. Begin:\n\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--mode", choices=["eos", "fixed"], required=True)
    p.add_argument("--num", type=int, default=128)
    p.add_argument("--concurrency", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--tag", default=None)
    p.add_argument("--out", default="results/s04_retract.csv")
    args = p.parse_args()

    flush_cache(args.url)
    t0 = time.perf_counter()

    def one(i):
        return generate(prompt=prompt(i), max_new_tokens=args.max_new_tokens,
                        ignore_eos=(args.mode == "fixed"), temperature=0.8,  # 다양한 출력 길이를 위해 약간의 sampling
                        url=args.url)

    with ThreadPoolExecutor(args.concurrency) as ex:
        results = list(ex.map(one, range(args.num)))
    wall = time.perf_counter() - t0
    for r in results:
        r.pop("text", None)
        r.pop("token_times", None)

    row = {"tag": args.tag or args.mode, "mode": args.mode}
    row.update(summarize(results, wall))
    out_lens = sorted(r["completion_tokens"] or 0 for r in results)
    row["completion_len_p50"] = out_lens[len(out_lens) // 2]
    row["completion_len_mean"] = sum(out_lens) / len(out_lens)
    for k in ("wall_time_s", "output_tok_per_s", "ttft_ms_p50", "ttft_ms_p99", "e2e_ms_p50", "e2e_ms_p99",
              "completion_len_p50", "completion_len_mean"):
        print(f"{k:<22} {row[k]:.1f}" if isinstance(row[k], float) else f"{k:<22} {row[k]}")
    append_csv(args.out, row)
    print("\n이제 서버 로그를 보세요:  python common/logparse.py logs/server_<tag>.log")
    print("           grep -E 'Retract' logs/server_<tag>.log | head")


if __name__ == "__main__":
    main()
