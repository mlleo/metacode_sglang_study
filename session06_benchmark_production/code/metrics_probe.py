"""Prometheus `/metrics`를 읽어 핵심 지표를 사람이 읽는 형태로 보여 준다.
서버를 `--enable-metrics`로 띄워야 합니다.

    bash common/launch.sh metrics --enable-metrics
    python session06_benchmark_production/code/metrics_probe.py              # 현재 값
    python session06_benchmark_production/code/metrics_probe.py --watch 2    # 2초마다 갱신 (부하를 걸면서 보기)

histogram(TTFT, ITL)은 누적 bucket이라 '서버가 켜진 뒤 전체'의 분포입니다. 구간별로 보려면 두 번 읽어 빼야 합니다.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from client import DEFAULT_URL  # noqa: E402

_LINE = re.compile(r'^(sglang:[a-z_]+)(\{[^}]*\})?\s+([-+eE0-9.naNinf]+)$')
GAUGES = ["num_running_reqs", "num_queue_reqs", "token_usage", "num_used_tokens", "gen_throughput", "cache_hit_rate"]
COUNTERS = ["prompt_tokens_total", "generation_tokens_total"]
HISTS = ["time_to_first_token_seconds", "inter_token_latency_seconds"]


def scrape(url: str) -> dict:
    text = requests.get(f"{url}/metrics", timeout=5).text
    out: dict = {}
    for line in text.splitlines():
        m = _LINE.match(line.strip())
        if not m:
            continue
        name, labels, val = m.group(1)[len("sglang:"):], m.group(2) or "", float(m.group(3))
        le = re.search(r'le="([^"]+)"', labels)
        if name.endswith("_bucket") and le:
            out.setdefault(name, []).append((float(le.group(1)), val))
        else:
            out[name] = out.get(name, 0.0) + val if name.endswith("_total") or name.endswith("_count") or name.endswith("_sum") else val
    return out


def hist_quantile(buckets, q):
    """Prometheus histogram_quantile과 같은 방식의 근사 (bucket 경계 선형 보간)."""
    buckets = sorted(buckets)
    total = buckets[-1][1] if buckets else 0
    if total == 0:
        return float("nan")
    rank, prev_le, prev_c = q * total, 0.0, 0.0
    for le, c in buckets:
        if c >= rank:
            if le == float("inf"):
                return prev_le
            return prev_le + (le - prev_le) * ((rank - prev_c) / (c - prev_c) if c > prev_c else 0)
        prev_le, prev_c = le, c
    return prev_le


def show(m: dict) -> None:
    for g in GAUGES:
        if g in m:
            print(f"  {g:<26} {m[g]:,.3f}")
    for c in COUNTERS:
        if c in m:
            print(f"  {c:<26} {m[c]:,.0f}")
    for h in HISTS:
        b = m.get(h + "_bucket")
        if b:
            print(f"  {h:<26} p50≈{hist_quantile(b, .5) * 1000:.1f} ms  p99≈{hist_quantile(b, .99) * 1000:.1f} ms  "
                  f"(n={int(m.get(h + '_count', 0))})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--watch", type=float, default=0)
    args = p.parse_args()
    while True:
        m = scrape(args.url)
        if not m:
            print("sglang: 지표가 없습니다. 서버를 --enable-metrics로 띄웠나요?")
            return
        print(time.strftime("%H:%M:%S"))
        show(m)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
