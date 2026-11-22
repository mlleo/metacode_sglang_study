"""SGLang 서버 로그 파서 — 스케줄러가 매 iteration 남기는 3종 로그를 표로 만든다.

    Prefill batch, #new-seq: 5, #new-token: 1234, #cached-token: 4096, token usage: 0.31,
        #running-req: 12, #queue-req: 40, ...
    Decode batch, #running-req: 17, #token: 8931, token usage: 0.44, ..., gen throughput (token/s): 1820, #queue-req: 40
    KV cache pool is full. Retract requests. #retracted_reqs: 8, #new_tokens_gained: ..., #new_token_ratio: 0.1043 -> 0.4312

로그 필드 이름은 SGLang 버전마다 조금씩 바뀝니다. 파서는 "key: number" 쌍을 모두 뽑아내는
방식이라 새 필드가 생겨도 대부분 그대로 동작합니다.

사용 예:
    python common/logparse.py logs/server_baseline.log                # 요약
    python common/logparse.py logs/a.log logs/b.log                   # 여러 개 비교
    python common/logparse.py logs/a.log --csv results/a_iters.csv    # iteration별 CSV
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

# "key: value" 쌍. key는 '#'으로 시작하거나 영문/공백/괄호/슬래시 포함 가능
_KV = re.compile(r"(#?[A-Za-z][\w\-\s()/]*?):\s*(-?\d+(?:\.\d+)?)(?=[,\s]|$)")
_TS = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_RATIO = re.compile(r"new_token_ratio:\s*([\d.]+)\s*->\s*([\d.]+)")


def _kind(line: str) -> Optional[str]:
    if "Prefill batch" in line:
        return "prefill"
    if "Decode batch" in line:
        return "decode"
    if "Retract requests" in line or "#retracted_reqs" in line:
        return "retract"
    return None


def parse_line(line: str) -> Optional[Dict]:
    kind = _kind(line)
    if kind is None:
        return None
    row: Dict = {"kind": kind}
    m = _TS.match(line)
    if m:
        row["ts"] = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
    body = line.split("batch", 1)[-1] if kind != "retract" else line
    for k, v in _KV.findall(body):
        k = k.strip()
        if k.startswith("new_token_ratio") or k == "#new_token_ratio":
            continue
        row[k] = float(v) if "." in v else int(v)
    # 'full token usage' 같은 변형 이름을 'token usage'로 정규화
    for alt in ("full token usage",):
        if alt in row and "token usage" not in row:
            row["token usage"] = row[alt]
    if kind == "retract":
        m = _RATIO.search(line)
        if m:
            row["ratio_before"], row["ratio_after"] = float(m.group(1)), float(m.group(2))
    return row


def parse_file(path: str) -> List[Dict]:
    rows = []
    with open(path, errors="ignore") as f:
        for line in f:
            r = parse_line(line)
            if r:
                rows.append(r)
    return rows


def summarize(rows: List[Dict]) -> Dict[str, float]:
    pre = [r for r in rows if r["kind"] == "prefill"]
    dec = [r for r in rows if r["kind"] == "decode"]
    ret = [r for r in rows if r["kind"] == "retract"]

    def s(xs, k):
        return sum(x.get(k, 0) for x in xs)

    def mx(xs, k):
        return max((x.get(k, 0) for x in xs), default=0)

    new_tok, cached = s(pre, "#new-token"), s(pre, "#cached-token")
    out = {
        "prefill_batches": len(pre),
        "decode_log_lines": len(dec),
        "sum_new_token": new_tok,
        "sum_cached_token": cached,
        "cache_hit_rate": cached / (new_tok + cached) if (new_tok + cached) else 0.0,
        "new_seq_mean": s(pre, "#new-seq") / len(pre) if pre else 0.0,
        "new_seq_max": mx(pre, "#new-seq"),
        "queue_req_max": max(mx(pre, "#queue-req"), mx(dec, "#queue-req")),
        "running_req_max": max(mx(pre, "#running-req"), mx(dec, "#running-req")),
        "token_usage_max": max(mx(pre, "token usage"), mx(dec, "token usage")),
        "gen_throughput_max": mx(dec, "gen throughput (token/s)"),
        "retract_events": len(ret),
        "retracted_reqs": s(ret, "#retracted_reqs"),
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+")
    p.add_argument("--csv", default=None, help="(로그가 하나일 때) iteration별 행을 CSV로 저장")
    args = p.parse_args()

    summaries = []
    for path in args.logs:
        rows = parse_file(path)
        sm = summarize(rows)
        sm["log"] = os.path.basename(path)
        summaries.append(sm)
        if args.csv and len(args.logs) == 1:
            keys: List[str] = []
            for r in rows:
                for k in r:
                    if k not in keys:
                        keys.append(k)
            os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
            with open(args.csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
            print(f"iteration 행 {len(rows)}개 → {args.csv}")

    keys = ["log"] + [k for k in summaries[0] if k != "log"]
    width = max(len(k) for k in keys)
    for k in keys:
        vals = []
        for sm in summaries:
            v = sm[k]
            vals.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        print(f"{k:<{width}}  " + "  ".join(f"{v:>18}" for v in vals))


if __name__ == "__main__":
    main()
