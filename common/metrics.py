"""지연시간 분포 요약 유틸리티 (numpy 없이 동작)."""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Dict, Iterable, List, Sequence


def percentile(xs: Sequence[float], q: float) -> float:
    """선형 보간 percentile (numpy.percentile 기본값과 동일한 방식). q는 0~100."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def dist(xs: Iterable[float], scale: float = 1000.0) -> Dict[str, float]:
    """mean / p50 / p95 / p99 / max. 기본 단위 변환: 초 → ms."""
    xs = [x * scale for x in xs if x is not None]
    if not xs:
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan"), "p99": float("nan"), "max": float("nan")}
    return {
        "mean": sum(xs) / len(xs),
        "p50": percentile(xs, 50),
        "p95": percentile(xs, 95),
        "p99": percentile(xs, 99),
        "max": max(xs),
    }


def summarize(results: List[dict], wall_time: float) -> Dict[str, float]:
    """loadgen 결과 리스트 → 한 줄 요약 (CSV 한 행)."""
    ok = [r for r in results if not r.get("error")]
    out_tok = sum(r.get("completion_tokens") or 0 for r in ok)
    in_tok = sum(r.get("prompt_tokens") or 0 for r in ok)
    cached = sum(r.get("cached_tokens") or 0 for r in ok)
    all_itls = [x for r in ok for x in r.get("itls", [])]
    row: Dict[str, float] = {
        "num_requests": len(results),
        "num_errors": len(results) - len(ok),
        "wall_time_s": wall_time,
        "req_per_s": len(ok) / wall_time if wall_time > 0 else float("nan"),
        "output_tok_per_s": out_tok / wall_time if wall_time > 0 else float("nan"),
        "total_tok_per_s": (in_tok + out_tok) / wall_time if wall_time > 0 else float("nan"),
        "prompt_tokens": in_tok,
        "cached_tokens": cached,
        "cache_hit_rate": cached / in_tok if in_tok else 0.0,
    }
    for name, values in (
        ("ttft_ms", [r.get("ttft") for r in ok]),
        ("itl_ms", all_itls),
        ("e2e_ms", [r.get("e2e") for r in ok]),
    ):
        for k, v in dist(values).items():
            row[f"{name}_{k}"] = v
    return row


def append_csv(path: str, row: Dict[str, object]) -> None:
    """CSV에 한 행 추가 (파일이 없으면 헤더 생성). 열 순서는 첫 행 기준."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)
    if exists:
        with open(path, newline="") as f:
            header = next(csv.reader(f), None)
        fieldnames = header or list(row.keys())
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)  # 새 열은 뒤에 붙지만 기존 행엔 비어 있음
    else:
        fieldnames = list(row.keys())
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow({k: _fmt(v) for k, v in row.items()})


def _fmt(v: object) -> object:
    if isinstance(v, float):
        return round(v, 4)
    return v


def save_json(path: str, obj: object) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def print_row(row: Dict[str, object], keys: Sequence[str] = (
    "tag", "concurrency", "req_per_s", "output_tok_per_s",
    "ttft_ms_p50", "ttft_ms_p95", "itl_ms_p50", "itl_ms_p95", "e2e_ms_p50", "cache_hit_rate",
)) -> None:
    parts = []
    for k in keys:
        if k in row:
            v = row[k]
            parts.append(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}")
    print("  ".join(parts))
