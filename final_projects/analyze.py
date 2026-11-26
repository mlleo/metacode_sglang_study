#!/usr/bin/env python3
"""각 profile에서 기준 통과 여부, seed별 차이, 짝별 95% t 구간을 계산합니다."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

# Two-sided 95% Student t critical values, degrees of freedom 1–9.
T975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262)
OBJECTIVES = {
    "prefix": ("ttft_ms_p95", False),
    "mixed": ("document_ttft_ms_mean", False),
    "batch": ("output_tok_per_s", True),
}


def paired_interval(deltas: list[float]) -> tuple[float, float | None, float | None]:
    average = statistics.mean(deltas)
    if not 2 <= len(deltas) <= 10:
        return average, None, None
    half = T975[len(deltas) - 2] * statistics.stdev(deltas) / math.sqrt(len(deltas))
    return average, average - half, average + half


def load_campaign(path: Path) -> tuple[dict, dict]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != 1 or manifest["track"] not in OBJECTIVES or "baseline" not in manifest["config_names"]:
        raise ValueError("지원하지 않는 manifest입니다.")
    rows = {}
    workloads = {}
    with (path / "runs.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["profile"], row["config"], int(row["seed"]))
            if key in rows:
                raise ValueError(f"중복된 실험 {key}")
            workload_key = (key[0], key[2])
            digest = row["workload_sha256"]
            if len(digest) != 64 or workloads.setdefault(workload_key, digest) != digest:
                raise ValueError(f"같은 profile·seed에서 입력이 달라졌습니다. {key}")
            excluded = {"config", "profile", "seed", "workload_sha256", "log", "log_start_byte", "log_end_byte"}
            rows[key] = {k: float(value) for k, value in row.items() if k not in excluded}
            if not all(math.isfinite(value) for value in rows[key].values()):
                raise ValueError(f"유한하지 않은 측정값 {key}")
            if rows[key]["completed"] != rows[key]["expected_requests"]:
                raise ValueError(f"완료 요청 수가 예상과 다릅니다. {key}")
    expected = {(profile, config, seed) for profile in manifest["profiles"]
                for config in manifest["config_names"] for seed in manifest["seeds"]}
    if not expected or rows.keys() != expected:
        raise ValueError("manifest와 runs.csv의 profile·설정·seed가 일치해야 합니다.")
    for profile in manifest["profiles"]:
        counts = {values["completed"] for key, values in rows.items() if key[0] == profile}
        if len(counts) != 1 or min(counts) < 1:
            raise ValueError(f"{profile}의 완료 요청 수가 일치해야 합니다.")
    return manifest, rows


def summarize_campaign(manifest: dict, rows: dict, chat_itl: float, chat_gap: float, deadline: float) -> list[dict]:
    track = manifest["track"]
    metric, maximize = OBJECTIVES[track]
    tests = {
        "prefix": (),
        "mixed": (("chat_itl_ms_p99", chat_itl), ("chat_gap_ms_max", chat_gap)),
        "batch": (("wall_s", deadline),),
    }[track]
    configs = manifest["config_names"]
    reference = "baseline"
    seeds = manifest["seeds"]
    summaries = []
    for profile in manifest["profiles"]:
        before = [rows[(profile, reference, seed)][metric] for seed in seeds]
        per_profile = []
        for config in configs:
            samples = [rows[(profile, config, seed)] for seed in seeds]
            values = [row[metric] for row in samples]
            diffs = [candidate - baseline for candidate, baseline in zip(values, before)]
            average, low, high = paired_interval(diffs)
            excesses = []
            if manifest["mode"] == "gpu" and profile != "pilot":
                for field, limit in tests:
                    for seed, sample in zip(seeds, samples):
                        if sample[field] > limit:
                            excesses.append(f"s{seed} {field}={sample[field]:.3f}>{limit:g}")
                status = "fail" if excesses else "pass"
            else:
                status = "smoke-only"
            per_profile.append({"profile": profile, "config": config, "baseline": reference,
                                "objective": metric, "objective_mean": statistics.mean(values),
                                "delta_mean": average, "ci95_low": low, "ci95_high": high,
                                "seed_deltas": ";".join(f"{seed}:{delta:.6g}" for seed, delta in zip(seeds, diffs)),
                                "status": status, "violations": "; ".join(excesses)})
        summaries.extend(sorted(per_profile, key=lambda row: (row["status"] != "pass",
                                                                (-1 if maximize else 1) * row["objective_mean"])))
    return summaries


def draw(path: Path, summaries: list[dict], profiles: list[str]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(profiles), figsize=(5.5 * len(profiles), 4), squeeze=False)
    for ax, profile in zip(axes[0], profiles):
        selected = [row for row in summaries if row["profile"] == profile]
        for index, row in enumerate(selected):
            color = {"pass": "#32748B", "fail": "#BC5741", "smoke-only": "#757575"}[row["status"]]
            delta = row["delta_mean"]
            low, high = row["ci95_low"], row["ci95_high"]
            error = [[delta - low], [high - delta]] if low is not None else None
            ax.errorbar(delta, index, xerr=error, fmt="o", capsize=4, color=color)
        ax.axvline(0, color="#999999", linewidth=1)
        ax.set_yticks(range(len(selected)), [f"{r['config']} ({r['status']})" for r in selected])
        ax.set_title(profile)
        ax.set_xlabel(f"{selected[0]['objective']} delta vs {selected[0]['baseline']}")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--chat-itl-ms", type=float, default=100.0)
    parser.add_argument("--chat-gap-ms", type=float, default=100.0)
    parser.add_argument("--deadline-s", type=float, default=12.0)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    if any(not 0 < value < float("inf") for value in (args.chat_itl_ms, args.chat_gap_ms, args.deadline_s)):
        parser.error("기준값은 양수여야 합니다.")
    try:
        manifest, rows = load_campaign(args.campaign)
        summaries = summarize_campaign(manifest, rows, args.chat_itl_ms, args.chat_gap_ms, args.deadline_s)
        for row in summaries:
            low, high = row["ci95_low"], row["ci95_high"]
            interval = f"[{low:+.3f}, {high:+.3f}]" if low is not None else "—"
            print(f"{row['profile']:<9} {row['config']:<15} {row['status']:<10} "
                  f"{row['objective']}={row['objective_mean']:.3f} "
                  f"Δ={row['delta_mean']:+.3f} 95% CI {interval}")
            if row["violations"]:
                print(f"  {row['violations']}")
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with args.csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
                writer.writeheader()
                writer.writerows(summaries)
        if args.plot:
            draw(args.plot, summaries, manifest["profiles"])
    except (KeyError, ValueError, OSError, TypeError) as error:
        parser.exit(1, f"분석 중단. {error}\n")


if __name__ == "__main__":
    main()
