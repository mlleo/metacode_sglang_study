"""프로젝트 요청 구성과 관측값. 시간은 한 클라이언트의 monotonic clock으로 잽니다."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from client import flush_cache, generate, random_ids  # noqa: E402
from loadgen import make_random_requests  # noqa: E402
from metrics import percentile, summarize  # noqa: E402

TRACKS = ("prefix", "mixed", "batch")
PROFILES = ("pilot", "standard", "stress")


def make_workload(track: str, profile: str, seed: int) -> tuple[list[dict], int]:
    pilot = profile == "pilot"
    stress = profile == "stress"
    if track == "prefix":
        # Same total input length and load across standard and stress.
        return make_random_requests(
            8 if pilot else 96, 64 if pilot else 512, 12 if pilot else 64,
            shared_prefix_len=128 if pilot else 1536,
            num_prefix_groups=2 if pilot else (24 if stress else 8), seed=seed,
        ), 4 if pilot else 32
    if track == "mixed":
        chats = make_random_requests(4 if pilot else 16, 96 if pilot else 256, 40 if pilot else 512, seed=seed)
        docs = make_random_requests(2 if stress else 1, 512 if pilot else 16000, 8, seed=seed + 100_000)
        for req in chats:
            req["role"] = "chat"
        for req in docs:
            req["role"] = "document"
            req["idx"] += len(chats)
        return chats + docs, len(chats) + len(docs)
    if track == "batch":
        specs = make_random_requests(8 if pilot else 96, 128 if pilot else 512, 12 if pilot else 128, seed=seed)
        if stress:
            for req in specs[::4]:
                req["max_new_tokens"] = 384
        return specs, 4 if pilot else 32
    raise ValueError(f"알 수 없는 workload {track}")


def measure_request(
    spec: dict, origin: float, url: str, timeout: float,
    on_first_token: Callable[[], None] | None = None,
) -> dict:
    record = {key: value for key, value in spec.items() if key != "input_ids"}
    record["input_len"] = len(spec["input_ids"])
    try:
        response = generate(
            input_ids=spec["input_ids"], max_new_tokens=spec["max_new_tokens"],
            url=url, timeout=timeout, on_first_token=on_first_token,
        )
        start = response.pop("t_start")
        response["t_start_rel"] = start - origin
        response["finish_s"] = start - origin + response["e2e"]
        response["token_times"] = [value - origin for value in response["token_times"]]
        response.pop("text")
        record.update(response)
        if record["completion_tokens"] != spec["max_new_tokens"] or record["prompt_tokens"] != record["input_len"]:
            raise ValueError("서버에서 보고한 입력 또는 출력 길이가 요청과 다릅니다.")
        if record["ttft"] is None or not record["token_times"]:
            raise ValueError("첫 토큰 시각을 기록하지 못했습니다.")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        record["error"] = str(error) or type(error).__name__
    return record


def execute_workload(track: str, profile: str, seed: int, url: str, timeout: float) -> dict:
    specs, concurrency = make_workload(track, profile, seed)
    # An unrelated warm-up exercises the same endpoint before each measured cold start.
    warm = {"idx": -1, "input_ids": random_ids(32, seed=90281), "max_new_tokens": 8}
    response = measure_request(warm, time.perf_counter(), url, timeout)
    if response.get("error"):
        raise RuntimeError(f"워밍업 실패. {response['error']}")
    if not flush_cache(url):
        raise RuntimeError("cache flush가 실패했습니다.")

    origin = time.perf_counter()

    def send(spec, on_first_token=None):
        return measure_request(spec, origin, url, timeout, on_first_token)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        if track == "mixed":
            chats = [spec for spec in specs if spec["role"] == "chat"]
            docs = [spec for spec in specs if spec["role"] == "document"]
            ready = [threading.Event() for _ in chats]
            futures = [executor.submit(send, spec, event.set) for spec, event in zip(chats, ready)]
            gate = time.perf_counter() + min(15.0, timeout)
            all_streaming = all(event.wait(max(0.0, gate - time.perf_counter())) for event in ready)
            if all_streaming:
                # Early decode, after every chat has received its first chunk.
                time.sleep(0.025)
                futures.extend(executor.submit(send, spec) for spec in docs)
            records = [future.result() for future in futures]
        else:
            records = list(executor.map(send, specs))

    wall_s = time.perf_counter() - origin
    errors = [f"req {row['idx']}: {row['error']}" for row in records if row.get("error")]
    if len(records) != len(specs):
        errors.append(f"{len(specs)}개 중 {len(records)}개만 실행했습니다. chat 첫 토큰 gate를 확인하세요.")
    metrics = None
    if not errors:
        try:
            metrics = summarize_workload(track, records, wall_s)
        except ValueError as error:
            errors.append(str(error))
    digest = hashlib.sha256(json.dumps(specs, sort_keys=True).encode("utf-8")).hexdigest()
    return {"expected_requests": len(specs), "client_concurrency": concurrency, "workload_sha256": digest,
            "errors": errors, "metrics": metrics, "requests": records}


def summarize_workload(track: str, records: list[dict], wall_s: float) -> dict:
    base = summarize(records, wall_s)
    result = {
        "completed": len(records), "output_tok_per_s": base["output_tok_per_s"],
        "wall_s": wall_s, "ttft_ms_p95": base["ttft_ms_p95"],
    }
    if track == "prefix":
        if any(row["cached_tokens"] is None for row in records):
            raise ValueError("cached_tokens가 응답에 없습니다.")
        result.update(cache_hit_rate=base["cache_hit_rate"],
                      uncached_input_tokens=base["prompt_tokens"] - base["cached_tokens"])
    elif track == "mixed":
        chats = [row for row in records if row["role"] == "chat"]
        docs = [row for row in records if row["role"] == "document"]
        overlaps = [sum(chat["token_times"][0] <= doc["t_start_rel"] < chat["token_times"][-1]
                        for chat in chats) for doc in docs]
        if min(overlaps) < math.ceil(len(chats) * 0.75):
            raise ValueError(f"긴 요청 도착 시 실제 스트리밍 겹침 {min(overlaps)}/{len(chats)}. 목표는 75% 이상입니다.")
        itls = [gap * 1000 for row in chats for gap in row["itls"]]
        gaps = [(end - start) * 1000 for row in chats
                for start, end in zip(row["token_times"], row["token_times"][1:])]
        if not itls or not gaps:
            raise ValueError("chat의 ITL 또는 청크 간격이 비어 있습니다.")
        result.update(chat_itl_ms_p99=percentile(itls, 99), chat_gap_ms_max=max(gaps),
                      document_ttft_ms_mean=sum(row["ttft"] for row in docs) * 1000 / len(docs),
                      min_chats_overlapping=min(overlaps), chat_requests=len(chats), document_requests=len(docs))
    elif track == "batch":
        # Every request is offered at job arrival. Finish time includes client queueing.
        result.update(job_finish_s_p95=percentile([row["finish_s"] for row in records], 95),
                      http_e2e_ms_p95=base["e2e_ms_p95"])
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("유한하지 않은 측정값이 있습니다.")
    return result


def timeline(records: list[dict], tag: str) -> dict:
    return {
        "tag": tag,
        "victims": {row["idx"]: row["token_times"] for row in records if row["role"] == "chat"},
        "aggressors": {row["idx"]: {"arrive_s": row["t_start_rel"], "ttft_ms": row["ttft"] * 1000}
                       for row in records if row["role"] == "document"},
    }
