#!/usr/bin/env python3
"""세 workload를 GPU 서버 또는 로컬 mock에서 실행하고 재분석할 파일을 저장합니다."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import re
import signal
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from workloads import PROFILES, TRACKS, execute_workload, make_workload, timeline, warm_up
from client import server_info

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = Path(__file__).with_name("configs.json")
MODEL = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
SGLANG_VERSION = "0.5.23"
RESERVED_FLAGS = {"--model", "--model-path", "--revision", "--host", "--port", "--log-level"}


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def select_configs(path: Path, track: str, names: list[str]) -> dict[str, list[str]]:
    source = json.loads(path.read_text(encoding="utf-8"))[track]
    common = source["common"]
    if not isinstance(common, list):
        raise ValueError("common은 server argument 배열이어야 합니다.")
    selected = {}
    for name in names:
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
            raise ValueError(f"설정 이름은 영문 소문자로 시작해야 합니다. {name}")
        extra = source["configs"][name]
        if not isinstance(extra, list):
            raise ValueError(f"{name}은 server argument 배열이어야 합니다.")
        args = [*common, *extra]
        if any(not isinstance(arg, str) or not arg for arg in args):
            raise ValueError(f"{name}의 server argument는 문자열이어야 합니다.")
        if any(arg.split("=", 1)[0] in RESERVED_FLAGS for arg in args):
            raise ValueError("모델은 MODEL 환경 변수로 지정하세요. 포트와 주소는 runner가 관리합니다.")
        if os.environ.get("MODEL", MODEL) == MODEL:
            args = ["--revision", MODEL_REVISION, *args]
        selected[name] = args
    return selected


def planned_runs(configs: list[str], seeds: list[int], order: str):
    if order == "grouped":
        yield from ((config, seeds) for config in configs)
    else:
        # Reverse condition order every other pair to distribute start-order effects.
        for index, seed in enumerate(seeds):
            names = configs if index % 2 == 0 else list(reversed(configs))
            yield from ((config, [seed]) for config in names)


def local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def server_environment(log_dir: Path, out: Path) -> tuple[dict[str, str], str]:
    port = local_port()
    url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update(PORT=str(port), SGLANG_URL=url, LOG_DIR=str(log_dir), RESULT_DIR=str(out),
               SGLANG_TORCH_PROFILER_DIR=str(out / "profiles"))
    return env, url


def metadata(args, settings: dict[str, list[str]], mode: str) -> dict:
    info = {
        "schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "mode": mode,
        "track": args.track, "profiles": args.profiles, "seeds": args.seeds, "config_names": args.configs,
        "settings": settings, "order": args.order, "python": sys.version.split()[0],
        "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
    }
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    if git.returncode == 0:
        info["commit"] = git.stdout.strip()
    elif os.environ.get("COURSE_REVISION"):
        info["commit"] = os.environ["COURSE_REVISION"]
    if mode == "gpu":
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True, capture_output=True, check=True,
        )
        installed = importlib.metadata.version("sglang")
        if installed != SGLANG_VERSION:
            raise ValueError(f"SGLang {SGLANG_VERSION}이 필요합니다. 설치 버전은 {installed}입니다. bash common/setup_pod.sh를 실행하세요.")
        model = os.environ.get("MODEL", MODEL)
        info.update(gpu=gpu.stdout.strip().splitlines(), model=model,
                    model_revision=MODEL_REVISION if model == MODEL else None,
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    sglang=installed, torch=importlib.metadata.version("torch"))
    return info


def execute(args, settings) -> None:
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    mode = "mock" if args.mock_url else "gpu"
    info = metadata(args, settings, mode)
    write_json(out / "manifest.json", info)
    if args.mock_url and server_info(args.mock_url).get("mock") is not True:
        raise ValueError("--mock-url에는 common/mock_server.py 주소를 넣으세요.")
    paths = out / "raw"
    paths.mkdir()
    rows = []
    for number, (name, seeds) in enumerate(planned_runs(args.configs, args.seeds, args.order), 1):
        log_dir = out / "logs" / f"{number:02d}_{name}"
        log = log_dir / "server_project.log"
        owns_server = not args.mock_url
        env, url = server_environment(log_dir, out) if owns_server else (None, args.mock_url)
        try:
            if owns_server:
                subprocess.run(["bash", "common/launch.sh", "project", *settings[name]], cwd=ROOT, env=env, check=True)
            for seed in seeds:
                for profile in args.profiles:
                    tag = f"{name}_{profile}_s{seed}"
                    print(f"== {args.track} / {tag} ==", flush=True)
                    specs, concurrency = make_workload(args.track, profile, seed)
                    try:
                        warm_up(specs, url, args.timeout)
                    except RuntimeError as error:
                        write_json(paths / f"{tag}.json", {"errors": [str(error)], "stage": "warmup"})
                        raise
                    log_start = log.stat().st_size if owns_server else None
                    measure = execute_workload(args.track, specs, concurrency, url, args.timeout)
                    log_end = log.stat().st_size if owns_server else None
                    write_json(paths / f"{tag}.json", measure)
                    if measure["errors"]:
                        raise RuntimeError(f"{tag}: {'; '.join(measure['errors'])}")
                    if args.track == "mixed":
                        write_json(paths / f"{tag}.timeline.json", timeline(measure["requests"], tag))
                    row = {"config": name, "profile": profile, "seed": seed,
                           "workload_sha256": measure["workload_sha256"],
                           "expected_requests": measure["expected_requests"], **measure["metrics"]}
                    if owns_server:
                        row.update(log=str(log.relative_to(out)), log_start_byte=log_start,
                                   log_end_byte=log_end)
                    rows.append(row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
        finally:
            if owns_server and (log_dir / "server.pid").exists():
                subprocess.run(["bash", "common/stop.sh"], cwd=ROOT, env=env, check=True)
    with (out / "runs.csv").open("w", newline="", encoding="utf-8") as data:
        writer = csv.DictWriter(data, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)}개 유효 run → {out / 'runs.csv'}", flush=True)


def comma_list(value: str, choices=None) -> list[str]:
    values = value.split(",")
    if any(not item for item in values) or len(set(values)) != len(values):
        raise ValueError("쉼표 목록에 빈 값이나 중복 값이 있습니다.")
    if choices and any(item not in choices for item in values):
        raise ValueError(f"허용 값은 {', '.join(choices)}입니다.")
    return values


def handle_term(signum, _frame):
    raise SystemExit(128 + signum)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track", choices=TRACKS)
    parser.add_argument("--out", type=Path, required=True, help="아직 존재하지 않는 결과 디렉터리")
    parser.add_argument("--configs", default="baseline", help="쉼표로 구분한 설정 이름")
    parser.add_argument("--seeds", default="11", help="쉼표로 구분한 0 이상의 정수")
    parser.add_argument("--profiles", default="standard", help="pilot, standard 또는 stress")
    parser.add_argument("--order", default="grouped", choices=("grouped", "paired"))
    parser.add_argument("--config-file", type=Path, default=CONFIGS)
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout 초")
    parser.add_argument("--mock-url", help="로컬 mock 서버. pilot과 baseline으로 확인")
    args = parser.parse_args()
    try:
        args.configs = comma_list(args.configs)
        if "baseline" not in args.configs:
            raise ValueError("모든 campaign에 baseline을 포함하세요.")
        seed_names = comma_list(args.seeds)
        if any(not value.isdecimal() for value in seed_names):
            raise ValueError("seed는 0 이상의 정수여야 합니다.")
        args.seeds = [int(value) for value in seed_names]
        if len(set(args.seeds)) != len(args.seeds):
            raise ValueError("seed가 중복됩니다.")
        args.profiles = comma_list(args.profiles, PROFILES)
        if args.mock_url and (args.profiles != ["pilot"] or args.configs != ["baseline"]):
            raise ValueError("mock에서는 --profiles pilot --configs baseline을 쓰세요.")
        if not 0 < args.timeout < float("inf"):
            raise ValueError("timeout은 양수여야 합니다.")
        settings = select_configs(args.config_file, args.track, args.configs)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    bypass = ",".join(filter(None, (os.environ.get("no_proxy"), os.environ.get("NO_PROXY"), "127.0.0.1", "localhost")))
    os.environ.update(NO_PROXY=bypass, no_proxy=bypass)
    signal.signal(signal.SIGTERM, handle_term)
    try:
        execute(args, settings)
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        parser.exit(1, f"실행 중단. {error}\n")


if __name__ == "__main__":
    main()
