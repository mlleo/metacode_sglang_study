#!/usr/bin/env python3
"""runs.csv가 가리키는 서버 로그의 해당 측정 구간을 stdout으로 보냅니다."""

import argparse
import csv
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("config")
    parser.add_argument("profile")
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    root = args.campaign.resolve()
    try:
        with (root / "runs.csv").open(newline="", encoding="utf-8") as file:
            rows = [row for row in csv.DictReader(file) if (row["config"], row["profile"], int(row["seed"]))
                    == (args.config, args.profile, args.seed)]
        if len(rows) != 1 or not rows[0].get("log"):
            raise ValueError("이 조건의 원본 서버 로그가 없습니다.")
        row = rows[0]
        path = (root / row["log"]).resolve()
        if root not in path.parents:
            raise ValueError("로그는 campaign 디렉터리 안에 있어야 합니다.")
        start, end = int(row["log_start_byte"]), int(row["log_end_byte"])
        if not 0 <= start <= end <= path.stat().st_size:
            raise ValueError("로그 위치가 파일 크기와 맞지 않습니다.")
        with path.open("rb") as file:
            file.seek(start)
            result = file.read(end - start).decode("utf-8", errors="replace")
        sys.stdout.write(result)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"로그 추출 중단. {error}\n")


if __name__ == "__main__":
    main()
