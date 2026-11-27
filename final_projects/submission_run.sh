#!/usr/bin/env bash
# 복사한 뒤 결과를 제출하는 repository와 일치하도록 아래 네 값을 수정하세요.
set -euo pipefail
repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
track=prefix
candidate=lpm
seeds=22,33,44
output=${1:-results/final/my-confirm}
python3 final_projects/run.py "$track" --configs "baseline,$candidate" --seeds "$seeds" \
  --profiles standard,stress --order paired --out "$output"
