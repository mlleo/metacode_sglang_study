#!/usr/bin/env bash
# 저장소 루트에서 bash docs/worked-results/project/run.sh results/project.csv
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
OUT="${1:-results/s06_project.csv}"
if [ -e "$OUT" ]; then
  echo "결과 파일이 이미 있습니다. 새 CSV 경로를 지정하세요." >&2
  exit 1
fi

bash common/launch.sh project --enable-metrics
trap 'bash common/stop.sh' EXIT
for seed in 1 2 3; do
  for input_len in 512 2048; do
    python3 common/loadgen.py --num 32 --input-len "$input_len" --output-len 256 \
      --concurrency 8 --flush --seed "$seed" --tag "project_in${input_len}_r${seed}" --out "$OUT"
  done
done
