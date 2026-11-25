#!/usr/bin/env bash
# 도착률(open-loop, Poisson) sweep — "서버 용량을 넘는 순간 대기열이 폭발한다"를 보는 실험.
#
#   bash session02_architecture_batching/code/sweep_rate.sh <tag> [도착률 목록] [입력] [출력] [요청 수]
#   예) bash session02_architecture_batching/code/sweep_rate.sh rate "2 4 8 16 24 32" 512 256 200
set -euo pipefail
cd "$(dirname "$0")/../.."
source common/env.sh

TAG="${1:?tag 필요}"
RATES="${2:-2 4 8 16 24 32}"
IN="${3:-512}"
OUT="${4:-256}"
N="${5:-200}"
FILE="${RESULT_DIR}/s02_${TAG}.jsonl"

for R in $RATES; do
  echo ">>> request-rate=${R}/s num-prompts=${N}"
  LOG="${LOG_DIR}/bench_${TAG}_r${R}.log"
  python3 -m sglang.benchmark.serving \
    --backend sglang --host 127.0.0.1 --port "${PORT}" --model "${MODEL}" \
    --dataset-name random-ids \
    --random-input-len "${IN}" --random-output-len "${OUT}" --random-range-ratio 1.0 \
    --num-prompts "${N}" --request-rate "${R}" \
    --flush-cache --warmup-requests 2 --seed 7 \
    --output-file "${FILE}" > "$LOG" 2>&1 || {
      tail -n 40 "$LOG" >&2
      exit 1
    }
  grep -E "Traffic request rate|Successful requests|Request throughput|Output token throughput|Median TTFT|P99 TTFT|Median ITL|Peak concurrent" "$LOG"
done
echo "saved ${FILE}"
