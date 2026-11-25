#!/usr/bin/env bash
# 공식 벤치마크 도구로 동시성(closed-loop) sweep.
#
#   bash session02_architecture_batching/code/sweep_concurrency.sh <tag> [동시성 목록] [입력] [출력]
#   예) bash session02_architecture_batching/code/sweep_concurrency.sh base "1 2 4 8 16 32 64 128" 512 256
#
# 결과: results/s02_<tag>.jsonl (한 줄 = 한 동시성). CSV 변환:
#   python session02_architecture_batching/code/bench_to_csv.py results/s02_<tag>.jsonl --tag <tag>
set -euo pipefail
cd "$(dirname "$0")/../.."
source common/env.sh

TAG="${1:?tag 필요}"
CONCS="${2:-1 2 4 8 16 32 64 128}"
IN="${3:-512}"
OUT="${4:-256}"
FILE="${RESULT_DIR}/s02_${TAG}.jsonl"

for C in $CONCS; do
  # 동시성의 4배(최소 32)만큼 요청을 보내야 '정상 상태(steady state)' 구간이 충분히 생긴다
  N=$(( C * 4 > 32 ? C * 4 : 32 ))
  echo ">>> max-concurrency=${C} num-prompts=${N}"
  python3 -m sglang.benchmark.serving \
    --backend sglang --host 127.0.0.1 --port "${PORT}" --model "${MODEL}" \
    --dataset-name random-ids \
    --random-input-len "${IN}" --random-output-len "${OUT}" --random-range-ratio 1.0 \
    --num-prompts "${N}" --max-concurrency "${C}" \
    --flush-cache --warmup-requests 2 --seed "${C}" \
    --output-file "${FILE}" > "${LOG_DIR}/bench_${TAG}_c${C}.log" 2>&1 || {
      tail -n 40 "${LOG_DIR}/bench_${TAG}_c${C}.log" >&2
      exit 1
    }
  grep -E "Successful requests|Max request concurrency|Request throughput|Output token throughput|Median TTFT|P99 TTFT|Median ITL|P99 ITL" \
    "${LOG_DIR}/bench_${TAG}_c${C}.log"
  if [[ "$(awk '/^Successful requests:/ {print $3}' "${LOG_DIR}/bench_${TAG}_c${C}.log")" != "$N" ]]; then
    echo "요청 ${N}개가 모두 완료되지 않았습니다. ${LOG_DIR}/bench_${TAG}_c${C}.log를 확인하세요." >&2
    exit 1
  fi
done
echo "saved ${FILE}"
