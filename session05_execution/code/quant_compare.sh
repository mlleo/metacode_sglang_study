#!/usr/bin/env bash
# BF16 vs FP8 weight vs FP8 KV cache: 메모리(담을 수 있는 token 수)와 속도를 한 번에 비교 (서버 3번, 약 10분)
#
#   bash session05_execution/code/quant_compare.sh
#
#   bf16    : 기본
#   fp8w    : --quantization fp8          (weight를 로딩 시점에 FP8로 양자화. Ada/Hopper 이상)
#   fp8kv   : --kv-cache-dtype fp8_e4m3   (KV cache만 FP8)
#
# 각 설정마다: 시작 로그에서 weight 메모리 / KV token 수를 뽑고, 동시성 1·16·64로 부하를 건다.
set -euo pipefail
cd "$(dirname "$0")/../.."
source common/env.sh

declare -A CONF=(
  [bf16]=""
  [fp8w]="--quantization fp8"
  [fp8kv]="--kv-cache-dtype fp8_e4m3"
)
SUMMARY="${RESULT_DIR}/s05_quant_memory.txt"
: > "$SUMMARY"

for TAG in bf16 fp8w fp8kv; do
  # shellcheck disable=SC2086
  if ! bash common/launch.sh "q_${TAG}" ${CONF[$TAG]}; then
    echo "[${TAG}] 서버 시작 실패 — 이 GPU/backend 조합이 지원되지 않을 수 있습니다. 로그: logs/server_q_${TAG}.log" | tee -a "$SUMMARY"
    bash common/stop.sh > /dev/null || true
    continue
  fi
  {
    echo "== ${TAG} =="
    grep -E "Load weight end|KV Cache is allocated" "${LOG_DIR}/server_q_${TAG}.log" | sed -E 's/^\[[^]]*\] //'
  } | tee -a "$SUMMARY"
  python3 common/loadgen.py --num 64 --input-len 512 --output-len 256 --concurrency 1,16,64 \
    --tag "$TAG" --out "${RESULT_DIR}/s05_quant.csv"
  bash common/stop.sh > /dev/null
done
echo "memory summary → ${SUMMARY}"
echo "speed          → ${RESULT_DIR}/s05_quant.csv"
