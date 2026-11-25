#!/usr/bin/env bash
# 서버 없이 '한 batch'의 prefill / decode step 시간을 잰다 (python -m sglang.benchmark.one_batch).
# HTTP 도착부터 재는 대신 한 batch의 준비·forward·sampling·동기화를 잰다.
#
#   bash session05_execution/code/onebatch_sweep.sh <tag> [추가 server 인자...]
#   예) bash session05_execution/code/onebatch_sweep.sh base
#       bash session05_execution/code/onebatch_sweep.sh nograph --cuda-graph-backend-decode disabled
#       bash session05_execution/code/onebatch_sweep.sh triton  --attention-backend triton
#       bash session05_execution/code/onebatch_sweep.sh fp8w    --quantization fp8
#
# 결과: results/s05_onebatch.jsonl 에 행 추가 (run_name = tag)
# 모델 로딩 + CUDA graph capture 때문에 실행당 1–2분.
set -euo pipefail
cd "$(dirname "$0")/../.."
source common/env.sh
TAG="${1:?tag 필요}"; shift || true
BS="${BS:-1 4 16 64 128}"
IN="${IN:-512}"
OUT="${OUT:-64}"

if [ -s "${LOG_DIR}/server.pid" ]; then
  echo "[오류] 이 실습의 서버가 실행 중입니다. bash common/stop.sh 먼저."; exit 1
fi

# shellcheck disable=SC2086
python3 -m sglang.benchmark.one_batch \
  --model-path "${MODEL}" \
  --batch-size ${BS} --input-len "${IN}" --output-len "${OUT}" \
  --run-name "${TAG}" --result-filename "${RESULT_DIR}/s05_onebatch.jsonl" \
  "$@" > "${LOG_DIR}/onebatch_${TAG}.log" 2>&1 || {
    tail -n 40 "${LOG_DIR}/onebatch_${TAG}.log" >&2
    exit 1
  }
grep -E "^Prefill\.|^Decode\.  median|^Total\.|max_total_num_tokens" "${LOG_DIR}/onebatch_${TAG}.log"
echo "saved → ${RESULT_DIR}/s05_onebatch.jsonl"
