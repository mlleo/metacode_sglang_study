#!/usr/bin/env bash
# 작은 KV pool(기본 30000 token)에서 admission/retraction 동작을 세 가지 설정으로 비교 (서버 3번, 약 8–10분)
#
#   bash session04_scheduler/code/retraction_sweep.sh [max_total_tokens]
#
#   eos        : ignore_eos=False, 기본 설정              → 낙관적 예약, retraction 예상
#   eos_cons2  : ignore_eos=False, --schedule-conservativeness 2.0  → 초기 예약 비율 0.7×2 → 1.0 (상한)
#   fixed      : ignore_eos=True                          → 전부 예약, retraction 거의 없음
set -euo pipefail
cd "$(dirname "$0")/../.."
source common/env.sh
POOL="${1:-30000}"

run() {
  local TAG="$1" MODE="$2"; shift 2
  bash common/launch.sh "rt_${TAG}" --max-total-tokens "${POOL}" "$@"
  python3 session04_scheduler/code/retraction_demo.py --mode "$MODE" --tag "$TAG" --out "${RESULT_DIR}/s04_retract.csv"
  bash common/stop.sh > /dev/null
  echo "--- ${TAG} 로그 요약 ---"
  python3 common/logparse.py "${LOG_DIR}/server_rt_${TAG}.log" \
    | grep -E "retract_events|retracted_reqs|new_seq_mean|running_req_max|queue_req_max|token_usage_max|sum_new_token"
}

run eos eos
run eos_cons2 eos --schedule-conservativeness 2.0
run fixed fixed
echo "done → ${RESULT_DIR}/s04_retract.csv"
