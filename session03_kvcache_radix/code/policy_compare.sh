#!/usr/bin/env bash
# LPM vs FCFS: 메모리가 빠듯할 때 cache-aware 스케줄링이 cache hit와 throughput을 얼마나 바꾸는가.
#
#   bash session03_kvcache_radix/code/policy_compare.sh [max_total_tokens]
#
# 워크로드 (generated-shared-prefix): system prompt 16종 × 각 16개 질문, 순서는 섞여서 도착
#   prefix 2048 token × 16종 ≈ 3.3만 token  >  KV pool (기본 24000 token)  → 전부 동시에 캐시할 수 없음
# 서버를 두 번(lpm, fcfs) 띄우므로 약 6–8분 걸린다.
set -euo pipefail
cd "$(dirname "$0")/../.."
source common/env.sh

POOL="${1:-24000}"
OUT="${RESULT_DIR}/s03_policy.jsonl"

for POLICY in lpm fcfs; do
  bash common/launch.sh "policy_${POLICY}" --schedule-policy "${POLICY}" --max-total-tokens "${POOL}"
  trap 'bash common/stop.sh > /dev/null' EXIT
  LOG="${LOG_DIR}/bench_policy_${POLICY}.log"
  # max-concurrency를 96으로 둔다: 대기열이 128을 넘으면 LPM은 비용 때문에 스스로 FCFS로 바뀐다
  #   (schedule_policy.py: WAITING_QUEUE_PREFIX_MATCH_MAX = 128)
  python3 -m sglang.benchmark.serving \
    --backend sglang --host 127.0.0.1 --port "${PORT}" --model "${MODEL}" \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups 16 --gsp-prompts-per-group 16 \
    --gsp-system-prompt-len 2048 --gsp-question-len 128 --gsp-output-len 64 \
    --max-concurrency 96 --flush-cache --seed 1 \
    --output-file "${OUT}" > "$LOG" 2>&1 || {
      tail -n 40 "$LOG" >&2
      exit 1
    }
  grep -E "Successful requests|Request throughput|Output token throughput|Median TTFT|P99 TTFT" "$LOG"
  bash common/stop.sh > /dev/null
  trap - EXIT
  echo "--- ${POLICY} 로그 요약 ---"
  python3 common/logparse.py "${LOG_DIR}/server_policy_${POLICY}.log" | grep -E "cache_hit_rate|sum_new_token|sum_cached|queue_req_max|token_usage_max"
done
echo "saved ${OUT}"
