#!/usr/bin/env bash
# 부하를 건 상태에서 torch profiler로 decode step 몇 개를 녹화한다.
#
#   bash session06_benchmark_production/code/profile_decode.sh <tag> <동시성> [추가 server 인자...]
#   예) bash session06_benchmark_production/code/profile_decode.sh c1  1
#       bash session06_benchmark_production/code/profile_decode.sh c32 32
#       bash session06_benchmark_production/code/profile_decode.sh c1_nograph 1 --cuda-graph-backend-decode disabled
#
# 결과: profiles/<tag>/*.trace.json.gz → 로컬로 내려받아 https://ui.perfetto.dev 에 끌어다 놓기
#       요약: python session06_benchmark_production/code/trace_summary.py profiles/<tag>
set -euo pipefail
cd "$(dirname "$0")/../.."
source common/env.sh
TAG="${1:?tag}"; CONC="${2:?concurrency}"; shift 2
PROFILE_ROOT="${STUDY_ROOT}/profiles/${TAG}"
mkdir -p "$PROFILE_ROOT"
OUT_DIR=$(mktemp -d "${PROFILE_ROOT}/run.XXXXXX")

SGLANG_PROFILE_BY_STAGE_DECODE_MIN_BS="$CONC" bash common/launch.sh "prof_${TAG}" "$@"
trap 'bash common/stop.sh > /dev/null' EXIT

# 부하를 보내기 전에 단계를 지정한다. capture는 decode batch 크기가 CONC에 닿은 뒤 시작한다.
curl -fsS --max-time 30 -X POST "${SGLANG_URL}/start_profile" -H "Content-Type: application/json" \
  -d "{\"output_dir\": \"${OUT_DIR}\", \"num_steps\": 10, \"activities\": [\"CPU\", \"GPU\"], \"profile_by_stage\": true}"
python3 common/loadgen.py --num "$CONC" --concurrency "$CONC" --input-len 256 --output-len 128 \
  --warmup 0 --tag "prof_${TAG}"

for i in $(seq 1 20); do
  if find "$OUT_DIR" -type f -iname '*-decode.trace.json*' -size +0c -print -quit | grep -q .; then
    break
  fi
  sleep 1
done
if ! find "$OUT_DIR" -type f -iname '*-decode.trace.json*' -size +0c -print -quit | grep -q .; then
  echo "decode trace가 생성되지 않았습니다. logs/server_prof_${TAG}.log에서 실제 decode batch 크기와 profiler 상태를 확인하세요." >&2
  exit 1
fi
echo "trace files:"
ls -lh "$OUT_DIR"
