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
OUT_DIR="${STUDY_ROOT}/profiles/${TAG}"
mkdir -p "$OUT_DIR"

bash common/launch.sh "prof_${TAG}" "$@"

# 1) 배경 부하: decode가 길게 이어지도록 출력 2048
python3 common/loadgen.py --num "$CONC" --concurrency "$CONC" --input-len 256 --output-len 2048 \
  --warmup 0 --tag "prof_${TAG}" > /dev/null &
LOAD_PID=$!
sleep 4   # prefill이 끝나고 decode가 안정될 때까지 (출력 2048이라 batch=1에서도 약 10초 이어짐)

# 2) decode step 10개 녹화 (끝나면 자동 정지)
curl -s -X POST "${SGLANG_URL}/start_profile" -H "Content-Type: application/json" \
  -d "{\"output_dir\": \"${OUT_DIR}\", \"num_steps\": 10, \"activities\": [\"CPU\", \"GPU\"]}"
wait $LOAD_PID || true
sleep 3
bash common/stop.sh > /dev/null
echo "trace files:"
ls -lh "$OUT_DIR"
