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

bash common/launch.sh "prof_${TAG}" "$@"
trap 'bash common/stop.sh > /dev/null' EXIT

# 1) 배경 부하: decode가 길게 이어지도록 출력 2048
python3 common/loadgen.py --num "$CONC" --concurrency "$CONC" --input-len 256 --output-len 2048 \
  --warmup 0 --tag "prof_${TAG}" > /dev/null &
LOAD_PID=$!
sleep "${PROFILE_DELAY:-0.8}"
if ! kill -0 "$LOAD_PID" 2>/dev/null; then
  echo "프로파일 시작 전에 부하가 끝났습니다. PROFILE_DELAY를 줄여 다시 실행하세요." >&2
  exit 1
fi

# 2) decode step 10개 녹화 (끝나면 자동 정지)
curl -fsS -X POST "${SGLANG_URL}/start_profile" -H "Content-Type: application/json" \
  -d "{\"output_dir\": \"${OUT_DIR}\", \"num_steps\": 10, \"activities\": [\"CPU\", \"GPU\"]}"
wait "$LOAD_PID"
for i in $(seq 1 20); do
  if find "$OUT_DIR" -type f -name '*.trace.json*' -size +0c -print -quit | grep -q .; then
    break
  fi
  sleep 1
done
if ! find "$OUT_DIR" -type f -name '*.trace.json*' -size +0c -print -quit | grep -q .; then
  echo "trace 파일이 생성되지 않았습니다. logs/server_prof_${TAG}.log를 확인하세요." >&2
  exit 1
fi
echo "trace files:"
ls -lh "$OUT_DIR"
