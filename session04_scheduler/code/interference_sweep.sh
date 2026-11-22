#!/usr/bin/env bash
# chunked prefill 설정을 바꿔 가며 interference.py를 반복한다. (서버를 4번 띄움, 약 8–10분)
#
#   bash session04_scheduler/code/interference_sweep.sh
#
#   nochunk     : --chunked-prefill-size -1        (청크 없음: 16K prefill을 한 번에)
#   chunk4096   : 기본값 (RTX 4090에서 자동으로 4096)
#   chunk1024   : --chunked-prefill-size 1024
#   chunk1024mx : --chunked-prefill-size 1024 --enable-mixed-chunk  (prefill 조각과 decode를 한 batch에 섞기)
set -euo pipefail
cd "$(dirname "$0")/../.."
source common/env.sh

declare -A CONF=(
  [nochunk]="--chunked-prefill-size -1"
  [chunk4096]=""
  [chunk1024]="--chunked-prefill-size 1024"
  [chunk1024mx]="--chunked-prefill-size 1024 --enable-mixed-chunk"
)

for TAG in nochunk chunk4096 chunk1024 chunk1024mx; do
  # shellcheck disable=SC2086
  bash common/launch.sh "$TAG" ${CONF[$TAG]}
  python3 session04_scheduler/code/interference.py --tag "$TAG" \
    --out "${RESULT_DIR}/s04_interference.csv" --timeline "${RESULT_DIR}/s04_timeline_${TAG}.json"
  bash common/stop.sh > /dev/null
done
echo "done → ${RESULT_DIR}/s04_interference.csv"
