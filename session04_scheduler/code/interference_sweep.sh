#!/usr/bin/env bash
# chunked prefill 설정을 바꿔 가며 interference.py를 반복한다. (서버를 5번 띄움, 약 10–12분)
#
#   bash session04_scheduler/code/interference_sweep.sh  # 수업용 두 설정
#   bash session04_scheduler/code/interference_sweep.sh nochunk chunk4096 chunk1024 chunk1024mx chunk1024pd
#   AGGRESSOR_IN=8000 bash session04_scheduler/code/interference_sweep.sh chunk1024 chunk1024mx
#
#   nochunk     : --chunked-prefill-size -1        (청크 없음: 16K prefill을 한 번에)
#   chunk4096   : 기본값 (RTX 4090에서 자동으로 4096)
#   chunk1024   : --chunked-prefill-size 1024
#   chunk1024mx : --chunked-prefill-size 1024 --enable-mixed-chunk  (prefill 조각과 decode를 한 batch에 섞기)
#   chunk1024pd : --chunked-prefill-size 1024 --prefill-decode-interval 1  (prefill 조각 뒤 decode 1번 보장)
#
# 이론 §4.3: 기본 설정에서는 조각들이 연달아 돌아서 chunk만 줄여서는 피해자의 멈춤이 줄지 않는다.
#           mx / pd 두 장치가 있어야 멈춤이 '조각 하나' 크기로 줄어든다. 그걸 확인하는 실험.
set -euo pipefail
cd "$(dirname "$0")/../.."
source common/env.sh

declare -A CONF=(
  [nochunk]="--chunked-prefill-size -1"
  [chunk4096]="--chunked-prefill-size 4096"
  [chunk1024]="--chunked-prefill-size 1024"
  [chunk1024mx]="--chunked-prefill-size 1024 --enable-mixed-chunk"
  [chunk1024pd]="--chunked-prefill-size 1024 --prefill-decode-interval 1"
)

if [ "$#" -eq 0 ]; then
  set -- chunk1024 chunk1024mx
fi

for TAG in "$@"; do
  if [[ ! -v 'CONF[$TAG]' ]]; then
    echo "알 수 없는 설정: $TAG" >&2
    exit 2
  fi
  # shellcheck disable=SC2086
  bash common/launch.sh "$TAG" ${CONF[$TAG]}
  STATUS=0
  python3 session04_scheduler/code/interference.py --tag "$TAG" \
    --aggressor-in "${AGGRESSOR_IN:-16000}" --aggressor-at "${AGGRESSOR_AT:-0.5}" \
    --victim-out "${VICTIM_OUT:-768}" \
    --out "${RESULT_DIR}/s04_interference.csv" --timeline "${RESULT_DIR}/s04_timeline_${TAG}.json" || STATUS=$?
  bash common/stop.sh > /dev/null
  if [ "$STATUS" -ne 0 ]; then
    exit "$STATUS"
  fi
done
echo "done → ${RESULT_DIR}/s04_interference.csv"
