#!/usr/bin/env bash
# RunPod Pod을 처음 띄운 직후 한 번 실행하는 셋업 스크립트.
#   bash common/setup_pod.sh
#
# 하는 일
#   1) GPU / driver가 SGLang 0.5.23 (CUDA 13)을 돌릴 수 있는지 확인
#   2) SGLang이 없으면 고정 버전으로 설치 (lmsysorg/sglang 이미지면 건너뜀)
#   3) 실습용 패키지(matplotlib, pandas) 설치
#   4) 표준 모델 weight 미리 다운로드
#   5) 환경 정보를 logs/env.txt에 기록 → 과제 README의 "실험 환경"에 그대로 붙이세요
set -euo pipefail
cd "$(dirname "$0")/.."
source common/env.sh

echo "== 1) GPU / driver 확인 =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "${DRIVER_MAJOR}" -lt 580 ]; then
  cat <<EOF
[경고] NVIDIA driver ${DRIVER_MAJOR}.x 감지. SGLang ${SGLANG_VERSION}는 CUDA 13 (driver >= 580)이 필요합니다.
       → Pod을 Terminate하고, Deploy 화면의 CUDA 버전 필터에서 13.0 이상을 골라 다시 띄우세요.
       (docs/runpod_guide.md "CUDA 필터" 참고)
EOF
  exit 1
fi

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"; then
  echo "[경고] Python $(python3 -V 2>&1) — SGLang ${SGLANG_VERSION}는 Python 3.11 이상이 필요합니다."
  echo "       Python 3.11+가 들어 있는 템플릿을 고르거나, docs/runpod_guide.md의 방법 B(공식 Docker 이미지)를 쓰세요."
  exit 1
fi

echo "== 2) SGLang 설치 확인 =="
INSTALLED=$(python3 -c "import sglang; print(sglang.__version__)" 2>/dev/null || echo "none")
if [ "$INSTALLED" = "$SGLANG_VERSION" ]; then
  echo "sglang ${INSTALLED} 이미 설치됨 → 건너뜀"
else
  echo "sglang ${INSTALLED} → ${SGLANG_VERSION} 설치 (수 분 소요)"
  # 시스템 Python이 'externally managed'(PEP 668)로 막혀 있는 이미지도 있어서 --break-system-packages를 함께 준다.
  # (Pod은 일회용 컨테이너이므로 시스템 Python에 설치해도 괜찮다)
  python3 -m pip install -q --upgrade pip uv 2>/dev/null \
    || python3 -m pip install -q --break-system-packages --upgrade pip uv
  python3 -m uv pip install --system --break-system-packages --prerelease=allow "sglang==${SGLANG_VERSION}"
fi

echo "== 3) 실습 도구 설치 =="
python3 -m pip install -q matplotlib pandas requests 2>/dev/null \
  || python3 -m pip install -q --break-system-packages matplotlib pandas requests

echo "== 4) 모델 다운로드: ${MODEL} =="
python3 - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("${MODEL}", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
print("downloaded to", p)
PY

echo "== 5) 환경 기록 → ${LOG_DIR}/env.txt =="
{
  echo "date: $(date -Is)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  python3 -c "import sglang, torch; print('sglang', sglang.__version__); print('torch', torch.__version__, 'cuda', torch.version.cuda)"
  echo "model: ${MODEL}"
  echo "runpod_pod_id: ${RUNPOD_POD_ID:-unknown}"
} | tee "${LOG_DIR}/env.txt"

echo
echo "셋업 완료. 다음: bash common/launch.sh baseline"
