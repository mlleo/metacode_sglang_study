# 스터디 공통 환경 변수. 모든 스크립트가 `source common/env.sh`로 읽습니다.
# 값을 바꾸고 싶으면 이 파일을 고치지 말고, 실행 전에 export 하세요.
#   예) MODEL=Qwen/Qwen3-0.6B bash common/launch.sh small

export SGLANG_VERSION="${SGLANG_VERSION:-0.5.23}"   # 스터디 전체에서 고정 (코드 리딩 기준 버전)
export MODEL="${MODEL:-Qwen/Qwen3-1.7B}"            # 표준 모델
export PORT="${PORT:-30000}"
export SGLANG_URL="${SGLANG_URL:-http://127.0.0.1:${PORT}}"

# RunPod에서는 /workspace가 (Volume Disk를 붙였다면) Stop 후에도 유지되는 경로입니다.
export STUDY_ROOT="${STUDY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [ -d /workspace ]; then
  export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
fi
export SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-${STUDY_ROOT}/profiles}"
export LOG_DIR="${LOG_DIR:-${STUDY_ROOT}/logs}"
export RESULT_DIR="${RESULT_DIR:-${STUDY_ROOT}/results}"
mkdir -p "$LOG_DIR" "$RESULT_DIR"
