#!/usr/bin/env bash
# SGLang 서버를 백그라운드로 띄우고, 준비될 때까지 기다린 뒤, 시작 설정을 요약해 보여준다.
#
#   bash common/launch.sh <tag> [추가 server 인자...]
#
# 예)
#   bash common/launch.sh baseline
#   bash common/launch.sh noradix   --disable-radix-cache
#   bash common/launch.sh tiny_pool --max-total-tokens 8192
#   MODEL=Qwen/Qwen3-0.6B bash common/launch.sh small
#
# 로그는 logs/server_<tag>.log 에 남습니다. 이 로그가 실습의 1차 데이터입니다.
# 서버 끄기: bash common/stop.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source common/env.sh

TAG="${1:?사용법: bash common/launch.sh <tag> [server args...]}"
shift || true
LOG="${LOG_DIR}/server_${TAG}.log"

if [ -s "${LOG_DIR}/server.pid" ]; then
  echo "[오류] 이전 실행 기록이 남아 있습니다. 먼저 bash common/stop.sh"
  exit 1
fi

if curl -s -o /dev/null -w "%{http_code}" "${SGLANG_URL}/health" 2>/dev/null | grep -q 200; then
  echo "[오류] 이미 ${SGLANG_URL} 에 서버가 떠 있습니다. 먼저 bash common/stop.sh"
  exit 1
fi

echo "launch: model=${MODEL} tag=${TAG} extra='$*'"
echo "log   : ${LOG}"
# 실행한 명령을 로그 첫 줄에 남겨 두면 나중에 "이 로그는 어떤 설정이었지?"를 바로 알 수 있다.
echo "# CMD: python3 -m sglang.launch_server --model-path ${MODEL} --host 127.0.0.1 --port ${PORT} --log-level info $*" > "$LOG"
nohup setsid python3 -m sglang.launch_server \
  --model-path "${MODEL}" \
  --host 127.0.0.1 --port "${PORT}" \
  --log-level info \
  "$@" >> "$LOG" 2>&1 &
echo $! > "${LOG_DIR}/server.pid"

echo -n "waiting for server"
ready=0
for i in $(seq 1 "${STARTUP_POLLS:-150}"); do
  if curl -fsS --max-time 2 "${SGLANG_URL}/health" >/dev/null 2>&1; then
    echo " ready (${i}x2s)"
    ready=1
    break
  fi
  if ! kill -0 "$(cat "${LOG_DIR}/server.pid")" 2>/dev/null; then
    echo
    echo "[오류] 서버 프로세스가 종료되었습니다. 로그 마지막 40줄:"
    tail -n 40 "$LOG"
    bash common/stop.sh >/dev/null
    exit 1
  fi
  echo -n "."
  sleep 2
done
if [ "$ready" -eq 0 ]; then
  printf '\n[오류] 서버 준비 시간 초과. 로그: %s\n' "$LOG" >&2
  tail -n 40 "$LOG" >&2
  bash common/stop.sh >/dev/null
  exit 1
fi

echo "---- 시작 설정 요약 (로그에서 발췌) ----"
grep -E "max_total_num_tokens=|KV Cache is allocated|CUDA graph end|Load weight end" "$LOG" | tail -n 8 || true
echo "----------------------------------------"
echo "로그 실시간 보기: tail -f ${LOG} | grep -E 'Prefill|Decode|Retract'"
