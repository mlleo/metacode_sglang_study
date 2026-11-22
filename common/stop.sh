#!/usr/bin/env bash
# 실행 중인 SGLang 서버(와 자식 프로세스: scheduler, detokenizer)를 모두 종료한다.
#   bash common/stop.sh
cd "$(dirname "$0")/.."
source common/env.sh

if [ -f "${LOG_DIR}/server.pid" ]; then
  PID=$(cat "${LOG_DIR}/server.pid")
  kill "$PID" 2>/dev/null || true
  rm -f "${LOG_DIR}/server.pid"
fi
# SGLang은 하위 프로세스 이름을 'sglang::scheduler', 'sglang::detokenizer'로 바꿔 둔다.
pkill -f "sglang.launch_server" 2>/dev/null || true
pkill -f "sglang::" 2>/dev/null || true

for i in $(seq 1 30); do
  if ! pgrep -f "sglang::|sglang.launch_server" >/dev/null; then
    break
  fi
  sleep 1
done
pkill -9 -f "sglang::|sglang.launch_server" 2>/dev/null || true
sleep 1
echo "stopped. GPU 메모리:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
