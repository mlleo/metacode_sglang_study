#!/usr/bin/env bash
# launch.sh가 시작한 서버 세션만 종료한다.
#   bash common/stop.sh
cd "$(dirname "$0")/.."
source common/env.sh

PID_FILE="${LOG_DIR}/server.pid"
if [ -s "$PID_FILE" ]; then
  read -r PID < "$PID_FILE"
  if [[ ! "$PID" =~ ^[1-9][0-9]+$ ]]; then
    echo "[오류] 잘못된 서버 PID: $PID_FILE" >&2
    exit 1
  fi
  if pgrep -s "$PID" -f 'sglang([.:]|_server)' >/dev/null; then
    pkill -TERM -s "$PID" || true
    for i in $(seq 1 30); do
      if ! pgrep -s "$PID" -f 'sglang([.:]|_server)' >/dev/null; then
        break
      fi
      sleep 1
    done
    pkill -KILL -s "$PID" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
fi
echo "서버 종료 확인. GPU 메모리:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true
