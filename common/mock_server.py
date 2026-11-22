"""GPU 없이 실습 스크립트를 디버깅하기 위한 '가짜' SGLang 서버.

스터디 원칙: GPU는 측정 장비다. 스크립트 작성·디버깅은 노트북에서 이 mock 서버로 끝내고,
RunPod에서는 실행만 합니다.

    python common/mock_server.py --port 30000          # 터미널 1
    python common/client.py --input-len 512            # 터미널 2
    python common/loadgen.py --num 16 --concurrency 1,4

흉내 내는 것 (숫자는 실제 GPU와 전혀 무관):
  - /generate (stream / non-stream), /v1/chat/completions (non-stream), /health, /flush_cache, /server_info, /metrics
  - prefill 시간 ∝ cache miss token 수, decode step 시간 ∝ 1 + 0.02 × running 요청 수
  - 아주 단순한 prefix cache (이전 요청들과의 최장 공통 prefix, 마지막 token 제외, 끝난 요청의 출력 포함)
    → meta_info.cached_tokens
  - stdout에 SGLang 형식의 'Prefill batch' / 'Decode batch' 로그
흉내 내지 않는 것: 실제 batching, 메모리 한계, retraction, 텍스트 생성
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PREFILL_S_PER_TOKEN = 0.00005   # 20k tok/s
DECODE_STEP_S = 0.008

_lock = threading.Lock()
_cache: list = []               # 이전 프롬프트들 (token 튜플)
_running = 0
_queue = 0


def _log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def _match(ids: tuple) -> int:
    best = 0
    for prev in _cache:
        n = 0
        for a, b in zip(prev, ids):
            if a != b:
                break
            n += 1
        best = max(best, n)
    return best


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # http.server 기본 접근 로그 끄기
        pass

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.path = self.path.split("?", 1)[0]
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path in ("/server_info", "/get_server_info"):
            self._json(200, {"model_path": "mock", "max_total_num_tokens": 100000, "mock": True})
        elif self.path == "/metrics":
            body = (f'sglang:num_running_reqs{{model_name="mock"}} {_running}\n'
                    f'sglang:num_queue_reqs{{model_name="mock"}} {_queue}\n'
                    'sglang:token_usage{model_name="mock"} 0.1\n').encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        global _running, _queue
        self.path = self.path.split("?", 1)[0]
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/flush_cache":
            with _lock:
                _cache.clear()
            self._json(200, {"ok": True})
            return
        if self.path == "/v1/chat/completions":
            n_out = int(req.get("max_tokens", 16))
            prompt_len = sum(len(m.get("content", "")) for m in req.get("messages", []))
            time.sleep(n_out * DECODE_STEP_S)
            self._json(200, {"choices": [{"message": {"role": "assistant", "content": " tok" * n_out}}],
                             "usage": {"prompt_tokens": prompt_len, "completion_tokens": n_out}})
            return
        if self.path != "/generate":
            self._json(404, {"error": "not found"})
            return

        ids = tuple(req.get("input_ids") or [ord(c) for c in req.get("text", "")])
        sp = req.get("sampling_params", {})
        max_new = int(sp.get("max_new_tokens", 16))
        stream = bool(req.get("stream", False))

        gen_ids = list(range(1000, 1000 + max_new))   # 가짜 출력 token
        with _lock:
            cached = min(_match(ids), max(len(ids) - 1, 0))   # SGLang처럼 마지막 token은 항상 다시 계산
            _cache.append(ids)
            _running += 1
            running = _running
        new_tok = len(ids) - cached
        _log(f"Prefill batch, #new-seq: 1, #new-token: {new_tok}, #cached-token: {cached}, "
             f"token usage: {min(0.99, running * 0.05):.2f}, #running-req: {running}, #queue-req: {_queue}, "
             f"cuda graph: False, input throughput (token/s): 20000.00")
        time.sleep(new_tok * PREFILL_S_PER_TOKEN)

        meta = {"prompt_tokens": len(ids), "cached_tokens": cached, "completion_tokens": 0,
                "finish_reason": None}
        try:
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
            text = ""
            for i in range(max_new):
                if i > 0:
                    time.sleep(DECODE_STEP_S * (1 + 0.02 * _running))
                text += " tok"
                meta = dict(meta, completion_tokens=i + 1)
                if i % 32 == 31:
                    _log(f"Decode batch, #running-req: {_running}, #token: {_running * 600}, "
                         f"token usage: {min(0.99, _running * 0.05):.2f}, cuda graph: True, "
                         f"gen throughput (token/s): {_running / DECODE_STEP_S:.2f}, #queue-req: {_queue}")
                if stream:
                    self._chunk(b"data: " + json.dumps({"text": text, "output_ids": gen_ids[:i + 1],
                                                        "meta_info": meta}).encode() + b"\n\n")
            meta["finish_reason"] = {"type": "length", "length": max_new}
            with _lock:   # SGLang처럼 끝난 요청의 입력 + 출력(마지막 token 제외)을 캐시에 등록
                _cache.append(ids + tuple(gen_ids[:-1]))
            if stream:
                self._chunk(b"data: [DONE]\n\n")
                self._chunk(b"")
            else:
                self._json(200, {"text": text, "output_ids": gen_ids, "meta_info": meta})
        finally:
            with _lock:
                _running -= 1

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=30000)
    args = p.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    _log(f"mock SGLang server on :{args.port} (GPU 없음, 숫자는 가짜입니다)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
