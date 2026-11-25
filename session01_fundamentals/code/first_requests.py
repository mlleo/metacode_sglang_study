"""SGLang 서버에 보내는 첫 요청들: native API, streaming, OpenAI 호환 API, 동시 요청.

먼저 서버를 띄우세요:  bash common/launch.sh baseline
    python session01_fundamentals/code/first_requests.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from client import DEFAULT_URL, generate, random_ids  # noqa: E402

URL = DEFAULT_URL
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-1.7B")


def section(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# 1) native /generate — 응답에 meta_info (prompt_tokens, cached_tokens, ...)가 같이 온다
section("1) native /generate (non-stream)")
r = requests.post(f"{URL}/generate", json={
    "text": "The capital of France is",
    "sampling_params": {"max_new_tokens": 16, "temperature": 0},
}).json()
print("text     :", repr(r["text"]))
print("meta_info:", json.dumps({k: r["meta_info"].get(k) for k in
                                ("prompt_tokens", "completion_tokens", "cached_tokens", "finish_reason", "e2e_latency")},
                               ensure_ascii=False))

# 2) streaming — token이 도착하는 대로 출력, TTFT와 ITL 측정
section("2) streaming: 첫 토큰과 토큰 사이 간격 재기")
t0 = time.perf_counter()
first = None
prev_len = 0
with requests.post(f"{URL}/generate", stream=True, json={
    "text": "GPU stands for",
    "sampling_params": {"max_new_tokens": 64, "temperature": 0},
    "stream": True,
}) as resp:
    for line in resp.iter_lines():
        if not line or not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        chunk = json.loads(data)
        if first is None:
            first = time.perf_counter() - t0
        text = chunk["text"]
        print(text[prev_len:], end="", flush=True)   # 기본 설정에서 text는 '누적' 문자열
        prev_len = len(text)
print(f"\n→ TTFT {first * 1000:.1f} ms")
m = generate(prompt="A graphics processor is", max_new_tokens=64)
itl = sorted(m["itls"])
print(f"→ (common/client.py로 다시 측정) TTFT {m['ttft'] * 1000:.1f} ms, "
      f"ITL p50 {itl[len(itl) // 2] * 1000:.2f} ms, p99 {itl[int(len(itl) * 0.99)] * 1000:.2f} ms")

# 3) OpenAI 호환 API — 기존 OpenAI SDK 코드가 base_url만 바꾸면 그대로 동작
section("3) OpenAI 호환 /v1/chat/completions")
r = requests.post(f"{URL}/v1/chat/completions", json={
    "model": MODEL,
    "messages": [{"role": "user", "content": "In one sentence, what is a KV cache?"}],
    "max_tokens": 64,
    "temperature": 0,
    # Qwen3는 기본으로 <think> 추론 블록을 만든다. 실습에서는 꺼서 짧게.
    "chat_template_kwargs": {"enable_thinking": False},
}).json()
print(r["choices"][0]["message"]["content"])
print("usage:", r.get("usage"))

# 4) 동시 요청 — 8개를 순서대로 vs 한꺼번에
section("4) 8개 요청: 순차 vs 동시")
# 두 실험이 서로의 prefix cache를 재사용하지 않도록 프롬프트를 따로 만든다
reqs_seq = [random_ids(256, seed=100 + i) for i in range(8)]
reqs_par = [random_ids(256, seed=200 + i) for i in range(8)]

t0 = time.perf_counter()
for ids in reqs_seq:
    generate(input_ids=ids, max_new_tokens=128)
seq_t = time.perf_counter() - t0

t0 = time.perf_counter()
with ThreadPoolExecutor(8) as ex:
    list(ex.map(lambda ids: generate(input_ids=ids, max_new_tokens=128), reqs_par))
par_t = time.perf_counter() - t0
print(f"순차: {seq_t:.2f}s   동시: {par_t:.2f}s   → {seq_t / par_t:.1f}배")
print("서버 로그에서 '#running-req: 8' 이 찍힌 Decode batch 줄을 찾아보세요:")
print("  grep 'Decode batch, #running-req: 8,' logs/server_baseline.log | tail -3")
