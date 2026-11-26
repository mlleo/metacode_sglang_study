"""SGLang 서버에 요청을 보내고 TTFT / ITL / E2E를 측정하는 공통 클라이언트.

모든 회차 실습이 이 파일을 재사용합니다. 의존성은 `requests` 하나뿐입니다.

사용 예:
    python common/client.py --prompt "The capital of France is" --max-new-tokens 32
    python common/client.py --input-len 1024 --max-new-tokens 64      # 랜덤 token id 1024개
    python common/client.py --flush                                   # radix cache 비우기

라이브러리로 사용:
    import sys; sys.path.insert(0, "common")
    from client import generate, flush_cache, random_ids
    r = generate(input_ids=random_ids(512, seed=0), max_new_tokens=64)
    print(r["ttft"], r["cached_tokens"])
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional

import requests

DEFAULT_URL = os.environ.get("SGLANG_URL", f"http://127.0.0.1:{os.environ.get('PORT', '30000')}")

# Qwen3 vocab은 151,936개. 특수 토큰(151643 이상)을 피하고, 너무 작은 id(바이트 토큰)도 피한다.
_SAFE_ID_LOW, _SAFE_ID_HIGH = 1_000, 100_000


def random_ids(n: int, seed: Optional[int] = None) -> List[int]:
    """길이가 정확히 n인 랜덤 token id 리스트.

    텍스트 대신 token id를 보내면 '프롬프트 길이'를 토크나이저와 무관하게 정확히 통제할 수 있다.
    같은 seed → 같은 시퀀스이므로 prefix 공유 실험에 쓰기 좋다.
    """
    rng = random.Random(seed)
    return [rng.randrange(_SAFE_ID_LOW, _SAFE_ID_HIGH) for _ in range(n)]


def _payload(
    prompt: Optional[str],
    input_ids: Optional[List[int]],
    max_new_tokens: int,
    temperature: float,
    ignore_eos: bool,
    stream: bool,
    extra_sampling: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if (prompt is None) == (input_ids is None):
        raise ValueError("prompt 또는 input_ids 중 정확히 하나만 지정하세요.")
    sampling = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "ignore_eos": ignore_eos,
    }
    if extra_sampling:
        sampling.update(extra_sampling)
    body: Dict[str, Any] = {"sampling_params": sampling, "stream": stream}
    if prompt is not None:
        body["text"] = prompt
    else:
        body["input_ids"] = input_ids
    return body


def generate(
    prompt: Optional[str] = None,
    input_ids: Optional[List[int]] = None,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    ignore_eos: bool = True,
    stream: bool = True,
    url: str = DEFAULT_URL,
    timeout: float = 600.0,
    extra_sampling: Optional[Dict[str, Any]] = None,
    on_first_token: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """SGLang native `/generate`를 호출하고 지연시간 지표를 돌려준다.

    on_first_token은 첫 토큰이 담긴 스트리밍 청크를 받았을 때 한 번 호출합니다.

    반환 dict:
        ttft            첫 token이 클라이언트에 도착할 때까지 (s). stream=False면 None
        itls            token 사이 간격 리스트 (s). 청크 하나에 k개 token이 오면 간격을 k로 나눠 k번 기록
        e2e             요청 전체 (s)
        prompt_tokens / completion_tokens / cached_tokens   서버 meta_info 값
        text            생성된 텍스트 (누적)
        t_start         요청 시작 시각 (time.perf_counter 기준, 타임라인 그릴 때 사용)
        stream_done     스트림에서 [DONE]을 받았는지 (stream=True일 때)
        token_times     청크가 도착한 시각들 (perf_counter 기준, stream=True일 때만). Session 4 ITL 타임라인용
    """
    body = _payload(prompt, input_ids, max_new_tokens, temperature, ignore_eos, stream, extra_sampling)
    t0 = time.perf_counter()

    if not stream:
        r = requests.post(f"{url}/generate", json=body, timeout=timeout)
        r.raise_for_status()
        out = r.json()
        e2e = time.perf_counter() - t0
        meta = out.get("meta_info", {})
        return {
            "ttft": None,
            "itls": [],
            "e2e": e2e,
            "prompt_tokens": meta.get("prompt_tokens"),
            "completion_tokens": meta.get("completion_tokens"),
            "cached_tokens": meta.get("cached_tokens", 0),
            "text": out.get("text", ""),
            "t_start": t0,
            "token_times": [],
        }

    ttft = None
    itls: List[float] = []
    token_times: List[float] = []
    last_t = None
    last_n = 0
    meta: Dict[str, Any] = {}
    text = ""
    stream_done = False
    with requests.post(f"{url}/generate", json=body, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                stream_done = True
                break
            chunk = json.loads(data)
            now = time.perf_counter()
            meta = chunk.get("meta_info", meta)
            text = chunk.get("text", text)
            n = meta.get("completion_tokens", last_n + 1)
            new_tokens = max(n - last_n, 0)
            if new_tokens == 0:
                continue  # 토큰 없이 meta만 갱신된 청크
            token_times.append(now)
            if ttft is None:
                ttft = now - t0
                if on_first_token is not None:
                    on_first_token()
                # 첫 청크에 token이 여러 개 실려 오면 나머지는 ITL 0으로 취급하지 않고 버린다.
            else:
                gap = (now - last_t) / new_tokens
                itls.extend([gap] * new_tokens)
            last_t, last_n = now, n
    e2e = time.perf_counter() - t0
    return {
        "ttft": ttft,
        "itls": itls,
        "e2e": e2e,
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": meta.get("completion_tokens", last_n),
        "cached_tokens": meta.get("cached_tokens", 0),
        "text": text,
        "t_start": t0,
        "token_times": token_times,
        "stream_done": stream_done,
    }


def flush_cache(url: str = DEFAULT_URL, wait_s: float = 30.0) -> bool:
    """radix cache를 비운다.

    SGLang은 실행 중이거나 대기 중인 요청이 있으면 flush를 거절한다(HTTP 400).
    직전 요청의 스트림이 끝났어도 서버 안에서 정리 중일 수 있으므로, `timeout` 인자로
    "서버가 한가해질 때까지 최대 wait_s초 기다렸다가 flush"하게 한다 (v0.5.23 flush_wrapper).
    실패하면 경고를 출력한다 — cold 측정이 몰래 warm이 되는 것을 막기 위해.
    """
    try:
        r = requests.post(f"{url}/flush_cache", params={"timeout": wait_s}, timeout=wait_s + 10)
        ok = r.status_code == 200
    except requests.RequestException as e:
        ok, r = False, e
    if not ok:
        import sys
        print(f"[경고] flush_cache 실패: {getattr(r, 'text', r)!r}. 이후 측정에 cache hit가 섞일 수 있습니다.",
              file=sys.stderr)
    return ok


def server_info(url: str = DEFAULT_URL) -> Dict[str, Any]:
    """`/server_info` (구버전은 `/get_server_info`)."""
    for path in ("/server_info", "/get_server_info"):
        try:
            r = requests.get(f"{url}{path}", timeout=10)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
    raise RuntimeError("server_info를 가져오지 못했습니다. 서버가 떠 있나요?")


def wait_ready(url: str = DEFAULT_URL, timeout: float = 900.0) -> None:
    """`/health`가 200을 돌려줄 때까지 기다린다 (모델 로딩 + CUDA graph capture 포함)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(f"{url}/health", timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"{timeout}s 안에 서버가 준비되지 않았습니다.")


def _main() -> None:
    p = argparse.ArgumentParser(description="SGLang /generate 단일 요청 측정")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--prompt", default=None)
    p.add_argument("--input-len", type=int, default=None, help="랜덤 token id 프롬프트 길이")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--no-stream", action="store_true")
    p.add_argument("--allow-eos", action="store_true", help="EOS에서 멈추도록 허용 (기본: ignore_eos)")
    p.add_argument("--flush", action="store_true", help="radix cache만 비우고 종료")
    args = p.parse_args()

    if args.flush:
        print("flush_cache:", flush_cache(args.url))
        return

    if args.prompt is None and args.input_len is None:
        args.prompt = "The capital of France is"
    ids = random_ids(args.input_len, args.seed) if args.input_len else None
    r = generate(
        prompt=args.prompt if ids is None else None,
        input_ids=ids,
        max_new_tokens=args.max_new_tokens,
        stream=not args.no_stream,
        ignore_eos=not args.allow_eos,
        url=args.url,
    )
    itl_ms = sorted(x * 1000 for x in r["itls"])
    summary = {
        "prompt_tokens": r["prompt_tokens"],
        "cached_tokens": r["cached_tokens"],
        "completion_tokens": r["completion_tokens"],
        "ttft_ms": None if r["ttft"] is None else round(r["ttft"] * 1000, 2),
        "itl_ms_median": round(itl_ms[len(itl_ms) // 2], 2) if itl_ms else None,
        "e2e_ms": round(r["e2e"] * 1000, 2),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.prompt is not None:
        print("--- text ---")
        print(r["text"])


if __name__ == "__main__":
    _main()
