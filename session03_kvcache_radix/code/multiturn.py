"""멀티턴 대화에서 radix cache가 '이전 대화 전체'를 재사용하는지 확인한다.

두 가지 방식으로 같은 대화를 6턴 진행하고 턴마다 cached_tokens를 비교한다.
    ids  : 이전 프롬프트 token + 모델이 실제로 낸 output_ids를 그대로 이어 붙임 (이상적)
    text : 이전 대화를 '텍스트'로 이어 붙인 뒤 서버가 다시 토크나이즈 (흔한 구현)

    python session03_kvcache_radix/code/multiturn.py
"""

from __future__ import annotations

import os
import sys

import requests
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from client import DEFAULT_URL, flush_cache  # noqa: E402

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-1.7B")
SYSTEM = "<|im_start|>system\nYou are a concise assistant. Answer in 2-3 sentences.<|im_end|>\n"
QUESTIONS = [
    "What is a KV cache?",
    "Why does it grow with context length?",
    "How does prefix caching help a chatbot?",
    "What is a radix tree?",
    "What happens when GPU memory runs out?",
    "Summarize our conversation in one sentence.",
]


def user_turn(q: str) -> str:
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"


def call(payload: dict) -> dict:
    payload["sampling_params"] = {"max_new_tokens": 80, "temperature": 0}
    r = requests.post(f"{DEFAULT_URL}/generate", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)

    print("== 방식 ids: token 그대로 이어 붙이기 ==")
    flush_cache()
    ids = tok.encode(SYSTEM, add_special_tokens=False)
    for i, q in enumerate(QUESTIONS, 1):
        ids += tok.encode(user_turn(q), add_special_tokens=False)
        out = call({"input_ids": ids})
        m = out["meta_info"]
        print(f"turn {i}: prompt={m['prompt_tokens']:>5} cached={m['cached_tokens']:>5} "
              f"hit={m['cached_tokens'] / m['prompt_tokens']:.0%}")
        ids += out["output_ids"] + tok.encode("\n", add_special_tokens=False)

    print("\n== 방식 text: 텍스트로 이어 붙이고 서버가 다시 토크나이즈 ==")
    flush_cache()
    text = SYSTEM
    for i, q in enumerate(QUESTIONS, 1):
        text += user_turn(q)
        out = call({"text": text})
        m = out["meta_info"]
        print(f"turn {i}: prompt={m['prompt_tokens']:>5} cached={m['cached_tokens']:>5} "
              f"hit={m['cached_tokens'] / m['prompt_tokens']:.0%}")
        # 서버는 기본으로 special token(<|im_end|>)을 지운 텍스트를 돌려준다 → 우리가 다시 붙인다
        text += out["text"].strip() + "<|im_end|>\n"

    print("""
비교할 것
  - ids 방식: turn k의 cached ≈ (turn k의 prompt) − (새 user 질문 길이) 인가?
  - text 방식: 어느 턴에서 cached가 기대보다 작아지나? 이전 답변의 '어느 지점'에서 매칭이 끊겼을까?
    (힌트: 모델이 낸 token 시퀀스 → 텍스트 → 다시 토크나이즈 했을 때 같은 token 시퀀스가 나온다는 보장이 없다.
           공백/줄바꿈 처리, .strip(), special token 제거, chat template의 <think> 블록 제거 등)
  - 실제 서비스에서 OpenAI 호환 /v1/chat/completions를 쓰면 매 턴 chat template이 '전체 대화'를 다시 렌더링한다.
    어떤 template은 이전 턴의 일부를 바꿔 렌더링해서 cache hit를 깨뜨린다 (과제 Part C 선택 실험).
""")


if __name__ == "__main__":
    main()
