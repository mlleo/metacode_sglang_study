"""radix cache가 '무엇을' 재사용하는지 `meta_info.cached_tokens`로 하나씩 확인한다.

token id를 직접 보내므로 공유 길이가 정확합니다 (토크나이저 경계 효과 없음).

    bash common/launch.sh baseline
    python session03_kvcache_radix/code/prefix_basics.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from client import DEFAULT_URL, flush_cache, generate, random_ids  # noqa: E402

import requests  # noqa: E402

P = random_ids(1000, seed=1)       # 공통 system prompt 역할 (1000 token)
Q1 = random_ids(50, seed=2)
Q2 = random_ids(50, seed=3)
OTHER = random_ids(1050, seed=4)   # 전혀 다른 프롬프트


def ask(label, ids, max_new_tokens=16, expect=""):
    r = generate(input_ids=ids, max_new_tokens=max_new_tokens)
    print(f"{label:<34} prompt={r['prompt_tokens']:>5}  cached={r['cached_tokens']:>5}  "
          f"TTFT={r['ttft'] * 1000:7.1f} ms   예상: {expect}")
    return r


flush_cache()
print("flush 완료 — tree가 비었습니다\n")
r1 = ask("1) P + Q1  (처음)", P + Q1, expect="0 — 빈 tree")
ask("2) P + Q2", P + Q2, expect="≈1000 — P까지 공유, Q에서 갈라짐(split)")
ask("3) P + Q1  (완전히 같은 요청)", P + Q1, expect="1049 — 마지막 1 token은 항상 다시 계산")
ask("4) P 만", P, expect="999 — 역시 마지막 1 token 제외")
ask("5) OTHER", OTHER, expect="0 — 공유 prefix 없음")

# 6) 생성된 token도 tree에 남는가? — P+Q1에 대한 '출력 token id'를 받아서 이어 붙여 보낸다
#    (non-stream 응답에는 output_ids가 들어 있다. temperature=0이라 1번과 같은 출력)
resp = requests.post(f"{DEFAULT_URL}/generate", json={
    "input_ids": P + Q1, "sampling_params": {"max_new_tokens": 16, "temperature": 0, "ignore_eos": True},
}).json()
gen_ids = resp["output_ids"]
ask("6) P + Q1 + (1번의 출력 16개) + Q2", P + Q1 + gen_ids + Q2,
    expect="≈1065 — 출력 token의 KV도 tree에 남아 있음")

print("""
확인할 것
  - 2번: cached가 정확히 1000인가? 그렇다면 radix tree가 P 끝에서 노드를 split한 것
  - 3·4번: 왜 '전체 길이 - 1'일까?  (힌트: 마지막 token의 logits가 있어야 다음 token을 뽑을 수 있다
           → schedule_batch.py `_compute_max_prefix_len`: max_prefix_len = input_len - 1)
  - 6번: 멀티턴 대화에서 '이전 답변'까지 재사용된다는 뜻. 단, 텍스트로 다시 토크나이즈하면 깨질 수 있다 (multiturn.py)
  - TTFT: cached가 클수록 줄어드나요? 줄어든 비율이 (1 - cached/prompt)와 비슷한가요?
""")
