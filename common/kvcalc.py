"""모델 config.json으로부터 KV cache 크기와 "GPU에 몇 token이 들어가는가"를 계산한다.

    KV bytes / token = 2 (K, V) × num_layers × num_kv_heads × head_dim × dtype_bytes

사용 예:
    python common/kvcalc.py Qwen/Qwen3-1.7B                        # HF hub에서 config만 받아 계산
    python common/kvcalc.py Qwen/Qwen3-1.7B --gpu-gib 24 --mem-fraction 0.70 --params-b 1.72
    python common/kvcalc.py ./my_model_dir --kv-dtype fp8
    python common/kvcalc.py Qwen/Qwen3-1.7B --params-b 1.7          # weight 크기를 직접 지정

주의: 실제 SGLang은 activation, CUDA graph 버퍼 등을 위해 메모리를 추가로 쓰므로
이 계산은 "상한" 추정치입니다. 실제 값은 서버 로그의 `max_total_num_tokens` 또는
`KV Cache is allocated. #tokens: ...` 줄과 비교해 보세요 (Session 3 실습).
"""

from __future__ import annotations

import argparse
import json
import os

DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp8": 1, "fp32": 4, "fp4": 0.5}


def load_config(model: str) -> dict:
    if os.path.isdir(model):
        with open(os.path.join(model, "config.json")) as f:
            return json.load(f)
    if model.endswith(".json") and os.path.exists(model):
        with open(model) as f:
            return json.load(f)
    from huggingface_hub import hf_hub_download  # SGLang 설치 시 함께 설치됨

    path = hf_hub_download(model, "config.json")
    with open(path) as f:
        return json.load(f)


def kv_bytes_per_token(cfg: dict, kv_dtype: str = "bf16") -> float:
    cfg = cfg.get("text_config", cfg)  # 멀티모달 모델 대비
    layers = cfg["num_hidden_layers"]
    q_heads = cfg["num_attention_heads"]
    kv_heads = cfg.get("num_key_value_heads", q_heads)
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // q_heads
    return 2 * layers * kv_heads * head_dim * DTYPE_BYTES[kv_dtype]


def human(nbytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:,.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PiB"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model", help="HF repo id, 로컬 모델 폴더, 또는 config.json 경로")
    p.add_argument("--kv-dtype", default="bf16", choices=list(DTYPE_BYTES))
    p.add_argument("--gpu-gib", type=float, default=24.0, help="GPU 메모리 (RTX 4090 = 24)")
    p.add_argument("--mem-fraction", type=float, default=0.70,
                   help="SGLang --mem-fraction-static (weight + KV pool 비율). "
                        "v0.5.23은 24GB GPU에서 activation/CUDA graph 몫을 빼고 자동으로 약 0.7 근처를 고른다")
    p.add_argument("--params-b", type=float, default=None, help="파라미터 수(십억). 모르면 비워두기")
    p.add_argument("--weight-bytes", type=float, default=2.0, help="파라미터당 byte (bf16=2, fp8=1)")
    args = p.parse_args()

    cfg = load_config(args.model)
    c = cfg.get("text_config", cfg)
    per_tok = kv_bytes_per_token(cfg, args.kv_dtype)
    q_heads = c["num_attention_heads"]
    kv_heads = c.get("num_key_value_heads", q_heads)
    head_dim = c.get("head_dim") or c["hidden_size"] // q_heads

    print(f"model            : {args.model}")
    print(f"layers           : {c['num_hidden_layers']}")
    print(f"q heads / kv heads: {q_heads} / {kv_heads}  (GQA 비율 {q_heads // kv_heads}:1)")
    print(f"head_dim         : {head_dim}")
    print(f"KV dtype         : {args.kv_dtype}")
    print(f"KV per token     : {per_tok:,.0f} B  ({human(per_tok)})")
    if kv_heads != q_heads:
        print(f"  (MHA였다면      : {human(per_tok * q_heads / kv_heads)} / token)")
    print()
    for n_req, ctx in [(1, 1024), (1, 32768), (16, 2048), (64, 2048), (256, 4096)]:
        print(f"  {n_req:>4} req × {ctx:>6} tok  → {human(per_tok * n_req * ctx)}")

    if args.params_b:
        w = args.params_b * 1e9 * args.weight_bytes
        budget = args.gpu_gib * 1024**3 * args.mem_fraction - w
        print()
        print(f"weights          : {human(w)}")
        print(f"KV pool 예산      : {args.gpu_gib} GiB × {args.mem_fraction} − weights = {human(budget)}")
        print(f"→ 최대 약 {int(budget // per_tok):,} tokens를 KV cache에 담을 수 있음 (상한 추정)")


if __name__ == "__main__":
    main()
