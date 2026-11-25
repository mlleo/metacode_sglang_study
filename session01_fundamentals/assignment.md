# Session 1 과제 — "숫자로 말하는 추론의 기초"

> 마감: Session 2 시작 24시간 전 PR · GPU 예산 **$0.9 이내** (Part C만 GPU 필요)
> 제출: `submissions/session01/<github-id>/README.md` ([templates/report_template.md](../templates/report_template.md) 기반)

## Part A. KV cache 손계산 (GPU 불필요, 30분)

아래 모델의 `config.json`을 Hugging Face에서 열어 값을 직접 읽고 표를 채우세요.
`common/kvcalc.py`로 검산해도 좋지만 **공식에 대입한 식을 README에 남기세요.**

| 모델 | layers | q heads | kv heads | head_dim | KV / token (BF16) | weights (BF16) |
|---|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen3-1.7B` | | | | | | |
| `Qwen/Qwen3-8B` | | | | | | |
| `meta-llama/Llama-3.1-8B-Instruct` | | | | | | |

질문 (각 2–3문장):

1. RTX 4090(24 GB)에서 `mem_fraction_static = 0.7`이라고 할 때, 각 모델의 KV pool에는 몇 token이 들어가는가?
   그 pool로 **context 4K짜리 사용자를 동시에 몇 명** 받을 수 있는가?
2. 8B 모델을 BF16으로 4090에 올리면 무슨 일이 생기는가? 이 문제를 풀 수 있는 방법 두 가지를 추측해 보세요. (Session 5에서 하나를 직접 실험합니다.)
3. Qwen3-8B가 MHA(= kv heads = q heads)였다면 1번 답은 어떻게 바뀌는가?

## Part B. Roofline 손계산 (GPU 불필요, 20분)

RTX 4090 기준값: BF16 ≈ 165 TFLOPS, 메모리 대역폭 ≈ 1.0 TB/s

1. Qwen3-1.7B의 batch=1 decode step 이론 하한(ms)과, 그때의 최대 token/s를 계산하세요.
2. linear layer만 생각할 때, batch가 몇일 때부터 decode가 compute-bound가 되는가? (AI ≈ B를 ridge와 비교)
3. 그런데 context가 2K인 요청 B개를 decode하면 KV 읽기 bytes는 얼마인가? **B = 8, 32, 128** 각각에 대해 weight bytes와 KV bytes를 나란히 적고,
   "batch를 키우면 decode가 compute-bound가 된다"는 2번 답이 현실에서 왜 잘 안 맞는지 설명하세요.

## Part C. 측정 (GPU 필요, $0.9 이내)

수업에서 저장한 `s01_manual.csv`, `s01_sglang.csv`와 함께 아래 한 가지를 측정합니다. `s01_hf.csv`가 있으면 추가로 볼 수 있습니다.

**추가 측정: SGLang에서 입력 길이 → TTFT**

```bash
bash common/launch.sh baseline
for n in 128 512 2048; do
  python3 common/loadgen.py --num 3 --concurrency 1 --flush --input-len "$n" --output-len 8 \
    --tag "s01_${n}" --out results/s01_ttft_vs_inlen.csv
done
bash common/stop.sh
```

시간이 더 있으면 4K와 8K를 추가하거나 같은 길이를 반복해 중앙값을 구하세요.

제출할 그래프 (로컬에서 `common/plot.py`로):

1. **batch → 처리량** · HF manual의 decode 처리량과 SGLang의 요청 전체 처리량을 각각 표시합니다. 둘 다 출력 64 token으로 맞춥니다. prefill을 포함한 두 흐름을 비교하려면 수업의 선택 실습 2인 HF `generate()` 결과와 SGLang 결과를 사용합니다. 서버 측정에는 HTTP 시간이 포함됩니다.
2. **입력 길이 → 응답 시작 시간** · HF manual의 prefill 시간과 SGLang의 클라이언트 TTFT를 두 선으로 그리고 각각의 측정 범위를 표시합니다.

## Part D. 해석 (README 본문)

1. 4문장 (I expected / I changed / I measured / I found) — Part C 그래프 1번에 대해
2. "같은 GPU, 같은 weight인데 SGLang이 HF보다 빠른 이유" 가설 **3개**를 적고, 오늘 데이터로 **가장 그럴듯한 것 하나**를 고른 근거를 쓰세요.
   (정답을 맞히는 게 목적이 아닙니다. Session 2, 5에서 하나씩 검증합니다.)
3. 그래프 2번의 세 점에서 입력이 길어지면 무엇이 보이나요? 4K와 8K까지 측정했다면 기울기가 바뀌는지도 살펴보세요.

## 채점 기준 (10점)

| 항목 | 점수 | 기준 |
|---|---:|---|
| Part A 정확성 | 2 | 공식 대입 과정이 보이고 숫자가 맞음 |
| Part B 추론 | 2 | KV 읽기가 batch 효과를 상쇄한다는 점을 숫자로 보임 |
| Part C 재현성 | 2 | 실행 명령·환경(`env.txt`)·원시 CSV 포함, 다른 사람이 그대로 재실행 가능 |
| Part D 해석 | 3 | 예측과 결과의 차이를 lecture의 개념으로 설명 |
| GPU 사용 기록 | 1 | 시간·비용 기록, 예산 이내 |

## 다음 회차 발표 (2–3명 × 5분)

[templates/presentation_template.md](../templates/presentation_template.md) 형식. 특히 **Part D-2의 가설**을 발표해 주세요 — 서로 다른 가설이 나올수록 좋습니다.
