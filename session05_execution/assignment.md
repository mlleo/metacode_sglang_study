# Session 5 과제 — 1-page Optimization Note

> 마감: Session 6 시작 24시간 전 PR · GPU 예산 **$1.5 이내**
> 제출: `submissions/session05/<github-id>/README.md` (**1페이지** — 그래프 1–2개 + 짧은 글)
> 이 노트는 최종 프로젝트의 씨앗이 됩니다.

## 과제

아래 최적화 중 **하나**를 골라, "**언제 효과가 있고 언제 사라지는가**"를 하나의 축으로 측정하세요.
기본 과제는 수업의 CUDA graph ON/OFF 결과를 재사용하고, 새 입력 길이 하나에서 같은 batch 두 개를 두 설정으로 비교하는 것입니다. 이 경로로 만점을 받을 수 있습니다. 다른 최적화는 같은 질문을 다룰 수 있을 때 골라 주세요.

```bash
BS="1 16" IN=2048 bash session05_execution/code/onebatch_sweep.sh base_2k
BS="1 16" IN=2048 bash session05_execution/code/onebatch_sweep.sh nograph_2k --cuda-graph-backend-decode disabled
python3 session05_execution/code/onebatch_to_csv.py results/s05_onebatch.jsonl
```

수업 결과 파일은 로컬에서 새 결과와 합쳐 비교합니다. 두 입력 길이에서 batch별 ON/OFF 차이가 달라졌는지 보고, 차이가 작아도 그대로 기록합니다. 한 번 실행한 경우에는 워밍업 뒤 여러 decode step의 중앙값을 썼다고 적으세요. 차이가 작고 예산이 남으면 두 조건을 한 번 더 실행해 순위가 유지되는지 확인합니다.

| 최적화 | 비교 | 바꿔 볼 축 (하나 선택) |
|---|---|---|
| CUDA graph | decode graph ON vs OFF | batch 크기 / 모델 크기 (0.6B vs 1.7B vs 4B) |
| FP8 weight | BF16 vs `--quantization fp8` | batch 크기 (1 → 128): 어디서 이득이 사라지나 |
| FP8 KV cache | BF16 vs `fp8_e4m3` | context 길이 (512 → 8K): KV 읽기 비중이 커질수록? / 최대 동시성 |
| Attention backend | flashinfer vs triton | context 길이 |
| Speculative (NGRAM) | off vs NGRAM | 동시성 1 → 32 / `--speculative-num-draft-tokens` 4·8·16 |
| Overlap scheduling | ON vs `--disable-overlap-schedule` | 모델 크기 / batch 크기 |
| torch.compile (도전) | off vs `--enable-torch-compile` | batch 크기 (시작 시간도 기록!) |

## 노트 형식 (이 순서, 1페이지)

```markdown
# <최적화 이름>: <한 줄 결론>
예) FP8 weight: batch ≤ 16에서는 decode가 1.7배 빨라지지만, batch 128에서는 1.1배로 줄어든다

## Hypothesis
§3 식의 어느 항을 줄이는가? 그래서 어떤 조건에서 효과가 크고, 어떤 조건에서 사라질 것이라 예측하는가? (숫자로)

## Setup
GPU / 모델 / 고정한 것 / 바꾼 것 / 반복 횟수 / 실행 명령 (한 블록)

## Result
그래프 1개 (x = 바꾼 축, y = 효과 배율 또는 지표, 두 선) + 핵심 숫자 표 (3–5행)

## Why?
결과를 §3 식으로 설명. 예측이 틀렸다면 무엇을 놓쳤는가.
가능하면 근거 하나 더: 로그(`cuda graph: True/False`, `accept len`), KV #tokens, 프로파일 등

## Cost & Limits
이 최적화의 대가 (정확도, 메모리, 시작 시간, 하드웨어 제약), 그리고 이 실험으로 말할 수 없는 것
```

## 도구

- 서버 없이 step 시간: `bash session05_execution/code/onebatch_sweep.sh <tag> [인자]` (환경 변수 `BS`, `IN`, `OUT`로 sweep 범위 지정)
- 서버 부하: `common/loadgen.py` 또는 `session02_architecture_batching/code/sweep_concurrency.sh`
- 다른 모델: `MODEL=Qwen/Qwen3-4B bash ...` (4B BF16 ≈ 8 GB, 4090에 충분히 들어감)

## 채점 기준 (10점)

| 항목 | 점수 |
|---|---:|
| Hypothesis가 §3 식과 연결되고 숫자로 예측됨 | 2 |
| 공정한 비교 (바꾼 변수, warm-up, 실행 횟수 기록) | 2 |
| "축에 따라 효과가 어떻게 달라졌는지"을 데이터로 보임 | 3 |
| Why — 예측과의 차이를 설명 | 2 |
| 1페이지 준수 + GPU 기록 | 1 |

## 다음 회차 발표

서로 다른 최적화를 고른 3명 (각 4분). 발표 후 "6개 최적화를 §3 식 위에 배치하기"를 반 전체가 함께 합니다.
