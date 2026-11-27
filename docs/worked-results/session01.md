# 1. 첫 요청과 추론 시간

## 실습에서 확인한 값

수동 HF 실험은 입력 512, 출력 64로 맞췄습니다. 첫 출력은 prefill에서 나오고, 이어서 63번의 decode forward를 잽니다. SGLang에는 같은 길이의 요청을 보냈습니다. native 응답에 `completion_tokens`가 있었고, streaming 응답에서는 TTFT와 ITL을 따로 쟀습니다. OpenAI 호환 요청은 `choices`와 `usage`를 반환했습니다.

| batch | HF decode 중앙값 (ms/step) | HF decode 처리량 (tok/s) |
|---:|---:|---:|
| 1 | 41.62 | 24.8 |
| 8 | 41.74 | 196.8 |
| 32 | 42.22 | 768.2 |

한 step의 시간이 거의 같은데 처리량이 늘었습니다. batch 32는 한 step마다 최대 32개 요청의 다음 토큰을 함께 만듭니다. SGLang에서 잰 처리량은 HTTP 요청 전체 시간을 분모로 씁니다. 두 숫자에는 다른 시간 구간이 포함되므로 성능 배율을 해석할 때 먼저 범위를 적었습니다.

## 과제 예시

| 모델 | layers | Q / KV heads | head dim | BF16 KV (KiB/token) | BF16 weight (GiB) | 4,096-token 요청 수 상한 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 28 | 16 / 8 | 128 | 112 | 3.204 | 31 |
| Qwen3-8B | 36 | 32 / 8 | 128 | 144 | 15.274 | 2 |
| Qwen2.5-7B-Instruct | 28 | 28 / 4 | 128 | 56 | 14.175 | 12 |

Qwen3-1.7B 예시 계산은 `2 × 28 × 8 × 128 × 2 = 114688 B/token`입니다. KV 예산을 `24 GiB × 0.7 − 3.44 GB`로 놓고, 단위를 맞춰 계산하면 약 127,291 token입니다. 실행할 때는 activation과 런타임 버퍼, scheduler의 제한도 필요합니다. Qwen3-8B가 MHA였다면 같은 예산에 2,778 token 정도만 들어갑니다.

4090의 대역폭을 1 TB/s로 근사하면 weight 3.44 GB를 읽는 데 3.44 ms가 필요합니다. 연산 165 TFLOPS를 쓰는 단순 모델의 교점은 batch 약 165입니다. 문맥이 2,048 token이면 BF16 KV 읽기는 batch 8·32·128에서 각각 약 1.88, 7.52, 30.06 GB로 커집니다. weight 크기만 넣은 교점이 실제 실험을 설명하지 못할 수 있습니다.

SGLang TTFT 중앙값은 입력 128·512·2,048에서 7.57·7.59·14.57 ms였습니다. 128에서 512로 늘어난 구간은 차이가 작습니다. 입력 길이별 요청이 세 개뿐이라 작은 차이까지 해석하려면 측정을 반복합니다.

[배치별 그래프 입력](data/s01_plot_batch.csv) · [입력 길이별 그래프 입력](data/s01_plot_input.csv) · [HF 수동 CSV](data/s01_manual.csv) · [HF generate CSV](data/s01_hf.csv) · [SGLang CSV](data/s01_sglang.csv) · [TTFT CSV](data/s01_ttft_vs_inlen.csv)

4090의 단순 대역폭 모델에서 3.44 ms/step의 역수를 취하면 batch 1의 상한은 약 291 tok/s입니다. 같은 예산에서 BF16 Qwen3-8B는 4,096-token 요청 두 개를 담는 것으로 계산됐습니다. 실제로 필요한 런타임 여유를 빼면 용량은 더 작아질 수 있습니다. 모델이 맞지만 KV가 좁으면 context나 동시성을 줄이고, weight 여유가 부족하면 더 작은 모델이나 weight 양자화를 검토합니다.

### 과제 D 보고서 문장 예시

“배치가 커지면 step 시간도 크게 늘 것으로 예상했다. 입력 512와 출력 64를 고정하고 batch만 1에서 32로 늘렸다. HF 수동 decode의 step 중앙값은 41.62→42.22 ms, 처리량은 24.8→768.2 tok/s였다. 한 번의 forward로 더 많은 요청의 다음 토큰을 만들면서 처리량이 약 31배가 됐다.”

엔진 차이의 후보로 Python·CUDA launch 비용, attention backend, 요청을 묶는 스케줄링을 적을 수 있습니다. 이 데이터에서 batch 크기와 함께 처리량이 크게 변했으므로 먼저 step마다 처리한 요청 수를 맞춰 보겠습니다. 같은 batch에서 CUDA graph를 끄고 launch가 늘어나는지는 [Session 5](session05.md)와 [Session 6](session06.md)의 비교로 따로 확인합니다.

## 재현

[실습 1–4](../../session01_fundamentals/lab.md)와 [과제 Part C](../../session01_fundamentals/assignment.md)를 실행했습니다. H100의 수동 실험은 `manual_decode.py --bw-gbs 3350`을 사용했습니다.

## 어떤 값으로 결론을 내렸나

1. [HF 수동 CSV](data/s01_manual.csv)의 `experiment=batch`만 골라 `batch`, `decode_step_ms_p50`, `decode_tok_per_s`를 나란히 봅니다. batch 1→32에서 step 중앙값이 41.62→42.22 ms인데 처리량은 약 31배입니다. 여기서 “한 step에 더 많은 토큰을 만들었다”는 결론을 냅니다.
2. 같은 길이에서 엔진 전체 시간까지 비교하려면 [HF generate CSV](data/s01_hf.csv)와 [SGLang CSV](data/s01_sglang.csv)의 `e2e_s`와 `output_tok_per_s`를 봅니다. 두 표가 출력 64개를 썼는지 먼저 확인합니다. 수동 HF의 forward만 잰 값은 단계 구분용으로 따로 표시했습니다.
3. 입력 길이 효과는 HF의 `experiment=inlen`의 `prefill_ms`와 SGLang의 `ttft_ms_p50`을 분리해 그립니다. TTFT에는 요청 준비, 대기, prefill, 전송이 들어갑니다. 작은 차이가 반복되는지 먼저 확인하고, 병목을 더 좁히려면 Session 6에서 step과 GPU 사이의 빈 구간을 봅니다.
