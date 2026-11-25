# 5. CUDA graph 실험 노트

## 질문과 가설

decode graph를 켜면 step마다 여러 CUDA 커널을 다시 요청하는 비용이 줄어듭니다. batch가 작으면 그 비용 비중이 클 것으로 예상했습니다. 입력을 512에서 2,048로 늘리면 KV 읽기 비중이 커지므로, 특히 batch 16에서 graph를 켰을 때의 배율이 줄어들 것으로 예상했습니다.

## 실습과 과제의 측정값

같은 H100과 Qwen3-1.7B BF16에서 출력 64로 고정했습니다. 512 실습은 batch 1·16·64·128, 2,048 과제는 batch 1·16을 쟀습니다. 표의 시간은 워밍업 뒤 decode step 중앙값입니다. 각 조건의 서버 실행은 한 번입니다.

| 입력 | batch | ON (ms) | OFF (ms) | OFF / ON | KV 읽기 추정 비중 |
|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 2.499 | 10.616 | 4.25× | 1.8% |
| 512 | 16 | 3.008 | 11.305 | 3.76× | 22.5% |
| 512 | 64 | 4.386 | 11.305 | 2.58× | 53.7% |
| 512 | 128 | 6.006 | 11.428 | 1.90× | 69.9% |
| 2,048 | 1 | 2.585 | 11.232 | 4.35× | 6.5% |
| 2,048 | 16 | 3.914 | 11.969 | 3.06× | 52.6% |

[요약 CSV](data/s05_onebatch.csv) · [one_batch 원본 JSONL](data/s05_onebatch.jsonl)

## 어떤 값으로 결론을 내렸나

1. `input_len`, `output_len`, `batch_size`가 같은 행을 묶어 `median_decode_latency_ms`의 OFF/ON을 계산했습니다. batch 16에서 3.76→3.06배로 줄었고 batch 1에서는 비슷했습니다. 절대 감소량도 적어 두면 배율이 달라진 이유를 구분할 수 있습니다. batch 16의 감소량은 8.30→8.06 ms였습니다.
2. `kv_share_pct`는 context와 배치에서 추정한 바이트 비율입니다. `bw_util_pct`도 추정 읽기량을 step 시간으로 나눈 값으로, GPU 메모리 컨트롤러의 실측 카운터와 의미가 다릅니다. H100 데이터에는 `onebatch_to_csv.py --peak-gbs 3350`을 사용했습니다.
3. 시작 로그에서 `Capture target decode CUDA graph end. bs=`를 확인합니다. 이 실험의 ON 설정은 batch 128까지 capture했습니다. `one_batch`에 상한 48을 넘겨도 측정 목록의 최대인 128까지 capture했습니다. 실제 서빙에서 상한을 찾을 때는 HTTP 서버의 시작 로그를 보고, 요청 동시성과 매 step의 `#running-req`를 구분합니다.
4. CPU launch 설명을 확인하려면 [Session 6 trace](session06.md)에서 같은 batch의 `cudaGraphLaunch`와 `cudaLaunchKernel`을 봅니다. 그래프 녹화에 드는 시작 시간과 여유 메모리, 작은 1.7B 모델 한 종류로 재었다는 점도 노트 끝에 남깁니다.

## 재현

[실습 1–2](../../session05_execution/lab.md)와 [기본 과제](../../session05_execution/assignment.md)의 비교입니다.
