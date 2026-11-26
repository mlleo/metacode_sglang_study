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

HTTP 서버에서 `bash common/launch.sh graph_limit --cuda-graph-max-bs-decode 48`로 별도 확인했습니다. `common/loadgen.py`에 `--num 128 --input-len 256 --output-len 256 --concurrency 16,64 --flush --tag graph_limit`를 넘겼습니다. 시작 로그의 capture 목록은 48까지였고, `Decode batch`의 실제 요청 16개에서 `cuda graph: True`, 64개에서 `False`를 봤습니다. 마치면 `bash common/stop.sh`로 서버를 종료합니다. [서빙 로그 발췌](data/s05_http_graph_limit.txt)

## 선택 실습에서 읽을 것

FP8 weight는 로딩 로그의 weight 메모리를 3.28→1.92 GB로 줄였습니다. FP8 KV는 같은 모델 weight에서 `#tokens`를 579,196→1,158,393으로 늘렸습니다. 그런데 동시성 1의 ITL 중앙값은 BF16 2.27 ms, FP8 KV 2.38 ms였습니다. 저장할 수 있는 토큰 수와 한 step의 속도를 따로 봐야 합니다. `Load weight end`와 `KV Cache is allocated`에서 실제 dtype·용량을 확인하고, 부하 CSV의 `itl_ms_p50`과 `output_tok_per_s`를 같은 동시성끼리 비교합니다. [양자화 CSV](data/s05_quant.csv) · [메모리 로그 발췌](data/s05_quant_memory.txt)

NGRAM은 두 서버의 메모리 비율을 0.7, decode graph 상한을 8로 맞췄습니다. 동시성 1에서 `per_req_decode_tok_per_s`는 반복적인 코드 출력에서 444→3,586, 창작 출력에서 447→418이었습니다. `completion_len_mean`은 두 경우 모두 400이었습니다. 먼저 같은 workload와 동시성의 CSV 행을 비교한 뒤, 서버 로그의 `accept len`으로 한 검증 step에 몇 토큰이 나왔는지 봅니다. 반복 구간에서는 15.12, 창작 구간에서는 1.18인 줄이 있었습니다. 이 발췌 값은 각 구간의 예시이며 전체 실행 평균은 아닙니다. 여러 토큰이 한 청크에 오면 클라이언트는 간격을 나눠 기록하므로 체감 출력 간격도 관심사라면 청크 도착 간격을 따로 잽니다. [NGRAM CSV](data/s05_spec.csv) · [서버 로그 발췌](data/s05_spec_excerpt.txt)

attention backend를 바꾼 실행은 입력 512, 출력 64에서 H100의 자동 선택 기준선과 Triton을 비교했습니다. batch 16에서 `prefill_latency_ms`는 42.97→644.44 ms, `median_decode_latency_ms`는 3.008→2.971 ms였습니다. standalone 도구는 각 batch의 prefill을 한 번씩 기록합니다. 첫 실행이나 컴파일의 영향인지 확인하려면 해당 shape를 워밍업한 뒤 prefill을 반복합니다. 0.04 ms 정도인 decode 차이도 이 한 번의 순위로 결론 내리지 않습니다. 로그에 실제 backend를 남기세요. 이 표는 4090의 flashinfer 대 Triton 비교 결과로 사용할 수 없습니다. [backend CSV](data/s05_triton.csv)

## 다른 최적화를 과제로 고른 경우

| 바꿀 조건 | 먼저 읽을 값 |
|---|---|
| graph 또는 overlap에서 모델 크기 | 모델별로 ON/OFF를 따로 묶고, decode 시간·TTFT·ITL과 그래프 capture 시간 또는 처리량을 적습니다. [Overlap 예시](session04.md#선택-실습에서-overlap-scheduling-읽기) |
| FP8 weight의 batch / FP8 KV의 context | weight 비교는 동일한 KV dtype, KV 비교는 동일한 weight를 유지합니다. 용량과 속도를 함께 보고 동일한 프롬프트에서 출력도 확인합니다. |
| Attention backend의 context | backend를 양쪽에 명시하고 각 context에서 워밍업한 prefill과 decode 중앙값을 따로 비교합니다. |
| NGRAM의 동시성 / draft 크기 | 같은 prompt와 출력 조건에서 `accept len`과 요청당 decode tok/s를 함께 기록합니다. draft 크기와 동시성은 하나씩 바꿉니다. |
| `torch.compile`의 batch | 시작 시간, 첫 요청, 워밍업 뒤 step 중앙값을 분리합니다. 같은 batch의 반복 실행에서 컴파일 비용을 회수할 수 있는지 확인합니다. |

위 조건을 모두 조합해 측정한 데이터는 없습니다. 우선 비교점 두 개를 정하고, 제출 노트에는 가설·설정·결과·근거·한계를 한 페이지에 맞춥니다.
