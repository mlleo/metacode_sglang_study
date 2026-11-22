# 용어집

| 용어 | 한 줄 정의 | 처음 나오는 회차 |
|---|---|---|
| Autoregressive generation | 이전까지의 token을 보고 다음 token 하나를 예측하는 것을 반복하는 생성 방식 | 1 |
| Prefill | 프롬프트 전체를 한 번의 forward로 처리해 KV를 만들고 첫 token을 뽑는 단계. 연산량이 지배적(compute-bound) | 1 |
| Decode | token 하나씩 forward하는 단계. weight·KV 읽기가 지배적(memory-bound) | 1 |
| Extend | SGLang 용어로 "prefill의 일반화": 이미 캐시된 prefix 뒤에 새 token들을 이어 붙여 계산 (`ForwardMode.EXTEND`) | 3, 5 |
| KV cache | 각 layer·token의 Key/Value 벡터를 저장해 두는 메모리. 재계산(O(n²))을 저장(O(n))으로 바꾼다 | 1 |
| GQA | Grouped-Query Attention. 여러 query head가 KV head 하나를 공유 → KV cache 축소 | 1 |
| Arithmetic intensity | FLOPs / 읽은 bytes. GPU의 (peak FLOPS / 대역폭) 보다 작으면 memory-bound | 1 |
| TTFT | Time To First Token. 요청 → 첫 token 도착 | 1 |
| ITL | Inter-Token Latency. 연속한 두 token 사이 간격 (분포로 본다) | 1 |
| TPOT | Time Per Output Token. (E2E − TTFT) / (출력 token 수 − 1) | 1 |
| E2E latency | 요청 → 마지막 token | 1 |
| Throughput | 단위 시간당 처리량 (req/s, output tok/s, total tok/s) | 1 |
| p50 / p95 / p99 | 분포의 50/95/99번째 백분위. tail latency는 p95 이상 | 1 |
| Static batching | batch를 한 번 만들면 전부 끝날 때까지 새 요청을 못 넣는 방식 | 2 |
| Continuous batching | 매 iteration마다 끝난 요청을 빼고 새 요청을 넣는 방식 (iteration-level scheduling) | 2 |
| Closed-loop / Open-loop | 동시성 고정(끝나면 다음 요청) / 도착률 고정(서버 상태와 무관하게 도착) 부하 | 2 |
| Little's law | 평균 동시 요청 수 = 도착률 × 평균 체류 시간 (L = λW) | 2 |
| SLO / Goodput | 서비스 목표(예: p95 TTFT < 500 ms) / SLO를 만족한 요청만 센 throughput | 2, 6 |
| TokenizerManager / Scheduler / DetokenizerManager | SGLang의 3개 핵심 프로세스 (토크나이즈 / batch 결정·GPU 실행 / 텍스트 복원) | 2 |
| ReqToTokenPool | 요청 × 위치 → KV 슬롯 번호를 담는 표 | 3 |
| TokenToKVPool | 실제 K/V 텐서가 사는 GPU 메모리 풀 (슬롯 번호로 접근) | 3 |
| Page size | KV를 할당·매칭하는 단위 token 수 (기본 1) | 3 |
| Prefix caching | 같은 prefix의 KV를 요청 간에 재사용하는 것 | 3 |
| Radix tree | 공통 prefix를 공유하는 압축 trie. SGLang은 token 시퀀스를 key로 KV 슬롯을 저장 | 3 |
| RadixAttention | radix tree 기반 prefix 캐시 + 그 KV를 읽는 attention의 조합을 부르는 SGLang의 이름 | 3 |
| Lock ref | 실행 중 요청이 쓰는 노드를 eviction에서 보호하는 참조 카운트 | 3 |
| LRU eviction | 메모리가 모자라면 오래 안 쓴 leaf부터 지운다 | 3 |
| LPM | Longest Prefix Match 스케줄링: 캐시 hit가 긴 요청부터 처리 (v0.5.23 기본값은 `fcfs`) | 3 |
| Admission control | 이번 iteration에 어떤 대기 요청을 prefill에 넣을지 예산으로 결정 (`PrefillAdder`) | 4 |
| new_token_ratio | 앞으로 생성될 token을 `max_new_tokens × ratio`만큼만 예약하는 낙관적 추정치 | 4 |
| Retraction | decode 중 KV가 부족하면 실행 중 요청을 쫓아내 대기열로 되돌리는 것 (나중에 다시 prefill) | 4 |
| Chunked prefill | 긴 prefill을 N token 조각으로 나눠 여러 iteration에 걸쳐 처리 | 4 |
| Head-of-line blocking | 앞의 큰 작업 하나 때문에 뒤의 작은 작업들이 기다리는 현상 | 4 |
| Overlap scheduling | GPU가 batch N을 실행하는 동안 CPU가 batch N+1을 준비하는 파이프라이닝 | 4 |
| ForwardBatch | GPU 실행용으로 텐서화된 batch (`input_ids`, `positions`, `out_cache_loc` ...) | 5 |
| Attention backend | attention 커널 구현 선택 (FlashInfer, Triton, FA3 ...) | 5 |
| CUDA graph | 커널 launch 시퀀스를 녹화해 한 번에 재생 → CPU launch overhead 제거 | 5 |
| Quantization (W8 / KV FP8) | weight 또는 KV를 8bit로 저장 → 읽을 bytes 감소 | 5 |
| Speculative decoding | 싸게 여러 token을 추측(draft)하고 큰 모델이 한 번에 검증(verify) | 5 |
| Accept length | speculative decoding에서 verify 한 번에 평균 몇 token이 확정되는가 | 5 |
| Profiling trace | 커널·CPU 작업 타임라인 (Perfetto / chrome://tracing에서 연다) | 6 |
| PD disaggregation | prefill과 decode를 다른 GPU(서버)로 분리하는 배치 | 6 |
| HiCache | GPU → CPU 메모리 → 디스크로 이어지는 계층형 KV 캐시 | 6 |
