# SGLang v0.5.23 코드 리딩 지도

SGLang은 릴리스마다 파일이 쪼개지고 이름이 바뀝니다. 이 지도는 **v0.5.23 태그** 기준입니다.
라인 번호는 적지 않습니다 (금방 틀려지므로). 대신 **파일 + 심볼 이름**으로 찾으세요.

```bash
# 로컬에서 (GPU 필요 없음)
git clone --depth 1 -b v0.5.23 https://github.com/sgl-project/sglang.git
cd sglang/python/sglang/srt
grep -n "def get_next_batch_to_run" managers/scheduler.py
```

> 💡 **읽는 순서 원칙**: 전체를 읽지 않습니다. "요청 하나"를 따라가며 함수 이름만 연결하고,
> 그 회차의 질문에 답하는 데 필요한 함수 2–3개만 본문을 읽습니다.

---

## 0. 전체 구조 — 프로세스 3개 + ZMQ

```text
             HTTP (FastAPI, entrypoints/http_server.py)
                          │  /generate, /v1/chat/completions
                          ▼
┌──────────────────────────────────────────┐
│ TokenizerManager   (메인 프로세스, asyncio)  │  managers/tokenizer_manager.py
│  - 텍스트 → token id, 요청 상태(ReqState) 관리 │
└───────────────┬──────────────────────────┘
                │ ZMQ (TokenizedGenerateReqInput)
                ▼
┌──────────────────────────────────────────┐
│ Scheduler   (프로세스 "sglang::scheduler")  │  managers/scheduler.py
│  - waiting_queue / running_batch          │
│  - 매 iteration: 다음 batch 결정 → GPU 실행  │
│  - TpModelWorker → ModelRunner → 모델       │  managers/tp_worker.py, model_executor/model_runner.py
│  - RadixCache / KV pool 소유               │  mem_cache/
└───────────────┬──────────────────────────┘
                │ ZMQ (BatchTokenIDOutput: 새 token id들)
                ▼
┌──────────────────────────────────────────┐
│ DetokenizerManager ("sglang::detokenizer")│  managers/detokenizer_manager.py
│  - token id → 텍스트 (incremental decode)   │
└───────────────┬──────────────────────────┘
                │ ZMQ (BatchStrOutput)
                ▼
          TokenizerManager → HTTP 응답 / SSE 스트림
```

`ps -ef | grep sglang` 를 Pod에서 실행하면 이 프로세스들이 실제로 보입니다 (Session 2 실습).

---

## 회차별 읽을 곳

### Session 1–2: 요청의 일생과 스케줄러 루프

| 무엇 | 파일 | 심볼 |
|---|---|---|
| 서버 진입점 | `launch_server.py`, `srt/entrypoints/http_server.py` | `run_server`, `generate_request` (route) |
| 프로세스 띄우기 | `srt/entrypoints/engine.py` | `run_scheduler_process`, `run_detokenizer_process` 사용처 |
| 토크나이즈 & 전송 | `srt/managers/tokenizer_manager.py` | `TokenizerManager.generate_request`, `_tokenize_one_request` |
| 요청 객체 | `srt/managers/schedule_batch.py` | `class Req`, `class ScheduleBatch` |
| 요청 수신 → 큐 | `srt/managers/scheduler.py` | `handle_generate_request`, `_add_request_to_queue` |
| **스케줄러 루프** | `srt/managers/scheduler.py` | `event_loop_normal`, `event_loop_overlap` |
| 다음 batch 결정 | `srt/managers/scheduler.py` | `get_next_batch_to_run` → `get_new_batch_prefill` / `update_running_batch` |
| 실행 & 결과 처리 | `srt/managers/scheduler.py` | `run_batch`, `process_batch_result` |
| 로그 한 줄 만드는 곳 | `srt/managers/scheduler_components/metrics_reporter.py` | `report_prefill_stats` (`Prefill batch, ...`), decode 쪽 `Decode batch, ...` |
| detokenize | `srt/managers/detokenizer_manager.py` | `DetokenizerManager.event_loop`, `handle_batch_token_id_out` |

### Session 3: KV cache 메모리와 RadixAttention

| 무엇 | 파일 | 심볼 |
|---|---|---|
| 레이어 구조 개요 (공식 README) | `srt/mem_cache/README.md` | allocator / pool / radix cache 역할 분리 |
| 요청 → token 슬롯 매핑 | `srt/mem_cache/memory_pool.py` | `class ReqToTokenPool` (`req_to_token` 2D 텐서) |
| 실제 K/V 텐서 | `srt/mem_cache/memory_pool.py` | `class MHATokenToKVPool` (`k_buffer`, `v_buffer` per layer) |
| 빈 슬롯 관리 | `srt/mem_cache/allocator/token.py`, `allocator/paged.py` | `TokenToKVPoolAllocator.alloc/free`, `PagedTokenToKVPoolAllocator` |
| **radix tree (교과서 구현)** | `srt/mem_cache/radix_cache.py` | `TreeNode`, `RadixKey`, `RadixCache.match_prefix`, `_match_prefix_helper`, `_split_node`, `insert`, `_insert_helper`, `checkpoint`(구버전 `cache_finished_req`/`cache_unfinished_req`), `evict`, `inc_lock_ref`/`dec_lock_ref` |
| prefix 재사용 상한 | `srt/managers/schedule_batch.py` | `Req.init_next_round_input`, `_compute_max_prefix_len` (`input_len - 1`) |
| in-batch prefix caching | `srt/managers/schedule_policy.py` | `_compute_prefix_matches`, `IN_BATCH_PREFIX_CACHING_*`, `WAITING_QUEUE_PREFIX_MATCH_MAX = 128` |
| radix tree (v0.5.23 기본 런타임 경로) | `srt/mem_cache/registry.py` → `unified_radix_cache.py`, `unified_cache/unified_tree_core.py` | `default_radix_cache_factory` → `create_unified_radix_cache`, `UnifiedTreeCore.match_prefix` |
| eviction 정책 | `srt/mem_cache/evict_policy.py` | `LRUStrategy`, `LFUStrategy`, ... (`--radix-eviction-policy`) |
| cache-aware 정렬 | `srt/managers/schedule_policy.py` | `SchedulePolicy.calc_priority`, `_sort_by_longest_prefix` (LPM) |

> ⚠️ **v0.5.23의 함정**: 일반 dense 모델도 기본적으로 `UnifiedRadixCache`(Full/SWA/Mamba를 한 틀로 다루는 일반화 버전)를 씁니다.
> 알고리즘(최장 prefix 매칭, 노드 split, lock, LRU leaf eviction)은 같으므로 **개념은 `radix_cache.py`로 배우고**,
> "실제로는 어디서 도는가"는 `registry.py`의 `default_radix_cache_factory`로 확인합니다.

### Session 4: 스케줄러 심화

| 무엇 | 파일 | 심볼 |
|---|---|---|
| prefill admission | `srt/managers/schedule_policy.py` | `class PrefillAdder`, `add_one_req`, `add_chunked_req`, `AddReqResult` (`CONTINUE`/`NO_TOKEN`/`OTHER`), `rem_total_tokens` |
| new_token_ratio (낙관적 예약) | `srt/managers/scheduler_components/new_token_ratio_tracker.py` | `NewTokenRatioTracker` (`init`/`min`/`decay`, `estimate_new_token_ratio_after_retract`) |
| 기본값 | `srt/environ.py` | `SGLANG_INIT_NEW_TOKEN_RATIO=0.7`, `SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR=0.14`, `SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS=600` |
| retraction | `srt/managers/schedule_batch.py` | `ScheduleBatch.retract_decode`, `_get_decode_retraction_order` |
| retraction 로그 | `srt/managers/scheduler.py` | `"KV cache pool is full. Retract requests."` |
| chunked prefill | `srt/managers/scheduler.py` | `chunked_req`, `stash_chunked_request` |
| overlap 스케줄러 | `srt/managers/scheduler.py` | `event_loop_overlap`, `result_queue`, `is_disable_overlap_for_batch` |
| GPU별 기본값 | `srt/arg_groups/memory_hook.py` | `chunked_prefill_size`, decode CUDA graph `max_bs`를 GPU 메모리로 결정 (4090: 4096 / 48) |
| 스케줄링 플래그 정의 | `srt/arg_groups/fields/schedule.py` | `schedule_policy` (기본 `fcfs`), `chunked_prefill_size`, `max_running_requests`, `max_total_tokens`, `schedule_conservativeness`, `enable_mixed_chunk` |

### Session 5: 모델 실행과 커널

| 무엇 | 파일 | 심볼 |
|---|---|---|
| ScheduleBatch → ForwardBatch | `srt/managers/tp_worker.py` | `TpModelWorker.forward_batch_generation` (`ForwardBatch.init_new`) |
| ForwardBatch / ForwardMode | `srt/model_executor/forward_batch_info.py` | `ForwardMode.EXTEND/DECODE/MIXED/IDLE`, `ForwardBatch` (`input_ids`, `positions`, `seq_lens`, `out_cache_loc`, `extend_prefix_lens`, `extend_seq_lens`) |
| 모델 실행 | `srt/model_executor/model_runner.py` | `ModelRunner.forward`, `_forward_raw` |
| CUDA graph | `srt/model_executor/runner/decode_cuda_graph_runner.py` | `DecodeCudaGraphRunner.can_run_graph`, `capture`, `execute` |
| 모델 정의 | `srt/models/qwen3.py` | `Qwen3Attention` → `RadixAttention` 호출부 |
| attention 진입 | `srt/layers/radix_attention.py` | `RadixAttention.forward` (얇은 shim: backend로 위임) |
| backend 계약 | `srt/layers/attention/base_attn_backend.py` | `AttentionBackend.init_forward_metadata`, `forward` → `forward_extend` / `forward_decode` |
| backend 구현 | `srt/layers/attention/flashinfer_backend.py`, `triton_backend.py` | 각 클래스의 `forward_extend`, `forward_decode` |
| logits & sampling | `srt/layers/logits_processor.py`, `srt/layers/sampler.py` | `LogitsProcessor.forward`, `Sampler.forward` |
| 실행 관련 플래그 | `srt/arg_groups/fields/exec_.py` | `attention_backend`, `cuda_graph_backend_decode`, `disable_decode_cuda_graph`, `enable_torch_compile` |
| speculative decoding | `srt/speculative/`, `srt/arg_groups/fields/spec.py` | `speculative_algorithm` (EAGLE, EAGLE3, NGRAM, ...) |

### Session 6: 측정 도구

| 무엇 | 파일 |
|---|---|
| 서빙 벤치마크 | `benchmark/serving.py` (`python -m sglang.benchmark.serving`; 구 경로 `sglang.bench_serving`도 동작) |
| 서버 없는 단일 batch 벤치마크 | `benchmark/one_batch.py` (`python -m sglang.benchmark.one_batch`) |
| 프로파일러 | `profiler.py` (`python -m sglang.profiler`), HTTP `/start_profile` / `/stop_profile` |
| Prometheus metrics | `--enable-metrics` → `/metrics` (`docs/.../references/production_metrics.mdx`) |

---

## 버전이 다를 때

SGLang을 다른 버전으로 설치했다면:

```bash
python -c "import sglang, os; print(sglang.__version__, os.path.dirname(sglang.__file__))"
grep -rn "def get_next_batch_to_run" $(python -c "import sglang, os; print(os.path.dirname(sglang.__file__))")
```

심볼 이름이 없으면 비슷한 이름(`get_next_batch`, `schedule`)으로 grep하세요. 구조(TokenizerManager → Scheduler → Detokenizer)는 2024년 이후 거의 변하지 않았습니다.
