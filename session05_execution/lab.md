# Session 5 실습 — step 시간을 분해하고, 하나씩 줄여 보기
> **2시간 수업 기준:** 필수 1·2·6 / 선택 3–5 (과제 주제로 연결)

> 실습 55분 (2시간 수업 기준) / GPU 예산 약 $1.0 / 결과물: `results/s05_*.jsonl|csv|txt`
> 모델 로딩이 여러 번 일어납니다. 각 단계는 스크립트가 알아서 띄우고 끕니다.

| # | 실습 | 시간 |
|---|---|---|
| 0 | 준비 + 코드 읽기 (ForwardBatch 찍어 보기) | 실습 직전 + 5분 |
| 1 | `one_batch`: batch별 decode step과 roofline ⭐ | 12분 |
| 2 | CUDA graph ON/OFF ⭐ | 10분 |
| 3 | Attention backend: flashinfer vs triton | 8분 |
| 4 | 양자화: BF16 vs FP8 weight vs FP8 KV ⭐ | 15분 |
| 5 | Speculative decoding (NGRAM) | 10분 |
| 6 | 정리 & Terminate | 5분 |

## 0. 준비

```bash
cd /workspace && git clone --branch main https://github.com/mlleo/metacode_sglang_study.git && cd metacode_sglang_study
bash common/setup_pod.sh
```

설치된 SGLang 코드에서 이론 §1의 필드를 직접 확인합니다 (GPU 불필요):

```bash
SG=$(python3 -c "import sglang, os; print(os.path.dirname(sglang.__file__))")
grep -n "class ForwardMode" -A12 $SG/srt/model_executor/forward_batch_info.py
grep -n "    out_cache_loc\|    extend_prefix_lens\|    seq_lens:" $SG/srt/model_executor/forward_batch_info.py
grep -n "self.attn = RadixAttention" -A8 $SG/srt/models/qwen3.py
```

---

## 1. `one_batch`: batch별 decode step과 roofline ⭐ (12분)

서버 없이 한 batch를 직접 돌립니다. HTTP와 서빙 큐를 건너뛰고, batch 준비부터 forward·sampling·GPU 동기화까지의 경과 시간을 잽니다.

```bash
BS="1 16 64 128" bash session05_execution/code/onebatch_sweep.sh base
python3 session05_execution/code/onebatch_to_csv.py results/s05_onebatch.jsonl
```

| batch | decode ms | decode tok/s | 달성 GB/s | BW% | KV% |
|---:|---:|---:|---:|---:|---:|
| 1 | | | | | |
| 16 | | | | | |
| 64 | | | | | |
| 128 | | | | | |

- batch 1 → 128에서 decode ms는 몇 배? Session 1의 HF manual loop 결과와 비교하면?
- batch=1의 BW%는? 100%와의 차이만 보고 launch 시간을 구할 수 있을까요? §3.1의 bytes 추정을 확인하세요.
- batch가 커질수록 `KV%`(step bytes 중 KV 비중)는 어떻게 변하나요? 그게 Session 1 과제 Part B-3의 답입니다.
- 로그(`logs/onebatch_base.log`)의 `Decode 0., Decode 1., …` 줄을 보면 첫 몇 step만 느린가요? 왜?

---

## 2. CUDA graph ON/OFF ⭐ (10분)

**예측 먼저:** batch=1에서 몇 배 느려질까요? batch=128에서는?

```bash
BS="1 16 64 128" bash session05_execution/code/onebatch_sweep.sh nograph --cuda-graph-backend-decode disabled
python3 session05_execution/code/onebatch_to_csv.py results/s05_onebatch.jsonl
```

| batch | base decode ms | nograph decode ms | 배율 |
|---:|---:|---:|---:|
| 1 | | | |
| 16 | | | |
| 64 | | | |
| 128 | | | |

- 배율이 batch가 커질수록 줄어드나요? 왜일까요? (커널 하나의 GPU 시간이 길어지면 launch overhead 비중↓)
- v0.5.23의 `one_batch`는 측정 목록의 가장 큰 batch에 맞춰 decode graph 상한을 올립니다. `logs/onebatch_base.log`의 `Capture target decode CUDA graph end. bs=`에 128이 있는지 확인하세요. HTTP 서버의 기본 상한과 따로 해석합니다.
- Session 1의 "HF가 이론 하한보다 몇 배 느렸던 이유" 가설 중 무엇이 확인되었나요?

서빙에서 상한을 확인하려면 강사 로그의 `Decode batch`에서 `#running-req`와 `cuda graph: True/False`를 함께 봅니다. 비교할 때 동시성과 그 순간의 실제 batch 크기를 구분하세요.

---

## 3. Attention backend: flashinfer vs triton (8분)

```bash
bash session05_execution/code/onebatch_sweep.sh triton --attention-backend triton
python3 session05_execution/code/onebatch_to_csv.py results/s05_onebatch.jsonl
```

- prefill latency와 decode latency 중 어느 쪽 차이가 더 큰가요?
- (긴 context에서 차이가 커질 수 있습니다. 시간이 남으면 `IN=4096 BS="1 16" bash session05_execution/code/onebatch_sweep.sh triton4k --attention-backend triton`와 base 4k를 비교)
- Triton backend 코드는 읽기 쉽습니다. `$SG/srt/layers/attention/triton_backend.py`의 `_fill_kv_indptr_and_indices`와 `forward_decode`에서 `kv_indptr`/`kv_indices`(이론 §2.2의 CSR)가 어떻게 만들어져 커널(`$SG/kernels/ops/attention/decode_attention.py`)로 넘어가는지 찾아보세요.

---

## 4. 선택 실습 · BF16 vs FP8 weight vs FP8 KV (15분)

```bash
bash session05_execution/code/quant_compare.sh
cat results/s05_quant_memory.txt
```

| 설정 | weight mem (GB) | KV #tokens | ITL p50 @c=1 | output tok/s @c=16 | @c=64 |
|---|---:|---:|---:|---:|---:|
| bf16 | | | | | |
| fp8w | | | | | |
| fp8kv | | | | | |

- fp8w: weight 메모리가 절반이 됐나요? batch=1 ITL은 몇 배 빨라졌나요? 이론(2배)과의 차이는?
- fp8kv: KV #tokens가 약 2배가 됐나요? 속도는 빨라졌나요, 느려졌나요? (backend의 FP8 KV 지원 여부)
- 출력 품질 확인 (서버를 하나 띄워서 같은 프롬프트로 비교):

  ```bash
  bash common/launch.sh q_fp8w --quantization fp8
  python3 common/client.py --prompt "Explain why the sky is blue in two sentences." --max-new-tokens 60 --allow-eos
  bash common/stop.sh
  ```

> 설정 하나가 시작에 실패하면 스크립트는 건너뛰고 계속합니다. 수업이나 과제는 CUDA graph 실험으로 이어갈 수 있습니다. `logs/server_q_<tag>.log`의 마지막 줄을 보고 이유를 기록하세요 (그것도 결과입니다).

---

## 5. Speculative decoding — NGRAM (10분)

```bash
bash common/launch.sh spec_off --mem-fraction-static 0.7 --cuda-graph-max-bs-decode 8
python3 session05_execution/code/spec_ngram.py --tag off --out results/s05_spec.csv
bash common/stop.sh

bash common/launch.sh spec_ngram --speculative-algorithm NGRAM --speculative-num-draft-tokens 16 \
  --speculative-ngram-max-bfs-breadth 10 --mem-fraction-static 0.7 --cuda-graph-max-bs-decode 8
python3 session05_execution/code/spec_ngram.py --tag ngram --out results/s05_spec.csv
grep "accept len" logs/server_spec_ngram.log | tail -5
bash common/stop.sh
```

| workload | concurrency | off tok/s (요청당) | ngram tok/s | 배율 | accept len (로그) |
|---|---:|---:|---:|---:|---:|
| repetitive | 1 | | | | |
| repetitive | 8 | | | | |
| creative | 1 | | | | |
| creative | 8 | | | | |

- 두 서버에 같은 KV 메모리 비율과 decode graph 상한을 적용합니다. speculative 검증 방식은 서버가 따로 정할 수 있으므로 시작 로그도 남기세요.
- 어떤 workload/동시성에서 이득이 가장 큰가요? 손해를 보는 칸이 있나요?
- `accept len`과 배율의 관계는? (배율 ≈ accept len ÷ (verify step 비용 / 일반 step 비용))

---

## 6. 정리 & Terminate

```bash
bash common/stop.sh
tar czf s05_$(date +%m%d).tgz results logs
```

로컬 그래프:

```bash
python3 common/plot.py results/s05_onebatch.csv --x batch_size \
  --y median_decode_latency_ms,bw_util_pct --group run_name --logx --out figures/s05_decode.png
```

## 트러블슈팅

| 증상 | 해결 |
|---|---|
| `one_batch`가 "서버가 GPU를 쓰고 있습니다" | `bash common/stop.sh` |
| `--quantization fp8` 시작 실패 | GPU가 FP8을 지원하지 않음 (3090 등) → 결과에 기록하고 건너뜀 |
| NGRAM 서버 시작 실패 | `--cuda-graph-max-bs-decode 8` 등 문서 예시 인자가 빠졌는지 확인. 계속 실패하면 로그를 첨부해 강사에게 |
| one_batch 결과 jsonl에 같은 tag가 여러 번 | 다시 실행할 때마다 행이 추가됩니다. 필요하면 `results/s05_onebatch.jsonl`을 지우고 다시 |
