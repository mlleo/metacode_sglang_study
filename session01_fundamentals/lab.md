# Session 1 실습 — 직접 짠 decode loop, 그리고 첫 SGLang 서버
> **2시간 수업 기준:** 필수 0·1·3·4·5 / 선택 2 (별도의 HF `generate()` 비교)

> 실습 55분 (2시간 수업 기준) / GPU 예산 약 $0.9 / 결과물: `results/s01_*.csv`, `logs/server_baseline.log`, `logs/env.txt`

| # | 실습 | 시간 | 핵심 관찰 |
|---|---|---|---|
| 0 | Pod 준비 (실습 직전) | 10분 | driver ≥ 580, SGLang 0.5.23 |
| 1 | 직접 짠 decode loop | 15분 | prefill ∝ 입력 길이, decode step은 거의 일정, batch를 키워도 step 시간은 조금만 증가 |
| 2 | HF `generate()` 기준선 | 8분 | batch별 throughput / E2E |
| 3 | 첫 SGLang 서버 | 12분 | 시작 로그 읽기, native / streaming / OpenAI API |
| 4 | 같은 workload를 SGLang으로 | 10분 | 같은 GPU, 같은 weight인데 속도가 다르다 |
| 5 | 정리 & Terminate | 10분 | 결과 다운로드 → **Terminate** |

---

## 0. Pod 준비 (수업 시작할 때)

[docs/runpod_guide.md](../docs/runpod_guide.md) 순서대로: RTX 4090 · **CUDA 13.0+ 필터** · Container Disk 40 GB.

```bash
cd /workspace
git clone https://github.com/<your-id>/metacode_sglang_study.git
cd metacode_sglang_study
bash common/setup_pod.sh          # 끝나면 logs/env.txt 생성
```

`setup_pod.sh`가 끝나기를 기다리는 동안 **lecture.md의 예측 질문 4개에 숫자로 답을 적어 두세요.**

---

## 1. 직접 짠 decode loop (15분)

`code/manual_decode.py`는 `generate()`를 쓰지 않고 prefill 1번 + 후속 decode을 손으로 돌리며 각각의 시간을 잽니다.
먼저 코드를 30초만 훑어보세요. `run()` 함수 안의 두 `model(...)` 호출이 각각 prefill과 decode입니다.

```bash
python3 session01_fundamentals/code/manual_decode.py --out results/s01_manual.csv
```

맨 처음 출력되는 줄을 보세요:

```text
weights = 3.44 GB → batch=1 decode 이론 하한 ≈ 3.41 ms/step (대역폭 1008 GB/s 가정)
```

### 관찰 표 (노트에 옮겨 적기)

| experiment | batch | in_len | prefill_ms | decode_step_ms_p50 | decode_tok_per_s |
|---|---:|---:|---:|---:|---:|
| batch | 1 | 512 | | | |
| batch | 8 | 512 | | | |
| batch | 32 | 512 | | | |
| inlen | 1 | 128 | | | |
| inlen | 1 | 512 | | | |
| inlen | 1 | 2048 | | | |
| nocache | 1 | 512 | | | |

### 해석 질문

1. **inlen 실험**: 입력 128 → 2048 (16배)일 때 prefill_ms는 몇 배? decode_step_ms는?
   → prefill은 입력에 따라 크게 늘고, decode step은 거의 그대로여야 합니다. 왜일까요? (이론 §3, §5)
2. **batch 실험**: batch 1 → 32에서 decode_step_ms는 몇 배? decode_tok_per_s는 몇 배?
   → step 시간이 32배보다 적게 늘었다면 batch로 처리량을 얻은 것입니다. weight를 여러 요청이 같이 쓰는 모델로 이 결과를 설명해 보세요.
3. batch=1 decode_step_ms를 **이론 하한**과 비교하세요. 몇 배 느린가요?
   그 차이는 어디서 올까요? 힌트: `nvidia-smi`로 batch=1 실행 중 GPU utilization을 보면 생각보다 낮습니다. GPU가 비는 이유로 CPU 준비나 동기화가 있을 수 있습니다. Session 6에서는 trace로 확인합니다.
4. **nocache**: decode_step_ms가 batch 실험의 batch=1 값보다 몇 배인가요? 출력이 64가 아니라 1024라면 어떻게 될까요?

> ⚠️ 이 스크립트는 step마다 `torch.cuda.synchronize()`를 부릅니다. 정확한 측정을 위해서지만, 그 자체가 약간의 overhead입니다.
> "측정 도구가 측정 대상을 바꾼다"는 것도 기억해 두세요.

---

## 2. HF `generate()` 기준선 (8분)

```bash
python3 session01_fundamentals/code/hf_generate_baseline.py --batches 1,8,32 --out results/s01_hf.csv
```

| batch | e2e_s | output_tok_per_s | peak_mem_gib |
|---:|---:|---:|---:|
| 1 | | | |
| 8 | | | |
| 32 | | | |

- `peak_mem_gib`이 batch에 따라 어떻게 늘어나나요? 늘어난 만큼을 KV 공식(112 KiB × batch × 640 token)으로 설명할 수 있나요?
- 이 숫자는 **"32개 요청이 정확히 같은 순간, 같은 길이로 도착했다"** 는 비현실적으로 유리한 조건입니다. 실제 서비스에선 요청이 제각각 도착합니다. 그럼 `generate()`는 어떻게 해야 할까요? (→ Session 2)

이제 HF 모델이 GPU 메모리를 잡고 있지 않은지 확인하세요 (스크립트가 끝나면 해제됩니다):

```bash
nvidia-smi --query-gpu=memory.used --format=csv
```

---

## 3. 첫 SGLang 서버 (12분)

### 3-1. 서버 띄우기

```bash
bash common/launch.sh baseline
```

첫 실행은 1–3분 걸립니다 (weight 로딩 → KV pool 할당 → CUDA graph capture). 기다리는 동안 다른 터미널에서:

```bash
tail -f logs/server_baseline.log
```

### 3-2. 시작 로그 읽기 — 서버가 스스로 내린 결정들

로그에서 아래 줄들을 찾아 값을 적으세요.

```bash
grep -E "Load weight end|KV Cache is allocated|max_total_num_tokens=|CUDA graph end" logs/server_baseline.log
```

| 로그 항목 | 값 | 의미 |
|---|---|---|
| `Load weight end. ... mem usage=` | | weight가 차지한 GPU 메모리 (GB) |
| `KV Cache is allocated. #tokens:` | | KV pool에 담을 수 있는 token 수 |
| `max_total_num_tokens=` | | 위와 같은 값 (스케줄러가 쓰는 예산) |
| `chunked_prefill_size=` | | 4090에서 기본 4096 (Session 4) |
| `max_running_requests=` | | 동시에 decode할 수 있는 요청 수 상한 |
| `Capture target decode CUDA graph end. bs=[...]` | | decode CUDA graph를 녹화한 batch 크기들 (Session 5) |

**과제와 연결**: `#tokens` 값을 `common/kvcalc.py`로 계산한 값(약 12만)과 비교하세요.

```bash
python3 common/kvcalc.py Qwen/Qwen3-1.7B --params-b 1.72
```

### 3-3. 요청 보내기

```bash
python3 session01_fundamentals/code/first_requests.py
```

네 가지를 순서대로 해 봅니다.

1. **native `/generate`** — 응답의 `meta_info`에 `prompt_tokens`, `cached_tokens`가 있습니다. `cached_tokens`는 Session 3의 주인공입니다.
2. **streaming** — 글자가 흘러나오는 속도가 곧 ITL입니다. HF batch=1 decode_step_ms와 비교해 보세요.
3. **OpenAI 호환 API** — OpenAI SDK를 쓰는 기존 코드는 `base_url`만 바꾸면 SGLang에 붙습니다.
4. **8개 순차 vs 동시** — 동시에 보내면 몇 배 빨라지나요? 서버 로그에서 확인:

   ```bash
   grep "Decode batch" logs/server_baseline.log | tail -3
   ```

   `#running-req: 8` 이 보이면 8개 요청이 **한 batch로 묶여** decode되고 있다는 뜻입니다. 누가 묶었을까요? (→ Session 2의 Scheduler)

### 3-4. 프로세스 구조 엿보기 (1분)

```bash
ps -eo pid,comm,args | grep -E "sglang" | grep -v grep
```

`sglang::scheduler`, `sglang::detokenizer` 프로세스가 보입니다. 서버는 프로세스 하나가 아닙니다 (→ Session 2).

---

## 4. 같은 workload를 SGLang으로 (10분)

```bash
python3 session01_fundamentals/code/sglang_same_workload.py --batches 1,8,32 --out-len 64 --out results/s01_sglang.csv
```

| batch | HF manual decode tok/s | SGLang 요청 전체 tok/s | SGLang batch wall time (s) | SGLang ITL p50 (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | | | | | |
| 8 | | | | | |
| 32 | | | | | |

토론 (조별 3분):
- 두 파일에서 처리량의 분모로 잰 시간이 어디서 시작해 어디서 끝나는지 써 보세요. 차이가 생길 요인을 세 가지 적습니다. (힌트: 이론 §8 표)
- SGLang batch=1 ITL은 이론 하한 3.4 ms에 얼마나 가까운가요?
- HF는 batch 32가 "동시에 도착"해야만 가능했습니다. SGLang은 요청이 따로 도착해도 묶어 줍니다. 이 차이는 이 표에 드러나 있나요? (드러나 있지 않습니다 — 그래서 Session 2에서 다른 측정 방법이 필요합니다.)

---

## 5. 정리 & Terminate (10분)

```bash
bash common/stop.sh
tar czf s01_$(date +%m%d).tgz results logs
```

1. `s01_MMDD.tgz`를 Jupyter 파일 브라우저로 다운로드 (또는 [runpod_guide.md §5](../docs/runpod_guide.md#5-결과를-pod-밖으로-꺼내기))
2. RunPod 콘솔 → Pods → **Terminate** → 목록에서 사라졌는지 확인
3. `templates/gpu_usage_log.csv` 형식으로 오늘 사용 기록 남기기

그래프는 로컬에서:

```bash
python3 common/plot.py results/s01_hf.csv results/s01_sglang.csv \
  --x batch --y output_tok_per_s,e2e_s --group engine --logx --out figures/s01_hf_vs_sglang.png
```

## 트러블슈팅

| 증상 | 해결 |
|---|---|
| `manual_decode.py`에서 `logits_to_keep` 관련 TypeError | transformers 버전이 오래됨. `setup_pod.sh`가 설치한 환경(SGLang 0.5.23 → transformers 5.x)에서 실행하세요 |
| HF 실행 중 OOM | 다른 프로세스(서버)가 GPU를 잡고 있음 → `bash common/stop.sh` |
| `launch.sh`가 "이미 서버가 떠 있습니다" | `bash common/stop.sh` 후 다시 |
| `first_requests.py` 3번에서 `chat_template_kwargs` 오류 | 해당 키를 지우고 실행 (모델이 `<think>` 블록을 출력할 수 있음) |
