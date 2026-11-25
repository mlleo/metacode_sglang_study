# Session 2 실습 — 서버 내부를 들여다보고, 부하를 걸어 곡선을 그린다
> **2시간 수업 기준:** 필수 2·3·4·6 / 선택 1(py-spy), 5(open-loop — 과제나 강사 시연)

> 실습 55분 (2시간 수업 기준) / GPU 예산 약 $0.8 / 결과물: `results/s02_*.jsonl|csv|json`, `logs/server_baseline.log`

| # | 실습 | 시간 |
|---|---|---|
| 0 | Pod 준비 + 서버 시작 | 실습 직전 |
| 1 | 프로세스 구조와 스케줄러 스택 엿보기 | 8분 |
| 2 | 요청 하나를 로그로 추적 | 7분 |
| 3 | Continuous batching 타임라인 | 10분 |
| 4 | 동시성 sweep (closed-loop) ⭐ | 20분 |
| 5 | 도착률 sweep (open-loop) | 12분 |
| 6 | 정리 & Terminate | 8분 |

## 0. 준비

```bash
cd /workspace && git clone --branch main https://github.com/mlleo/metacode_sglang_study.git && cd metacode_sglang_study
bash common/setup_pod.sh
bash common/launch.sh baseline
```

터미널 두 개를 쓰세요 (`tmux`). 터미널 2에서는 실습 내내 로그를 띄워 둡니다.

```bash
tail -f logs/server_baseline.log | grep --line-buffered -E "Prefill batch|Decode batch"
```

---

## 1. 프로세스 구조와 스케줄러 스택 엿보기 (8분)

```bash
ps -eo pid,ppid,comm,args | grep -E "sglang" | grep -v grep
```

- 어떤 프로세스가 GPU를 쓰고 있나요? `nvidia-smi`의 Processes 칸 PID와 비교하세요. (Scheduler만 GPU 메모리를 잡고 있어야 합니다.)

**스케줄러가 지금 무엇을 하고 있는지 스택 보기** (`py-spy`는 SGLang 의존성으로 설치됩니다):

```bash
SCHED_PID=$(pgrep -f "sglang::scheduler" | head -1)
py-spy dump --pid $SCHED_PID | head -40
```

- 요청이 없을 때: 어떤 함수에서 대기하고 있나요? (`event_loop_overlap` → 요청 수신 함수)
- 다른 터미널에서 `python3 common/loadgen.py --num 32 --concurrency 16 --output-len 256`을 돌리는 동안 다시 dump하면?
  `run_batch`, `forward`, `process_batch_result` 같은 이름이 보이면 성공입니다.

> `py-spy`가 `Permission denied`면 컨테이너에 ptrace 권한이 없는 것입니다. 건너뛰고 [code_map.md](../docs/code_map.md)의 표로 대신하세요.

---

## 2. 요청 하나를 로그로 추적 (7분)

```bash
python3 common/client.py --flush
python3 common/client.py --input-len 2000 --max-new-tokens 200
```

로그 터미널에서 방금 요청이 만든 줄을 찾으세요.

| 확인 | 기대값 | 실제 |
|---|---|---|
| `Prefill batch`의 `#new-seq` | 1 | |
| `#new-token` | 2000 (또는 근처) | |
| `#cached-token` | 0 (방금 flush했으므로) | |
| 그다음 `Decode batch` 줄 수 | 200 step ÷ 40 ≈ 5줄 | |
| `Decode batch`의 `#running-req` | 1 | |
| `gen throughput (token/s)` | ≈ 1000 / ITL(ms) | |

`client.py`가 출력한 `itl_ms_median`과 `gen throughput`이 서로 맞나요? (1000 / ITL ≈ token/s)

---

## 3. Continuous batching 타임라인 (10분)

```bash
python3 session02_architecture_batching/code/staggered.py \
  --out results/s02_staggered.json --plot figures/s02_timeline.png
```

출력 표와 그림(`figures/s02_timeline.png`, Jupyter에서 열기)을 보고:

1. B, C, E(출력 64)는 A(출력 512)보다 **먼저 끝났나요?** static batching이었다면?
2. 각 요청의 TTFT가 수십 ms 수준인가요? 즉 "앞 요청이 끝나길 기다리지 않고" 도착 직후 prefill을 받았나요?
3. 로그 터미널에서 `#running-req`가 1 → 2 → 3 → … → 줄어드는 흐름을 찾으세요. 이것이 매 iteration 재구성되는 batch입니다.
4. A의 `itl_ms_mean`은 혼자 돌 때(실습 2)보다 커졌나요? 커졌다면 원인은 두 가지입니다: batch가 커져서 / 다른 요청의 prefill이 끼어들어서. 어느 쪽이 클까요? (Session 4에서 분리해서 잽니다)

---

## 4. 동시성 sweep — latency-throughput 곡선 ⭐ (20분)

```bash
bash session02_architecture_batching/code/sweep_concurrency.sh base "1 8 32 64 128" 512 256
python3 session02_architecture_batching/code/bench_to_csv.py results/s02_base.jsonl --tag base
```

> 약 8–10분 걸립니다. 기다리는 동안 로그 터미널에서 `#running-req`와 `token usage`가 동시성에 따라 어떻게 변하는지 보세요.
> 특히 동시성 128에서 `#queue-req`가 0보다 커지는 순간이 있나요?

결과 표 채우기 (`results/s02_base.csv`):

| 동시성 | output tok/s | median TTFT | p99 TTFT | median ITL | p99 ITL |
|---:|---:|---:|---:|---:|---:|
| 1 | | | | | |
| 8 | | | | | |
| 32 | | | | | |
| 128 | | | | | |

해석 질문:

1. 동시성 1 → 32에서 throughput은 몇 배, median ITL은 몇 배? Session 1의 "memory-bound 공짜 구간"과 같은 현상인가요?
2. throughput 증가가 둔해지는 지점은 어디인가요? 그 지점에서 `token usage`는 몇이었나요?
3. p99 TTFT가 median TTFT보다 훨씬 큰 지점이 있나요? 그 요청들은 무엇을 기다렸을까요?
4. **(중요)** 동시성 1에서 decode batch 크기(`#running-req`)는 정확히 동시성과 같나요? 아니라면 왜일까요?

---

## 5. 도착률 sweep — open-loop (12분)

4번의 결과에서 **동시성 64일 때의 request throughput(req/s)** 을 확인하세요. 그 근처가 대략 서버의 용량입니다.
그 값의 50% / 100% / 150%를 포함하도록 도착률을 정해 돌립니다. (예: 용량이 16 req/s 근처라면)

```bash
bash session02_architecture_batching/code/sweep_rate.sh rate "4 8 12 16 20 24" 512 256 200
python3 session02_architecture_batching/code/bench_to_csv.py results/s02_rate.jsonl --tag rate
```

- 도착률이 용량을 넘는 순간 median TTFT는 어떻게 되나요? (선형으로 늘지 않고 **폭발**해야 합니다)
- `Peak concurrent requests`는 도착률에 따라 어떻게 변하나요? 안정 상태의 평균 동시성은 Little's law로 확인하고, 순식간의 최고값과 구분하세요.
- closed-loop 동시성 64와 open-loop 용량 근처의 결과를 비교하면, 같은 throughput에서 TTFT가 같나요?

---

## 6. 정리 & Terminate (8분)

```bash
bash common/stop.sh
python3 common/logparse.py logs/server_baseline.log > results/s02_log_summary.txt
tar czf s02_$(date +%m%d).tgz results logs figures
```

다운로드 → **Terminate** → 사용 기록.

로컬에서 그래프:

```bash
python3 common/plot.py results/s02_base.csv --x max_concurrency \
  --y output_throughput,p95_ttft_ms,median_itl_ms --logx --out figures/s02_concurrency.png
python3 common/plot.py results/s02_base.csv --x output_throughput --y p95_ttft_ms \
  --annotate max_concurrency --out figures/s02_curve.png
python3 common/plot.py results/s02_rate.csv --x request_rate --y median_ttft_ms,output_throughput \
  --out figures/s02_rate.png
```

## 트러블슈팅

| 증상 | 해결 |
|---|---|
| `sglang.benchmark.serving`이 tokenizer를 못 찾음 | `--model Qwen/Qwen3-1.7B`(또는 `--tokenizer`) 인자 추가 |
| sweep이 너무 오래 걸림 | 동시성 목록에서 128을 빼거나 출력 길이를 128로 줄이세요. 줄였다면 README에 명시 |
| 동시성 1의 TTFT가 이상하게 큼 | 첫 요청에 JIT/워밍업이 섞임 → `--warmup-requests`를 늘리세요 |
