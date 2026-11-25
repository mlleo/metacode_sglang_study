# Session 3 실습 — KV pool 실측과 RadixAttention 관찰
> **2시간 수업 기준:** 필수 0(로컬)·1·2·3·6 / 선택 4, 5 (5는 서버를 두 번 띄워 6–8분 → 강사 시연 추천)

> 실습 55분 (2시간 수업 기준) / GPU 예산 약 $0.8 / 결과물: `results/s03_*.csv|jsonl`, `logs/server_*.log`

| # | 실습 | 시간 | GPU |
|---|---|---|---|
| 0 | radix tree 시뮬레이터 (로컬에서도 가능) | 5분 | 불필요 |
| 1 | KV pool 크기: 손계산 vs 실측 | 10분 | ✅ |
| 2 | `cached_tokens` 기본기 | 10분 | ✅ |
| 3 | prefix 길이 sweep (cold vs warm) ⭐ | 12분 | ✅ |
| 4 | 멀티턴: token으로 이어 붙이기 vs 텍스트로 이어 붙이기 | 8분 | ✅ |
| 5 | LPM vs FCFS (메모리가 빠듯할 때) ⭐ | 12분 | ✅ |
| 6 | 정리 & Terminate | 8분 | |

## 0. radix tree 시뮬레이터 (Pod을 띄우는 동안, 로컬에서)

이론 §4의 손 추적을 그대로 돌려 봅니다.

```bash
python3 session03_kvcache_radix/code/radix_sim.py ABCDE ABCXY ABCDEZ ABQR --capacity 14
python3 session03_kvcache_radix/code/radix_sim.py ABCDE ABCXY ABQRS --capacity 9      # eviction 발생
```

- 두 번째 명령에서 `DE`가 지워지는 이유는? (LRU, leaf, lock 0)
- `--capacity 7`로 줄이면 무슨 일이 생기나요? 실제 SGLang이라면 이때 어떻게 할까요? (→ Session 4)

---

## 1. KV pool 크기: 손계산 vs 실측 (10분)

새 Pod라면 실습 저장소를 받고 환경을 준비합니다. 로컬 시뮬레이터는 계속 로컬에서 실행합니다.

```bash
cd /workspace
git clone --branch main https://github.com/mlleo/metacode_sglang_study.git
cd metacode_sglang_study
bash common/setup_pod.sh
```

```bash
bash common/launch.sh baseline
grep -E "Load weight end|KV Cache is allocated|max_total_num_tokens=" logs/server_baseline.log
```

| 항목 | 값 |
|---|---|
| `#tokens` (KV Cache is allocated) | |
| `K size`와 `V size`의 합 (로그에 GB로 표시, 계산은 GiB 기준) | |
| (K + V) ÷ #tokens (KiB/token) | |
| 손계산 (Session 1 과제) | 112 KiB/token |

일치하나요? 이제 `mem_fraction_static`을 바꿔서 서버가 고르는 token 수가 어떻게 바뀌는지 예측 → 확인합니다.

```bash
bash common/stop.sh
bash common/launch.sh mf05 --mem-fraction-static 0.5
grep -E "KV Cache is allocated|max_total_num_tokens=" logs/server_mf05.log
```

- 예측: 24 GiB × 0.5 − weights 3.2 GiB = ? GiB → ? token
- 실측과 차이가 나면 GPU의 실제 가용 메모리와 weight 외에 이미 사용 중인 메모리를 확인하세요.

```bash
bash common/stop.sh
bash common/launch.sh baseline          # 이후 실습은 기본 설정으로
```

---

## 2. `cached_tokens` 기본기 (10분)

```bash
python3 session03_kvcache_radix/code/prefix_basics.py
```

출력의 "예상" 칸과 실제 `cached` 값을 비교하세요. 특히:

- 3번(완전히 같은 요청)이 왜 `prompt − 1`인가?
- 6번에서 **생성된 token**까지 재사용되는가?
- `TTFT`가 `cached`에 따라 얼마나 줄었나? `(1 − cached/prompt)` 비율과 비슷한가요? 아니라면 남는 고정 비용은 무엇일까요?

로그로 교차 확인:

```bash
grep "Prefill batch" logs/server_baseline.log | tail -6
```

`#new-token` + `#cached-token` = prompt 길이인가요?

---

## 3. prefix 길이 sweep — cold vs warm ⭐ (12분)

```bash
python3 session03_kvcache_radix/code/prefix_sweep.py --out results/s03_prefix_sweep.csv
```

| prefix_len | cold TTFT p50 (ms) | warm TTFT p50 (ms) | warm cached_mean | 절약 비율 |
|---:|---:|---:|---:|---:|
| 0 | | | | |
| 1024 | | | | |
| 4096 | | | | |
| 8192 | | | | |

- cold TTFT는 prefix 길이에 선형인가요? 8K에서 기울기가 커진다면 왜? (attention은 길이의 제곱, chunked prefill 4096)
- warm TTFT는 prefix 길이와 거의 무관해야 합니다. 그런가요? 조금씩 늘어난다면 무엇 때문일까요?
  (힌트: 새로 계산하는 64 token도 attention에서는 **앞의 8K token 전부**를 읽어야 한다)

---

## 4. 멀티턴 (8분)

```bash
python3 session03_kvcache_radix/code/multiturn.py
```

- `ids` 방식에서 turn 6의 hit rate는? (90% 이상이어야 정상)
- `text` 방식에서 hit rate가 떨어지는 턴이 있나요? 있다면 어느 지점에서 매칭이 끊겼는지 추측해 보세요.
- 서비스를 만든다면 어떤 방식으로 대화 기록을 관리하겠습니까?

---

## 5. LPM vs FCFS ⭐ (12분)

KV pool을 일부러 24,000 token으로 줄이고(`--max-total-tokens`), prefix 2048 token짜리 그룹 16개(총 약 3.3만 token)를 섞어 보냅니다.
**캐시에 전부 들어가지 않으므로** 어떤 순서로 처리하느냐가 중요해집니다.

```bash
bash common/stop.sh
bash session03_kvcache_radix/code/policy_compare.sh 24000
python3 session02_architecture_batching/code/bench_to_csv.py results/s03_policy.jsonl
```

| 정책 | cache_hit_rate (로그) | sum_new_token | output tok/s | median TTFT | p99 TTFT |
|---|---:|---:|---:|---:|---:|
| lpm | | | | | |
| fcfs | | | | | |

- LPM이 hit rate를 올렸나요? throughput은?
- p99 TTFT는 어느 쪽이 더 나쁜가요? 왜 그럴까요? (이론 §6.1 starvation)
- **차이가 안 보인다면**: pool이 충분히 작지 않거나, 요청이 너무 적어 대기열이 짧은 것입니다. `--max-total-tokens 16000`으로 다시 해 보세요.
  "차이가 안 나는 조건"을 찾는 것도 좋은 결과입니다.

---

## 6. 정리 & Terminate

```bash
bash common/stop.sh
tar czf s03_$(date +%m%d).tgz results logs
```

로컬 그래프:

```bash
python3 common/plot.py results/s03_prefix_sweep.csv --x prefix_len --y ttft_ms_p50 --group mode \
  --out figures/s03_prefix_sweep.png
```
