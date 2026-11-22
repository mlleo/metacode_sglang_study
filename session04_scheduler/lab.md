# Session 4 실습 — 스케줄러를 궁지에 몰아넣기
> **2시간 수업 기준:** 필수 1·2·4 / 선택 3 (1, 2는 각각 스크립트가 10분 안팎 돔 — 시작해 두고 로그 읽기)

> 실습 55분 (2시간 수업 기준) / GPU 예산 약 $0.8 / 결과물: `results/s04_*.csv|json`, `logs/server_*.log`
> 이번 회차는 **서버를 여러 번 재시작**합니다. 스크립트가 자동으로 띄우고 끄므로, 기다리는 동안 로그를 읽으세요.

| # | 실습 | 시간 |
|---|---|---|
| 0 | 준비 | 수업 시작 시 (발표 중) |
| 1 | Admission: 낙관적 예약 vs 보수적 예약, 그리고 retraction ⭐ | 20분 |
| 2 | Head-of-line blocking과 chunked prefill ⭐ | 20분 |
| 3 | Overlap scheduling ON/OFF | 12분 |
| 4 | 정리 & Terminate | 8분 |

## 0. 준비

```bash
cd /workspace && git clone https://github.com/<your-id>/metacode_sglang_study.git && cd metacode_sglang_study
bash common/setup_pod.sh
```

---

## 1. Admission & retraction ⭐ (20분)

KV pool을 30,000 token으로 좁히고, "아주 긴 이야기를 써 줘" 요청 128개를 한꺼번에 보냅니다. 같은 요청을 세 가지 설정으로 보냅니다.

| tag | 요청 | 서버 | 기대 |
|---|---|---|---|
| `eos` | `ignore_eos=False` | 기본 | 많이 받아들이고, 나중에 retract |
| `eos_cons2` | `ignore_eos=False` | `--schedule-conservativeness 2.0` | 덜 받아들이고, retract 감소 |
| `fixed` | `ignore_eos=True` | 기본 | 전부 예약 → 가장 적게 받아들임, retract 거의 없음 |

**먼저 예측을 적으세요:** 세 설정의 `running_req_max`, `retract_events`, `wall_time_s` 순위는?

```bash
bash session04_scheduler/code/retraction_sweep.sh 30000
```

(약 8–10분. 첫 번째 서버가 도는 동안 다른 터미널에서 `tail -f logs/server_rt_eos.log | grep --line-buffered -E "Prefill|Retract"`)

결과 표 (`results/s04_retract.csv` + 스크립트가 출력한 로그 요약):

| tag | running_req_max | new_seq_mean | retract_events | retracted_reqs | sum_new_token | wall_time_s | e2e p99 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| eos | | | | | | | |
| eos_cons2 | | | | | | | |
| fixed | | | | | | | |

해석:

1. `eos`의 retract 로그에서 `#new_token_ratio: a -> b`를 찾으세요. a와 b는 몇인가요? 이론 §2의 표(0.7에서 시작해 0.098까지 감소)와 맞나요?
2. `sum_new_token`(prefill한 총 token)이 `fixed`보다 `eos`에서 큰가요? 그 차이가 **retraction의 재계산 비용**입니다.
3. 셋 중 어느 것이 가장 빨리 끝났나요(wall_time)? "retraction은 나쁘다"고 단정할 수 있나요?
4. 출력 길이(`completion_len_mean`)는 세 경우 비슷한가요? 다르면 무엇이 비교를 흐리나요? (`temperature=0.8`, EOS)

---

## 2. Head-of-line blocking과 chunked prefill ⭐ (20분)

16명이 streaming 중일 때 16K token 프롬프트가 두 번 끼어듭니다. 설정 5가지에서 같은 시나리오를 반복합니다.

| tag | 서버 인자 | 이론 |
|---|---|---|
| nochunk | `--chunked-prefill-size -1` | 16K를 한 번에 |
| chunk4096 | (기본) | 조각 4개가 **연달아** (§4.3) |
| chunk1024 | `--chunked-prefill-size 1024` | 조각 16개가 연달아 |
| chunk1024mx | + `--enable-mixed-chunk` | 조각마다 decode를 섞음 (§4.4) |
| chunk1024pd | + `--prefill-decode-interval 1` | 조각 뒤 decode 1번 보장 (§4.5) |

**예측 먼저:** 각 설정에서 피해자의 최악 멈춤(`victim_worst_gap_ms_mean`)은 몇 ms일까요? (힌트: Session 1 과제에서 잰 TTFT vs 입력 길이. 그리고 이론 §4.3의 ⚠️)

```bash
bash session04_scheduler/code/interference_sweep.sh
```

| tag | victim ITL p50 | victim ITL p99 | 최악 멈춤 평균 | 가해자 TTFT | 피해자 E2E 평균 |
|---|---:|---:|---:|---:|---:|
| nochunk | | | | | |
| chunk4096 | | | | | |
| chunk1024 | | | | | |
| chunk1024mx | | | | | |
| chunk1024pd | | | | | |

해석:

1. nochunk / chunk4096 / chunk1024의 **최악 멈춤이 거의 같다면**, 그게 §4.3의 "조각이 연달아 돈다"는 증거입니다. 그런가요? 다르다면 왜일까요?
2. mx와 pd에서는 최악 멈춤이 얼마로 줄었나요? 그 값은 "1024 token prefill 하나의 시간"과 비슷한가요?
3. 가해자의 TTFT는 설정마다 얼마나 늘었나요? 무엇과 무엇을 바꾼 건가요?
4. ITL **p50**은 다섯 설정에서 비슷하고 **p99와 max**만 다를 것입니다. 왜 p50만 보면 이 문제를 못 찾을까요? mx의 p50이 다른 설정보다 큰가요? (§4.4의 대가)
5. 서버 로그로 조각이 어떻게 들어갔는지 확인하세요:

   ```bash
   grep -E "Prefill batch|Decode batch" logs/server_chunk1024.log | grep -A2 "#new-token: 1024" | head -20
   grep -E "Prefill batch|Decode batch" logs/server_chunk1024pd.log | grep -A2 "#new-token: 1024" | head -20
   ```

   `chunk1024`에서는 `#new-token: 1024` 줄이 **연속으로** 찍히나요? `chunk1024pd`에서는 그 사이에 decode가 끼어드나요?
   (Decode 줄은 40 step마다만 찍히므로, 조각 사이 간격은 로그 타임스탬프로 보는 게 더 정확합니다.)

타임라인 그래프 (로컬):

```bash
python session04_scheduler/code/plot_timeline.py results/s04_timeline_nochunk.json \
  results/s04_timeline_chunk1024.json results/s04_timeline_chunk1024mx.json results/s04_timeline_chunk1024pd.json \
  --out figures/s04_itl_timeline.png
```

---

## 3. Overlap scheduling ON/OFF (12분)

CPU overhead가 잘 보이도록 **작은 모델(0.6B), 작은 batch**에서 비교합니다.

```bash
export MODEL=Qwen/Qwen3-0.6B
bash common/launch.sh ov_on
python common/loadgen.py --num 32 --input-len 128 --output-len 512 --concurrency 1,4,16 --tag ov_on --out results/s04_overlap.csv
bash common/stop.sh

bash common/launch.sh ov_off --disable-overlap-schedule
python common/loadgen.py --num 32 --input-len 128 --output-len 512 --concurrency 1,4,16 --tag ov_off --out results/s04_overlap.csv
bash common/stop.sh
unset MODEL
```

| concurrency | ITL p50 ON | ITL p50 OFF | 차이 (%) | output tok/s ON | OFF |
|---:|---:|---:|---:|---:|---:|
| 1 | | | | | |
| 4 | | | | | |
| 16 | | | | | |

- 차이가 가장 큰 동시성은? 왜 그 지점일까요? (GPU step 시간이 짧을수록 CPU 시간의 비중↑)
- 0.6B에서 ITL이 1.7B와 비교해 얼마나 짧은가요? weight 크기 비율(≈ 1/3)만큼 줄었나요? 덜 줄었다면 남은 시간은 무엇일까요?

---

## 4. 정리 & Terminate

```bash
bash common/stop.sh
tar czf s04_$(date +%m%d).tgz results logs
```

다운로드 → **Terminate** → 사용 기록.

## 트러블슈팅

| 증상 | 해결 |
|---|---|
| `eos`에서도 retraction이 안 생김 | 모델이 일찍 EOS를 냄 → `retraction_demo.py --max-new-tokens 2048` 또는 pool을 20000으로 |
| `--prefill-decode-interval`이 인식되지 않음 | SGLang 버전 확인 (`v0.5.23`). 없으면 mx 결과만으로 비교 |
| `nochunk`에서 서버 OOM | 16K 한 번에 prefill하는 activation 메모리 부족 → `--mem-fraction-static 0.6`을 함께 주고 README에 기록 |
| interference 결과가 실행마다 크게 다름 | 2회 반복해 평균. 가해자 도착 시각(`--aggressor-at`)이 피해자 시작과 겹치지 않는지 확인 |
