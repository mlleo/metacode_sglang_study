# Session 4 실습 — 스케줄러를 궁지에 몰아넣기
> **2시간 수업 기준:** 학생은 2·4를 실행합니다. 1은 수업에서 제공된 로그로, 3은 선택입니다.

> 실습 55분 (2시간 수업 기준) / GPU 예산 약 $0.8 / 결과물: `results/s04_*.csv|json`, `logs/server_*.log`
> 이번 회차는 **서버를 여러 번 재시작**합니다. 스크립트가 자동으로 띄우고 끄므로, 기다리는 동안 로그를 읽으세요.

| # | 실습 | 시간 |
|---|---|---|
| 0 | 준비 | 실습 직전 |
| 1 | Admission과 retraction 로그 읽기 | 8분 |
| 2 | Head-of-line blocking과 chunked prefill ⭐ | 20분 |
| 3 | Overlap scheduling ON/OFF | 12분 |
| 4 | 정리 & Terminate | 8분 |

## 0. 준비

```bash
cd /workspace && git clone --branch main https://github.com/mlleo/metacode_sglang_study.git && cd metacode_sglang_study
bash common/setup_pod.sh
```

---

## 1. Admission & retraction 로그 읽기

KV pool을 30,000 token으로 좁히고, "아주 긴 이야기를 써 줘" 요청 128개를 한꺼번에 보냅니다. 세 실행은 같은 prompt 묶음을 씁니다. 강사가 미리 받은 결과에서 `Prefill batch`와 `Retract requests`를 찾아 봅니다. 아래 명령으로 데이터를 다시 만들 수 있습니다.

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
2. `sum_new_token`(prefill한 총 token)이 `fixed`보다 `eos`에서 큰가요? 그 차이에는 retraction 뒤 재계산과 실행마다 달라진 요청 길이의 효과가 함께 들어갑니다. retraction 로그에서 재계산의 증거를 따로 찾으세요.
3. 셋 중 어느 것이 가장 빨리 끝났나요(wall_time)? "retraction은 나쁘다"고 단정할 수 있나요?
4. 출력 길이(`completion_len_mean`)는 세 경우 비슷한가요? 다르면 무엇이 비교를 흐리나요? (`temperature=0.8`, EOS)

---

## 2. Head-of-line blocking과 chunked prefill ⭐ (20분)

16개 요청이 생성 중일 때 16K token짜리 요청을 한 번 보냅니다. chunk 크기와 workload는 그대로 두고 mixed만 바꿉니다.

| tag | 서버 인자 | 먼저 예상할 것 |
|---|---|---|
| chunk1024 | `--chunked-prefill-size 1024` | prefill 조각이 이어질 때 기존 요청의 다음 token은 언제 오는가? |
| chunk1024mx | 위 설정 + `--enable-mixed-chunk` | 같은 forward에서 기존 요청도 진행하면 긴 멈춤이 얼마나 달라지는가? |

```bash
bash session04_scheduler/code/interference_sweep.sh
```

실행 도중 출력하는 `스트리밍 중 n/16`을 확인합니다. 비교할 때 이 수가 작은 경우나 실험이 겹치지 않아 중단된 경우에는 두 조건 모두에서 도착 시각을 조정하세요. 예를 들어 `AGGRESSOR_AT=0.3 bash session04_scheduler/code/interference_sweep.sh`를 사용합니다. 아직 첫 token이 오기 전이었다면 더 늦추거나 `VICTIM_OUT`을 늘립니다.

| tag | victim ITL p50 | victim ITL p99 | 최악 멈춤 평균 | 긴 요청 TTFT |
|---|---:|---:|---:|---:|
| chunk1024 | | | | |
| chunk1024mx | | | | |

- 긴 요청이 들어온 시각 전후를 그래프에서 확인하세요. p99와 최대 간격이 한 방향으로 변했나요?
- mixed에서 긴 요청의 TTFT도 달라졌나요? 두 사용자가 겪은 변화를 각각 한 문장으로 씁니다.
- 강사 자료에 interval 또는 unchunked 결과가 있으면 같은 그림에 추가합니다. 직접 비교하려면 스크립트에 `nochunk chunk4096 chunk1024 chunk1024mx chunk1024pd`를 넘깁니다.

```bash
# 결과를 다운로드한 뒤 로컬에서 실행
python3 session04_scheduler/code/plot_timeline.py results/s04_timeline_chunk1024.json \
  results/s04_timeline_chunk1024mx.json --out figures/s04_itl_timeline.png
```

---

## 3. Overlap scheduling ON/OFF (12분)

CPU overhead가 잘 보이도록 **작은 모델(0.6B), 작은 batch**에서 비교합니다.

```bash
export MODEL=Qwen/Qwen3-0.6B
bash common/launch.sh ov_on
python3 common/loadgen.py --num 32 --input-len 128 --output-len 512 --concurrency 1,4,16 --tag ov_on --out results/s04_overlap.csv
bash common/stop.sh

bash common/launch.sh ov_off --disable-overlap-schedule
python3 common/loadgen.py --num 32 --input-len 128 --output-len 512 --concurrency 1,4,16 --tag ov_off --out results/s04_overlap.csv
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
| interference 결과가 실행마다 크게 다름 | 2회 반복해 평균. 긴 요청이 기존 요청의 스트리밍 구간에 들어왔는지 확인 |
