# 강사 가이드

## 1. 개강 전 (1–2주 전)

### GPU 리허설

H100 한 장에서 설치, 실행, 결과 파일을 먼저 확인하고, 수강생 환경은 RTX 4090 RunPod에서 짧게 재현합니다. H100의 시간과 메모리 수치는 학생용 예상 범위에 넣지 않습니다. 새 결과를 만들 때는 GPU와 Pod가 다른 작업과 겹치지 않게 한 장을 지정하고 빈 체크아웃에서 시작합니다.

```bash
# 리허설할 커밋의 저장소 루트
export CUDA_VISIBLE_DEVICES=0
export PORT=30123
source common/env.sh
bash common/setup_pod.sh

# Session 1 · HF forward, Session 1–3 · HTTP, batching, prefix
python3 session01_fundamentals/code/manual_decode.py --experiment batch --out results/s01_manual.csv
bash common/launch.sh baseline --chunked-prefill-size 4096
python3 session01_fundamentals/code/first_requests.py
bash session02_architecture_batching/code/sweep_concurrency.sh rehearsal "1 8" 512 256
python3 session02_architecture_batching/code/bench_to_csv.py results/s02_rehearsal.jsonl
python3 session03_kvcache_radix/code/prefix_basics.py
bash common/stop.sh

# Session 4–6 · 간섭, graph, trace, CSV 소비
bash session04_scheduler/code/interference_sweep.sh
BS="1 16" bash session05_execution/code/onebatch_sweep.sh base
BS="1 16" bash session05_execution/code/onebatch_sweep.sh nograph --cuda-graph-backend-decode disabled
python3 session05_execution/code/onebatch_to_csv.py results/s05_onebatch.jsonl
bash session06_benchmark_production/code/profile_decode.sh c1 1
python3 session06_benchmark_production/code/trace_summary.py profiles/c1
python3 session06_benchmark_production/code/cost_calc.py results/s02_rehearsal.csv \
  --usd-per-hour 0.60 --throughput output_throughput --load max_concurrency \
  --slo p95_ttft_ms=500 --slo p95_itl_ms=50
```

`logs/env.txt`와 결과를 한 묶음으로 보관하고, 실행한 커밋과 펼칠 자료를 기록합니다. 입출력 길이, 성공 요청 수, 빈 CSV, 잘못된 숫자를 먼저 확인합니다. Session 4는 긴 요청이 기존 스트림과 겹친 실행만 비교합니다. Session 6은 Perfetto에서 받은 파일을 한 번 열어 봅니다.

RTX 4090 Pod에서는 수강생과 같은 이미지로 아래 항목을 확인합니다.

| 회차 | 4090에서 확인할 것 |
|---|---|
| 1 | 새 Pod 준비 시간, HF와 SGLang 실행 및 결과 내려받기 |
| 2 | 필수 sweep 소요 시간과 로그, 두 p95 열과 완료 요청 수 |
| 3 | 자동 설정으로 선택된 pool 크기와 page size, 반복 요청의 `cached_tokens` |
| 4 | 두 조건의 실제 overlap, retraction 예시 로그 확보, 최대 ITL과 긴 요청 TTFT 비교 |
| 5 | 기본 graph가 적용되는 batch 크기, CUDA graph ON/OFF. 선택 FP8 실습은 4090에서 별도 확인 |
| 6 | trace 다운로드와 비용 분석. ON/OFF trace와 Session 2 기준 CSV를 수업 전에 준비 |

수강생이 받을 기준 데이터는 `env.txt`, 명령, 원시 로그와 CSV, 작은 trace 하나를 같이 둡니다. 예측 문제에는 정답표 대신 어떤 근거를 봐야 하는지 적고, 수치는 장비가 표시된 예시로 제공합니다. 개인 계정의 한도나 장비 장애로 실행이 어려우면 같은 지표를 기준 데이터에서 분석해 제출할 수 있게 합니다.

### 공지할 것

- RunPod 가입, $20 크레딧, **CUDA 필터** 사용법 ([runpod_guide.md](runpod_guide.md))
- 레포 fork, 로컬에서 mock 서버 동작 확인 (Session 1 사전 준비)
- 템플릿 방식(A/B) 확정

## 2. 매 회차 운영

### 발표 (15분)

- PR 마감(수업 24시간 전) 후 **서로 다른 결과/해석**을 보인 2–3명을 고릅니다. 잘한 순서가 아닙니다.
- 예측이 크게 틀린 과제가 가장 좋은 소재입니다. "왜 틀렸나"가 이론 시간의 도입부가 됩니다.
- 발표 후 질의는 [presentation_template.md](../templates/presentation_template.md)의 질문 3개를 돌아가며 던지게 합니다.

### 이론 (40분)

- `lecture.md`는 수업 + 복습용 완결 자료라 40분에 다 할 수 없습니다. 상단의 **40분 진행안**을 따라가고, 📖 절(코드 발췌, 심화, 자기 점검 문제와 답)은 복습 과제로 안내하세요.
- **칠판에서 직접 계산하는 절**(S1 §5 roofline, S3 §4.4 tree 손 추적, S4 §2.5 admission 예제)은 줄이지 마세요.
- 각 lecture 끝의 "실습 전 예측 질문"은 실습 직전에 학생들이 숫자로 적게 합니다. 실습 후 실제 값과 비교하는 시간이 가장 학습 효과가 큽니다.

### 실습 (55분)

- Pod는 리허설에서 잰 준비 시간만큼 실습보다 앞서 시작합니다. 발표와 강의 앞부분은 로컬에서 진행합니다.
- 서버를 여러 번 띄우는 회차(4, 5)는 스크립트가 자동으로 launch/stop합니다. 기다리는 동안 로그 읽기 질문을 던지세요.
- **종료 10분 전 일괄 "결과 다운로드 → Terminate"** 안내. 이걸 안 하면 반드시 누군가는 밤새 Pod을 켜 둡니다.

### 과제 안내 (10분)

- 과제마다 "GPU 필요 부분"과 "GPU 불필요 부분"이 나뉘어 있습니다. GPU 불필요 부분부터 하라고 안내하세요.
- 예산 상한을 다시 말해 주세요.

## 3. 자주 막히는 지점

| 증상 | 원인 | 대응 |
|---|---|---|
| Pod이 계속 재시작 | 낮은 driver 호스트 | CUDA 필터 13.0+ |
| 서버 시작 3분 이상 | FlashInfer JIT, CUDA graph capture | 정상. 두 번째부터 빨라짐 (같은 Pod 내) |
| `CUDA out of memory` at startup | 이전 서버가 안 꺼짐 | `bash common/stop.sh`, `nvidia-smi` |
| 결과가 옆 사람과 20% 이상 다름 | 다른 호스트 (CPU, 클럭) | `logs/env.txt` 비교. S6 lab 1에서 정식으로 다룸 |
| 과제 그래프를 GPU에서 그림 | 습관 | 로컬 `common/plot.py` 안내, 원칙 4 |
| `sglang.bench_serving` deprecated 경고 | v0.5.23에서 경로 변경 | `sglang.benchmark.serving` 사용 (스크립트는 이미 반영) |

## 4. 채점

- 각 과제 `assignment.md`에 10점 기준표가 있습니다. 최종 프로젝트는 20점.
- 공통 감점: 원시 데이터 없음, 실행 명령 없음, 환경 기록 없음.
- 공통 가점(재량): 예측이 틀린 이유를 코드/로그 근거로 설명한 경우.

## 5. 버전을 올리고 싶다면

SGLang은 매달 릴리스가 나옵니다. 다음 기수에 버전을 올릴 때:

1. `common/env.sh`의 `SGLANG_VERSION` 변경
2. [code_map.md](code_map.md)의 심볼을 새 버전에서 `grep`으로 재확인
3. 플래그 확인: `python3 -m sglang.launch_server --help | grep -E "cuda-graph|chunked|schedule|quantization|speculative"`
4. 로그 형식 확인: `common/logparse.py`로 새 로그를 파싱해 필드가 다 나오는지
5. 벤치마크 경로 확인: `python3 -m sglang.benchmark.serving --help`
6. 각 lab을 리허설

## 6. 이전 자료와의 관계

이 레포는 이전 스터디 자료(`mlleo/metacode_study`)를 참고해 재구성했습니다. 주요 변경:

- 8회 → 6회로 재편, 각 회차를 "발표 → 이론 → 실습 → 과제" 구조로 통일
- 코드는 md 안 조각 대신 **실행 가능한 스크립트**(`common/`, `sessionNN/code/`)로 분리, mock 서버로 로컬 디버깅 가능
- SGLang 버전 고정(v0.5.23)과 코드 경로 재확인 (예: v0.5.23의 기본 prefix cache는 `UnifiedRadixCache`, 기본 스케줄 정책은 `fcfs`, `checkpoint`가 `cache_finished_req`를 대체, 벤치마크 경로 `sglang.benchmark.*`)
- 이전 자료의 HiCache L3/RDMA, 브레이크포인트 디버깅 등 고급 주제는 Session 6 §5의 지도로 축소 (최종 프로젝트 주제로 선택 가능)
- 예산 $10–15 → $20 기준으로 재배분, 회차별 상한 명시
