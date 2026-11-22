# Inside an LLM Inference Engine — SGLang 스터디 (6회차)

> **목표:** "SGLang을 써봤다"가 아니라, 어떤 inference engine을 만나도
> *batch는 어떻게 만들고, KV cache는 어떻게 관리하고, GPU에서 병목은 어디인가*를
> **측정으로** 설명할 수 있게 되는 것.

| 항목 | 내용 |
|---|---|
| 대상 | SGLang을 처음 써보는 부트캠프 수강생 (Python, Linux 기초, Transformer 큰 그림 정도) |
| 형식 | 6회 × 2시간 (지난 과제 발표 → 이론 → 실습 → 과제 안내) |
| 기준 버전 | **SGLang v0.5.23** (2026-10-28 릴리스) — 모든 코드 리딩 경로와 플래그가 이 버전 기준 |
| 표준 모델 | `Qwen/Qwen3-1.7B` (BF16, GQA 16:8) — 필요 시 `Qwen/Qwen3-0.6B` |
| 표준 GPU | RunPod **RTX 4090 24GB** (Community Cloud), CUDA 13 필터 필수 |
| 예산 | 1인 **$20** — 회차별 배분은 [docs/runpod_guide.md](docs/runpod_guide.md#예산-20-배분) |

---

## 커리큘럼

| 회차 | 주제 | 핵심 질문 | 실습 하이라이트 | 과제 |
|---|---|---|---|---|
| [1](session01_fundamentals/) | LLM 추론의 기초 & 첫 SGLang 서버 | token 하나는 어떻게 만들어지고, 왜 `generate()`로는 서빙이 안 되는가? | 직접 짠 decode loop로 prefill/decode 분리 측정, HF vs SGLang | KV 계산 + baseline 비교 리포트 |
| [2](session02_architecture_batching/) | SGLang 아키텍처 & Continuous Batching | 요청 하나는 서버 안에서 어떤 길을 지나고, 여러 요청은 어떻게 한 batch가 되는가? | 서버 로그로 iteration 관찰, 동시성 sweep → latency-throughput 곡선 | SLO 기준 sweet spot 찾기 + 아키텍처 다이어그램 |
| [3](session03_kvcache_radix/) | KV Cache 메모리 관리 & RadixAttention | 이미 계산한 prompt를 다시 계산하지 않으려면? | `cached_tokens` 관찰, prefix 길이 sweep, 멀티턴 재사용, LPM vs FCFS | radix tree 손 추적 + prefix 효과 곡선 |
| [4](session04_scheduler/) | 스케줄러 심화: Admission · Retraction · Chunked Prefill | 메모리가 모자랄 때, 긴 프롬프트가 끼어들 때 스케줄러는 무엇을 하는가? | 작은 KV pool로 retraction 유도, 긴 prefill이 다른 요청 ITL을 망가뜨리는 것 재현 | Long-prompt interference 리포트 |
| [5](session05_execution/) | Model Execution & 커널 수준 최적화 | GPU 위에서 한 step은 무엇으로 이루어지고, 무엇을 줄일 수 있는가? | `one_batch`로 step 시간 분해, CUDA graph / attention backend / FP8 / speculative decoding ablation | 1-page Optimization Note |
| [6](session06_benchmark_production/) | Benchmarking · Profiling · 비용 | "30% 빨라졌다"를 어떻게 믿게 만들고, 1M token에 얼마가 드는가? | torch profiler trace 읽기, Prometheus metrics, $/1M tokens 비교 | **최종 프로젝트** |

각 회차 폴더 구성:

```text
sessionNN_xxx/
├── README.md       # 회차 개요, 타임테이블, 사전 준비, 진행 체크리스트
├── lecture.md      # 이론: 수업용(★) + 복습용(📖) 완결 자료 — 코드 발췌, 풀이 예제, 자기 점검 문제
├── lab.md          # 실습 가이드 (명령어 그대로 복붙 → 결과 해석)
├── assignment.md   # 과제 명세 + 채점 기준 + 다음 회차 발표 형식
└── code/           # 실습/과제용 스크립트
```

---

## 매 회차 진행 방식 (120분)

| 시간 | 순서 | 내용 |
|---|---|---|
| 0:00–0:15 | **지난 과제 발표** | 2명 × 5분 발표 + 질의 (1회차는 오리엔테이션·환경 점검으로 대체). **이 시간에 Pod Deploy + setup 시작** |
| 0:15–0:55 | **이론** | `lecture.md`의 ★ 절 — 핵심 질문 → 개념 → 코드 → 예측 질문 |
| 0:55–1:50 | **실습** | `lab.md`의 필수 실습 — GPU ON → 측정 → 결과 저장 → **GPU OFF** |
| 1:50–2:00 | **과제 안내** | `assignment.md` — 무엇을, 어떤 형식으로, 다음 시간에 누가 발표하는가 |

- `lecture.md`는 **수업 + 복습용 완결 자료**입니다. ★ 절이 수업 핵심, 📖 절은 복습·심화(코드 발췌, 풀이 예제, 자기 점검 문제와 답). 강사는 필요한 절만 골라 씁니다.
- `lab.md`는 2시간 수업에 다 하기엔 많습니다. 각 회차 `README.md`의 "실습" 칸에 적힌 번호가 필수이고, 나머지는 선택·과제·강사 시연용입니다.

---

## 빠른 시작 (수강생)

```bash
# 0) 로컬(노트북)에서: 레포 fork → clone
git clone https://github.com/<your-id>/metacode_sglang_study.git
cd metacode_sglang_study

# 1) 로컬에서 GPU 없이 스크립트 디버깅 (mock 서버) — GPU 비용 0원
python common/mock_server.py &                 # 가짜 SGLang 서버
python common/client.py --input-len 512        # 측정 클라이언트가 잘 도는지 확인

# 2) RunPod Pod 안에서 (docs/runpod_guide.md 참고)
git clone https://github.com/<your-id>/metacode_sglang_study.git && cd metacode_sglang_study
bash common/setup_pod.sh                       # driver 확인, SGLang 0.5.23 설치, 모델 다운로드
bash common/launch.sh baseline                 # 서버 시작 (logs/server_baseline.log)
python common/client.py --prompt "The capital of France is" --allow-eos
bash common/stop.sh
```

## 공통 도구 (`common/`)

| 파일 | 용도 |
|---|---|
| `env.sh` | 버전·모델·포트·경로 공통 변수 |
| `setup_pod.sh` | Pod 첫 셋업 (driver 확인 → 설치 → 모델 다운로드 → 환경 기록) |
| `launch.sh` / `stop.sh` | 서버를 tag별 로그와 함께 띄우고 끄기 |
| `client.py` | 단일 요청 + TTFT/ITL/E2E/`cached_tokens` 측정 (라이브러리 겸용) |
| `loadgen.py` | closed-loop(동시성) / open-loop(도착률) 부하 생성, prefix 공유 workload |
| `metrics.py` | p50/p95/p99, CSV 저장 |
| `logparse.py` | 서버 로그(Prefill/Decode/Retract) → 요약·CSV |
| `kvcalc.py` | `config.json` → KV bytes/token, GPU에 들어가는 token 수 |
| `plot.py` | 결과 CSV → 그래프 PNG (로컬에서 실행) |
| `mock_server.py` | GPU 없이 스크립트를 개발하기 위한 가짜 서버 |

## 문서

- [docs/runpod_guide.md](docs/runpod_guide.md) — GPU 선택, CUDA 13 필터, 템플릿, Stop vs Terminate, **$20 예산 배분**
- [docs/code_map.md](docs/code_map.md) — SGLang v0.5.23 코드 리딩 지도 (회차별로 어떤 파일을 읽는지)
- [docs/glossary.md](docs/glossary.md) — 용어집
- [docs/instructor_guide.md](docs/instructor_guide.md) — 강사용: 사전 리허설, 발표 운영, 채점, 자주 막히는 지점
- [submissions/README.md](submissions/README.md) — 과제 제출 방법 (fork → PR)

## 스터디 원칙

1. **"왜"를 공부한다.** 플래그 이름을 외우지 말고, 그 플래그가 없으면 무엇이 반복 계산되는지 설명한다.
2. **모든 주장에는 측정값이 있다.** "빨라진 것 같다" ❌ → "p50 TTFT 412 ms → 96 ms" ✅
3. **한 번에 변수 하나만 바꾼다.** GPU·모델·입출력 길이·동시성·sampling을 고정하고 하나만 바꾼다.
4. **GPU는 측정 장비다.** 코드는 로컬에서 mock 서버로 다 짜 오고, Pod에서는 실행·저장·종료만 한다.
5. **예측 먼저, 측정은 나중에.** 실행 전에 결과를 숫자로 예측하고 적는다. 틀린 예측이 가장 많이 가르친다.

> Build less. Measure more. Understand why.
