# Session 2 — SGLang 아키텍처 & Continuous Batching

> **핵심 질문:** 요청 하나는 서버 안에서 어떤 길을 지나고, 제각각 도착하는 여러 요청은 어떻게 한 batch가 되는가?

## 학습 목표

- [ ] SGLang 서버의 3개 프로세스(TokenizerManager / Scheduler / DetokenizerManager)와 역할을 그릴 수 있다
- [ ] 요청 하나의 일생을 함수 이름 수준으로 따라갈 수 있다
- [ ] 스케줄러 루프 한 iteration과 "prefill 우선" 규칙을 설명할 수 있다
- [ ] static batching과 continuous batching의 차이를 타임라인으로 보여줄 수 있다
- [ ] closed-loop / open-loop 부하의 차이, Little's law, SLO·goodput을 설명할 수 있다
- [ ] `sglang.benchmark.serving`으로 sweep을 돌리고 latency-throughput 곡선을 해석할 수 있다

## 타임테이블 (120분)

| 시간 | 순서 | 자료 |
|---|---|---|
| 0:00–0:15 | **Session 1 과제 발표** (2명) — "SGLang이 HF보다 빠른 이유" 가설 | |
| 0:15–0:55 | **이론** — ★ 절 (§0–§6). 📖 절은 복습 자료 (발표 시작 전에 Pod Deploy + setup을 걸어 두면 이론 중에 설치가 끝남) | [lecture.md](lecture.md) |
| 0:55–1:50 | **실습** — 실습 2·3·4·6 (1, 5는 선택/과제) | [lab.md](lab.md) |
| 1:50–2:00 | **과제 안내** | [assignment.md](assignment.md) |

## 발표 진행 팁 (강사)

발표에서 나온 가설을 칠판에 적고, 오늘 이론으로 지울 수 있는 것은 지웁니다.
남는 것(예: "CUDA graph 때문", "attention 커널 때문")은 Session 5까지 붙여 둡니다.

## 강사 체크리스트

- [ ] 리허설에서 동시성 sweep 소요 시간 측정 (반 전체 기준 시간 안내)
- [ ] 리허설 결과로 lab 4번의 "용량 근처 도착률"을 미리 정해 두기 (lab 5번 `sweep_rate.sh` 인자)
- [ ] §5 latency-throughput 곡선은 칠판에 먼저 그려 놓고, 실습 후 실제 데이터로 겹쳐 그리기

## 이번 회차 GPU 예산: $2.0 (수업 $0.8 + 과제 $1.2)
