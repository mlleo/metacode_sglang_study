# Session 3 — KV Cache 메모리 관리 & RadixAttention

> **핵심 질문:** KV 메모리를 단편화 없이 나눠 쓰고, 이미 계산한 prompt를 다시 계산하지 않으려면?

## 학습 목표

- [ ] 순진한 KV 할당의 단편화 문제와 두 단계 풀(ReqToTokenPool → TokenToKVPool)을 그림으로 설명할 수 있다
- [ ] 서버 로그의 KV pool 크기를 config로 손계산해 검증할 수 있다
- [ ] radix tree의 match / split / insert / evict를 손으로 추적할 수 있다
- [ ] `cached_tokens`가 왜 `prompt − 1`이 되는지, 출력도 캐시된다는 것을 설명할 수 있다
- [ ] LPM과 FCFS의 차이, 그리고 v0.5.23 기본값이 무엇인지 안다
- [ ] prefix cache가 깨지는 실무 패턴(앞부분이 변하는 프롬프트, 재토크나이즈, template)을 안다

## 타임테이블 (160분)

| 시간 | 순서 | 자료 |
|---|---|---|
| 0:00–0:25 | **Session 2 과제 발표** — capacity 숫자 비교, 다이어그램 리뷰 | |
| 0:25–1:15 | **이론** §0–§7 | [lecture.md](lecture.md) |
| 1:15–1:25 | 휴식 — Pod Deploy + setup (이 동안 lab 0번 시뮬레이터는 로컬에서) | |
| 1:25–2:30 | **실습** 1–6 | [lab.md](lab.md) |
| 2:30–2:40 | **과제 안내** | [assignment.md](assignment.md) |

## 강사 체크리스트

- [ ] §4 손 추적 예제를 칠판에 미리 그려 두고, 학생이 다음 단계를 맞히게 진행
- [ ] lab 5번(`policy_compare.sh`)은 서버를 두 번 띄워 6–8분 걸림 → 시작 후 기다리는 동안 lab 4 결과 토론
- [ ] 리허설에서 lab 5의 LPM/FCFS 차이가 보이는 pool 크기를 미리 확인 (24000이 기본, 안 보이면 16000)
- [ ] Session 2 발표에서 나온 "입력 2K → capacity 감소" 설명과 오늘 §1(KV 메모리)을 연결

## 이번 회차 GPU 예산: $2.0 (수업 $0.8 + 과제 $1.2)
