# Session 4 — 스케줄러 심화: Admission · Retraction · Chunked Prefill · Overlap

> **핵심 질문:** 메모리가 모자랄 때, 긴 프롬프트가 끼어들 때, CPU가 병목일 때 스케줄러는 무엇을 하는가?

## 학습 목표

- [ ] 한 iteration에서 `PrefillAdder`가 쓰는 세 가지 예산과 `new_token_ratio`를 설명하고, 승인 수를 손으로 계산할 수 있다
- [ ] retraction이 언제, 누구에게 일어나고, 비용이 무엇인지 로그로 보일 수 있다
- [ ] `ignore_eos` 벤치마크가 스케줄러 동작을 바꾼다는 것을 안다
- [ ] head-of-line blocking을 재현하고 chunked prefill / mixed chunk의 TTFT ↔ ITL 교환을 측정할 수 있다
- [ ] overlap scheduling이 무엇을 숨기는지 설명할 수 있다
- [ ] 로그로 "메모리 부족 vs 설정 상한 vs 긴 prefill 간섭"을 진단할 수 있다

## 타임테이블 (160분)

| 시간 | 순서 | 자료 |
|---|---|---|
| 0:00–0:25 | **Session 3 과제 발표** — 손 추적 vs 시뮬레이터 차이, 공유 비율 곡선 | |
| 0:25–1:15 | **이론** §0–§6 | [lecture.md](lecture.md) |
| 1:15–1:25 | 휴식 — Pod Deploy + setup | |
| 1:25–2:30 | **실습** 1–4 | [lab.md](lab.md) |
| 2:30–2:40 | **과제 안내** | [assignment.md](assignment.md) |

## 강사 체크리스트

- [ ] §2 숫자 예제를 학생과 함께 계산 (과제 Part A와 같은 형식)
- [ ] 리허설에서 lab 1의 `eos`가 실제로 retraction을 일으키는지 확인. 안 생기면 pool 크기/`max_new_tokens` 조정값을 공지
- [ ] lab 1, 2는 각각 서버를 3–4번 띄움 → 학생들이 동시에 시작하도록 안내 (기다리는 시간에 로그 읽기 과제 부여)
- [ ] "p50은 같은데 p99만 다르다"는 순간을 꼭 짚기 — Session 6 벤치마크 방법론으로 이어짐

## 이번 회차 GPU 예산: $2.0 (수업 $0.8 + 과제 $1.2)
