# Session 5 — Model Execution & 커널 수준 최적화

> **핵심 질문:** GPU 위에서 decode step 하나는 무엇으로 이루어지고, 각 최적화는 그중 무엇을 줄이는가?

## 학습 목표

- [ ] `ScheduleBatch → ForwardBatch`의 주요 필드(`input_ids`, `positions`, `seq_lens`, `out_cache_loc`, `extend_*`)를 예제로 채울 수 있다
- [ ] attention backend의 역할과 EXTEND/DECODE의 차이를 설명할 수 있다
- [ ] decode step 시간 분해식을 쓰고, 측정값으로 "달성 대역폭"을 계산할 수 있다
- [ ] CUDA graph가 줄이는 것, 동작 조건(고정 shape, padding, max_bs), 대가를 설명할 수 있다
- [ ] FP8 weight / FP8 KV 양자화의 효과를 메모리와 속도로 구분해 측정할 수 있다
- [ ] speculative decoding이 이득인 조건과 손해인 조건을 설명할 수 있다

## 타임테이블 (160분)

| 시간 | 순서 | 자료 |
|---|---|---|
| 0:00–0:25 | **Session 4 과제 발표** — 추천 설정이 다른 2–3명 | |
| 0:25–1:15 | **이론** §0–§7 | [lecture.md](lecture.md) |
| 1:15–1:25 | 휴식 — Pod Deploy + setup | |
| 1:25–2:30 | **실습** 0–6 | [lab.md](lab.md) |
| 2:30–2:40 | **과제 안내** — 최종 프로젝트로 이어진다는 점 강조 | [assignment.md](assignment.md) |

## 강사 체크리스트

- [ ] 리허설에서 `--quantization fp8`, `--kv-cache-dtype fp8_e4m3`, NGRAM이 **4090 + v0.5.23에서 시작되는지** 확인. 안 되는 조합은 수업 전에 공지
- [ ] §3 식을 칠판 한쪽에 계속 남겨 두고, 실습 결과가 나올 때마다 해당 항에 숫자를 적기
- [ ] Session 1 발표에서 칠판에 남겨 둔 "HF vs SGLang 가설" 목록을 오늘 마무리
- [ ] 이번 회차가 GPU를 가장 많이 씀 ($1.0) → 실습 시작 시 잔액 확인

## 이번 회차 GPU 예산: $2.5 (수업 $1.0 + 과제 $1.5)
