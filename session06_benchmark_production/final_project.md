# 최종 프로젝트 — 하나의 inference 질문에 측정으로 답하기

> 기간: Session 6 이후 2주 (강사가 공지) · GPU 예산 **$4.0** (+ 남은 예비비)
> 제출: `submissions/final/<github-id>/` PR · 발표: 7분 + 질의 3분 (데모데이 또는 온라인 — 강사 공지)

## 원칙

- 거대한 서비스를 만드는 것이 아닙니다. **작고 명확한 시스템 질문 하나**를 고릅니다.
- 좋은 질문: "공유 prefix 비율이 몇 %부터 radix cache가 throughput을 2배 이상 올리는가?"
- 너무 큰 질문: "SGLang으로 production 챗봇을 만든다."
- 한 가지 기준 설정과 두세 개 비교점이면 충분합니다. Session 6 §1에 따라 결과를 다시 측정하고 불확실성을 함께 설명하세요. 수업 결과는 같은 환경과 workload일 때 기준선 후보로 쓸 수 있습니다.

## 주제 예시 (골라도 되고, 변형해도 되고, 새로 제안해도 됩니다)

| | 주제 | 질문 | 출발점 |
|---|---|---|---|
| A | Radix Cache Explorer | 동시성과 메모리를 고정하면 공유 비율에 따라 cache 이득이 어떻게 변하나? | S3 과제 Part B |
| B | Capacity Planner | 입력/출력 길이 분포(예: ShareGPT 실제 분포 `--dataset-name sharegpt`)에서 SLO 만족 capacity는? random-ids 결과와 얼마나 다른가? | S2 과제 |
| C | Long-Prompt Interference | chunk 크기·mixed chunk·`max-prefill-tokens` 조합으로 TTFT/ITL Pareto front 그리기 | S4 과제 |
| D | Optimization Ablation | Session 5의 비교를 다른 입력 길이 하나에서 재현하면 효과가 유지되나? | S5 과제 |
| E | Cost-aware Configuration | 4090 vs L4 vs (선택) A40/L40S에서 SLO 만족 $/1M tokens 비교. 비싼 GPU가 더 쌀 수 있는가? | S6 §4 |
| F | Scheduler Hack | SGLang 소스를 고쳐(예: retraction 정렬 키, LPM 임계값 128, eviction 정책) 전후 비교. editable 설치 필요 | S4 코드 리딩 |
| G | SGLang vs 다른 엔진 | 같은 GPU·모델·workload에서 vLLM과 비교. 순위가 아니라 **workload에 따라 왜 달라지는가** | S1–S5 전부 |
| H | Speculative decoding 심화 | NGRAM vs EAGLE3(공개 draft가 있는 모델로) — accept length와 동시성의 관계 | S5 |
| I | Structured output 비용 | JSON schema 제약 생성이 throughput/ITL에 주는 비용은? 같은 입력에서 간단한 스키마와 복잡한 스키마의 차이는? | S6 §5 |

> F, G는 설치·환경 문제로 시간이 더 듭니다. 강사와 먼저 상의하세요.

## 제출 구조

```text
submissions/final/<github-id>/
├── README.md         # 아래 8단 구조
├── run.sh            # 모든 실험을 처음부터 재현하는 명령 (서버 launch 포함)
├── code/             # 직접 쓴 스크립트
├── results/          # 원시 CSV/JSONL (+ 서버 로그 요약)
└── figures/          # 그래프
```

`README.md` 8단 구조:

1. **Question** — 한 문장
2. **Hypothesis** — 숫자로. 어떤 이론(회차/절)에 근거했는가
3. **Setup** — 환경(`env.txt`), 서버 인자 전체, workload, 부하 모델, 반복 횟수
4. **Baseline** — 비교 기준
5. **Change** — 바꾼 것 (하나의 축, 또는 명시적인 2축 grid)
6. **Result** — 그래프 2–3개, 표 (비교 결과와 95% CI. 같은 seed로 짝지었다면 쌍별 차이의 구간)
7. **Explanation** — 왜? 로그/metrics/profile 근거 하나 이상
8. **Limitations & Next** — 말할 수 없는 것, GPU 30분이 더 있다면 할 실험

## 발표 (7분)

Problem → Hypothesis → Experiment → Result (그래프 1장) → Why → What I would test next.
[templates/presentation_template.md](../templates/presentation_template.md)의 질의 가이드로 서로 질문합니다.

## 채점 기준 (20점)

| 항목 | 점수 | 기준 |
|---|---:|---|
| 질문의 명확성 | 3 | 한 문장, 측정 가능, 범위가 적절 |
| 실험 설계 | 4 | 체크리스트 전부, 변수 통제, 반복·CI |
| 재현성 | 3 | `run.sh` 하나로 재실행 가능, 원시 데이터 포함 |
| 결과와 해석 | 6 | 이론과 연결, 근거(로그/profile), 예측과의 차이 설명 |
| 한계 인식 | 2 | 일반화할 수 없는 부분을 정확히 |
| 비용 관리 | 2 | GPU 기록, 예산 준수, 작은 리허설 → 본 실험 순서 |

## 일정 제안

| 시점 | 할 일 |
|---|---|
| Session 6 직후 | 주제 확정 → 이슈로 등록 (`[Final] <이름>: <질문>`) → 강사 피드백 |
| +3일 | 작은 설정으로 파이프라인 완성 (GPU $0.5 이내) |
| +10일 | 본 실험 + 반복 |
| +14일 | README + 발표 |
