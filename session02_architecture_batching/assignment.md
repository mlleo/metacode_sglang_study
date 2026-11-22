# Session 2 과제 — "이 서버는 몇 명을 받을 수 있는가?"

> 마감: Session 3 시작 24시간 전 PR · GPU 예산 **$1.2 이내**
> 제출: `submissions/session02/<github-id>/`

## 시나리오

여러분은 사내 코딩 어시스턴트를 RTX 4090 한 장 + Qwen3-1.7B로 운영하려고 합니다. 팀장이 묻습니다.

> "응답 시작이 0.5초 안에, 글자 나오는 속도가 끊기지 않게(token 간격 50 ms 이하) 하려면 **동시에 몇 명까지** 받을 수 있어?
> 그리고 사용자가 질문을 길게 쓰면(입력 2K) 그 숫자는 어떻게 바뀌어?"

## Part A. 측정 (GPU)

SLO: **p95 TTFT ≤ 500 ms, p95 ITL ≤ 50 ms**

| 실험 | 입력 / 출력 | 방법 |
|---|---|---|
| A1 (수업 결과 재사용 가능) | 512 / 256 | 동시성 sweep 1–128 |
| A2 | **2048** / 256 | 동시성 sweep — 같은 목록 |
| A3 | 512 / **1024** | 동시성 sweep — 1, 4, 16, 64만 (비용 절약) |

```bash
bash common/launch.sh baseline
bash session02_architecture_batching/code/sweep_concurrency.sh in2k "1 2 4 8 16 32 64 128" 2048 256
bash session02_architecture_batching/code/sweep_concurrency.sh out1k "1 4 16 64" 512 1024
bash common/stop.sh
```

각 실험에 대해:

```bash
python session02_architecture_batching/code/bench_to_csv.py results/s02_in2k.jsonl --tag in2k
python session02_architecture_batching/code/find_capacity.py results/s02_in2k.csv \
    --slo p95_ttft_ms=500 --slo p95_itl_ms=50
```

## Part B. 분석

1. **그래프 2장**
   - 동시성 → output throughput (A1, A2, A3 세 선)
   - output throughput → p95 TTFT (latency-throughput 곡선, 점마다 동시성 표시)
2. **표**: 실험별 capacity(SLO 만족 최대 동시성)와 그때의 throughput
3. **팀장에게 보내는 답변 3–5문장.** 숫자, 그리고 "어떤 조건이 바뀌면 이 숫자가 틀려지는가"를 포함하세요.
4. **왜?** 입력이 2K로 늘었을 때 capacity가 줄어든 이유를 두 가지 메커니즘으로 설명하세요.
   (힌트 ①: 새 요청의 prefill이 길어지면 기존 요청의 ITL이 어떻게 되나? ② KV pool ≈ 12만 token에 요청 하나가 차지하는 양은?)
   서버 로그(`common/logparse.py`)의 `token usage`·`#queue-req` 최대값을 근거로 쓰세요.
5. A3(출력 1K)는 throughput과 TTFT 중 무엇에 더 큰 영향을 줬나요? Little's law로 설명해 보세요.

## Part C. 아키텍처 다이어그램 (GPU 불필요)

"`POST /generate` 하나가 들어와 첫 token이 클라이언트에 도착하기까지"를 그림으로 그리세요 (손그림 사진, draw.io, Mermaid 모두 OK).

요구사항:

- 프로세스 경계 3개 (TokenizerManager / Scheduler / DetokenizerManager)와 그 사이 통신 방식
- **SGLang v0.5.23의 실제 함수 이름 8개 이상** ([docs/code_map.md](../docs/code_map.md) 참고. 직접 `grep`해서 찾은 것 2개 이상 포함)
- 그림 위에 **TTFT에 들어가는 구간**과 **ITL에 들어가는 구간**을 표시
- `get_next_batch_to_run`이 prefill과 decode 중 무엇을 고르는지에 대한 규칙 한 줄

## 채점 기준 (10점)

| 항목 | 점수 |
|---|---:|
| 측정 재현성 (명령, env.txt, 원시 jsonl) | 2 |
| capacity를 SLO 기준으로 정확히 산출 | 2 |
| 2K 입력 효과를 로그 근거로 두 메커니즘 설명 | 3 |
| 아키텍처 다이어그램 (함수 이름, TTFT/ITL 구간) | 2 |
| GPU 기록, 예산 준수 | 1 |

## 다음 회차 발표

- 서로 다른 capacity 숫자가 나온 2–3명을 고릅니다 (같은 GPU인데 왜 다른가? — 측정 방법·노이즈 토론).
- 다이어그램 1개를 골라 함께 리뷰합니다.
