# Session NN 과제 — <이름 / GitHub ID>

## 0. 한 줄 요약
> 예: Qwen3-1.7B / RTX 4090에서 동시성을 1→64로 올리면 output throughput은 11배, p95 TTFT는 6배 증가했다. SLO(p95 TTFT < 300 ms) 기준 sweet spot은 동시성 16.

## 1. 실험 환경
`logs/env.txt` 내용을 그대로 붙이세요.

```text
date:
GPU / driver:
sglang / torch / cuda:
model:
```

서버 실행 명령 (로그 첫 줄 `# CMD:` 복사):

```bash
python3 -m sglang.launch_server --model-path ... 
```

## 2. 질문과 예측 (실행 전에 작성!)

| 질문 | 예측 (숫자로) | 근거 |
|---|---|---|
| | | |

## 3. 실험 설계

- 고정한 것: GPU, 모델, 입력 길이, 출력 길이, 요청 수, sampling (temperature=0, ignore_eos)
- 바꾼 것 (하나만):
- 반복 횟수 / warm-up:
- 실행한 명령:

```bash
```

## 4. 결과

| 조건 | metric 1 | metric 2 | ... |
|---|---:|---:|---|
| | | | |

![graph](figures/xxx.png)

## 5. 해석 — I expected / I changed / I measured / I found

- **I expected**:
- **I changed**:
- **I measured**:
- **I found**:

왜 그런 결과가 나왔는가? (이번 회차 이론 + 코드 위치)

## 6. 한계
- 이 실험으로 말할 수 없는 것:

## 7. GPU 사용 기록

| 날짜 | GPU | 시작 | 종료 | 단가 ($/h) | 비용 | 한 일 |
|---|---|---|---|---:|---:|---|
| | | | | | | |
