# 실습과 과제 예시 결과

한 장의 H100에서 여섯 회차의 필수 실습과 기본 과제 경로를 실행했습니다. 수업에서 결과를 어떻게 읽고 보고서에 옮기는지 보여 주는 예시입니다. RTX 4090에서 얻은 값과 숫자 자체를 비교하기 전에 GPU, backend, 동시성, 입력·출력 길이와 측정 범위를 맞추세요.

| 회차 | 실습에서 찾아볼 것 | 과제에서 찾아볼 것 |
|---|---|---|
| [1. 첫 요청과 추론 시간](session01.md) | HF 수동 decode·generate, SGLang 첫 요청과 측정 범위 | KV·roofline 계산, 두 그래프, 해석 문장 |
| [2. 동시성과 SLO](session02.md) | 유휴 스택, 요청 로그와 타임라인, 동시성·도착률 sweep | 512·2,048의 SLO, 팀장 보고, 요청 경로 그림 |
| [3. Radix tree와 공유 prefix](session03.md) | KV pool, cold/warm, 멀티턴, LPM·FCFS | 손 추적, 공유율 곡선, 코드 리딩, 선택 대조군 설계 |
| [4. 긴 입력과 출력 끊김](session04.md) | admission·retraction 로그, 16K 간섭, overlap | 슬롯 계산, 8K·16K SLO, 타임라인, 코드 리딩 |
| [5. CUDA graph와 실행 최적화](session05.md) | decode step·graph 상한, backend, FP8, NGRAM | 기본 ON/OFF 노트, 다른 최적화에서 볼 지표 |
| [6. 반복·trace·비용](session06.md) | 반복 변동, Perfetto, Prometheus, SLO를 만족하는 비용 | 최종 프로젝트 리허설, 짝별 비교와 신뢰구간 |

실측한 실험에는 CSV나 로그를 연결했습니다. 측정하지 않은 선택 변형에는 고정할 조건과 읽을 지표를 적었습니다. 최종 프로젝트는 입력 길이 한 가지를 바꾼 완성 예시를 싣습니다.

## 실행 환경

| 항목 | 값 |
|---|---|
| GPU / 드라이버 | H100 SXM 80 GB HBM3 1장 / 580.126.16 |
| SGLang / PyTorch / Python | 0.5.23 / 2.14.1+cu130 / 3.12.16 |
| 모델 | Qwen3-1.7B BF16, HF revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| 기본 attention backend | FA3 |
| 메모리 대역폭 기준 | 3,350 GB/s |

H100에서는 이미 설치된 SGLang과 해시를 확인한 모델 캐시를 썼습니다. 새 RTX 4090 RunPod의 설치 시간, 다운로드, 과금은 강사 리허설에서 따로 확인합니다. [강사 가이드](../instructor_guide.md)

표와 그래프의 입력 파일은 [data/](data/)에 있습니다. 로컬 저장소에서 `bash docs/worked-results/plot_examples.sh`를 실행하면 예시 그래프를 다시 그릴 수 있습니다. 파일을 바꿔 직접 그린 그림이나 보고서에는 GPU와 설정을 함께 적으세요. 같은 명령을 다시 실행할 때는 이전 파일을 복사해 두면 누적된 행을 새 반복으로 착각하는 일을 피할 수 있습니다.

## 내 결과를 읽는 순서

1. **비교 조건 확인.** GPU·모델·입출력 길이·동시성·바꾼 플래그를 표로 적습니다. 요청 수와 완료 수가 맞는지 확인하고, 실패한 실행은 원인과 함께 따로 기록합니다.
2. **사용자 지표 찾기.** 빨리 시작해야 하면 TTFT, 출력이 끊기면 ITL, 비용이 질문이면 output tok/s와 요금을 봅니다. 집계 기간과 p50·p95·p99 중 무엇을 골랐는지도 적습니다.
3. **서버에서 근거 찾기.** `Prefill batch`, `Decode batch`, `#cached-token`, `#queue-req`를 CSV와 대조합니다. 시간 차이가 어디서 생겼는지 알고 싶으면 trace에서 같은 step의 CPU 호출과 GPU 커널을 같이 엽니다.
4. **한 문장으로 남기기.** 무엇을 바꿨는지, 어떤 값이 얼마만큼 변했는지, 로그나 trace에서 무엇과 맞았는지 순서대로 씁니다. 실행 간 차이가 작거나 다른 설명도 가능하면 무엇을 한 번 더 측정할지 적습니다.
