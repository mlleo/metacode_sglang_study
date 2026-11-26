# 최종 프로젝트 실행 가이드

[과제 일정과 채점 기준](../session06_benchmark_production/final_project.md)을 먼저 읽으세요. 이 폴더의 runner는 기준 서버 기동, 트래픽 생성, 원시 데이터 수집, 서버 종료를 한 번에 처리합니다. 프로젝트 데이터는 길이가 통제된 토큰 ID로 만듭니다. 실험은 서빙 성능을 질문으로 삼습니다.

## 1. 어떤 트래픽을 보내나

| 트랙 | `standard` | `stress`에서 바꾸는 것 | 판정 |
|---|---|---|---|
| `prefix` | 요청 96개, client 동시성 32, 입력 2,048 중 공유 prefix 1,536, 출력 64, prefix 그룹 8개 | 그룹을 24개로 늘려 같은 pool에 더 많은 prefix가 경쟁하게 합니다. | 모두 완료한 조건에서 p95 TTFT가 짧은 설정. `cache_hit_rate`, `uncached_input_tokens`, output tok/s로 차이를 설명합니다. |
| `mixed` | 채팅 16개(입력 256, 출력 512)가 시작한 뒤 문서 1개(입력 16,000, 출력 8)를 보냅니다. | 문서 2개를 함께 보냅니다. | 채팅 p99 ITL과 최장 출력 청크 공백이 한도 안에 든 설정에서 문서 평균 TTFT가 짧은 설정. 실제로 겹친 요청 수와 타임라인을 확인합니다. |
| `batch` | 요청 96개를 하나의 작업으로 올립니다. client worker 32개, 입력 512, 출력 128입니다. | 네 요청 중 하나의 출력만 384로 늘립니다. | 전체 작업이 마감 안에 끝난 설정에서 output tok/s가 높은 조건. 배치 제출부터 잰 `wall_s`와 `job_finish_s_p95`를 읽습니다. |

`mixed`는 채팅 16개의 첫 출력 청크가 도착한 뒤 25 ms에 문서 요청을 보냅니다. 각 문서 요청이 시작할 때 채팅의 최소 75%가 실제로 스트리밍 중이어야 유효합니다. 청크에 토큰 여러 개가 담기면 토큰 간격 추정치와 화면이 잠깐 멈춘 시간이 달라질 수 있습니다. `chat_itl_ms_p99`는 토큰 간격 추정치, `chat_gap_ms_max`는 실제 청크 도착 사이의 최댓값입니다.

`batch`에서 worker를 기다린 시간도 작업 시간에 포함됩니다. 요청별 `http_e2e_ms_p95`는 HTTP 전송부터 받기까지입니다. 마감은 배치 전체의 `wall_s`에 적용합니다.

## 2. GPU 없이 연결 확인

로컬에 Python 3.11 이상과 `requests`를 설치합니다. 그래프를 그리려면 `matplotlib`도 설치합니다. 저장소 루트에서 실행하세요.

```bash
python3 -m pip install requests matplotlib
python3 common/mock_server.py --port 30000
```

다른 터미널에서 본인 트랙으로 바꿔 실행합니다. 결과 경로는 새 이름을 사용하세요.

```bash
python3 final_projects/run.py mixed --profiles pilot \
  --mock-url http://127.0.0.1:30000 --out results/final/mock-mixed
python3 final_projects/analyze.py results/final/mock-mixed
```

결과에 `smoke-only`가 뜨고 `raw/`가 생기면 연결이 된 것입니다. mock은 요청 처리와 파일 생성 확인에 씁니다. GPU 기록과 같은 결과 폴더에 두면 혼동할 수 있으니 `mock-` 이름을 유지합니다. 마치면 첫 터미널에서 `Ctrl-C`로 끕니다.

## 3. GPU 리허설과 후보 비교

수업과 같은 RunPod 이미지를 쓰고 [RunPod 가이드](../docs/runpod_guide.md)에 따라 준비합니다. 아래는 `prefix` 예시입니다. 설정 이름은 [configs.json](configs.json)에서 확인할 수 있습니다. 기본 모델은 검증에 사용한 Qwen3-1.7B revision으로 고정됩니다. 첫 실행에서는 해당 revision의 파일을 내려받을 수 있습니다. [모델 revision](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e)

```bash
bash common/setup_pod.sh
python3 final_projects/run.py prefix --profiles pilot --out results/final/prefix-pilot
python3 final_projects/run.py prefix --configs baseline,lpm,lfu \
  --profiles standard --seeds 11 --out results/final/prefix-screen
```

| 트랙 | `baseline` | 후보 1 | 후보 2 |
|---|---|---|---|
| `prefix` | FCFS + LRU | `lpm` · LPM + LRU | `lfu` · FCFS + LFU |
| `mixed` | chunk 1,024 | `mixed_chunk` · chunk 1,024에서 mixed prefill | `decode_interval` · chunk 1,024에서 prefill 다음 decode 간격 1 |
| `batch` | server running limit 8 | `cap16` · 16 | `cap32` · 32 |

자기 후보를 추가할 땐 `configs.json`을 복사하고 `--config-file`로 넘깁니다. 기준선의 인자와 다른 점을 한 줄로 적으세요. 모델과 HTTP 주소는 전 조건에서 같게 유지합니다. [SGLang v0.5.23 스케줄링 인자](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/arg_groups/fields/schedule.py) · [캐시 인자](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/arg_groups/fields/memory.py)

## 4. 한 후보를 확인하고 stress 결과 읽기

후보는 첫 비교에서 고릅니다. 비교에 쓰지 않은 seed로 기준선과 후보를 세 번씩 측정합니다. 아래는 `lpm`을 골랐을 때의 명령입니다.

```bash
python3 final_projects/run.py prefix --configs baseline,lpm --profiles standard,stress \
  --seeds 22,33,44 --order paired --out results/final/prefix-confirm
```

`--order paired`는 각 seed에서 설정 순서를 번갈아 바꿉니다. 서버 시작 비용을 먼저 확인하세요. 기본 `grouped`는 설정마다 한 번만 기동하고 여러 seed를 처리합니다. 이 모드를 사용했다면 기동 순서와 환경 변화가 결과에 섞일 수 있다는 점을 적으세요. 같은 seed·profile에서 workload 해시가 같아야 합니다. 각 측정 전에 가장 긴 입력과 같은 길이로 별도 요청을 보내 prefill을 워밍업합니다. cache를 비운 뒤 측정과 로그 기록을 시작합니다.

GPU 실험이 끝나면 결과를 다운로드하고 **Terminate**합니다. 아래 분석은 로컬 저장소에서 실행합니다.

```bash
python3 final_projects/analyze.py results/final/prefix-confirm \
  --csv results/final/prefix-confirm/summary.csv --plot figures/final/prefix.png
python3 final_projects/extract_log.py results/final/prefix-confirm lpm standard 22 \
  > results/final/lpm-standard-s22.log
python3 common/logparse.py results/final/lpm-standard-s22.log
```

`mixed`는 `--chat-itl-ms 100 --chat-gap-ms 100`, `batch`는 `--deadline-s 12`가 예시 기본값입니다. 반에서 공지한 값을 **후보를 보기 전에** 정하고 분석 명령에 넣습니다. 한 번이라도 한도를 넘으면 해당 조건은 실패로 표시합니다. 작은 리허설은 `smoke-only`입니다. 반복 세 개에 대한 95% 구간은 후보−기준선의 짝별 차이로 계산합니다. `prefix`와 `mixed`는 음수가 지연 단축이고, `batch`는 양수가 처리량 증가입니다.

`mixed`의 타임라인을 두 조건에서 비교할 수 있습니다. `raw/`의 같은 seed·profile 파일을 선택하세요.

```bash
mkdir -p figures/final
python3 session04_scheduler/code/plot_timeline.py \
  results/final/mixed-confirm/raw/{baseline,mixed_chunk}_standard_s22.timeline.json \
  --out figures/final/mixed-timeline.png
```

## 5. 결과를 읽고 문제를 고치는 순서

1. **`manifest.json`**에서 GPU, 모델, 실행 버전, 플래그, seed를 적습니다. `runs.csv`에서 해시가 같고 예상 요청 수만큼 완료됐는지 확인합니다.
2. 트랙의 성공 기준을 적용합니다. 넘어선 run이 있다면 seed와 값을 적습니다. 차이의 구간이 0 근처에 넓게 걸치면 순위에 확신을 두기 어렵습니다.
3. `extract_log.py`로 해당 구간만 꺼냅니다. `prefix`는 `#cached-token`과 `#new-token`, `mixed`는 문서 도착선과 청크가 비는 구간, `batch`는 `#running-req`와 `#queue-req`를 확인합니다. 로그의 초 단위 시각과 클라이언트 타임라인을 같은 정밀도로 해석하지 않습니다.
4. 시작 실패나 요청 오류가 나면 `logs/`와 `raw/`를 봅니다. 실패 campaign에는 `runs.csv`가 생성되지 않습니다. 원인을 기록하고 새 출력 경로에서 비교를 다시 실행합니다. runner가 멈춘 뒤에도 해당 결과 폴더에 `logs/*/server.pid`가 남았다면 그 폴더만 지정해 `LOG_DIR=... bash common/stop.sh`로 확인합니다.

[트랙별 분석 예시](examples/README.md) · [보고서 템플릿](submission_template.md) · [강사 리허설](instructor.md)
