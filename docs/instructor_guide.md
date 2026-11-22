# 강사 가이드

## 1. 개강 전 (1–2주 전)

### 리허설은 필수입니다

이 레포의 스크립트는 SGLang v0.5.23 소스(플래그 정의, 로그 형식, 벤치마크 도구 인자)를 직접 확인하고 작성했고,
**GPU 없이 mock 서버로 클라이언트·파서·그래프 도구를 검증**했습니다. 하지만 **실제 RTX 4090 + RunPod에서 끝까지 돌려 본 것은 아닙니다.**
수업 전에 강사가 한 번씩 돌려 아래를 확인하고, 숫자를 강의 노트의 "예측 질문" 정답으로 채워 두세요.

| 회차 | 리허설에서 확인할 것 | 실패하면 |
|---|---|---|
| 공통 | CUDA 13 필터로 4090 Pod이 뜨는지, `setup_pod.sh` 소요 시간, 템플릿 A/B 중 무엇을 쓸지 | 필터 13.0+ 확인, 안 되면 `lmsysorg/sglang:v0.5.23-runtime` 이미지(B)로 |
| 1 | `manual_decode.py`의 `logits_to_keep` 인자, `first_requests.py`의 `chat_template_kwargs` | 해당 인자 제거 (주석 참고) |
| 2 | `py-spy dump` 권한, sweep 소요 시간 | py-spy는 선택 실습으로 |
| 3 | `policy_compare.sh`에서 LPM/FCFS 차이가 보이는 pool 크기 | 16000으로 줄여 공지 |
| 4 | `retraction_sweep.sh`의 `eos`에서 retraction이 실제로 일어나는지, `nochunk`에서 OOM 여부 | `--max-new-tokens 2048` / pool 20000 / `--mem-fraction-static 0.6` |
| 5 | `--quantization fp8`, `--kv-cache-dtype fp8_e4m3`, NGRAM이 4090에서 시작되는지 | 안 되는 조합은 lab에서 제외하고 공지 |
| 6 | `/start_profile`로 trace가 생성되는지, 파일 크기 | `num_steps`를 5로 |

리허설 비용은 회차당 약 $1–1.5입니다.

### 공지할 것

- RunPod 가입, $20 크레딧, **CUDA 필터** 사용법 ([runpod_guide.md](runpod_guide.md))
- 레포 fork, 로컬에서 mock 서버 동작 확인 (Session 1 사전 준비)
- 템플릿 방식(A/B) 확정

## 2. 매 회차 운영

### 발표 (25분)

- PR 마감(수업 24시간 전) 후 **서로 다른 결과/해석**을 보인 2–3명을 고릅니다. 잘한 순서가 아닙니다.
- 예측이 크게 틀린 과제가 가장 좋은 소재입니다. "왜 틀렸나"가 이론 시간의 도입부가 됩니다.
- 발표 후 질의는 [presentation_template.md](../templates/presentation_template.md)의 질문 3개를 돌아가며 던지게 합니다.

### 이론 (50분)

- 각 `lecture.md` 맨 위의 시간 배분표를 따르세요. ★ 표시가 없어도 **칠판에서 직접 계산하는 절**(S1 §4, S3 §4, S4 §2)은 줄이지 마세요.
- 매 회차 끝의 "예측 질문"은 실습 직전에 학생들이 숫자로 적게 합니다. 실습 후 실제 값과 비교하는 시간이 가장 학습 효과가 큽니다.

### 실습 (65분)

- **휴식 시간에 Pod Deploy + setup**을 시작하게 하세요. 설치에 3–8분 걸립니다.
- 서버를 여러 번 띄우는 회차(4, 5)는 스크립트가 자동으로 launch/stop합니다. 기다리는 동안 로그 읽기 질문을 던지세요.
- **종료 10분 전 일괄 "결과 다운로드 → Terminate"** 안내. 이걸 안 하면 반드시 누군가는 밤새 Pod을 켜 둡니다.

### 과제 안내 (10분)

- 과제마다 "GPU 필요 부분"과 "GPU 불필요 부분"이 나뉘어 있습니다. GPU 불필요 부분부터 하라고 안내하세요.
- 예산 상한을 다시 말해 주세요.

## 3. 자주 막히는 지점

| 증상 | 원인 | 대응 |
|---|---|---|
| Pod이 계속 재시작 | 낮은 driver 호스트 | CUDA 필터 13.0+ |
| 서버 시작 3분 이상 | FlashInfer JIT, CUDA graph capture | 정상. 두 번째부터 빨라짐 (같은 Pod 내) |
| `CUDA out of memory` at startup | 이전 서버가 안 꺼짐 | `bash common/stop.sh`, `nvidia-smi` |
| 결과가 옆 사람과 20% 이상 다름 | 다른 호스트 (CPU, 클럭) | `logs/env.txt` 비교. S6 lab 1에서 정식으로 다룸 |
| 과제 그래프를 GPU에서 그림 | 습관 | 로컬 `common/plot.py` 안내, 원칙 4 |
| `sglang.bench_serving` deprecated 경고 | v0.5.23에서 경로 변경 | `sglang.benchmark.serving` 사용 (스크립트는 이미 반영) |

## 4. 채점

- 각 과제 `assignment.md`에 10점 기준표가 있습니다. 최종 프로젝트는 20점.
- 공통 감점: 원시 데이터 없음, 실행 명령 없음, 환경 기록 없음.
- 공통 가점(재량): 예측이 틀린 이유를 코드/로그 근거로 설명한 경우.

## 5. 버전을 올리고 싶다면

SGLang은 매달 릴리스가 나옵니다. 다음 기수에 버전을 올릴 때:

1. `common/env.sh`의 `SGLANG_VERSION` 변경
2. [code_map.md](code_map.md)의 심볼을 새 버전에서 `grep`으로 재확인
3. 플래그 확인: `python -m sglang.launch_server --help | grep -E "cuda-graph|chunked|schedule|quantization|speculative"`
4. 로그 형식 확인: `common/logparse.py`로 새 로그를 파싱해 필드가 다 나오는지
5. 벤치마크 경로 확인: `python -m sglang.benchmark.serving --help`
6. 각 lab을 리허설

## 6. 이전 자료와의 관계

이 레포는 이전 스터디 자료(`mlleo/metacode_study`)를 참고해 재구성했습니다. 주요 변경:

- 8회 → 6회로 재편, 각 회차를 "발표 → 이론 → 실습 → 과제" 구조로 통일
- 코드는 md 안 조각 대신 **실행 가능한 스크립트**(`common/`, `sessionNN/code/`)로 분리, mock 서버로 로컬 디버깅 가능
- SGLang 버전 고정(v0.5.23)과 코드 경로 재확인 (예: v0.5.23의 기본 prefix cache는 `UnifiedRadixCache`, 기본 스케줄 정책은 `fcfs`, `checkpoint`가 `cache_finished_req`를 대체, 벤치마크 경로 `sglang.benchmark.*`)
- 이전 자료의 HiCache L3/RDMA, 브레이크포인트 디버깅 등 고급 주제는 Session 6 §5의 지도로 축소 (최종 프로젝트 주제로 선택 가능)
- 예산 $10–15 → $20 기준으로 재배분, 회차별 상한 명시
