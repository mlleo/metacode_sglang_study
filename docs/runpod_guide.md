# RunPod 실습 환경 & 예산 가이드

수강생은 자기 RunPod 계정에서 실습과 과제를 합니다. **$20**은 수업·과제·최종 프로젝트를 합친 계획 기준입니다. Deploy 화면의 시간당 가격과 본인 잔액을 먼저 확인하세요.

> **3계명**
> 1. GPU를 켜기 전에 무엇을 실행할지 정해 둔다 (스크립트는 로컬에서 mock 서버로 미리 디버깅).
> 2. 실습이 끝나면 결과를 내려받고 **Terminate**한다.
> 3. Billing 페이지에서 잔액을 자주 확인한다.

---

## 1. GPU 선택

| GPU | VRAM | 아키텍처 | 이 스터디에서 | 비고 |
|---|---:|---|---|---|
| **RTX 4090** ⭐ | 24 GB | Ada (SM89) | **표준.** 모든 실습 가능 | FP8 tensor core 지원 → Session 5 FP8 실습 가능 |
| RTX 3090 | 24 GB | Ampere (SM86) | Session 5 FP8 실습을 뺀 나머지 가능 | 4090보다 저렴, FP8 하드웨어 가속 없음 |
| L4 | 24 GB | Ada (SM89) | 가능. 메모리 대역폭이 4090의 약 1/3 | "decode는 memory-bound" 비교 실험용으로 흥미로움 |
| A40 / L40S | 48 GB | Ampere / Ada | 가능하지만 이 스터디엔 과함 | 큰 모델 실험을 하고 싶은 최종 프로젝트용 |

- 같은 RTX 4090이라도 지역과 재고에 따라 달라집니다. 실습 시간을 계산할 때는 **Deploy 화면에 표시되는 시간당 가격**을 씁니다. [Runpod GPU 가격표](https://www.runpod.io/gpu-models)
- 결과 비교를 위해 **같은 반은 같은 GPU 종류**를 쓰는 것이 원칙입니다. 4090이 없으면 강사에게 알리고 다른 GPU를 쓴 뒤, README에 GPU 이름을 반드시 적으세요.

## 2. CUDA 필터 (중요!)

SGLang v0.5.23은 **CUDA 13 / PyTorch 2.14** 기반이라 **NVIDIA driver 580 이상**인 호스트에서만 돕니다.
RunPod Community Cloud 호스트는 driver 버전이 제각각입니다.

- Deploy 화면에서 **CUDA Version 필터 → 13.0 이상**을 선택하고 GPU를 고르세요.
- 필터 없이 띄우면 컨테이너가 `CUDA driver version is insufficient` 같은 오류로 시작하지 않거나,
  `common/setup_pod.sh`가 driver 경고를 내고 멈춥니다. 그때는 바로 Terminate하고 필터를 걸어 다시 띄우세요.

## 3. 템플릿 선택 — 둘 중 하나

### 방법 A (권장): Runpod PyTorch 템플릿 + 설치 스크립트

1. Pods → **+ Deploy** → Community Cloud → RTX 4090 → **CUDA 13.0+ 필터**
2. Template: **Runpod PyTorch** 계열 중 Python 3.11 이상, CUDA 13을 지원하는 이미지. **Additional filters → CUDA Versions**에서 호스트 조건을 따로 지정합니다. [Runpod 배포 문서](https://docs.runpod.io/pods/manage-pods#deploy-a-pod)
3. **Container Disk 40 GB 이상** (PyTorch 2.14 + SGLang 의존성 + 모델 weight)
4. Volume Disk: 0 GB로 둬도 됩니다 (매번 Terminate할 것이므로). 수업 사이에 Stop으로 이어 쓸 계획이면 20 GB 정도.
5. Deploy → `Running`이 되면 **Connect → Web Terminal**을 엽니다. `Start` 버튼이 보이면 누른 후 **Open Web Terminal**을 선택합니다. JupyterLab의 Terminal도 사용할 수 있습니다. [Runpod 접속 문서](https://docs.runpod.io/pods/connect-to-a-pod)
6. 터미널에서:

   ```bash
   cd /workspace
   git clone --branch main https://github.com/mlleo/metacode_sglang_study.git
   cd metacode_sglang_study
   bash common/setup_pod.sh      # 3–8분: SGLang 0.5.23 설치 + 모델 다운로드
   ```

   템플릿의 PyTorch 버전과 상관없이 `setup_pod.sh`가 SGLang이 요구하는 PyTorch(2.14)를 함께 설치합니다.

### 방법 B: SGLang 공식 Docker 이미지를 커스텀 템플릿으로

설치 시간이 아까울 때 쓰는 방법입니다. 대신 이미지가 커서(약 14–16 GB) **처음 pull하는 데 시간이 걸립니다**.

1. Templates → New Template
   - Container Image: `lmsysorg/sglang:v0.5.23-runtime` (개발 도구가 빠진 작은 이미지)
     또는 `lmsysorg/sglang:v0.5.23`
   - Container Disk: 50 GB
   - Container Start Command: `sleep infinity`
2. 이 템플릿으로 Deploy (CUDA 13 필터 동일하게)
3. Web Terminal로 접속 → 위와 같이 clone → `bash common/setup_pod.sh` (SGLang 설치는 자동으로 건너뜁니다)

> 강사 메모: 방법 B는 Web Terminal / SSH 접속 방식이 이미지에 따라 다를 수 있습니다. 수업 전 리허설에서
> 어느 방법을 쓸지 정하고, 반 전체가 같은 방법을 쓰게 하세요 ([instructor_guide.md](instructor_guide.md)).

## 4. 터미널 운영 팁

- Web Terminal은 브라우저 탭을 닫으면 끊길 수 있습니다. 서버는 `common/launch.sh`가 **백그라운드(nohup)** 로 띄우므로
  터미널이 끊겨도 살아 있습니다. 다시 접속해서 `tail -f logs/server_<tag>.log`로 이어 보면 됩니다.
- 터미널 두 개가 필요하면 `tmux` (`tmux new -s lab`, 분할: `Ctrl-b %`, 이동: `Ctrl-b ←/→`)를 쓰세요.
  (`tmux`가 없으면 `apt-get install -y tmux`)
- GPU 상태: `watch -n 1 nvidia-smi`

## 5. 결과를 Pod 밖으로 꺼내기

Terminate하면 디스크가 통째로 사라집니다. **꺼내고 나서** Terminate하세요.

| 방법 | 명령 / 위치 | 비고 |
|---|---|---|
| Jupyter 파일 브라우저 | `results/`, `logs/` 우클릭 → Download | 가장 쉬움 (방법 A) |
| 압축 후 다운로드 | `tar czf s01_results.tgz results logs` | 파일이 많을 때 |
| `runpodctl send` | Pod: `runpodctl send s01_results.tgz` → 출력된 코드로 로컬에서 `runpodctl receive <code>` | RunPod 이미지에 보통 포함 |
| git push | 결과만 커밋해서 fork에 push | **토큰을 노트북·스크립트에 하드코딩하지 마세요.** 쓰고 나면 토큰 폐기 |

그래프는 **로컬에서** `python3 common/plot.py ...`로 그립니다. GPU 위에서 matplotlib 돌리는 데 돈을 쓰지 마세요.

## 6. Stop vs Terminate

| | Stop (■) | Terminate (🗑) |
|---|---|---|
| GPU 과금 | 멈춤 | 멈춤 |
| 디스크 과금 | **계속됨** (Volume Disk 유지 비용) | 멈춤 |
| 데이터 | Volume Disk(`/workspace`)을 붙였다면 그 내용은 유지, Container Disk는 사라짐 | Container Disk와 Pod의 Volume Disk는 사라짐. 별도로 연결한 Network Volume은 유지 |
| 언제 | 같은 날 1–2시간 쉬었다 이어갈 때 | 그날 실습이 끝났을 때 — **기본값** |

Terminate 후 Pods 목록에서 카드가 **완전히 사라졌는지** 확인하세요.

### (선택) 깜빡 방지 안전장치

로컬 CLI로 Pod를 만들 때 `runpodctl pod create`에 `--stop-after 3h`를 추가할 수 있습니다. 타이머 전에 결과를 내려받으세요. [Runpod CLI 옵션](https://docs.runpod.io/runpodctl/reference/runpodctl-remove-pods)

Stop은 디스크 과금이 남으므로, 다음 날 꼭 Terminate하세요.

---

## 예산 ($20) 배분

계획 단가: **시간당 $0.60** (4090 Community 최저가보다 넉넉하게 잡은 값. Pod 셋업·이미지 pull 시간도 과금된다고 가정)

| 회차 | 수업 실습 | 과제 | 소계 | GPU 시간 (대략) |
|---|---:|---:|---:|---:|
| 1. 기초 & 첫 서버 | $0.9 (셋업 포함) | $0.9 | $1.8 | 3.0 h |
| 2. 아키텍처 & batching | $0.8 | $1.2 | $2.0 | 3.3 h |
| 3. KV cache & Radix | $0.8 | $1.2 | $2.0 | 3.3 h |
| 4. 스케줄러 | $0.8 | $1.2 | $2.0 | 3.3 h |
| 5. 실행 & 최적화 | $1.0 | $1.5 | $2.5 | 4.2 h |
| 6. 벤치마크 & 비용 | $0.8 | — | $0.8 | 1.3 h |
| 최종 프로젝트 | — | $4.0 | $4.0 | 6.7 h |
| **예비비** | | | **$4.9** | 실수·재실행·Stop 디스크 비용 |
| **합계** | | | **$20.0** | |

**실습 시작 전에 쓸 시간을 정하세요.**

- 사용할 수 있는 시간(분)은 `60 × GPU에 쓸 금액 ÷ 표시된 시간당 가격`입니다. 예를 들어 $0.80과 $0.60/h면 약 80분입니다. 여기서 설치, 서버 재시작, 다운로드 시간을 먼저 뺍니다. 55분 실습이 0:55에 시작하면 0:40–0:50 사이에 준비를 시작하는 편이 맞습니다. 수업 도입부터 1:50까지 켜 두면 같은 단가에서 약 $1.10입니다. 저장 공간 요금은 별도로 확인하세요.
- 과제에서는 수업 CSV를 다시 쓰고 필수 비교부터 실행합니다. 예산이 빠듯하면 강사에게 같은 문항을 풀 수 있는 기준 데이터를 받아 로컬에서 분석하세요.
- 결과는 각 실험이 끝날 때 내려받습니다. `common/stop.sh`는 SGLang 프로세스를 종료합니다. GPU 임대를 마치려면 RunPod 콘솔에서 Pod를 종료해야 합니다.
- 모든 GPU 사용은 [templates/gpu_usage_log.csv](../templates/gpu_usage_log.csv) 형식으로 기록합니다.
- 비싼 실험(긴 sweep)은 **작은 설정으로 먼저 한 번** 돌려 스크립트가 끝까지 도는지 확인하고 본 실험을 돌립니다.
- 모델 다운로드는 Pod마다 다시 일어납니다 (Qwen3-1.7B ≈ 4 GB, 보통 1분 내외).

## 자주 겪는 문제

| 증상 | 원인 / 해결 |
|---|---|
| Pod이 `Running`이 안 되고 재시작 반복 | CUDA 필터 없이 driver가 낮은 호스트에 배치됨 → Terminate, 필터 13.0+로 재배포 |
| `setup_pod.sh`가 driver 경고 후 종료 | 위와 같음 |
| `No space left on device` | Container Disk 부족 → 40 GB 이상으로 재배포 |
| 서버가 `CUDA out of memory`로 죽음 | 다른 서버가 이미 떠 있음 → `bash common/stop.sh` 후 `nvidia-smi`로 메모리 0 확인 |
| `launch.sh`가 오래 기다림 | 첫 실행은 weight 로딩 + CUDA graph capture + (FlashInfer JIT) 때문에 1–3분 걸릴 수 있음. `tail -f logs/server_<tag>.log`로 진행 확인 |
| 4090이 품절 | 같은 반이 다른 GPU를 쓰게 되면 README에 GPU 명시. 숫자 비교는 같은 GPU끼리만 |
