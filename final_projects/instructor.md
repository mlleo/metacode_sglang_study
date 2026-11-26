# 최종 프로젝트 운영과 사전 리허설

## 1. 배정 전에 확인할 것

- 1주차에 workload를 발표합니다. 4주차 말 선호를 받아 트랙당 인원을 비슷하게 배정하고 초과분을 추첨합니다. 배정 뒤 학생 개인별 기준선과 성공 기준은 해당 트랙에서 같게 공지합니다.
- 5–8주차는 일반 과제와 별도로 프로젝트를 진행합니다. 5주차에는 스크립트 실행 기록, 6주차에는 후보 선택 근거, 7주차에는 paired CSV를 한 번씩 확인합니다.
- 실제 수강생과 같은 RTX 4090 RunPod 이미지에서 [실행 가이드](README.md)의 새 Pod 설치, 다운로드, `pilot`, 종료까지 잽니다. H100은 서버와 실험 로직 검증에 쓸 수 있습니다. 4090의 시간과 비용은 RunPod에서 잽니다.
- `mixed`에서 실제 overlap과 채팅의 최대 gap을 먼저 봅니다. `batch`는 HTTP E2E와 전체 작업 시간을 각각 보고, baseline과 한 후보가 의미 있게 갈리는 목표인지 확인합니다. 예시 100 ms와 12초를 그대로 써도 되는지는 학생 GPU에서 체크하고 배정 전에 공지합니다. pilot은 요구되는 성공 지표를 정하는 데 쓰기보다 설치와 기록을 검증하는 용도입니다.

## 2. 같은 코드로 세 트랙 돌리기

GPU를 한 장 지정한 뒤 저장소 루트에서 실행합니다. 수강생이 실제로 쓸 커밋을 적어 두세요. 출력 디렉터리는 이전 것과 이름이 달라야 합니다.

```bash
bash common/setup_pod.sh
for track in prefix mixed batch; do
  python3 final_projects/run.py "$track" --profiles pilot --out "results/final-rehearsal/${track}-pilot"
done
python3 final_projects/run.py prefix --configs baseline,lpm,lfu --profiles standard,stress \
  --seeds 11,22,33 --out results/final-rehearsal/prefix
python3 final_projects/run.py mixed --configs baseline,mixed_chunk,decode_interval --profiles standard,stress \
  --seeds 11,22,33 --out results/final-rehearsal/mixed
python3 final_projects/run.py batch --configs baseline,cap16,cap32 --profiles standard,stress \
  --seeds 11,22,33 --out results/final-rehearsal/batch
```

로컬로 결과를 옮겨 `analyze.py`와 `extract_log.py`를 각 트랙에서 실제로 실행합니다. 프로젝트 본 실험 예시는 [실행 가이드](README.md)의 `--order paired`를 사용합니다. 전체 사전 리허설의 `grouped`에는 시동 비용을 줄이기 위한 순서가 들어갑니다. 리허설 결과를 공유할 때는 그 순서도 적습니다.

| 무엇을 확인하나 | 확인할 파일 |
|---|---|
| 조건별 동일 input과 모든 요청 완료 | `manifest.json`, `runs.csv`의 설정·seed·workload 해시·완료 수 |
| 표와 로그가 같은 run을 가리킴 | `extract_log.py` 출력과 `raw/<설정>_<profile>_s<seed>.json` |
| 채팅과 문서 요청 겹침 | `min_chats_overlapping`과 두 조건의 `timeline.json` |
| deadline 점검 | `wall_s`, 시작·종료가 있는 원시 JSON, `job_finish_s_p95` |
| 빈 CSV나 오류가 통과하지 않음 | 실패가 난 campaign의 manifest·로그, 분석 도구의 오류 출력 |
| 비용과 서버 종료 | 실제 Billing과 남아 있는 GPU 프로세스 확인, 결과 보관 후 Pod 정리 |

## 3. 발표 운영

같은 트랙 발표는 가능하면 이어서 배치합니다. 첫 발표에서 바꾼 기준 하나를 다음 발표에도 줍니다. 예를 들어 mixed의 최대 청크 공백을 더 엄격하게 바꾸고, 어떤 설정이 남는지 학생의 표에서 바로 찾아보게 합니다. 발표의 장비가 다르면 절대 수치 옆에 GPU를 표시하고, 각자의 기준선 대비 변화와 실험 조건을 함께 봅니다.

평가는 실험 설계, 지표의 의미, 근거의 일치 여부로 합니다. 기준선을 유지하는 결론이나 조건부 추천도 같은 기준으로 평가합니다. 수강생 예산이나 장비 문제로 다시 측정하기 어렵다면 저장된 기준 캠페인 하나를 주고, 실측에 참여한 부분과 제공 데이터에서 분석한 부분을 보고서에 적게 합니다.
