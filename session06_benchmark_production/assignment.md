# Session 6 과제 = 최종 프로젝트

최종 프로젝트는 8주 과정의 별도 활동입니다. 4주차 말에 workload를 배정하고 5–8주차에 POC를 진행합니다. 이번 수업의 반복·trace·비용 실습을 각자의 프로젝트에 적용합니다. [일정과 제출 기준](final_project.md) · [실행 코드와 예시](../final_projects/README.md)

## 수업 실습 결과 제출 (간단히)

`submissions/session06/<github-id>/README.md`에 아래 세 가지만 붙여 주세요 (발표 없음).

- `repeat_stats.py` 출력 + "내 이전 과제의 주장 중 이 노이즈 수준에서 다시 봐야 할 것" 1개
- `trace_summary.py` 출력 3개(c1, c32, c1_nograph)와 Perfetto 스크린샷 1장, 관찰 3줄. c32와 c1_nograph는 수업에 제공된 trace를 사용해도 됩니다. 각 trace의 GPU를 적어 주세요.
- 내 결과 중 SLO를 지키는 가장 싼 $/1M tokens와 그 설정. 모두 실패했다면 어떤 지표가 기준을 넘었는지 적습니다.
