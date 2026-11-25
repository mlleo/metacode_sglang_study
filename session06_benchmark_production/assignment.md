# Session 6 과제 = 최종 프로젝트

이번 회차의 과제는 **최종 프로젝트**입니다 → [final_project.md](final_project.md)

Session 6 직후 할 일:

1. 주제를 골라 레포에 **이슈**로 등록: `[Final] <이름>: <질문 한 문장>`
   - 본문: 가설(숫자), 바꿀 변수, 고정할 조건, 예상 GPU 시간
2. 강사 피드백 반영 후 작은 리허설 → 본 실험
3. 마감·발표 일정은 강사 공지를 따릅니다

## 수업 실습 결과 제출 (간단히)

`submissions/session06/<github-id>/README.md`에 아래 세 가지만 붙여 주세요 (발표 없음).

- `repeat_stats.py` 출력 + "내 이전 과제의 주장 중 이 노이즈 수준에서 다시 봐야 할 것" 1개
- `trace_summary.py` 출력 3개(c1, c32, c1_nograph)와 Perfetto 스크린샷 1장, 관찰 3줄. c32와 c1_nograph는 수업에 제공된 trace를 사용해도 됩니다. 각 trace의 GPU를 적어 주세요.
- 내 결과 중 SLO를 지키는 가장 싼 $/1M tokens와 그 설정
