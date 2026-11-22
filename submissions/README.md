# 과제 제출 방법

1. 이 레포를 본인 계정으로 **fork**
2. 본인 fork에서 아래 경로에 과제 폴더 생성

   ```text
   submissions/sessionNN/<github-id>/
   ├── README.md          # templates/report_template.md 복사해서 작성
   ├── results/           # CSV, JSON (원시 결과)
   ├── logs/              # 서버 로그 (용량 큰 것은 gzip)
   ├── figures/           # PNG
   └── code/              # 직접 고치거나 새로 쓴 스크립트
   ```

3. 원본 레포에 **Pull Request** — 제목: `[S02] <이름> 과제`
4. 마감: **다음 회차 시작 24시간 전** (강사가 발표자를 고르고 피드백을 남길 시간)

## 규칙

- 원시 결과(CSV/로그)를 꼭 함께 올리세요. 그래프만 있으면 검증할 수 없습니다.
- 50 MB 넘는 파일(프로파일 trace 등)은 올리지 말고 README에 "어떻게 다시 만드는지"를 적으세요.
- 토큰·API key·RunPod key가 들어간 파일은 절대 커밋하지 마세요.
- 다른 사람 결과를 참고했다면 README에 출처(GitHub ID)를 적으세요. 같은 GPU에서 비교하는 건 환영입니다.

## 발표자 선정

- 강사는 PR을 보고 **서로 다른 결과/해석**을 보여줄 2–3명을 고릅니다 (잘한 사람 순이 아님).
- 예측이 크게 틀린 과제가 가장 좋은 발표 소재입니다.
