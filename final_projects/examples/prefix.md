# 공유 prefix 상담 서비스

## 어떤 비교를 하는가

같은 96개 요청을 보내고, 각 반복에서 prefix 그룹 수·입출력 길이·동시성·KV pool을 고정합니다. `baseline`과 `lpm`은 요청 순서 정책만 다르고, `baseline`과 `lfu`는 cache eviction 정책만 다릅니다. 설정 이름이 아니라 manifest의 실제 인자를 비교합니다. `standard`에서 고른 후보는 `stress`의 더 많은 prefix 그룹에서 다시 확인합니다.

## 지표에서 근거 찾기

1. `runs.csv`에서 같은 seed의 `workload_sha256`과 `completed`를 확인합니다. `ttft_ms_p95`가 줄었는지 seed별로 차이를 냅니다.
2. `cache_hit_rate`가 오르고 `uncached_input_tokens`가 줄었는지 확인합니다. 같은 요청량에서 이 세 방향이 맞으면 prefill 재계산을 줄였다는 설명과 일치합니다.
3. 해당 seed만 로그로 잘라 `#cached-token`, `#new-token`을 합합니다. 토큰이 덜 계산됐는데 TTFT가 늘었다면 `#queue-req`, p95와 p50의 차이, 늦은 요청의 `group`을 봅니다.
4. 두 설정의 짝별 차이와 구간을 적습니다. 표준 트래픽에서만 이겼다면 캐시 재사용이 많은 환경이라는 조건을 붙입니다.

[수업 H100의 LPM/FCFS 비교](../../docs/worked-results/session03.md#선택-실습에서-읽을-것)에서는 24,000-slot pool에 여러 prefix를 섞었습니다. 두 설정이 각각 256개를 완료했고, LPM에서 로그 cache hit 83.5%, FCFS에서 45.5%였습니다. 출력 처리량은 각각 5,074와 2,489 tok/s였습니다. 이 결과에서 다음 프로젝트의 가설을 세울 수 있습니다. 새 트래픽의 TTFT는 프로젝트 CSV에서 확인합니다.
