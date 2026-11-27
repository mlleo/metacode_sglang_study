# 마감이 있는 배치 생성

## 어떤 비교를 하는가

요청 96개가 같은 시각에 들어온 하나의 작업으로 봅니다. client worker 32개와 decode graph 상한은 세 설정에서 같습니다. 비교하는 값은 서버의 running-request 상한 8·16·32입니다. 마감은 `standard`와 `stress` 각각에서 배치 전체가 끝난 시각에 적용합니다. 실험 시작 전에 기준을 정하세요.

## 지표에서 근거 찾기

1. `completed`와 `expected_requests`가 96인지 확인합니다. `wall_s`는 가장 늦게 끝난 요청까지 잰 값입니다. output tok/s 후보를 비교할 때는 마감을 지킨 실행만 고려합니다.
2. `job_finish_s_p95`와 `http_e2e_ms_p95`의 차이를 봅니다. 원시 파일에서 `t_start_rel`이 큰 요청을 읽으면 client worker가 비기까지 기다렸음을 알 수 있습니다. 서버 안의 대기는 `#queue-req`가 보여 줍니다.
3. 같은 seed의 로그에서 `#running-req`와 `cuda graph`를 확인합니다. 요청이 32개 동시 제출돼도 매 step에서 32개를 처리한다는 뜻은 아닙니다. 더 큰 상한이 실제로 더 큰 decode batch로 이어졌는지 근거를 남깁니다.
4. stress에서는 96개 중 24개의 출력이 길어집니다. 처리량이 높은데도 배치 끝이 밀릴 수 있으므로 profile별 목표 통과 여부와 마지막 몇 개의 `finish_s`를 기록합니다.

[수업 H100의 HTTP graph 상한 예시](../../docs/worked-results/session05.md#재현)에서는 실제 `#running-req` 16에서 graph가 적용됐고 64에서 적용되지 않았습니다. 프로젝트에서 client 동시성과 실제 batch 크기를 따로 보는 이유입니다.

같은 수업 실행에서 입력·출력 256인 요청을 128개씩 보냈습니다. 두 조건 모두 전부 완료했습니다. client 동시성 16은 5.69초에 5,759 tok/s, 64는 6.73초에 4,866 tok/s였습니다. 연습용 마감을 6초로 두면 16만 통과합니다. 요청의 p95 HTTP E2E인 717 ms와 3,376 ms를 사용하면 전체 배치 마감을 잘못 판정합니다. 이 수업 비교에서는 client 동시성을 바꿨습니다. 프로젝트에서는 client worker 수를 고정한 상태에서 서버 상한을 바꾸므로 각각의 통제 변수를 보고서에 적으세요. [수업 CSV](data/batch_graph_reference.csv) · [서빙 로그](../../docs/worked-results/data/s05_http_graph_limit.txt)
