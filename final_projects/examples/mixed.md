# 채팅 중 긴 문서 요청

## 어떤 비교를 하는가

`baseline`과 `mixed_chunk`는 chunk 크기 1,024를 공유합니다. `decode_interval`도 같은 크기에서 prefill 뒤에 decode 기회를 한 번 주는 설정입니다. `standard`에서 문서 하나, `stress`에서 문서 두 개를 넣습니다. `min_chats_overlapping`과 문서 도착선을 먼저 봅니다. 겹침이 없던 구간을 비교하면 prefill 간섭을 설명할 수 없습니다.

## 지표에서 근거 찾기

1. 각 seed에서 채팅 `chat_itl_ms_p99`와 `chat_gap_ms_max`에 한도를 적용합니다. 한 번 넘은 설정은 그 profile에서 실패입니다. 평균 문서 TTFT의 순위는 통과한 설정에서 봅니다.
2. 같은 seed의 timeline 두 개를 열고 문서 도착선 뒤의 빈 구간을 찾습니다. 같은 채팅 id에서 연속된 청크 시각을 빼면 화면의 멈춤 시간을 확인할 수 있습니다.
3. 같은 run의 `Prefill batch`와 `Decode batch`를 순서대로 봅니다. 클라이언트와 서버 로그의 시계 단위가 다르므로 개별 kernel에 시간을 귀속하려면 extend trace로 한 번 더 확인합니다.
4. 스트리밍 한도를 더 낮춰 선택이 바뀌는지 봅니다. 문서 두 개가 온 상황에서 어느 설정도 통과하지 못했다면 기준 미달로 보고하고 추가 실험을 제안합니다.

[수업 H100 타임라인](../../docs/worked-results/session04.md)에서는 16K 문서의 mixed OFF가 p99 ITL 3.70 ms를 기록했지만 최대 ITL은 151.82 ms였습니다. ON은 p99가 9.90 ms로 늘고 최대 ITL은 14.02 ms로 줄었습니다. p99와 최장 공백을 함께 두는 이유입니다.
