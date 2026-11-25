# 3. Radix tree와 공유 prefix

## 실습에서 확인한 값

기본 서버는 579,196 KV token을, `--mem-fraction-static 0.5`는 336,445개를 할당했습니다. 기본 로그의 K·V 크기 합은 약 61.86 GiB입니다. 579,196으로 나누면 Session 1에서 계산한 약 112 KiB/token과 맞습니다.

`prefix_basics.py`에서 처음 요청은 0개, 서로 다른 suffix 앞에 1,000개를 공유한 요청은 정확히 1,000개를 재사용했습니다. 1,050개 입력을 다시 보내면 1,049개가 cached였습니다. 마지막 위치의 logits가 필요해서 마지막 토큰은 다시 forward합니다. 8,192-token prefix에서 cold TTFT 중앙값은 62.04 ms, warm은 9.37 ms였습니다. [prefix sweep CSV](data/s03_prefix_sweep.csv)

## 과제 A 손 추적

| 요청 | cached | 새 할당 | split | eviction |
|---|---:|---:|---|---|
| SYSTEMHELLO | 0 | 11 | | |
| SYSTEMBYE | 6 | 3 | SYSTEM / HELLO | |
| SYSTEMHELP | 9 | 1 | HEL / LO | |
| CHAT | 0 | 4 | | BYE |
| SYSTEMBYEBYE | 6 | 6 | | LO, P, HEL |

네 번째 요청 뒤에는 CHAT과 SYSTEM 아래의 두 leaf가 남습니다. 다섯 번째 요청은 SYSTEM을 재사용하며 그 경로를 보호합니다. 오래된 leaf부터 슬롯을 돌려받은 뒤 새로운 BYEBYE를 달 수 있습니다. 시뮬레이터는 요청을 하나씩 진행하므로 실제 서버의 공유 경로 lock 수와 동시성을 그대로 그리지 않습니다.

## 과제 B 공유 비율

| 공유 비율 | cached token 비율 | output tok/s | p50 TTFT (ms) |
|---:|---:|---:|---:|
| 0% | 0.0% | 3,080 | 113.7 |
| 50% | 46.1% | 3,773 | 79.5 |
| 75% | 69.1% | 4,316 | 52.5 |

75% 공유에서 처리량은 약 1.40배가 됐고, 첫 토큰 지연은 절반 이하로 내려갔습니다. 입력을 덜 계산해도 생성 중에는 계속 KV를 읽습니다. shared prefix가 75%인데 hit가 69.1%인 이유로, 첫 요청들이 동시에 prefill에 들어간 구간을 확인할 수 있습니다.

[과제 그래프](figures/s03_assignment_prefix.png) · [부하 CSV](data/s03_share.csv)

## 재현

[실습 0–4](../../session03_kvcache_radix/lab.md)와 [과제 A·B](../../session03_kvcache_radix/assignment.md)의 명령을 사용했습니다. cold/warm sweep의 H100 서버는 4090 수업 설정과 같은 chunk 크기 4,096으로 시작했습니다. 처리량 비교는 입력 총 2,048, 출력 128, 동시성 16으로 고정했습니다.

## 어떤 값으로 결론을 내렸나

1. 한 요청에서 `meta_info.cached_tokens`를 `prompt_tokens`와 비교하고, 서버 로그의 `#new-token + #cached-token`으로 확인합니다. 같은 prefix를 유지하고 suffix만 다르게 보내면 어느 지점까지 재사용했는지 드러납니다.
2. 동시성 실험은 [부하 CSV](data/s03_share.csv)에서 `shared_prefix_len + input_len = 2048`인지 확인합니다. `cache_hit_rate`는 실제로 재사용한 입력 토큰 비율입니다. 설정한 공유율 75%와 비교해 첫 batch의 miss 가능성을 설명합니다.
3. `ttft_ms_p50` 개선 배율과 `output_tok_per_s` 개선 배율을 따로 냅니다. 두 값이 다르면 decode가 남아 있다는 설명을 붙일 수 있습니다. 그 설명을 더 검증하려면 출력 길이를 하나 더 골라 같은 방식으로 재거나, 75% 공유를 그대로 두고 cache만 끕니다.
