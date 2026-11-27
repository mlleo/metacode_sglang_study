# 3. Radix tree와 공유 prefix

## 실습에서 확인한 값

기본 서버는 579,196 KV token을, `--mem-fraction-static 0.5`는 336,445개를 할당했습니다. 기본 로그의 K·V 크기 합은 약 61.86 GiB입니다. 579,196으로 나누면 Session 1에서 계산한 약 112 KiB/token과 맞습니다.

`prefix_basics.py`에서 처음 요청은 0개, 서로 다른 suffix 앞에 1,000개를 공유한 요청은 정확히 1,000개를 재사용했습니다. 1,050개 입력을 다시 보내면 1,049개가 cached였습니다. 마지막 위치의 logits가 필요해서 마지막 토큰은 다시 forward합니다. 8,192-token prefix에서 cold TTFT 중앙값은 62.04 ms, warm은 9.37 ms였습니다. [prefix sweep CSV](data/s03_prefix_sweep.csv)

실습 0에서 `ABCDE → ABCXY → ABQRS`를 용량 9로 돌리면 세 번째 요청은 `AB` 두 슬롯을 재사용하고 `DE`를 제거합니다. 용량 7에서는 `DE`와 `XY`가 모두 빠집니다. `evicted`와 `free`를 같이 보고, 새 요청이 사용 중인 `AB`는 보호되는지 트리에서 확인합니다.

멀티턴의 6번째 요청은 token ID를 이어 붙였을 때 `472/491`, 텍스트를 다시 토크나이즈했을 때 `475/495`가 hit여서 둘 다 약 96%였습니다. 분모도 달라졌으므로 `cached` 숫자만으로 어느 쪽이 더 낫다고 고르기 어렵습니다. 같은 이전 출력을 두 방식으로 넣고 token ID가 처음 달라지는 위치를 기록하면, 템플릿이나 경계에서 재토크나이즈한 차이를 더 직접적으로 찾을 수 있습니다. [6개 턴 출력](data/s03_multiturn.txt)

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

[부하 CSV](data/s03_share.csv)

## 재현

[실습 0–4](../../session03_kvcache_radix/lab.md)와 [과제 A·B](../../session03_kvcache_radix/assignment.md)의 명령을 사용했습니다. cold/warm sweep의 H100 서버는 4090 수업 설정과 같은 chunk 크기 4,096으로 시작했습니다. 처리량 비교는 입력 총 2,048, 출력 128, 동시성 16으로 고정했습니다.

## 어떤 값으로 결론을 내렸나

1. 한 요청에서 `meta_info.cached_tokens`를 `prompt_tokens`와 비교하고, 서버 로그의 `#new-token + #cached-token`으로 확인합니다. 같은 prefix를 유지하고 suffix만 다르게 보내면 어느 지점까지 재사용했는지 드러납니다.
2. 동시성 실험은 [부하 CSV](data/s03_share.csv)에서 `shared_prefix_len + input_len = 2048`인지 확인합니다. `cache_hit_rate`는 실제로 재사용한 입력 토큰 비율입니다. 설정한 공유율 75%와 비교해 첫 batch의 miss 가능성을 설명합니다.
3. `ttft_ms_p50` 개선 배율과 `output_tok_per_s` 개선 배율을 따로 냅니다. 두 값이 다르면 decode가 남아 있다는 설명을 붙일 수 있습니다. 그 설명을 더 검증하려면 출력 길이를 하나 더 골라 같은 방식으로 재거나, 75% 공유를 그대로 두고 cache만 끕니다.

## 선택 실습에서 읽을 것

24,000-token pool에서 LPM과 FCFS는 각각 256개 요청을 완료했습니다. 서버 로그의 cache hit 비율은 83.5%와 45.5%였고, 처리량은 약 5,074와 2,489 tok/s였습니다. `sum_new_token`도 97,323과 321,682로 방향이 맞습니다. 같은 workload에서 대기 순서가 달라져 prefill 재사용량도 달라진 결과입니다. p99 TTFT는 이 실행에서 각각 2,375와 2,492 ms였으므로, LPM이 언제나 tail을 늦춘다고 결론 내릴 수 없습니다. 공유 종류와 부하를 바꿔 다시 확인할 수 있습니다. [정책별 CSV](data/s03_policy.csv)

## 과제 D 코드 리딩

1. `default_radix_cache_factory`의 마지막 분기에서 `create_unified_radix_cache`를 따라갑니다. 기본 Qwen3 dense 모델은 `FULL` component의 `UnifiedRadixCache`를 만듭니다. [캐시 팩토리](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/mem_cache/registry.py#L134-L174)
2. 이 모델에서는 `--disable-radix-cache`를 주어도 `UnifiedRadixCache`를 만듭니다. 일반 세션에서 `_match_tree`가 `empty_match_result`를 반환하므로 기존 prefix KV를 가져오지 않습니다. [비활성화 분기](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/mem_cache/registry.py#L76-L90) · [prefix 조회](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/mem_cache/unified_radix_cache.py#L580-L588)
3. `RadixCache._split_node`에서 새 부모는 기존 child를 쓰던 요청들의 경로에도 들어갑니다. child의 `lock_ref`를 새 부모에 넘기면, 그 요청들이 끝날 때까지 두 조각의 슬롯을 보호할 수 있습니다. [split 코드](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/mem_cache/radix_cache.py#L620-L633)

## 과제 C를 고른 경우

| 비교 | 확인할 근거 |
|---|---|
| 기본 page vs page 16 | 동일한 요청의 `cached_tokens`와 `#new-token`. 1,000-token 공통 prefix가 어디까지 페이지 단위로 맞는지 계산합니다. |
| Chat template의 thinking ON/OFF | 같은 대화에서 서버가 보고한 `usage.prompt_tokens_details.cached_tokens`와 token ID가 처음 달라지는 위치를 턴마다 적습니다. 템플릿으로 달라진 이전 assistant 턴도 확인합니다. |
| LRU vs LFU | pool과 스케줄링 정책을 고정한 뒤 일부 prefix에 반복 요청을 모읍니다. cache hit, 재계산한 입력 토큰 수, p95 TTFT를 비교합니다. |

cache ON/OFF 대조군은 공유율 75%를 양쪽에 고정합니다. 이 네 변형에는 H100 실측 예시가 없습니다.
