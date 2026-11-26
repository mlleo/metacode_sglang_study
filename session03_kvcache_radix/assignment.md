# Session 3 과제 — "prefix cache는 언제, 얼마나 이득인가"

> 마감: Session 4 시작 24시간 전 PR · GPU 예산 **$1.2 이내**
> 제출: `submissions/session03/<github-id>/`

## Part A. Radix tree 손 추적 (GPU 불필요)

KV pool 슬롯 16개, LRU eviction, 요청은 **순서대로 하나씩** 실행되고 끝난다고 가정합니다.
요청이 실행될 때는 `입력[:-1]`로 match합니다.

```text
req1  SYSTEMHELLO
req2  SYSTEMBYE
req3  SYSTEMHELP
req4  CHAT
req5  SYSTEMBYEBYE
```

각 요청 후의 tree를 그리고(노드 key, 슬롯 수), 아래 표를 채우세요.

| 요청 | cached | 새로 할당 | split 발생 노드 | evict된 노드 |
|---|---:|---:|---|---|
| req1 | | | | |
| … | | | | |

그다음 `radix_sim.py`로 검증하세요.

```bash
python3 session03_kvcache_radix/code/radix_sim.py SYSTEMHELLO SYSTEMBYE SYSTEMHELP CHAT SYSTEMBYEBYE --capacity 16
```

손 추적과 시뮬레이터 결과가 **다른 곳이 있다면** 어느 쪽이 맞는지, 왜 그런지 설명하세요. (시뮬레이터가 단순화한 부분도 찾아보세요. 힌트: lock 처리, 마지막 token.)

## Part B. "언제 이득인가" 곡선 (GPU)

수업의 `prefix_sweep.py`는 **요청을 하나씩** 보냈습니다. 실제 서비스는 동시에 들어옵니다.
동시성 16에서 **공유 비율**에 따라 throughput과 TTFT가 어떻게 변하는지 재 보세요.

입력은 모두 2048 token으로 고정하고 공유 prefix 0%, 50%, 75%를 비교합니다. 25%와 95%는 시간이 남을 때 추가하세요.

```bash
bash common/launch.sh baseline
for SHARE in 0 1024 1536; do
  SUFFIX=$((2048 - SHARE))
  python3 common/loadgen.py --num 64 --concurrency 16 --shared-prefix-len $SHARE --input-len $SUFFIX \
    --output-len 128 --flush --tag share$SHARE --out results/s03_share.csv
done
bash common/stop.sh

# 선택: radix cache를 끄고 같은 실험 (75%만)
bash common/launch.sh noradix --disable-radix-cache
python3 common/loadgen.py --num 64 --concurrency 16 --shared-prefix-len 1536 --input-len 512 \
  --output-len 128 --flush --tag noradix1536 --out results/s03_share.csv
bash common/stop.sh
```

제출:

1. 그래프: 공유 비율 → output throughput, p50 TTFT (두 subplot)
2. 각 조건의 `cache_hit_rate`를 비교하세요. 첫 요청 뒤에 나머지가 모두 hit한다고 가정한 상한은 공유 비율 × (요청 수 − 1) / 요청 수입니다. 동시에 들어온 첫 batch에서는 이보다 작을 수 있습니다.
3. 해석: 공유 비율 75%일 때 throughput이 몇 배 늘었나요? **TTFT 개선 폭과 throughput 개선 폭이 다르다면** 왜일까요?
   (힌트: prefix hit가 아끼는 것은 주로 prefill입니다. 생성 중에는 KV를 계속 읽습니다. Session 1 §3, §5)
4. 선택 대조군을 실행했다면 공유 75%에서 cache ON/OFF를 비교하세요. ON 상태의 공유 0%와 비교할 때는 prefix 구성도 함께 바뀐다는 점을 적습니다.

## Part C. 선택 실험 (GPU 예산에 여유가 있으면 하나, $0.3 이내)

- **C1. page_size**: `--page-size 16`으로 띄우고 `prefix_basics.py`를 돌려 cached 값이 어떻게 바뀌는지 (16의 배수로 잘리는지) 확인. 장점은 무엇일까?
- **C2. chat template과 cache**: `--enable-cache-report`로 서버를 띄우고, OpenAI 호환 `/v1/chat/completions`로 6턴 대화를 진행하며
  `usage.prompt_tokens_details.cached_tokens`를 기록. `chat_template_kwargs: {"enable_thinking": true}`와 `false`를 비교하고,
  hit rate가 다르다면 Qwen3 chat template이 이전 턴의 무엇을 바꾸는지 template 파일(`tokenizer_config.json`의 `chat_template`)을 읽고 설명.
- **C3. eviction 정책**: `policy_compare.sh`를 응용해 `--radix-eviction-policy lru` vs `lfu`를 비교. 어떤 workload에서 LFU가 유리할까?

## Part D. 코드 리딩 (GPU 불필요)

`mem_cache/registry.py`의 `default_radix_cache_factory`를 읽고 답하세요.

1. Qwen3-1.7B(일반 dense 모델)를 기본 설정으로 띄우면 어떤 클래스의 prefix cache가 만들어지나요? 그 근거가 되는 코드 줄을 인용하세요.
2. `--disable-radix-cache`를 주면 무엇이 만들어지나요? 그 객체의 `match_prefix`는 무엇을 반환할까요?
3. `radix_cache.py`의 `_split_node`에서 `new_node.lock_ref = child.lock_ref`인 이유는? (split 전에 child를 쓰던 요청을 생각해 보세요)

## 채점 기준 (10점)

| 항목 | 점수 |
|---|---:|
| Part A 손 추적 정확성 + 시뮬레이터와의 차이 분석 | 3 |
| Part B 측정 재현성 + 그래프 | 2 |
| Part B 해석 (TTFT vs throughput 개선 폭 차이) | 2 |
| Part D 코드 리딩 | 2 |
| GPU 기록 | 1 |
