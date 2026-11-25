#!/usr/bin/env bash
# 로컬에서 실행. matplotlib, pandas 설치 필요.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
DATA=docs/worked-results/data
OUT="${1:-figures/worked-results}"
mkdir -p "$OUT"

python3 common/plot.py "$DATA/s01_plot_batch.csv" --x batch --y tokens_per_s --group scope --logx --out "$OUT/s01_batch.png"
python3 common/plot.py "$DATA/s01_plot_input.csv" --x input_len --y ms --group scope --logx --out "$OUT/s01_input.png"
python3 common/plot.py "$DATA/s02_base.csv" "$DATA/s02_in2k.csv" --x max_concurrency --y output_throughput \
  --group tag --logx --out "$OUT/s02_throughput.png"
python3 common/plot.py "$DATA/s02_base.csv" "$DATA/s02_in2k.csv" --x output_throughput --y p95_ttft_ms \
  --group tag --annotate max_concurrency --out "$OUT/s02_latency.png"
python3 common/plot.py "$DATA/s03_share.csv" --x shared_prefix_len --y output_tok_per_s,ttft_ms_p50 --out "$OUT/s03_share.png"
python3 session04_scheduler/code/plot_timeline.py "$DATA/s04_timeline_chunk1024.json" \
  "$DATA/s04_timeline_chunk1024mx.json" --out "$OUT/s04_itl.png"
python3 common/plot.py "$DATA/s05_onebatch.csv" --x batch_size --y median_decode_latency_ms,kv_share_pct \
  --group run_name --logx --out "$OUT/s05_graph.png"
