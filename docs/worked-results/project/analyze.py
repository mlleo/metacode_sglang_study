"""입력 길이만 다른 run.sh 결과에서 seed별 짝을 묶어 두 지표의 차이를 계산한다."""

import argparse
import csv
import math
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("csv")
args = parser.parse_args()

metrics = {"ttft_ms_p95": "p95 TTFT (ms)", "output_tok_per_s": "output tok/s"}
pairs = {seed: {} for seed in (1, 2, 3)}
with open(args.csv, newline="") as f:
    for row in csv.DictReader(f):
        length = int(row["input_len"])
        seed = int(row["tag"].rsplit("_r", maxsplit=1)[1])
        if seed not in pairs or length not in (512, 2048) or length in pairs[seed]:
            parser.error("seed 1·2·3에 대해 입력 512와 2048의 결과가 각각 한 행 있어야 합니다.")
        if (row["num_errors"], row["num_requests"], row["concurrency"], row["output_len"]) != ("0", "32", "8", "256"):
            parser.error(f"{row['tag']}의 완료 요청 수나 workload가 예상과 다릅니다.")
        pairs[seed][length] = row

if any(len(pair) != 2 for pair in pairs.values()):
    parser.error("세 seed에서 모두 512와 2048의 짝이 필요합니다.")

for field, label in metrics.items():
    short = [float(pair[512][field]) for pair in pairs.values()]
    long = [float(pair[2048][field]) for pair in pairs.values()]
    deltas = [b - a for a, b in zip(short, long)]
    mean = statistics.mean(deltas)
    half = 4.303 * statistics.stdev(deltas) / math.sqrt(3)  # 양측 95% t, 자유도 2
    print(f"{label}: 512={statistics.mean(short):.2f}, 2048={statistics.mean(long):.2f}")
    print(f"  seed별 차이: {', '.join(f'{x:+.2f}' for x in deltas)}")
    print(f"  평균 차이 {mean:+.2f}, 95% CI [{mean-half:+.2f}, {mean+half:+.2f}]")
