import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from common.metrics import percentile
from session03_kvcache_radix.code import radix_sim
from session06_benchmark_production.code import trace_summary


class CourseToolTests(unittest.TestCase):
    def test_one_large_gap_does_not_make_p99_equal_to_max(self):
        gaps = [10] * 99 + [1000]
        self.assertAlmostEqual(percentile(gaps, 99), 19.9)
        self.assertEqual(max(gaps), 1000)

    def test_radix_sim_releases_duplicate_final_token_slot(self):
        tree = radix_sim.RadixTree(8)
        tokens = list("ABCDE")
        for _ in range(12):
            matched, node = tree.match(tokens[:-1])
            node.lock_ref += 1
            allocated, evicted = tree.alloc(len(tokens) - len(matched))
            tree.insert(tokens, matched + allocated)
            node.lock_ref -= 1
            self.assertFalse(evicted)
            self.assertEqual(len(tree.free), 3)
        self.assertEqual(len(tree.match(tokens)[0]), 5)

    def test_overlapping_gpu_events_use_latest_end(self):
        events = [
            {"ph": "X", "cat": "kernel", "ts": 0, "dur": 100, "name": "gemm"},
            {"ph": "X", "cat": "gpu_memcpy", "ts": 50, "dur": 1, "name": "memcpy"},
            {"ph": "X", "cat": "kernel", "ts": 52, "dur": 0, "name": "empty"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            path.write_text(json.dumps(events))
            summary = trace_summary.summarize([str(path)])
        self.assertEqual(summary["gpu_busy_pct"], 100)
        self.assertEqual(summary["num_kernels"], 2)

    def test_slo_limits_distinguish_load_throughput_and_cost(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sweep.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerows([
                    ["max_concurrency", "output_throughput", "p95_ttft_ms", "p95_itl_ms"],
                    [8, 4500, 250, 40],
                    [16, 4000, 350, 42],
                    [64, 9000, 450, 55],
                ])
            slos = ["--slo", "p95_ttft_ms=500", "--slo", "p95_itl_ms=50"]
            capacity = subprocess.check_output([
                sys.executable,
                root / "session02_architecture_batching/code/find_capacity.py",
                path, *slos,
            ], text=True)
            cost = subprocess.check_output([
                sys.executable,
                root / "session06_benchmark_production/code/cost_calc.py",
                path, "--usd-per-hour", "0.59", "--utilization", "0.3",
                "--throughput", "output_throughput", "--load", "max_concurrency", *slos,
            ], text=True)
        self.assertIn("최대 측정 부하: max_concurrency=16", capacity)
        self.assertIn("최대 throughput: max_concurrency=8", capacity)
        self.assertIn("max_concurrency=8", cost)


if __name__ == "__main__":
    unittest.main()
