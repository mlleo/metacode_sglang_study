"""CPU checks for paired final-project results and workload accounting."""

import csv
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final_projects"))
from analyze import load_campaign, paired_interval, summarize_campaign
from run import planned_runs, select_configs
from workloads import make_workload, measure_request, summarize_workload


class FinalProjectTests(unittest.TestCase):
    def test_request_identity_and_stress_cases(self):
        for track in ("prefix", "mixed", "batch"):
            self.assertEqual(make_workload(track, "standard", 11), make_workload(track, "standard", 11))
            self.assertNotEqual(make_workload(track, "standard", 11), make_workload(track, "standard", 22))
        base, c1 = make_workload("prefix", "standard", 11)
        stress, c2 = make_workload("prefix", "stress", 11)
        self.assertEqual((len(base), c1), (len(stress), c2))
        self.assertTrue(all(len(req["input_ids"]) == 2048 for req in base + stress))
        self.assertEqual(len({row["group"] for row in stress}), 24)
        stress, _ = make_workload("mixed", "stress", 11)
        self.assertEqual(sum(req["role"] == "document" for req in stress), 2)
        stress, _ = make_workload("batch", "stress", 11)
        self.assertEqual(sum(req["max_new_tokens"] == 384 for req in stress), 24)

    def test_paired_order_reverses(self):
        expected = [("baseline", [11]), ("cap32", [11]), ("cap32", [22]), ("baseline", [22])]
        self.assertEqual(list(planned_runs(["baseline", "cap32"], [11, 22], "paired")), expected)

    def test_config_cannot_take_over_server_address(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(json.dumps({"batch": {"common": [], "configs": {"bad": ["--port=1234"]}}}))
            with self.assertRaisesRegex(ValueError, "포트"):
                select_configs(path, "batch", ["bad"])

    def test_mixed_uses_actual_chunk_gaps_and_rejects_no_overlap(self):
        common = {"e2e": 0.5, "ttft": 0.01, "completion_tokens": 3, "prompt_tokens": 10, "cached_tokens": 0}
        chat = {**common, "role": "chat", "token_times": [0.01, 0.04, 0.10], "itls": [0.015, 0.03]}
        doc = {**common, "role": "document", "t_start_rel": 0.09, "itls": [0.005]}
        metrics = summarize_workload("mixed", [chat, doc], 0.5)
        self.assertAlmostEqual(metrics["chat_gap_ms_max"], 60.0)
        self.assertLess(metrics["chat_itl_ms_p99"], 31.0)
        doc["t_start_rel"] = 0.11
        with self.assertRaisesRegex(ValueError, "겹침"):
            summarize_workload("mixed", [chat, doc], 0.5)

    def test_batch_deadline_counts_client_queue_and_uses_each_repeat(self):
        manifest = {"track": "batch", "mode": "gpu", "config_names": ["baseline", "fast"],
                    "profiles": ["standard"], "seeds": [11, 22, 33]}
        rows = {(profile, config, seed): {"wall_s": wall, "output_tok_per_s": throughput}
                for profile in manifest["profiles"] for config in manifest["config_names"]
                for seed, wall, throughput in zip(manifest["seeds"],
                                                  [9, 9, 9] if config == "baseline" else [6, 7, 13],
                                                  [100, 101, 102] if config == "baseline" else [201, 202, 203])}
        summary = summarize_campaign(manifest, rows, 100, 100, 12)
        self.assertEqual([row["config"] for row in summary], ["baseline", "fast"])
        self.assertEqual(summary[1]["status"], "fail")
        self.assertIn("s33", summary[1]["violations"])
        self.assertEqual(paired_interval([101, 101, 101]), (101, 101, 101))
        self.assertEqual(paired_interval([101]), (101, None, None))
        samples = [{"e2e": 1, "ttft": 0.1, "completion_tokens": 3, "prompt_tokens": 10, "cached_tokens": 0,
                    "finish_s": finish, "itls": [0.1]} for finish in [1, 3]]
        metrics = summarize_workload("batch", samples, 3.1)
        self.assertAlmostEqual(metrics["job_finish_s_p95"], 2.9)
        self.assertEqual(metrics["http_e2e_ms_p95"], 1000.0)

    def test_incomplete_stream_is_an_error(self):
        response = {"t_start": 10.0, "e2e": 0.5, "ttft": 0.1, "completion_tokens": 3,
                    "prompt_tokens": 1, "token_times": [10.1], "text": "mock", "stream_done": False}
        spec = {"idx": 0, "input_ids": [1000], "max_new_tokens": 3}
        with patch("workloads.generate", return_value=response):
            result = measure_request(spec, 10.0, "http://localhost", 1.0)
        self.assertIn("[DONE]", result["error"])

    def test_campaign_rejects_input_mismatch_and_missing_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            manifest = {"schema": 1, "track": "prefix", "config_names": ["baseline", "lpm"],
                        "profiles": ["standard"], "seeds": [11]}
            (path / "manifest.json").write_text(json.dumps(manifest))
            fields = ["config", "profile", "seed", "workload_sha256", "expected_requests", "completed"]
            with (path / "runs.csv").open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(fields)
                writer.writerow(["baseline", "standard", 11, "a" * 64, 96, 96])
            with self.assertRaisesRegex(ValueError, "일치"):
                load_campaign(path)
            with (path / "runs.csv").open("a", newline="") as file:
                csv.writer(file).writerow(["lpm", "standard", 11, "b" * 64, 96, 96])
            with self.assertRaisesRegex(ValueError, "입력"):
                load_campaign(path)


if __name__ == "__main__":
    unittest.main()
