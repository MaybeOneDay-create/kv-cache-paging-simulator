import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kv_cache.experiments import run, run_long_context
from kv_cache.workload import make_workload, split_total_length


class ExperimentTests(unittest.TestCase):
    def test_seeded_workload(self):
        self.assertEqual(make_workload(3, seed=7), make_workload(3, seed=7))
        split = make_workload(3, seed=7, prompt_fraction=0.5)
        self.assertTrue(all(s["prompt_tokens"] + s["generated_tokens"] == s["L"]
                            for s in split))
        self.assertEqual(split_total_length(100, 0.5), (50, 50))

    def test_long_context_strata(self):
        with TemporaryDirectory() as directory:
            path = run_long_context(directory, requests_per_length=2)
            with open(path, encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 5 * 3 * 4)
            self.assertEqual({int(r["actual_tokens"]) for r in rows},
                             {512, 1024, 2048, 4096})
            self.assertTrue(all(int(r["requests"]) == 2 for r in rows))
            self.assertTrue(all(0 <= float(r["oom_rate"]) <= 1 for r in rows))

    def test_long_context_explicit_prompt_split(self):
        with TemporaryDirectory() as directory:
            path = run_long_context(directory, requests_per_length=1, prompt_fraction=0.5)
            with open(path, encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 60)

    def test_main_csv_has_all_modes(self):
        with TemporaryDirectory() as directory:
            sample = make_workload(8, seed=42)
            with patch("kv_cache.experiments.run_long_context", return_value="long_context.csv"):
                with patch("kv_cache.experiments.make_workload", return_value=sample):
                    result = run(directory)
            self.assertEqual(result["naive_total"], 8 * 128)
            self.assertGreater(result["ratio"], 1)
            path = Path(directory) / "results.csv"
            with path.open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual({r["mode"] for r in rows},
                             {"naive", "paged", "paged_dynamic"})
            with (Path(directory) / "fragmentation.csv").open(encoding="utf-8-sig", newline="") as f:
                waste = {r["mode"]: r for r in csv.DictReader(f)}
            self.assertGreater(float(waste["naive"]["internal_waste_ratio"]),
                               float(waste["paged"]["internal_waste_ratio"]))
            config = json.loads((Path(directory) / "experiment_config.json").read_text(encoding="utf-8"))
            self.assertIsNone(config["prompt_fraction"])


if __name__ == "__main__":
    unittest.main()
