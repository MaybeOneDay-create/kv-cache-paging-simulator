import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from kv_cache.trace_experiment import evaluate


class TraceExperimentTests(unittest.TestCase):
    def test_numeric_trace_produces_matching_comparison_files(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "trace.json"
            output = Path(directory) / "output"
            source.write_text(json.dumps({
                "metadata": {"dataset_id": "sharegpt-test"},
                "requests": [
                    {"id": 0, "ta": 0.0, "dur": 8.0, "L": 192, "M": 512,
                     "prompt_tokens": 96, "generated_tokens": 96},
                    {"id": 1, "ta": 0.1, "dur": 8.0, "L": 640, "M": 1024,
                     "prompt_tokens": 320, "generated_tokens": 320},
                ]}), encoding="utf-8")
            manifest = evaluate(str(source), str(output))
            self.assertEqual(manifest["requests"], 2)
            with (output / "results.csv").open(encoding="utf-8-sig", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 36)
            with (output / "length_buckets.csv").open(
                    encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 144)
            self.assertTrue((output / "comparison.svg").exists())

    def test_invalid_total_is_rejected(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "trace.json"
            source.write_text(json.dumps({"requests": [
                {"id": 0, "ta": 0, "dur": 1, "L": 10, "M": 20,
                 "prompt_tokens": 5, "generated_tokens": 4}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "token lengths"):
                evaluate(str(source), str(Path(directory) / "out"))
