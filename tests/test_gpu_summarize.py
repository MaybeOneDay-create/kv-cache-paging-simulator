import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from gpu_validation.summarize_runs import summarize


class SummarizeTests(unittest.TestCase):
    def payload(self, backend, speed, fail=False):
        results = [dict(batch_size=1, completed=1,
                        output_tokens_per_s=speed,
                        peak_device_memory_mib=1000, error=None)]
        if fail:
            results.append(dict(batch_size=2, completed=0,
                                output_tokens_per_s=0,
                                peak_device_memory_mib=1200, error="CUDA OOM"))
        else:
            results.append(dict(batch_size=2, completed=2,
                                output_tokens_per_s=2 * speed,
                                peak_device_memory_mib=1100, error=None))
        return dict(backend=backend, model="m", revision="commit", dtype="float16",
                    input_tokens=512, output_tokens=128, max_model_len=4096,
                    prompt_ids_sha256="hash", gpu_name="GPU", gpu_index=0,
                    gpu_uuid="GPU-123",
                    gpu_total_memory_mib=16000, engine_version="1",
                    torch="2", transformers="4", planned_batch_sizes=[1, 2],
                    results=results)

    def test_three_pairs_keep_failures_out_of_mean(self):
        with TemporaryDirectory() as directory:
            paths = []
            for index in range(3):
                for backend in ("hf_static", "vllm"):
                    payload = self.payload(backend, 100 + index * 10,
                                           fail=(index == 1 and backend == "hf_static"))
                    path = Path(directory) / f"{index}_{backend}.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    paths.append(str(path))
            output = str(Path(directory) / "sub" / "summary.csv")
            rows = summarize(paths, output)
            self.assertEqual(rows[0]["hf_tokens_per_s_mean"], 110)
            self.assertEqual(rows[1]["hf_successful_runs"], 2)
            self.assertEqual(rows[1]["hf_failed_runs"], 1)
            self.assertEqual(rows[1]["hf_tokens_per_s_mean"], 220)
            self.assertEqual(rows[1]["vllm_successful_runs"], 3)
            with Path(output).open(encoding="utf-8-sig", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)

    def test_reject_mismatched_revision(self):
        with TemporaryDirectory() as directory:
            paths = []
            for index in range(3):
                for backend in ("hf_static", "vllm"):
                    payload = self.payload(backend, 100)
                    if index == 2 and backend == "vllm":
                        payload["revision"] = "other"
                    path = Path(directory) / f"{index}_{backend}.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    paths.append(str(path))
            with self.assertRaisesRegex(ValueError, "revision"):
                summarize(paths, str(Path(directory) / "summary.csv"))
