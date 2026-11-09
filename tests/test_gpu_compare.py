import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from gpu_validation.compare import compare


class CompareTests(unittest.TestCase):
    def payload(self, backend, failed=False):
        results = [dict(batch_size=1, completed=1, output_tokens_per_s=100,
                        peak_device_memory_mib=1000, error=None)]
        if failed:
            results.append(dict(batch_size=2, completed=0, output_tokens_per_s=0,
                                peak_device_memory_mib=1200, error="CUDA OOM"))
        else:
            results.append(dict(batch_size=2, completed=2, output_tokens_per_s=180,
                                peak_device_memory_mib=1100, error=None))
        return dict(backend=backend, model="same/model", revision="fixed-commit",
                    dtype="float16",
                    input_tokens=512, output_tokens=128, max_model_len=4096,
                    prompt_ids_sha256="samehash", gpu_name="same GPU",
                    gpu_index=0, gpu_uuid="GPU-123", gpu_total_memory_mib=16000,
                    planned_batch_sizes=[1, 2, 4], results=results)

    def test_failed_and_unrun_batches_are_distinct(self):
        with TemporaryDirectory() as directory:
            hf = Path(directory) / "hf.json"
            vl = Path(directory) / "vl.json"
            out = Path(directory) / "comparison.csv"
            hf.write_text(json.dumps(self.payload("hf_static", failed=True)), encoding="utf-8")
            vl.write_text(json.dumps(self.payload("vllm")), encoding="utf-8")
            rows = compare(str(hf), str(vl), str(out))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[1]["hf_error"], "CUDA OOM")
            self.assertEqual(rows[2]["hf_error"], "not_run")
            with out.open(encoding="utf-8-sig", newline="") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 3)

    def test_mismatched_trace_is_rejected(self):
        with TemporaryDirectory() as directory:
            hf = Path(directory) / "hf.json"
            vl = Path(directory) / "vl.json"
            out = Path(directory) / "comparison.csv"
            a, b = self.payload("hf_static"), self.payload("vllm")
            b["prompt_ids_sha256"] = "different"
            hf.write_text(json.dumps(a), encoding="utf-8")
            vl.write_text(json.dumps(b), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prompt_ids_sha256"):
                compare(str(hf), str(vl), str(out))

    def test_mismatched_model_revision_is_rejected(self):
        with TemporaryDirectory() as directory:
            hf = Path(directory) / "hf.json"
            vl = Path(directory) / "vl.json"
            out = Path(directory) / "comparison.csv"
            a, b = self.payload("hf_static"), self.payload("vllm")
            b["revision"] = "another-commit"
            hf.write_text(json.dumps(a), encoding="utf-8")
            vl.write_text(json.dumps(b), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "revision"):
                compare(str(hf), str(vl), str(out))

    def test_initialization_failure_is_not_compared_as_unrun(self):
        with TemporaryDirectory() as directory:
            hf = Path(directory) / "hf.json"
            vl = Path(directory) / "vl.json"
            out = Path(directory) / "comparison.csv"
            a, b = self.payload("hf_static"), self.payload("vllm")
            a["status"] = "initialization_failed"
            hf.write_text(json.dumps(a), encoding="utf-8")
            vl.write_text(json.dumps(b), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "initialization failed"):
                compare(str(hf), str(vl), str(out))

    def test_different_physical_gpu_is_rejected(self):
        with TemporaryDirectory() as directory:
            hf = Path(directory) / "hf.json"
            vl = Path(directory) / "vl.json"
            out = Path(directory) / "comparison.csv"
            a, b = self.payload("hf_static"), self.payload("vllm")
            b["gpu_uuid"] = "GPU-456"
            hf.write_text(json.dumps(a), encoding="utf-8")
            vl.write_text(json.dumps(b), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "gpu_uuid"):
                compare(str(hf), str(vl), str(out))
