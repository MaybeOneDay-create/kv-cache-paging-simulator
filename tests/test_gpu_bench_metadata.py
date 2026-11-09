import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from gpu_validation.bench import (configure_multiprocessing, failure_kind,
                                  save_json, selected_gpu_identity)


class BenchMetadataTests(unittest.TestCase):
    def test_vllm_spawn_after_cuda_probe(self):
        with patch.dict("gpu_validation.bench.os.environ",
                        {"VLLM_WORKER_MULTIPROC_METHOD": "fork"}):
            configure_multiprocessing("vllm")
            self.assertEqual(os.environ["VLLM_WORKER_MULTIPROC_METHOD"], "spawn")

    def test_failure_kind_distinguishes_cuda_and_engine_capacity(self):
        self.assertEqual(failure_kind(RuntimeError("CUDA out of memory")), "cuda_oom")
        self.assertEqual(failure_kind(RuntimeError(
            "No available memory for the cache blocks")), "engine_kv_capacity")
        self.assertEqual(failure_kind(ValueError("bad model path")), "other_error")

    def test_initialization_record_can_be_saved_before_engine_load(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "failure.json"
            save_json(str(path), {"status": "initialization_failed", "error_kind": "cuda_oom"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["error_kind"],
                             "cuda_oom")

    def test_identifies_selected_physical_gpu(self):
        with patch("gpu_validation.bench.subprocess.check_output",
                   return_value="GPU-123, 16384\n"):
            self.assertEqual(selected_gpu_identity(), ("GPU-123", 16384))
        with patch("gpu_validation.bench.subprocess.check_output",
                   return_value="GPU-123, 16384\nGPU-456, 16384\n"):
            with patch.dict("gpu_validation.bench.os.environ",
                            {"CUDA_VISIBLE_DEVICES": "1"}):
                self.assertEqual(selected_gpu_identity(1), ("GPU-456", 16384))
            with patch.dict("gpu_validation.bench.os.environ",
                            {"CUDA_VISIBLE_DEVICES": "0"}):
                with self.assertRaisesRegex(RuntimeError, "CUDA_VISIBLE_DEVICES=1"):
                    selected_gpu_identity(1)
