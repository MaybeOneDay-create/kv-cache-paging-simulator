import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gpu_validation import run_ab
from gpu_validation.run_ab import pressure_rows, write_pressure_summary


class PressureSummaryTests(unittest.TestCase):
    def test_runner_continues_after_one_engine_initialization_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "scan"
            observed = []

            def fake_capture(command, log_path, _env):
                if log_path.name.startswith(("hf_static_", "vllm_")) and log_path.suffix == ".log":
                    stem = log_path.stem
                    observed.append(stem)
                    result_path = output / f"{stem}.json"
                    failure = stem == "vllm_1"
                    result_path.write_text(json.dumps({
                        "status": "initialization_failed" if failure else "complete",
                        "error_kind": "other_error" if failure else None,
                        "results": [] if failure else [
                            {"batch_size": 1, "completed": 1, "error": None},
                        ],
                    }), encoding="utf-8")
                    return 1 if failure else 0
                return 0

            argv = ["run_ab.py", "--model", "fake", "--revision", "fixed",
                    "--batch-sizes", "1", "--continue-on-failure",
                    "--output-dir", str(output)]
            with patch("sys.argv", argv), patch.object(run_ab, "capture", fake_capture):
                run_ab.main()
            self.assertEqual(observed, [
                "hf_static_1", "vllm_1", "hf_static_2", "vllm_2",
                "hf_static_3", "vllm_3",
            ])
            with (output / "pressure_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual(len(saved), 6)
            self.assertEqual(saved[1]["outcome"], "initialization_failed")
            self.assertFalse((output / "summary.csv").exists())
            self.assertIn("Skipped", (output / "comparison_1.log").read_text())

    def test_distinguishes_batch_failure_unattempted_and_initialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "hf_static_1.json").write_text(json.dumps({
                "status": "complete",
                "results": [
                    {"batch_size": 1, "completed": 1, "error": None,
                     "output_tokens_per_s": 12.5, "peak_device_memory_mib": 300},
                    {"batch_size": 2, "completed": 0, "error": "CUDA out of memory",
                     "error_kind": "cuda_oom", "peak_device_memory_mib": 500},
                ],
            }), encoding="utf-8")
            (output / "vllm_1.json").write_text(json.dumps({
                "status": "initialization_failed", "error_kind": "other_error",
                "results": [],
            }), encoding="utf-8")
            statuses = [
                {"repetition": 1, "backend": "hf_static", "stem": "hf_static_1", "returncode": 0},
                {"repetition": 1, "backend": "vllm", "stem": "vllm_1", "returncode": 1},
                {"repetition": 2, "backend": "vllm", "stem": "vllm_2", "returncode": 137},
            ]
            rows = pressure_rows(output, [1, 2, 4], statuses)
            self.assertEqual([row["outcome"] for row in rows], [
                "success", "failed", "not_run",
                "initialization_failed", "initialization_failed", "initialization_failed",
                "process_exit", "process_exit", "process_exit",
            ])
            self.assertEqual(rows[1]["error_kind"], "cuda_oom")
            self.assertEqual(rows[4]["error_kind"], "other_error")
            write_pressure_summary(output, [1, 2, 4], statuses)
            with (output / "pressure_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual(len(saved), 9)
            self.assertEqual(saved[-1]["outcome"], "process_exit")
            self.assertEqual(json.loads((output / "run_status.json").read_text()), statuses)


if __name__ == "__main__":
    unittest.main()
