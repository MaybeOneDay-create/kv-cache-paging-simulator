import json
import shutil
import tempfile
import unittest
from pathlib import Path

from gpu_validation.verify_evidence import verify


SOURCE = Path(__file__).resolve().parents[1] / "review_outputs" / "gpu_formal" / "main_512_128"


class EvidenceVerificationTests(unittest.TestCase):
    def test_preserved_colab_runs_and_detects_tampering(self):
        self.assertEqual(verify(str(SOURCE))["status"], "verified")
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "run"
            shutil.copytree(SOURCE, copied)
            summary = copied / "summary.csv"
            summary.write_bytes(summary.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "summary.csv differs"):
                verify(str(copied))

            shutil.copy2(SOURCE / "summary.csv", summary)
            command = copied / "vllm_1.command.json"
            arguments = json.loads(command.read_text(encoding="utf-8"))
            arguments[arguments.index("--input-tokens") + 1] = "1024"
            command.write_text(json.dumps(arguments), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "--input-tokens differs"):
                verify(str(copied))
