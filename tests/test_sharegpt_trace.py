from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from gpu_validation.prepare_sharegpt_trace import build_trace, sha256_file
from kv_cache.simulator import simulate


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        return [1, 2] + [3 for message in messages
                         for _ in message["content"].split()]

    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return [4 for _ in text.split()]


class ShareGPTTraceTests(unittest.TestCase):
    def test_real_lengths_with_explicitly_synthetic_timing(self):
        records = [
            {"id": "a", "conversations": [
                {"from": "human", "value": "one two three"},
                {"from": "gpt", "value": "four five"}]},
            {"id": "b", "conversations": [
                {"from": "human", "value": "alpha beta"},
                {"from": "gpt", "value": "gamma delta epsilon"}]},
        ]
        first = build_trace(records, FakeTokenizer(), 2, 42, 2, 32, 0.15,
                            (8.0, 25.0))
        second = build_trace(records, FakeTokenizer(), 2, 42, 2, 32, 0.15,
                             (8.0, 25.0))
        self.assertEqual(first, second)
        self.assertEqual({r["source_id"] for r in first}, {"a", "b"})
        self.assertTrue(all(r["prompt_tokens"] + r["generated_tokens"] == r["L"]
                            for r in first))
        self.assertTrue(all(r["dur"] > 0 for r in first))
        self.assertEqual(simulate(first, 32, "paged_dynamic")["rejected"], 0)

    def test_insufficient_eligible_conversations(self):
        records = [{"conversations": [{"from": "human", "value": "only user"}]}]
        with self.assertRaisesRegex(ValueError, "eligible conversations"):
            build_trace(records, FakeTokenizer(), 1, 1, 2, 32, 0.15,
                        (8.0, 25.0))

    def test_sha256_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sample"
            path.write_bytes(b"abc")
            self.assertEqual(sha256_file(path),
                             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
