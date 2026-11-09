import unittest

from kv_cache.simulator import simulate


class SimulatorTests(unittest.TestCase):
    def test_time_weighted_utilization(self):
        workload = [
            {"id": 1, "ta": 0, "dur": 10, "L": 16, "M": 16},
            {"id": 2, "ta": 9, "dur": 1, "L": 16, "M": 16},
        ]
        # One block for 9 seconds, two blocks for 1 second: 11/20.
        result = simulate(workload, 2, "paged")
        self.assertAlmostEqual(result["avg_util"], 0.55)
        self.assertEqual(result["peak_concurrency"], 2)

    def test_empty_workload(self):
        result = simulate([], 2, "paged")
        self.assertEqual(result["oom_rate"], 0)
        self.assertEqual(result["avg_util"], 0)

    def test_departure_frees_capacity_before_same_time_arrival(self):
        workload = [
            {"id": 1, "ta": 0, "dur": 1, "L": 16, "M": 16,
             "prompt_tokens": 8, "generated_tokens": 8},
            {"id": 2, "ta": 1, "dur": 1, "L": 16, "M": 16,
             "prompt_tokens": 8, "generated_tokens": 8},
        ]
        for mode in ("naive", "paged", "paged_dynamic"):
            with self.subTest(mode=mode):
                result = simulate(workload, 1, mode)
                self.assertEqual(result["rejected"], 0)
                self.assertEqual(result["peak_concurrency"], 1)

    def test_dynamic_growth_can_fail_after_admission(self):
        workload = [
            {"id": 1, "ta": 0, "dur": 10, "L": 32, "M": 32},
            {"id": 2, "ta": 0, "dur": 10, "L": 32, "M": 32},
        ]
        result = simulate(workload, 2, "paged_dynamic")
        self.assertEqual(result["rejected"], 1)
        self.assertEqual(result["accepted"], 1)
        self.assertEqual(result["peak_concurrency"], 2)

    def test_prompt_prefill_can_fail_at_arrival(self):
        workload = [
            {"id": 1, "ta": 0, "dur": 10, "L": 32, "M": 32,
             "prompt_tokens": 24, "generated_tokens": 8},
        ]
        result = simulate(workload, 1, "paged_dynamic")
        self.assertEqual(result["rejected_ids"], {1})
        self.assertEqual(result["peak_concurrency"], 0)

    def test_prompt_and_generation_must_sum_to_total(self):
        bad = [{"id": 1, "ta": 0, "dur": 1, "L": 32, "M": 32,
                "prompt_tokens": 16, "generated_tokens": 15}]
        with self.assertRaises(ValueError):
            simulate(bad, 3, "paged_dynamic")


if __name__ == "__main__":
    unittest.main()
