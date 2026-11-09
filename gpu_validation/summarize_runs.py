"""Summarize repeated, paired engine runs without hiding failures."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

if __package__:
    from .compare import SAME_FIELDS
else:
    from compare import SAME_FIELDS


def summarize(paths: list[str], output_path: str) -> list[dict]:
    if len(paths) < 6 or len(paths) % 2:
        raise ValueError("provide at least three hf_static/vllm JSON pairs")
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    first = payloads[0]
    if not first.get("gpu_uuid"):
        raise ValueError("all runs must record a physical GPU UUID")
    plan = first.get("planned_batch_sizes")
    if not isinstance(plan, list) or not plan:
        raise ValueError("planned_batch_sizes must be nonempty")
    for index, payload in enumerate(payloads):
        if payload.get("status", "complete") != "complete":
            raise ValueError(f"run {index + 1} failed during initialization")
        expected_backend = "hf_static" if index % 2 == 0 else "vllm"
        if payload.get("backend") != expected_backend:
            raise ValueError(f"run {index + 1} must use {expected_backend}")
        if payload.get("planned_batch_sizes") != plan:
            raise ValueError("planned batch sizes differ across runs")
        fields = [field for field in SAME_FIELDS if payload.get(field) != first.get(field)]
        if fields:
            raise ValueError("A/B configuration mismatch: " + ", ".join(fields))
        if index >= 2:
            prior_same_backend = payloads[index - 2]
            for field in ("engine_version", "torch", "transformers"):
                if payload.get(field) != prior_same_backend.get(field):
                    raise ValueError(f"{field} changed across {expected_backend} runs")
    rows = []
    repetitions = len(paths) // 2
    for batch in plan:
        row = {"batch_size": batch, "repetitions": repetitions}
        for backend, offset in (("hf", 0), ("vllm", 1)):
            outcomes = []
            for payload in payloads[offset::2]:
                outcome = next((item for item in payload["results"]
                                if item["batch_size"] == batch), None)
                outcomes.append(outcome)
            successful = [item for item in outcomes
                          if item is not None and item.get("error") is None
                          and item.get("completed") == batch]
            row[f"{backend}_successful_runs"] = len(successful)
            row[f"{backend}_failed_runs"] = sum(
                item is not None and item.get("error") is not None for item in outcomes)
            row[f"{backend}_not_run"] = sum(item is None for item in outcomes)
            for key, label in (("output_tokens_per_s", "tokens_per_s"),
                               ("peak_device_memory_mib", "peak_memory_mib")):
                values = [item[key] for item in successful if item.get(key) is not None]
                row[f"{backend}_{label}_mean"] = statistics.mean(values) if values else None
                row[f"{backend}_{label}_stdev"] = (
                    statistics.stdev(values) if len(values) > 1 else None)
        rows.append(row)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", required=True,
                        help="hf_1.json vllm_1.json hf_2.json vllm_2.json ...")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = summarize(args.pairs, args.output)
    print(f"Summarized {len(args.pairs) // 2} paired runs across {len(rows)} batch sizes")


if __name__ == "__main__":
    main()
