"""Compare two real GPU benchmark JSON files without causal overclaiming."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SAME_FIELDS = ("model", "revision", "dtype", "input_tokens", "output_tokens", "max_model_len",
               "batch_warmups", "prompt_ids_sha256", "gpu_name", "gpu_index", "gpu_uuid",
               "gpu_total_memory_mib")


def compare(hf_path: str, vllm_path: str, output_path: str) -> list[dict]:
    hf = json.loads(Path(hf_path).read_text(encoding="utf-8"))
    vllm = json.loads(Path(vllm_path).read_text(encoding="utf-8"))
    if hf.get("backend") != "hf_static" or vllm.get("backend") != "vllm":
        raise ValueError("expected hf_static JSON followed by vllm JSON")
    if hf.get("status", "complete") != "complete" or vllm.get("status", "complete") != "complete":
        raise ValueError("initialization failed; inspect the raw benchmark JSON files")
    if not hf.get("gpu_uuid") or not vllm.get("gpu_uuid"):
        raise ValueError("both runs must record a physical GPU UUID")
    differences = [key for key in SAME_FIELDS if hf.get(key) != vllm.get(key)]
    if differences:
        raise ValueError("A/B configuration mismatch: " + ", ".join(differences))
    hf_rows = {r["batch_size"]: r for r in hf["results"]}
    vllm_rows = {r["batch_size"]: r for r in vllm["results"]}
    plan = hf.get("planned_batch_sizes")
    if not isinstance(plan, list) or plan != vllm.get("planned_batch_sizes"):
        raise ValueError("planned batch size sets differ")
    rows = []
    for batch in plan:
        a, b = hf_rows.get(batch, {}), vllm_rows.get(batch, {})
        rows.append(dict(
            batch_size=batch,
            hf_completed=a.get("completed"), vllm_completed=b.get("completed"),
            hf_tokens_per_s=a.get("output_tokens_per_s"),
            vllm_tokens_per_s=b.get("output_tokens_per_s"),
            hf_peak_device_memory_mib=a.get("peak_device_memory_mib"),
            vllm_peak_device_memory_mib=b.get("peak_device_memory_mib"),
            hf_error_kind=a.get("error_kind", "not_run"),
            vllm_error_kind=b.get("error_kind", "not_run"),
            hf_error=a.get("error", "not_run"), vllm_error=b.get("error", "not_run"),
        ))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["batch_size"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf", required=True)
    parser.add_argument("--vllm", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = compare(args.hf, args.vllm, args.output)
    print(f"Compared {len(rows)} matching batch sizes; wrote {args.output}")
    print("Interpret as an engine-level A/B comparison, not an isolated paging effect.")


if __name__ == "__main__":
    main()
