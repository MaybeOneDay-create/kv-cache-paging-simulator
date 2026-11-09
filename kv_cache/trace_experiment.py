"""Evaluate a numeric dialogue-length trace with the same CPU allocators."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from kv_cache.allocator import BLOCK_SIZE, blocks_for
from kv_cache.simulator import simulate
from kv_cache.svg_chart import write_comparison

CAPACITIES = (500, 1000, 2000, 3000, 6000, 9000, 12000, 16000, 20000,
              24000, 28000, 32000)
MODES = ("naive", "paged", "paged_dynamic")
BUCKETS = ((128, 511), (512, 1023), (1024, 2047), (2048, 4096))


def evaluate(trace_path: str, output_dir: str) -> dict:
    source = Path(trace_path)
    raw = source.read_bytes()
    payload = json.loads(raw)
    requests = payload.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("trace requests must be a nonempty list")
    if len({r["id"] for r in requests}) != len(requests):
        raise ValueError("duplicate request IDs")
    for request in requests:
        if (request["prompt_tokens"] <= 0 or request["generated_tokens"] <= 0
                or request["L"] != request["prompt_tokens"] + request["generated_tokens"]
                or request["L"] > request["M"]):
            raise ValueError("invalid request token lengths")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    naive_total = sum(blocks_for(r["M"]) for r in requests)
    paged_total = sum(blocks_for(r["L"]) for r in requests)
    rows = []
    bucket_rows = []
    for capacity in CAPACITIES:
        for mode in MODES:
            result = simulate(requests, capacity, mode)
            rows.append((capacity, mode, result["oom_rate"], result["avg_util"],
                         result["peak_concurrency"], result["accepted"],
                         result["rejected"]))
            for low, high in BUCKETS:
                ids = {r["id"] for r in requests if low <= r["L"] <= high}
                failed = len(ids & result["rejected_ids"])
                bucket_rows.append((capacity, mode, low, high, len(ids), failed,
                                    failed / len(ids) if ids else None))
    with (output / "results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("capacity_blocks", "mode", "oom_rate", "avg_util",
                         "peak_concurrency", "accepted", "rejected"))
        writer.writerows(rows)
    with (output / "length_buckets.csv").open("w", encoding="utf-8-sig",
                                               newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("capacity_blocks", "mode", "min_tokens", "max_tokens",
                         "requests", "oom_count", "oom_rate"))
        writer.writerows(bucket_rows)
    useful = sum(r["L"] for r in requests)
    with (output / "fragmentation.csv").open("w", encoding="utf-8-sig",
                                             newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("mode", "reserved_token_slots", "useful_token_slots",
                         "unused_token_slots", "internal_waste_ratio"))
        for mode, reserved in (("naive", naive_total * BLOCK_SIZE),
                               ("paged", paged_total * BLOCK_SIZE)):
            writer.writerow((mode, reserved, useful, reserved - useful,
                             (reserved - useful) / reserved))
    write_comparison(str(output / "comparison.svg"), rows, naive_total, paged_total)
    manifest = dict(trace_sha256=hashlib.sha256(raw).hexdigest(),
                    source_metadata=payload.get("metadata"), requests=len(requests),
                    block_size=BLOCK_SIZE, capacities=CAPACITIES,
                    modes=MODES, length_buckets=BUCKETS,
                    method="same arrival/duration trace for each allocator")
    (output / "experiment_config.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
