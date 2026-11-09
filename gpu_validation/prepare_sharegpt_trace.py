"""Convert pinned ShareGPT conversations to a numeric KV workload.

The conversation text determines token lengths. Arrival times and durations are
synthetic because the published conversation file has no request timestamps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

DATASET_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
DATASET_REVISION = "192ab2185289094fc556ec8ce5ce1e8e587154ca"
DATASET_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"
DATASET_SHA256 = "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def conversation_pairs(record: dict, tokenizer, min_total: int, max_context: int):
    messages = []
    for turn in record.get("conversations", []):
        role = {"human": "user", "gpt": "assistant"}.get(turn.get("from"))
        text = turn.get("value")
        if role is None or not isinstance(text, str) or not text.strip():
            continue
        if role == "assistant" and messages and messages[-1]["role"] == "user":
            prompt_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True)
            response_ids = tokenizer.encode(text, add_special_tokens=False)
            total = len(prompt_ids) + len(response_ids)
            if (prompt_ids and response_ids and min_total <= total <= max_context):
                yield len(prompt_ids), len(response_ids)
        messages.append({"role": role, "content": text})


def build_trace(records: list[dict], tokenizer, count: int, seed: int,
                min_total: int, max_context: int, arrival_mean: float,
                duration_range: tuple[float, float]) -> list[dict]:
    if count <= 0 or min_total < 2 or max_context < min_total or arrival_mean <= 0:
        raise ValueError("invalid trace parameters")
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    requests = []
    arrival = 0.0
    for index in indices:
        record = records[index]
        if not isinstance(record, dict):
            continue
        candidates = list(conversation_pairs(record, tokenizer, min_total, max_context))
        if not candidates:
            continue
        prompt, generated = rng.choice(candidates)
        arrival += rng.expovariate(1.0 / arrival_mean)
        requests.append(dict(
            id=len(requests), ta=arrival, dur=rng.uniform(*duration_range),
            L=prompt + generated, M=max_context,
            prompt_tokens=prompt, generated_tokens=generated,
            source_id=str(record.get("id", index))))
        if len(requests) == count:
            break
    if len(requests) < count:
        raise ValueError(f"only {len(requests)} eligible conversations; expected {count}")
    return requests


def prepare(input_path: str, output_path: str, model: str, revision: str,
            count: int = 800, seed: int = 42, min_total: int = 128,
            max_context: int = 4096) -> dict:
    from transformers import AutoTokenizer

    source = Path(input_path)
    source_sha256 = sha256_file(source)
    if source_sha256 != DATASET_SHA256:
        raise ValueError("ShareGPT file SHA256 differs from pinned dataset")
    with source.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("ShareGPT source must be a JSON array")
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    requests = build_trace(records, tokenizer, count, seed, min_total,
                           max_context, 0.15, (8.0, 25.0))
    payload = dict(
        metadata=dict(dataset_id=DATASET_ID, dataset_revision=DATASET_REVISION,
                      dataset_file=DATASET_FILE, dataset_sha256=source_sha256,
                      tokenizer_model=model, tokenizer_revision=revision,
                      requests=count, seed=seed, min_total_tokens=min_total,
                      max_context=max_context, arrival_mean=0.15,
                      duration_range=[8.0, 25.0],
                      timing="synthetic Poisson arrivals and uniform durations"),
        requests=requests)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="pinned ShareGPT JSON file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True, help="immutable tokenizer commit")
    parser.add_argument("--count", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-total", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=4096)
    args = parser.parse_args()
    payload = prepare(args.input, args.output, args.model, args.revision,
                      args.count, args.seed, args.min_total, args.max_context)
    print(f"Wrote {len(payload['requests'])} numeric requests to {args.output}")


if __name__ == "__main__":
    main()
