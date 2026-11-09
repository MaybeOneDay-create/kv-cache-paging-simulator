"""Verify a run_ab.py output directory against its raw runs and summaries."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

if __package__:
    from .compare import compare
    from .summarize_runs import summarize
else:
    from compare import compare
    from summarize_runs import summarize


CONFIG_FIELDS = {
    "model": "model", "revision": "revision", "dtype": "dtype",
    "input_tokens": "input_tokens", "output_tokens": "output_tokens",
    "max_model_len": "max_model_len", "gpu_index": "gpu_index",
    "batch_warmups": "batch_warmups", "batch_sizes": "planned_batch_sizes",
    "gpu_memory_utilization": "gpu_memory_utilization",
}


def option_values(command: list[str], name: str) -> list[str]:
    if command.count(name) != 1:
        raise ValueError(f"command must contain exactly one {name}")
    start = command.index(name) + 1
    end = start
    while end < len(command) and not command[end].startswith("--"):
        end += 1
    if end == start:
        raise ValueError(f"command has no value for {name}")
    return command[start:end]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def required(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty evidence file: {path}")
    return path


def verify(directory: str) -> dict:
    folder = Path(directory).resolve()
    config_path = required(folder / "run_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repetitions = config["repetitions"]
    if not isinstance(repetitions, int) or repetitions < 3:
        raise ValueError("formal evidence needs at least three paired repetitions")
    batch_sizes = config["batch_sizes"]
    if not isinstance(batch_sizes, list) or not batch_sizes:
        raise ValueError("run_config batch_sizes missing")

    evidence = [config_path, required(folder / "nvidia_smi_before.txt")]
    pairs = []
    all_payloads = []
    for repetition in range(1, repetitions + 1):
        for backend in ("hf_static", "vllm"):
            stem = f"{backend}_{repetition}"
            result_path = required(folder / f"{stem}.json")
            command_path = required(folder / f"{stem}.command.json")
            log_path = required(folder / f"{stem}.log")
            smi_path = required(folder / f"nvidia_smi_after_{stem}.txt")
            evidence.extend((result_path, command_path, log_path, smi_path))
            command = json.loads(command_path.read_text(encoding="utf-8"))
            if not isinstance(command, list) or option_values(command, "--backend") != [backend]:
                raise ValueError(f"command/backend mismatch: {stem}")
            for config_key in CONFIG_FIELDS:
                flag = "--" + config_key.replace("_", "-")
                expected = config[config_key]
                values = [str(value) for value in expected] if isinstance(expected, list) else [str(expected)]
                if option_values(command, flag) != values:
                    raise ValueError(f"{stem}: {flag} differs from run_config")
            if Path(option_values(command, "--output")[0]).name != result_path.name:
                raise ValueError(f"{stem}: output path differs from result filename")
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if payload.get("status") != "complete":
                raise ValueError(f"initialization did not complete: {stem}")
            for config_key, payload_key in CONFIG_FIELDS.items():
                if payload.get(payload_key) != config.get(config_key):
                    raise ValueError(f"{stem}: {payload_key} differs from run_config")
            if payload.get("block_size") != 16:
                raise ValueError(f"{stem}: block_size is not 16")
            pairs.append(str(result_path))
            all_payloads.append(payload)
        comparison_path = required(folder / f"comparison_{repetition}.csv")
        evidence.extend((comparison_path, required(folder / f"comparison_{repetition}.log")))
    summary_path = required(folder / "summary.csv")
    evidence.extend((summary_path, required(folder / "summary.log")))

    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        for repetition in range(1, repetitions + 1):
            recalculated = temp / f"comparison_{repetition}.csv"
            compare(pairs[2 * (repetition - 1)], pairs[2 * (repetition - 1) + 1],
                    str(recalculated))
            original = folder / f"comparison_{repetition}.csv"
            if original.read_bytes() != recalculated.read_bytes():
                raise ValueError(f"comparison_{repetition}.csv differs from raw JSON")
        recalculated_summary = temp / "summary.csv"
        rows = summarize(pairs, str(recalculated_summary))
        if summary_path.read_bytes() != recalculated_summary.read_bytes():
            raise ValueError("summary.csv differs from raw JSON")

    return {
        "status": "verified", "directory": str(folder),
        "repetitions": repetitions, "batch_sizes": batch_sizes,
        "gpu_uuid": all_payloads[0]["gpu_uuid"],
        "result_rows": len(rows),
        "files": [{"name": path.name, "bytes": path.stat().st_size,
                   "sha256": digest(path)} for path in evidence],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify(args.dir)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Verified {result['repetitions']} pairs and {result['result_rows']} batches; wrote {target}")


if __name__ == "__main__":
    main()
