"""Run paired GPU engine experiments in fresh processes and retain raw evidence."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import distributions
from pathlib import Path


HERE = Path(__file__).resolve().parent


def pressure_rows(output: Path, batch_sizes: list[int], statuses: list[dict]) -> list[dict]:
    """Preserve failed and unattempted batches without treating them as OOM."""
    rows = []
    for status in statuses:
        result_path = output / f"{status['stem']}.json"
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        outcomes = {item["batch_size"]: item for item in payload.get("results", [])}
        for batch in batch_sizes:
            item = outcomes.get(batch)
            if item is not None:
                outcome = "success" if item.get("error") is None and item.get("completed") == batch else "failed"
                error_kind = item.get("error_kind")
            elif payload.get("status") == "initialization_failed":
                outcome, error_kind = "initialization_failed", payload.get("error_kind")
            elif payload.get("status") == "complete":
                outcome, error_kind = "not_run", None
            else:
                outcome, error_kind = "process_exit", None
            rows.append(dict(repetition=status["repetition"], backend=status["backend"],
                             batch_size=batch, outcome=outcome, error_kind=error_kind,
                             completed=item.get("completed") if item else None,
                             output_tokens_per_s=item.get("output_tokens_per_s") if item else None,
                             peak_device_memory_mib=item.get("peak_device_memory_mib") if item else None,
                             process_returncode=status["returncode"],
                             raw_result=result_path.name if result_path.is_file() else None))
    return rows


def write_pressure_summary(output: Path, batch_sizes: list[int], statuses: list[dict]) -> None:
    rows = pressure_rows(output, batch_sizes, statuses)
    with (output / "pressure_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "run_status.json").write_text(
        json.dumps(statuses, ensure_ascii=False, indent=2), encoding="utf-8")


def capture(args: list[str], path: Path, env: dict[str, str]) -> int:
    with path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(args, stdout=handle, stderr=subprocess.STDOUT,
                                env=env, check=False)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--batch-warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--continue-on-failure", action="store_true",
                        help="pressure scans: retain all pairs despite initialization/process failures")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.repetitions < 3:
        parser.error("formal A/B requires at least three paired repetitions")
    if (args.input_tokens <= 0 or args.output_tokens <= 0
            or args.input_tokens + args.output_tokens > args.max_model_len
            or not args.batch_sizes or any(n <= 0 for n in args.batch_sizes)
            or args.batch_sizes != sorted(set(args.batch_sizes))
            or args.batch_warmups < 1
            or args.gpu_index < 0 or not 0 < args.gpu_memory_utilization <= 1):
        parser.error("invalid lengths, batch sizes, GPU index or utilization")

    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output-dir must be empty; choose a new directory for each scan")
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    config = vars(args).copy()
    config.update(python=sys.version, executable=sys.executable,
                  platform=platform.platform(), cuda_visible_devices=env["CUDA_VISIBLE_DEVICES"],
                  installed_packages=sorted(
                      f"{dist.metadata['Name']}=={dist.version}"
                      for dist in distributions() if dist.metadata.get("Name")))
    (output / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    capture(["nvidia-smi"], output / "nvidia_smi_before.txt", env)

    pairs: list[str] = []
    statuses: list[dict] = []
    all_complete = True
    for repetition in range(1, args.repetitions + 1):
        pair_complete = True
        for backend in ("hf_static", "vllm"):
            stem = f"{backend}_{repetition}"
            result_path = output / f"{stem}.json"
            command = [sys.executable, str(HERE / "bench.py"),
                       "--backend", backend, "--model", args.model,
                       "--revision", args.revision, "--dtype", args.dtype,
                       "--input-tokens", str(args.input_tokens),
                       "--output-tokens", str(args.output_tokens),
                       "--max-model-len", str(args.max_model_len),
                       "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                       "--gpu-index", str(args.gpu_index),
                       "--batch-sizes", *map(str, args.batch_sizes),
                       "--batch-warmups", str(args.batch_warmups),
                       "--output", str(result_path)]
            (output / f"{stem}.command.json").write_text(
                json.dumps(command, indent=2), encoding="utf-8")
            code = capture(command, output / f"{stem}.log", env)
            capture(["nvidia-smi"], output / f"nvidia_smi_after_{stem}.txt", env)
            status = {"repetition": repetition, "backend": backend,
                      "stem": stem, "returncode": code}
            statuses.append(status)
            if code and not args.continue_on_failure:
                raise SystemExit(f"{stem} exited {code}; inspect {stem}.log and .json")
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                payload = {}
            if code or payload.get("status") != "complete":
                pair_complete = False
                all_complete = False
            pairs.append(str(result_path))
        if pair_complete:
            code = capture([sys.executable, str(HERE / "compare.py"),
                            "--hf", pairs[-2], "--vllm", pairs[-1],
                            "--output", str(output / f"comparison_{repetition}.csv")],
                           output / f"comparison_{repetition}.log", env)
            if code:
                raise SystemExit(f"comparison_{repetition} failed; inspect its log")
        else:
            (output / f"comparison_{repetition}.log").write_text(
                "Skipped: at least one engine did not initialize or exit successfully.\n",
                encoding="utf-8")
    if args.continue_on_failure:
        write_pressure_summary(output, args.batch_sizes, statuses)
    if all_complete:
        code = capture([sys.executable, str(HERE / "summarize_runs.py"),
                        "--pairs", *pairs, "--output", str(output / "summary.csv")],
                       output / "summary.log", env)
        if code:
            raise SystemExit("summary failed; inspect summary.log")
    elif args.continue_on_failure:
        (output / "summary.log").write_text(
            "Skipped: at least one engine failed initialization; see pressure_summary.csv.\n",
            encoding="utf-8")
    print(f"Saved {args.repetitions} paired runs to {output}")


if __name__ == "__main__":
    main()
