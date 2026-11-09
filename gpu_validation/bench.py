"""Run one side of a real GPU A/B test. Requires a dedicated CUDA instance.

Run this file in separate processes for hf_static and vllm; never label the
result as a measurement of the metadata-only allocator in kv_cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import threading
import time
from pathlib import Path


def failure_kind(error: BaseException) -> str:
    name = type(error).__name__.lower()
    message = str(error).lower()
    if "outofmemory" in name or "cuda out of memory" in message:
        return "cuda_oom"
    if ("no available memory for the cache blocks" in message
            or "insufficient kv cache" in message):
        return "engine_kv_capacity"
    return "other_error"


def save_json(path: str, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def device_memory_mib(gpu_index: int = 0) -> int | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5)
        return int(output.splitlines()[0].strip())
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return None


def selected_gpu_identity(gpu_index: int = 0) -> tuple[str, int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,memory.total", "--format=csv,noheader,nounits"],
        text=True, timeout=5)
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if not 0 <= gpu_index < len(rows):
        raise RuntimeError(f"GPU index {gpu_index} outside {len(rows)} physical GPUs")
    uuid, total = (part.strip() for part in rows[gpu_index].split(",", 1))
    if not uuid or not total.isdigit():
        raise RuntimeError("could not read GPU UUID and total memory")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if len(rows) > 1 and visible not in (str(gpu_index), uuid):
        raise RuntimeError(
            f"set CUDA_VISIBLE_DEVICES={gpu_index} to isolate the selected GPU")
    return uuid, int(total)


class MemorySampler:
    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self.peak_mib = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.is_set():
            value = device_memory_mib(self.gpu_index)
            if value is not None:
                self.peak_mib = max(self.peak_mib or value, value)
            self.stop.wait(0.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()


def make_ids(tokenizer, input_tokens):
    seed = tokenizer.encode("The quick brown fox studies memory paging. ",
                            add_special_tokens=False)
    if not seed:
        raise ValueError("tokenizer produced an empty seed")
    prefix = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    if input_tokens <= len(prefix):
        raise ValueError("input_tokens too small")
    return prefix + (seed * input_tokens)[:input_tokens - len(prefix)]


def configure_multiprocessing(backend: str) -> None:
    # vLLM's Python API normally forks its engine process. This script probes
    # torch.cuda before engine creation, so the child must be spawned instead.
    if backend == "vllm":
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def run(args):
    configure_multiprocessing(args.backend)
    import torch
    import transformers
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("benchmark requires exactly one CUDA-visible GPU")
    gpu_uuid, gpu_total_memory_mib = selected_gpu_identity(args.gpu_index)
    model_kwargs = {"revision": args.revision} if args.revision else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **model_kwargs)
    ids = make_ids(tokenizer, args.input_tokens)
    if args.input_tokens + args.output_tokens > args.max_model_len:
        raise ValueError("input + output exceeds max_model_len")
    if args.backend == "hf_static":
        from transformers import AutoModelForCausalLM, StaticCache
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=getattr(torch, args.dtype), **model_kwargs).to("cuda")

        def infer(batch):
            batch_ids = torch.tensor([ids] * batch, device="cuda")
            cache = StaticCache(config=model.config, max_cache_len=args.max_model_len)
            with torch.inference_mode():
                output = model.generate(
                    input_ids=batch_ids, do_sample=False, min_new_tokens=args.output_tokens,
                    max_new_tokens=args.output_tokens, past_key_values=cache,
                    pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                                  else tokenizer.eos_token_id))
            torch.cuda.synchronize()
            return int(output.shape[0] * (output.shape[1] - len(ids)))
        engine_version = transformers.__version__
    else:
        import vllm
        from vllm import LLM, SamplingParams
        engine = LLM(model=args.model, dtype=args.dtype, block_size=16,
                     max_model_len=args.max_model_len,
                     gpu_memory_utilization=args.gpu_memory_utilization,
                     enable_prefix_caching=False,
                     **({"revision": args.revision,
                         "tokenizer_revision": args.revision} if args.revision else {}))
        sampling = SamplingParams(temperature=0, max_tokens=args.output_tokens,
                                  min_tokens=args.output_tokens, ignore_eos=True)

        def infer(batch):
            prompts = [{"prompt_token_ids": ids} for _ in range(batch)]
            output = engine.generate(prompts, sampling, use_tqdm=False)
            return sum(len(item.outputs[0].token_ids) for item in output)
        engine_version = vllm.__version__

    # Warmup separately so one-time initialization is not included in timing.
    warmup_tokens = infer(1)
    if warmup_tokens != args.output_tokens:
        raise RuntimeError(
            f"warmup generated {warmup_tokens} tokens; expected {args.output_tokens}")
    results = []
    for batch in args.batch_sizes:
        expected = batch * args.output_tokens
        started = None
        elapsed = None
        with MemorySampler(args.gpu_index) as sampler:
            try:
                # Compile and fill shape-specific paths before the timed pass.
                for _ in range(args.batch_warmups):
                    warmup_generated = infer(batch)
                    if warmup_generated != expected:
                        raise RuntimeError(
                            f"batch warmup generated {warmup_generated} tokens; expected {expected}")
                started = time.perf_counter()
                generated = infer(batch)
                elapsed = time.perf_counter() - started
                if generated != expected:
                    raise RuntimeError(
                        f"generated {generated} tokens; expected {expected}")
                error = None
                error_kind = None
            except Exception as exc:
                if started is not None and elapsed is None:
                    elapsed = time.perf_counter() - started
                generated = 0
                error = f"{type(exc).__name__}: {exc}"
                error_kind = failure_kind(exc)
        results.append(dict(batch_size=batch, completed=batch if error is None else 0,
                            generated_tokens=generated, elapsed_s=elapsed,
                            output_tokens_per_s=generated / elapsed if elapsed and not error else None,
                            peak_device_memory_mib=sampler.peak_mib,
                            error_kind=error_kind, error=error))
        if error:
            break
    payload = dict(status="complete", backend=args.backend, model=args.model,
                   revision=args.revision,
                   dtype=args.dtype,
                   input_tokens=args.input_tokens, output_tokens=args.output_tokens,
                   prompt_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                   max_model_len=args.max_model_len, block_size=16,
                   batch_warmups=args.batch_warmups,
                   planned_batch_sizes=args.batch_sizes,
                   gpu_memory_utilization=args.gpu_memory_utilization,
                   python=platform.python_version(), torch=torch.__version__,
                   transformers=transformers.__version__, engine_version=engine_version,
                   gpu_name=torch.cuda.get_device_name(0), gpu_index=args.gpu_index,
                   gpu_uuid=gpu_uuid,
                   gpu_total_memory_mib=gpu_total_memory_mib, results=results)
    save_json(args.output, payload)
    print(args.output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("hf_static", "vllm"), required=True)
    parser.add_argument("--model", required=True, help="same local path or model ID for both runs")
    parser.add_argument("--revision", help="same immutable model/tokenizer commit for both runs")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--gpu-index", type=int, default=0,
                        help="physical nvidia-smi GPU index; isolate it via CUDA_VISIBLE_DEVICES")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--batch-warmups", type=int, default=2,
                        help="untimed warmup calls per batch size before measurement")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (any(n <= 0 for n in args.batch_sizes) or len(set(args.batch_sizes)) != len(args.batch_sizes)
            or args.input_tokens <= 0 or args.output_tokens <= 0
            or args.batch_warmups < 1):
        parser.error("all lengths and batch sizes must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu-memory-utilization must be in (0, 1]")
    if args.gpu_index < 0:
        parser.error("gpu-index must be nonnegative")
    try:
        run(args)
    except Exception as exc:
        save_json(args.output, dict(
            status="initialization_failed", backend=args.backend, model=args.model,
            revision=args.revision, dtype=args.dtype, input_tokens=args.input_tokens,
            output_tokens=args.output_tokens, max_model_len=args.max_model_len,
            batch_warmups=args.batch_warmups,
            planned_batch_sizes=args.batch_sizes,
            error_kind=failure_kind(exc), error=f"{type(exc).__name__}: {exc}",
            results=[]))
        raise


if __name__ == "__main__":
    main()
