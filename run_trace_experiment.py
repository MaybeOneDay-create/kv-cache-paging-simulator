"""Run the CPU allocator comparison on a pretokenized ShareGPT trace."""
import argparse

from kv_cache.trace_experiment import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output-dir", default="review_outputs/sharegpt")
    args = parser.parse_args()
    manifest = evaluate(args.trace, args.output_dir)
    print(f"Evaluated {manifest['requests']} requests in {args.output_dir}")


if __name__ == "__main__":
    main()
