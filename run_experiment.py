# -*- coding: utf-8 -*-
"""实验入口：运行 KV Cache 分页 vs 朴素连续分配的对比实验。

用法：
    python run_experiment.py
结果：默认按已确认的 50% prompt 比例运行；--sensitivity 生成三组敏感性结果。
"""
import argparse
from pathlib import Path

from kv_cache.experiments import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KV Cache CPU simulation")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prompt-fraction", type=float, default=0.5,
                        help="fraction of total L used by prompt; default is the confirmed 50 percent")
    parser.add_argument("--sensitivity", action="store_true",
                        help="run 25, 50 and 75 percent prompt fractions")
    args = parser.parse_args()
    if not 0 < args.prompt_fraction < 1:
        parser.error("prompt-fraction must be between zero and one")
    if args.sensitivity:
        root = Path(args.output_dir or "review_outputs")
        for fraction in (0.25, 0.5, 0.75):
            output = root / "sensitivity" / str(fraction)
            print(f"\n=== prompt fraction {fraction:.0%}: {output} ===")
            run(str(output), prompt_fraction=fraction)
        print("\n三组比例为敏感性分析；正式主实验采用 50%。")
    else:
        label = f"{args.prompt_fraction * 100:g}pct"
        output = args.output_dir or str(Path("review_outputs") / f"formal_{label}")
        run(output, prompt_fraction=args.prompt_fraction)
