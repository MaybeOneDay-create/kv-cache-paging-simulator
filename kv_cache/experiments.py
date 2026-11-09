# -*- coding: utf-8 -*-
"""
experiments.py —— 角色 D（测试与评测）：对比实验 / 可视化 / CSV 导出
====================================================================
容量扫描 + 出图 + 出 CSV，结果直接对应评分指标：
  - 碎片下降与累计预留块需求（不等于 GPU 显存利用率）
  - 单卡并发序列数提升（同容量峰值并发与零 OOM 容量网格）
  - 长上下文 OOM 比例下降（OOM 率随容量变化曲线）
"""


import csv
import json
import os

from kv_cache.allocator import BLOCK_SIZE, blocks_for
from kv_cache.simulator import simulate
from kv_cache.workload import make_workload, split_total_length
from kv_cache.svg_chart import write_comparison


def run(output_dir: str = ".", prompt_fraction: float | None = None):
    os.makedirs(output_dir, exist_ok=True)
    workload = make_workload(800, 42, prompt_fraction=prompt_fraction)

    # 整份负载的『总量对比』（朴素按最坏情况预留 vs 分页按实际）
    naive_total = sum(blocks_for(s["M"]) for s in workload)
    paged_total = sum(blocks_for(s["L"]) for s in workload)
    saving = (naive_total - paged_total) / naive_total * 100
    useful_tokens = sum(s["L"] for s in workload)
    naive_internal_waste = 1 - useful_tokens / (naive_total * BLOCK_SIZE)
    paged_internal_waste = 1 - useful_tokens / (paged_total * BLOCK_SIZE)

    # 容量扫描（压到需求附近，让两种方案的 OOM 差距显现）
    capacities = [500, 1000, 2000, 3000, 6000, 9000, 12000, 16000, 20000, 24000, 28000, 32000]

    rows = []
    per_mode = {"naive": [], "paged": [], "paged_dynamic": []}
    for C in capacities:
        for mode in per_mode:
            m = simulate(workload, C, mode)
            rows.append((C, mode, m["oom_rate"], m["avg_util"],
                         m["peak_concurrency"], m["accepted"], m["rejected"]))
            per_mode[mode].append(m)

    # 在扫描网格上达到零 OOM 的最小容量；不是精确边界。
    min_zero_oom = {}
    for mode in per_mode:
        mzo = None
        for C, m in zip(capacities, per_mode[mode]):
            if m["oom_rate"] == 0:
                mzo = C
                break
        min_zero_oom[mode] = mzo
    capacity_ratio = (min_zero_oom["naive"] / min_zero_oom["paged"]) if min_zero_oom["naive"] and min_zero_oom["paged"] else None

    # 写 CSV（报告表格数据直接来这）
    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["capacity_blocks", "mode", "oom_rate", "avg_util",
                    "peak_concurrency", "accepted", "rejected"])
        for r in rows:
            w.writerow(r)

    with open(os.path.join(output_dir, "fragmentation.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["mode", "reserved_token_slots", "useful_token_slots",
                    "unused_token_slots", "internal_waste_ratio"])
        for mode, reserved, waste_ratio in (("naive", naive_total * BLOCK_SIZE, naive_internal_waste),
                                            ("paged", paged_total * BLOCK_SIZE, paged_internal_waste)):
            w.writerow([mode, reserved, useful_tokens, reserved - useful_tokens, waste_ratio])

    long_context_path = run_long_context(output_dir, prompt_fraction=prompt_fraction)
    with open(os.path.join(output_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump({"main": {"requests": 800, "seed": 42, "arrival_mean": 0.15,
                            "total_length_range": [128, 1024], "max_context": 2048},
                   "long_context": {"requests_per_length": 100, "seed": 2026,
                                    "total_lengths": [512, 1024, 2048, 4096],
                                    "max_context": 4096},
                   "block_size": BLOCK_SIZE, "prompt_fraction": prompt_fraction,
                   "dynamic_growth_timing": "uniform generated tokens over request duration",
                   "dynamic_cpu_swap_blocks": 0}, f, ensure_ascii=False, indent=2)
    svg_path = os.path.join(output_dir, "comparison.svg")
    write_comparison(svg_path, rows, naive_total, paged_total)
    print("已生成矢量图：", svg_path)

    # 控制台摘要
    print("=" * 64)
    print("任务一 KV Cache 分页内存管理 —— 仿真结果摘要")
    print("=" * 64)
    print(f"工作负载：{len(workload)} 条请求（泊松高并发到达，实际长度 128~1024 token）")
    print(f"整份负载所需 KV 块：朴素(按2048预留)={naive_total}  分页(按实际)={paged_total}")
    print(f">>> 按最终长度计算的累计预留块需求减少 {saving:.1f}%（不是实测碎片率）")
    print(f">>> 内部未用 token 槽比例：连续预留 {naive_internal_waste:.1%}，分页末块 {paged_internal_waste:.1%}")
    if capacity_ratio:
        print(f">>> 扫描网格上的零 OOM 容量：朴素 {min_zero_oom['naive']} 块，"
              f"一次性分页 {min_zero_oom['paged']} 块，逐步扩页 {min_zero_oom['paged_dynamic']} 块")
    print("-" * 64)
    print(f"{'容量':>8} {'方案':>7} {'OOM率':>8} {'平均利用率':>10} {'峰值并发':>8}")
    for r in rows:
        print(f"{r[0]:>8} {r[1]:>7} {r[2]*100:>7.1f}% {r[3]*100:>9.1f}% {r[4]:>8}")

    # 画图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        naive_oom = [r[2] for r in rows if r[1] == "naive"]
        paged_oom = [r[2] for r in rows if r[1] == "paged"]
        dynamic_oom = [r[2] for r in rows if r[1] == "paged_dynamic"]
        naive_util = [r[3] for r in rows if r[1] == "naive"]
        paged_util = [r[3] for r in rows if r[1] == "paged"]
        dynamic_util = [r[3] for r in rows if r[1] == "paged_dynamic"]

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        axes[0].plot(capacities, [o * 100 for o in naive_oom], "o-", label="朴素连续分配(基线)")
        axes[0].plot(capacities, [o * 100 for o in paged_oom], "s-", label="分页分配(本设计)")
        axes[0].plot(capacities, [o * 100 for o in dynamic_oom], "^-", label="逐步扩页(无换出)")
        axes[0].set_title("不同显存容量下的 OOM 拒绝率")
        axes[0].set_xlabel("KV 缓存容量(块)"); axes[0].set_ylabel("OOM 率(%)")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(capacities, [u * 100 for u in naive_util], "o-", label="朴素连续分配(基线)")
        axes[1].plot(capacities, [u * 100 for u in paged_util], "s-", label="分页分配(本设计)")
        axes[1].plot(capacities, [u * 100 for u in dynamic_util], "^-", label="逐步扩页(无换出)")
        axes[1].set_title("不同容量下的平均 GPU 块占用率")
        axes[1].set_xlabel("KV 缓存容量(块)"); axes[1].set_ylabel("平均块占用率(%)")
        axes[1].legend(); axes[1].grid(alpha=0.3)

        axes[2].bar(["朴素(按2048预留)", "分页(按实际)"], [naive_total, paged_total],
                    color=["#c0392b", "#27ae60"])
        title3 = f"整份负载累计预留块需求\n(按实际长度减少 {saving:.0f}%"
        if capacity_ratio:
            title3 += f"，零 OOM 容量比 {capacity_ratio:.1f}:1)"
        axes[2].set_title(title3)
        axes[2].set_ylabel("KV 块总数")
        for i, v in enumerate([naive_total, paged_total]):
            axes[2].text(i, v, str(v), ha="center", va="bottom")

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "comparison.png"), dpi=130)
        print("\n已生成图片：", os.path.join(output_dir, "comparison.png"))
    except Exception as e:  # noqa
        print("\n（未生成图片，原因为：", e, "—— 不影响指标与 CSV）")

    return dict(saving=saving, ratio=capacity_ratio,
                naive_total=naive_total, paged_total=paged_total,
                naive_internal_waste=naive_internal_waste,
                paged_internal_waste=paged_internal_waste,
                long_context_path=long_context_path, svg_path=svg_path)


def run_long_context(output_dir: str = ".", requests_per_length: int = 100,
                     prompt_fraction: float | None = None) -> str:
    """Stratified synthetic stress test with equal requests per length."""
    lengths = (512, 1024, 2048, 4096)
    if requests_per_length <= 0:
        raise ValueError("requests_per_length must be positive")
    workload = make_workload(requests_per_length * len(lengths), seed=2026, arrival_mean=0.12,
                             L_range=(512, 4096), M=4096)
    for i, request in enumerate(workload):
        request["L"] = lengths[i % len(lengths)]
        if prompt_fraction is not None:
            prompt, generated = split_total_length(request["L"], prompt_fraction)
            request.update(prompt_tokens=prompt, generated_tokens=generated)
    path = os.path.join(output_dir, "long_context.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["capacity_blocks", "mode", "actual_tokens", "requests",
                         "oom_count", "oom_rate"])
        for capacity in (1000, 2000, 4000, 8000, 16000):
            for mode in ("naive", "paged", "paged_dynamic"):
                result = simulate(workload, capacity, mode)
                for length in lengths:
                    ids = {s["id"] for s in workload if s["L"] == length}
                    failed = len(ids & result["rejected_ids"])
                    writer.writerow([capacity, mode, length, len(ids), failed,
                                     failed / len(ids)])
    return path


if __name__ == "__main__":
    run(".")
