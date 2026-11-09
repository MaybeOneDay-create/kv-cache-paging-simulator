# -*- coding: utf-8 -*-
"""
workload.py —— 角色 D（测试与评测）：工作负载建模
================================================
生成一份高并发压力的工作负载（泊松到达流），贴近真实 LLM 服务 trace。
  - arrival_mean 越小 -> 到达越密 -> 并发越高（制造显存压力）
  - M 是模型最大上下文：朴素方案按此『最坏情况』预留，造成大量内部碎片
"""


import random


def split_total_length(total_tokens: int, prompt_fraction: float) -> tuple[int, int]:
    if total_tokens < 2 or not 0 < prompt_fraction < 1:
        raise ValueError("invalid total length or prompt fraction")
    prompt = min(total_tokens - 1, max(1, round(total_tokens * prompt_fraction)))
    return prompt, total_tokens - prompt


def make_workload(n_seqs: int = 800, seed: int = 42, arrival_mean: float = 0.15,
                 dur_range=(8.0, 25.0), L_range=(128, 1024), M: int = 2048,
                 prompt_fraction: float | None = None):
    if prompt_fraction is not None and not 0 < prompt_fraction < 1:
        raise ValueError("prompt_fraction must be between zero and one")
    rng = random.Random(seed)
    workload = []
    t = 0.0
    for i in range(n_seqs):
        t += rng.expovariate(1.0 / arrival_mean)   # 指数分布到达间隔（泊松流）
        duration = rng.uniform(*dur_range)          # 该请求占用 KV 的时长
        actual = rng.randint(*L_range)              # prompt + 生成的总 KV 长度
        request = dict(id=i, ta=t, dur=duration, L=actual, M=M)
        if prompt_fraction is not None:
            prompt, generated = split_total_length(actual, prompt_fraction)
            request.update(prompt_tokens=prompt, generated_tokens=generated)
        workload.append(request)
    return workload
