# -*- coding: utf-8 -*-
"""
simulator.py —— 角色 C（运行时/IO 实现）：离散事件仿真
====================================================
给定显存容量 C，按时间顺序处理『到达 / 离开』事件，统计：
  - oom_rate         OOM 拒绝率（长上下文 OOM 比例）
  - avg_util         平均显存利用率
  - peak_concurrency 峰值并发序列数（单卡并发序列数）
"""


from collections import defaultdict

from kv_cache.allocator import AllocError, BLOCK_SIZE, KVBlockAllocator, NaiveContiguousAllocator


def simulate(workload, capacity, mode):
    if mode == "paged_dynamic":
        return simulate_dynamic(workload, capacity)
    if mode not in ("naive", "paged"):
        raise ValueError("unknown mode")
    if mode == "naive":
        alloc = NaiveContiguousAllocator(capacity)
        reserve = lambda s: alloc.reserve(s["id"], s["M"])
        release = lambda s: alloc.release(s["id"])
        held = lambda s: s["id"] in alloc._held
    else:  # paged
        alloc = KVBlockAllocator(capacity)
        reserve = lambda s: alloc.allocate_on_demand(s["id"], s["L"])
        release = lambda s: alloc.release(s["id"])
        held = lambda s: s["id"] in alloc.page_tables

    # 事件：(时间, 类型, 序列)
    events = []
    for s in workload:
        events.append((s["ta"], "arr", s))
        events.append((s["ta"] + s["dur"], "dep", s))
    events.sort(key=lambda e: e[0])

    rejected = 0
    rejected_ids = set()
    accepted = 0
    occupied_block_time = 0.0
    previous_time = events[0][0] if events else 0.0
    active = defaultdict(int)
    peak = 0

    for _time, typ, s in events:
        occupied_block_time += (capacity - alloc.free) * (_time - previous_time)
        previous_time = _time
        if typ == "arr":
            if reserve(s):
                accepted += 1
                active[s["id"]] = 1
            else:
                rejected += 1  # OOM，请求被拒
                rejected_ids.add(s["id"])
        else:  # dep
            if held(s):
                release(s)
                active[s["id"]] = 0
        peak = max(peak, sum(active.values()))

    return dict(
        oom_rate=rejected / len(workload) if workload else 0.0,
        avg_util=(occupied_block_time / (capacity * (events[-1][0] - events[0][0]))
                  if len(events) > 1 and events[-1][0] > events[0][0] else 0.0),
        peak_concurrency=peak,
        accepted=accepted,
        rejected=rejected,
        rejected_ids=rejected_ids,
    )


def simulate_dynamic(workload, capacity):
    """Allocate as a request generates tokens, with no CPU swap budget.

    L is generated uniformly over dur. This timing model is synthetic and
    separate from real GPU measurements.
    """
    alloc = KVBlockAllocator(capacity, cpu_capacity_blocks=0)
    events = []
    for s in workload:
        if s["L"] < 0 or s["dur"] <= 0:
            raise ValueError("invalid request length or duration")
        prompt = s.get("prompt_tokens", 0)
        generated = s.get("generated_tokens", s["L"])
        if prompt < 0 or generated < 0 or prompt + generated != s["L"]:
            raise ValueError("prompt + generated must equal total L")
        events.append((s["ta"], 0, s["id"], "arr", 0))
        completed = 0
        while completed < generated:
            count = min(generated - completed,
                        BLOCK_SIZE - ((prompt + completed) % BLOCK_SIZE))
            time = s["ta"] + s["dur"] * (completed + 1) / max(1, generated)
            events.append((time, 1, s["id"], "grow", count))
            completed += count
        events.append((s["ta"] + s["dur"], 2, s["id"], "dep", 0))
    events.sort()
    active = set()
    running = set()
    failed = set()
    requests = {s["id"]: s for s in workload}
    occupied_block_time = 0.0
    previous_time = events[0][0] if events else 0.0
    peak = 0
    for time, _, seq_id, typ, count in events:
        occupied_block_time += (capacity - alloc.free) * (time - previous_time)
        previous_time = time
        if typ == "arr":
            alloc.new_sequence(seq_id)
            try:
                for _ in range(requests[seq_id].get("prompt_tokens", 0)):
                    alloc.append_token(seq_id)
                active.add(seq_id)
                if requests[seq_id].get("prompt_tokens", 0):
                    running.add(seq_id)
                    peak = max(peak, len(running))
            except AllocError:
                failed.add(seq_id)
                alloc.release(seq_id)
        elif typ == "grow" and seq_id in active:
            try:
                for _ in range(count):
                    alloc.append_token(seq_id)
                running.add(seq_id)
                peak = max(peak, len(running))
            except AllocError:
                failed.add(seq_id)
                alloc.release(seq_id)
                active.remove(seq_id)
                running.discard(seq_id)
        elif typ == "dep" and seq_id in active:
            alloc.release(seq_id)
            active.remove(seq_id)
            running.discard(seq_id)
    duration = events[-1][0] - events[0][0] if events else 0
    return dict(oom_rate=len(failed) / len(workload) if workload else 0.0,
                avg_util=occupied_block_time / (capacity * duration) if duration else 0.0,
                peak_concurrency=peak, accepted=len(workload) - len(failed),
                rejected=len(failed), rejected_ids=failed)
