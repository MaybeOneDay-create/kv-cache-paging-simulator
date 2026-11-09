"""任务一：KV Cache 分页内存管理子系统（对标 vLLM PagedAttention）。

包结构（对应设计文档 4 人分工）：
  allocator.py   —— 角色 B（内存/系统实现）：KVBlockAllocator / PageTable / NaiveContiguousAllocator
  workload.py     —— 角色 D（测试与评测）：工作负载建模
  simulator.py    —— 角色 C（运行时/IO 实现）：离散事件仿真
  experiments.py  —— 角色 D（测试与评测）：对比实验 / 可视化 / CSV
"""
