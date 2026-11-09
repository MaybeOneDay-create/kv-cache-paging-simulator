# 任务一：KV Cache 分页内存管理子系统（对标 vLLM PagedAttention）

> **正式 CPU 口径**：`L` 为 prompt 加生成 token 的总缓存长度，合成负载将 prompt 设为 `L` 的 50%。正式 CSV 和 SVG 位于 `review_outputs/formal_50pct/`；历史未拆分数据留在 `review_outputs/` 根目录，仅供追溯。CPU 数值不能充当真实 GPU 测量值。

运行 `python run_experiment.py` 会重现 50% 正式 CPU 结果；`python run_experiment.py --sensitivity` 会分别生成 25%、50%、75% 三组敏感性结果。每组的 `experiment_config.json` 记录种子、长度范围、块大小和比例。

> **课程**：2026 专业实习 —— A 类选题 **A1：面向大模型推理优化的操作系统级任务**
> **模块**：任务一（4 个任务中任选其一，本仓库完成“任务一”）
> **定位**：用 CPU 离散事件仿真器研究 KV Cache 分页机制，并在云端 GPU 上比较真实推理引擎。正式 CPU 合成实验采用 50% prompt；Colab T4 上已完成 512+128 token 的三轮引擎 A/B，长上下文与压力边界待测。

逐项验收状态见 [任务一验收清单](docs/任务一验收清单.md)。

---

## 1. 为什么做这个（问题背景）

大模型自回归生成时，每一层都要缓存历史 token 的 Key/Value 张量（KV Cache）。
传统做法是**每个请求一到来就按“最大上下文长度”连续预留一整段显存**：

- 实际生成长度往往远小于最大长度 → 大量**内部碎片**；
- 显存被碎片“虚占”，真正能并发的请求数被压低 → 高并发下频繁 **OOM 拒单**；
- 无法共享前缀（beam search / 并行采样 / 多轮对话前缀）。

本设计借鉴 **vLLM 的 PagedAttention**：把 KV Cache 像**操作系统虚拟内存**一样**分页、按需分配、用页表映射、支持写时复制与换出**，从而：

- 减少按最大长度预留造成的浪费；分页末块仍可能有少量未使用槽位；
- 等量显存下可并发更多序列 → 同等负载零 OOM 所需显存大幅下降；
- 支持前缀共享 → 多请求复用同一份前缀物理块。

---

## 2. OS 概念 ↔ KV Cache 概念映射（写报告 / 答辩直接用）

| 操作系统概念 | 本设计中的对应物 | 说明 |
|---|---|---|
| 物理页帧 (page frame) | 物理 KV 块 (block) | 显存中可独立分配的最小单位 |
| 虚拟页 (virtual page) | 逻辑 token 序列块 | 序列第 i 个 16-token 块 |
| 页表 (page table) | `PageTable` | 逻辑块号 → 物理块号 映射 |
| 按需调页 (demand paging) | `append_token` | 生成跨越块边界时才分配新块 |
| 写时复制 (COW) | `share_prefix` | beam / 并行采样共享前缀 |
| 页面置换 (page replacement) | `evict` | 显存不足时换出到 CPU 内存 |
| 块级位图与空闲帧栈 | `BitmapFramePool` | O(1) 取还固定大小页框，位图校验占用状态 |
| 内部碎片 | 朴素方案的“按最大值预留” | 本设计要消除的对象 |

> 块大小 `BLOCK_SIZE = 16`（类比页大小），页数 = `ceil(tokens / 16)`。

---

## 3. 角色分工与包结构

代码按设计文档的 4 个角色拆分为 `kv_cache` 包：

```
任务一_KV_Cache分页仿真器/
├── README.md                  # 本文件
├── requirements.txt           # CPU 阶段无第三方依赖；注明可选出图依赖
├── .gitignore                 # 忽略缓存与运行产物
├── start.bat                  # Windows 一键启动
├── start.sh                   # macOS / Linux 一键启动
├── run_experiment.py          # 实验入口
└── kv_cache/                  # 核心代码包
    ├── __init__.py            # 包说明 + 4 角色分工
    ├── allocator.py           # 角色 B（内存/系统实现）
    │   ├── PageTable                      # 页表
    │   ├── KVBlockAllocator               # 分页分配器（本设计）
    │   └── NaiveContiguousAllocator       # 朴素连续分配器（基线对照）
    ├── workload.py            # 角色 D（测试与评测）：泊松高并发负载建模
    ├── simulator.py           # 角色 C（运行时/IO）：离散事件仿真
    └── experiments.py         # 角色 D（测试与评测）：容量扫描 / 出图 / 出 CSV
```

| 角色 | 负责文件 | 内容 |
|---|---|---|
| A 架构设计 | （见设计文档 / 本 README 第 2 节映射） | OS 概念映射、接口契约 |
| B 内存/系统实现 | `allocator.py` | `PageTable`、`KVBlockAllocator`、`NaiveContiguousAllocator` |
| C 运行时/IO | `simulator.py` | 到达/离开离散事件引擎，统计 OOM 率、利用率、峰值并发 |
| D 测试与评测 | `workload.py` + `experiments.py` | 负载生成、容量扫描、可视化、CSV |

---

## 4. 环境要求

- Python **3.10+**（本轮本地验证：Python 3.11）
- CPU 仿真、CSV、SVG 和测试只需 Python 标准库；可选安装 `matplotlib>=3.7` 额外生成 PNG
- CPU 仿真阶段无需 GPU / CUDA / torch；完整任务仍需另做云端 GPU 验证

---

## 5. 一键启动（推荐）

### Windows
双击 `start.bat`，或命令行：
```bat
start.bat
```
脚本会直接运行实验，无需联网安装依赖。

### macOS / Linux
```bash
bash start.sh
```

### 手动运行
```bash
python run_experiment.py
```

默认运行后会在 `review_outputs/formal_50pct/` 中生成：
- `results.csv` —— 各容量下三种方案的 OOM 率 / 块占用率 / 峰值并发
- `comparison.svg` —— 无第三方绘图库也会生成的对比图
- `comparison.png` —— 安装 matplotlib 后附加生成的对比图
- `long_context.csv` —— 512 / 1024 / 2048 / 4096 token 分组的 OOM 压测结果
- `fragmentation.csv` —— 两种预留策略的内部未用 token 槽比例
- `experiment_config.json` —— 本轮实验的负载与时序参数

---

## 6. 实验结果（CPU 仿真，来自 800 条泊松高并发请求）

负载参数：800 条请求，指数分布到达（泊松流，`arrival_mean=0.15`），实际长度 128~1024 token，模型最大上下文 `M=2048`，块大小 16。

**总量对比**
- 整份负载所需 KV 块：朴素（按 2048 预留）= **102 400**；分页（按实际）= **29 545**
- **按最终长度计算的累计预留块需求减少约 71.1%**。这是预留量对比，不是实测碎片率或 GPU 显存节省率。
- **内部未用 token 槽比例**：连续预留约 71.5%，按实际长度分页的末块约 1.3%；计算式和原始分子、分母见 `fragmentation.csv`。

**零 OOM 所需最小容量（对应“单卡可承载并发序列数提升”）**
- 在当前扫描容量网格上，朴素方案需 **20 000** 块；一次性分页和逐步扩页均需 **6 000** 块。这些不是精确边界，也不是 GPU 实测值。

**容量扫描（节选）**

| 容量(块) | 方案 | OOM 率 | 平均块占用率 | 峰值并发 |
|---:|---|---:|---:|---:|
| 3000 | 朴素(基线) | 76.5% | 86.7% | 23 |
| 3000 | 一次性分页 | 21.0% | 82.2% | 93 |
| 3000 | 逐步扩页 | 8.6% | 76.8% | 118 |
| 6000 | 朴素(基线) | 54.8% | 85.0% | 46 |
| 6000 | 一次性分页 | 0.0% | 55.7% | 132 |
| 6000 | 逐步扩页 | 0.0% | 41.9% | 132 |
| 9000 | 朴素(基线) | 33.4% | 84.4% | 70 |
| 9000 | 一次性分页 | 0.0% | 37.1% | 132 |
| 9000 | 逐步扩页 | 0.0% | 27.9% | 132 |
| 16000 | 朴素(基线) | 1.0% | 71.4% | 125 |
| 16000 | 一次性分页 | 0.0% | 20.9% | 132 |
| 16000 | 逐步扩页 | 0.0% | 15.7% | 132 |
| 20000 | 朴素(基线) | 0.0% | 57.7% | 132 |
| 20000 | 一次性分页 | 0.0% | 16.7% | 132 |
| 20000 | 逐步扩页 | 0.0% | 12.6% | 132 |

> `avg_util` 是仿真期间已占 GPU 块数除以容量的时间加权平均值；数值较低表示占用较少，并不单独证明服务效率较高。逐步扩页在请求到达时装入 prompt，生成 token 在持续时间内均匀产生；失败请求在扩页 OOM 时释放。完整数据见 `review_outputs/formal_50pct/results.csv`。

**长上下文压测**：`review_outputs/formal_50pct/long_context.csv` 在每个长度档使用 100 条请求，共 400 条；每个容量、每种方案各输出一行失败数和失败率。逐步扩页的请求可能在已经运行后因扩页失败，因此某些长请求的完成率会低于一次性准入；解读结果时要同时考虑准入策略和已消耗的计算时间。此处的 CPU 侧“换出”仅保存块元数据，尚未代表真实 KV 张量搬运。

**回归与覆盖率**：在项目根目录执行 `python -m unittest discover -s tests -v` 和 `python tests/measure_coverage.py`。2026-11-09 在本地 Python 3.11 上，35 个单测通过；按标准库 `trace` 计算整个 `kv_cache` 包的可执行行覆盖为 477/629（75.8%）。覆盖率脚本会在低于 70% 时返回非零状态。

**ShareGPT 长度轨迹**：固定数据集 SHA-256 和 Qwen tokenizer revision 后，`gpu_validation/prepare_sharegpt_trace.py` 已生成 800 条数值请求（`review_outputs/sharegpt/trace.json`），`run_trace_experiment.py` 的三方案结果见 `review_outputs/sharegpt/cpu_results/`。真实对话提供 prompt/回复 token 长度；到达与持续时间是合成的。该轨迹的 prompt 占比为 69.7%，应与正式 50% prompt 合成主实验分别解读。复跑命令与来源哈希见 `docs/量化评测报告.md`。云端引擎 A/B 的原始数据见 `review_outputs/gpu_formal/main_512_128/`。

---

## 7. 对应评分指标（设计文档中的量化点）

| 评分点 | 本仓库如何体现 |
|---|---|
| 预留浪费下降 | 按实际长度比按最大长度少预留 71.1% 的块；内部未用 token 槽比例 71.5%→1.3% |
| 单卡并发序列数提升 | CPU 仿真峰值活跃序列 23→93（同 3000 块）；零 OOM 扫描容量 20000→6000。真实 GPU 同时活跃序列数待测 |
| 长上下文 OOM 比例下降 | CPU 各长度档结果见 `long_context.csv`；逐步扩页可能在运行中失败，不能保证每档都优于连续预留。真实 GPU 长上下文边界待测 |
| 新技术 / 场景覆盖 | PagedAttention 思想 + OS 虚拟内存类比；分配器提供 COW、前缀共享和模拟换出；云端引擎完成 512+128 token 三轮 A/B |

---

## 8. 二次扩展接口（已预留，便于加分）

`allocator.py` 中已实现的仿真接口：
- `KVBlockAllocator.share_prefix(...)` / `fork(...)` —— 引用计数与写时复制前缀共享
- `KVBlockAllocator.evict(...)` / `translate(...)` —— 模拟换出和访问时换入；仅保存块元数据，不保存真实 KV 张量

LRU 换出已在元数据层实现。已确认的 GPU 验证方式是使用真实推理引擎做 A/B，同时用本项目仿真器验证机制；本分配器不会直接管理真实 KV 张量。设计与评测细节见 `docs/系统设计文档.md` 和 `docs/量化评测报告.md`。

Colab T4 的环境检查、两引擎单批 smoke test 和 512 输入 + 128 输出的三轮配对 A/B 已完成。正式原始证据在 `review_outputs/gpu_formal/main_512_128/`，汇总与解释在 `docs/量化评测报告.md`；复跑方法见 `gpu_validation/Colab快速开始.md`。这组数据比较的是完整引擎，长上下文和容量压力仍待实测；已确认的后续实验配置见 `docs/GPU长上下文与压力测试计划.md`。

---

## 9. 快速理解代码路径（新人上手）

1. `workload.py::make_workload` —— 生成一份高并发压力负载。
2. `allocator.py::KVBlockAllocator` —— 看“分页怎么分配/释放/共享”。
3. `allocator.py::NaiveContiguousAllocator` —— 看“朴素方案为什么浪费”。
4. `simulator.py::simulate` —— 离散事件引擎怎么统计 OOM / 利用率。
5. `experiments.py::run` —— 容量扫描、出图、出 CSV，复现第 6 节结果。
