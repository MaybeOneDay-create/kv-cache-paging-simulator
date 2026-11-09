# 云端 GPU 验证记录模板（待实际运行）

本文件是待执行的实验记录规范，不包含 GPU 推理实测结果。已确认先尝试免费 GPU。2026-11-09 的 Colab `nvidia-smi` 截图已显示单张 Tesla T4、15,360 MiB 显存、驱动 580.82.07、空闲 0 MiB 使用量；这只证明运行时已分配 GPU，不证明模型或 A/B 已跑通。模型候选为固定 revision 的 Qwen2.5-1.5B-Instruct，仍须现场 smoke test。不能把 `review_outputs/formal_50pct` 中的 CPU 仿真数字填写为 GPU 测量值。

**免费环境（2026-11-09 核查）**：先尝试 Google Colab 免费 GPU；其资源类型和用量不保证。Kaggle 可作备选，但 P100 已于 2026-09-15 退役，不能照旧文档指定 P100；T4×2 实例需要只暴露其中一张卡。实际会话先运行 `nvidia-smi` 确认 GPU、空闲显存和驱动，再按 vLLM 官方安装指南选择兼容版本。[Colab FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html)、[Kaggle P100 退役公告](https://www.kaggle.com/product-announcements/735239)、[Kaggle Notebook 说明](https://www.kaggle.com/docs/notebooks)、[vLLM GPU 安装](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)

## 已准备的 A/B 入口

在选定 GPU 实例和模型后，先按所用实例的 CUDA 环境安装兼容的 PyTorch、Transformers 与 vLLM；vLLM 的安装方式应以对应版本的官方 GPU 安装文档为准。两组实验使用同一个模型路径、精度、输入/输出 token 数、最大上下文和批大小，**分别在干净进程中执行**：

### 实际执行顺序

1. 在免费云平台创建 CUDA 实例，先记录 `nvidia-smi`、Python、驱动、CUDA 和 GPU 可用显存；测试期间不要让其他进程占用选中的卡。单卡实例可直接运行；若机器有两张 T4，在 Linux 命令前加 `CUDA_VISIBLE_DEVICES=0`，并在脚本中指定 `--gpu-index 0`。脚本要求恰好一张 CUDA 可见 GPU，记录所选物理 GPU 的 UUID 与总显存；A/B 比较会核对 UUID。
2. 把本项目目录上传或克隆到实例，安装与该实例匹配的 PyTorch、Transformers、vLLM。将安装命令和 `pip freeze` 保存到实验目录。CPU 仿真无需第三方依赖；GPU 依赖按实例环境单独安装。
3. 选择两个引擎都支持、模型许可允许使用、上下文至少覆盖测试长度的同一个模型。记录模型仓库与不可变 commit hash；正式 A/B 在 `--revision` 填入相同 hash。若使用本地模型目录，先冻结目录内容，并保存权重清单及哈希。
4. 先用 `--batch-sizes 1` 跑 smoke test，检查两份 JSON 中 `status=complete`、`error=null`、`generated_tokens=output_tokens`、`prompt_ids_sha256` 相同。初始化失败会写出 `status=initialization_failed` 与原始错误；先保存环境和日志并修正，再做正式扫描。
5. 正式实验每轮分别启动两个新进程，运行下方命令；至少重复三轮，每轮使用不同输出文件名。每轮都调用 `compare.py`，保留原始 JSON、比较 CSV、控制台日志和设备监控快照。
6. 按长度档、批大小和显存预算扫描。长上下文档的 `input_tokens + output_tokens` 不得超过模型支持范围及 `max_model_len`。一次只改变一个自变量；每组都重复三轮。
7. 对任务一的四项指标分别保存证据：设备显存与有效 KV 容量口径、CPU 仿真的内外碎片数据、同卡可完成的并发请求数、长上下文下的失败率及 OOM 分类。引擎日志中的 KV 使用率、抢占或换出事件可以补充解释。当前 `bench.py` 输出批级吞吐、耗时、设备显存采样和错误；它不能单独证明 GPU 的 KV 碎片率，也不提供请求级延迟。首 token 延迟 P50/P99 是合并版任务二的指标，任务一可选测，不作为本任务的必达门槛。

```bash
python gpu_validation/bench.py --backend hf_static --model MODEL_ID --revision COMMIT_HASH --input-tokens 512 --output-tokens 128 --max-model-len 4096 --batch-sizes 1 2 4 8 16 --output hf_static.json
python gpu_validation/bench.py --backend vllm --model MODEL_ID --revision COMMIT_HASH --input-tokens 512 --output-tokens 128 --max-model-len 4096 --batch-sizes 1 2 4 8 16 --output vllm.json
python gpu_validation/compare.py --hf hf_static.json --vllm vllm.json --output gpu_ab_comparison.csv
```

两种 smoke test 通过后，可用 `gpu_validation/run_ab.py` 自动执行三轮配对扫描，逐轮保存命令、完整日志、JSON、`nvidia-smi` 快照和 CSV，并生成 `summary.csv`。它在初始化失败时停止，保留失败日志，不把失败轮次伪装为成功数据。Colab 可直接使用 `gpu_validation/Colab快速开始.md` 中的命令。

2026-11-09 首次 vLLM smoke test 在引擎子进程初始化时失败，日志根因为 `Cannot re-initialize CUDA in forked subprocess`。脚本在引擎创建前查询了 CUDA，而 vLLM Python API 默认使用 `fork`；现已在 vLLM 分支设置 `VLLM_WORKER_MULTIPROC_METHOD=spawn`，且入口有 `if __name__ == "__main__"` 保护。正式运行前仍须复跑 smoke test 并保存首次失败和修复后日志；不能将这次初始化失败记为容量 OOM。[vLLM 多进程排障](https://docs.vllm.ai/en/latest/usage/troubleshooting/)、[vLLM 环境变量](https://docs.vllm.ai/en/latest/configuration/env_vars/)

三轮数据分别命名为 `hf_1.json` / `vllm_1.json` 等后，运行：

```bash
python gpu_validation/summarize_runs.py --pairs hf_1.json vllm_1.json hf_2.json vllm_2.json hf_3.json vllm_3.json --output gpu_ab_summary.csv
```

汇总表分别给出成功、失败、未运行轮数，以及成功轮次吞吐和设备显存的均值、标准差。失败轮次不会被当作零吞吐并入均值；报告必须同时展示失败轮数。三轮配置或引擎版本不一致会被拒绝。初始化失败的记录也会被拒绝，需单独列入环境或模型加载失败清单。

脚本使用同一 tokenizer 生成固定 token ID 输入，先预热并校验实际输出 token 数，再记录批量完成数、生成 token/s、耗时、`nvidia-smi` 采样到的设备显存峰值和错误。正式版在每个批大小的计时前额外预热两次，避免把首次 shape 的 Triton JIT 编译混入吞吐；预热失败会作为该批次失败记录。比较脚本拒绝模型、revision、精度、长度、预热次数、prompt token ID 哈希或 GPU 型号不一致的结果，并区分失败批次与尚未运行的批次。`hf_static` 为每批显式构造 Transformers `StaticCache(config=model.config, max_cache_len=max_model_len)` 并交给 `generate`；`vllm` 使用 16-token block。2026-11-09 的 Colab T4 单批 smoke test 两后端均成功，但正式扫描尚未运行。[Transformers 缓存参数](https://huggingface.co/docs/transformers/main_classes/text_generation)、[vLLM 离线推理示例](https://docs.vllm.ai/en/latest/examples/basic/offline_inference/)、[vLLM token ID 输入格式](https://docs.vllm.ai/en/stable/api/vllm/inputs/)

## 固定配置

每组 A/B 应记录：实验日期、云平台与实例、GPU 型号和显存、驱动与 CUDA、Python 与 PyTorch、推理引擎及版本、模型仓库和精确 revision、参数量、权重与 KV dtype、最大上下文、KV block 大小、KV 预算、prefix caching 开关、swap/offload 配置、并发上限、随机种子。保存启动命令、完整服务日志与 `nvidia-smi` 快照。

## 负载

1. 主负载：固定一份 prompt 与输出长度的请求轨迹，记录每条请求的到达时间、输入 token 数、期望输出 token 数及实际输出数。先做 warmup，再至少重复三轮。
2. 长上下文：按 512、1024、2048、4096 token 等长度分组。长度必须按模型 tokenizer 的 token 数确定，且不超过模型支持范围。
3. 共享前缀：相同前缀、不同后缀的请求组；单独切换 prefix caching，并记录前缀命中指标。避免把前缀缓存收益混入基础分页对比。
4. 压力扫描：保持模型、精度、请求轨迹不变，扫描 KV 预算和并发上限，找到失败边界。达到预算边界时需区分请求拒绝、引擎抢占重算、CPU 换出与进程 CUDA OOM。

## 逐轮记录

| run_id | 配置哈希 | 实际完成/总请求 | 吞吐 token/s | GPU 峰值显存 | 有效 KV 容量或占用 | 最大成功并发 | 抢占/换出次数 | OOM 分类 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 待运行 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

保存原始请求级 JSON/CSV、监控样本和启动日志；图表应由原始数据重新生成。汇总数字须注明多轮重复的均值和波动，不只展示表现最好的一轮。

## 解释边界

本项目已确定采用真实推理引擎 A/B 加 CPU 仿真器验证机制。不同引擎的吞吐和延迟差异包含调度器、算子内核、编译与缓存策略差异，不能全部归因为 KV 分页。本项目分配器不控制真实 KV 张量，报告须把仿真机制结论和真实引擎性能结论分别陈述。

`hf_static` 按 `max_cache_len` 预留静态缓存；vLLM 根据 `gpu_memory_utilization` 估算或限制自己的 KV 块池。二者不是同一个可精确对齐的 KV 字节预算，因此正式报告应同时列出各自配置、设备显存绝对值、成功批量和失败边界，不得把设备显存差值直接称为“分页节省的显存”。[Transformers StaticCache 参数](https://huggingface.co/docs/transformers/main_classes/text_generation)、[vLLM 缓存参数](https://docs.vllm.ai/en/latest/configuration/engine_args/)

当前 vLLM V1 在 KV 块不足时可抢占请求并重算；不能仅凭“请求最终完成”判断没有容量压力。出现抢占日志或指标时应连同吞吐和耗时报告，而不能把它写成 CPU KV 张量换出。本项目的 LRU 换出是 CPU 元数据仿真机制，与此行为分开解释。[vLLM 预抢占与重算说明](https://docs.vllm.ai/en/latest/design/arch_overview/)、[vLLM 生产指标](https://docs.vllm.ai/en/latest/usage/metrics/)
