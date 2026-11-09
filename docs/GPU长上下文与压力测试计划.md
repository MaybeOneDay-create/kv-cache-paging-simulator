# 任务一：GPU 长上下文与容量压力测试计划

本计划只用于后续真实 GPU 运行，当前**没有**长上下文 GPU 结果。用户已确认 `L = prompt token 数 + 生成 token 数`；GPU 长上下文固定生成 128 token，增加 prompt 长度。此选择与正式 CPU 合成实验的 50% prompt 比例不同，报告中分别解释。

## 固定条件与长度档

沿用已成功的 Qwen/Qwen2.5-1.5B-Instruct revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`、float16、Transformers StaticCache / vLLM 0.33.0、16-token 分页块、`max_model_len=4096`。每组两侧使用相同的 token ID prompt、同一块 GPU、相同批量和两次批级预热。每组在新进程中配对运行三轮，保存原始 JSON、完整日志、命令和 `nvidia-smi` 快照。

| 总 KV 长度 L | prompt token | 生成 token | 目的 |
|---:|---:|---:|---|
| 512 | 384 | 128 | 短长度参考 |
| 1024 | 896 | 128 | 中等长度 |
| 2048 | 1920 | 128 | 长度扩展 |
| 4096 | 3968 | 128 | 当前模型配置下的上限档 |

主负载已有 512 输入 + 128 输出（`L=640`）的三轮数据，不能把它冒充上表的 `L=512` 或 `L=1024`。跨新会话先重新运行 `nvidia-smi`、版本检查和单批 smoke test；确认输出仍是 CUDA 设备、两侧 `status=complete` 后再进入正式扫描。新会话 GPU UUID 可能不同，每个长度档内部仍须同一 UUID；跨档设备不同要在汇总中标出，不能写成同一块卡的连续扫描。

## 顺序

1. **先测 L=4096**：批量 `1 2 4 8 16`，每侧三轮。它优先补齐最长档。若初始化或批量失败，保留原始错误，不把环境错误算作容量 OOM。
2. **依次测 L=2048、1024、512**：各用相同批量与重复次数。每完成一个长度档立即将整个结果目录打包带回本地，避免免费会话中断丢失数据。
3. **扩展 L=4096 的压力批量**：从 `16 32 64` 开始，单独记录失败类型、完成数、耗时与抢占日志。若仍全部成功，再审视 GPU 剩余配额和日志后决定是否扩大到更高批量；不能把当前最大成功批量当作精确容量边界。
4. **KV 预算敏感性**：在同一 GPU 和固定负载下，分别试 `gpu_memory_utilization=0.6、0.75、0.9` 的 vLLM 新进程；每档至少三轮。该参数是引擎预算口径，不是两引擎精确相同的 KV 字节预算。Transformers StaticCache 的结果单独列出，不将预算变化解释为分页单因素作用。

`bench.py` 在第一个失败批量后停止，因此汇总必须区分“成功”“失败”“未运行”。压力扫描使用 `run_ab.py --continue-on-failure`：即使某侧初始化或进程失败，仍运行余下配对，保存 `run_status.json` 与逐轮逐批的 `pressure_summary.csv`。进程退出而没有可读 JSON 时标为 `process_exit`，不能推断为 OOM；初始化错误也单独标记。若全部初始化成功，仍生成通常的 `summary.csv`。每次扫描须使用空目录，防止旧证据混入。vLLM 可以通过排队或抢占完成超过瞬时容量的提交批量；仅凭批量完成数不能称为实际同时活跃序列数。显存指标是设备已用峰值，采样区间包含预热；它不能单独证明 KV 内部利用率或碎片率。

## Colab 单组命令模板

恢复 GPU、重新上传并解压 `review_outputs/gpu_validation_bundle.zip`、安装与 smoke test 成功后，先运行最长档：

```python
!CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn /content/kv_task/.venv/bin/python gpu_validation/run_ab.py --model Qwen/Qwen2.5-1.5B-Instruct --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --dtype float16 --input-tokens 3968 --output-tokens 128 --max-model-len 4096 --batch-sizes 1 2 4 8 16 --batch-warmups 2 --repetitions 3 --gpu-index 0 --continue-on-failure --output-dir /content/kv_task/gpu_results/long_L4096
```

结束后在新代码单元打包下载：

```python
import shutil
from google.colab import files

archive = shutil.make_archive(
    "/content/long_L4096_evidence", "zip",
    "/content/kv_task/gpu_results", "long_L4096"
)
files.download(archive)
```

其他长度档只替换 `--input-tokens` 和 `--output-dir`：`1920` / `long_L2048`、`896` / `long_L1024`、`384` / `long_L512`。容量压力档改用 `--batch-sizes 16 32 64`，输出到新的 `pressure_L4096` 目录；预算敏感性每档也用不同输出目录。正式命令、原始文件和结果分析在实际运行后再补入量化报告。免费 Colab GPU 配额当前已满；用户不方便完成 Kaggle 手机验证，故不以 Kaggle 作为当前执行路径。
