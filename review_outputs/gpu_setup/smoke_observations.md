# Colab 单批 smoke test 观察记录

以下为 2026-11-09 当前对话中用户提供的 Colab 输出。原始 JSON 与日志仍需从 Colab 带回，故此记录仅作为过程证据。

| 后端 | 模型 / GPU | 状态 | 生成 token | 耗时 | 输出吞吐 | 设备显存峰值采样 |
|---|---|---|---:|---:|---:|---:|
| Transformers StaticCache | Qwen2.5-1.5B-Instruct / T4 | `complete` | 128 | 3.694701084999906 s | 34.644209925308004 token/s | 3,565 MiB |
| vLLM 0.33.0 | 同模型 / T4 | `complete` | 128 | 2.0852984509999715 s | 61.38210093553738 token/s | 13,165 MiB |

`compare.py` 在云端对两份原始 JSON 返回退出码 0，确认了配置字段与同卡 UUID 一致。首次 vLLM 初始化失败的真正原因是 CUDA 已在父进程初始化后又被 `fork` 到子进程；将 `VLLM_WORKER_MULTIPROC_METHOD=spawn` 后单批成功。失败原始日志和 JSON 在 Colab 分别保存为 `vllm_smoke_fork_failure.log`、`vllm_smoke_fork_failure.json`，尚未带回本地。

vLLM 单批日志出现 Triton kernel JIT 编译提示，故上述单批吞吐不能作正式 A/B 性能结论。两后端的设备显存值也包含不同的引擎预留与运行机制，不能把差值称为分页节省的 KV 显存。
