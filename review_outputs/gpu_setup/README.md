# Colab GPU 环境证据（2026-11-09）

用户在当前对话提供的 Colab 截图已复制到本目录用于复核：

| 文件 | 可见证据 | SHA-256 |
|---|---|---|
| `colab_t4_nvidia_smi.png` | 单张 Tesla T4；显存 15,360 MiB；驱动 580.82.07；`nvidia-smi` 报告 CUDA 13.0；采样时 GPU 显存使用 0 MiB | `5e7a3a66d9d4746b76301809b77cf7019110421ce96517cebdb2ae8fb8f9d74c` |
| `colab_torch_cuda.png` | 实验包上传、解压成功；PyTorch 2.14.1+cu130；`torch.version.cuda=13.0`；`torch.cuda.is_available()` 为 True | `6c6da8d149981b12a8364106e747802f565b6e7de872e001a5fd8a0cf0b53858` |
| `colab_vllm_install_tail.png` | 独立虚拟环境安装结束，列表含 vLLM 0.33.0、Transformers 5.20.2 与 PyTorch 2.14.0+cu132 | `b26395267e5c0024143e3ba920dfcd9bab317dcb32f8adcca04729d2491e66a6` |
| `colab_venv_cuda_smoke.png` | 独立环境创建 CUDA 张量成功，导入 vLLM 与 Transformers 成功；输出 `OK 2.14.0+cu132 13.2 0.33.0 5.20.2` | `aaf5b97970cdba0af7e9a77252ea302e4e943f92f2472413a1f52a25b58fdb72` |
| `hf_static_smoke.png` | Transformers 静态缓存单批推理成功：`status=complete`，生成 128 token，耗时约 3.69 s，设备显存峰值采样 3,565 MiB | `d0f44468786ea48edf3ba699e325a0ffcce17def983176f2d5724bc9d1a0ca57` |

这些截图验证了 GPU 运行时、文件传输、CUDA 张量创建及一次 Transformers 单批生成。正式多轮 A/B、显存压力和 OOM 边界仍未实测；正式实验须保存完整原始 JSON、环境版本、命令、日志及比较 CSV。

原始文件已从 Colab 下载到 `smoke_evidence/`，原包 `smoke_evidence.zip` 的 SHA-256 为 `dba53798c1541766fd6de4041b3facc23f7dc3c3efc5d0128f670a28ffedcbcf`。本地 `compare.py` 复核返回退出码 0；两份成功 JSON 均记录 GPU UUID `GPU-1d320edd-f5e3-da67-7540-e7d84812a199`、相同 prompt 哈希 `dcfabcee843e67d741efaf2e4e9ac4c9f42cef4d6eba0b072c96ade6deca558d` 和固定模型 revision。vLLM 成功日志第 161–163 行报告可用 KV 缓存 9.07 GiB、339,648 token 容量及针对 4096-token 请求的理论并发 82.92 倍；这是引擎初始化时的估计，不是实测最大同时活跃请求数。
