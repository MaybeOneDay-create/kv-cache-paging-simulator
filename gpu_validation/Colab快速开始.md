# Colab 免费 GPU 快速开始

本页记录 Colab 上的安装、smoke test、三轮引擎 A/B 与证据下载命令。2026-11-09 的 Tesla T4 主负载结果和原始文件已保存于 `review_outputs/gpu_formal/main_512_128/`；新会话仍需先检查实际 GPU 与环境。长上下文和压力测试计划见 `docs/云端GPU验证记录模板.md`。

## 1. 打开 GPU

访问 [Google Colab](https://colab.research.google.com/)，登录 Google 账号并新建笔记本。点“代码执行程序 → 更改运行时类型 → 硬件加速器 → T4 GPU（或其他可用 GPU）”，保存，点击右上角“连接”。运行第一个代码单元：

```python
!nvidia-smi
!python --version
!python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

保存完整输出。若 `nvidia-smi` 失败或显示无可用 GPU，先停止，不要把后续 CPU 执行当作 GPU 验证。[Colab GPU 官方说明](https://colab.research.google.com/notebooks/pro.ipynb)

## 2. 上传实验包

本地现成包：`review_outputs/gpu_validation_bundle.zip`。在 Colab 新代码单元运行：

```python
from google.colab import files
uploaded = files.upload()  # 选择 gpu_validation_bundle.zip
import zipfile
with zipfile.ZipFile("gpu_validation_bundle.zip") as archive:
    archive.extractall("/content/kv_task")
%cd /content/kv_task
```

包中含 GPU A/B 脚本和脱敏的 ShareGPT 数值长度轨迹，不含 673 MB 原始对话、模型权重或本地虚拟环境。

## 3. 安装并做单批 smoke test

先保存环境信息，再按 [vLLM 官方 GPU 安装说明](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)安装。2026-11-09 Colab T4 会话显示预装 PyTorch 2.14.1+cu130；使用独立虚拟环境，避免直接覆盖 Colab 的系统包。`uv` 按当前驱动自动选 PyTorch CUDA 后端；安装结束后记录版本。下面是单卡 T4 的起始配置；如果实际显存或驱动不兼容，先记录报错并调整，不要覆盖原始日志。

```python
!python -m pip -q install uv
!uv venv /content/kv_task/.venv --python python3
!uv pip install --python /content/kv_task/.venv/bin/python 'torch==2.14.0' 'vllm==0.33.0' 'transformers==5.20.2' --torch-backend=auto
!uv pip freeze --python /content/kv_task/.venv/bin/python > environment.txt
!CUDA_VISIBLE_DEVICES=0 /content/kv_task/.venv/bin/python gpu_validation/bench.py --backend hf_static --model Qwen/Qwen2.5-1.5B-Instruct --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --dtype float16 --input-tokens 512 --output-tokens 128 --max-model-len 4096 --batch-sizes 1 --gpu-index 0 --output hf_smoke.json
!CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn /content/kv_task/.venv/bin/python gpu_validation/bench.py --backend vllm --model Qwen/Qwen2.5-1.5B-Instruct --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --dtype float16 --input-tokens 512 --output-tokens 128 --max-model-len 4096 --batch-sizes 1 --gpu-index 0 --output vllm_smoke.json
!/content/kv_task/.venv/bin/python gpu_validation/compare.py --hf hf_smoke.json --vllm vllm_smoke.json --output smoke_comparison.csv
```

两种引擎由两个命令分别启动为新进程；若前一个进程异常退出、残留 GPU 占用或 Colab 重启，先检查 `nvidia-smi` 再启动下一种。JSON 应同时为 `status=complete`，同卡 `gpu_uuid` 与 prompt 哈希应一致。**smoke test 成功只证明流程可运行，不等于完成正式 A/B**。

以上版本与 2026-11-09 已成功的 T4 主负载一致。新会话若驱动、CUDA 后端或安装解析与原记录不同，先保存环境输出并重新做 smoke test；不能把不同版本的结果直接并入原来的三轮均值。

## 4. 正式 A/B

按 `docs/云端GPU验证记录模板.md` 的同卡、同模型、同 dtype、同长度、同批大小要求，各引擎至少三轮。`gpu_validation/run_ab.py` 自动为两种引擎分别启动全新进程，保存每轮原始 JSON、完整日志、`nvidia-smi` 快照、比较 CSV、环境版本和汇总表。示例：

```python
!CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn /content/kv_task/.venv/bin/python gpu_validation/run_ab.py --model Qwen/Qwen2.5-1.5B-Instruct --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --dtype float16 --input-tokens 512 --output-tokens 128 --max-model-len 4096 --batch-sizes 1 2 4 8 16 --batch-warmups 2 --repetitions 3 --gpu-index 0 --output-dir /content/kv_task/gpu_results/main_512_128
```

只在两种 smoke test 均通过后运行。运行结束应显示 `Saved 3 paired runs to ...`。随后按 512、1024、2048、4096 token 长度档与不同 KV 预算做压力测试。只在实测后写入 GPU 结论。Colab 免费会话可能中断，需及时下载每轮原始文件。

## 结果带回本地

正式实验完成后，将整个结果目录打包下载，保留六轮原始 JSON、日志、配置、`nvidia-smi` 快照和比较 CSV：

```python
import shutil
from google.colab import files

archive = shutil.make_archive(
    "/content/main_512_128_evidence", "zip",
    "/content/kv_task/gpu_results", "main_512_128"
)
files.download(archive)
```

smoke test 的单独文件也可按需下载：

```python
for name in ["hf_smoke.json", "vllm_smoke.json", "smoke_comparison.csv", "environment.txt"]:
    files.download(name)
```

若要我继续代为检查和完成正式扫描，请把 `nvidia-smi` 的输出及 smoke test JSON 带回当前对话，或保持已登录的 Colab 浏览器标签页打开。
