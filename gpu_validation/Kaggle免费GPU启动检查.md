# Kaggle 免费 GPU 启动检查

Colab 主负载三轮 A/B 已完成并保存在 `review_outputs/gpu_formal/main_512_128/`。Colab 免费 GPU 配额用尽后，可在 Kaggle 尝试后续长上下文和压力实验。新平台先核查实际硬件、驱动、Python 与联网状态；不要把不同会话的吞吐合并成同卡重复实验。

1. 登录 [Kaggle Notebooks](https://www.kaggle.com/code)，新建 Notebook。
2. 在右侧 **Settings → Session options** 选择 **Accelerator → GPU T4×2**，确认 **Internet** 已开启。实际可用资源受账号与平台配额影响。用户首次截图中 GPU 和 Internet 均为灰色，页面提示 **Get phone verified**；此时需由账号持有人自行决定是否在 Kaggle 完成手机验证，验证后刷新页面再检查选项。验证码不要写入项目或发送给他人。若仍不可用，记录界面提示即可。
3. 在第一个代码单元运行：

```python
!nvidia-smi
!python --version
!python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
```

若有两张 T4，后续推理命令加 `CUDA_VISIBLE_DEVICES=0`，只使用第一张；`bench.py` 要求 Python 进程恰好看到一张 CUDA GPU，同时通过 `nvidia-smi` 记录物理 GPU UUID。GPU 型号与可用显存以实际输出为准。

4. 在右侧 **Input → Add Input → Upload** 上传本项目的 `review_outputs/gpu_validation_bundle.zip`。上传后先查找位置：

```python
from pathlib import Path
matches = list(Path('/kaggle/input').rglob('gpu_validation_bundle.zip'))
print(matches)
```

请先保存上述设备输出与上传文件路径。安装包版本、模型下载和压力扫描命令应依据该会话的实际环境确定；若 Internet 或 GPU 不可用，先记录错误，不把 CPU 运行当作 GPU 验证。

Kaggle 官方文档说明 Notebook 的 Session options 可配置 Internet 和 GPU，上传文件可作为 Input；2026-09 的平台公告表示 P100 已退役，T4×2 是保留的选项：[Notebook 文档](https://www.kaggle.com/docs/notebooks)、[GPU 变更公告](https://www.kaggle.com/product-announcements/735239)。
