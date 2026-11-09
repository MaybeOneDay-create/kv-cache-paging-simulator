#!/bin/bash
# KV Cache 分页内存管理仿真器 —— macOS / Linux 一键启动
set -e
cd "$(dirname "$0")"
echo "============================================================"
echo "  KV Cache 分页内存管理仿真器 —— 一键启动"
echo "============================================================"
echo "运行对比实验 ..."
python3 run_experiment.py
echo
echo "运行完成：正式结果见 review_outputs/formal_50pct 文件夹"
