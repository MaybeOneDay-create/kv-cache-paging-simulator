@echo off
cd /d "%~dp0"
chcp 65001 >nul
echo ============================================================
echo   KV Cache 分页内存管理仿真器 —— Windows 一键启动
echo ============================================================
echo 运行对比实验 ...
python run_experiment.py
if errorlevel 1 (
    echo 实验运行失败，请确认本机已安装 Python 3.10+ 并加入 PATH。
    pause
    exit /b 1
)
echo.
echo 运行完成：正式结果见 review_outputs\formal_50pct 文件夹
pause
