@echo off
chcp 65001 >nul
title VideoMind - 自动化视频内容处理系统
color 0A

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║      VideoMind - 自动化视频内容处理系统       ║
echo ║         从视频链接到结构化笔记                ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

REM 获取脚本所在目录
cd /d "%~dp0"

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.8或更高版本
    echo.
    pause
    exit /b 1
)

echo [信息] 正在启动VideoMind...
echo.

REM 尝试直接运行videomind命令（如果已安装）
where videomind >nul 2>&1
if not errorlevel 1 (
    echo [信息] 使用已安装的videomind命令...
    echo.
    videomind
    goto :end
)

REM 如果videomind未安装，使用Python模块方式运行
echo [信息] 使用Python模块方式运行...
echo.

REM 检查是否在正确的目录
if not exist "cli\main.py" (
    echo [错误] 未找到cli\main.py文件，请确保在videomind目录下运行此脚本
    echo.
    pause
    exit /b 1
)

python -m cli.main

:end
if errorlevel 1 (
    echo.
    echo [错误] 启动失败，请检查：
    echo   1. Python是否已正确安装
    echo   2. 是否已安装依赖：pip install -r requirements.txt
    echo   3. 是否配置了API密钥
    echo.
    pause
    exit /b 1
)

pause
