@echo off
chcp 65001 >nul
title VideoMind - 安装脚本
color 0B

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║      VideoMind - 安装脚本                   ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

REM 获取脚本所在目录
cd /d "%~dp0"

REM 检查Python是否安装
echo [步骤 1/4] 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.8或更高版本
    echo 下载地址: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

python --version
echo [✓] Python环境检查通过
echo.

REM 检查pip是否可用
echo [步骤 2/4] 检查pip...
pip --version >nul 2>&1
if errorlevel 1 (
    echo [错误] pip未安装或不可用
    echo 请先安装pip: python -m ensurepip --upgrade
    echo.
    pause
    exit /b 1
)

pip --version
echo [✓] pip检查通过
echo.

REM 升级pip
echo [步骤 3/4] 升级pip...
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [警告] pip升级失败，继续安装...
)
echo [✓] pip升级完成
echo.

REM 安装依赖
echo [步骤 4/4] 安装项目依赖...
echo.

if exist "requirements.txt" (
    echo 正在安装依赖包（可能需要几分钟）...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 依赖安装失败
        echo 请检查网络连接或使用国内镜像源：
        echo pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
        echo.
        pause
        exit /b 1
    )
    echo [✓] 依赖安装完成
) else (
    echo [警告] 未找到requirements.txt文件
)

echo.

REM 安装项目（开发模式）
echo [步骤 5/5] 安装VideoMind项目...
pip install -e .
if errorlevel 1 (
    echo [错误] 项目安装失败
    echo.
    pause
    exit /b 1
)

echo [✓] 项目安装完成
echo.

REM 验证安装
echo [验证] 检查安装结果...
where videomind >nul 2>&1
if errorlevel 1 (
    echo [警告] videomind命令未找到，但可以直接运行：python -m cli.main
) else (
    echo [✓] videomind命令已成功安装
)

echo.
echo ═══════════════════════════════════════════════════════════
echo   安装完成！
echo ═══════════════════════════════════════════════════════════
echo.
echo 现在您可以使用以下方式启动：
echo.
echo   方式1: 双击 启动.bat
echo   方式2: 在命令行输入 videomind
echo   方式3: python -m cli.main
echo.
echo 注意：首次使用前请配置API密钥（在config.yaml中）
echo.
pause
