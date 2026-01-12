@echo off
chcp 65001 >nul
title VideoMind - 环境检查
color 0C

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║      VideoMind - 环境检查工具              ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

REM 获取脚本所在目录
cd /d "%~dp0"

set ERROR_COUNT=0

echo [检查项 1] Python环境
echo ──────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [✗] Python未安装或不在PATH中
    set /a ERROR_COUNT+=1
) else (
    python --version
    echo [✓] Python已安装
)
echo.

echo [检查项 2] pip工具
echo ──────────────────────────────────────────────────────────
pip --version >nul 2>&1
if errorlevel 1 (
    echo [✗] pip未安装或不可用
    set /a ERROR_COUNT+=1
) else (
    pip --version
    echo [✓] pip可用
)
echo.

echo [检查项 3] FFmpeg
echo ──────────────────────────────────────────────────────────
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [✗] FFmpeg未安装或不在PATH中（音频处理需要）
    echo     下载地址: https://ffmpeg.org/download.html
    set /a ERROR_COUNT+=1
) else (
    for /f "tokens=3" %%i in ('ffmpeg -version ^| findstr "version"') do set FFMPEG_VER=%%i
    echo [✓] FFmpeg已安装: %FFMPEG_VER%
)
echo.

echo [检查项 4] 项目文件
echo ──────────────────────────────────────────────────────────
if not exist "cli\main.py" (
    echo [✗] 未找到cli\main.py（关键文件缺失）
    set /a ERROR_COUNT+=1
) else (
    echo [✓] 项目文件完整
)
echo.

echo [检查项 5] videomind命令
echo ──────────────────────────────────────────────────────────
where videomind >nul 2>&1
if errorlevel 1 (
    echo [!] videomind命令未安装（可使用python -m cli.main运行）
) else (
    echo [✓] videomind命令已安装
)
echo.

echo [检查项 6] 配置文件
echo ──────────────────────────────────────────────────────────
if exist "config.yaml" (
    echo [✓] 配置文件存在
) else (
    if exist "config.yaml.example" (
        echo [!] 配置文件不存在，但有示例文件
        echo     建议运行: copy config.yaml.example config.yaml
    ) else (
        echo [✗] 配置文件缺失
        set /a ERROR_COUNT+=1
    )
)
echo.

echo [检查项 7] API密钥配置
echo ──────────────────────────────────────────────────────────
if exist "config.yaml" (
    findstr /i "openai_api_key" config.yaml | findstr /v "^#" | findstr /v "^$" >nul 2>&1
    if errorlevel 1 (
        echo [!] 未检测到API密钥配置
        echo     请在config.yaml中配置或设置环境变量OPENAI_API_KEY
    ) else (
        echo [✓] 检测到API密钥配置（请确认密钥有效）
    )
) else (
    echo [!] 无法检查（配置文件不存在）
)
echo.

echo ═══════════════════════════════════════════════════════════
echo   检查完成
echo ═══════════════════════════════════════════════════════════
echo.

if %ERROR_COUNT% EQU 0 (
    echo [✓] 所有关键检查项通过，可以正常使用！
) else (
    echo [✗] 发现 %ERROR_COUNT% 个问题，请先解决这些问题
)

echo.
echo 提示：
echo   - 如果videomind命令未安装，运行 安装.bat
echo   - 如果缺少依赖，运行: pip install -r requirements.txt
echo   - 如果FFmpeg未安装，请从官网下载并添加到PATH
echo.

pause
