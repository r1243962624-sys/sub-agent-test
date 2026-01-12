"""
VideoMind 主命令行界面
使用 Typer 和 Rich 创建美观的命令行界面
"""

import sys
import os

# 添加父目录到路径，以便导入audioop_fix
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 尝试修复audioop模块缺失问题（Python 3.14兼容性）
try:
    from audioop_fix import *
    print("[INFO] audioop模块修复已加载")
except ImportError as e:
    print(f"[WARNING] 无法加载audioop修复模块: {e}")

from pathlib import Path
from typing import Optional, List
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax

from core.processor import VideoProcessor
from models.config import Config, ModelProvider, WhisperModelSize
from models.template import TemplateType
from utils.config_manager import get_config_manager, get_config
from utils.logger import setup_logger, log
from utils.validator import validate_video_url
from utils.exceptions import VideoMindError

# 创建Typer应用
app = typer.Typer(
    name="videomind",
    help="自动化视频内容处理系统 - 从视频链接到结构化笔记",
    add_completion=False,
    rich_markup_mode="rich",
)

# 创建Rich控制台
console = Console()


def show_banner():
    """显示应用横幅"""
    banner = """
    [bold blue]╔══════════════════════════════════════════════════════════╗[/bold blue]
    [bold blue]║[/bold blue]      [bold cyan]VideoMind[/bold cyan] - 自动化视频内容处理系统       [bold blue]║[/bold blue]
    [bold blue]║[/bold blue]         [italic]从视频链接到结构化笔记[/italic]                [bold blue]║[/bold blue]
    [bold blue]╚══════════════════════════════════════════════════════════╝[/bold blue]
    """
    console.print(banner)


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="显示详细日志"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件路径"),
):
    """
    VideoMind - 自动化视频内容处理系统
    """
    # 显示横幅
    show_banner()

    # 设置日志
    config = get_config()
    if verbose:
        config.log.level = "DEBUG"

    setup_logger(config.log)

    # 如果没有提供子命令，进入交互模式
    if ctx.invoked_subcommand is None:
        interactive_mode()


@app.command(name="process", help="处理单个视频")
def process_video(
    url: str = typer.Argument(..., help="视频URL"),
    template: str = typer.Option("study_notes", "--template", "-t", help="使用的模板名称"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="输出目录"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="使用的模型名称"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="转写语言"),
    keep_files: bool = typer.Option(False, "--keep-files", help="保留中间文件"),
    no_cache: bool = typer.Option(False, "--no-cache", help="禁用缓存"),
):
    """
    处理单个视频并生成结构化笔记
    """
    try:
        # 验证URL
        url = validate_video_url(url)

        # 获取配置
        config = get_config()

        # 覆盖配置选项
        if output_dir:
            config.download.output_dir = output_dir
        if model:
            config.api.default_model = model
        if language:
            config.processing.whisper_language = language
        config.processing.keep_intermediate_files = keep_files

        # 创建处理器
        processor = VideoProcessor(config)

        # 显示处理信息
        info_table = Table(title="处理信息", show_header=False, box=None)
        info_table.add_row("视频URL", f"[cyan]{url}[/cyan]")
        info_table.add_row("使用模板", f"[green]{template}[/green]")
        info_table.add_row("输出目录", f"[yellow]{config.download.output_dir}[/yellow]")
        info_table.add_row("使用模型", f"[magenta]{config.api.default_model}[/magenta]")
        console.print(info_table)
        console.print()

        # 创建进度条
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            # 添加任务
            task = progress.add_task("[cyan]处理视频...", total=100)

            # 处理视频
            result = processor.process_video(
                url=url,
                template_name=template,
                use_cache=not no_cache,
                progress_callback=lambda p: progress.update(task, completed=p)
            )

        # 显示结果
        console.print()
        console.print(Panel.fit(
            "[bold green]✅ 处理完成！[/bold green]",
            border_style="green"
        ))

        # 显示结果摘要
        summary_table = Table(title="处理结果摘要", box=None)
        summary_table.add_column("项目", style="cyan")
        summary_table.add_column("值", style="green")

        summary_table.add_row("视频标题", result.video_info.title or "未知")
        summary_table.add_row("处理状态", result.status.value)
        summary_table.add_row("总耗时", f"{result.total_duration:.1f}秒" if result.total_duration is not None else "未知")
        summary_table.add_row("转写文本长度", f"{len(result.transcript or '')}字符")
        summary_table.add_row("笔记长度", f"{len(result.structured_notes or '')}字符")

        if result.template_used:
            summary_table.add_row("使用模板", result.template_used)

        console.print(summary_table)

        # 显示输出文件路径
        if result.video_info.output_path and result.video_info.output_path.exists():
            console.print()
            console.print(f"[bold]输出文件:[/bold] [cyan]{result.video_info.output_path}[/cyan]")

            # 显示笔记预览
            if result.structured_notes:
                console.print()
                console.print("[bold]笔记预览:[/bold]")
                preview = result.structured_notes[:500] + ("..." if len(result.structured_notes) > 500 else "")
                console.print(Markdown(preview))

    except VideoMindError as e:
        console.print(f"[bold red]错误:[/bold red] {str(e)}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]未知错误:[/bold red] {str(e)}")
        sys.exit(1)


@app.command(name="batch", help="批量处理视频")
def batch_process(
    urls_file: Path = typer.Argument(..., help="包含视频URL的文件（每行一个URL）"),
    template: str = typer.Option("study_notes", "--template", "-t", help="使用的模板名称"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="输出目录"),
    max_workers: int = typer.Option(1, "--workers", "-w", help="最大并行处理数"),
):
    """
    批量处理多个视频
    """
    try:
        # 读取URL文件
        if not urls_file.exists():
            console.print(f"[bold red]错误:[/bold red] 文件不存在: {urls_file}")
            sys.exit(1)

        with open(urls_file, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]

        if not urls:
            console.print("[bold yellow]警告:[/bold yellow] URL文件为空")
            return

        console.print(f"[bold]找到 {len(urls)} 个视频URL[/bold]")

        # 获取配置
        config = get_config()
        if output_dir:
            config.download.output_dir = output_dir

        # 创建处理器
        processor = VideoProcessor(config)

        # 批量处理
        results = processor.batch_process(
            urls=urls,
            template_name=template,
            max_workers=max_workers
        )

        # 显示统计信息
        success_count = sum(1 for r in results if r.status.value == "completed")
        failed_count = len(urls) - success_count

        console.print()
        console.print(Panel.fit(
            f"[bold]批量处理完成[/bold]\n"
            f"成功: [green]{success_count}[/green] | 失败: [red]{failed_count}[/red]",
            border_style="blue"
        ))

        # 显示失败详情（如果有）
        if failed_count > 0:
            console.print()
            console.print("[bold red]失败详情:[/bold red]")
            for i, result in enumerate(results):
                if result.status.value != "completed":
                    console.print(f"  {i+1}. {result.video_info.url}")
                    if result.error:
                        console.print(f"     错误: {result.error}")

    except Exception as e:
        console.print(f"[bold red]错误:[/bold red] {str(e)}")
        sys.exit(1)


@app.command(name="config", help="配置管理")
def config_management(
    show: bool = typer.Option(False, "--show", "-s", help="显示当前配置"),
    set_key: Optional[str] = typer.Option(None, "--set", help="设置配置项（格式: section.key=value）"),
    reset: bool = typer.Option(False, "--reset", help="重置为默认配置"),
):
    """
    管理系统配置
    """
    config_manager = get_config_manager()

    if reset:
        config_manager.create_default_config()
        console.print("[bold green]✅ 配置已重置为默认值[/bold green]")
        return

    if set_key:
        try:
            # 解析设置项
            if "=" not in set_key:
                console.print("[bold red]错误:[/bold red] 格式应为 section.key=value")
                sys.exit(1)

            key_part, value = set_key.split("=", 1)
            if "." not in key_part:
                console.print("[bold red]错误:[/bold red] 格式应为 section.key=value")
                sys.exit(1)

            section, key = key_part.split(".", 1)

            # 更新配置
            updates = {section: {key: value}}
            config_manager.update_config(updates)

            console.print(f"[bold green]✅ 配置已更新: {section}.{key} = {value}[/bold green]")

        except Exception as e:
            console.print(f"[bold red]错误:[/bold red] {str(e)}")
            sys.exit(1)

    if show or not (set_key or reset):
        config = config_manager.get_config()

        # 显示配置
        console.print("[bold cyan]当前配置:[/bold cyan]")
        console.print()

        # API配置
        api_table = Table(title="API配置", box=None)
        api_table.add_column("项目", style="cyan")
        api_table.add_column("值", style="green")

        api_table.add_row("模型提供商", config.api.model_provider.value)
        api_table.add_row("默认模型", config.api.default_model)
        api_table.add_row("OpenAI密钥", "[red]已设置[/red]" if config.api.openai_api_key else "[yellow]未设置[/yellow]")
        api_table.add_row("Anthropic密钥", "[red]已设置[/red]" if config.api.anthropic_api_key else "[yellow]未设置[/yellow]")
        api_table.add_row("温度", str(config.api.temperature))
        api_table.add_row("最大Token数", str(config.api.max_tokens))

        console.print(api_table)
        console.print()

        # 下载配置
        download_table = Table(title="下载配置", box=None)
        download_table.add_column("项目", style="cyan")
        download_table.add_column("值", style="green")

        download_table.add_row("输出目录", str(config.download.output_dir))
        download_table.add_row("临时目录", str(config.download.temp_dir))
        download_table.add_row("下载超时", f"{config.download.download_timeout}秒")
        download_table.add_row("最大下载速度", f"{config.download.max_download_speed}KB/s" if config.download.max_download_speed > 0 else "不限速")

        console.print(download_table)
        console.print()

        # 处理配置
        processing_table = Table(title="处理配置", box=None)
        processing_table.add_column("项目", style="cyan")
        processing_table.add_column("值", style="green")

        processing_table.add_row("Whisper模型", config.processing.whisper_model.value)
        processing_table.add_row("转写语言", config.processing.whisper_language or "自动检测")
        processing_table.add_row("默认模板", config.processing.default_template)
        processing_table.add_row("保留中间文件", "是" if config.processing.keep_intermediate_files else "否")
        processing_table.add_row("最大重试次数", str(config.processing.max_retries))

        console.print(processing_table)
        console.print()

        # AI配置
        ai_table = Table(title="AI功能配置", box=None)
        ai_table.add_column("项目", style="cyan")
        ai_table.add_column("值", style="green")

        # 成本控制
        ai_table.add_row("成本监控", "启用" if config.ai.enable_cost_monitoring else "禁用")
        ai_table.add_row("每日预算", f"${config.ai.daily_budget:.2f}" if config.ai.daily_budget else "无限制")
        ai_table.add_row("每月预算", f"${config.ai.monthly_budget:.2f}" if config.ai.monthly_budget else "无限制")

        # Prompt优化
        ai_table.add_row("Prompt优化", "启用" if config.ai.enable_prompt_optimization else "禁用")
        ai_table.add_row("优化级别", config.ai.default_optimization_level)

        # 批量处理
        ai_table.add_row("最大并发任务", str(config.ai.max_concurrent_batch_tasks))
        ai_table.add_row("最大工作线程", str(config.ai.max_workers_per_batch))

        # 模型监控
        ai_table.add_row("模型监控", "启用" if config.ai.enable_model_monitoring else "禁用")
        ai_table.add_row("数据保留天数", str(config.ai.performance_data_retention_days))

        # 高级功能
        ai_table.add_row("上下文管理", "启用" if config.ai.enable_context_management else "禁用")
        ai_table.add_row("输出验证", "启用" if config.ai.enable_output_validation else "禁用")
        ai_table.add_row("流式输出", "启用" if config.ai.enable_streaming_output else "禁用")

        console.print(ai_table)


@app.command(name="template", help="模板管理")
def template_management(
    list_templates: bool = typer.Option(False, "--list", "-l", help="列出所有模板"),
    show: Optional[str] = typer.Option(None, "--show", "-s", help="显示指定模板详情"),
    create: Optional[Path] = typer.Option(None, "--create", "-c", help="从JSON文件创建模板"),
    delete: Optional[str] = typer.Option(None, "--delete", "-d", help="删除模板"),
):
    """
    管理提示模板
    """
    from core.template_engine import TemplateEngine

    template_engine = TemplateEngine()

    if list_templates:
        templates = template_engine.list_templates()

        table = Table(title="可用模板", box=None)
        table.add_column("名称", style="cyan")
        table.add_column("类型", style="green")
        table.add_column("描述", style="yellow")
        table.add_column("变量", style="magenta")

        for template in templates:
            variables = ", ".join([v.name.strip("{}") for v in template.variables])
            table.add_row(
                template.name,
                template.type.value,
                template.description[:50] + ("..." if len(template.description) > 50 else ""),
                variables or "无"
            )

        console.print(table)

    elif show:
        try:
            template = template_engine.get_template(show)

            # 显示模板详情
            console.print(f"[bold cyan]模板: {template.name}[/bold cyan]")
            console.print(f"[bold]类型:[/bold] {template.type.value}")
            console.print(f"[bold]描述:[/bold] {template.description}")
            console.print(f"[bold]版本:[/bold] {template.version}")

            if template.author:
                console.print(f"[bold]作者:[/bold] {template.author}")

            if template.tags:
                console.print(f"[bold]标签:[/bold] {', '.join(template.tags)}")

            # 显示变量
            if template.variables:
                console.print()
                console.print("[bold]变量:[/bold]")
                for var in template.variables:
                    required = "必需" if var.required else "可选"
                    console.print(f"  • {var.name} - {var.description} ({required})")

            # 显示系统提示
            if template.system_prompt:
                console.print()
                console.print("[bold]系统提示:[/bold]")
                console.print(Syntax(template.system_prompt, "text", theme="monokai"))

            # 显示用户提示
            console.print()
            console.print("[bold]用户提示:[/bold]")
            console.print(Syntax(template.user_prompt, "text", theme="monokai"))

            # 显示模型配置
            console.print()
            console.print("[bold]模型配置:[/bold]")
            for key, value in template.model_parameters.items():
                console.print(f"  {key}: {value}")

        except Exception as e:
            console.print(f"[bold red]错误:[/bold red] {str(e)}")
            sys.exit(1)

    elif create:
        try:
            if not create.exists():
                console.print(f"[bold red]错误:[/bold red] 文件不存在: {create}")
                sys.exit(1)

            with open(create, "r", encoding="utf-8") as f:
                template_data = f.read()

            template = template_engine.import_template(template_data)
            console.print(f"[bold green]✅ 模板创建成功: {template.name}[/bold green]")

        except Exception as e:
            console.print(f"[bold red]错误:[/bold red] {str(e)}")
            sys.exit(1)

    elif delete:
        try:
            template_engine.delete_template(delete)
            console.print(f"[bold green]✅ 模板删除成功: {delete}[/bold green]")
        except Exception as e:
            console.print(f"[bold red]错误:[/bold red] {str(e)}")
            sys.exit(1)

    else:
        # 显示帮助
        console.print("[bold cyan]模板管理命令:[/bold cyan]")
        console.print("  videomind template --list         列出所有模板")
        console.print("  videomind template --show NAME    显示模板详情")
        console.print("  videomind template --create FILE  从JSON文件创建模板")
        console.print("  videomind template --delete NAME  删除模板")


@app.command(name="info", help="系统信息")
def system_info():
    """
    显示系统信息和状态
    """
    from core.downloader import VideoDownloader
    from core.audio_extractor import AudioExtractor
    from core.transcriber import Transcriber
    from core.llm_client import LLMClient

    config = get_config()

    # 显示系统信息
    console.print("[bold cyan]系统信息[/bold cyan]")
    console.print()

    # 检查FFmpeg
    extractor = AudioExtractor()
    ffmpeg_available = extractor.check_ffmpeg_available()
    ffmpeg_status = "[green]可用[/green]" if ffmpeg_available else "[red]不可用[/red]"

    # 检查Whisper模型
    transcriber = Transcriber(config.processing.whisper_model)
    model_info = transcriber.get_model_info()
    whisper_status = "[green]已加载[/green]" if model_info["loaded"] else "[red]未加载[/red]"

    # 检查API连接
    llm_client = LLMClient(config.api)
    api_test = llm_client.test_connection()
    api_status = "[green]连接正常[/green]" if api_test else "[red]连接失败[/red]"

    # 创建信息表
    info_table = Table(box=None)
    info_table.add_column("组件", style="cyan")
    info_table.add_column("状态", style="green")
    info_table.add_column("详情", style="yellow")

    info_table.add_row("FFmpeg", ffmpeg_status, "音频处理工具")
    info_table.add_row("Whisper", whisper_status, f"模型: {model_info.get('model_size', '未知')}")
    info_table.add_row("API连接", api_status, f"提供商: {config.api.model_provider.value}")
    info_table.add_row("输出目录", "[green]可写[/green]" if config.download.output_dir.exists() else "[red]不可写[/red]", str(config.download.output_dir))
    info_table.add_row("临时目录", "[green]可写[/green]" if config.download.temp_dir.exists() else "[red]不可写[/red]", str(config.download.temp_dir))

    console.print(info_table)

    # 显示支持的平台
    console.print()
    console.print("[bold cyan]支持的视频平台:[/bold cyan]")
    downloader = VideoDownloader(config.download)
    sites = downloader.get_supported_sites()[:10]  # 只显示前10个
    console.print(", ".join(sites))
    if len(sites) > 10:
        console.print(f"... 等 {len(downloader.get_supported_sites())} 个平台")


@app.command(name="clean", help="清理临时文件")
def cleanup(
    temp_files: bool = typer.Option(True, "--temp", help="清理临时文件"),
    cache: bool = typer.Option(False, "--cache", help="清理缓存"),
    results: bool = typer.Option(False, "--results", help="清理结果文件"),
    all: bool = typer.Option(False, "--all", help="清理所有文件"),
    days: int = typer.Option(7, "--days", help="清理超过指定天数的文件"),
):
    """
    清理系统文件
    """
    from storage.cache_manager import CacheManager
    from storage.result_storage import ResultStorage

    config = get_config()

    if all:
        temp_files = cache = results = True

    cleaned_count = 0

    # 清理临时文件
    if temp_files:
        temp_dir = config.download.temp_dir
        if temp_dir.exists():
            for file in temp_dir.glob("*"):
                try:
                    file.unlink()
                    cleaned_count += 1
                except Exception:
                    pass
            console.print(f"[green]清理了临时目录: {temp_dir}[/green]")

    # 清理缓存
    if cache:
        cache_manager = CacheManager()
        cache_manager.clear_cache("all")
        console.print("[green]清理了所有缓存[/green]")

    # 清理结果文件
    if results:
        result_storage = ResultStorage()
        result_storage.cleanup_old_results(days)
        console.print(f"[green]清理了超过{days}天的结果文件[/green]")

    if cleaned_count > 0 or cache or results:
        console.print(f"[bold green]✅ 清理完成[/bold green]")
    else:
        console.print("[yellow]没有需要清理的文件[/yellow]")


@app.command(name="version", help="显示版本信息")
def version_info():
    """
    显示版本信息
    """
    import videomind

    console.print(f"[bold cyan]VideoMind[/bold cyan] v{videomind.__version__}")
    console.print("自动化视频内容处理系统")
    console.print("https://github.com/yourusername/videomind")


@app.command(name="ai", help="AI功能管理")
def ai_management(
    cost: bool = typer.Option(False, "--cost", "-c", help="显示成本统计"),
    optimize: Optional[str] = typer.Option(None, "--optimize", "-o", help="优化指定模板或文件"),
    model_stats: bool = typer.Option(False, "--model-stats", "-m", help="显示模型性能统计"),
    batch_status: bool = typer.Option(False, "--batch-status", "-b", help="显示批量任务状态"),
    insights: bool = typer.Option(False, "--insights", "-i", help="显示AI功能洞察"),
    export: Optional[Path] = typer.Option(None, "--export", "-e", help="导出AI数据到文件"),
):
    """
    AI功能管理和监控
    """
    try:
        if cost:
            _show_cost_statistics()
        elif optimize:
            _optimize_prompt(optimize)
        elif model_stats:
            _show_model_statistics()
        elif batch_status:
            _show_batch_status()
        elif insights:
            _show_ai_insights()
        elif export:
            _export_ai_data(export)
        else:
            # 显示AI功能概览
            _show_ai_overview()

    except Exception as e:
        console.print(f"[bold red]错误:[/bold red] {str(e)}")
        sys.exit(1)


def _show_cost_statistics():
    """显示成本统计"""
    from core.cost_monitor import get_cost_monitor

    cost_monitor = get_cost_monitor()

    # 获取成本汇总
    daily_summary = cost_monitor.get_cost_summary()
    weekly_summary = cost_monitor.get_cost_summary(period="weekly")
    monthly_summary = cost_monitor.get_cost_summary(period="monthly")

    console.print("[bold cyan]成本统计[/bold cyan]")
    console.print()

    # 显示汇总表格
    summary_table = Table(title="成本汇总", box=None)
    summary_table.add_column("周期", style="cyan")
    summary_table.add_column("总成本", style="green")
    summary_table.add_column("总Token数", style="yellow")
    summary_table.add_column("请求数", style="magenta")
    summary_table.add_column("成功率", style="blue")

    for period, summary in [("今日", daily_summary), ("本周", weekly_summary), ("本月", monthly_summary)]:
        success_rate = (summary.success_count / summary.request_count * 100) if summary.request_count > 0 else 0
        summary_table.add_row(
            period,
            f"${summary.total_cost:.4f}",
            f"{summary.total_tokens:,}",
            str(summary.request_count),
            f"{success_rate:.1f}%"
        )

    console.print(summary_table)
    console.print()

    # 显示成本分解
    breakdown = cost_monitor.get_cost_breakdown(days=7)
    if breakdown:
        console.print("[bold]最近7天成本分解:[/bold]")
        breakdown_table = Table(box=None)
        breakdown_table.add_column("日期", style="cyan")
        breakdown_table.add_column("成本", style="green")
        breakdown_table.add_column("Token数", style="yellow")
        breakdown_table.add_column("请求数", style="magenta")

        for day in breakdown[:7]:  # 显示最近7天
            breakdown_table.add_row(
                day["date"],
                f"${day['cost']:.4f}",
                f"{day['tokens']:,}",
                str(day["requests"])
            )

        console.print(breakdown_table)
        console.print()

    # 显示优化建议
    suggestions = cost_monitor.get_optimization_suggestions()
    if suggestions:
        console.print("[bold yellow]成本优化建议:[/bold yellow]")
        for suggestion in suggestions[:5]:  # 最多显示5条建议
            console.print(f"• {suggestion}")


def _optimize_prompt(target: str):
    """优化prompt"""
    from core.prompt_optimizer import get_prompt_optimizer, PromptOptimizationLevel
    from core.template_engine import TemplateEngine
    from core.llm_client import LLMClient
    from models.config import get_config

    config = get_config()
    llm_client = LLMClient(config.api)
    optimizer = get_prompt_optimizer(llm_client)
    template_engine = TemplateEngine()

    console.print(f"[bold cyan]优化目标:[/bold cyan] {target}")
    console.print()

    try:
        # 检查是否是模板
        if target in template_engine.templates:
            template = template_engine.get_template(target)
            prompt = template.user_prompt
            source_type = "模板"
        else:
            # 假设是文件路径
            import os
            if os.path.exists(target):
                with open(target, "r", encoding="utf-8") as f:
                    prompt = f.read()
                source_type = "文件"
            else:
                # 假设是直接prompt
                prompt = target
                source_type = "文本"

        console.print(f"[bold]来源类型:[/bold] {source_type}")
        console.print(f"[bold]原始长度:[/bold] {len(prompt)} 字符")
        console.print()

        # 进行优化
        with console.status("[bold green]正在优化prompt...[/bold green]"):
            result = optimizer.optimize_prompt(
                prompt,
                optimization_level=PromptOptimizationLevel.BALANCED,
                target_model=config.api.default_model
            )

        # 显示优化结果
        console.print("[bold green]✅ 优化完成![/bold green]")
        console.print()

        # 显示统计
        stats_table = Table(title="优化统计", box=None)
        stats_table.add_column("指标", style="cyan")
        stats_table.add_column("原始", style="yellow")
        stats_table.add_column("优化后", style="green")
        stats_table.add_column("节省", style="magenta")

        stats_table.add_row(
            "字符数",
            str(result.analysis.original_length),
            str(result.analysis.optimized_length),
            f"{result.analysis.original_length - result.analysis.optimized_length} (-{((result.analysis.original_length - result.analysis.optimized_length)/result.analysis.original_length*100):.1f}%)"
        )

        stats_table.add_row(
            "Token数",
            str(result.analysis.token_estimate),
            str(result.analysis.token_estimate - result.tokens_saved),
            f"{result.tokens_saved} (-{(result.tokens_saved/result.analysis.token_estimate*100):.1f}%)"
        )

        stats_table.add_row(
            "成本",
            f"${result.analysis.cost_estimate:.6f}",
            f"${result.analysis.cost_estimate - result.cost_saved:.6f}",
            f"${result.cost_saved:.6f} (-{(result.cost_saved/result.analysis.cost_estimate*100):.1f}%)"
        )

        console.print(stats_table)
        console.print()

        # 显示质量评分
        console.print("[bold]质量评分:[/bold]")
        console.print(f"• 可读性: {result.analysis.readability_score:.2f}/1.0")
        console.print(f"• 清晰度: {result.analysis.clarity_score:.2f}/1.0")
        console.print(f"• 具体性: {result.analysis.specificity_score:.2f}/1.0")
        console.print()

        # 显示发现的问题
        if result.analysis.issues_found:
            console.print("[bold yellow]发现的问题:[/bold yellow]")
            for issue in result.analysis.issues_found:
                console.print(f"• {issue}")
            console.print()

        # 显示优化建议
        if result.analysis.suggestions:
            console.print("[bold green]优化建议:[/bold green]")
            for suggestion in result.analysis.suggestions[:3]:  # 最多显示3条
                console.print(f"• {suggestion}")
            console.print()

        # 显示优化前后对比
        console.print("[bold]优化前后对比:[/bold]")
        console.print()
        console.print("[bold yellow]优化前:[/bold yellow]")
        console.print(Syntax(result.original_prompt[:500] + ("..." if len(result.original_prompt) > 500 else ""), "text", theme="monokai"))
        console.print()
        console.print("[bold green]优化后:[/bold green]")
        console.print(Syntax(result.optimized_prompt[:500] + ("..." if len(result.optimized_prompt) > 500 else ""), "text", theme="monokai"))

        # 询问是否保存优化结果
        console.print()
        save = typer.confirm("是否保存优化结果到文件？", default=False)
        if save:
            output_file = Path(f"optimized_{target if source_type == '文件' else 'prompt'}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result.optimized_prompt)
            console.print(f"[bold green]✅ 优化结果已保存到: {output_file}[/bold green]")

    except Exception as e:
        console.print(f"[bold red]优化失败:[/bold red] {str(e)}")
        raise


def _show_model_statistics():
    """显示模型性能统计"""
    from core.model_monitor import get_model_monitor

    model_monitor = get_model_monitor()

    # 获取模型性能数据
    performances = model_monitor.list_model_performances(min_requests=5, sort_by="performance_score", descending=True)

    console.print("[bold cyan]模型性能统计[/bold cyan]")
    console.print()

    if not performances:
        console.print("[yellow]暂无足够的模型性能数据[/yellow]")
        return

    # 显示性能表格
    perf_table = Table(title="模型性能排名", box=None)
    perf_table.add_column("排名", style="cyan")
    perf_table.add_column("模型", style="green")
    perf_table.add_column("提供商", style="yellow")
    perf_table.add_column("性能评分", style="magenta")
    perf_table.add_column("平均响应时间", style="blue")
    perf_table.add_column("成功率", style="green")
    perf_table.add_column("平均成本", style="red")

    for i, perf in enumerate(performances[:10], 1):  # 显示前10个
        success_rate = (perf.successful_requests / perf.total_requests * 100) if perf.total_requests > 0 else 0
        perf_table.add_row(
            str(i),
            perf.model_name,
            perf.provider,
            f"{perf.performance_score:.1f}/100",
            f"{perf.avg_response_time:.2f}s",
            f"{success_rate:.1f}%",
            f"${perf.avg_cost_per_request:.6f}"
        )

    console.print(perf_table)
    console.print()

    # 显示性能趋势（第一个模型）
    if performances:
        main_model = performances[0].model_name
        trends = model_monitor.get_performance_trends(main_model, days=7)

        if trends and trends.get("response_time"):
            console.print(f"[bold]{main_model} 最近7天性能趋势:[/bold]")

            trend_table = Table(box=None)
            trend_table.add_column("日期", style="cyan")
            trend_table.add_column("响应时间", style="green")
            trend_table.add_column("成功率", style="yellow")
            trend_table.add_column("请求数", style="magenta")

            for i in range(min(7, len(trends["response_time"]))):
                date = trends["response_time"][i][0]
                response_time = trends["response_time"][i][1]
                success_rate = trends["success_rate"][i][1] if i < len(trends["success_rate"]) else 0
                requests = trends["daily_requests"][i][1] if i < len(trends["daily_requests"]) else 0

                trend_table.add_row(
                    date,
                    f"{response_time:.2f}s",
                    f"{success_rate:.1f}%",
                    str(requests)
                )

            console.print(trend_table)
            console.print()

    # 显示模型推荐
    console.print("[bold]模型推荐示例:[/bold]")
    recommendations = [
        ("总结任务", "balanced", None, None),
        ("分析任务", "high_quality", 0.05, 10.0),
        ("创意任务", "high_quality", None, None),
    ]

    for task_type, quality, budget, time_constraint in recommendations:
        try:
            recommendation = model_monitor.recommend_model(
                task_type=task_type,
                budget_constraint=budget,
                time_constraint=time_constraint,
                quality_requirement=quality
            )

            console.print(f"• [bold]{task_type}[/bold]: {recommendation.recommended_model}")
            console.print(f"  理由: {recommendation.reasoning}")
            console.print(f"  置信度: {recommendation.confidence_score:.1%}")
            if recommendation.alternative_models:
                console.print(f"  备选: {', '.join(recommendation.alternative_models)}")
            console.print()

        except Exception as e:
            console.print(f"• [bold]{task_type}[/bold]: 推荐失败 - {str(e)}")


def _show_batch_status():
    """显示批量任务状态"""
    from core.batch_manager import get_batch_manager

    batch_manager = get_batch_manager()

    # 获取任务列表
    tasks = batch_manager.list_tasks(limit=10)

    console.print("[bold cyan]批量任务状态[/bold cyan]")
    console.print()

    if not tasks:
        console.print("[yellow]暂无批量任务[/yellow]")
        return

    # 显示任务表格
    tasks_table = Table(title="批量任务列表", box=None)
    tasks_table.add_column("任务ID", style="cyan")
    tasks_table.add_column("状态", style="green")
    tasks_table.add_column("视频数", style="yellow")
    tasks_table.add_column("进度", style="magenta")
    tasks_table.add_column("创建时间", style="blue")
    tasks_table.add_column("模板", style="green")

    for task in tasks:
        # 根据状态设置颜色
        status_color = {
            "pending": "yellow",
            "running": "green",
            "paused": "blue",
            "completed": "cyan",
            "failed": "red",
            "cancelled": "grey"
        }.get(task.status.value, "white")

        tasks_table.add_row(
            task.task_id[:8] + "...",
            f"[{status_color}]{task.status.value}[/{status_color}]",
            str(len(task.urls)),
            f"{task.progress*100:.1f}%",
            task.created_at[:10],
            task.template_name
        )

    console.print(tasks_table)
    console.print()

    # 显示性能统计
    stats = batch_manager.get_performance_stats()
    console.print("[bold]批量处理统计:[/bold]")
    console.print(f"• 总任务数: {stats['total_tasks']}")
    console.print(f"• 总视频数: {stats['total_videos']}")
    console.print(f"• 成功率: {stats['success_rate']:.1f}%")
    console.print(f"• 总成本: ${stats['total_cost']:.4f}")
    console.print(f"• 平均处理时间: {stats['avg_processing_time']:.1f}秒")
    console.print(f"• 平均成本/视频: ${stats['avg_cost_per_video']:.6f}")

    # 显示活动任务详情
    active_tasks = [t for t in tasks if t.status.value in ["running", "paused"]]
    if active_tasks:
        console.print()
        console.print("[bold]活动任务详情:[/bold]")
        for task in active_tasks:
            progress = batch_manager.get_task_progress(task.task_id)
            if progress:
                console.print(f"• [bold]{task.task_id[:8]}...[/bold]: {progress.processed_items}/{progress.total_items} 完成")
                if progress.estimated_time_remaining:
                    console.print(f"  预计剩余时间: {progress.estimated_time_remaining:.1f}秒")
                if progress.current_speed:
                    console.print(f"  当前速度: {progress.current_speed:.2f} 视频/秒")


def _show_ai_insights():
    """显示AI功能洞察"""
    from core.cost_monitor import get_cost_monitor
    from core.model_monitor import get_model_monitor
    from core.batch_manager import get_batch_manager

    cost_monitor = get_cost_monitor()
    model_monitor = get_model_monitor()
    batch_manager = get_batch_manager()

    console.print("[bold cyan]AI功能洞察[/bold cyan]")
    console.print()

    # 成本洞察
    console.print("[bold]成本洞察:[/bold]")
    cost_suggestions = cost_monitor.get_optimization_suggestions()
    for suggestion in cost_suggestions[:3]:
        console.print(f"• {suggestion}")
    console.print()

    # 模型性能洞察
    console.print("[bold]模型性能洞察:[/bold]")
    model_insights = model_monitor.get_performance_insights()
    for insight in model_insights[:5]:
        console.print(f"• {insight}")
    console.print()

    # 批量处理洞察
    console.print("[bold]批量处理洞察:[/bold]")
    batch_stats = batch_manager.get_performance_stats()
    if batch_stats["total_videos"] > 0:
        console.print(f"• 已处理 {batch_stats['total_videos']} 个视频，成功率 {batch_stats['success_rate']:.1f}%")
        console.print(f"• 总成本 ${batch_stats['total_cost']:.4f}，平均 ${batch_stats['avg_cost_per_video']:.6f}/视频")
        console.print(f"• 平均处理时间 {batch_stats['avg_processing_time']:.1f} 秒/视频")

        # 提供优化建议
        if batch_stats["success_rate"] < 80:
            console.print("• [yellow]建议: 成功率较低，检查网络连接和API密钥[/yellow]")
        if batch_stats["avg_cost_per_video"] > 0.05:
            console.print("• [yellow]建议: 平均成本较高，考虑使用更经济的模型[/yellow]")
    else:
        console.print("• 暂无批量处理数据")

    console.print()

    # 系统建议
    console.print("[bold]系统建议:[/bold]")
    console.print("1. 定期检查成本预算，避免意外费用")
    console.print("2. 使用模型推荐功能选择最适合任务的模型")
    console.print("3. 批量处理视频以提高效率")
    console.print("4. 优化prompt以减少token使用和成本")
    console.print("5. 监控模型性能，及时发现问题")


def _export_ai_data(file_path: Path):
    """导出AI数据"""
    from core.cost_monitor import get_cost_monitor
    from core.model_monitor import get_model_monitor

    cost_monitor = get_cost_monitor()
    model_monitor = get_model_monitor()

    console.print(f"[bold cyan]导出AI数据到: {file_path}[/bold cyan]")
    console.print()

    try:
        # 确定导出格式
        if file_path.suffix.lower() == ".csv":
            format = "csv"
        else:
            format = "json"

        # 导出成本数据
        cost_file = file_path.parent / f"cost_data{file_path.suffix}"
        cost_monitor.export_cost_data(format=format, file_path=cost_file)
        console.print(f"[green]✅ 成本数据已导出到: {cost_file}[/green]")

        # 导出模型性能数据
        model_file = file_path.parent / f"model_performance{file_path.suffix}"
        model_monitor.export_performance_data(format=format, file_path=model_file)
        console.print(f"[green]✅ 模型性能数据已导出到: {model_file}[/green]")

        console.print()
        console.print("[bold green]✅ 所有数据导出完成![/bold green]")

    except Exception as e:
        console.print(f"[bold red]导出失败:[/bold red] {str(e)}")
        raise


def _show_ai_overview():
    """显示AI功能概览"""
    from core.cost_monitor import get_cost_monitor
    from core.model_monitor import get_model_monitor
    from core.batch_manager import get_batch_manager

    cost_monitor = get_cost_monitor()
    model_monitor = get_model_monitor()
    batch_manager = get_batch_manager()

    console.print("[bold cyan]AI功能概览[/bold cyan]")
    console.print()

    # 成本概览
    daily_cost = cost_monitor.get_period_cost(
        start_date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat(),
        end_date=datetime.now().isoformat()
    )
    monthly_cost = cost_monitor.get_period_cost(
        start_date=datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat(),
        end_date=datetime.now().isoformat()
    )

    console.print("[bold]成本概览:[/bold]")
    console.print(f"• 今日成本: ${daily_cost:.4f}")
    console.print(f"• 本月成本: ${monthly_cost:.4f}")
    console.print()

    # 模型概览
    model_count = len(model_monitor.list_model_performances(min_requests=0))
    top_models = model_monitor.list_model_performances(min_requests=5, sort_by="performance_score", descending=True)[:3]

    console.print("[bold]模型概览:[/bold]")
    console.print(f"• 已监控模型: {model_count} 个")
    if top_models:
        console.print(f"• 最佳模型: {top_models[0].model_name} (评分: {top_models[0].performance_score:.1f}/100)")
    console.print()

    # 批量处理概览
    batch_stats = batch_manager.get_performance_stats()
    console.print("[bold]批量处理概览:[/bold]")
    console.print(f"• 总任务数: {batch_stats['total_tasks']}")
    console.print(f"• 总视频数: {batch_stats['total_videos']}")
    console.print(f"• 成功率: {batch_stats['success_rate']:.1f}%")
    console.print()

    # 可用命令
    console.print("[bold]可用命令:[/bold]")
    console.print("• videomind ai --cost          显示成本统计")
    console.print("• videomind ai --optimize FILE 优化prompt或模板")
    console.print("• videomind ai --model-stats   显示模型性能统计")
    console.print("• videomind ai --batch-status  显示批量任务状态")
    console.print("• videomind ai --insights      显示AI功能洞察")
    console.print("• videomind ai --export FILE   导出AI数据")


def interactive_mode():
    """
    交互模式：提示用户输入URL并处理视频
    """
    console.print("[bold cyan]📹 欢迎使用 VideoMind 交互模式[/bold cyan]")
    console.print("[dim]提示：直接输入视频URL即可开始处理，输入 'quit' 或 'exit' 退出[/dim]")
    console.print()

    # 获取配置
    config = get_config()
    
    # 创建处理器
    processor = VideoProcessor(config)

    # 主循环
    while True:
        try:
            # 提示输入URL
            console.print()
            url = typer.prompt(
                "[bold cyan]请输入视频URL[/bold cyan]",
                default="",
            ).strip()

            # 检查退出命令
            if url.lower() in ["quit", "exit", "q"]:
                console.print("[bold yellow]👋 再见！[/bold yellow]")
                break

            # 检查是否为空
            if not url:
                console.print("[bold yellow]⚠️  请输入有效的视频URL[/bold yellow]")
                continue

            # 验证URL
            try:
                url = validate_video_url(url)
            except Exception as e:
                console.print(f"[bold red]❌ URL验证失败:[/bold red] {str(e)}")
                continue

            # 显示处理信息
            info_table = Table(title="处理信息", show_header=False, box=None)
            info_table.add_row("视频URL", f"[cyan]{url}[/cyan]")
            info_table.add_row("使用模板", f"[green]study_notes[/green]")
            info_table.add_row("输出目录", f"[yellow]{config.download.output_dir}[/yellow]")
            info_table.add_row("使用模型", f"[magenta]{config.api.default_model}[/magenta]")
            console.print(info_table)
            console.print()

            # 创建进度条
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as progress:
                # 添加任务
                task = progress.add_task("[cyan]处理视频...", total=100)

                # 处理视频
                result = processor.process_video(
                    url=url,
                    template_name="study_notes",
                    use_cache=True,
                    progress_callback=lambda p: progress.update(task, completed=p)
                )

            # 显示结果
            console.print()
            if result.status.value == "completed":
                console.print(Panel.fit(
                    "[bold green]✅ 处理完成！[/bold green]",
                    border_style="green"
                ))

                # 显示结果摘要
                summary_table = Table(title="处理结果摘要", box=None)
                summary_table.add_column("项目", style="cyan")
                summary_table.add_column("值", style="green")

                summary_table.add_row("视频标题", result.video_info.title or "未知")
                summary_table.add_row("处理状态", result.status.value)
                summary_table.add_row("总耗时", f"{result.total_duration:.1f}秒" if result.total_duration is not None else "未知")
                summary_table.add_row("转写文本长度", f"{len(result.transcript or '')}字符")
                summary_table.add_row("笔记长度", f"{len(result.structured_notes or '')}字符")

                if result.template_used:
                    summary_table.add_row("使用模板", result.template_used)

                console.print(summary_table)

                # 显示输出文件路径
                if result.video_info.output_path and result.video_info.output_path.exists():
                    console.print()
                    console.print(f"[bold]输出文件:[/bold] [cyan]{result.video_info.output_path}[/cyan]")

                    # 显示笔记预览
                    if result.structured_notes:
                        console.print()
                        console.print("[bold]笔记预览:[/bold]")
                        preview = result.structured_notes[:500] + ("..." if len(result.structured_notes) > 500 else "")
                        console.print(Markdown(preview))
            else:
                console.print(Panel.fit(
                    f"[bold red]❌ 处理失败[/bold red]\n{result.error or '未知错误'}",
                    border_style="red"
                ))

            # 询问是否继续
            console.print()
            continue_processing = typer.confirm(
                "[bold cyan]是否继续处理其他视频？[/bold cyan]",
                default=True
            )
            if not continue_processing:
                console.print("[bold yellow]👋 再见！[/bold yellow]")
                break

        except KeyboardInterrupt:
            console.print()
            console.print("[bold yellow]👋 再见！[/bold yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]❌ 发生错误:[/bold red] {str(e)}")
            console.print("[dim]您可以继续输入其他URL，或输入 'quit' 退出[/dim]")


if __name__ == "__main__":
    app()