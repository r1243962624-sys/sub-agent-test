"""
配置数据模型
定义系统的配置结构和验证规则
"""

from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field, validator
from pathlib import Path


class ModelProvider(str, Enum):
    """模型提供商枚举"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    GOOGLE = "google"
    LOCAL = "local"


class WhisperModelSize(str, Enum):
    """Whisper 模型大小枚举"""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class LogLevel(str, Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class APIConfig(BaseModel):
    """API 配置"""
    openai_api_key: Optional[str] = Field(None, description="OpenAI API 密钥")
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API 密钥")
    deepseek_api_key: Optional[str] = Field(None, description="DeepSeek API 密钥")
    google_api_key: Optional[str] = Field(None, description="Google API 密钥")

    model_provider: ModelProvider = Field(ModelProvider.OPENAI, description="模型提供商")
    default_model: str = Field("gpt-4-turbo-preview", description="默认模型名称")

    max_tokens: Optional[int] = Field(None, description="最大生成 token 数（None表示不限制）")
    temperature: float = Field(0.7, description="温度参数", ge=0.0, le=2.0)
    top_p: float = Field(1.0, description="Top-p 采样参数", ge=0.0, le=1.0)

    @validator("default_model")
    def validate_model_name(cls, v, values):
        """验证模型名称与提供商匹配"""
        provider = values.get("model_provider", ModelProvider.OPENAI)

        if provider == ModelProvider.OPENAI:
            if not v.startswith(("gpt-", "text-")):
                raise ValueError(f"OpenAI 模型名称应以 'gpt-' 或 'text-' 开头: {v}")
        elif provider == ModelProvider.ANTHROPIC:
            if not v.startswith("claude-"):
                raise ValueError(f"Anthropic 模型名称应以 'claude-' 开头: {v}")
        elif provider == ModelProvider.DEEPSEEK:
            if not v.startswith("deepseek-"):
                raise ValueError(f"DeepSeek 模型名称应以 'deepseek-' 开头: {v}")
        elif provider == ModelProvider.GOOGLE:
            if not v.startswith(("gemini-", "models/")):
                raise ValueError(f"Google 模型名称应以 'gemini-' 或 'models/' 开头: {v}")

        return v


class DownloadConfig(BaseModel):
    """下载配置"""
    download_timeout: int = Field(300, description="下载超时时间（秒）", ge=30, le=1800)
    max_download_speed: int = Field(0, description="最大下载速度（KB/s，0为不限速）", ge=0)
    output_dir: Path = Field(Path("./output"), description="输出目录")
    temp_dir: Path = Field(Path("./temp"), description="临时文件目录")

    ytdlp_options: Dict[str, Any] = Field(
        default_factory=lambda: {
            "format": "best[height<=720]",
            "quiet": True,
            "no_warnings": True,
        },
        description="yt-dlp 自定义参数"
    )

    @validator("output_dir", "temp_dir")
    def validate_paths(cls, v):
        """验证路径并确保目录存在"""
        v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v


class ProcessingConfig(BaseModel):
    """处理配置"""
    whisper_model: WhisperModelSize = Field(WhisperModelSize.BASE, description="Whisper 模型大小")
    whisper_language: Optional[str] = Field("auto", description="转写语言（auto为自动检测）")

    keep_intermediate_files: bool = Field(False, description="是否保留中间文件")
    max_retries: int = Field(3, description="最大重试次数", ge=0, le=10)
    retry_delay: int = Field(5, description="重试延迟（秒）", ge=1, le=60)

    default_template: str = Field("study_notes", description="默认模板名称")


class LogConfig(BaseModel):
    """日志配置"""
    level: LogLevel = Field(LogLevel.INFO, description="日志级别")
    file: Optional[Path] = Field(Path("./logs/videomind.log"), description="日志文件路径")
    format: str = Field(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        description="日志格式"
    )

    @validator("file")
    def validate_log_file(cls, v):
        """验证日志文件路径"""
        if v:
            v.parent.mkdir(parents=True, exist_ok=True)
        return v


class AIConfig(BaseModel):
    """AI功能配置"""
    # 成本控制
    enable_cost_monitoring: bool = Field(True, description="启用成本监控")
    daily_budget: Optional[float] = Field(None, description="每日预算（美元）", ge=0.0)
    monthly_budget: Optional[float] = Field(None, description="每月预算（美元）", ge=0.0)
    total_budget: Optional[float] = Field(None, description="总预算（美元）", ge=0.0)

    # Prompt优化
    enable_prompt_optimization: bool = Field(True, description="启用Prompt优化")
    default_optimization_level: str = Field("balanced", description="默认优化级别", pattern="^(minimal|balanced|aggressive)$")
    auto_optimize_templates: bool = Field(True, description="自动优化模板")

    # 批量处理
    max_concurrent_batch_tasks: int = Field(3, description="最大并发批量任务数", ge=1, le=10)
    max_workers_per_batch: int = Field(2, description="每个批量任务的最大工作线程数", ge=1, le=5)
    batch_retry_attempts: int = Field(3, description="批量任务重试次数", ge=0, le=10)

    # 模型性能监控
    enable_model_monitoring: bool = Field(True, description="启用模型性能监控")
    performance_data_retention_days: int = Field(90, description="性能数据保留天数", ge=7, le=365)
    auto_model_recommendation: bool = Field(True, description="自动模型推荐")

    # 高级AI功能
    enable_context_management: bool = Field(True, description="启用上下文管理")
    max_context_length: Optional[int] = Field(None, description="最大上下文长度（None表示不限制）")
    enable_output_validation: bool = Field(True, description="启用输出格式验证")
    enable_streaming_output: bool = Field(True, description="启用流式输出")

    # 错误处理和恢复
    max_api_retries: int = Field(3, description="最大API重试次数", ge=0, le=10)
    retry_backoff_factor: float = Field(1.5, description="重试退避因子", ge=1.0, le=5.0)
    fallback_model_enabled: bool = Field(True, description="启用降级模型")

    # 缓存和性能
    enable_response_caching: bool = Field(True, description="启用响应缓存")
    cache_ttl_hours: int = Field(24, description="缓存存活时间（小时）", ge=1, le=168)
    enable_performance_logging: bool = Field(True, description="启用性能日志记录")


class Config(BaseModel):
    """系统总配置"""
    api: APIConfig = Field(default_factory=APIConfig, description="API 配置")
    download: DownloadConfig = Field(default_factory=DownloadConfig, description="下载配置")
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig, description="处理配置")
    log: LogConfig = Field(default_factory=LogConfig, description="日志配置")
    ai: AIConfig = Field(default_factory=AIConfig, description="AI功能配置")

    # 高级配置
    http_proxy: Optional[str] = Field(None, description="HTTP 代理")
    https_proxy: Optional[str] = Field(None, description="HTTPS 代理")

    class Config:
        """Pydantic 配置"""
        validate_assignment = True
        arbitrary_types_allowed = True