"""
自定义异常类
定义系统专用的异常类型
"""


class VideoMindError(Exception):
    """VideoMind 基础异常类"""
    pass


class ConfigError(VideoMindError):
    """配置错误"""
    pass


class DownloadError(VideoMindError):
    """下载错误"""
    pass


class NetworkError(VideoMindError):
    """网络错误"""
    pass


class AudioExtractionError(VideoMindError):
    """音频提取错误"""
    pass


class TranscriptionError(VideoMindError):
    """转写错误"""
    pass


class LLMError(VideoMindError):
    """大模型错误"""
    pass


class TemplateError(VideoMindError):
    """模板错误"""
    pass


class ValidationError(VideoMindError):
    """验证错误"""
    pass


class StorageError(VideoMindError):
    """存储错误"""
    pass


class ProcessingError(VideoMindError):
    """处理错误"""
    pass


class RetryExhaustedError(VideoMindError):
    """重试耗尽错误"""
    pass


class APIKeyError(VideoMindError):
    """API密钥错误"""
    pass


class ModelNotSupportedError(VideoMindError):
    """模型不支持错误"""
    pass


class RateLimitError(VideoMindError):
    """速率限制错误"""
    pass


class TimeoutError(VideoMindError):
    """超时错误"""
    pass


class FileSystemError(VideoMindError):
    """文件系统错误"""
    pass


class InvalidURLError(VideoMindError):
    """无效URL错误"""
    pass


class UnsupportedPlatformError(VideoMindError):
    """不支持的平台错误"""
    pass


class FFmpegError(VideoMindError):
    """FFmpeg错误"""
    pass


class WhisperError(VideoMindError):
    """Whisper错误"""
    pass