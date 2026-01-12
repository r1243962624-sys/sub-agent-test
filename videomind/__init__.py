"""
VideoMind - 自动化视频内容处理系统
"""

__version__ = "0.1.0"
__author__ = "VideoMind Team"
__email__ = "contact@videomind.ai"

# 导出主要模块
from core.processor import VideoProcessor
from cli.main import app

__all__ = [
    "VideoProcessor",
    "app",
    "__version__",
    "__author__",
    "__email__",
]