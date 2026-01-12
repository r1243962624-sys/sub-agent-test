"""
日志配置模块
配置系统的日志记录
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from models.config import LogConfig, LogLevel


def setup_logger(config: Optional[LogConfig] = None) -> logger:
    """
    设置日志记录器

    Args:
        config: 日志配置

    Returns:
        logger: 配置好的日志记录器
    """
    # 移除默认的处理器
    logger.remove()

    if config is None:
        config = LogConfig()

    # 设置日志级别
    level = config.level.value

    # 控制台输出格式
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    # 添加控制台处理器
    logger.add(
        sys.stderr,
        format=console_format,
        level=level,
        colorize=True,
    )

    # 添加文件处理器（如果配置了文件）
    if config.file:
        # 确保日志目录存在
        config.file.parent.mkdir(parents=True, exist_ok=True)

        # 文件输出格式（更详细）
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} - "
            "{message}"
        )

        logger.add(
            str(config.file),
            format=file_format,
            level=level,
            rotation="10 MB",  # 每个文件最大10MB
            retention="30 days",  # 保留30天
            compression="zip",  # 压缩旧日志
            encoding="utf-8",
        )

    # 设置全局日志级别
    logger.level(level)

    return logger


def get_logger(name: str) -> logger:
    """
    获取指定名称的日志记录器

    Args:
        name: 记录器名称

    Returns:
        logger: 日志记录器
    """
    return logger.bind(name=name)


# 全局日志记录器
log = get_logger("videomind")