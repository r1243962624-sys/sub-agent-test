"""
验证器模块
验证输入参数和配置
"""

import re
import os
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

from utils.exceptions import InvalidURLError, ValidationError, FileSystemError


def validate_url(url: str) -> Tuple[bool, str]:
    """
    验证URL格式

    Args:
        url: 要验证的URL

    Returns:
        Tuple[bool, str]: (是否有效, 错误信息)
    """
    if not url or not isinstance(url, str):
        return False, "URL不能为空"

    # 基本URL格式验证
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False, "URL格式无效"
    except Exception:
        return False, "URL解析失败"

    # 检查支持的视频平台
    url_lower = url.lower()
    supported_patterns = [
        r"youtube\.com/watch\?v=",
        r"youtu\.be/",
        r"bilibili\.com/video/",
        r"vimeo\.com/",
        r"\.mp4$",
        r"\.mkv$",
        r"\.webm$",
        r"\.avi$",
    ]

    for pattern in supported_patterns:
        if re.search(pattern, url_lower):
            return True, "URL有效"

    return False, "不支持的视频平台或文件格式"


def validate_output_dir(directory: Path, create_if_not_exists: bool = True) -> Tuple[bool, str]:
    """
    验证输出目录

    Args:
        directory: 目录路径
        create_if_not_exists: 如果目录不存在是否创建

    Returns:
        Tuple[bool, str]: (是否有效, 错误信息)
    """
    try:
        directory = Path(directory)

        if directory.exists():
            if not directory.is_dir():
                return False, "路径不是目录"
            # 检查是否可写
            test_file = directory / ".test_write"
            try:
                test_file.touch()
                test_file.unlink()
            except Exception:
                return False, "目录不可写"
        else:
            if create_if_not_exists:
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    return False, f"创建目录失败: {str(e)}"
            else:
                return False, "目录不存在"

        return True, "目录有效"

    except Exception as e:
        return False, f"验证目录失败: {str(e)}"


def validate_api_key(api_key: str, provider: str) -> Tuple[bool, str]:
    """
    验证API密钥格式

    Args:
        api_key: API密钥
        provider: 提供商名称

    Returns:
        Tuple[bool, str]: (是否有效, 错误信息)
    """
    if not api_key or not isinstance(api_key, str):
        return False, "API密钥不能为空"

    api_key = api_key.strip()

    # 根据提供商进行基本格式验证
    if provider.lower() == "openai":
        # OpenAI API密钥通常以'sk-'开头
        if not api_key.startswith("sk-"):
            return False, "OpenAI API密钥应以'sk-'开头"
        if len(api_key) < 20:
            return False, "API密钥长度过短"

    elif provider.lower() == "anthropic":
        # Anthropic API密钥格式
        if len(api_key) < 20:
            return False, "API密钥长度过短"

    elif provider.lower() == "deepseek":
        # DeepSeek API密钥格式
        if len(api_key) < 20:
            return False, "API密钥长度过短"

    else:
        # 其他提供商，只做基本检查
        if len(api_key) < 10:
            return False, "API密钥长度过短"

    return True, "API密钥格式有效"


def validate_file_path(file_path: Path, check_exists: bool = True, check_readable: bool = True) -> Tuple[bool, str]:
    """
    验证文件路径

    Args:
        file_path: 文件路径
        check_exists: 是否检查文件存在
        check_readable: 是否检查文件可读

    Returns:
        Tuple[bool, str]: (是否有效, 错误信息)
    """
    try:
        file_path = Path(file_path)

        if check_exists and not file_path.exists():
            return False, "文件不存在"

        if file_path.exists():
            if not file_path.is_file():
                return False, "路径不是文件"

            if check_readable:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        f.read(1)  # 尝试读取一个字节
                except Exception:
                    return False, "文件不可读"

        return True, "文件路径有效"

    except Exception as e:
        return False, f"验证文件路径失败: {str(e)}"


def validate_video_url(url: str) -> str:
    """
    验证视频URL并返回规范化URL

    Args:
        url: 视频URL

    Returns:
        str: 规范化后的URL

    Raises:
        InvalidURLError: URL无效
    """
    is_valid, error_msg = validate_url(url)
    if not is_valid:
        raise InvalidURLError(error_msg)

    # 规范化URL
    url = url.strip()

    # 处理YouTube短链接
    if "youtu.be/" in url:
        # 将 youtu.be/xxx 转换为 youtube.com/watch?v=xxx
        video_id = url.split("youtu.be/")[-1].split("?")[0]
        url = f"https://www.youtube.com/watch?v={video_id}"

    # 确保使用HTTPS
    if url.startswith("http://"):
        url = url.replace("http://", "https://")

    return url


def validate_config(config_dict: dict) -> Tuple[bool, str, dict]:
    """
    验证配置字典

    Args:
        config_dict: 配置字典

    Returns:
        Tuple[bool, str, dict]: (是否有效, 错误信息, 清理后的配置)
    """
    try:
        # 这里可以添加更复杂的配置验证逻辑
        cleaned_config = {}

        # 验证API配置
        if "api" in config_dict:
            api_config = config_dict["api"]
            cleaned_api = {}

            # 验证模型提供商
            if "model_provider" in api_config:
                provider = api_config["model_provider"]
                valid_providers = ["openai", "anthropic", "deepseek"]
                if provider not in valid_providers:
                    return False, f"不支持的模型提供商: {provider}", {}
                cleaned_api["model_provider"] = provider

            # 验证模型名称
            if "default_model" in api_config:
                model = api_config["default_model"]
                if not model or len(model) < 3:
                    return False, "模型名称无效", {}
                cleaned_api["default_model"] = model

            # 验证API密钥（如果提供）
            if "openai_api_key" in api_config:
                key = api_config["openai_api_key"]
                if key:
                    is_valid, error = validate_api_key(key, "openai")
                    if not is_valid:
                        return False, f"OpenAI API密钥无效: {error}", {}
                    cleaned_api["openai_api_key"] = key

            if "anthropic_api_key" in api_config:
                key = api_config["anthropic_api_key"]
                if key:
                    is_valid, error = validate_api_key(key, "anthropic")
                    if not is_valid:
                        return False, f"Anthropic API密钥无效: {error}", {}
                    cleaned_api["anthropic_api_key"] = key

            if "deepseek_api_key" in api_config:
                key = api_config["deepseek_api_key"]
                if key:
                    is_valid, error = validate_api_key(key, "deepseek")
                    if not is_valid:
                        return False, f"DeepSeek API密钥无效: {error}", {}
                    cleaned_api["deepseek_api_key"] = key

            cleaned_config["api"] = cleaned_api

        # 验证下载配置
        if "download" in config_dict:
            download_config = config_dict["download"]
            cleaned_download = {}

            # 验证输出目录
            if "output_dir" in download_config:
                output_dir = Path(download_config["output_dir"])
                is_valid, error = validate_output_dir(output_dir, create_if_not_exists=False)
                if not is_valid:
                    return False, f"输出目录无效: {error}", {}
                cleaned_download["output_dir"] = str(output_dir)

            # 验证临时目录
            if "temp_dir" in download_config:
                temp_dir = Path(download_config["temp_dir"])
                is_valid, error = validate_output_dir(temp_dir, create_if_not_exists=True)
                if not is_valid:
                    return False, f"临时目录无效: {error}", {}
                cleaned_download["temp_dir"] = str(temp_dir)

            cleaned_config["download"] = cleaned_download

        return True, "配置有效", cleaned_config

    except Exception as e:
        return False, f"配置验证失败: {str(e)}", {}


def validate_template_variables(template_variables: dict, provided_variables: dict) -> Tuple[bool, str, dict]:
    """
    验证模板变量

    Args:
        template_variables: 模板定义的变量
        provided_variables: 提供的变量值

    Returns:
        Tuple[bool, str, dict]: (是否有效, 错误信息, 验证后的变量)
    """
    try:
        validated_vars = {}

        for var_name, var_def in template_variables.items():
            var_name_clean = var_name.strip("{}")

            if var_name_clean in provided_variables:
                value = provided_variables[var_name_clean]

                # 检查必需变量
                if var_def.get("required", False) and value is None:
                    return False, f"必需变量 '{var_name}' 未提供", {}

                # 类型验证（简化版）
                expected_type = var_def.get("type", "str")
                if value is not None:
                    if expected_type == "str" and not isinstance(value, str):
                        return False, f"变量 '{var_name}' 应为字符串类型", {}
                    elif expected_type == "int" and not isinstance(value, int):
                        return False, f"变量 '{var_name}' 应为整数类型", {}
                    elif expected_type == "float" and not isinstance(value, (int, float)):
                        return False, f"变量 '{var_name}' 应为数值类型", {}
                    elif expected_type == "bool" and not isinstance(value, bool):
                        return False, f"变量 '{var_name}' 应为布尔类型", {}

                validated_vars[var_name] = value
            elif var_def.get("required", False):
                return False, f"必需变量 '{var_name}' 未提供", {}
            elif "default" in var_def:
                validated_vars[var_name] = var_def["default"]

        return True, "变量验证通过", validated_vars

    except Exception as e:
        return False, f"变量验证失败: {str(e)}", {}