"""
配置管理器
负责加载、保存和管理系统配置
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from models.config import Config, ModelProvider, WhisperModelSize, LogLevel


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: Optional[Path] = None, env_path: Optional[Path] = None):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径（YAML）
            env_path: 环境变量文件路径（.env）
        """
        self.config_path = config_path or Path("config.yaml")
        self.env_path = env_path or Path(".env")
        self.config: Optional[Config] = None

        # 加载环境变量
        self._load_env_vars()

        # 加载配置文件
        self._load_config()

    def _load_env_vars(self):
        """加载环境变量"""
        # #region agent log
        import json
        import time
        try:
            debug_log_path = Path(__file__).parent.parent.parent / ".cursor" / "debug.log"
            debug_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"config_manager.py:_load_env_vars:36","message":"检查.env文件路径","data":{"env_path":str(self.env_path),"exists":self.env_path.exists()},"timestamp":int(time.time()*1000)}) + "\n")
        except Exception as e:
            import sys
            print(f"[DEBUG] Log write failed: {e}", file=sys.stderr)
        # #endregion
        if self.env_path.exists():
            load_dotenv(self.env_path)
            # #region agent log
            try:
                debug_log_path = Path(__file__).parent.parent.parent / ".cursor" / "debug.log"
                debug_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"config_manager.py:_load_env_vars:40","message":".env文件加载完成","data":{"path":str(self.env_path)},"timestamp":int(time.time()*1000)}) + "\n")
            except Exception as e:
                import sys
                print(f"[DEBUG] Log write failed: {e}", file=sys.stderr)
            # #endregion
        else:
            # 尝试从当前目录和上级目录查找 .env 文件
            load_dotenv()
            # #region agent log
            try:
                debug_log_path = Path(__file__).parent.parent.parent / ".cursor" / "debug.log"
                debug_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"config_manager.py:_load_env_vars:43","message":".env文件不存在，尝试默认位置","data":{},"timestamp":int(time.time()*1000)}) + "\n")
            except Exception as e:
                import sys
                print(f"[DEBUG] Log write failed: {e}", file=sys.stderr)
            # #endregion

    def _load_config(self):
        """加载配置文件"""
        try:
            if self.config_path.exists():
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config_data = yaml.safe_load(f) or {}
            else:
                config_data = {}

            # 从环境变量覆盖配置
            config_data = self._merge_env_vars(config_data)

            # #region agent log
            import json
            import time
            try:
                debug_log_path = Path(__file__).parent.parent.parent / ".cursor" / "debug.log"
                debug_log_path.parent.mkdir(parents=True, exist_ok=True)
                api_config_after = config_data.get("api", {})
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C,D","location":"config_manager.py:_load_config:54","message":"合并后的配置数据","data":{"model_provider":str(api_config_after.get("model_provider")),"default_model":api_config_after.get("default_model")},"timestamp":int(time.time()*1000)}) + "\n")
            except Exception as e:
                import sys
                print(f"[DEBUG] Log write failed: {e}", file=sys.stderr)
            # #endregion

            # 创建配置对象
            self.config = Config(**config_data)
            
            # #region agent log
            try:
                debug_log_path = Path(__file__).parent.parent.parent / ".cursor" / "debug.log"
                debug_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"config_manager.py:_load_config:57","message":"Config对象创建完成","data":{"model_provider":self.config.api.model_provider.value,"default_model":self.config.api.default_model},"timestamp":int(time.time()*1000)}) + "\n")
            except Exception as e:
                import sys
                print(f"[DEBUG] Log write failed: {e}", file=sys.stderr)
            # #endregion

            # 设置代理环境变量
            self._setup_proxy()

        except Exception as e:
            raise ValueError(f"加载配置文件失败: {e}")

    def _merge_env_vars(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """合并环境变量到配置数据"""
        # #region agent log
        import json
        import time
        try:
            debug_log_path = Path(__file__).parent.parent.parent / ".cursor" / "debug.log"
            debug_log_path.parent.mkdir(parents=True, exist_ok=True)
            model_provider_env = os.getenv("MODEL_PROVIDER")
            default_model_env = os.getenv("DEFAULT_MODEL")
            api_config_before = config_data.get("api", {})
            with open(debug_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"config_manager.py:_merge_env_vars:65","message":"环境变量值检查","data":{"MODEL_PROVIDER":model_provider_env,"DEFAULT_MODEL":default_model_env,"config_before":str(api_config_before.get("model_provider"))},"timestamp":int(time.time()*1000)}) + "\n")
        except Exception as e:
            import sys
            print(f"[DEBUG] Log write failed: {e}", file=sys.stderr)
        # #endregion
        # API 配置
        api_config = config_data.get("api", {})
        if os.getenv("OPENAI_API_KEY"):
            api_config["openai_api_key"] = os.getenv("OPENAI_API_KEY")
        if os.getenv("ANTHROPIC_API_KEY"):
            api_config["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY")
        if os.getenv("DEEPSEEK_API_KEY"):
            api_config["deepseek_api_key"] = os.getenv("DEEPSEEK_API_KEY")
        if os.getenv("MODEL_PROVIDER"):
            api_config["model_provider"] = ModelProvider(os.getenv("MODEL_PROVIDER"))
            # #region agent log
            try:
                debug_log_path = Path(__file__).parent.parent.parent / ".cursor" / "debug.log"
                debug_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B,C","location":"config_manager.py:_merge_env_vars:76","message":"MODEL_PROVIDER环境变量合并","data":{"value":os.getenv("MODEL_PROVIDER"),"merged":str(api_config.get("model_provider"))},"timestamp":int(time.time()*1000)}) + "\n")
            except Exception as e:
                import sys
                print(f"[DEBUG] Log write failed: {e}", file=sys.stderr)
            # #endregion
        if os.getenv("DEFAULT_MODEL"):
            api_config["default_model"] = os.getenv("DEFAULT_MODEL")
            # #region agent log
            try:
                debug_log_path = Path(__file__).parent.parent.parent / ".cursor" / "debug.log"
                debug_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B,C","location":"config_manager.py:_merge_env_vars:78","message":"DEFAULT_MODEL环境变量合并","data":{"value":os.getenv("DEFAULT_MODEL"),"merged":api_config.get("default_model")},"timestamp":int(time.time()*1000)}) + "\n")
            except Exception as e:
                import sys
                print(f"[DEBUG] Log write failed: {e}", file=sys.stderr)
            # #endregion

        # 下载配置
        download_config = config_data.get("download", {})
        if os.getenv("OUTPUT_DIR"):
            download_config["output_dir"] = Path(os.getenv("OUTPUT_DIR"))
        if os.getenv("TEMP_DIR"):
            download_config["temp_dir"] = Path(os.getenv("TEMP_DIR"))

        # 处理配置
        processing_config = config_data.get("processing", {})
        if os.getenv("WHISPER_MODEL"):
            processing_config["whisper_model"] = WhisperModelSize(os.getenv("WHISPER_MODEL"))
        if os.getenv("WHISPER_LANGUAGE"):
            processing_config["whisper_language"] = os.getenv("WHISPER_LANGUAGE")
        if os.getenv("KEEP_INTERMEDIATE_FILES"):
            processing_config["keep_intermediate_files"] = os.getenv("KEEP_INTERMEDIATE_FILES").lower() == "true"
        if os.getenv("MAX_RETRIES"):
            processing_config["max_retries"] = int(os.getenv("MAX_RETRIES"))
        if os.getenv("RETRY_DELAY"):
            processing_config["retry_delay"] = int(os.getenv("RETRY_DELAY"))
        if os.getenv("DEFAULT_TEMPLATE"):
            processing_config["default_template"] = os.getenv("DEFAULT_TEMPLATE")

        # 日志配置
        log_config = config_data.get("log", {})
        if os.getenv("LOG_LEVEL"):
            log_config["level"] = LogLevel(os.getenv("LOG_LEVEL"))
        if os.getenv("LOG_FILE"):
            log_config["file"] = Path(os.getenv("LOG_FILE"))

        # AI配置
        ai_config = config_data.get("ai", {})
        if os.getenv("ENABLE_COST_MONITORING"):
            ai_config["enable_cost_monitoring"] = os.getenv("ENABLE_COST_MONITORING").lower() == "true"
        if os.getenv("DAILY_BUDGET"):
            ai_config["daily_budget"] = float(os.getenv("DAILY_BUDGET"))
        if os.getenv("MONTHLY_BUDGET"):
            ai_config["monthly_budget"] = float(os.getenv("MONTHLY_BUDGET"))
        if os.getenv("TOTAL_BUDGET"):
            ai_config["total_budget"] = float(os.getenv("TOTAL_BUDGET"))
        if os.getenv("ENABLE_PROMPT_OPTIMIZATION"):
            ai_config["enable_prompt_optimization"] = os.getenv("ENABLE_PROMPT_OPTIMIZATION").lower() == "true"
        if os.getenv("DEFAULT_OPTIMIZATION_LEVEL"):
            ai_config["default_optimization_level"] = os.getenv("DEFAULT_OPTIMIZATION_LEVEL")
        if os.getenv("MAX_CONCURRENT_BATCH_TASKS"):
            ai_config["max_concurrent_batch_tasks"] = int(os.getenv("MAX_CONCURRENT_BATCH_TASKS"))
        if os.getenv("ENABLE_MODEL_MONITORING"):
            ai_config["enable_model_monitoring"] = os.getenv("ENABLE_MODEL_MONITORING").lower() == "true"

        # 代理配置
        if os.getenv("HTTP_PROXY"):
            config_data["http_proxy"] = os.getenv("HTTP_PROXY")
        if os.getenv("HTTPS_PROXY"):
            config_data["https_proxy"] = os.getenv("HTTPS_PROXY")

        # 更新配置数据
        if api_config:
            config_data["api"] = api_config
        if download_config:
            config_data["download"] = download_config
        if processing_config:
            config_data["processing"] = processing_config
        if log_config:
            config_data["log"] = log_config
        if ai_config:
            config_data["ai"] = ai_config

        return config_data

    def _setup_proxy(self):
        """设置代理环境变量"""
        if self.config and self.config.http_proxy:
            os.environ["HTTP_PROXY"] = self.config.http_proxy
        if self.config and self.config.https_proxy:
            os.environ["HTTPS_PROXY"] = self.config.https_proxy

    def save_config(self, config: Optional[Config] = None):
        """
        保存配置到文件

        Args:
            config: 要保存的配置对象，如果为None则保存当前配置
        """
        if config is None:
            config = self.config

        if config is None:
            raise ValueError("没有可保存的配置")

        # 转换为字典
        config_dict = config.dict(exclude_none=True)

        # 保存到YAML文件
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)

    def get_config(self) -> Config:
        """获取当前配置"""
        if self.config is None:
            raise ValueError("配置未加载")
        return self.config

    def update_config(self, updates: Dict[str, Any]):
        """
        更新配置

        Args:
            updates: 要更新的配置项
        """
        if self.config is None:
            raise ValueError("配置未加载")

        # 递归更新配置
        self._update_dict(self.config.dict(), updates)

        # 重新创建配置对象
        self.config = Config(**self.config.dict())

        # 保存更新后的配置
        self.save_config()

    def _update_dict(self, target: Dict[str, Any], updates: Dict[str, Any]):
        """递归更新字典"""
        for key, value in updates.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_dict(target[key], value)
            else:
                target[key] = value

    def get_api_key(self, provider: ModelProvider) -> Optional[str]:
        """获取指定提供商的API密钥"""
        if self.config is None:
            return None

        if provider == ModelProvider.OPENAI:
            return self.config.api.openai_api_key
        elif provider == ModelProvider.ANTHROPIC:
            return self.config.api.anthropic_api_key
        elif provider == ModelProvider.DEEPSEEK:
            return self.config.api.deepseek_api_key
        else:
            return None

    def validate_config(self) -> bool:
        """验证配置是否有效"""
        if self.config is None:
            return False

        # 检查必要的API密钥
        provider = self.config.api.model_provider
        api_key = self.get_api_key(provider)

        if not api_key:
            return False

        # 检查输出目录是否可写
        try:
            output_dir = self.config.download.output_dir
            test_file = output_dir / ".test_write"
            test_file.touch()
            test_file.unlink()
        except Exception:
            return False

        return True

    def create_default_config(self) -> Config:
        """创建默认配置"""
        default_config = Config()
        self.config = default_config
        self.save_config()
        return default_config


# 全局配置管理器实例
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> Config:
    """获取当前配置"""
    return get_config_manager().get_config()