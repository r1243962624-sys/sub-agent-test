"""
大模型客户端模块
支持多种大模型API提供商
"""

import os
import time
import json
from typing import Optional, Dict, Any, List
from enum import Enum

import openai
import anthropic
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from models.config import ModelProvider, APIConfig
from utils.exceptions import LLMError, APIKeyError, ModelNotSupportedError, RateLimitError


class LLMClient:
    """大模型客户端"""

    def __init__(self, config: APIConfig):
        """
        初始化大模型客户端

        Args:
            config: API配置
        """
        # #region agent log
        import json
        import time
        from pathlib import Path
        try:
            debug_log_path = Path(__file__).parent.parent.parent / ".cursor" / "debug.log"
            debug_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"llm_client.py:__init__:24","message":"LLMClient初始化","data":{"model_provider":config.model_provider.value,"default_model":config.default_model},"timestamp":int(time.time()*1000)}) + "\n")
        except Exception as e:
            import sys
            print(f"[DEBUG] Log write failed: {e}", file=sys.stderr)
        # #endregion
        self.config = config
        self.provider = config.model_provider
        self.api_key = self._get_api_key()
        self._setup_client()

    def _get_api_key(self) -> str:
        """获取API密钥"""
        if self.provider == ModelProvider.OPENAI:
            key = self.config.openai_api_key
        elif self.provider == ModelProvider.ANTHROPIC:
            key = self.config.anthropic_api_key
        elif self.provider == ModelProvider.DEEPSEEK:
            key = self.config.deepseek_api_key
        elif self.provider == ModelProvider.GOOGLE:
            key = self.config.google_api_key or self.config.openai_api_key  # 优先使用google_api_key，其次使用openai_api_key
        else:
            raise ModelNotSupportedError(f"不支持的模型提供商: {self.provider}")

        if not key:
            raise APIKeyError(f"{self.provider.value} API密钥未设置")

        return key

    def _setup_client(self):
        """设置客户端"""
        if self.provider == ModelProvider.OPENAI:
            # 使用老张API (laozhang.ai) 作为OpenAI兼容接口
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://api.laozhang.ai/v1"
            )
        elif self.provider == ModelProvider.ANTHROPIC:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        elif self.provider == ModelProvider.DEEPSEEK:
            # DeepSeek使用OpenAI兼容的API
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )
        elif self.provider == ModelProvider.GOOGLE:
            # Google Gemini通过老张API (laozhang.ai) 作为OpenAI兼容接口
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://api.laozhang.ai/v1"
            )
        else:
            raise ModelNotSupportedError(f"不支持的模型提供商: {self.provider}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, LLMError)),
        reraise=True
    )
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        生成文本

        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            **kwargs: 其他参数（如temperature, max_tokens等）

        Returns:
            str: 生成的文本

        Raises:
            LLMError: 生成失败
            RateLimitError: 速率限制
        """
        logger.info(f"使用 {self.provider.value} 生成文本 (模型: {self.config.default_model})")

        try:
            # 合并参数
            params = {
                "model": self.config.default_model,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            }
            # 只有当max_tokens不为None时才添加
            if self.config.max_tokens is not None:
                params["max_tokens"] = self.config.max_tokens
            params.update(kwargs)
            
            # #region agent log
            import json
            import time
            from pathlib import Path
            try:
                debug_log_path = Path(__file__).parent.parent.parent / ".cursor" / "debug.log"
                debug_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"llm_client.py:generate:124","message":"参数合并后","data":{"max_tokens":params.get("max_tokens"),"config_max_tokens":self.config.max_tokens,"kwargs_max_tokens":kwargs.get("max_tokens")},"timestamp":int(time.time()*1000)}) + "\n")
            except Exception as e:
                import sys
                print(f"[DEBUG] Log write failed: {e}", file=sys.stderr)
            # #endregion

            if self.provider == ModelProvider.OPENAI:
                return self._generate_openai(prompt, system_prompt, **params)
            elif self.provider == ModelProvider.ANTHROPIC:
                return self._generate_anthropic(prompt, system_prompt, **params)
            elif self.provider == ModelProvider.DEEPSEEK:
                return self._generate_deepseek(prompt, system_prompt, **params)
            elif self.provider == ModelProvider.GOOGLE:
                return self._generate_google(prompt, system_prompt, **params)
            else:
                raise ModelNotSupportedError(f"不支持的模型提供商: {self.provider}")

        except openai.RateLimitError as e:
            logger.error(f"OpenAI速率限制: {e}")
            raise RateLimitError(f"API速率限制: {str(e)}")
        except anthropic.RateLimitError as e:
            logger.error(f"Anthropic速率限制: {e}")
            raise RateLimitError(f"API速率限制: {str(e)}")
        except openai.AuthenticationError as e:
            logger.error(f"API认证失败: {e}")
            raise APIKeyError(f"API密钥无效: {str(e)}")
        except Exception as e:
            logger.error(f"文本生成失败: {e}")
            raise LLMError(f"文本生成失败: {str(e)}")

    def _generate_openai(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """使用OpenAI生成文本"""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        # #region agent log
        import json
        import time
        from pathlib import Path
        try:
            debug_log_path = Path(__file__).parent.parent.parent / ".cursor" / "debug.log"
            debug_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"llm_client.py:_generate_openai:159","message":"调用API前","data":{"max_tokens":kwargs.get("max_tokens"),"prompt_length":len(prompt),"system_prompt_length":len(system_prompt) if system_prompt else 0},"timestamp":int(time.time()*1000)}) + "\n")
        except Exception as e:
            import sys
            print(f"[DEBUG] Log write failed: {e}", file=sys.stderr)
        # #endregion
        
        response = self.client.chat.completions.create(
            messages=messages,
            **kwargs
        )

        # #region agent log
        try:
            finish_reason = response.choices[0].finish_reason if hasattr(response.choices[0], 'finish_reason') else None
            content_length = len(response.choices[0].message.content) if response.choices[0].message.content else 0
            with open(debug_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"llm_client.py:_generate_openai:164","message":"API响应后","data":{"finish_reason":finish_reason,"content_length":content_length,"max_tokens":kwargs.get("max_tokens")},"timestamp":int(time.time()*1000)}) + "\n")
        except Exception as e:
            import sys
            print(f"[DEBUG] Log write failed: {e}", file=sys.stderr)
        # #endregion

        # 检查finish_reason，如果是length则记录警告
        if hasattr(response.choices[0], 'finish_reason') and response.choices[0].finish_reason == "length":
            logger.warning(f"生成内容因max_tokens限制被截断 (max_tokens={kwargs.get('max_tokens')}, content_length={len(response.choices[0].message.content)})")
            logger.warning(f"建议增加max_tokens参数或使用续写功能")

        return response.choices[0].message.content

    def _generate_anthropic(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """使用Anthropic生成文本"""
        # Anthropic API参数调整
        anthropic_params = kwargs.copy()

        # 重命名参数
        if "max_tokens" in anthropic_params:
            anthropic_params["max_tokens_to_sample"] = anthropic_params.pop("max_tokens")

        # 构建消息
        messages = [{"role": "user", "content": prompt}]

        # #region agent log
        import json
        import time
        from pathlib import Path
        try:
            debug_log_path = Path(__file__).parent.parent.parent / ".cursor" / "debug.log"
            debug_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"llm_client.py:_generate_anthropic:217","message":"调用Anthropic API前","data":{"max_tokens_to_sample":anthropic_params.get("max_tokens_to_sample"),"prompt_length":len(prompt)},"timestamp":int(time.time()*1000)}) + "\n")
        except Exception as e:
            import sys
            print(f"[DEBUG] Log write failed: {e}", file=sys.stderr)
        # #endregion
        
        response = self.client.messages.create(
            messages=messages,
            system=system_prompt,
            **anthropic_params
        )

        # #region agent log
        try:
            stop_reason = response.stop_reason if hasattr(response, 'stop_reason') else None
            content_length = len(response.content[0].text) if response.content and len(response.content) > 0 else 0
            with open(debug_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"llm_client.py:_generate_anthropic:223","message":"Anthropic API响应后","data":{"stop_reason":stop_reason,"content_length":content_length},"timestamp":int(time.time()*1000)}) + "\n")
        except Exception as e:
            import sys
            print(f"[DEBUG] Log write failed: {e}", file=sys.stderr)
        # #endregion

        return response.content[0].text

    def _generate_deepseek(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """使用DeepSeek生成文本"""
        # DeepSeek使用OpenAI兼容的API
        return self._generate_openai(prompt, system_prompt, **kwargs)

    def _generate_google(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """使用Google生成文本"""
        # Google通过老张API使用OpenAI兼容的API
        return self._generate_openai(prompt, system_prompt, **kwargs)

    def generate_with_template(self, template, variables: Dict[str, Any]) -> str:
        """
        使用模板生成文本

        Args:
            template: 模板对象（Template）或模板字典
            variables: 模板变量

        Returns:
            str: 生成的文本
        """
        try:
            from models.template import Template
            
            # 如果传入的是字典，转换为Template对象
            if isinstance(template, dict):
                template = Template.from_dict(template)
            
            # 渲染模板
            from core.template_engine import TemplateEngine
            engine = TemplateEngine()
            rendered_prompt = engine.render_template(template, variables)

            # 获取系统提示
            system_prompt = template.system_prompt

            # 获取模型参数（支持新旧字段名）
            model_config = template.model_parameters or {}
            
            # #region agent log
            import json
            import time
            from pathlib import Path
            try:
                debug_log_path = Path(__file__).parent.parent.parent / ".cursor" / "debug.log"
                debug_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"llm_client.py:generate_with_template:223","message":"模板参数","data":{"model_config":model_config,"template_name":template.name if hasattr(template, 'name') else 'unknown'},"timestamp":int(time.time()*1000)}) + "\n")
            except Exception as e:
                import sys
                print(f"[DEBUG] Log write failed: {e}", file=sys.stderr)
            # #endregion

            # 生成文本
            result = self.generate(
                prompt=rendered_prompt,
                system_prompt=system_prompt,
                **model_config
            )
            
            # #region agent log
            try:
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"llm_client.py:generate_with_template:240","message":"生成结果长度","data":{"result_length":len(result) if result else 0},"timestamp":int(time.time()*1000)}) + "\n")
            except Exception as e:
                import sys
                print(f"[DEBUG] Log write failed: {e}", file=sys.stderr)
            # #endregion
            
            return result

        except Exception as e:
            logger.error(f"使用模板生成文本失败: {e}")
            raise LLMError(f"模板生成失败: {str(e)}")

    def stream_generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        """
        流式生成文本

        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            **kwargs: 其他参数

        Yields:
            str: 生成的文本片段
        """
        logger.info(f"流式生成文本 (模型: {self.config.default_model})")

        try:
            # 合并参数
            params = {
                "model": self.config.default_model,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "stream": True,
            }
            # 只有当max_tokens不为None时才添加
            if self.config.max_tokens is not None:
                params["max_tokens"] = self.config.max_tokens
            params.update(kwargs)

            if self.provider == ModelProvider.OPENAI:
                yield from self._stream_generate_openai(prompt, system_prompt, **params)
            elif self.provider == ModelProvider.ANTHROPIC:
                yield from self._stream_generate_anthropic(prompt, system_prompt, **params)
            elif self.provider == ModelProvider.DEEPSEEK:
                yield from self._stream_generate_deepseek(prompt, system_prompt, **params)
            elif self.provider == ModelProvider.GOOGLE:
                yield from self._stream_generate_google(prompt, system_prompt, **params)
            else:
                raise ModelNotSupportedError(f"不支持的模型提供商: {self.provider}")

        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            raise LLMError(f"流式生成失败: {str(e)}")

    def _stream_generate_openai(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        """OpenAI流式生成"""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat.completions.create(
            messages=messages,
            **kwargs
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _stream_generate_anthropic(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        """Anthropic流式生成"""
        # Anthropic API参数调整
        anthropic_params = kwargs.copy()
        anthropic_params.pop("stream", None)  # Anthropic不需要stream参数

        if "max_tokens" in anthropic_params:
            anthropic_params["max_tokens_to_sample"] = anthropic_params.pop("max_tokens")

        # 构建消息
        messages = [{"role": "user", "content": prompt}]

        with self.client.messages.stream(
            messages=messages,
            system=system_prompt,
            **anthropic_params
        ) as stream:
            for text in stream.text_stream:
                yield text

    def _stream_generate_deepseek(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        """DeepSeek流式生成"""
        # DeepSeek使用OpenAI兼容的API
        yield from self._stream_generate_openai(prompt, system_prompt, **kwargs)

    def _stream_generate_google(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        """Google流式生成"""
        # Google通过老张API使用OpenAI兼容的API
        yield from self._stream_generate_openai(prompt, system_prompt, **kwargs)

    def get_available_models(self) -> List[str]:
        """获取可用的模型列表"""
        try:
            if self.provider == ModelProvider.OPENAI:
                models = self.client.models.list()
                return [model.id for model in models.data]
            elif self.provider == ModelProvider.ANTHROPIC:
                # Anthropic目前只有几个模型
                return [
                    "claude-3-opus-20240229",
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307",
                ]
            elif self.provider == ModelProvider.DEEPSEEK:
                return [
                    "deepseek-chat",
                    "deepseek-coder",
                ]
            elif self.provider == ModelProvider.GOOGLE:
                # Google Gemini模型列表（通过老张API）
                return [
                    "gemini-3-flash-preview-thinking",
                    "gemini-2.5-pro",
                    "gemini-2.0-flash-exp",
                    "gemini-2.0-flash-thinking-exp",
                ]
            else:
                return []
        except Exception as e:
            logger.warning(f"获取模型列表失败: {e}")
            return []

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """获取模型信息"""
        try:
            if self.provider == ModelProvider.OPENAI:
                # OpenAI没有直接的模型信息API
                return {
                    "provider": "openai",
                    "model": model_name,
                    "supports_streaming": True,
                    "max_tokens": 4096 if "gpt-3.5" in model_name else 8192 if "gpt-4" in model_name else 4096,
                }
            elif self.provider == ModelProvider.ANTHROPIC:
                return {
                    "provider": "anthropic",
                    "model": model_name,
                    "supports_streaming": True,
                    "max_tokens": 4096,
                }
            elif self.provider == ModelProvider.DEEPSEEK:
                return {
                    "provider": "deepseek",
                    "model": model_name,
                    "supports_streaming": True,
                    "max_tokens": 4096,
                }
            elif self.provider == ModelProvider.GOOGLE:
                return {
                    "provider": "google",
                    "model": model_name,
                    "supports_streaming": True,
                    "max_tokens": 8192 if "gemini-2.5" in model_name else 4096,
                }
            else:
                return {}
        except Exception:
            return {}

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int, model: Optional[str] = None) -> float:
        """估算API调用成本"""
        model = model or self.config.default_model

        # 价格表（美元/1000 tokens）
        pricing = {
            # OpenAI
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
            "gpt-4.1-mini-2025-04-14": {"input": 0.005, "output": 0.015},
            # Anthropic
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
            # DeepSeek
            "deepseek-chat": {"input": 0.00014, "output": 0.00028},
            "deepseek-coder": {"input": 0.00014, "output": 0.00028},
            # Google
            "gemini-3-flash-preview-thinking": {"input": 0.002, "output": 0.006},
            "gemini-2.5-pro": {"input": 0.005, "output": 0.015},
            "gemini-2.0-flash-exp": {"input": 0.001, "output": 0.003},
            "gemini-2.0-flash-thinking-exp": {"input": 0.0015, "output": 0.0045},
        }

        if model not in pricing:
            logger.warning(f"未知模型价格: {model}")
            return 0.0

        price = pricing[model]
        cost = (prompt_tokens / 1000 * price["input"]) + (completion_tokens / 1000 * price["output"])
        return cost

    def test_connection(self) -> bool:
        """测试API连接"""
        try:
            # 尝试一个简单的请求
            test_prompt = "Hello, this is a connection test. Please respond with 'OK'."
            response = self.generate(test_prompt, max_tokens=10)

            if response and "OK" in response.upper():
                logger.success(f"{self.provider.value} API连接测试成功")
                return True
            else:
                logger.warning(f"{self.provider.value} API连接测试返回异常响应")
                return False

        except Exception as e:
            logger.error(f"{self.provider.value} API连接测试失败: {e}")
            return False

    def cleanup(self):
        """清理资源"""
        try:
            # 目前没有需要显式清理的资源
            pass
        except Exception as e:
            logger.warning(f"清理LLM客户端资源时发生错误: {e}")