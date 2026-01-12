"""
ASR转写模块
使用Whisper进行语音转写
"""

import os
import warnings
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

import whisper
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from models.config import WhisperModelSize
from models.video import VideoInfo
from utils.exceptions import TranscriptionError, WhisperError

# 确保FFmpeg在PATH中（Windows）
def _ensure_ffmpeg_in_path():
    """确保FFmpeg在PATH环境变量中"""
    if os.name == 'nt':  # Windows
        possible_paths = [
            r"C:\ProgramData\chocolatey\bin",
            r"C:\ffmpeg\bin",
            r"C:\Program Files\ffmpeg\bin",
        ]
        for path in possible_paths:
            if os.path.exists(path) and path not in os.environ.get("PATH", ""):
                os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
                logger.info(f"Added FFmpeg path to environment: {path}")
                break

# 在模块加载时自动添加FFmpeg路径
_ensure_ffmpeg_in_path()


class Transcriber:
    """语音转写器"""

    def __init__(self, model_size: WhisperModelSize = WhisperModelSize.BASE, language: Optional[str] = None):
        """
        初始化转写器

        Args:
            model_size: Whisper模型大小
            language: 转写语言（None为自动检测）
        """
        self.model_size = model_size
        self.language = language
        self.model: Optional[whisper.Whisper] = None
        self._load_model()

    def _load_model(self):
        """加载Whisper模型"""
        logger.info(f"加载Whisper模型: {self.model_size.value}")

        try:
            # 忽略警告
            warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

            # 加载模型
            self.model = whisper.load_model(self.model_size.value)

            logger.success(f"Whisper模型加载完成: {self.model_size.value}")
        except Exception as e:
            logger.error(f"加载Whisper模型失败: {e}")
            raise WhisperError(f"无法加载Whisper模型: {str(e)}")

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((TranscriptionError, WhisperError)),
        reraise=True
    )
    def transcribe(self, audio_path: Path, language: Optional[str] = None) -> Dict[str, Any]:
        """
        转写音频文件

        Args:
            audio_path: 音频文件路径
            language: 转写语言（None为自动检测）

        Returns:
            Dict[str, Any]: 转写结果

        Raises:
            TranscriptionError: 转写失败
            WhisperError: Whisper错误
        """
        if not audio_path.exists():
            raise TranscriptionError(f"音频文件不存在: {audio_path}")

        if self.model is None:
            raise TranscriptionError("Whisper模型未加载")

        logger.info(f"开始转写音频: {audio_path}")

        try:
            # 设置转写参数
            transcribe_kwargs = {
                "verbose": False,
                "task": "transcribe",
                "fp16": False,  # CPU上禁用FP16
            }

            # 设置语言
            if language:
                transcribe_kwargs["language"] = language
            elif self.language and self.language != "auto":
                transcribe_kwargs["language"] = self.language

            # 执行转写
            logger.debug(f"转写参数: {transcribe_kwargs}")
            result = self.model.transcribe(str(audio_path), **transcribe_kwargs)

            # 验证结果
            if not result or "text" not in result:
                raise TranscriptionError("转写结果为空")

            # 处理结果
            processed_result = self._process_transcription_result(result)

            logger.success(f"音频转写完成: {audio_path}")
            logger.info(f"转写文本长度: {len(processed_result['text'])} 字符")
            if processed_result.get("language"):
                logger.info(f"检测到语言: {processed_result['language']}")

            return processed_result

        except Exception as e:
            logger.error(f"音频转写失败: {e}")
            raise TranscriptionError(f"转写失败: {str(e)}")

    def _process_transcription_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """处理转写结果"""
        processed = {
            "text": result.get("text", "").strip(),
            "language": result.get("language", ""),
            "segments": [],
            "statistics": {}
        }

        # 处理分段信息
        if "segments" in result:
            for segment in result["segments"]:
                processed_segment = {
                    "id": segment.get("id"),
                    "start": segment.get("start", 0.0),
                    "end": segment.get("end", 0.0),
                    "text": segment.get("text", "").strip(),
                    "temperature": segment.get("temperature", 0.0),
                    "avg_logprob": segment.get("avg_logprob", 0.0),
                    "compression_ratio": segment.get("compression_ratio", 0.0),
                    "no_speech_prob": segment.get("no_speech_prob", 0.0),
                }
                processed["segments"].append(processed_segment)

        # 计算统计信息
        text = processed["text"]
        processed["statistics"] = {
            "char_count": len(text),
            "word_count": len(text.split()),
            "segment_count": len(processed["segments"]),
            "total_duration": sum(seg["end"] - seg["start"] for seg in processed["segments"]) if processed["segments"] else 0,
        }

        return processed

    def transcribe_audio_for_video(self, video_info: VideoInfo, language: Optional[str] = None) -> VideoInfo:
        """
        为视频信息对象转写音频

        Args:
            video_info: 视频信息对象
            language: 转写语言

        Returns:
            VideoInfo: 更新后的视频信息对象
        """
        if not video_info.audio_path:
            raise TranscriptionError("音频文件路径未设置")

        try:
            # 执行转写
            result = self.transcribe(video_info.audio_path, language)

            # 保存转写结果到文件
            transcript_path = self._save_transcript(video_info, result)

            # 更新视频信息
            video_info.transcript_path = transcript_path
            video_info.transcript = result["text"]

            return video_info

        except Exception as e:
            logger.error(f"为视频转写音频失败: {e}")
            raise

    def _save_transcript(self, video_info: VideoInfo, result: Dict[str, Any]) -> Path:
        """保存转写结果到文件"""
        try:
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{video_info.get_safe_filename()}_transcript_{timestamp}"

            # 保存完整结果（JSON格式）
            json_path = video_info.audio_path.parent / f"{filename}.json"
            import json
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            # 保存纯文本
            txt_path = video_info.audio_path.parent / f"{filename}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(result["text"])

            logger.debug(f"转写结果已保存: {txt_path}")
            return txt_path

        except Exception as e:
            logger.warning(f"保存转写结果失败: {e}")
            # 创建临时文件
            temp_file = Path(tempfile.mktemp(suffix=".txt"))
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(result["text"])
            return temp_file

    def batch_transcribe(self, audio_paths: List[Path], language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        批量转写音频文件

        Args:
            audio_paths: 音频文件路径列表
            language: 转写语言

        Returns:
            List[Dict[str, Any]]: 转写结果列表
        """
        results = []
        total = len(audio_paths)

        for i, audio_path in enumerate(audio_paths, 1):
            logger.info(f"转写进度: {i}/{total} - {audio_path.name}")

            try:
                result = self.transcribe(audio_path, language)
                results.append({
                    "audio_path": audio_path,
                    "success": True,
                    "result": result,
                    "error": None
                })
            except Exception as e:
                logger.error(f"转写失败 {audio_path}: {e}")
                results.append({
                    "audio_path": audio_path,
                    "success": False,
                    "result": None,
                    "error": str(e)
                })

        return results

    def detect_language(self, audio_path: Path) -> str:
        """
        检测音频语言

        Args:
            audio_path: 音频文件路径

        Returns:
            str: 语言代码
        """
        if not audio_path.exists():
            raise TranscriptionError(f"音频文件不存在: {audio_path}")

        if self.model is None:
            raise TranscriptionError("Whisper模型未加载")

        try:
            # 加载音频并检测语言
            audio = whisper.load_audio(str(audio_path))
            audio = whisper.pad_or_trim(audio)

            # 创建mel频谱图
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

            # 检测语言
            _, probs = self.model.detect_language(mel)
            detected_language = max(probs, key=probs.get)

            logger.info(f"检测到语言: {detected_language} (置信度: {probs[detected_language]:.2%})")
            return detected_language

        except Exception as e:
            logger.error(f"语言检测失败: {e}")
            return "unknown"

    def get_supported_languages(self) -> List[str]:
        """获取支持的语言列表"""
        try:
            # Whisper支持的语言代码
            languages = [
                "en", "zh", "es", "fr", "de", "ja", "ko", "ru", "pt", "it",
                "nl", "tr", "pl", "sv", "fi", "no", "da", "ar", "hi", "th",
                "vi", "he", "el", "id", "ms", "cs", "hu", "ro", "sk", "uk",
                "bg", "hr", "lt", "sl", "et", "lv", "sw", "tl", "af", "bn",
                "ca", "eu", "fa", "gl", "ha", "is", "jw", "km", "lo", "mk",
                "mn", "my", "ne", "pa", "si", "sq", "sr", "su", "ta", "te",
                "ur", "zu"
            ]
            return sorted(languages)
        except Exception:
            return ["en", "zh", "es", "fr", "de", "ja", "ko"]  # 返回主要语言

    def transcribe_with_timestamps(self, audio_path: Path, language: Optional[str] = None) -> List[Tuple[float, float, str]]:
        """
        转写音频并返回带时间戳的文本

        Args:
            audio_path: 音频文件路径
            language: 转写语言

        Returns:
            List[Tuple[float, float, str]]: (开始时间, 结束时间, 文本) 列表
        """
        result = self.transcribe(audio_path, language)

        timestamps = []
        for segment in result.get("segments", []):
            timestamps.append((
                segment.get("start", 0.0),
                segment.get("end", 0.0),
                segment.get("text", "").strip()
            ))

        return timestamps

    def cleanup(self):
        """清理资源"""
        try:
            if self.model is not None:
                # Whisper模型没有显式的清理方法
                # 可以尝试释放GPU内存
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self.model = None
                logger.debug("Whisper模型资源已释放")

        except Exception as e:
            logger.warning(f"清理转写器资源时发生错误: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.model is None:
            return {"loaded": False}

        try:
            import torch
            device = str(self.model.device)
            dtype = str(next(self.model.parameters()).dtype)

            return {
                "loaded": True,
                "model_size": self.model_size.value,
                "device": device,
                "dtype": dtype,
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "language": self.language or "auto",
            }
        except Exception:
            return {
                "loaded": True,
                "model_size": self.model_size.value,
                "language": self.language or "auto",
            }