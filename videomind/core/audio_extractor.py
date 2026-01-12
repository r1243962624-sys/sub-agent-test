"""
音频提取模块
从视频文件中提取音频
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

import pydub
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from models.video import VideoInfo
from utils.exceptions import AudioExtractionError, FFmpegError


class AudioExtractor:
    """音频提取器"""

    def __init__(self, temp_dir: Optional[Path] = None, ffmpeg_path: Optional[str] = None):
        """
        初始化音频提取器

        Args:
            temp_dir: 临时目录路径
            ffmpeg_path: FFmpeg可执行文件路径
        """
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "videomind_audio"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # 设置FFmpeg路径
        self.ffmpeg_path = ffmpeg_path or "ffmpeg"
        # 如果未指定，尝试使用已知位置
        if self.ffmpeg_path == "ffmpeg":
            known_paths = [
                "C:\\Users\\WaveI\\ffmpeg-8.0.1-essentials_build\\bin\\ffmpeg.exe",
                "C:\\FFmpeg\\bin\\ffmpeg.exe",
                "C:\\Program Files\\FFmpeg\\bin\\ffmpeg.exe",
            ]
            for path in known_paths:
                if Path(path).exists():
                    self.ffmpeg_path = path
                    break

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((FFmpegError, AudioExtractionError)),
        reraise=True
    )
    def extract_audio(self, video_path: Path, output_format: str = "wav") -> Tuple[Path, float]:
        """
        从视频文件中提取音频

        Args:
            video_path: 视频文件路径
            output_format: 输出音频格式（wav, mp3, flac等）

        Returns:
            Tuple[Path, float]: (音频文件路径, 音频时长（秒）)

        Raises:
            AudioExtractionError: 音频提取失败
            FFmpegError: FFmpeg处理失败
        """
        if not video_path.exists():
            raise AudioExtractionError(f"视频文件不存在: {video_path}")

        logger.info(f"开始提取音频: {video_path}")

        try:
            # 生成输出文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{video_path.stem}_{timestamp}.{output_format}"
            output_path = self.temp_dir / output_filename

            # 使用FFmpeg提取音频
            self._extract_with_ffmpeg(video_path, output_path, output_format)

            # 验证音频文件
            if not output_path.exists():
                raise AudioExtractionError("音频文件未成功创建")

            # 获取音频信息
            audio_info = self._get_audio_info(output_path)

            logger.success(f"音频提取完成: {output_path} (时长: {audio_info['duration']:.2f}秒)")
            return output_path, audio_info["duration"]

        except subprocess.SubprocessError as e:
            logger.error(f"FFmpeg子进程错误: {e}")
            raise FFmpegError(f"FFmpeg处理失败: {str(e)}")
        except Exception as e:
            logger.error(f"音频提取过程中发生未知错误: {e}")
            raise AudioExtractionError(f"音频提取失败: {str(e)}")

    def _extract_with_ffmpeg(self, video_path: Path, output_path: Path, output_format: str):
        """使用FFmpeg提取音频"""
        ffmpeg_cmd = [
            self.ffmpeg_path,
            "-i", str(video_path),  # 输入文件
            "-vn",  # 禁用视频流
            "-acodec", "pcm_s16le",  # PCM 16位小端
            "-ar", "16000",  # 采样率16kHz（Whisper推荐）
            "-ac", "1",  # 单声道
            "-y",  # 覆盖输出文件
            str(output_path)
        ]

        logger.debug(f"执行FFmpeg命令: {' '.join(ffmpeg_cmd)}")

        try:
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() or "未知FFmpeg错误"
                logger.error(f"FFmpeg错误输出: {error_msg}")
                raise FFmpegError(f"FFmpeg处理失败: {error_msg}")

        except subprocess.TimeoutExpired:
            raise FFmpegError("FFmpeg处理超时")

    def _get_audio_info(self, audio_path: Path) -> dict:
        """获取音频文件信息"""
        try:
            audio = pydub.AudioSegment.from_file(str(audio_path))
            return {
                "duration": len(audio) / 1000.0,  # 转换为秒
                "channels": audio.channels,
                "sample_width": audio.sample_width,
                "frame_rate": audio.frame_rate,
                "frame_count": audio.frame_count(),
                "max_dBFS": audio.max_dBFS,
            }
        except Exception as e:
            logger.warning(f"无法获取音频详细信息: {e}")
            # 尝试使用FFmpeg获取基本信息
            return self._get_audio_info_with_ffmpeg(audio_path)

    def _get_audio_info_with_ffmpeg(self, audio_path: Path) -> dict:
        """使用FFmpeg获取音频信息"""
        ffprobe_cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path)
        ]

        try:
            result = subprocess.run(
                ffprobe_cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                duration = float(result.stdout.strip())
                return {"duration": duration}
            else:
                return {"duration": 0.0}

        except Exception:
            return {"duration": 0.0}

    def extract_audio_for_video(self, video_info: VideoInfo) -> VideoInfo:
        """
        为视频信息对象提取音频

        Args:
            video_info: 视频信息对象

        Returns:
            VideoInfo: 更新后的视频信息对象
        """
        if not video_info.video_path:
            raise AudioExtractionError("视频文件路径未设置")

        try:
            audio_path, audio_duration = self.extract_audio(video_info.video_path)
            video_info.audio_path = audio_path

            # 更新音频时长信息
            if not video_info.duration:
                video_info.duration = audio_duration

            return video_info

        except Exception as e:
            logger.error(f"为视频提取音频失败: {e}")
            raise

    def convert_audio_format(self, input_path: Path, output_format: str = "wav") -> Path:
        """
        转换音频格式

        Args:
            input_path: 输入音频文件路径
            output_format: 输出格式

        Returns:
            Path: 转换后的音频文件路径
        """
        if not input_path.exists():
            raise AudioExtractionError(f"输入音频文件不存在: {input_path}")

        logger.info(f"转换音频格式: {input_path} -> {output_format}")

        try:
            # 加载音频
            audio = pydub.AudioSegment.from_file(str(input_path))

            # 生成输出文件名
            output_filename = f"{input_path.stem}_converted.{output_format}"
            output_path = self.temp_dir / output_filename

            # 导出音频
            audio.export(str(output_path), format=output_format)

            logger.success(f"音频格式转换完成: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"音频格式转换失败: {e}")
            raise AudioExtractionError(f"音频格式转换失败: {str(e)}")

    def normalize_audio(self, audio_path: Path, target_dBFS: float = -20.0) -> Path:
        """
        标准化音频音量

        Args:
            audio_path: 音频文件路径
            target_dBFS: 目标音量（分贝）

        Returns:
            Path: 标准化后的音频文件路径
        """
        if not audio_path.exists():
            raise AudioExtractionError(f"音频文件不存在: {audio_path}")

        logger.info(f"标准化音频音量: {audio_path}")

        try:
            # 加载音频
            audio = pydub.AudioSegment.from_file(str(audio_path))

            # 计算当前音量
            current_dBFS = audio.dBFS

            # 计算增益
            gain = target_dBFS - current_dBFS

            # 应用增益
            normalized_audio = audio.apply_gain(gain)

            # 生成输出文件名
            output_filename = f"{audio_path.stem}_normalized.{audio_path.suffix[1:]}"
            output_path = self.temp_dir / output_filename

            # 导出音频
            normalized_audio.export(str(output_path), format=audio_path.suffix[1:])

            logger.success(f"音频标准化完成: {output_path} (增益: {gain:.2f}dB)")
            return output_path

        except Exception as e:
            logger.error(f"音频标准化失败: {e}")
            raise AudioExtractionError(f"音频标准化失败: {str(e)}")

    def split_audio(self, audio_path: Path, segment_duration: int = 600) -> list:
        """
        分割音频文件（用于处理长音频）

        Args:
            audio_path: 音频文件路径
            segment_duration: 每个片段的时长（秒）

        Returns:
            list: 分割后的音频文件路径列表
        """
        if not audio_path.exists():
            raise AudioExtractionError(f"音频文件不存在: {audio_path}")

        logger.info(f"分割音频文件: {audio_path} (每段{segment_duration}秒)")

        try:
            # 加载音频
            audio = pydub.AudioSegment.from_file(str(audio_path))
            total_duration = len(audio) / 1000.0  # 转换为秒

            # 计算分段数量
            num_segments = int(total_duration // segment_duration) + 1

            segments = []
            for i in range(num_segments):
                start_time = i * segment_duration * 1000  # 转换为毫秒
                end_time = min((i + 1) * segment_duration * 1000, len(audio))

                # 提取片段
                segment = audio[start_time:end_time]

                # 生成输出文件名
                output_filename = f"{audio_path.stem}_part{i+1:03d}.{audio_path.suffix[1:]}"
                output_path = self.temp_dir / output_filename

                # 导出片段
                segment.export(str(output_path), format=audio_path.suffix[1:])
                segments.append(output_path)

            logger.success(f"音频分割完成: 共{len(segments)}个片段")
            return segments

        except Exception as e:
            logger.error(f"音频分割失败: {e}")
            raise AudioExtractionError(f"音频分割失败: {str(e)}")

    def cleanup(self, audio_path: Optional[Path] = None):
        """
        清理音频文件

        Args:
            audio_path: 要清理的音频文件路径，如果为None则清理整个临时目录
        """
        try:
            if audio_path and audio_path.exists():
                audio_path.unlink()
                logger.debug(f"已删除音频文件: {audio_path}")
            elif audio_path is None:
                # 清理整个临时目录
                for file in self.temp_dir.glob("*"):
                    try:
                        file.unlink()
                    except Exception:
                        pass
                logger.debug(f"已清理音频临时目录: {self.temp_dir}")

        except Exception as e:
            logger.warning(f"清理音频文件时发生错误: {e}")

    def check_ffmpeg_available(self) -> bool:
        """检查FFmpeg是否可用"""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False