"""
视频下载模块
使用 yt-dlp 下载视频
"""

import os
import tempfile
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import yt_dlp
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from models.video import VideoInfo, VideoPlatform
from models.config import DownloadConfig
from utils.exceptions import DownloadError, NetworkError


class VideoDownloader:
    """视频下载器"""

    def __init__(self, config: DownloadConfig):
        """
        初始化视频下载器

        Args:
            config: 下载配置
        """
        self.config = config
        self.ydl_opts = self._build_ydl_options()

    def _build_ydl_options(self, format_selector: Optional[str] = None, platform: Optional[VideoPlatform] = None) -> Dict[str, Any]:
        """构建 yt-dlp 选项
        
        Args:
            format_selector: 格式选择器，如果为None则不指定格式（让yt-dlp自动选择）
            platform: 视频平台，用于特殊配置
        """
        base_opts = {
            # 输出配置
            "outtmpl": str(self.config.temp_dir / "%(title)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "ignoreerrors": False,
            "no_color": True,

            # 元数据提取
            "writethumbnail": False,
            "writeinfojson": True,
            "writesubtitles": False,
            "writeautomaticsub": False,

            # 网络配置
            "socket_timeout": self.config.download_timeout,
            "retries": 10,
            "fragment_retries": 10,
            "skip_unavailable_fragments": True,
        }

        # Bilibili特殊处理：完全不设置format选项，让yt-dlp使用默认行为
        if platform == VideoPlatform.BILIBILI:
            # 对于Bilibili，不设置format选项，让yt-dlp自动选择并合并
            # 只设置merge_output_format确保输出为mp4
            base_opts["merge_output_format"] = "mp4"
            # 不设置format选项，让yt-dlp使用默认的格式选择逻辑
        else:
            # 其他平台的标准处理
            if format_selector is not None:
                base_opts["format"] = format_selector
                base_opts["merge_output_format"] = "mp4"
            else:
                # 不指定format时，仍然设置merge_output_format以确保合并为mp4
                base_opts["merge_output_format"] = "mp4"

        # 合并自定义选项（但Bilibili要移除format选项）
        if self.config.ytdlp_options:
            custom_opts = self.config.ytdlp_options.copy()
            # 对于Bilibili，移除自定义选项中的format，避免冲突
            if platform == VideoPlatform.BILIBILI and "format" in custom_opts:
                logger.debug("移除Bilibili自定义选项中的format设置，使用默认格式选择")
                custom_opts.pop("format")
            base_opts.update(custom_opts)

        # 限制下载速度
        if self.config.max_download_speed > 0:
            base_opts["ratelimit"] = self.config.max_download_speed * 1024  # 转换为字节

        return base_opts

    def download(self, url: str) -> VideoInfo:
        """
        下载视频

        Args:
            url: 视频URL

        Returns:
            VideoInfo: 视频信息对象

        Raises:
            DownloadError: 下载失败
            NetworkError: 网络错误
        """
        logger.info(f"开始下载视频: {url}")

        # 先提取视频信息，判断平台
        try:
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
                info_dict = ydl.extract_info(url, download=False)
                video_info = self._extract_video_info(info_dict)
                platform = video_info.platform
                logger.info(f"检测到视频平台: {platform}, 标题: {video_info.title}")
        except Exception as e:
            logger.warning(f"预提取视频信息失败，将使用默认格式选择器: {e}")
            platform = None

        # 根据平台选择合适的格式选择器列表
        if platform == VideoPlatform.BILIBILI:
            # Bilibili完全不设置format选项，让yt-dlp使用默认行为
            # 这样yt-dlp会自动选择最佳格式并合并视频和音频
            format_selectors = [
                None,  # 让yt-dlp自动选择（不设置format选项）
            ]
            logger.info("检测到Bilibili视频，使用yt-dlp默认格式选择（自动合并视频和音频）")
        else:
            # 其他平台使用标准格式选择器
            format_selectors = [
                "best[height<=720]/best",  # 首选：720p以下
                "best/bestvideo+bestaudio",  # 备选1：最佳格式或分离的视频+音频
                "best",  # 备选2：最佳可用格式
                None,  # 备选3：让yt-dlp自动选择
            ]

        last_error = None
        for attempt, format_selector in enumerate(format_selectors, 1):
            try:
                logger.debug(f"尝试下载（第{attempt}次），格式选择器: {format_selector or '自动'}")
                
                # 构建选项
                ydl_opts = self._build_ydl_options(format_selector, platform)
                ydl_opts["outtmpl"] = str(self.config.temp_dir / "%(id)s.%(ext)s")
                ydl_opts["writedescription"] = True

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # 提取视频信息（如果之前没有提取）
                    if platform is None:
                        info_dict = ydl.extract_info(url, download=False)
                        video_info = self._extract_video_info(info_dict)
                    else:
                        # 重新提取以确保信息完整
                        info_dict = ydl.extract_info(url, download=False)
                        video_info = self._extract_video_info(info_dict)

                    # 下载视频
                    logger.info(f"下载视频: {video_info.title}")
                    result = ydl.download([url])

                    if result != 0:
                        raise DownloadError(f"下载失败，返回码: {result}")

                    # 查找下载的文件
                    video_file = self._find_downloaded_file(video_info, ydl_opts["outtmpl"])
                    if not video_file or not video_file.exists():
                        raise DownloadError("无法找到下载的视频文件")

                    video_info.video_path = video_file

                    # 保存元数据
                    self._save_metadata(video_info, info_dict)

                    logger.success(f"视频下载完成: {video_info.title}")
                    return video_info

            except yt_dlp.utils.DownloadError as e:
                error_msg = str(e)
                last_error = e
                logger.warning(f"下载尝试 {attempt} 失败: {error_msg}")
                
                # 如果是格式相关的错误，继续尝试下一个格式选择器
                if "format" in error_msg.lower() or "not available" in error_msg.lower():
                    if attempt < len(format_selectors):
                        logger.info(f"格式选择失败，尝试下一个格式选择器...")
                        continue
                else:
                    # 其他类型的错误，直接抛出
                    logger.error(f"yt-dlp 下载错误: {e}")
                    raise DownloadError(f"视频下载失败: {str(e)}")
                    
            except yt_dlp.utils.ExtractorError as e:
                logger.error(f"提取器错误: {e}")
                raise DownloadError(f"无法提取视频信息: {str(e)}")
            except DownloadError as e:
                # 如果是我们自己的DownloadError，且不是格式问题，直接抛出
                if attempt < len(format_selectors):
                    logger.warning(f"下载失败，尝试下一个格式选择器...")
                    continue
                raise
            except Exception as e:
                logger.error(f"下载过程中发生未知错误: {e}")
                if attempt < len(format_selectors):
                    logger.warning(f"发生错误，尝试下一个格式选择器...")
                    continue
                raise DownloadError(f"下载失败: {str(e)}")

        # 所有格式选择器都失败了
        if last_error:
            logger.error(f"所有格式选择器都失败，最后错误: {last_error}")
            raise DownloadError(f"视频下载失败: {str(last_error)}")
        else:
            raise DownloadError("视频下载失败: 未知错误")

    def _extract_video_info(self, info_dict: Dict[str, Any]) -> VideoInfo:
        """从信息字典中提取视频信息"""
        # 基本视频信息
        video_info = VideoInfo(
            url=info_dict.get("webpage_url", info_dict.get("url", "")),
            title=info_dict.get("title", "未知标题"),
            duration=info_dict.get("duration"),
            description=info_dict.get("description"),
            thumbnail_url=info_dict.get("thumbnail"),
            channel=info_dict.get("uploader"),
            metadata=info_dict
        )

        # 提取上传日期
        upload_date = info_dict.get("upload_date")
        if upload_date:
            try:
                video_info.upload_date = datetime.strptime(upload_date, "%Y%m%d")
            except ValueError:
                pass

        # 提取平台信息
        extractor = info_dict.get("extractor_key", "").lower()
        if "youtube" in extractor:
            video_info.platform = VideoPlatform.YOUTUBE
        elif "bilibili" in extractor:
            video_info.platform = VideoPlatform.BILIBILI
        elif "vimeo" in extractor:
            video_info.platform = VideoPlatform.VIMEO

        return video_info

    def _find_downloaded_file(self, video_info: VideoInfo, outtmpl: str) -> Optional[Path]:
        """查找下载的视频文件"""
        # 确保outtmpl是字符串
        if not isinstance(outtmpl, str):
            outtmpl = str(outtmpl)

        # 确保video_id是字符串
        video_id_raw = video_info.metadata.get("id", "")
        if isinstance(video_id_raw, dict):
            # 如果是字典，尝试获取id字段或转换为字符串
            video_id = video_id_raw.get("id", str(video_id_raw))
        else:
            video_id = str(video_id_raw)
        # 首先尝试最常见的路径：视频ID + .mp4
        common_path = self.config.temp_dir / f"{video_id}.mp4"
        if common_path.exists():
            return common_path

        possible_extensions = [".mp4", ".mkv", ".webm", ".flv", ".avi"]

        for ext in possible_extensions:
            # 使用ID作为文件名
            filename = outtmpl.replace("%(id)s", video_id).replace("%(ext)s", ext[1:])
            file_path = Path(filename)

            # 检查文件是否存在
            if file_path.exists():
                return file_path

            # 尝试使用标题作为文件名
            if video_info.title:
                safe_title = video_info.get_safe_filename()
                filename = str(self.config.temp_dir / f"{safe_title}{ext}")
                file_path = Path(filename)
                if file_path.exists():
                    return file_path

        # 在临时目录中查找最近创建的文件
        temp_files = list(self.config.temp_dir.glob("*"))
        if temp_files:
            # 按修改时间排序，获取最新的文件
            latest_file = max(temp_files, key=lambda x: x.stat().st_mtime)
            # 检查文件大小（避免选择太小的文件）
            if latest_file.stat().st_size > 1024 * 1024:  # 大于1MB
                return latest_file

        return None

    def _save_metadata(self, video_info: VideoInfo, info_dict: Dict[str, Any]):
        """保存视频元数据到JSON文件"""
        try:
            metadata_file = video_info.video_path.with_suffix(".json")
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(info_dict, f, ensure_ascii=False, indent=2, default=str)

            logger.debug(f"元数据已保存到: {metadata_file}")
        except Exception as e:
            logger.warning(f"保存元数据失败: {e}")

    def get_video_info(self, url: str) -> VideoInfo:
        """
        获取视频信息（不下载）

        Args:
            url: 视频URL

        Returns:
            VideoInfo: 视频信息对象
        """
        logger.info(f"获取视频信息: {url}")

        try:
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
                info_dict = ydl.extract_info(url, download=False)
                video_info = self._extract_video_info(info_dict)
                return video_info

        except yt_dlp.utils.DownloadError as e:
            logger.error(f"获取视频信息失败: {e}")
            raise DownloadError(f"无法获取视频信息: {str(e)}")
        except Exception as e:
            logger.error(f"获取视频信息时发生未知错误: {e}")
            raise DownloadError(f"获取视频信息失败: {str(e)}")

    def cleanup(self, video_info: VideoInfo):
        """
        清理下载的文件

        Args:
            video_info: 视频信息对象
        """
        try:
            if video_info.video_path and video_info.video_path.exists():
                video_info.video_path.unlink()
                logger.debug(f"已删除视频文件: {video_info.video_path}")

            # 删除元数据文件
            metadata_file = video_info.video_path.with_suffix(".json")
            if metadata_file.exists():
                metadata_file.unlink()
                logger.debug(f"已删除元数据文件: {metadata_file}")

        except Exception as e:
            logger.warning(f"清理文件时发生错误: {e}")

    def check_ffmpeg(self) -> bool:
        """检查FFmpeg是否可用"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def get_supported_sites(self) -> list:
        """获取支持的网站列表"""
        try:
            with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                extractors = ydl._ies  # 获取所有提取器
                sites = [ie.IE_NAME for ie in extractors.values()]
                return sorted(sites)
        except Exception:
            return ["youtube", "bilibili", "vimeo"]  # 返回常见站点