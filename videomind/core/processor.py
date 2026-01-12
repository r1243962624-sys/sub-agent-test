"""
视频处理器模块
协调各个模块完成完整的视频处理流程
"""

import concurrent.futures
from pathlib import Path
from typing import Optional, List, Callable, Dict, Any
from datetime import datetime

from loguru import logger
from tqdm import tqdm

from models.config import Config
from models.video import ProcessingResult, VideoInfo, ProcessingStatus, ProcessingStep
from core.downloader import VideoDownloader
from core.audio_extractor import AudioExtractor
from core.transcriber import Transcriber
from core.llm_client import LLMClient
from core.template_engine import TemplateEngine
from storage.cache_manager import CacheManager
from storage.result_storage import ResultStorage
from utils.exceptions import VideoMindError


class VideoProcessor:
    """视频处理器"""

    def __init__(self, config: Config):
        """
        初始化视频处理器

        Args:
            config: 系统配置
        """
        self.config = config

        # 初始化各个模块
        self.downloader = VideoDownloader(config.download)
        self.audio_extractor = AudioExtractor(config.download.temp_dir)
        self.transcriber = Transcriber(
            model_size=config.processing.whisper_model,
            language=config.processing.whisper_language
        )
        self.llm_client = LLMClient(config.api)
        self.template_engine = TemplateEngine()

        # 初始化存储模块
        self.cache_manager = CacheManager()
        self.result_storage = ResultStorage(config.download.output_dir)

        # 验证系统状态
        self._validate_system()

    def _validate_system(self):
        """验证系统状态"""
        # 检查FFmpeg
        if not self.audio_extractor.check_ffmpeg_available():
            logger.warning("FFmpeg不可用，音频提取可能失败")

        # 检查API连接
        if not self.llm_client.test_connection():
            logger.warning("API连接测试失败，大模型功能可能不可用")

    def process_video(
        self,
        url: str,
        template_name: str = "study_notes",
        use_cache: bool = True,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> ProcessingResult:
        """
        处理单个视频

        Args:
            url: 视频URL
            template_name: 使用的模板名称
            use_cache: 是否使用缓存
            progress_callback: 进度回调函数

        Returns:
            ProcessingResult: 处理结果
        """
        # 创建处理结果对象
        result = ProcessingResult(video_info=VideoInfo(url=url))
        result.start_processing()

        try:
            # 步骤1: 检查缓存
            if use_cache:
                cached_result = self.cache_manager.get_cached_processing_result(url)
                if cached_result:
                    logger.info(f"使用缓存结果: {url}")
                    # 将字典转换为ProcessingResult对象
                    if isinstance(cached_result, dict):
                        try:
                            # Pydantic v2使用model_validate
                            result = ProcessingResult.model_validate(cached_result)
                        except Exception as e:
                            logger.warning(f"缓存结果转换失败，创建新对象: {e}")
                            # 如果转换失败，创建一个新的ProcessingResult对象
                            video_info_dict = cached_result.get("video_info", {"url": url})
                            if isinstance(video_info_dict, dict):
                                video_info = VideoInfo.model_validate(video_info_dict)
                            else:
                                video_info = VideoInfo(url=url)
                            result = ProcessingResult(video_info=video_info)
                    else:
                        result = cached_result
                    result.status = ProcessingStatus.COMPLETED
                    
                    # 如果缓存结果有structured_notes，也保存文件
                    if result.structured_notes:
                        try:
                            # 保存文件
                            self.result_storage.save_result(result, format="json")
                            self.result_storage.save_result(result, format="markdown")
                            
                            # 设置output_path
                            if not result.video_info.output_path:
                                # 查找最新生成的markdown文件
                                markdown_files = list(self.result_storage.markdown_dir.glob("*.md"))
                                if markdown_files:
                                    result.video_info.output_path = sorted(markdown_files, key=lambda p: p.stat().st_mtime)[-1]
                        except Exception as e:
                            logger.warning(f"保存缓存结果失败: {e}")
                    
                    # 清除该URL的缓存（确保下次处理时重新生成，不使用缓存）
                    try:
                        self.cache_manager.clear_cache_for_url(url)
                    except Exception as e:
                        logger.warning(f"清除缓存失败: {e}")
                    
                    if progress_callback:
                        progress_callback(100)
                    return result

            # 步骤2: 下载视频
            download_step = result.add_step("download")
            try:
                video_info = self.downloader.download(url)
                result.video_info = video_info
                download_step.complete({
                    "video_path": str(video_info.video_path),
                    "duration": video_info.duration
                })
                if progress_callback:
                    progress_callback(20)
            except Exception as e:
                download_step.fail(str(e))
                result.fail_processing(f"视频下载失败: {str(e)}")
                return result

            # 步骤3: 提取音频
            audio_step = result.add_step("audio_extraction")
            try:
                video_info = self.audio_extractor.extract_audio_for_video(video_info)
                result.video_info = video_info
                audio_step.complete({
                    "audio_path": str(video_info.audio_path),
                    "audio_duration": video_info.duration
                })
                if progress_callback:
                    progress_callback(40)
            except Exception as e:
                audio_step.fail(str(e))
                result.fail_processing(f"音频提取失败: {str(e)}")
                return result

            # 步骤4: 语音转写
            transcribe_step = result.add_step("transcription")
            try:
                # 检查转写缓存
                if use_cache:
                    cached_transcript = self.cache_manager.get_cached_transcript(url)
                    if cached_transcript:
                        logger.info(f"使用缓存转写: {url}")
                        logger.debug(f"缓存转写文本长度: {len(cached_transcript)}")
                        # get_cached_transcript 返回的是字符串，不是字典
                        video_info.transcript = cached_transcript
                        result.video_info = video_info
                        result.transcript = cached_transcript
                        transcribe_step.complete({
                            "from_cache": True,
                            "transcript_length": len(cached_transcript)
                        })
                    else:
                        logger.warning(f"缓存转写不存在或为空: {url}")
                        video_info = self.transcriber.transcribe_audio_for_video(video_info)
                        result.video_info = video_info
                        result.transcript = video_info.transcript
                        transcribe_step.complete({
                            "from_cache": False,
                            "transcript_length": len(video_info.transcript or "")
                        })
                        # 缓存转写结果
                        self.cache_manager.cache_transcript(url, video_info.transcript)
                else:
                    video_info = self.transcriber.transcribe_audio_for_video(video_info)
                    result.video_info = video_info
                    result.transcript = video_info.transcript
                    transcribe_step.complete({
                        "from_cache": False,
                        "transcript_length": len(video_info.transcript or "")
                    })

                if progress_callback:
                    progress_callback(60)
            except Exception as e:
                transcribe_step.fail(str(e))
                result.fail_processing(f"语音转写失败: {str(e)}")
                return result

            # 步骤5: 生成结构化笔记
            generate_step = result.add_step("generation")
            try:
                # 获取模板
                template = self.template_engine.get_template(template_name)

                # 检查 transcript 是否存在
                if not result.transcript:
                    logger.error(f"转写文本为空，无法生成笔记。result.transcript={result.transcript}, video_info.transcript={video_info.transcript}")
                    generate_step.fail("转写文本为空")
                    result.fail_processing("转写文本为空，无法生成笔记")
                    return result
                
                # 准备变量
                variables = {
                    "transcript": result.transcript,
                }
                
                logger.info(f"准备生成笔记，变量键: {list(variables.keys())}, transcript值类型: {type(variables.get('transcript'))}, transcript是否为空: {not variables.get('transcript')}")
                logger.debug(f"准备生成笔记，变量: {list(variables.keys())}, transcript长度: {len(result.transcript) if result.transcript else 0}, transcript类型: {type(result.transcript)}")

                # 生成笔记
                structured_notes = self.llm_client.generate_with_template(
                    template=template,
                    variables=variables
                )

                result.structured_notes = structured_notes
                result.template_used = template_name
                result.notes_length = len(structured_notes)

                generate_step.complete({
                    "template": template_name,
                    "notes_length": len(structured_notes)
                })
                if progress_callback:
                    progress_callback(80)
            except Exception as e:
                generate_step.fail(str(e))
                result.fail_processing(f"笔记生成失败: {str(e)}")
                return result

            # 步骤6: 保存结果
            save_step = result.add_step("saving")
            try:
                # 保存Markdown文件
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{video_info.get_safe_filename()}_{timestamp}.md"
                output_path = self.config.download.output_dir / filename

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(result.structured_notes)

                video_info.output_path = output_path
                result.video_info = video_info

                # 保存完整结果
                self.result_storage.save_result(result, format="json")
                self.result_storage.save_result(result, format="markdown")

                # 清除该URL的缓存（确保下次处理时重新生成，不使用缓存）
                try:
                    self.cache_manager.clear_cache_for_url(url)
                except Exception as e:
                    logger.warning(f"清除缓存失败: {e}")

                save_step.complete({
                    "output_path": str(output_path),
                    "file_size": output_path.stat().st_size
                })
                if progress_callback:
                    progress_callback(100)
            except Exception as e:
                save_step.fail(str(e))
                result.fail_processing(f"结果保存失败: {str(e)}")
                return result

            # 步骤7: 清理中间文件
            if not self.config.processing.keep_intermediate_files:
                cleanup_step = result.add_step("cleanup")
                try:
                    self._cleanup_intermediate_files(result)
                    cleanup_step.complete({})
                except Exception as e:
                    cleanup_step.fail(str(e))
                    logger.warning(f"清理中间文件失败: {e}")

            # 完成处理
            result.complete_processing()
            logger.success(f"视频处理完成: {url}")

            return result

        except Exception as e:
            logger.error(f"视频处理过程中发生未知错误: {e}")
            result.fail_processing(f"处理失败: {str(e)}")
            return result

    def batch_process(
        self,
        urls: List[str],
        template_name: str = "study_notes",
        max_workers: int = 1
    ) -> List[ProcessingResult]:
        """
        批量处理多个视频

        Args:
            urls: 视频URL列表
            template_name: 使用的模板名称
            max_workers: 最大并行处理数

        Returns:
            List[ProcessingResult]: 处理结果列表
        """
        results = []

        if max_workers <= 1:
            # 顺序处理
            for url in tqdm(urls, desc="处理视频"):
                try:
                    result = self.process_video(url, template_name)
                    results.append(result)
                except Exception as e:
                    logger.error(f"处理视频失败 {url}: {e}")
                    result = ProcessingResult(video_info=VideoInfo(url=url))
                    result.fail_processing(str(e))
                    results.append(result)
        else:
            # 并行处理
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交任务
                future_to_url = {
                    executor.submit(self.process_video, url, template_name): url
                    for url in urls
                }

                # 收集结果
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_url),
                    total=len(urls),
                    desc="处理视频"
                ):
                    url = future_to_url[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"处理视频失败 {url}: {e}")
                        result = ProcessingResult(video_info=VideoInfo(url=url))
                        result.fail_processing(str(e))
                        results.append(result)

        return results

    def _cleanup_intermediate_files(self, result: ProcessingResult):
        """清理中间文件"""
        try:
            # 清理视频文件
            if result.video_info.video_path and result.video_info.video_path.exists():
                result.video_info.video_path.unlink()

            # 清理音频文件
            if result.video_info.audio_path and result.video_info.audio_path.exists():
                result.video_info.audio_path.unlink()

            # 清理转写文件
            if result.video_info.transcript_path and result.video_info.transcript_path.exists():
                result.video_info.transcript_path.unlink()

            logger.debug("中间文件已清理")

        except Exception as e:
            logger.warning(f"清理中间文件时发生错误: {e}")

    def get_processing_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        try:
            # 获取结果存储统计
            storage_stats = self.result_storage.get_storage_statistics()

            # 获取缓存统计
            cache_stats = self.cache_manager.get_cache_statistics()

            return {
                "storage": storage_stats,
                "cache": cache_stats,
                "config": {
                    "model_provider": self.config.api.model_provider.value,
                    "whisper_model": self.config.processing.whisper_model.value,
                    "default_template": self.config.processing.default_template,
                }
            }

        except Exception as e:
            logger.error(f"获取处理统计信息失败: {e}")
            return {}

    def validate_video_url(self, url: str) -> bool:
        """
        验证视频URL

        Args:
            url: 视频URL

        Returns:
            bool: 是否有效
        """
        try:
            video_info = self.downloader.get_video_info(url)
            return video_info is not None
        except Exception:
            return False

    def get_supported_templates(self) -> List[str]:
        """获取支持的模板列表"""
        templates = self.template_engine.list_templates()
        return [t.name for t in templates]

    def cleanup(self):
        """清理资源"""
        try:
            self.transcriber.cleanup()
            self.llm_client.cleanup()
            logger.debug("处理器资源已清理")
        except Exception as e:
            logger.warning(f"清理处理器资源时发生错误: {e}")