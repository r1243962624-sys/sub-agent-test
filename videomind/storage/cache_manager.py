"""
缓存管理模块
用于缓存视频处理结果和转录文本
"""

import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime

from loguru import logger


class CacheManager:
    """缓存管理器"""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录路径，默认为 ./storage/cache
        """
        if cache_dir is None:
            self.cache_dir = Path("./storage/cache")
        else:
            self.cache_dir = Path(cache_dir)
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_cache_dir = self.cache_dir / "transcripts"
        self.result_cache_dir = self.cache_dir / "results"
        self.transcript_cache_dir.mkdir(parents=True, exist_ok=True)
        self.result_cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, url: str) -> str:
        """生成缓存键"""
        return hashlib.md5(url.encode()).hexdigest()

    def get_cached_transcript(self, url: str) -> Optional[str]:
        """
        获取缓存的转录文本
        
        Args:
            url: 视频URL
            
        Returns:
            转录文本，如果不存在则返回None
        """
        cache_key = self._get_cache_key(url)
        cache_file = self.transcript_cache_dir / f"{cache_key}.txt"
        
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"读取缓存转录失败: {e}")
                return None
        
        return None

    def cache_transcript(self, url: str, transcript: str) -> None:
        """
        缓存转录文本
        
        Args:
            url: 视频URL
            transcript: 转录文本
        """
        cache_key = self._get_cache_key(url)
        cache_file = self.transcript_cache_dir / f"{cache_key}.txt"
        
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(transcript)
            logger.debug(f"已缓存转录: {cache_key}")
        except Exception as e:
            logger.warning(f"缓存转录失败: {e}")

    def get_cached_processing_result(self, url: str) -> Optional[Dict[str, Any]]:
        """
        获取缓存的处理结果
        
        Args:
            url: 视频URL
            
        Returns:
            处理结果字典，如果不存在则返回None
        """
        cache_key = self._get_cache_key(url)
        cache_file = self.result_cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"读取缓存结果失败: {e}")
                return None
        
        return None

    def cache_processing_result(self, result: Union[Dict[str, Any], Any]) -> None:
        """
        缓存处理结果
        
        Args:
            result: 处理结果字典或ProcessingResult对象
        """
        # 如果是Pydantic模型，转换为字典
        if hasattr(result, "to_dict"):
            result_dict = result.to_dict()
        elif hasattr(result, "dict"):
            result_dict = result.dict()
        elif isinstance(result, dict):
            result_dict = result
        else:
            result_dict = result.__dict__ if hasattr(result, "__dict__") else {}
        
        # 从result_dict中获取URL
        if "video_info" in result_dict:
            video_info = result_dict.get("video_info", {})
            url = video_info.get("url", "") if isinstance(video_info, dict) else getattr(video_info, "url", "")
        else:
            url = result_dict.get("url", "")
            
        if not url:
            return
        
        cache_key = self._get_cache_key(url)
        cache_file = self.result_cache_dir / f"{cache_key}.json"
        
        try:
            # 确保所有Path对象和datetime转换为可序列化的格式
            def convert_for_json(obj):
                if isinstance(obj, Path):
                    return str(obj)
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
            
            result_for_json = convert_for_json(result_dict)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result_for_json, f, ensure_ascii=False, indent=2, default=str)
            logger.debug(f"已缓存处理结果: {cache_key}")
        except Exception as e:
            logger.warning(f"缓存处理结果失败: {e}")

    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计信息字典
        """
        transcript_count = len(list(self.transcript_cache_dir.glob("*.txt")))
        result_count = len(list(self.result_cache_dir.glob("*.json")))
        
        # 计算缓存大小
        transcript_size = sum(
            f.stat().st_size for f in self.transcript_cache_dir.glob("*.txt")
        )
        result_size = sum(
            f.stat().st_size for f in self.result_cache_dir.glob("*.json")
        )
        
        return {
            "transcript_count": transcript_count,
            "result_count": result_count,
            "transcript_size_mb": transcript_size / (1024 * 1024),
            "result_size_mb": result_size / (1024 * 1024),
            "total_size_mb": (transcript_size + result_size) / (1024 * 1024),
        }

    def clear_cache_for_url(self, url: str) -> None:
        """
        清除特定URL的缓存
        
        Args:
            url: 视频URL
        """
        cache_key = self._get_cache_key(url)
        
        # 清除转写缓存
        transcript_cache_file = self.transcript_cache_dir / f"{cache_key}.txt"
        if transcript_cache_file.exists():
            try:
                transcript_cache_file.unlink()
                logger.debug(f"已删除转写缓存: {cache_key}")
            except Exception as e:
                logger.warning(f"删除转写缓存失败: {e}")
        
        # 清除结果缓存
        result_cache_file = self.result_cache_dir / f"{cache_key}.json"
        if result_cache_file.exists():
            try:
                result_cache_file.unlink()
                logger.debug(f"已删除结果缓存: {cache_key}")
            except Exception as e:
                logger.warning(f"删除结果缓存失败: {e}")

    def clear_cache(self, cache_type: str = "all") -> None:
        """
        清除缓存
        
        Args:
            cache_type: 缓存类型，"transcripts"、"results"或"all"
        """
        if cache_type in ["transcripts", "all"]:
            for cache_file in self.transcript_cache_dir.glob("*.txt"):
                try:
                    cache_file.unlink()
                    logger.info(f"已删除缓存文件: {cache_file}")
                except Exception as e:
                    logger.warning(f"删除缓存文件失败: {e}")
        
        if cache_type in ["results", "all"]:
            for cache_file in self.result_cache_dir.glob("*.json"):
                try:
                    cache_file.unlink()
                    logger.info(f"已删除缓存文件: {cache_file}")
                except Exception as e:
                    logger.warning(f"删除缓存文件失败: {e}")
