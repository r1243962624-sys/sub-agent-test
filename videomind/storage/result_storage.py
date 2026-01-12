"""
结果存储模块
用于保存处理结果到不同格式的文件
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta

from loguru import logger


class ResultStorage:
    """结果存储器"""

    def __init__(self, output_dir: str = "./output"):
        """
        初始化结果存储器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.markdown_dir = self.output_dir / "markdown"
        self.json_dir = self.output_dir / "json"
        self.yaml_dir = self.output_dir / "yaml"
        self.pickle_dir = self.output_dir / "pickle"
        
        for dir_path in [self.markdown_dir, self.json_dir, self.yaml_dir, self.pickle_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _get_filename(self, result: Union[Dict[str, Any], Any], format: str) -> Path:
        """
        生成文件名
        
        Args:
            result: 处理结果字典或ProcessingResult对象
            format: 文件格式
            
        Returns:
            文件路径
        """
        # 如果是Pydantic模型，转换为字典
        if hasattr(result, "to_dict"):
            result_dict = result.to_dict()
        elif hasattr(result, "dict"):
            result_dict = result.dict()
        elif isinstance(result, dict):
            result_dict = result
        else:
            # 尝试作为对象访问属性
            result_dict = result.__dict__ if hasattr(result, "__dict__") else {}
        
        # 从结果中提取标识信息
        # 如果result_dict有video_info，从中获取url和title
        if "video_info" in result_dict:
            video_info = result_dict.get("video_info", {})
            url = video_info.get("url", "unknown") if isinstance(video_info, dict) else getattr(video_info, "url", "unknown")
            title = video_info.get("title", "untitled") if isinstance(video_info, dict) else getattr(video_info, "title", "untitled")
        else:
            url = result_dict.get("url", "unknown")
            title = result_dict.get("title", "untitled")
        
        # 清理标题，移除非法字符
        safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_"))[:50]
        safe_title = safe_title.strip().replace(" ", "_")
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 根据格式选择目录和扩展名
        if format == "markdown":
            dir_path = self.markdown_dir
            ext = ".md"
        elif format == "json":
            dir_path = self.json_dir
            ext = ".json"
        elif format == "yaml":
            dir_path = self.yaml_dir
            ext = ".yaml"
        else:
            dir_path = self.output_dir
            ext = f".{format}"
        
        filename = f"{safe_title}_{timestamp}{ext}"
        return dir_path / filename

    def save_result(self, result: Union[Dict[str, Any], Any], format: str = "markdown") -> Path:
        """
        保存处理结果
        
        Args:
            result: 处理结果字典或ProcessingResult对象
            format: 文件格式，支持 "markdown", "json", "yaml"
            
        Returns:
            保存的文件路径
        """
        # 如果是Pydantic模型，转换为字典
        if hasattr(result, "to_dict"):
            result_dict = result.to_dict()
        elif hasattr(result, "dict"):
            result_dict = result.dict()
        elif isinstance(result, dict):
            result_dict = result
        else:
            # 尝试作为对象访问属性
            result_dict = result.__dict__ if hasattr(result, "__dict__") else {}
            logger.warning("无法将result转换为字典，使用__dict__")
        
        file_path = self._get_filename(result, format)
        
        try:
            if format == "markdown":
                # 从result中获取structured_notes作为markdown内容
                if hasattr(result, "structured_notes") and result.structured_notes:
                    content = result.structured_notes
                elif isinstance(result_dict, dict):
                    content = result_dict.get("structured_notes", result_dict.get("content", ""))
                else:
                    content = ""
                    
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
            
            elif format == "json":
                # 确保所有Path对象转换为字符串
                def convert_paths(obj):
                    if isinstance(obj, Path):
                        return str(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_paths(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_paths(item) for item in obj]
                    elif isinstance(obj, datetime):
                        return obj.isoformat()
                    return obj
                
                result_for_json = convert_paths(result_dict)
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(result_for_json, f, ensure_ascii=False, indent=2, default=str)
            
            elif format == "yaml":
                try:
                    import yaml
                    # 确保所有Path对象转换为字符串
                    def convert_paths(obj):
                        if isinstance(obj, Path):
                            return str(obj)
                        elif isinstance(obj, dict):
                            return {k: convert_paths(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_paths(item) for item in obj]
                        elif isinstance(obj, datetime):
                            return obj.isoformat()
                        return obj
                    
                    result_for_yaml = convert_paths(result_dict)
                    with open(file_path, "w", encoding="utf-8") as f:
                        yaml.dump(result_for_yaml, f, allow_unicode=True, default_flow_style=False)
                except ImportError:
                    logger.warning("yaml模块未安装，无法保存yaml格式")
                    raise
            
            else:
                raise ValueError(f"不支持的格式: {format}")
            
            logger.info(f"已保存结果到: {file_path}")
            return file_path
        
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
            raise

    def get_storage_statistics(self) -> Dict[str, Any]:
        """
        获取存储统计信息
        
        Returns:
            存储统计信息字典
        """
        stats = {
            "markdown_count": len(list(self.markdown_dir.glob("*.md"))),
            "json_count": len(list(self.json_dir.glob("*.json"))),
            "yaml_count": len(list(self.yaml_dir.glob("*.yaml"))) + len(list(self.yaml_dir.glob("*.yml"))),
            "pickle_count": len(list(self.pickle_dir.glob("*.pkl"))),
        }
        
        # 计算总大小
        total_size = 0
        for dir_path in [self.markdown_dir, self.json_dir, self.yaml_dir, self.pickle_dir]:
            for file_path in dir_path.glob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        
        stats["total_size_mb"] = total_size / (1024 * 1024)
        stats["total_files"] = sum(stats.values()) - stats["total_size_mb"]
        
        return stats

    def cleanup_old_results(self, days: int = 30) -> int:
        """
        清理旧的结果文件
        
        Args:
            days: 保留天数，超过此天数的文件将被删除
            
        Returns:
            删除的文件数量
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_count = 0
        
        for dir_path in [self.markdown_dir, self.json_dir, self.yaml_dir, self.pickle_dir]:
            for file_path in dir_path.glob("*"):
                if file_path.is_file():
                    try:
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_time < cutoff_date:
                            file_path.unlink()
                            deleted_count += 1
                            logger.debug(f"已删除旧文件: {file_path}")
                    except Exception as e:
                        logger.warning(f"删除文件失败 {file_path}: {e}")
        
        logger.info(f"清理完成，删除了 {deleted_count} 个旧文件")
        return deleted_count
