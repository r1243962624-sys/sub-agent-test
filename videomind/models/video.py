"""
视频数据模型
定义视频信息和处理结果的结构
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Field, validator


class ProcessingStatus(str, Enum):
    """处理状态枚举"""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    EXTRACTING_AUDIO = "extracting_audio"
    TRANSCRIBING = "transcribing"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VideoPlatform(str, Enum):
    """视频平台枚举"""
    YOUTUBE = "youtube"
    BILIBILI = "bilibili"
    VIMEO = "vimeo"
    OTHER = "other"


class VideoInfo(BaseModel):
    """视频信息"""
    url: str = Field(..., description="视频URL")
    title: Optional[str] = Field(None, description="视频标题")
    platform: Optional[VideoPlatform] = Field(None, description="视频平台")
    duration: Optional[float] = Field(None, description="视频时长（秒）")
    upload_date: Optional[datetime] = Field(None, description="上传日期")
    description: Optional[str] = Field(None, description="视频描述")
    thumbnail_url: Optional[str] = Field(None, description="缩略图URL")
    channel: Optional[str] = Field(None, description="频道/上传者")

    # 文件路径
    video_path: Optional[Path] = Field(None, description="视频文件路径")
    audio_path: Optional[Path] = Field(None, description="音频文件路径")
    transcript_path: Optional[Path] = Field(None, description="转写文本路径")
    output_path: Optional[Path] = Field(None, description="输出文件路径")

    # 转写文本
    transcript: Optional[str] = Field(None, description="转写文本内容")

    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="原始元数据")

    @validator("platform", pre=True, always=True)
    def detect_platform(cls, v, values):
        """根据URL自动检测平台"""
        if v is not None:
            return v

        url = values.get("url", "")
        if not url:
            return VideoPlatform.OTHER

        url_lower = url.lower()
        if "youtube.com" in url_lower or "youtu.be" in url_lower:
            return VideoPlatform.YOUTUBE
        elif "bilibili.com" in url_lower:
            return VideoPlatform.BILIBILI
        elif "vimeo.com" in url_lower:
            return VideoPlatform.VIMEO
        else:
            return VideoPlatform.OTHER

    def get_safe_filename(self) -> str:
        """获取安全的文件名（用于保存文件）"""
        if self.title:
            # 移除非法字符，限制长度
            safe_title = "".join(c for c in self.title if c.isalnum() or c in " ._-")
            safe_title = safe_title[:100]  # 限制长度
            return safe_title.strip()
        else:
            # 使用URL的哈希值
            import hashlib
            url_hash = hashlib.md5(self.url.encode()).hexdigest()[:8]
            return f"video_{url_hash}"


class ProcessingStep(BaseModel):
    """处理步骤信息"""
    name: str = Field(..., description="步骤名称")
    status: ProcessingStatus = Field(..., description="步骤状态")
    start_time: Optional[datetime] = Field(None, description="开始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")
    duration: Optional[float] = Field(None, description="耗时（秒）")
    error: Optional[str] = Field(None, description="错误信息")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="步骤元数据")

    def start(self):
        """开始步骤"""
        self.start_time = datetime.now()
        self.status = ProcessingStatus.PENDING

    def complete(self, metadata: Dict[str, Any] = None):
        """完成步骤"""
        self.end_time = datetime.now()
        self.status = ProcessingStatus.COMPLETED
        if self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
        if metadata:
            self.metadata.update(metadata)

    def fail(self, error: str):
        """步骤失败"""
        self.end_time = datetime.now()
        self.status = ProcessingStatus.FAILED
        self.error = error
        if self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()


class ProcessingResult(BaseModel):
    """处理结果"""
    video_info: VideoInfo = Field(..., description="视频信息")
    status: ProcessingStatus = Field(ProcessingStatus.PENDING, description="总体状态")
    steps: List[ProcessingStep] = Field(default_factory=list, description="处理步骤")
    start_time: Optional[datetime] = Field(None, description="开始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")
    total_duration: Optional[float] = Field(None, description="总耗时（秒）")

    # 处理结果
    transcript: Optional[str] = Field(None, description="转写文本")
    structured_notes: Optional[str] = Field(None, description="结构化笔记")
    template_used: Optional[str] = Field(None, description="使用的模板")

    # 统计信息
    video_duration: Optional[float] = Field(None, description="视频时长（秒）")
    audio_duration: Optional[float] = Field(None, description="音频时长（秒）")
    transcript_length: Optional[int] = Field(None, description="转写文本长度")
    notes_length: Optional[int] = Field(None, description="笔记长度")
    token_usage: Optional[Dict[str, int]] = Field(None, description="Token 使用量")

    # 错误信息
    error: Optional[str] = Field(None, description="错误信息")
    error_details: Optional[Dict[str, Any]] = Field(None, description="错误详情")

    def start_processing(self):
        """开始处理"""
        self.start_time = datetime.now()
        self.status = ProcessingStatus.PENDING

    def add_step(self, step_name: str) -> ProcessingStep:
        """添加处理步骤"""
        step = ProcessingStep(name=step_name, status=ProcessingStatus.PENDING)
        step.start()
        self.steps.append(step)
        return step

    def complete_processing(self):
        """完成处理"""
        self.end_time = datetime.now()
        self.status = ProcessingStatus.COMPLETED
        if self.start_time:
            self.total_duration = (self.end_time - self.start_time).total_seconds()

    def fail_processing(self, error: str, details: Dict[str, Any] = None):
        """处理失败"""
        self.end_time = datetime.now()
        self.status = ProcessingStatus.FAILED
        self.error = error
        self.error_details = details or {}
        if self.start_time:
            self.total_duration = (self.end_time - self.start_time).total_seconds()

    def get_current_step(self) -> Optional[ProcessingStep]:
        """获取当前正在进行的步骤"""
        for step in reversed(self.steps):
            if step.status in [ProcessingStatus.PENDING, ProcessingStatus.DOWNLOADING,
                              ProcessingStatus.EXTRACTING_AUDIO, ProcessingStatus.TRANSCRIBING,
                              ProcessingStatus.GENERATING]:
                return step
        return None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于JSON序列化）"""
        result = self.dict(exclude={"video_info": {"video_path", "audio_path", "transcript_path", "output_path"}})
        result["video_info"] = self.video_info.dict(exclude={"video_path", "audio_path", "transcript_path", "output_path"})
        return result