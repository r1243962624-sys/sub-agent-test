"""
批量处理管理器
智能管理批量视频处理任务，优化并发和资源使用
"""

import asyncio
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue

from loguru import logger

from models.config import Config
from core.processor import VideoProcessor
from models.video import ProcessingResult
from core.cost_monitor import CostMonitor, get_cost_monitor


class BatchTaskStatus(str, Enum):
    """批量任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchPriority(str, Enum):
    """批量任务优先级"""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class BatchTask:
    """批量任务"""
    task_id: str
    urls: List[str]
    template_name: str
    config: Config
    priority: BatchPriority = BatchPriority.NORMAL
    max_workers: int = 1
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    status: BatchTaskStatus = BatchTaskStatus.PENDING
    results: List[ProcessingResult] = field(default_factory=list)
    progress: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchProgress:
    """批量处理进度"""
    task_id: str
    total_items: int
    processed_items: int
    successful_items: int
    failed_items: int
    progress_percentage: float
    estimated_time_remaining: Optional[float] = None  # 秒
    current_speed: Optional[float] = None  # 项目/秒
    current_workers: int = 0


class BatchManager:
    """批量处理管理器"""

    def __init__(self, max_concurrent_tasks: int = 3, max_workers_per_task: int = 2):
        """
        初始化批量处理管理器

        Args:
            max_concurrent_tasks: 最大并发任务数
            max_workers_per_task: 每个任务的最大工作线程数
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_workers_per_task = max_workers_per_task

        # 任务管理
        self.tasks: Dict[str, BatchTask] = {}
        self.task_queue = queue.PriorityQueue()
        self.active_tasks: Dict[str, threading.Thread] = {}

        # 线程池
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent_tasks * max_workers_per_task,
            thread_name_prefix="batch_worker"
        )

        # 成本监控
        self.cost_monitor = get_cost_monitor()

        # 状态锁
        self._lock = threading.Lock()

        # 性能统计
        self.performance_stats = {
            "total_tasks": 0,
            "total_videos": 0,
            "total_success": 0,
            "total_failed": 0,
            "total_cost": 0.0,
            "avg_processing_time": 0.0,
        }

        logger.info(f"批量处理管理器初始化完成，最大并发任务: {max_concurrent_tasks}")

    def create_batch_task(
        self,
        urls: List[str],
        template_name: str,
        config: Config,
        priority: BatchPriority = BatchPriority.NORMAL,
        max_workers: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        创建批量任务

        Args:
            urls: 视频URL列表
            template_name: 模板名称
            config: 配置
            priority: 任务优先级
            max_workers: 最大工作线程数
            metadata: 元数据

        Returns:
            str: 任务ID
        """
        try:
            # 生成任务ID
            task_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.tasks)}"

            # 限制最大工作线程数
            if max_workers is None:
                max_workers = min(len(urls), self.max_workers_per_task)
            else:
                max_workers = min(max_workers, self.max_workers_per_task, len(urls))

            # 创建任务
            task = BatchTask(
                task_id=task_id,
                urls=urls,
                template_name=template_name,
                config=config,
                priority=priority,
                max_workers=max_workers,
                metadata=metadata or {}
            )

            # 保存任务
            with self._lock:
                self.tasks[task_id] = task

            # 添加到队列（根据优先级）
            priority_value = {"high": 0, "normal": 1, "low": 2}[priority.value]
            self.task_queue.put((priority_value, task_id))

            # 更新统计
            self.performance_stats["total_tasks"] += 1
            self.performance_stats["total_videos"] += len(urls)

            logger.info(f"创建批量任务: {task_id}，包含 {len(urls)} 个视频，优先级: {priority.value}")
            return task_id

        except Exception as e:
            logger.error(f"创建批量任务失败: {e}")
            raise

    def start_task(self, task_id: str) -> bool:
        """
        启动任务

        Args:
            task_id: 任务ID

        Returns:
            bool: 是否成功启动
        """
        try:
            with self._lock:
                if task_id not in self.tasks:
                    logger.error(f"任务不存在: {task_id}")
                    return False

                task = self.tasks[task_id]
                if task.status != BatchTaskStatus.PENDING:
                    logger.warning(f"任务状态不是PENDING: {task.status}")
                    return False

                # 更新任务状态
                task.status = BatchTaskStatus.RUNNING
                task.started_at = datetime.now().isoformat()

            # 启动处理线程
            thread = threading.Thread(
                target=self._process_batch_task,
                args=(task_id,),
                name=f"batch_task_{task_id}",
                daemon=True
            )
            thread.start()

            with self._lock:
                self.active_tasks[task_id] = thread

            logger.info(f"启动批量任务: {task_id}")
            return True

        except Exception as e:
            logger.error(f"启动任务失败: {e}")
            with self._lock:
                if task_id in self.tasks:
                    self.tasks[task_id].status = BatchTaskStatus.FAILED
                    self.tasks[task_id].error = str(e)
            return False

    def _process_batch_task(self, task_id: str):
        """处理批量任务"""
        try:
            with self._lock:
                task = self.tasks[task_id]

            logger.info(f"开始处理批量任务: {task_id}，视频数: {len(task.urls)}")

            # 创建视频处理器
            processor = VideoProcessor(task.config)

            # 智能分批处理
            batches = self._create_intelligent_batches(task.urls, task.max_workers)

            total_batches = len(batches)
            processed_count = 0

            for batch_idx, batch_urls in enumerate(batches):
                logger.info(f"处理批次 {batch_idx + 1}/{total_batches}，包含 {len(batch_urls)} 个视频")

                # 处理当前批次
                batch_results = processor.batch_process(
                    urls=batch_urls,
                    template_name=task.template_name,
                    max_workers=min(len(batch_urls), task.max_workers)
                )

                # 更新任务结果
                with self._lock:
                    task.results.extend(batch_results)
                    processed_count += len(batch_urls)
                    task.progress = processed_count / len(task.urls)

                # 记录成本
                self._record_batch_costs(batch_results, task_id)

                # 检查是否需要暂停或取消
                with self._lock:
                    if task.status == BatchTaskStatus.PAUSED:
                        logger.info(f"任务暂停: {task_id}")
                        return
                    elif task.status == BatchTaskStatus.CANCELLED:
                        logger.info(f"任务取消: {task_id}")
                        return

            # 任务完成
            with self._lock:
                task.status = BatchTaskStatus.COMPLETED
                task.completed_at = datetime.now().isoformat()
                task.progress = 1.0

            # 更新统计
            self._update_performance_stats(task)

            logger.info(f"批量任务完成: {task_id}，成功: {self._count_successful_results(task.results)}")

        except Exception as e:
            logger.error(f"处理批量任务失败: {e}")
            with self._lock:
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    task.status = BatchTaskStatus.FAILED
                    task.error = str(e)

    def _create_intelligent_batches(self, urls: List[str], max_workers: int) -> List[List[str]]:
        """
        创建智能批次

        Args:
            urls: URL列表
            max_workers: 最大工作线程数

        Returns:
            List[List[str]]: 批次列表
        """
        # 简单分批：根据工作线程数平均分配
        batch_size = max(1, len(urls) // max_workers)
        batches = []

        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            if batch:
                batches.append(batch)

        # 如果批次太多，合并小批次
        if len(batches) > max_workers * 2:
            merged_batches = []
            current_batch = []
            target_size = len(urls) // max_workers

            for batch in batches:
                current_batch.extend(batch)
                if len(current_batch) >= target_size:
                    merged_batches.append(current_batch)
                    current_batch = []

            if current_batch:
                merged_batches.append(current_batch)

            batches = merged_batches

        return batches

    def _record_batch_costs(self, results: List[ProcessingResult], task_id: str):
        """记录批次成本"""
        try:
            for result in results:
                if result.llm_usage:
                    self.cost_monitor.record_cost(
                        provider=result.llm_usage.get("provider", "unknown"),
                        model=result.llm_usage.get("model", "unknown"),
                        prompt_tokens=result.llm_usage.get("prompt_tokens", 0),
                        completion_tokens=result.llm_usage.get("completion_tokens", 0),
                        operation_type="batch",
                        template_name=result.template_used,
                        video_id=result.video_info.url,
                        success=result.status.value == "completed",
                        error_message=result.error
                    )
        except Exception as e:
            logger.warning(f"记录批次成本失败: {e}")

    def _count_successful_results(self, results: List[ProcessingResult]) -> int:
        """计算成功结果数量"""
        return sum(1 for r in results if r.status.value == "completed")

    def _update_performance_stats(self, task: BatchTask):
        """更新性能统计"""
        successful_results = self._count_successful_results(task.results)
        failed_results = len(task.results) - successful_results

        # 计算平均处理时间
        processing_times = []
        total_cost = 0.0

        for result in task.results:
            if result.total_duration > 0:
                processing_times.append(result.total_duration)

            # 估算成本
            if result.llm_usage:
                cost = self.cost_monitor.calculate_cost(
                    result.llm_usage.get("model", "unknown"),
                    result.llm_usage.get("prompt_tokens", 0),
                    result.llm_usage.get("completion_tokens", 0)
                )
                total_cost += cost

        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0

        with self._lock:
            self.performance_stats["total_success"] += successful_results
            self.performance_stats["total_failed"] += failed_results
            self.performance_stats["total_cost"] += total_cost

            # 更新平均处理时间（加权平均）
            old_avg = self.performance_stats["avg_processing_time"]
            old_count = self.performance_stats["total_success"] - successful_results
            if old_count + successful_results > 0:
                new_avg = (old_avg * old_count + avg_time * successful_results) / (old_count + successful_results)
                self.performance_stats["avg_processing_time"] = new_avg

    def pause_task(self, task_id: str) -> bool:
        """
        暂停任务

        Args:
            task_id: 任务ID

        Returns:
            bool: 是否成功暂停
        """
        try:
            with self._lock:
                if task_id not in self.tasks:
                    return False

                task = self.tasks[task_id]
                if task.status != BatchTaskStatus.RUNNING:
                    return False

                task.status = BatchTaskStatus.PAUSED
                logger.info(f"暂停任务: {task_id}")
                return True

        except Exception as e:
            logger.error(f"暂停任务失败: {e}")
            return False

    def resume_task(self, task_id: str) -> bool:
        """
        恢复任务

        Args:
            task_id: 任务ID

        Returns:
            bool: 是否成功恢复
        """
        try:
            with self._lock:
                if task_id not in self.tasks:
                    return False

                task = self.tasks[task_id]
                if task.status != BatchTaskStatus.PAUSED:
                    return False

                task.status = BatchTaskStatus.RUNNING
                logger.info(f"恢复任务: {task_id}")
                return True

        except Exception as e:
            logger.error(f"恢复任务失败: {e}")
            return False

    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务

        Args:
            task_id: 任务ID

        Returns:
            bool: 是否成功取消
        """
        try:
            with self._lock:
                if task_id not in self.tasks:
                    return False

                task = self.tasks[task_id]
                if task.status in [BatchTaskStatus.COMPLETED, BatchTaskStatus.CANCELLED]:
                    return False

                task.status = BatchTaskStatus.CANCELLED
                task.completed_at = datetime.now().isoformat()

                # 如果任务正在运行，标记为取消
                if task_id in self.active_tasks:
                    # 注意：这里只是标记，实际线程可能还在运行
                    pass

                logger.info(f"取消任务: {task_id}")
                return True

        except Exception as e:
            logger.error(f"取消任务失败: {e}")
            return False

    def get_task_progress(self, task_id: str) -> Optional[BatchProgress]:
        """
        获取任务进度

        Args:
            task_id: 任务ID

        Returns:
            Optional[BatchProgress]: 进度信息
        """
        try:
            with self._lock:
                if task_id not in self.tasks:
                    return None

                task = self.tasks[task_id]

                # 计算统计
                successful_items = self._count_successful_results(task.results)
                failed_items = len(task.results) - successful_items
                processed_items = len(task.results)

                # 计算进度百分比
                progress_percentage = task.progress * 100

                # 估算剩余时间（如果任务已开始）
                estimated_time_remaining = None
                current_speed = None

                if task.started_at and task.status == BatchTaskStatus.RUNNING:
                    try:
                        start_time = datetime.fromisoformat(task.started_at)
                        elapsed_time = (datetime.now() - start_time).total_seconds()

                        if processed_items > 0 and elapsed_time > 0:
                            current_speed = processed_items / elapsed_time
                            remaining_items = len(task.urls) - processed_items
                            if current_speed > 0:
                                estimated_time_remaining = remaining_items / current_speed
                    except Exception:
                        pass

                return BatchProgress(
                    task_id=task_id,
                    total_items=len(task.urls),
                    processed_items=processed_items,
                    successful_items=successful_items,
                    failed_items=failed_items,
                    progress_percentage=progress_percentage,
                    estimated_time_remaining=estimated_time_remaining,
                    current_speed=current_speed,
                    current_workers=task.max_workers
                )

        except Exception as e:
            logger.error(f"获取任务进度失败: {e}")
            return None

    def get_task(self, task_id: str) -> Optional[BatchTask]:
        """
        获取任务详情

        Args:
            task_id: 任务ID

        Returns:
            Optional[BatchTask]: 任务详情
        """
        with self._lock:
            return self.tasks.get(task_id)

    def list_tasks(
        self,
        status_filter: Optional[BatchTaskStatus] = None,
        limit: int = 50
    ) -> List[BatchTask]:
        """
        列出任务

        Args:
            status_filter: 状态过滤器
            limit: 返回数量限制

        Returns:
            List[BatchTask]: 任务列表
        """
        with self._lock:
            tasks = list(self.tasks.values())

            if status_filter:
                tasks = [t for t in tasks if t.status == status_filter]

            # 按创建时间排序（最新的在前）
            tasks.sort(key=lambda x: x.created_at, reverse=True)

            return tasks[:limit]

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计

        Returns:
            Dict[str, Any]: 性能统计
        """
        with self._lock:
            stats = self.performance_stats.copy()

            # 添加当前活动任务数
            stats["active_tasks"] = len([t for t in self.tasks.values() if t.status == BatchTaskStatus.RUNNING])
            stats["pending_tasks"] = len([t for t in self.tasks.values() if t.status == BatchTaskStatus.PENDING])
            stats["total_tasks_in_system"] = len(self.tasks)

            # 计算成功率
            if stats["total_videos"] > 0:
                stats["success_rate"] = stats["total_success"] / stats["total_videos"] * 100
            else:
                stats["success_rate"] = 0.0

            # 平均成本
            if stats["total_success"] > 0:
                stats["avg_cost_per_video"] = stats["total_cost"] / stats["total_success"]
            else:
                stats["avg_cost_per_video"] = 0.0

            return stats

    def cleanup_old_tasks(self, days_to_keep: int = 7):
        """
        清理旧任务

        Args:
            days_to_keep: 保留天数
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            tasks_to_remove = []

            with self._lock:
                for task_id, task in self.tasks.items():
                    try:
                        created_at = datetime.fromisoformat(task.created_at)
                        if created_at < cutoff_date and task.status in [
                            BatchTaskStatus.COMPLETED,
                            BatchTaskStatus.FAILED,
                            BatchTaskStatus.CANCELLED
                        ]:
                            tasks_to_remove.append(task_id)
                    except Exception:
                        pass

                # 移除任务
                for task_id in tasks_to_remove:
                    del self.tasks[task_id]
                    if task_id in self.active_tasks:
                        del self.active_tasks[task_id]

            logger.info(f"清理了 {len(tasks_to_remove)} 个超过 {days_to_keep} 天的旧任务")

        except Exception as e:
            logger.error(f"清理旧任务失败: {e}")

    def optimize_batch_strategy(
        self,
        urls: List[str],
        available_workers: int,
        estimated_duration_per_video: float = 300.0
    ) -> Dict[str, Any]:
        """
        优化批量处理策略

        Args:
            urls: URL列表
            available_workers: 可用工作线程数
            estimated_duration_per_video: 预估每个视频处理时间（秒）

        Returns:
            Dict[str, Any]: 优化策略
        """
        total_videos = len(urls)

        if total_videos == 0:
            return {
                "recommended_workers": 0,
                "batch_size": 0,
                "estimated_total_time": 0,
                "strategy": "no_videos"
            }

        # 智能计算推荐的工作线程数
        recommended_workers = min(
            available_workers,
            total_videos,
            self.max_workers_per_task
        )

        # 根据视频数量调整
        if total_videos <= 3:
            recommended_workers = 1
            batch_strategy = "sequential"  # 顺序处理
        elif total_videos <= 10:
            recommended_workers = min(2, recommended_workers)
            batch_strategy = "small_batches"  # 小批量
        else:
            batch_strategy = "parallel_batches"  # 并行批次

        # 计算批次大小
        if batch_strategy == "sequential":
            batch_size = total_videos
            num_batches = 1
        elif batch_strategy == "small_batches":
            batch_size = max(2, total_videos // recommended_workers)
            num_batches = (total_videos + batch_size - 1) // batch_size
        else:  # parallel_batches
            batch_size = max(3, total_videos // recommended_workers)
            num_batches = (total_videos + batch_size - 1) // batch_size

        # 估算总时间
        estimated_total_time = (total_videos * estimated_duration_per_video) / recommended_workers

        return {
            "recommended_workers": recommended_workers,
            "batch_size": batch_size,
            "num_batches": num_batches,
            "batch_strategy": batch_strategy,
            "estimated_total_time": estimated_total_time,
            "estimated_time_per_batch": estimated_total_time / num_batches if num_batches > 0 else 0,
            "efficiency_score": min(1.0, recommended_workers / total_videos * 10) if total_videos > 0 else 0
        }

    def export_task_report(self, task_id: str, format: str = "markdown") -> str:
        """
        导出任务报告

        Args:
            task_id: 任务ID
            format: 报告格式（markdown, json）

        Returns:
            str: 报告内容
        """
        try:
            task = self.get_task(task_id)
            if not task:
                return f"任务不存在: {task_id}"

            progress = self.get_task_progress(task_id)

            if format.lower() == "json":
                import json
                report_data = {
                    "task": {
                        "task_id": task.task_id,
                        "status": task.status.value,
                        "priority": task.priority.value,
                        "created_at": task.created_at,
                        "started_at": task.started_at,
                        "completed_at": task.completed_at,
                        "total_videos": len(task.urls),
                        "template": task.template_name,
                        "max_workers": task.max_workers,
                        "error": task.error,
                        "metadata": task.metadata
                    },
                    "progress": {
                        "processed": progress.processed_items if progress else 0,
                        "successful": progress.successful_items if progress else 0,
                        "failed": progress.failed_items if progress else 0,
                        "percentage": progress.progress_percentage if progress else 0,
                    } if progress else None,
                    "results_summary": self._create_results_summary(task.results)
                }
                return json.dumps(report_data, ensure_ascii=False, indent=2)

            else:  # markdown
                # 创建Markdown报告
                report = f"""# 批量处理任务报告

## 任务信息
- **任务ID**: {task.task_id}
- **状态**: {task.status.value}
- **优先级**: {task.priority.value}
- **创建时间**: {task.created_at}
- **开始时间**: {task.started_at or '未开始'}
- **完成时间**: {task.completed_at or '未完成'}
- **总视频数**: {len(task.urls)}
- **使用模板**: {task.template_name}
- **最大工作线程**: {task.max_workers}

"""

                if progress:
                    report += f"""## 处理进度
- **已处理**: {progress.processed_items}/{len(task.urls)}
- **成功**: {progress.successful_items}
- **失败**: {progress.failed_items}
- **进度**: {progress.progress_percentage:.1f}%

"""
                    if progress.estimated_time_remaining:
                        report += f"- **预计剩余时间**: {progress.estimated_time_remaining:.1f}秒\n"
                    if progress.current_speed:
                        report += f"- **当前速度**: {progress.current_speed:.2f} 视频/秒\n"

                report += """## 结果摘要

"""

                summary = self._create_results_summary(task.results)
                for key, value in summary.items():
                    if isinstance(value, (int, float)) and key.endswith("_rate"):
                        report += f"- **{key}**: {value:.1f}%\n"
                    elif isinstance(value, (int, float)):
                        report += f"- **{key}**: {value}\n"
                    else:
                        report += f"- **{key}**: {value}\n"

                # 失败详情
                failed_results = [r for r in task.results if r.status.value != "completed"]
                if failed_results:
                    report += "\n## 失败详情\n\n"
                    for i, result in enumerate(failed_results[:10]):  # 最多显示10个
                        report += f"{i+1}. **{result.video_info.url}**\n"
                        if result.error:
                            report += f"   错误: {result.error}\n"
                        report += "\n"

                    if len(failed_results) > 10:
                        report += f"... 还有 {len(failed_results) - 10} 个失败项\n"

                return report

        except Exception as e:
            logger.error(f"导出任务报告失败: {e}")
            return f"导出报告失败: {str(e)}"

    def _create_results_summary(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """创建结果摘要"""
        total = len(results)
        successful = self._count_successful_results(results)
        failed = total - successful

        # 计算统计
        total_duration = sum(r.total_duration for r in results if r.total_duration > 0)
        avg_duration = total_duration / successful if successful > 0 else 0

        # 成本统计
        total_cost = 0.0
        total_tokens = 0
        for result in results:
            if result.llm_usage:
                cost = self.cost_monitor.calculate_cost(
                    result.llm_usage.get("model", "unknown"),
                    result.llm_usage.get("prompt_tokens", 0),
                    result.llm_usage.get("completion_tokens", 0)
                )
                total_cost += cost
                total_tokens += result.llm_usage.get("prompt_tokens", 0) + result.llm_usage.get("completion_tokens", 0)

        return {
            "total_results": total,
            "successful_results": successful,
            "failed_results": failed,
            "success_rate": (successful / total * 100) if total > 0 else 0,
            "total_processing_time": total_duration,
            "average_processing_time": avg_duration,
            "total_cost_usd": total_cost,
            "total_tokens": total_tokens,
            "avg_cost_per_successful": total_cost / successful if successful > 0 else 0,
        }

    def shutdown(self):
        """关闭批量管理器"""
        try:
            logger.info("正在关闭批量管理器...")

            # 取消所有任务
            with self._lock:
                for task_id in list(self.tasks.keys()):
                    if self.tasks[task_id].status in [BatchTaskStatus.PENDING, BatchTaskStatus.RUNNING, BatchTaskStatus.PAUSED]:
                        self.tasks[task_id].status = BatchTaskStatus.CANCELLED

            # 关闭执行器
            self.executor.shutdown(wait=False)

            logger.info("批量管理器已关闭")

        except Exception as e:
            logger.error(f"关闭批量管理器失败: {e}")


# 全局批量管理器实例
_batch_manager_instance: Optional[BatchManager] = None

def get_batch_manager() -> BatchManager:
    """获取全局批量管理器实例"""
    global _batch_manager_instance
    if _batch_manager_instance is None:
        _batch_manager_instance = BatchManager()
    return _batch_manager_instance