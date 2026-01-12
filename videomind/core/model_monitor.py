"""
模型性能监控模块
监控各模型的响应时间、成功率、成本效率等指标
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics

from loguru import logger

from models.config import ModelProvider
from core.cost_monitor import CostMonitor, get_cost_monitor


class ModelMetric(str, Enum):
    """模型监控指标"""
    RESPONSE_TIME = "response_time"
    SUCCESS_RATE = "success_rate"
    TOKEN_EFFICIENCY = "token_efficiency"
    COST_EFFICIENCY = "cost_efficiency"
    AVAILABILITY = "availability"


@dataclass
class ModelPerformance:
    """模型性能数据"""
    model_name: str
    provider: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    avg_response_time: float = 0.0
    min_response_time: float = 0.0
    max_response_time: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_cost_per_request: float = 0.0
    avg_tokens_per_request: int = 0
    last_used: Optional[str] = None
    performance_score: float = 0.0  # 综合性能评分（0-100）


@dataclass
class ModelRecommendation:
    """模型推荐"""
    task_type: str
    recommended_model: str
    alternative_models: List[str]
    reasoning: str
    estimated_cost: float
    estimated_time: float
    confidence_score: float  # 0-1


class ModelMonitor:
    """模型性能监控器"""

    def __init__(self, db_path: Optional[Path] = None):
        """
        初始化模型性能监控器

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path or Path("./data/model_performance.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 成本监控器
        self.cost_monitor = get_cost_monitor()

        # 性能基准
        self.performance_benchmarks = {
            "response_time": {
                "excellent": 2.0,   # 秒
                "good": 5.0,
                "poor": 10.0
            },
            "success_rate": {
                "excellent": 0.98,  # 98%
                "good": 0.95,
                "poor": 0.90
            },
            "cost_per_token": {
                "excellent": 0.0001,  # $/token
                "good": 0.0005,
                "poor": 0.001
            }
        }

        # 任务类型到模型的映射建议
        self.task_model_mapping = {
            "summarization": ["gpt-3.5-turbo", "claude-3-haiku", "deepseek-chat"],
            "analysis": ["gpt-4-turbo-preview", "claude-3-sonnet", "gpt-4"],
            "creative": ["claude-3-opus", "gpt-4", "gpt-4-turbo-preview"],
            "coding": ["deepseek-coder", "gpt-4", "claude-3-sonnet"],
            "translation": ["gpt-3.5-turbo", "claude-3-haiku", "deepseek-chat"],
            "qa": ["gpt-3.5-turbo", "claude-3-haiku", "deepseek-chat"],
        }

        # 初始化数据库
        self._init_database()

        logger.info(f"模型性能监控器初始化完成，数据库: {self.db_path}")

    def _init_database(self):
        """初始化数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 创建模型性能记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    response_time REAL,
                    success BOOLEAN DEFAULT 1,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    cost_usd REAL DEFAULT 0.0,
                    task_type TEXT,
                    template_name TEXT,
                    error_message TEXT
                )
            """)

            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON model_performance(model_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON model_performance(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_provider ON model_performance(provider)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_task_type ON model_performance(task_type)")

            # 创建模型性能汇总表（定期更新）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance_summary (
                    model_name TEXT PRIMARY KEY,
                    provider TEXT,
                    total_requests INTEGER DEFAULT 0,
                    successful_requests INTEGER DEFAULT 0,
                    failed_requests INTEGER DEFAULT 0,
                    total_response_time REAL DEFAULT 0.0,
                    avg_response_time REAL DEFAULT 0.0,
                    min_response_time REAL DEFAULT 0.0,
                    max_response_time REAL DEFAULT 0.0,
                    total_tokens INTEGER DEFAULT 0,
                    total_cost REAL DEFAULT 0.0,
                    avg_cost_per_request REAL DEFAULT 0.0,
                    avg_tokens_per_request INTEGER DEFAULT 0,
                    last_updated TEXT,
                    performance_score REAL DEFAULT 0.0
                )
            """)

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"初始化数据库失败: {e}")
            raise

    def record_model_performance(
        self,
        model_name: str,
        provider: str,
        response_time: float,
        success: bool = True,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        task_type: Optional[str] = None,
        template_name: Optional[str] = None,
        error_message: Optional[str] = None
    ):
        """
        记录模型性能数据

        Args:
            model_name: 模型名称
            provider: 提供商
            response_time: 响应时间（秒）
            success: 是否成功
            prompt_tokens: 输入token数
            completion_tokens: 输出token数
            task_type: 任务类型
            template_name: 模板名称
            error_message: 错误信息
        """
        try:
            # 计算成本
            cost = self.cost_monitor.calculate_cost(model_name, prompt_tokens, completion_tokens)
            total_tokens = prompt_tokens + completion_tokens

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO model_performance (
                    timestamp, model_name, provider, response_time, success,
                    prompt_tokens, completion_tokens, total_tokens, cost_usd,
                    task_type, template_name, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                model_name,
                provider,
                response_time,
                success,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cost,
                task_type,
                template_name,
                error_message
            ))

            conn.commit()
            conn.close()

            # 更新性能汇总
            self._update_performance_summary(model_name)

            logger.debug(f"记录模型性能: {model_name} - 响应时间: {response_time:.2f}s, 成功: {success}")

        except Exception as e:
            logger.error(f"记录模型性能失败: {e}")

    def _update_performance_summary(self, model_name: str):
        """更新模型性能汇总"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 获取最近30天的数据
            cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()

            cursor.execute("""
                SELECT
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_requests,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_requests,
                    SUM(response_time) as total_response_time,
                    AVG(response_time) as avg_response_time,
                    MIN(response_time) as min_response_time,
                    MAX(response_time) as max_response_time,
                    SUM(total_tokens) as total_tokens,
                    SUM(cost_usd) as total_cost,
                    provider
                FROM model_performance
                WHERE model_name = ? AND timestamp >= ?
                GROUP BY model_name, provider
            """, (model_name, cutoff_date))

            row = cursor.fetchone()

            if row:
                total_requests = row[0] or 0
                successful_requests = row[1] or 0
                failed_requests = row[2] or 0
                total_response_time = row[3] or 0.0
                avg_response_time = row[4] or 0.0
                min_response_time = row[5] or 0.0
                max_response_time = row[6] or 0.0
                total_tokens = row[7] or 0
                total_cost = row[8] or 0.0
                provider = row[9] or "unknown"

                avg_cost_per_request = total_cost / total_requests if total_requests > 0 else 0.0
                avg_tokens_per_request = total_tokens // total_requests if total_requests > 0 else 0

                # 计算性能评分
                performance_score = self._calculate_performance_score(
                    avg_response_time,
                    successful_requests / total_requests if total_requests > 0 else 0,
                    avg_cost_per_request,
                    total_requests
                )

                # 更新或插入汇总数据
                cursor.execute("""
                    INSERT OR REPLACE INTO model_performance_summary (
                        model_name, provider, total_requests, successful_requests,
                        failed_requests, total_response_time, avg_response_time,
                        min_response_time, max_response_time, total_tokens,
                        total_cost, avg_cost_per_request, avg_tokens_per_request,
                        last_updated, performance_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_name,
                    provider,
                    total_requests,
                    successful_requests,
                    failed_requests,
                    total_response_time,
                    avg_response_time,
                    min_response_time,
                    max_response_time,
                    total_tokens,
                    total_cost,
                    avg_cost_per_request,
                    avg_tokens_per_request,
                    datetime.now().isoformat(),
                    performance_score
                ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"更新性能汇总失败: {e}")

    def _calculate_performance_score(
        self,
        avg_response_time: float,
        success_rate: float,
        avg_cost_per_request: float,
        total_requests: int
    ) -> float:
        """计算综合性能评分（0-100）"""
        try:
            # 响应时间评分（0-30分）
            if avg_response_time <= self.performance_benchmarks["response_time"]["excellent"]:
                response_score = 30
            elif avg_response_time <= self.performance_benchmarks["response_time"]["good"]:
                response_score = 25
            elif avg_response_time <= self.performance_benchmarks["response_time"]["poor"]:
                response_score = 15
            else:
                response_score = 5

            # 成功率评分（0-40分）
            if success_rate >= self.performance_benchmarks["success_rate"]["excellent"]:
                success_score = 40
            elif success_rate >= self.performance_benchmarks["success_rate"]["good"]:
                success_score = 30
            elif success_rate >= self.performance_benchmarks["success_rate"]["poor"]:
                success_score = 20
            else:
                success_score = 5

            # 成本效率评分（0-30分）
            cost_per_token = avg_cost_per_request / 1000  # 转换为每千token成本
            if cost_per_token <= self.performance_benchmarks["cost_per_token"]["excellent"]:
                cost_score = 30
            elif cost_per_token <= self.performance_benchmarks["cost_per_token"]["good"]:
                cost_score = 25
            elif cost_per_token <= self.performance_benchmarks["cost_per_token"]["poor"]:
                cost_score = 15
            else:
                cost_score = 5

            # 数据量权重（请求越多，评分越可靠）
            data_weight = min(1.0, total_requests / 100)  # 100个请求达到最大权重

            total_score = (response_score + success_score + cost_score) * data_weight

            return min(100.0, total_score)

        except Exception:
            return 50.0  # 默认评分

    def get_model_performance(self, model_name: str) -> Optional[ModelPerformance]:
        """
        获取模型性能数据

        Args:
            model_name: 模型名称

        Returns:
            Optional[ModelPerformance]: 模型性能数据
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM model_performance_summary WHERE model_name = ?
            """, (model_name,))

            row = cursor.fetchone()
            conn.close()

            if not row:
                return None

            # 解析数据库行
            columns = [description[0] for description in cursor.description]
            data = dict(zip(columns, row))

            return ModelPerformance(
                model_name=data["model_name"],
                provider=data["provider"],
                total_requests=data["total_requests"],
                successful_requests=data["successful_requests"],
                failed_requests=data["failed_requests"],
                total_response_time=data["total_response_time"],
                avg_response_time=data["avg_response_time"],
                min_response_time=data["min_response_time"],
                max_response_time=data["max_response_time"],
                total_tokens=data["total_tokens"],
                total_cost=data["total_cost"],
                avg_cost_per_request=data["avg_cost_per_request"],
                avg_tokens_per_request=data["avg_tokens_per_request"],
                last_used=data["last_updated"],
                performance_score=data["performance_score"]
            )

        except Exception as e:
            logger.error(f"获取模型性能失败: {e}")
            return None

    def list_model_performances(
        self,
        min_requests: int = 5,
        sort_by: str = "performance_score",
        descending: bool = True
    ) -> List[ModelPerformance]:
        """
        列出所有模型性能数据

        Args:
            min_requests: 最小请求数要求
            sort_by: 排序字段
            descending: 是否降序

        Returns:
            List[ModelPerformance]: 模型性能列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 验证排序字段
            valid_sort_fields = [
                "model_name", "provider", "total_requests",
                "performance_score", "avg_response_time",
                "successful_requests", "total_cost", "avg_cost_per_request"
            ]

            if sort_by not in valid_sort_fields:
                sort_by = "performance_score"

            order = "DESC" if descending else "ASC"

            cursor.execute(f"""
                SELECT * FROM model_performance_summary
                WHERE total_requests >= ?
                ORDER BY {sort_by} {order}
            """, (min_requests,))

            rows = cursor.fetchall()
            conn.close()

            performances = []
            for row in rows:
                columns = [description[0] for description in cursor.description]
                data = dict(zip(columns, row))

                performances.append(ModelPerformance(
                    model_name=data["model_name"],
                    provider=data["provider"],
                    total_requests=data["total_requests"],
                    successful_requests=data["successful_requests"],
                    failed_requests=data["failed_requests"],
                    total_response_time=data["total_response_time"],
                    avg_response_time=data["avg_response_time"],
                    min_response_time=data["min_response_time"],
                    max_response_time=data["max_response_time"],
                    total_tokens=data["total_tokens"],
                    total_cost=data["total_cost"],
                    avg_cost_per_request=data["avg_cost_per_request"],
                    avg_tokens_per_request=data["avg_tokens_per_request"],
                    last_used=data["last_updated"],
                    performance_score=data["performance_score"]
                ))

            return performances

        except Exception as e:
            logger.error(f"列出模型性能失败: {e}")
            return []

    def get_performance_trends(
        self,
        model_name: str,
        days: int = 7
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        获取模型性能趋势

        Args:
            model_name: 模型名称
            days: 天数

        Returns:
            Dict[str, List[Tuple[str, float]]]: 性能趋势数据
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            # 获取每日数据
            cursor.execute("""
                SELECT
                    DATE(timestamp) as date,
                    COUNT(*) as daily_requests,
                    AVG(response_time) as avg_response_time,
                    AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as success_rate,
                    SUM(cost_usd) as daily_cost,
                    SUM(total_tokens) as daily_tokens
                FROM model_performance
                WHERE model_name = ? AND timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, (model_name, cutoff_date))

            rows = cursor.fetchall()
            conn.close()

            # 组织趋势数据
            trends = {
                "response_time": [],
                "success_rate": [],
                "daily_cost": [],
                "daily_tokens": [],
                "daily_requests": []
            }

            for row in rows:
                date_str = row[0]
                daily_requests = row[1] or 0
                avg_response_time = row[2] or 0.0
                success_rate = row[3] or 0.0
                daily_cost = row[4] or 0.0
                daily_tokens = row[5] or 0

                trends["response_time"].append((date_str, avg_response_time))
                trends["success_rate"].append((date_str, success_rate * 100))  # 转换为百分比
                trends["daily_cost"].append((date_str, daily_cost))
                trends["daily_tokens"].append((date_str, daily_tokens))
                trends["daily_requests"].append((date_str, daily_requests))

            return trends

        except Exception as e:
            logger.error(f"获取性能趋势失败: {e}")
            return {}

    def recommend_model(
        self,
        task_type: str,
        budget_constraint: Optional[float] = None,
        time_constraint: Optional[float] = None,
        quality_requirement: str = "balanced"  # balanced, high_quality, cost_effective
    ) -> ModelRecommendation:
        """
        推荐最适合的模型

        Args:
            task_type: 任务类型
            budget_constraint: 预算约束（美元）
            time_constraint: 时间约束（秒）
            quality_requirement: 质量要求

        Returns:
            ModelRecommendation: 模型推荐
        """
        try:
            # 获取候选模型
            candidate_models = self.task_model_mapping.get(task_type, [])
            if not candidate_models:
                candidate_models = ["gpt-3.5-turbo", "claude-3-haiku", "deepseek-chat"]

            # 获取模型性能数据
            model_performances = []
            for model_name in candidate_models:
                perf = self.get_model_performance(model_name)
                if perf and perf.total_requests >= 5:  # 至少有5个请求的数据
                    model_performances.append(perf)

            # 如果没有足够数据，使用默认推荐
            if not model_performances:
                return self._create_default_recommendation(task_type, candidate_models)

            # 根据约束和要求筛选和排序
            filtered_models = []
            for perf in model_performances:
                # 预算约束
                if budget_constraint and perf.avg_cost_per_request > budget_constraint:
                    continue

                # 时间约束
                if time_constraint and perf.avg_response_time > time_constraint:
                    continue

                filtered_models.append(perf)

            # 如果没有满足约束的模型，放宽约束
            if not filtered_models:
                filtered_models = model_performances

            # 根据质量要求排序
            if quality_requirement == "high_quality":
                # 优先性能评分和成功率
                filtered_models.sort(key=lambda x: (x.performance_score, x.successful_requests/x.total_requests if x.total_requests > 0 else 0), reverse=True)
            elif quality_requirement == "cost_effective":
                # 优先成本和token效率
                filtered_models.sort(key=lambda x: (x.avg_cost_per_request, -x.avg_tokens_per_request))
            else:  # balanced
                # 平衡考虑
                filtered_models.sort(key=lambda x: x.performance_score, reverse=True)

            # 选择推荐模型
            recommended_model = filtered_models[0] if filtered_models else model_performances[0]

            # 备选模型（排除推荐模型）
            alternative_models = [
                perf.model_name for perf in filtered_models[1:3]
                if perf.model_name != recommended_model.model_name
            ]

            # 估算成本和时间
            estimated_cost = recommended_model.avg_cost_per_request
            estimated_time = recommended_model.avg_response_time

            # 生成推荐理由
            reasoning = self._generate_recommendation_reasoning(
                recommended_model,
                task_type,
                budget_constraint,
                time_constraint,
                quality_requirement
            )

            # 计算置信度
            confidence_score = self._calculate_confidence_score(recommended_model)

            return ModelRecommendation(
                task_type=task_type,
                recommended_model=recommended_model.model_name,
                alternative_models=alternative_models,
                reasoning=reasoning,
                estimated_cost=estimated_cost,
                estimated_time=estimated_time,
                confidence_score=confidence_score
            )

        except Exception as e:
            logger.error(f"推荐模型失败: {e}")
            # 返回默认推荐
            return self._create_default_recommendation(task_type, ["gpt-3.5-turbo"])

    def _create_default_recommendation(
        self,
        task_type: str,
        candidate_models: List[str]
    ) -> ModelRecommendation:
        """创建默认推荐"""
        default_model = candidate_models[0] if candidate_models else "gpt-3.5-turbo"

        # 默认成本和时间估算
        default_costs = {
            "gpt-3.5-turbo": 0.002,
            "claude-3-haiku": 0.001,
            "deepseek-chat": 0.0003,
            "gpt-4": 0.03,
            "claude-3-sonnet": 0.015,
            "claude-3-opus": 0.075
        }

        default_times = {
            "gpt-3.5-turbo": 2.0,
            "claude-3-haiku": 1.5,
            "deepseek-chat": 3.0,
            "gpt-4": 5.0,
            "claude-3-sonnet": 3.0,
            "claude-3-opus": 8.0
        }

        estimated_cost = default_costs.get(default_model, 0.01)
        estimated_time = default_times.get(default_model, 3.0)

        return ModelRecommendation(
            task_type=task_type,
            recommended_model=default_model,
            alternative_models=candidate_models[1:3] if len(candidate_models) > 1 else [],
            reasoning=f"基于任务类型 '{task_type}' 的默认推荐。建议收集更多使用数据以获得更精准的推荐。",
            estimated_cost=estimated_cost,
            estimated_time=estimated_time,
            confidence_score=0.5  # 低置信度
        )

    def _generate_recommendation_reasoning(
        self,
        model_perf: ModelPerformance,
        task_type: str,
        budget_constraint: Optional[float],
        time_constraint: Optional[float],
        quality_requirement: str
    ) -> str:
        """生成推荐理由"""
        reasons = []

        # 基本性能
        reasons.append(f"模型 '{model_perf.model_name}' 在历史使用中表现稳定")

        # 成功率
        if model_perf.total_requests > 0:
            success_rate = model_perf.successful_requests / model_perf.total_requests
            if success_rate > 0.95:
                reasons.append("成功率高达 {:.1f}%".format(success_rate * 100))
            elif success_rate > 0.9:
                reasons.append("成功率良好 ({:.1f}%)".format(success_rate * 100))

        # 响应时间
        if model_perf.avg_response_time < 3.0:
            reasons.append(f"响应速度快 ({model_perf.avg_response_time:.1f}秒)")
        elif model_perf.avg_response_time < 8.0:
            reasons.append(f"响应时间适中 ({model_perf.avg_response_time:.1f}秒)")

        # 成本效率
        if model_perf.avg_cost_per_request < 0.01:
            reasons.append(f"成本效益高 (${model_perf.avg_cost_per_request:.4f}/请求)")

        # 任务类型匹配
        reasons.append(f"适合 '{task_type}' 类型的任务")

        # 约束考虑
        if budget_constraint:
            if model_perf.avg_cost_per_request <= budget_constraint:
                reasons.append(f"符合预算约束 (${budget_constraint:.4f})")
            else:
                reasons.append(f"略超预算，但性能更优")

        if time_constraint:
            if model_perf.avg_response_time <= time_constraint:
                reasons.append(f"符合时间约束 ({time_constraint:.1f}秒)")
            else:
                reasons.append(f"响应时间稍长，但输出质量更高")

        # 质量要求
        if quality_requirement == "high_quality":
            reasons.append("优先考虑输出质量")
        elif quality_requirement == "cost_effective":
            reasons.append("优先考虑成本效益")

        return "；".join(reasons)

    def _calculate_confidence_score(self, model_perf: ModelPerformance) -> float:
        """计算推荐置信度"""
        try:
            # 基于数据量的置信度
            data_confidence = min(1.0, model_perf.total_requests / 50)  # 50个请求达到最大置信度

            # 基于性能稳定性的置信度
            if model_perf.total_requests >= 10:
                # 计算变异系数（标准差/均值）
                # 这里简化处理，使用成功率稳定性
                success_rate = model_perf.successful_requests / model_perf.total_requests
                stability_confidence = success_rate  # 成功率越高越稳定
            else:
                stability_confidence = 0.5

            # 综合置信度
            confidence = (data_confidence * 0.6) + (stability_confidence * 0.4)

            return min(1.0, max(0.3, confidence))  # 确保在0.3-1.0之间

        except Exception:
            return 0.5

    def get_performance_insights(self) -> List[str]:
        """
        获取性能洞察

        Returns:
            List[str]: 洞察列表
        """
        insights = []

        try:
            # 获取所有模型性能
            model_performances = self.list_model_performances(min_requests=10)

            if not model_performances:
                return ["暂无足够的性能数据进行分析"]

            # 分析最佳性能模型
            best_model = max(model_performances, key=lambda x: x.performance_score)
            insights.append(f"最佳性能模型: {best_model.model_name} (评分: {best_model.performance_score:.1f}/100)")

            # 分析最经济模型
            if model_performances:
                economical_model = min(model_performances, key=lambda x: x.avg_cost_per_request)
                insights.append(f"最经济模型: {economical_model.model_name} (${economical_model.avg_cost_per_request:.4f}/请求)")

            # 分析最快模型
                fastest_model = min(model_performances, key=lambda x: x.avg_response_time)
                insights.append(f"最快模型: {fastest_model.model_name} ({fastest_model.avg_response_time:.1f}秒/请求)")

            # 检查问题模型
            problem_models = [
                m for m in model_performances
                if m.total_requests >= 10 and m.successful_requests / m.total_requests < 0.8
            ]
            for model in problem_models:
                success_rate = model.successful_requests / model.total_requests * 100
                insights.append(f"警告: {model.model_name} 成功率较低 ({success_rate:.1f}%)")

            # 成本分析
            total_cost = sum(m.total_cost for m in model_performances)
            if total_cost > 10:  # 如果总成本超过10美元
                insights.append(f"总API成本: ${total_cost:.2f}，建议设置预算限制")

            # 使用频率分析
            total_requests = sum(m.total_requests for m in model_performances)
            if total_requests > 100:
                most_used = max(model_performances, key=lambda x: x.total_requests)
                usage_percentage = most_used.total_requests / total_requests * 100
                insights.append(f"最常用模型: {most_used.model_name} ({usage_percentage:.1f}% 的使用量)")

        except Exception as e:
            logger.error(f"生成性能洞察失败: {e}")
            insights = ["生成性能洞察时发生错误"]

        return insights

    def export_performance_data(
        self,
        format: str = "json",
        file_path: Optional[Path] = None
    ) -> str:
        """
        导出性能数据

        Args:
            format: 导出格式（json, csv）
            file_path: 导出文件路径

        Returns:
            str: 导出内容或文件路径
        """
        try:
            # 获取所有性能数据
            model_performances = self.list_model_performances(min_requests=0)

            if format.lower() == "csv":
                import csv
                from io import StringIO

                output = StringIO()
                writer = csv.writer(output)

                # 写入标题
                writer.writerow([
                    "model_name", "provider", "total_requests", "successful_requests",
                    "failed_requests", "avg_response_time", "min_response_time",
                    "max_response_time", "total_tokens", "total_cost",
                    "avg_cost_per_request", "avg_tokens_per_request", "last_used",
                    "performance_score"
                ])

                # 写入数据
                for perf in model_performances:
                    writer.writerow([
                        perf.model_name,
                        perf.provider,
                        perf.total_requests,
                        perf.successful_requests,
                        perf.failed_requests,
                        f"{perf.avg_response_time:.3f}",
                        f"{perf.min_response_time:.3f}",
                        f"{perf.max_response_time:.3f}",
                        perf.total_tokens,
                        f"{perf.total_cost:.6f}",
                        f"{perf.avg_cost_per_request:.6f}",
                        perf.avg_tokens_per_request,
                        perf.last_used or "",
                        f"{perf.performance_score:.1f}"
                    ])

                content = output.getvalue()

            else:  # JSON
                data = [asdict(perf) for perf in model_performances]
                content = json.dumps(data, ensure_ascii=False, indent=2)

            if file_path:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return str(file_path)
            else:
                return content

        except Exception as e:
            logger.error(f"导出性能数据失败: {e}")
            raise

    def clear_old_data(self, days_to_keep: int = 90):
        """
        清理旧数据

        Args:
            days_to_keep: 保留天数
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()

            # 删除旧记录
            cursor.execute("DELETE FROM model_performance WHERE timestamp < ?", (cutoff_date,))
            deleted_count = cursor.rowcount

            # 重新计算汇总
            cursor.execute("DELETE FROM model_performance_summary")
            cursor.execute("SELECT DISTINCT model_name FROM model_performance")
            model_names = [row[0] for row in cursor.fetchall()]

            for model_name in model_names:
                self._update_performance_summary(model_name)

            conn.commit()
            conn.close()

            logger.info(f"清理了 {deleted_count} 条超过 {days_to_keep} 天的性能记录")

        except Exception as e:
            logger.error(f"清理旧数据失败: {e}")


# 全局模型监控器实例
_model_monitor_instance: Optional[ModelMonitor] = None

def get_model_monitor() -> ModelMonitor:
    """获取全局模型监控器实例"""
    global _model_monitor_instance
    if _model_monitor_instance is None:
        _model_monitor_instance = ModelMonitor()
    return _model_monitor_instance