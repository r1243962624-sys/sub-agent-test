"""
成本监控模块
实时监控API调用成本，提供预算控制和优化建议
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from loguru import logger

from models.config import ModelProvider
from utils.exceptions import BudgetExceededError


class CostPeriod(str, Enum):
    """成本统计周期"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    TOTAL = "total"


@dataclass
class CostRecord:
    """成本记录"""
    id: Optional[int] = None
    timestamp: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    operation_type: str = "generate"  # generate, stream, batch
    template_name: Optional[str] = None
    video_id: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class CostSummary:
    """成本汇总"""
    period: CostPeriod
    start_date: str
    end_date: str
    total_cost: float = 0.0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    request_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    cost_by_model: Dict[str, float] = None
    cost_by_provider: Dict[str, float] = None
    cost_by_template: Dict[str, float] = None


class CostMonitor:
    """成本监控器"""

    def __init__(self, db_path: Optional[Path] = None):
        """
        初始化成本监控器

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path or Path("./data/costs.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 预算设置
        self.daily_budget: Optional[float] = None
        self.monthly_budget: Optional[float] = None
        self.total_budget: Optional[float] = None

        # 价格表（美元/1000 tokens）
        self.pricing_table = {
            # OpenAI
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
            "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
            # Anthropic
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
            # DeepSeek
            "deepseek-chat": {"input": 0.00014, "output": 0.00028},
            "deepseek-coder": {"input": 0.00014, "output": 0.00028},
            "deepseek-reasoner": {"input": 0.00028, "output": 0.00056},
        }

        # 初始化数据库
        self._init_database()

        logger.info(f"成本监控器初始化完成，数据库: {self.db_path}")

    def _init_database(self):
        """初始化数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 创建成本记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cost_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    provider TEXT,
                    model TEXT,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    cost_usd REAL DEFAULT 0.0,
                    operation_type TEXT DEFAULT 'generate',
                    template_name TEXT,
                    video_id TEXT,
                    success BOOLEAN DEFAULT 1,
                    error_message TEXT
                )
            """)

            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON cost_records(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_provider ON cost_records(provider)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_model ON cost_records(model)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_template ON cost_records(template_name)")

            # 创建预算设置表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS budget_settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    setting_key TEXT UNIQUE NOT NULL,
                    setting_value REAL,
                    updated_at TEXT NOT NULL
                )
            """)

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"初始化数据库失败: {e}")
            raise

    def record_cost(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        operation_type: str = "generate",
        template_name: Optional[str] = None,
        video_id: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> CostRecord:
        """
        记录API调用成本

        Args:
            provider: API提供商
            model: 模型名称
            prompt_tokens: 输入token数
            completion_tokens: 输出token数
            operation_type: 操作类型
            template_name: 模板名称
            video_id: 视频ID
            success: 是否成功
            error_message: 错误信息

        Returns:
            CostRecord: 成本记录
        """
        try:
            # 计算成本
            cost = self.calculate_cost(model, prompt_tokens, completion_tokens)

            # 创建记录
            record = CostRecord(
                timestamp=datetime.now().isoformat(),
                provider=provider,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cost_usd=cost,
                operation_type=operation_type,
                template_name=template_name,
                video_id=video_id,
                success=success,
                error_message=error_message
            )

            # 保存到数据库
            self._save_record(record)

            # 检查预算
            self._check_budget()

            logger.debug(f"记录成本: {provider}/{model} - {cost:.6f} USD")
            return record

        except Exception as e:
            logger.error(f"记录成本失败: {e}")
            raise

    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """
        计算API调用成本

        Args:
            model: 模型名称
            prompt_tokens: 输入token数
            completion_tokens: 输出token数

        Returns:
            float: 成本（美元）
        """
        if model not in self.pricing_table:
            # 尝试模糊匹配
            for key in self.pricing_table:
                if model.startswith(key.split("-")[0]):
                    model = key
                    break

        if model not in self.pricing_table:
            logger.warning(f"未知模型价格: {model}")
            return 0.0

        price = self.pricing_table[model]
        cost = (prompt_tokens / 1000 * price["input"]) + (completion_tokens / 1000 * price["output"])
        return round(cost, 6)

    def _save_record(self, record: CostRecord):
        """保存记录到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO cost_records (
                    timestamp, provider, model, prompt_tokens, completion_tokens,
                    total_tokens, cost_usd, operation_type, template_name,
                    video_id, success, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.timestamp, record.provider, record.model,
                record.prompt_tokens, record.completion_tokens,
                record.total_tokens, record.cost_usd, record.operation_type,
                record.template_name, record.video_id, record.success,
                record.error_message
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"保存成本记录失败: {e}")
            raise

    def set_budget(self, daily: Optional[float] = None, monthly: Optional[float] = None, total: Optional[float] = None):
        """
        设置预算

        Args:
            daily: 每日预算（美元）
            monthly: 每月预算（美元）
            total: 总预算（美元）
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            if daily is not None:
                self.daily_budget = daily
                cursor.execute(
                    "INSERT OR REPLACE INTO budget_settings (setting_key, setting_value, updated_at) VALUES (?, ?, ?)",
                    ("daily_budget", daily, now)
                )

            if monthly is not None:
                self.monthly_budget = monthly
                cursor.execute(
                    "INSERT OR REPLACE INTO budget_settings (setting_key, setting_value, updated_at) VALUES (?, ?, ?)",
                    ("monthly_budget", monthly, now)
                )

            if total is not None:
                self.total_budget = total
                cursor.execute(
                    "INSERT OR REPLACE INTO budget_settings (setting_key, setting_value, updated_at) VALUES (?, ?, ?)",
                    ("total_budget", total, now)
                )

            conn.commit()
            conn.close()

            logger.info(f"预算设置更新: 每日=${daily}, 每月=${monthly}, 总计=${total}")

        except Exception as e:
            logger.error(f"设置预算失败: {e}")
            raise

    def _check_budget(self):
        """检查预算是否超支"""
        try:
            # 加载预算设置
            self._load_budget_settings()

            if not any([self.daily_budget, self.monthly_budget, self.total_budget]):
                return

            # 检查每日预算
            if self.daily_budget:
                today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                daily_cost = self.get_period_cost(today_start.isoformat(), datetime.now().isoformat())
                if daily_cost > self.daily_budget:
                    raise BudgetExceededError(f"每日预算超支: ${daily_cost:.2f} > ${self.daily_budget:.2f}")

            # 检查每月预算
            if self.monthly_budget:
                month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                monthly_cost = self.get_period_cost(month_start.isoformat(), datetime.now().isoformat())
                if monthly_cost > self.monthly_budget:
                    raise BudgetExceededError(f"每月预算超支: ${monthly_cost:.2f} > ${self.monthly_budget:.2f}")

            # 检查总预算
            if self.total_budget:
                total_cost = self.get_period_cost(None, None)  # 所有时间
                if total_cost > self.total_budget:
                    raise BudgetExceededError(f"总预算超支: ${total_cost:.2f} > ${self.total_budget:.2f}")

        except BudgetExceededError:
            raise
        except Exception as e:
            logger.warning(f"检查预算失败: {e}")

    def _load_budget_settings(self):
        """从数据库加载预算设置"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT setting_key, setting_value FROM budget_settings")
            rows = cursor.fetchall()
            conn.close()

            for key, value in rows:
                if value is not None:
                    if key == "daily_budget":
                        self.daily_budget = float(value)
                    elif key == "monthly_budget":
                        self.monthly_budget = float(value)
                    elif key == "total_budget":
                        self.total_budget = float(value)

        except Exception as e:
            logger.warning(f"加载预算设置失败: {e}")

    def get_period_cost(self, start_date: Optional[str], end_date: Optional[str]) -> float:
        """
        获取指定时间段的成本

        Args:
            start_date: 开始时间（ISO格式）
            end_date: 结束时间（ISO格式）

        Returns:
            float: 总成本（美元）
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            query = "SELECT SUM(cost_usd) FROM cost_records WHERE 1=1"
            params = []

            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)

            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)

            cursor.execute(query, params)
            result = cursor.fetchone()
            conn.close()

            return float(result[0] or 0.0)

        except Exception as e:
            logger.error(f"获取时间段成本失败: {e}")
            return 0.0

    def get_cost_summary(self, period: CostPeriod = CostPeriod.DAILY) -> CostSummary:
        """
        获取成本汇总

        Args:
            period: 统计周期

        Returns:
            CostSummary: 成本汇总
        """
        try:
            now = datetime.now()

            if period == CostPeriod.DAILY:
                start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = now
            elif period == CostPeriod.WEEKLY:
                start_date = now - timedelta(days=now.weekday())
                start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = now
            elif period == CostPeriod.MONTHLY:
                start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                end_date = now
            else:  # TOTAL
                start_date = None
                end_date = None

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 基础统计
            query = """
                SELECT
                    COUNT(*) as request_count,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failure_count,
                    SUM(prompt_tokens) as total_prompt_tokens,
                    SUM(completion_tokens) as total_completion_tokens,
                    SUM(total_tokens) as total_tokens,
                    SUM(cost_usd) as total_cost
                FROM cost_records
            """
            params = []

            if start_date:
                query += " WHERE timestamp >= ?"
                params.append(start_date.isoformat())

            if end_date and start_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())

            cursor.execute(query, params)
            row = cursor.fetchone()

            # 按模型统计
            cursor.execute(f"""
                SELECT model, SUM(cost_usd) as model_cost
                FROM cost_records
                {f"WHERE timestamp >= '{start_date.isoformat()}' AND timestamp <= '{end_date.isoformat()}'" if start_date else ""}
                GROUP BY model
                ORDER BY model_cost DESC
            """)
            cost_by_model = {row[0]: row[1] for row in cursor.fetchall()}

            # 按提供商统计
            cursor.execute(f"""
                SELECT provider, SUM(cost_usd) as provider_cost
                FROM cost_records
                {f"WHERE timestamp >= '{start_date.isoformat()}' AND timestamp <= '{end_date.isoformat()}'" if start_date else ""}
                GROUP BY provider
                ORDER BY provider_cost DESC
            """)
            cost_by_provider = {row[0]: row[1] for row in cursor.fetchall()}

            # 按模板统计
            cursor.execute(f"""
                SELECT template_name, SUM(cost_usd) as template_cost
                FROM cost_records
                WHERE template_name IS NOT NULL
                {f"AND timestamp >= '{start_date.isoformat()}' AND timestamp <= '{end_date.isoformat()}'" if start_date else ""}
                GROUP BY template_name
                ORDER BY template_cost DESC
            """)
            cost_by_template = {row[0]: row[1] for row in cursor.fetchall()}

            conn.close()

            summary = CostSummary(
                period=period,
                start_date=start_date.isoformat() if start_date else "",
                end_date=end_date.isoformat() if end_date else "",
                total_cost=row[6] or 0.0,
                total_tokens=row[5] or 0,
                prompt_tokens=row[3] or 0,
                completion_tokens=row[4] or 0,
                request_count=row[0] or 0,
                success_count=row[1] or 0,
                failure_count=row[2] or 0,
                cost_by_model=cost_by_model,
                cost_by_provider=cost_by_provider,
                cost_by_template=cost_by_template
            )

            return summary

        except Exception as e:
            logger.error(f"获取成本汇总失败: {e}")
            # 返回空汇总
            return CostSummary(
                period=period,
                start_date="",
                end_date=""
            )

    def get_cost_breakdown(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        获取成本分解（按天）

        Args:
            days: 天数

        Returns:
            List[Dict[str, Any]]: 每日成本分解
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    DATE(timestamp) as date,
                    SUM(cost_usd) as daily_cost,
                    SUM(total_tokens) as daily_tokens,
                    COUNT(*) as daily_requests
                FROM cost_records
                WHERE timestamp >= date('now', ?)
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """, (f"-{days} days",))

            breakdown = []
            for row in cursor.fetchall():
                breakdown.append({
                    "date": row[0],
                    "cost": row[1] or 0.0,
                    "tokens": row[2] or 0,
                    "requests": row[3] or 0
                })

            conn.close()
            return breakdown

        except Exception as e:
            logger.error(f"获取成本分解失败: {e}")
            return []

    def get_optimization_suggestions(self) -> List[str]:
        """
        获取成本优化建议

        Returns:
            List[str]: 优化建议列表
        """
        suggestions = []

        try:
            # 获取最近30天的数据
            summary = self.get_cost_summary(CostPeriod.MONTHLY)

            if summary.total_cost == 0:
                return ["暂无成本数据，无法提供优化建议"]

            # 分析成本分布
            if summary.cost_by_model:
                most_expensive_model = max(summary.cost_by_model.items(), key=lambda x: x[1])
                suggestions.append(f"最昂贵的模型是 {most_expensive_model[0]}，占总成本的 {most_expensive_model[1]/summary.total_cost*100:.1f}%")

            # 检查失败率
            if summary.request_count > 0:
                failure_rate = summary.failure_count / summary.request_count * 100
                if failure_rate > 10:
                    suggestions.append(f"API调用失败率较高 ({failure_rate:.1f}%)，建议检查网络连接和API密钥")

            # 检查token使用效率
            if summary.total_tokens > 0:
                avg_cost_per_token = summary.total_cost / summary.total_tokens * 1000
                suggestions.append(f"平均每千token成本: ${avg_cost_per_token:.4f}")

            # 检查是否有高成本模板
            if summary.cost_by_template:
                expensive_templates = [(k, v) for k, v in summary.cost_by_template.items() if v > summary.total_cost * 0.2]
                for template, cost in expensive_templates:
                    suggestions.append(f"模板 '{template}' 成本较高 (${cost:.2f})，考虑优化其prompt")

            # 通用建议
            suggestions.extend([
                "考虑使用更经济的模型（如gpt-3.5-turbo或claude-3-haiku）",
                "优化prompt长度，减少不必要的上下文",
                "批量处理视频以减少API调用次数",
                "设置预算限制以避免意外高额费用"
            ])

        except Exception as e:
            logger.error(f"生成优化建议失败: {e}")
            suggestions = ["无法生成优化建议"]

        return suggestions

    def export_cost_data(self, format: str = "json", file_path: Optional[Path] = None) -> str:
        """
        导出成本数据

        Args:
            format: 导出格式（json, csv）
            file_path: 导出文件路径

        Returns:
            str: 导出内容或文件路径
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM cost_records ORDER BY timestamp DESC")
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            conn.close()

            if format.lower() == "csv":
                import csv
                from io import StringIO

                output = StringIO()
                writer = csv.writer(output)
                writer.writerow(columns)
                writer.writerows(rows)

                content = output.getvalue()

            else:  # JSON
                data = []
                for row in rows:
                    data.append(dict(zip(columns, row)))

                content = json.dumps(data, ensure_ascii=False, indent=2)

            if file_path:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return str(file_path)
            else:
                return content

        except Exception as e:
            logger.error(f"导出成本数据失败: {e}")
            raise

    def clear_old_records(self, days_to_keep: int = 90):
        """
        清理旧记录

        Args:
            days_to_keep: 保留天数
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
            cursor.execute("DELETE FROM cost_records WHERE timestamp < ?", (cutoff_date,))

            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()

            logger.info(f"清理了 {deleted_count} 条超过 {days_to_keep} 天的成本记录")

        except Exception as e:
            logger.error(f"清理旧记录失败: {e}")
            raise


# 全局成本监控器实例
_cost_monitor_instance: Optional[CostMonitor] = None

def get_cost_monitor() -> CostMonitor:
    """获取全局成本监控器实例"""
    global _cost_monitor_instance
    if _cost_monitor_instance is None:
        _cost_monitor_instance = CostMonitor()
    return _cost_monitor_instance