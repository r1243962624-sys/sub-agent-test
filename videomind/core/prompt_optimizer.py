"""
Prompt优化模块
智能优化prompt，提高输出质量和成本效率
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from loguru import logger

from core.llm_client import LLMClient
from models.config import APIConfig


class PromptOptimizationLevel(str, Enum):
    """Prompt优化级别"""
    MINIMAL = "minimal"      # 最小优化，保持原意
    BALANCED = "balanced"    # 平衡优化，兼顾质量和成本
    AGGRESSIVE = "aggressive"  # 激进优化，最大程度压缩


@dataclass
class PromptAnalysis:
    """Prompt分析结果"""
    original_length: int
    optimized_length: int
    token_estimate: int
    cost_estimate: float
    readability_score: float  # 0-1，可读性评分
    clarity_score: float     # 0-1，清晰度评分
    specificity_score: float  # 0-1，具体性评分
    issues_found: List[str]
    suggestions: List[str]


@dataclass
class OptimizationResult:
    """优化结果"""
    original_prompt: str
    optimized_prompt: str
    analysis: PromptAnalysis
    optimization_level: PromptOptimizationLevel
    model_used: str
    tokens_saved: int
    cost_saved: float


class PromptOptimizer:
    """Prompt优化器"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        初始化Prompt优化器

        Args:
            llm_client: LLM客户端（可选）
        """
        self.llm_client = llm_client
        self.optimization_history: List[OptimizationResult] = []

        # 优化规则
        self.optimization_rules = {
            "remove_redundant_words": True,
            "simplify_sentence_structure": True,
            "remove_unnecessary_context": True,
            "optimize_formatting": True,
            "add_specific_instructions": True,
            "improve_clarity": True,
        }

        logger.info("Prompt优化器初始化完成")

    def optimize_prompt(
        self,
        prompt: str,
        optimization_level: PromptOptimizationLevel = PromptOptimizationLevel.BALANCED,
        target_model: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """
        优化Prompt

        Args:
            prompt: 原始prompt
            optimization_level: 优化级别
            target_model: 目标模型
            context: 上下文信息

        Returns:
            OptimizationResult: 优化结果
        """
        try:
            logger.info(f"开始优化prompt，级别: {optimization_level.value}")

            # 分析原始prompt
            analysis = self._analyze_prompt(prompt, context)

            # 根据优化级别选择优化策略
            if optimization_level == PromptOptimizationLevel.MINIMAL:
                optimized_prompt = self._minimal_optimization(prompt, analysis)
            elif optimization_level == PromptOptimizationLevel.AGGRESSIVE:
                optimized_prompt = self._aggressive_optimization(prompt, analysis, target_model)
            else:  # BALANCED
                optimized_prompt = self._balanced_optimization(prompt, analysis, target_model)

            # 分析优化后的prompt
            optimized_analysis = self._analyze_prompt(optimized_prompt, context)

            # 估算token和成本节省
            tokens_saved = analysis.token_estimate - optimized_analysis.token_estimate
            cost_saved = analysis.cost_estimate - optimized_analysis.cost_estimate

            # 创建结果
            result = OptimizationResult(
                original_prompt=prompt,
                optimized_prompt=optimized_prompt,
                analysis=optimized_analysis,
                optimization_level=optimization_level,
                model_used=target_model or "gpt-3.5-turbo",
                tokens_saved=tokens_saved,
                cost_saved=cost_saved
            )

            # 保存到历史
            self.optimization_history.append(result)

            logger.info(f"Prompt优化完成，节省 {tokens_saved} tokens (${cost_saved:.4f})")
            return result

        except Exception as e:
            logger.error(f"优化prompt失败: {e}")
            raise

    def _analyze_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> PromptAnalysis:
        """
        分析Prompt

        Args:
            prompt: 要分析的prompt
            context: 上下文信息

        Returns:
            PromptAnalysis: 分析结果
        """
        try:
            # 基础分析
            length = len(prompt)
            word_count = len(prompt.split())
            sentence_count = len(re.split(r'[.!?]+', prompt))

            # 估算token数（简单估算：1 token ≈ 4字符）
            token_estimate = len(prompt) // 4

            # 估算成本（基于gpt-3.5-turbo）
            cost_estimate = token_estimate / 1000 * 0.001

            # 分析问题
            issues = self._identify_issues(prompt)

            # 计算评分
            readability_score = self._calculate_readability_score(prompt, word_count, sentence_count)
            clarity_score = self._calculate_clarity_score(prompt, issues)
            specificity_score = self._calculate_specificity_score(prompt, context)

            # 生成建议
            suggestions = self._generate_suggestions(prompt, issues, context)

            return PromptAnalysis(
                original_length=length,
                optimized_length=length,  # 初始时相同
                token_estimate=token_estimate,
                cost_estimate=cost_estimate,
                readability_score=readability_score,
                clarity_score=clarity_score,
                specificity_score=specificity_score,
                issues_found=issues,
                suggestions=suggestions
            )

        except Exception as e:
            logger.error(f"分析prompt失败: {e}")
            # 返回基本分析结果
            return PromptAnalysis(
                original_length=len(prompt),
                optimized_length=len(prompt),
                token_estimate=len(prompt) // 4,
                cost_estimate=0.0,
                readability_score=0.5,
                clarity_score=0.5,
                specificity_score=0.5,
                issues_found=[],
                suggestions=[]
            )

    def _identify_issues(self, prompt: str) -> List[str]:
        """识别prompt中的问题"""
        issues = []

        # 检查长度
        if len(prompt) > 2000:
            issues.append("Prompt过长，可能超出模型上下文限制")

        # 检查模糊指令
        vague_patterns = [
            r"尽可能.*好", r"尽量.*", r"适当.*", r"合适.*",
            r"高质量", r"优秀", r"完美", r"最佳"
        ]
        for pattern in vague_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                issues.append("包含模糊指令，应更具体")
                break

        # 检查重复内容
        sentences = re.split(r'[.!?]+', prompt)
        if len(sentences) > 5:
            # 检查句子相似度
            for i in range(len(sentences) - 1):
                for j in range(i + 1, len(sentences)):
                    if self._calculate_similarity(sentences[i], sentences[j]) > 0.8:
                        issues.append("可能存在重复或冗余内容")
                        break

        # 检查格式问题
        if not re.search(r'#+ |\*\*.*\*\*|\[.*\]|\(.*\)', prompt):
            issues.append("缺乏结构化格式，考虑使用标题、列表等")

        # 检查具体性
        if len(re.findall(r'\d+', prompt)) < 2 and len(re.findall(r'具体|详细|明确', prompt)) < 1:
            issues.append("缺乏具体要求和细节")

        return issues

    def _calculate_readability_score(self, prompt: str, word_count: int, sentence_count: int) -> float:
        """计算可读性评分"""
        try:
            # 基于Flesch Reading Ease的简化版本
            if sentence_count == 0 or word_count == 0:
                return 0.5

            avg_sentence_length = word_count / sentence_count
            avg_word_length = len(prompt.replace(" ", "")) / word_count

            # 理想范围：句子长度10-20词，单词长度4-6字母
            sentence_score = max(0, 1 - abs(avg_sentence_length - 15) / 15)
            word_score = max(0, 1 - abs(avg_word_length - 5) / 3)

            return (sentence_score + word_score) / 2

        except Exception:
            return 0.5

    def _calculate_clarity_score(self, prompt: str, issues: List[str]) -> float:
        """计算清晰度评分"""
        base_score = 0.7  # 基础分

        # 根据问题扣分
        penalty = len(issues) * 0.1
        clarity_score = max(0.1, base_score - penalty)

        # 检查是否有明确指令
        if re.search(r'请.*|要求.*|必须.*|应该.*', prompt):
            clarity_score += 0.1

        # 检查是否有具体示例
        if re.search(r'例如.*|比如.*|示例.*|案例.*', prompt):
            clarity_score += 0.1

        return min(1.0, clarity_score)

    def _calculate_specificity_score(self, prompt: str, context: Optional[Dict[str, Any]]) -> float:
        """计算具体性评分"""
        score = 0.5

        # 检查数字和具体细节
        numbers = re.findall(r'\d+', prompt)
        if len(numbers) >= 2:
            score += 0.2

        # 检查具体名词
        specific_terms = re.findall(r'[\u4e00-\u9fff]{2,4}会议|[\u4e00-\u9fff]{2,4}报告|[\u4e00-\u9fff]{2,4}分析', prompt)
        if len(specific_terms) >= 1:
            score += 0.1

        # 检查格式要求
        if re.search(r'格式.*|结构.*|模板.*', prompt):
            score += 0.1

        # 检查输出要求
        if re.search(r'输出.*|结果.*|生成.*', prompt):
            score += 0.1

        # 上下文增强
        if context:
            # 检查是否使用了上下文中的具体信息
            context_used = any(str(value) in prompt for value in context.values() if value)
            if context_used:
                score += 0.1

        return min(1.0, score)

    def _generate_suggestions(self, prompt: str, issues: List[str], context: Optional[Dict[str, Any]]) -> List[str]:
        """生成优化建议"""
        suggestions = []

        # 基于问题生成建议
        for issue in issues:
            if "过长" in issue:
                suggestions.append("考虑缩短prompt，移除不必要的内容")
            elif "模糊指令" in issue:
                suggestions.append("将模糊指令替换为具体、可衡量的要求")
            elif "重复" in issue:
                suggestions.append("合并或删除重复的句子")
            elif "缺乏结构化" in issue:
                suggestions.append("使用标题、列表、编号等结构化格式")
            elif "缺乏具体" in issue:
                suggestions.append("添加具体数字、示例或详细要求")

        # 通用建议
        if len(prompt) < 100:
            suggestions.append("prompt可能过于简短，考虑添加更多上下文和要求")
        elif len(prompt) > 1000:
            suggestions.append("考虑将长prompt分解为多个部分或使用更简洁的表达")

        # 检查是否有明确的角色设定
        if not re.search(r'作为.*|你是.*|角色.*', prompt):
            suggestions.append("考虑添加明确的角色设定（如'你是一个专业的...'）")

        # 检查是否有输出格式要求
        if not re.search(r'格式.*|按照.*格式|输出.*格式', prompt):
            suggestions.append("明确指定输出格式（如Markdown、JSON、特定结构等）")

        # 上下文相关建议
        if context:
            suggestions.append("考虑在prompt中更充分地利用提供的上下文信息")

        return suggestions[:5]  # 最多返回5条建议

    def _minimal_optimization(self, prompt: str, analysis: PromptAnalysis) -> str:
        """最小优化：只修复明显问题"""
        optimized = prompt

        # 移除多余的空格和换行
        optimized = re.sub(r'\n\s*\n\s*\n', '\n\n', optimized)
        optimized = re.sub(r'[ \t]+', ' ', optimized)

        # 修复明显的语法问题
        optimized = re.sub(r'，\s*，', '，', optimized)
        optimized = re.sub(r'。\s*。', '。', optimized)

        # 移除明显的冗余短语
        redundant_phrases = [
            r'非常非常', r'特别特别', r'真的真的',
            r'毫无疑问地', r'毫无疑问的'
        ]
        for phrase in redundant_phrases:
            optimized = re.sub(phrase, '', optimized)

        return optimized.strip()

    def _balanced_optimization(self, prompt: str, analysis: PromptAnalysis, target_model: Optional[str]) -> str:
        """平衡优化：兼顾质量和效率"""
        optimized = self._minimal_optimization(prompt, analysis)

        # 如果使用LLM客户端，进行智能优化
        if self.llm_client and target_model:
            try:
                optimized = self._llm_optimization(optimized, "balanced", target_model)
            except Exception as e:
                logger.warning(f"LLM优化失败，使用规则优化: {e}")

        # 规则优化
        if not self.llm_client or not target_model:
            optimized = self._rule_based_optimization(optimized, "balanced")

        return optimized

    def _aggressive_optimization(self, prompt: str, analysis: PromptAnalysis, target_model: Optional[str]) -> str:
        """激进优化：最大程度压缩"""
        optimized = self._minimal_optimization(prompt, analysis)

        # 如果使用LLM客户端，进行智能优化
        if self.llm_client and target_model:
            try:
                optimized = self._llm_optimization(optimized, "aggressive", target_model)
            except Exception as e:
                logger.warning(f"LLM优化失败，使用规则优化: {e}")

        # 规则优化
        if not self.llm_client or not target_model:
            optimized = self._rule_based_optimization(optimized, "aggressive")

        # 额外压缩
        optimized = self._compress_prompt(optimized)

        return optimized

    def _llm_optimization(self, prompt: str, optimization_type: str, target_model: str) -> str:
        """使用LLM进行智能优化"""
        system_prompt = """你是一个专业的prompt优化专家。你的任务是优化用户提供的prompt，使其更加清晰、具体、高效。

优化原则：
1. 保持原意不变
2. 提高清晰度和具体性
3. 移除冗余和模糊内容
4. 优化结构和格式
5. 考虑token使用效率

请直接返回优化后的prompt，不要添加任何解释。"""

        user_prompt = f"""请优化以下prompt，优化类型：{optimization_type}

原始prompt：
{prompt}

请返回优化后的prompt："""

        try:
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=len(prompt) + 500,
                temperature=0.3
            )

            return response.strip()

        except Exception as e:
            logger.error(f"LLM优化失败: {e}")
            raise

    def _rule_based_optimization(self, prompt: str, optimization_type: str) -> str:
        """基于规则的优化"""
        optimized = prompt

        # 简化句子结构
        if self.optimization_rules["simplify_sentence_structure"]:
            # 将长句拆分为短句
            sentences = re.split(r'[。！？]', optimized)
            simplified_sentences = []
            for sentence in sentences:
                if len(sentence) > 50:
                    # 尝试在逗号处拆分
                    parts = re.split(r'[，,]', sentence)
                    if len(parts) > 1:
                        simplified_sentences.extend([p.strip() + '。' for p in parts if p.strip()])
                    else:
                        simplified_sentences.append(sentence + '。')
                else:
                    simplified_sentences.append(sentence + '。')
            optimized = ' '.join(simplified_sentences)

        # 移除冗余词汇
        if self.optimization_rules["remove_redundant_words"]:
            redundant_patterns = [
                (r'非常|特别|极其|十分', ''),
                (r'的的', '的'),
                (r'了了', '了'),
                (r'一定必须', '必须'),
                (r'应该要', '应该'),
            ]
            for pattern, replacement in redundant_patterns:
                optimized = re.sub(pattern, replacement, optimized)

        # 优化格式
        if self.optimization_rules["optimize_formatting"]:
            # 确保标题格式
            optimized = re.sub(r'^(#+)\s*([^#\n]+)$', r'\1 \2', optimized, flags=re.MULTILINE)
            # 统一列表格式
            optimized = re.sub(r'^\s*[-*•]\s+', '- ', optimized, flags=re.MULTILINE)

        # 提高清晰度
        if self.optimization_rules["improve_clarity"] and optimization_type != "minimal":
            # 添加具体指令
            if not re.search(r'请.*|要求.*|必须.*', optimized):
                optimized = "请" + optimized if not optimized.startswith("请") else optimized

            # 确保有输出格式要求
            if not re.search(r'格式.*|输出.*|结果.*', optimized):
                optimized += "\n\n请按照清晰的结构化格式输出结果。"

        return optimized.strip()

    def _compress_prompt(self, prompt: str) -> str:
        """压缩prompt"""
        compressed = prompt

        # 移除所有不必要的空格
        compressed = re.sub(r'\s+', ' ', compressed)

        # 缩短常见短语
        replacements = {
            r'例如': '如',
            r'比如': '如',
            r'也就是说': '即',
            r'换句话说': '换言之',
            r'与此同时': '同时',
            r'尽管如此': '但',
            r'首先': '一',
            r'其次': '二',
            r'最后': '三',
            r'非常重要': '重要',
            r'非常关键': '关键',
        }

        for pattern, replacement in replacements:
            compressed = re.sub(pattern, replacement, compressed)

        return compressed.strip()

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简单版本）"""
        if not text1 or not text2:
            return 0.0

        # 转换为字符集合
        set1 = set(text1)
        set2 = set(text2)

        if not set1 or not set2:
            return 0.0

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    def get_optimization_history(self, limit: int = 10) -> List[OptimizationResult]:
        """
        获取优化历史

        Args:
            limit: 返回结果数量限制

        Returns:
            List[OptimizationResult]: 优化历史
        """
        return self.optimization_history[-limit:]

    def batch_optimize(
        self,
        prompts: List[str],
        optimization_level: PromptOptimizationLevel = PromptOptimizationLevel.BALANCED,
        target_model: Optional[str] = None
    ) -> List[OptimizationResult]:
        """
        批量优化prompt

        Args:
            prompts: prompt列表
            optimization_level: 优化级别
            target_model: 目标模型

        Returns:
            List[OptimizationResult]: 优化结果列表
        """
        results = []
        for i, prompt in enumerate(prompts):
            try:
                logger.info(f"批量优化进度: {i+1}/{len(prompts)}")
                result = self.optimize_prompt(prompt, optimization_level, target_model)
                results.append(result)
            except Exception as e:
                logger.error(f"优化prompt失败 (索引 {i}): {e}")
                # 创建失败结果
                results.append(OptimizationResult(
                    original_prompt=prompt,
                    optimized_prompt=prompt,
                    analysis=PromptAnalysis(
                        original_length=len(prompt),
                        optimized_length=len(prompt),
                        token_estimate=len(prompt) // 4,
                        cost_estimate=0.0,
                        readability_score=0.5,
                        clarity_score=0.5,
                        specificity_score=0.5,
                        issues_found=["优化失败"],
                        suggestions=[]
                    ),
                    optimization_level=optimization_level,
                    model_used=target_model or "unknown",
                    tokens_saved=0,
                    cost_saved=0.0
                ))

        return results

    def export_optimization_report(self, results: List[OptimizationResult]) -> str:
        """
        导出优化报告

        Args:
            results: 优化结果列表

        Returns:
            str: 报告内容（Markdown格式）
        """
        if not results:
            return "# 优化报告\n\n没有优化结果。"

        total_tokens_saved = sum(r.tokens_saved for r in results)
        total_cost_saved = sum(r.cost_saved for r in results)
        avg_readability_improvement = sum(r.analysis.readability_score for r in results) / len(results)

        report = f"""# Prompt优化报告

## 统计摘要
- 优化数量: {len(results)}
- 总节省Token数: {total_tokens_saved}
- 总节省成本: ${total_cost_saved:.4f}
- 平均可读性评分: {avg_readability_improvement:.2f}/1.0

## 详细结果

"""

        for i, result in enumerate(results):
            report += f"""### 优化结果 #{i+1}

**优化级别**: {result.optimization_level.value}
**使用模型**: {result.model_used}

**原始Prompt长度**: {len(result.original_prompt)} 字符
**优化后长度**: {len(result.optimized_prompt)} 字符
**节省Token数**: {result.tokens_saved}
**节省成本**: ${result.cost_saved:.4f}

**分析结果**:
- 可读性评分: {result.analysis.readability_score:.2f}/1.0
- 清晰度评分: {result.analysis.clarity_score:.2f}/1.0
- 具体性评分: {result.analysis.specificity_score:.2f}/1.0

**发现的问题**:
{chr(10).join(f"- {issue}" for issue in result.analysis.issues_found) or "无"}

**优化建议**:
{chr(10).join(f"- {suggestion}" for suggestion in result.analysis.suggestions) or "无"}

**优化前**:
```
{result.original_prompt[:500]}{'...' if len(result.original_prompt) > 500 else ''}
```

**优化后**:
```
{result.optimized_prompt[:500]}{'...' if len(result.optimized_prompt) > 500 else ''}
```

---

"""

        return report


# 全局Prompt优化器实例
_prompt_optimizer_instance: Optional[PromptOptimizer] = None

def get_prompt_optimizer(llm_client: Optional[LLMClient] = None) -> PromptOptimizer:
    """获取全局Prompt优化器实例"""
    global _prompt_optimizer_instance
    if _prompt_optimizer_instance is None:
        _prompt_optimizer_instance = PromptOptimizer(llm_client)
    return _prompt_optimizer_instance