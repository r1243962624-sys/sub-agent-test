"""
模板数据模型
定义提示模板的结构和变量系统
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from pydantic import BaseModel, Field, validator


class TemplateType(str, Enum):
    """模板类型枚举"""
    MEETING_MINUTES = "meeting_minutes"
    STUDY_NOTES = "study_notes"
    TRAINING_SUMMARY = "training_summary"
    INTERVIEW_TRANSCRIPT = "interview_transcript"
    LECTURE_NOTES = "lecture_notes"
    PRESENTATION_SUMMARY = "presentation_summary"
    PODCAST_SUMMARY = "podcast_summary"
    WEBINAR_SUMMARY = "webinar_summary"
    TUTORIAL_GUIDE = "tutorial_guide"
    PRODUCT_REVIEW = "product_review"
    RESEARCH_PAPER_SUMMARY = "research_paper_summary"
    CUSTOM = "custom"


class TemplateVariable(BaseModel):
    """模板变量定义"""
    name: str = Field(..., description="变量名称")
    description: str = Field(..., description="变量描述")
    required: bool = Field(True, description="是否必需")
    default_value: Optional[Any] = Field(None, description="默认值")
    validation_rules: Optional[Dict[str, Any]] = Field(None, description="验证规则")

    @validator("name")
    def validate_variable_name(cls, v):
        """验证变量名称格式"""
        # 支持 {{{var}}} 和 {{var}} 格式
        if not (v.startswith("{{") and v.endswith("}}")):
            raise ValueError(f"变量名称应使用双花括号或三花括号包裹: {v}")
        if " " in v:
            raise ValueError(f"变量名称不能包含空格: {v}")
        return v


class Template(BaseModel):
    """提示模板"""
    name: str = Field(..., description="模板名称")
    type: TemplateType = Field(..., description="模板类型")
    description: str = Field(..., description="模板描述")

    # 提示内容
    system_prompt: Optional[str] = Field(None, description="系统提示")
    user_prompt: str = Field(..., description="用户提示")

    # 变量系统
    variables: List[TemplateVariable] = Field(default_factory=list, description="模板变量")

    # 模型参数
    model_parameters: Dict[str, Any] = Field(
        default_factory=lambda: {
            "temperature": 0.7,
            "max_tokens": None,
            "top_p": 1.0,
        },
        description="模型配置参数",
        alias="model_config"
    )

    # 元数据
    version: str = Field("1.0.0", description="模板版本")
    author: Optional[str] = Field(None, description="作者")
    tags: List[str] = Field(default_factory=list, description="标签")
    created_at: Optional[str] = Field(None, description="创建时间")
    updated_at: Optional[str] = Field(None, description="更新时间")

    @validator("user_prompt")
    def validate_prompt_contains_variables(cls, v, values):
        """验证提示中包含必要的变量"""
        variables = values.get("variables", [])
        for var in variables:
            if var.required and var.name not in v:
                raise ValueError(f"必需变量 {var.name} 未在提示中找到")
        return v

    def get_variable_names(self) -> List[str]:
        """获取所有变量名称"""
        return [var.name for var in self.variables]

    def validate_variables(self, provided_vars: Dict[str, Any]) -> Dict[str, Any]:
        """验证提供的变量值"""
        from loguru import logger
        import re
        
        validated_vars = {}

        for var in self.variables:
            # 去除所有花括号，支持 {{{var}}} 和 {{var}} 格式
            # 使用正则表达式去除所有花括号
            var_name = re.sub(r'^\{+|}+\}$', '', var.name)
            
            logger.debug(f"验证变量: 模板变量名={var.name}, 提取的变量名={var_name}, 提供的变量={list(provided_vars.keys())}")

            if var_name in provided_vars:
                value = provided_vars[var_name]
                # 检查值是否为None或空字符串
                if value is None:
                    logger.warning(f"变量 {var_name} 的值为 None")
                # 这里可以添加更复杂的验证逻辑
                validated_vars[var.name] = value
            elif var.required and var.default_value is None:
                logger.error(f"必需变量 {var.name} (提取为 {var_name}) 未在提供的变量中找到: {list(provided_vars.keys())}")
                raise ValueError(f"必需变量 {var.name} 未提供且无默认值")
            elif var.default_value is not None:
                validated_vars[var.name] = var.default_value

        return validated_vars

    def render(self, variables: Dict[str, Any]) -> str:
        """渲染模板，替换变量"""
        validated_vars = self.validate_variables(variables)

        rendered_prompt = self.user_prompt
        for var_name, value in validated_vars.items():
            # 确保值是字符串
            str_value = str(value) if value is not None else ""
            rendered_prompt = rendered_prompt.replace(var_name, str_value)

        return rendered_prompt

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于JSON序列化）"""
        return self.dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Template":
        """从字典创建模板"""
        return cls(**data)


# 预定义模板
PREDEFINED_TEMPLATES = {
    "meeting_minutes": Template(
        name="meeting_minutes",
        type=TemplateType.MEETING_MINUTES,
        description="会议纪要模板，将会议录音整理为结构化会议纪要",
        system_prompt="你是一个专业的会议纪要整理助手。请根据提供的会议录音文字稿，整理出结构清晰、重点突出的会议纪要。",
        user_prompt="""请将以下会议录音文字稿整理为结构化会议纪要：

会议录音文字稿：
{{{transcript}}}

请按照以下格式整理会议纪要：

# 会议纪要

## 基本信息
- **会议主题**: [请根据内容总结]
- **会议时间**: [如有提到请填写，否则留空]
- **参会人员**: [请从文字稿中提取参会人员]
- **会议地点**: [如有提到请填写，否则留空]

## 会议议程
[请列出讨论的主要议题]

## 讨论要点
[请总结每个议题的讨论内容和关键观点]

## 决策事项
[请列出会议中做出的决策和行动计划]

## 待办事项
[请列出需要跟进的任务，包括负责人和截止时间（如有）]

## 关键结论
[请总结会议的核心结论和成果]

## 下一步计划
[请描述后续的行动计划和安排]

请确保纪要内容准确、完整，重点突出，语言简洁明了。""",
        variables=[
            TemplateVariable(
                name="{{{transcript}}}",
                description="会议录音文字稿",
                required=True
            )
        ],
        model_parameters={
            "temperature": 0.3,
            "max_tokens": None,
            "top_p": 0.9,
        },
        tags=["meeting", "minutes", "business"],
        author="VideoMind Team"
    ),

    "study_notes": Template(
        name="study_notes",
        type=TemplateType.STUDY_NOTES,
        description="学习笔记模板，将学习视频内容整理为结构化笔记",
        system_prompt="""你是一位专业的课程内容整理专家。你的任务是将从视频转写的文字稿整理成结构化、易于理解的学习笔记。

## 角色定位 (Role)
你是课程内容整理专家，具备出色的内容理解、结构化和表达能力。

## 背景 (Background)
你将收到从视频内容转写的文字稿，这些文字稿可能包含：
- 课程讲解内容
- 知识点解释
- 操作步骤说明
- 案例和示例
- 问答和讨论

## 专业能力 (Profile)
- 深刻理解课程内容和知识结构
- 能够识别和提取关键知识点
- 擅长将非结构化内容转化为结构化笔记
- 具备优秀的逻辑组织和表达能力

## 核心技能 (Skills)
1. 内容分析：快速理解课程的核心内容和逻辑结构
2. 知识点提取：准确识别和提取关键知识点、概念和要点
3. 结构化组织：将内容组织成清晰的层级结构
4. 操作步骤梳理：将操作流程整理为清晰的步骤说明（SOP）
5. 难点解析：识别并详细解释复杂概念和难点

## 工作目标 (Goals)
1. 将转写的文字稿整理成清晰、结构化的学习笔记
2. 使用Markdown格式，包含适当的标题层级（H2-H4）
3. 提取关键知识点，便于快速复习
4. 整理操作步骤为标准的SOP格式
5. 确保内容准确、完整，便于理解和后续学习

## 约束条件 (Constrains)
1. 只基于提供的转写文字稿进行整理，不添加原文中没有的信息
2. 保持客观准确，不添加主观臆测
3. 使用Markdown格式输出，标题层级使用H2-H4
4. 知识点要条理清晰，使用列表和层级结构
5. 操作步骤要明确，使用有序列表和子项

## 输出格式 (OutputFormat)
使用Markdown格式，包含以下结构：
- 使用 ## 作为主要章节标题（H2）
- 使用 ### 作为小节标题（H3）
- 使用 #### 作为子小节标题（H4）
- 知识点使用列表格式
- 操作步骤使用有序列表
- 重要内容可以使用加粗强调

## 工作流程 (Workflow)
1. 通读转写文字稿，理解整体内容结构
2. 识别课程主题和核心知识点
3. 提取关键概念、定义和要点
4. 梳理操作步骤（如有），整理为SOP格式
5. 识别难点和复杂概念，进行详细解释
6. 组织内容为Markdown格式，使用适当的标题层级
7. 检查内容的准确性和完整性

## 示例 (Examples)
### 示例1：技术课程
```markdown
## 核心知识点

### 基础概念
- **概念1**: 定义和说明
- **概念2**: 定义和说明

### 操作步骤

#### 步骤1：准备工作
1. 子步骤1
2. 子步骤2

#### 步骤2：执行操作
1. 子步骤1
2. 子步骤2
```

### 示例2：理论课程
```markdown
## 主要内容

### 核心理论
- **理论要点1**: 详细说明
  - 关键点1
  - 关键点2

### 实际应用
- 应用场景1
- 应用场景2
```""",
        user_prompt="""请将以下转写的视频文字稿整理为结构化的学习笔记，使用Markdown格式，包含知识点和操作步骤：

{{{transcript}}}

请按照系统提示中的格式要求进行整理。""",
        variables=[
            TemplateVariable(
                name="{{{transcript}}}",
                description="视频内容文字稿",
                required=True
            )
        ],
        model_parameters={
            "temperature": 0.3,
            "max_tokens": None,
            "top_p": 0.9,
        },
        tags=["study", "learning", "education"],
        author="VideoMind Team"
    ),

    "training_summary": Template(
        name="training_summary",
        type=TemplateType.TRAINING_SUMMARY,
        description="培训总结模板，将培训视频内容整理为结构化总结",
        system_prompt="你是一个专业的培训总结助手。请根据提供的培训视频内容文字稿，整理出结构清晰、重点突出的培训总结。",
        user_prompt="""请将以下培训视频内容整理为结构化培训总结：

培训内容文字稿：
{{{transcript}}}

请按照以下格式整理培训总结：

# 培训总结

## 培训基本信息
- **培训主题**: [请根据内容总结]
- **培训讲师**: [如有提到请填写]
- **培训时长**: [如有提到请填写]
- **培训日期**: [如有提到请填写]

## 培训目标
[请总结培训的主要目标和预期成果]

## 核心内容
[请总结培训的核心内容和知识点]

## 技能要点
[请列出培训中讲解的关键技能和操作方法]

## 最佳实践
[请记录培训中分享的最佳实践和经验]

## 工具与资源
[请列出培训中提到的工具、资源或参考资料]

## 考核要点
[如有考核要求，请总结考核要点和标准]

## 行动计划
[请制定基于培训内容的学习或工作行动计划]

## 培训反馈
[请总结培训的优点和改进建议]

请确保总结内容准确、完整，重点突出，便于后续应用和分享。""",
        variables=[
            TemplateVariable(
                name="{{{transcript}}}",
                description="培训内容文字稿",
                required=True
            )
        ],
        model_parameters={
            "temperature": 0.35,
            "max_tokens": None,
            "top_p": 0.9,
        },
        tags=["training", "summary", "professional"],
        author="VideoMind Team"
    ),

    "presentation_summary": Template(
        name="presentation_summary",
        type=TemplateType.PRESENTATION_SUMMARY,
        description="演示文稿总结模板，将演示视频整理为结构化总结",
        system_prompt="你是一个专业的演示文稿总结助手。请根据提供的演示视频文字稿，整理出结构清晰、重点突出的演示总结。",
        user_prompt="""请将以下演示视频内容整理为结构化演示总结：

演示内容文字稿：
{{{transcript}}}

请按照以下格式整理演示总结：

# 演示总结

## 演示信息
- **演示主题**: [请根据内容总结]
- **演示者**: [如有提到请填写]
- **演示时长**: [如有提到请填写]
- **演示日期**: [如有提到请填写]

## 核心观点
[请总结演示的核心观点和主要论点]

## 关键数据
[请记录演示中提到的关键数据、统计和事实]

## 视觉元素
[请描述演示中的图表、图片、动画等视觉元素及其含义]

## 故事线
[请总结演示的故事线或逻辑结构]

## 观众互动
[如有观众互动环节，请记录互动内容和反馈]

## 行动号召
[请总结演示中的行动号召或下一步建议]

## 学习要点
[请总结从演示中学到的主要内容和启发]

请确保总结内容准确、完整，重点突出，便于分享和传播。""",
        variables=[
            TemplateVariable(
                name="{{{transcript}}}",
                description="演示内容文字稿",
                required=True
            )
        ],
        model_parameters={
            "temperature": 0.4,
            "max_tokens": None,
            "top_p": 0.9,
        },
        tags=["presentation", "summary", "business"],
        author="VideoMind Team"
    ),

    "podcast_summary": Template(
        name="podcast_summary",
        type=TemplateType.PODCAST_SUMMARY,
        description="播客节目总结模板，将播客音频整理为结构化总结",
        system_prompt="你是一个专业的播客总结助手。请根据提供的播客音频文字稿，整理出结构清晰、重点突出的播客总结。",
        user_prompt="""请将以下播客节目内容整理为结构化播客总结：

播客内容文字稿：
{{{transcript}}}

请按照以下格式整理播客总结：

# 播客总结

## 节目信息
- **节目名称**: [请根据内容总结]
- **主持人**: [请从文字稿中提取]
- **嘉宾**: [请从文字稿中提取]
- **节目时长**: [如有提到请填写]
- **发布日期**: [如有提到请填写]

## 讨论主题
[请列出讨论的主要话题和议题]

## 嘉宾观点
[请总结嘉宾的主要观点和见解]

## 精彩对话
[请记录节目中的精彩对话和讨论]

## 故事分享
[请记录节目中分享的故事和案例]

## 实用建议
[请总结节目中提供的实用建议和技巧]

## 争议话题
[如有争议话题，请客观记录各方观点]

## 节目亮点
[请总结节目的亮点和特别之处]

## 推荐理由
[请总结推荐收听的理由和收获]

请确保总结内容准确、完整，生动有趣，便于分享和讨论。""",
        variables=[
            TemplateVariable(
                name="{{{transcript}}}",
                description="播客内容文字稿",
                required=True
            )
        ],
        model_parameters={
            "temperature": 0.5,
            "max_tokens": None,
            "top_p": 0.95,
        },
        tags=["podcast", "summary", "entertainment"],
        author="VideoMind Team"
    ),

    "webinar_summary": Template(
        name="webinar_summary",
        type=TemplateType.WEBINAR_SUMMARY,
        description="网络研讨会总结模板，将网络研讨会视频整理为结构化总结",
        system_prompt="你是一个专业的网络研讨会总结助手。请根据提供的网络研讨会视频文字稿，整理出结构清晰、重点突出的研讨会总结。",
        user_prompt="""请将以下网络研讨会内容整理为结构化研讨会总结：

研讨会内容文字稿：
{{{transcript}}}

请按照以下格式整理研讨会总结：

# 网络研讨会总结

## 研讨会信息
- **研讨会主题**: [请根据内容总结]
- **主办方**: [如有提到请填写]
- **主讲人**: [请从文字稿中提取]
- **研讨会时长**: [如有提到请填写]
- **举办日期**: [如有提到请填写]

## 研讨会目标
[请总结研讨会的主要目标和预期成果]

## 核心内容
[请总结研讨会的核心内容和知识点]

## 技术演示
[如有技术演示，请记录演示内容和步骤]

## Q&A环节
[请记录问答环节的主要问题和回答]

## 资源分享
[请记录研讨会中分享的资源、工具和链接]

## 行业洞察
[请总结研讨会提供的行业洞察和趋势分析]

## 实践案例
[请记录研讨会中分享的实践案例和经验]

## 后续行动
[请总结研讨会建议的后续学习和行动]

请确保总结内容准确、完整，专业实用，便于后续学习和应用。""",
        variables=[
            TemplateVariable(
                name="{{{transcript}}}",
                description="研讨会内容文字稿",
                required=True
            )
        ],
        model_parameters={
            "temperature": 0.35,
            "max_tokens": None,
            "top_p": 0.9,
        },
        tags=["webinar", "summary", "professional"],
        author="VideoMind Team"
    ),

    "tutorial_guide": Template(
        name="tutorial_guide",
        type=TemplateType.TUTORIAL_GUIDE,
        description="教程指南模板，将教程视频整理为结构化操作指南",
        system_prompt="你是一个专业的教程整理助手。请根据提供的教程视频文字稿，整理出结构清晰、步骤明确的教程指南。",
        user_prompt="""请将以下教程视频内容整理为结构化教程指南：

教程内容文字稿：
{{{transcript}}}

请按照以下格式整理教程指南：

# 教程指南

## 教程概述
- **教程主题**: [请根据内容总结]
- **技能级别**: [初级/中级/高级]
- **预计学习时间**: [如有提到请填写]
- **所需工具**: [请列出需要的工具和软件]

## 学习目标
[请列出学完本教程后能够掌握的知识和技能]

## 前置知识
[请列出学习本教程前需要掌握的基础知识]

## 环境准备
[请详细说明环境配置和准备工作]

## 步骤详解
[请按照顺序详细说明每个操作步骤，包括：
1. 步骤名称
2. 具体操作
3. 预期结果
4. 注意事项]

## 代码示例
[如有代码，请提供完整可运行的代码示例]

## 常见问题
[请列出可能遇到的问题和解决方案]

## 测试验证
[请说明如何验证教程学习效果]

## 扩展学习
[请提供进一步学习的资源和方向]

请确保指南内容准确、详细，步骤清晰，便于跟随操作。""",
        variables=[
            TemplateVariable(
                name="{{{transcript}}}",
                description="教程内容文字稿",
                required=True
            )
        ],
        model_parameters={
            "temperature": 0.3,
            "max_tokens": None,
            "top_p": 0.9,
        },
        tags=["tutorial", "guide", "technical"],
        author="VideoMind Team"
    ),

    "product_review": Template(
        name="product_review",
        type=TemplateType.PRODUCT_REVIEW,
        description="产品评测模板，将产品评测视频整理为结构化评测报告",
        system_prompt="你是一个专业的产品评测整理助手。请根据提供的产品评测视频文字稿，整理出客观、全面的产品评测报告。",
        user_prompt="""请将以下产品评测视频内容整理为结构化评测报告：

评测内容文字稿：
{{{transcript}}}

请按照以下格式整理评测报告：

# 产品评测报告

## 产品信息
- **产品名称**: [请根据内容总结]
- **产品型号**: [如有提到请填写]
- **品牌**: [请从文字稿中提取]
- **评测日期**: [如有提到请填写]

## 评测概要
[请简要总结评测的主要内容和结论]

## 产品规格
[请记录产品的关键规格和参数]

## 外观设计
[请描述产品的外观设计、材质和做工]

## 功能特点
[请总结产品的主要功能和特点]

## 性能测试
[请记录性能测试的结果和数据]

## 使用体验
[请总结实际使用体验，包括优点和不足]

## 竞品对比
[如有竞品对比，请记录对比结果]

## 性价比分析
[请分析产品的价格和价值比]

## 购买建议
[请提供针对不同用户群体的购买建议]

## 总结评分
[请给出综合评分（1-10分）和理由]

请确保报告内容客观、全面，数据准确，便于消费者参考。""",
        variables=[
            TemplateVariable(
                name="{{{transcript}}}",
                description="评测内容文字稿",
                required=True
            )
        ],
        model_parameters={
            "temperature": 0.4,
            "max_tokens": None,
            "top_p": 0.9,
        },
        tags=["product", "review", "consumer"],
        author="VideoMind Team"
    ),

    "research_paper_summary": Template(
        name="research_paper_summary",
        type=TemplateType.RESEARCH_PAPER_SUMMARY,
        description="研究论文总结模板，将学术讲座视频整理为结构化论文总结",
        system_prompt="你是一个专业的学术论文总结助手。请根据提供的学术讲座视频文字稿，整理出结构清晰、重点突出的论文总结。",
        user_prompt="""请将以下学术讲座内容整理为结构化论文总结：

讲座内容文字稿：
{{{transcript}}}

请按照以下格式整理论文总结：

# 研究论文总结

## 论文信息
- **论文标题**: [请根据内容总结]
- **作者**: [请从文字稿中提取]
- **发表期刊/会议**: [如有提到请填写]
- **发表年份**: [如有提到请填写]

## 研究背景
[请总结研究的背景和问题提出]

## 研究目标
[请明确研究的主要目标和研究问题]

## 研究方法
[请详细说明研究采用的方法论、实验设计和数据分析方法]

## 主要发现
[请总结研究的主要发现和结果]

## 数据分析
[请记录关键的数据分析结果和统计]

## 讨论与解释
[请总结对研究结果的讨论和理论解释]

## 创新点
[请指出研究的创新点和贡献]

## 局限与展望
[请总结研究的局限性和未来研究方向]

## 实践意义
[请说明研究的实践意义和应用价值]

## 参考文献
[请记录讲座中提到的关键参考文献]

请确保总结内容准确、专业，逻辑清晰，便于学术交流和引用。""",
        variables=[
            TemplateVariable(
                name="{{{transcript}}}",
                description="讲座内容文字稿",
                required=True
            )
        ],
        model_parameters={
            "temperature": 0.3,
            "max_tokens": None,
            "top_p": 0.9,
        },
        tags=["research", "paper", "academic"],
        author="VideoMind Team"
    )
}