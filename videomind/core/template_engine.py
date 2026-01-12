"""
模板引擎模块
管理提示模板和变量替换
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from loguru import logger

from models.template import Template, TemplateVariable, TemplateType, PREDEFINED_TEMPLATES
from utils.exceptions import TemplateError


class TemplateEngine:
    """模板引擎"""

    def __init__(self, templates_dir: Optional[Path] = None):
        """
        初始化模板引擎

        Args:
            templates_dir: 模板目录路径
        """
        self.templates_dir = templates_dir or Path("templates")
        self.templates_dir.mkdir(parents=True, exist_ok=True)

        # 加载模板
        self.templates: Dict[str, Template] = {}
        self._load_templates()

    def _load_templates(self):
        """加载模板"""
        # 加载预定义模板
        self.templates.update(PREDEFINED_TEMPLATES)

        # 加载用户自定义模板
        self._load_custom_templates()

    def _load_custom_templates(self):
        """加载用户自定义模板"""
        try:
            # 查找JSON模板文件
            json_files = list(self.templates_dir.glob("*.json"))
            yaml_files = list(self.templates_dir.glob("*.yaml")) + list(self.templates_dir.glob("*.yml"))

            for file_path in json_files + yaml_files:
                try:
                    template = self._load_template_from_file(file_path)
                    if template.name not in self.templates:  # 不覆盖预定义模板
                        self.templates[template.name] = template
                        logger.debug(f"加载自定义模板: {template.name}")
                except Exception as e:
                    logger.warning(f"加载模板文件失败 {file_path}: {e}")

        except Exception as e:
            logger.error(f"加载自定义模板失败: {e}")

    def _load_template_from_file(self, file_path: Path) -> Template:
        """从文件加载模板"""
        if file_path.suffix.lower() in [".yaml", ".yml"]:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:  # JSON
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        return Template.from_dict(data)

    def get_template(self, name: str) -> Template:
        """
        获取模板

        Args:
            name: 模板名称

        Returns:
            Template: 模板对象

        Raises:
            TemplateError: 模板不存在
        """
        if name not in self.templates:
            raise TemplateError(f"模板不存在: {name}")

        return self.templates[name]

    def list_templates(self, template_type: Optional[TemplateType] = None) -> List[Template]:
        """
        列出模板

        Args:
            template_type: 模板类型过滤器

        Returns:
            List[Template]: 模板列表
        """
        templates = list(self.templates.values())

        if template_type:
            templates = [t for t in templates if t.type == template_type]

        return sorted(templates, key=lambda x: x.name)

    def create_template(self, template_data: Dict[str, Any]) -> Template:
        """
        创建新模板

        Args:
            template_data: 模板数据

        Returns:
            Template: 创建的模板对象
        """
        try:
            # 验证模板数据
            if "name" not in template_data:
                raise TemplateError("模板必须包含名称")

            if "user_prompt" not in template_data:
                raise TemplateError("模板必须包含用户提示")

            # 设置默认值
            template_data.setdefault("type", TemplateType.CUSTOM)
            template_data.setdefault("description", "")
            template_data.setdefault("variables", [])
            template_data.setdefault("model_parameters", {})
            template_data.setdefault("version", "1.0.0")
            template_data.setdefault("tags", [])
            template_data.setdefault("created_at", datetime.now().isoformat())
            template_data.setdefault("updated_at", datetime.now().isoformat())

            # 创建模板对象
            template = Template.from_dict(template_data)

            # 保存模板
            self.save_template(template)

            # 添加到内存缓存
            self.templates[template.name] = template

            logger.info(f"创建新模板: {template.name}")
            return template

        except Exception as e:
            logger.error(f"创建模板失败: {e}")
            raise TemplateError(f"创建模板失败: {str(e)}")

    def save_template(self, template: Template):
        """
        保存模板到文件

        Args:
            template: 模板对象
        """
        try:
            # 更新修改时间
            template.updated_at = datetime.now().isoformat()

            # 保存为JSON文件
            file_path = self.templates_dir / f"{template.name}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(template.to_dict(), f, ensure_ascii=False, indent=2)

            logger.debug(f"模板已保存: {file_path}")

        except Exception as e:
            logger.error(f"保存模板失败: {e}")
            raise TemplateError(f"保存模板失败: {str(e)}")

    def update_template(self, name: str, updates: Dict[str, Any]) -> Template:
        """
        更新模板

        Args:
            name: 模板名称
            updates: 更新内容

        Returns:
            Template: 更新后的模板对象
        """
        if name not in self.templates:
            raise TemplateError(f"模板不存在: {name}")

        try:
            # 获取现有模板
            template = self.templates[name]

            # 更新模板数据
            template_dict = template.dict()
            template_dict.update(updates)
            template_dict["updated_at"] = datetime.now().isoformat()

            # 重新创建模板对象
            updated_template = Template.from_dict(template_dict)

            # 保存更新
            self.save_template(updated_template)

            # 更新内存缓存
            self.templates[name] = updated_template

            logger.info(f"更新模板: {name}")
            return updated_template

        except Exception as e:
            logger.error(f"更新模板失败: {e}")
            raise TemplateError(f"更新模板失败: {str(e)}")

    def delete_template(self, name: str):
        """
        删除模板

        Args:
            name: 模板名称
        """
        if name not in self.templates:
            raise TemplateError(f"模板不存在: {name}")

        # 不能删除预定义模板
        if name in PREDEFINED_TEMPLATES:
            raise TemplateError(f"不能删除预定义模板: {name}")

        try:
            # 删除文件
            file_path = self.templates_dir / f"{name}.json"
            if file_path.exists():
                file_path.unlink()

            # 从内存缓存中删除
            del self.templates[name]

            logger.info(f"删除模板: {name}")

        except Exception as e:
            logger.error(f"删除模板失败: {e}")
            raise TemplateError(f"删除模板失败: {str(e)}")

    def render_template(self, template: Template, variables: Dict[str, Any]) -> str:
        """
        渲染模板

        Args:
            template: 模板对象
            variables: 模板变量

        Returns:
            str: 渲染后的提示文本
        """
        try:
            # 渲染模板（render方法内部会调用validate_variables，所以直接传入原始variables）
            rendered_prompt = template.render(variables)

            logger.debug(f"模板渲染完成: {template.name}")
            return rendered_prompt

        except Exception as e:
            logger.error(f"渲染模板失败: {e}")
            raise TemplateError(f"渲染模板失败: {str(e)}")

    def render_template_by_name(self, template_name: str, variables: Dict[str, Any]) -> str:
        """
        通过名称渲染模板

        Args:
            template_name: 模板名称
            variables: 模板变量

        Returns:
            str: 渲染后的提示文本
        """
        template = self.get_template(template_name)
        return self.render_template(template, variables)

    def validate_variables(self, template: Template, variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证模板变量

        Args:
            template: 模板对象
            variables: 模板变量

        Returns:
            Dict[str, Any]: 验证后的变量
        """
        return template.validate_variables(variables)

    def get_template_variables(self, template_name: str) -> List[TemplateVariable]:
        """
        获取模板变量列表

        Args:
            template_name: 模板名称

        Returns:
            List[TemplateVariable]: 模板变量列表
        """
        template = self.get_template(template_name)
        return template.variables

    def search_templates(self, query: str) -> List[Template]:
        """
        搜索模板

        Args:
            query: 搜索查询

        Returns:
            List[Template]: 匹配的模板列表
        """
        query = query.lower()
        results = []

        for template in self.templates.values():
            # 在名称、描述、标签中搜索
            if (query in template.name.lower() or
                query in template.description.lower() or
                any(query in tag.lower() for tag in template.tags)):
                results.append(template)

        return sorted(results, key=lambda x: x.name)

    def export_template(self, template_name: str, format: str = "json") -> str:
        """
        导出模板

        Args:
            template_name: 模板名称
            format: 导出格式（json, yaml）

        Returns:
            str: 导出的模板内容
        """
        template = self.get_template(template_name)

        if format.lower() == "yaml":
            return yaml.dump(template.dict(), default_flow_style=False, allow_unicode=True)
        else:  # JSON
            return json.dumps(template.dict(), ensure_ascii=False, indent=2)

    def import_template(self, content: str, format: str = "json") -> Template:
        """
        导入模板

        Args:
            content: 模板内容
            format: 导入格式（json, yaml）

        Returns:
            Template: 导入的模板对象
        """
        try:
            if format.lower() == "yaml":
                data = yaml.safe_load(content)
            else:  # JSON
                data = json.loads(content)

            return self.create_template(data)

        except Exception as e:
            logger.error(f"导入模板失败: {e}")
            raise TemplateError(f"导入模板失败: {str(e)}")

    def get_template_statistics(self) -> Dict[str, Any]:
        """获取模板统计信息"""
        total = len(self.templates)
        predefined = len(PREDEFINED_TEMPLATES)
        custom = total - predefined

        # 按类型统计
        type_counts = {}
        for template in self.templates.values():
            type_counts[template.type.value] = type_counts.get(template.type.value, 0) + 1

        return {
            "total_templates": total,
            "predefined_templates": predefined,
            "custom_templates": custom,
            "type_distribution": type_counts,
        }