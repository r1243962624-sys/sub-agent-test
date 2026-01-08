# Test Subagent Project

这是一个使用Claude Sub-Agent进行产品开发的测试项目。

## 项目结构

```
.
├── CLAUDE.md                                    # 主流程控制器配置
├── PRD.md                                       # 产品需求文档
├── .claude/
│   └── agents/                                  # Sub-Agent配置
│       ├── backenddeveloper.md                 # 后端开发工程师
│       ├── designer.md                          # UI/UX设计师
│       ├── frontenddeveloper.md                 # 前端开发工程师
│       ├── llmengineer.md                      # 大模型应用工程师
│       └── productmanager.md                    # 产品经理
└── README.md                                    # 项目说明文档
```

## 功能说明

本项目配置了多个专业的Sub-Agent，用于完整的产品开发流程：

- **产品经理** (productmanager): 需求分析、PRD生成
- **UI/UX设计师** (designer): 设计策略制定、设计规范生成
- **前端开发工程师** (frontenddeveloper): 前端技术方案、代码实现
- **后端开发工程师** (backenddeveloper): 后端架构设计、部署配置
- **大模型应用工程师** (llmengineer): AI功能方案设计、AI代码实现

## 使用方式

按照CLAUDE.md中的工作流程，通过问答式交互完成从需求收集到产品部署的全流程。

## 许可证

MIT License
