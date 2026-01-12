# VideoMind - 自动化视频内容处理系统

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

VideoMind 是一个自动化视频内容处理系统，能够将视频链接转换为结构化笔记。系统支持从 YouTube、Bilibili 等平台下载视频，提取音频，使用 Whisper 进行语音转写，并通过大模型 API 生成高质量的 Markdown 格式笔记。

## ✨ 核心功能

- **视频下载**：支持 YouTube、Bilibili 等主流视频平台
- **音频提取**：自动从视频中提取高质量音频
- **语音转写**：使用 OpenAI Whisper 进行高精度语音识别
- **AI 生成**：通过大模型 API 生成结构化笔记
- **模板系统**：预置会议纪要、培训笔记、学习总结等模板
- **批量处理**：支持批量处理多个视频链接

### 🧠 增强的AI功能

- **成本监控**：实时监控API使用成本，设置预算限制
- **Prompt优化**：智能优化prompt，提高输出质量和成本效率
- **批量处理管理**：智能批量任务调度和并发控制
- **模型性能监控**：监控各模型响应时间、成功率和成本效率
- **智能模型推荐**：根据任务类型自动推荐最佳模型
- **模板扩展**：支持演示文稿、播客、教程、产品评测等12种模板类型

## 🚀 快速开始

### 安装

1. 克隆项目：
```bash
git clone https://github.com/yourusername/videomind.git
cd videomind
```

2. 创建虚拟环境并安装依赖：
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

3. 安装 FFmpeg（音频处理必需）：
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt install ffmpeg`
- **Windows**: 从 [FFmpeg官网](https://ffmpeg.org/download.html) 下载并添加到 PATH

### 配置

1. 复制环境变量示例文件：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，配置 API 密钥：
```env
# OpenAI API 配置
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API 配置（可选）
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# 其他配置
MODEL_PROVIDER=openai  # 或 anthropic
DEFAULT_MODEL=gpt-4-turbo-preview  # 或 claude-3-opus-20240229
```

3. 配置模板（可选）：
```bash
cp config.yaml.example config.yaml
```

### 基本使用

处理单个视频：
```bash
videomind process "https://www.youtube.com/watch?v=example"
```

批量处理多个视频：
```bash
videomind batch --file urls.txt
```

查看配置：
```bash
videomind config show
```

管理模板：
```bash
videomind template list
videomind template use meeting_minutes
```

AI功能管理：
```bash
# 显示成本统计
videomind ai --cost

# 优化prompt或模板
videomind ai --optimize meeting_minutes
videomind ai --optimize ./my_prompt.txt

# 显示模型性能统计
videomind ai --model-stats

# 显示批量任务状态
videomind ai --batch-status

# 显示AI功能洞察
videomind ai --insights

# 导出AI数据
videomind ai --export ./ai_data.json
```

## 📁 项目结构

```
videomind/
├── core/                    # 核心处理模块
│   ├── downloader.py       # 视频下载
│   ├── audio_extractor.py  # 音频提取
│   ├── transcriber.py      # ASR 转写
│   ├── llm_client.py       # 大模型客户端
│   ├── template_engine.py  # 模板引擎
│   ├── processor.py        # 视频处理器
│   ├── cost_monitor.py     # 成本监控模块
│   ├── prompt_optimizer.py # Prompt优化模块
│   ├── batch_manager.py    # 批量处理管理器
│   └── model_monitor.py    # 模型性能监控模块
├── cli/                    # 命令行界面
│   └── main.py            # 主程序入口
├── models/                 # 数据模型
├── storage/               # 存储管理
├── utils/                 # 工具函数
├── templates/             # 模板文件
├── tests/                 # 测试代码
├── config.yaml           # 配置文件
├── .env                  # 环境变量
├── requirements.txt      # 依赖列表
└── README.md            # 项目说明
```

## 🔧 详细配置

### 支持的视频平台

- YouTube
- Bilibili
- Vimeo
- 其他 yt-dlp 支持的平台

### 支持的 AI 模型提供商

- OpenAI (GPT-4, GPT-3.5-turbo)
- Anthropic (Claude-3 系列)
- DeepSeek (DeepSeek-Chat, DeepSeek-Coder)
- 可扩展支持其他提供商

### AI功能配置

系统提供丰富的AI功能配置选项，可以在 `config.yaml` 文件中配置：

```yaml
ai:
  # 成本控制
  enable_cost_monitoring: true
  daily_budget: 10.0      # 每日预算（美元）
  monthly_budget: 100.0   # 每月预算（美元）

  # Prompt优化
  enable_prompt_optimization: true
  default_optimization_level: "balanced"  # minimal, balanced, aggressive

  # 批量处理
  max_concurrent_batch_tasks: 3
  max_workers_per_batch: 2

  # 模型性能监控
  enable_model_monitoring: true
  performance_data_retention_days: 90

  # 高级功能
  enable_context_management: true
  max_context_length: 4000
  enable_output_validation: true
```

也可以通过环境变量配置：
```bash
export ENABLE_COST_MONITORING=true
export DAILY_BUDGET=10.0
export ENABLE_PROMPT_OPTIMIZATION=true
```

### 输出格式

系统支持多种输出格式：
- Markdown (.md)
- 纯文本 (.txt)
- JSON (.json)

## 📝 使用示例

### 示例 1：处理会议视频

```bash
# 处理会议视频并生成会议纪要
videomind process "https://www.youtube.com/watch?v=meeting_video" \
  --template meeting_minutes \
  --output ./meetings/
```

### 示例 2：批量处理学习视频

```bash
# 创建包含视频链接的文件
echo "https://www.youtube.com/watch?v=video1" > urls.txt
echo "https://www.bilibili.com/video/BV1example" >> urls.txt

# 批量处理
videomind batch --file urls.txt --template study_notes
```

### 示例 3：自定义处理参数

```bash
# 使用特定模型和参数
videomind process "https://example.com/video" \
  --model gpt-4-turbo-preview \
  --temperature 0.7 \
  --max-tokens 2000 \
  --language zh
```

### 示例 4：使用AI功能优化

```bash
# 监控成本使用情况
videomind ai --cost

# 优化会议纪要模板的prompt
videomind ai --optimize meeting_minutes

# 查看模型性能，选择最佳模型
videomind ai --model-stats

# 批量处理时监控任务状态
videomind ai --batch-status

# 获取AI功能优化建议
videomind ai --insights
```

## 🛠️ 开发

### 运行测试

```bash
pytest tests/
```

### 代码格式化

```bash
black .
isort .
```

### 类型检查

```bash
mypy .
```

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 支持

如有问题，请：
1. 查看 [FAQ](docs/FAQ.md)
2. 提交 [Issue](https://github.com/yourusername/videomind/issues)
3. 查看 [文档](docs/)

---

**VideoMind** - 让视频内容更易理解，让知识获取更高效！