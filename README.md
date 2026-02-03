# Thesis Reading System

一个基于多Agent协作的智能论文阅读系统，用于深度分析学术论文并生成结构化的知识原子。

## 系统特点

- **智能PDF解析** - 自动提取论文标题、摘要、章节结构、参考文献
- **多Agent协作** - 5个专业Agent协同工作（Scholar、Code Architect、Validator、Critic、Knowledge Vault）
- **知识原子化** - 将论文知识转化为可复用的结构化原子（Concept、Method、Finding等）
- **LLM轮换** - 支持DeepSeek和Anthropic Claude轮换使用，提高稳定性
- **执行计划生成** - 自动生成详细的论文分析执行计划

## 系统架构

```
Thesis_Reading_System/
├── read_paper.py              # 主入口脚本
├── agents_system/             # Agent系统
│   ├── E2E-Learning-Orchestrator.md  # 学习计划生成Agent
│   ├── Scholar_Internalizer.md       # 学术分析Agent
│   ├── Code_Architect.md             # 代码架构分析Agent
│   ├── Scenario_Validator.md         # 场景验证Agent
│   ├── Strategic_Critic.md           # 战略评估Agent
│   ├── Knowledge_Vault.md            # 知识库管理Agent
│   └── runtime/               # 运行时模块
├── atoms/                     # 知识原子存储
├── manifests/                 # 索引和清单
├── outputs/                   # 分析输出
└── raw_papers/                # 原始PDF文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install pdfplumber openai anthropic
```

### 2. 配置环境变量

复制 `.env.example` 到 `.env` 并填入您的API密钥：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入API密钥。

### 3. 放置PDF文件

将要分析的论文PDF放入 `raw_papers/` 目录。

### 4. 运行分析

```bash
python3 read_paper.py
```

系统将自动：
1. 解析PDF文件
2. 生成学习计划
3. 深度分析关键章节
4. 生成知识原子并存储

## 输出说明

- `outputs/plan_*.md` - 学习计划
- `outputs/analysis_*.md` - 章节分析
- `atoms/` - 知识原子（JSON格式）
- `manifests/paper_index.json` - 论文索引

## 更新日志

### v0.1.0 (2026-02-03)

- 实现PDF解析器（支持双栏、无空格章节标题）
- 集成DeepSeek和Anthropic Claude
- 实现5个专业Agent
- 知识原子化存储
- 自动生成学习计划
- 深度分析关键章节

## 许可证

MIT License
