# Thesis Reading System - 安装指南

## 系统要求

- Python 3.8+
- 4GB+ RAM
- 网络连接（用于API调用）

## 安装步骤

### 1. 克隆仓库

```bash
git clone <repository-url>
cd Thesis_Reading_System
```

### 2. 创建虚拟环境（推荐）

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

如果没有 `requirements.txt`，手动安装核心依赖：

```bash
pip install pdfplumber openai anthropic
```

### 4. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入您的API密钥：

```bash
# DeepSeek API
DEEPSEEK_API_KEY=sk-your-deepseek-key
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-reasoner

# Anthropic API (Claude)
ANTHROPIC_API_KEY=sk-your-anthropic-key
ANTHROPIC_API_BASE=https://api.anthropic.com
ANTHROPIC_MODEL=claude-opus-4-5

# LLM轮换策略
LLM_ROTATION_STRATEGY=round_robin
```

### 5. 验证安装

```bash
# 测试配置
python3 check_config.py

# 测试LLM连接
python3 test_llm.py
```

### 6. 创建必要目录

```bash
mkdir -p raw_papers outputs atoms/concepts atoms/methods atoms/findings manifests
```

## 依赖说明

### 核心依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| pdfplumber | >= 0.10.0 | PDF文本提取 |
| openai | >= 1.0.0 | DeepSeek API客户端 |
| anthropic | >= 0.18.0 | Claude API客户端 |

### 可选依赖

```bash
# PDF解析备用方案
pip install PyMuPDF  # fitz

# 数据库支持（如需持久化）
pip install asyncpg psycopg2-binary

# 可视化（如需生成图表）
pip install matplotlib networkx
```

## 获取API密钥

### DeepSeek

1. 访问 https://platform.deepseek.com/
2. 注册账号
3. 在控制台创建API密钥
4. 复制密钥到 `.env` 文件

### Anthropic Claude

1. 访问 https://console.anthropic.com/
2. 注册账号
3. 在API Keys页面创建密钥
4. 复制密钥到 `.env` 文件

**注意**：如果使用第三方代理，需要修改 `ANTHROPIC_API_BASE`。

## 故障排查

### 问题：pdfplumber安装失败

```bash
# 安装系统依赖（Ubuntu/Debian）
sudo apt-get install python3-dev libpoppler-cpp-dev

# 或使用conda
conda install -c conda-forge pdfplumber
```

### 问题：openai包版本冲突

```bash
# 卸载旧版本
pip uninstall openai

# 安装最新版本
pip install openai>=1.0.0
```

### 问题：anthropic包导入错误

```bash
# 确保安装了正确的包
pip install anthropic --upgrade
```

### 问题：权限错误

```bash
# Linux/Mac
chmod +x read_paper.py

# 或使用python3直接运行
python3 read_paper.py
```

## 配置验证

运行以下命令验证配置：

```bash
python3 << 'PYEOF'
import sys
from pathlib import Path

print("检查Python版本...")
print(f"Python {sys.version}")

print("\n检查依赖包...")
try:
    import pdfplumber
    print("✓ pdfplumber")
except ImportError:
    print("✗ pdfplumber - 请运行: pip install pdfplumber")

try:
    import openai
    print("✓ openai")
except ImportError:
    print("✗ openai - 请运行: pip install openai")

try:
    import anthropic
    print("✓ anthropic")
except ImportError:
    print("✗ anthropic - 请运行: pip install anthropic")

print("\n检查目录结构...")
dirs = ['raw_papers', 'outputs', 'atoms', 'manifests']
for d in dirs:
    if Path(d).exists():
        print(f"✓ {d}/")
    else:
        print(f"✗ {d}/ - 请创建此目录")

print("\n检查环境变量...")
import os
if os.getenv("DEEPSEEK_API_KEY"):
    print("✓ DEEPSEEK_API_KEY")
else:
    print("✗ DEEPSEEK_API_KEY - 请在.env中配置")

if os.getenv("ANTHROPIC_API_KEY"):
    print("✓ ANTHROPIC_API_KEY")
else:
    print("✗ ANTHROPIC_API_KEY - 请在.env中配置")

print("\n配置检查完成！")
PYEOF
```

## Docker部署（可选）

创建 `Dockerfile`：

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libpoppler-cpp-dev \
    && rm -rf /var/lib//lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 创建必要目录
RUN mkdir -p raw_papers outputs atoms manifests

CMD ["python3", "read_paper.py"]
```

构建和运行：

```bash
docker build -t thesis-reader .
docker run -v $(pwd)/raw_papers:/app/raw_papers \
           -v $(pwd)/outputs:/app/outputs \
           --env-file .env \
           thesis-reader
```

## 下一步

安装完成后，请查看：
- [使用指南](USAGE.md) - 了解如何使用系统
- [README](README.md) - 系统概述和快速开始

如有问题，请查看项目Issue或联系维护者。
