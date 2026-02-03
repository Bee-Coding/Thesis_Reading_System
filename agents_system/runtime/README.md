# Thesis_Reading_System 运行时模块

## 概述

本目录包含多智能体系统的运行时代码实现，基于 `protocols/` 目录下定义的协议规范。

## 目录结构

```
runtime/
├── config/              # 配置文件
│   ├── settings.py      # 全局配置
│   └── agents.yaml      # Agent配置
├── dispatcher/          # 调度器模块
│   ├── __init__.py
│   ├── dispatcher.py    # 核心调度逻辑
│   ├── task_parser.py   # 任务包解析
│   └── dag_executor.py  # DAG执行引擎
├── agent_invoker/       # Agent调用模块
│   ├── __init__.py
│   ├── invoker.py       # LLM API封装
│   ├── prompt_builder.py # Prompt构建
│   └── response_parser.py # 响应解析
├── quality_gate/        # 质量门引擎
│   ├── __init__.py
│   ├── gate_engine.py   # 质量门执行
│   ├── checkers.py      # 检查器实现
│   └── scorer.py        # 评分逻辑
├── data_transfer/       # 数据传递模块
│   ├── __init__.py
│   ├── reference_resolver.py # 引用解析
│   └── context_manager.py    # 上下文管理
├── error_handler/       # 错误处理模块
│   ├── __init__.py
│   ├── handler.py       # 错误处理逻辑
│   ├── retry.py         # 重试策略
│   └── circuit_breaker.py # 熔断器
├── database/            # 数据库模块
│   ├── __init__.py
│   ├── connection.py    # 数据库连接
│   ├── models.py        # ORM模型
│   └── repository.py    # 数据访问层
└── main.py              # 主入口
```

## 模块依赖关系

```
main.py
    └── dispatcher/
            ├── agent_invoker/
            ├── quality_gate/
            ├── data_transfer/
            ├── error_handler/
            └── database/
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 设置 API 密钥等

# 3. 初始化数据库
python -m runtime.database.init

# 4. 运行调度器
python -m runtime.main --plan path/to/plan.json
```

## 配置说明

详见 `config/settings.py` 和 `config/agents.yaml`
