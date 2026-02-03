# Thesis_Reading_System 通用规范目录

## 概述

本目录包含多智能体系统的通用规范文件，定义了所有Agent共享的规则、工作流程、输出格式和质量评估标准。

## 文件清单

| 文件 | 说明 | 版本 |
|------|------|------|
| `common_rules.md` | 通用规则库，包含防幻觉协议和数据真实性原则 | 1.0 |
| `common_workflow.md` | 通用工作流程，定义6阶段标准流程 | 1.0 |
| `common_output.md` | 原子输出格式标准，定义Math/Code/Scenario/Strategic原子结构 | 1.0 |
| `common_quality_assessment.md` | 质量评估框架，定义各类原子的评分维度和等级标准 | 1.0 |
| `common_directory.md` | 目录结构标准，定义系统文件组织规范 | 1.0 |
| `common_profile.md` | Agent通用配置模板 | 1.0 |

## 规范层次结构

```
通用规范 (common/)
├── 规则层
│   └── common_rules.md          # 防幻觉、数据真实性、技术纯粹性
├── 流程层
│   └── common_workflow.md       # 6阶段标准工作流程
├── 输出层
│   └── common_output.md         # 原子JSON结构定义
├── 质量层
│   └── common_quality_assessment.md  # 评分维度和等级
└── 组织层
    ├── common_directory.md      # 目录结构
    └── common_profile.md        # Agent配置模板
```

## 与协议体系的关系

```
agents_system/
├── common/                 <-- 本目录：通用规范（Agent行为约束）
│   └── *.md
├── protocols/              <-- 调度协议（系统自动化调度）
│   ├── 01_task_package_protocol.md
│   ├── 02_agent_invocation_protocol.md
│   └── ...
└── *.md                    <-- Agent定义（具体角色实现）
```

**关系说明：**
- `common/` 定义Agent的行为规范和产出标准
- `protocols/` 定义系统如何自动化调度Agent
- Agent定义文件引用common规范，被protocols调度

## 使用指南

### 1. Agent开发者

在创建新Agent时，必须：
1. 引用 `common_rules.md` 中的规则
2. 遵循 `common_workflow.md` 的工作流程
3. 按 `common_output.md` 格式输出原子
4. 使用 `common_quality_assessment.md` 进行自评

### 2. 调度器开发者

在实现调度器时，应：
1. 参考 `common_output.md` 验证Agent产出
2. 使用 `common_quality_assessment.md` 评估质量
3. 结合 `protocols/` 目录下的协议实现自动化

### 3. 系统维护者

在维护系统时，需：
1. 保持各规范文件版本一致
2. 更新规范时同步更新相关Agent定义
3. 确保协议与规范的兼容性

## 核心原则

### 防幻觉原则 (来自 common_rules.md)

1. **数据真实性**：所有输出必须源自真实论文或源码
2. **强制锚定**：公式标明页码，代码标明行号
3. **无源则无果**：文献未提及则明确说明

### 质量标准 (来自 common_quality_assessment.md)

| 等级 | 分数范围 | 处理策略 |
|------|----------|----------|
| A级 | 85-100分 | 直接入库，优先推荐 |
| B级 | 70-84分 | 入库，标记需验证点 |
| C级 | 55-69分 | 限制入库，需人工审核 |
| D级 | 0-54分 | 拒绝入库，触发改进流程 |

## 版本历史

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| 1.0 | 2026-02-03 | 初始版本，整合所有通用规范 |

## 相关文档

- `../protocols/README.md` - 调度协议体系说明
- `../protocols/schemas/` - JSON Schema验证文件
- `../database/schema.sql` - 数据库表结构
