# Thesis_Reading_System 调度协议体系

## 概述

本目录定义了多智能体系统的完整调度协议，用于指导自动化调度器（Dispatcher）协调各Agent完成论文研读任务。

## 协议架构

```
+---------------------------------------------------------------------+
|                         协议层次结构                                  |
+---------------------------------------------------------------------+
|                                                                     |
|  +-------------------------------------------------------------+   |
|  |  01_task_package_protocol.md    任务包协议                    |   |
|  |  - Orchestrator输出的结构化任务包格式                          |   |
|  |  - 任务有向无环图(DAG)定义                                    |   |
|  |  - 各Agent的input/output规格                                 |   |
|  +-------------------------------------------------------------+   |
|                              |                                      |
|                              v                                      |
|  +-------------------------------------------------------------+   |
|  |  02_agent_invocation_protocol.md    Agent调用协议             |   |
|  |  - 调度器与Agent之间的通信格式                                 |   |
|  |  - System Prompt + User Prompt组装规则                        |   |
|  |  - LLM调用参数配置                                            |   |
|  +-------------------------------------------------------------+   |
|                              |                                      |
|                              v                                      |
|  +-------------------------------------------------------------+   |
|  |  03_data_transfer_protocol.md    数据传递协议                  |   |
|  |  - Agent之间的数据引用格式                                     |   |
|  |  - 依赖解析规则                                               |   |
|  |  - 上下文窗口管理策略                                          |   |
|  +-------------------------------------------------------------+   |
|                              |                                      |
|                              v                                      |
|  +-------------------------------------------------------------+   |
|  |  04_quality_gate_protocol.md    质量门协议                     |   |
|  |  - 质量检查点定义                                             |   |
|  |  - 评分规则和等级划分                                          |   |
|  |  - 熔断和降级策略                                             |   |
|  +-------------------------------------------------------------+   |
|                              |                                      |
|                              v                                      |
|  +-------------------------------------------------------------+   |
|  |  05_error_handling_protocol.md    错误处理协议                 |   |
|  |  - 错误分类和编码                                             |   |
|  |  - 重试策略                                                   |   |
|  |  - 降级处理流程                                               |   |
|  +-------------------------------------------------------------+   |
|                                                                     |
+---------------------------------------------------------------------+
```

## 协议文件清单

| 文件 | 说明 | 状态 |
|------|------|------|
| `01_task_package_protocol.md` | 任务包协议 | Done |
| `02_agent_invocation_protocol.md` | Agent调用协议 | Done |
| `03_data_transfer_protocol.md` | 数据传递协议 | Done |
| `04_quality_gate_protocol.md` | 质量门协议 | Done |
| `05_error_handling_protocol.md` | 错误处理协议 | Done |
| `schemas/` | JSON Schema定义 | Done |

## JSON Schema文件

| Schema文件 | 说明 |
|------------|------|
| `task_package.schema.json` | 任务包结构验证 |
| `agent_request.schema.json` | Agent调用请求验证 |
| `agent_response.schema.json` | Agent调用响应验证 |
| `atom_base.schema.json` | 原子基础结构验证 |

## 执行流程概览

```
1. 用户输入研读请求
       |
       v
2. 调度器调用 Orchestrator Agent
       |
       v
3. Orcrator 输出 01_plan.json (符合任务包协议)
       |
       v
4. 调度器解析任务包，构建执行DAG
       |
       v
5. 按DAG顺序调用各Agent (符合Agent调用协议)
   - 并行阶段: Scholar, Code, Validator 同时执行
   - 串行阶段: Knowledge_Vault -> Strategic_Critic
       |
       v
6. 每个阶段执行质量门检查 (符合质量门协议)
       |
       v
7. 遇到错误时按错误处理协议处理
       |
       v
8. 输出最终研读报告和决策建议
```

## 与现有系统的关系

```
agents_system/
|-- protocols/              <-- 本目录：调度协议定义
|   |-- *.md               <-- 协议文档
|   +-- schemas/           <-- JSON Schema
|-- common/                 <-- 通用规范（已有）
|   |-- common_rules.md    <-- 规则库
|   |-- common_workflow.md <-- 工作流程
|   |-- common_output.md   <-- 输出格式
|   +-- ...
|-- database/              <-- 数据库设计
|   +-- schema.sql
+-- *.md                   <-- Agent定义（已有）
```

## 版本信息

- **版本**: 1.0
- **创建时间**: 2026-02-03
- **适用范围**: Thesis_Reading_System 自动化调度
- **维护者**: System Architect

## 版本兼容性矩阵

| 协议文件 | 当前版本 | 兼容的Schema版本 | 兼容的common版本 |
|----------|----------|------------------|------------------|
| 01_task_package_protocol.md | 1.0 | task_package.schema.json v1.0 | common_output.md v1.0 |
| 02_agent_invocation_protocol.md | 1.0 | agent_request/response.schema.json v1.0 | common_workflow.md v1.0 |
| 03_data_transfer_protocol.md | 1.0 | - | common_output.md v1.0 |
| 04_quality_gate_protocol.md | 1.0 | - | common_quality_assessment.md v1.0 |
| 05_error_handling_protocol.md | 1.0 | - | common_rules.md v1.0 |

**版本更新规则：**
- 主版本号变更（如1.0→2.0）：不兼容的协议变更
- 次版本号变更（如1.0→1.1）：向后兼容的功能增强
- 更新协议时必须同步更新兼容性矩阵

## 快速开始

1. 阅读 `01_task_package_protocol.md` 了解任务包格式
2. 阅读 `02_agent_invocation_protocol.md` 了解如何调用Agent
3. 参考 `schemas/` 目录下的JSON Schema进行数据验证
4. 实现调度器时遵循各协议的规范

## 相关文档

- `../common/common_rules.md` - 通用规则库
- `../common/common_workflow.md` - 通用工作流程
- `../common/common_output.md` - 原子输出格式
- `../common/common_quality_assessment.md` - 质量评估框架
- `../database/schema.sql` - 数据库表结构
