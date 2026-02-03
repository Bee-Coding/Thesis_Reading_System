# 错误处理协议 (Error Handling Protocol)

## 1. 概述

### 1.1 协议目的

本协议定义了多智能体系统中的标准化错误处理机制，确保：

- **错误可分类**：所有错误都有明确的分类和编码，便于自动化处理
- **处理可预测**：针对每类错误都有预定义的处理策略和恢复路径
- **系统高可用**：通过重试、降级、熔断等机制保证系统持续运行
- **问题可追溯**：完整的错误链路追踪，支持根因分析和持续改进

### 1.2 错误处理原则

| 原则 | 说明 | 实施要求 |
|------|------|----------|
| **快速失败** | 在不可恢复错误发生时立即终止，避免资源浪费 | 设置明确的失败条件 |
| **优雅降级** | 在部分功能不可用时，提供可接受的替代方案 | 实现降级策略 |
| **自动恢复** | 对可恢复错误，系统应能自动尝试恢复 | 实现重试机制 |
| **渐进式披露** | 错误信息应根据用户角色和场景逐步披露 | 分层错误消息 |
| **持续改进** | 错误数据应用于系统优化和预防措施 | 错误分析和反馈循环 |

### 1.3 协议版本

- **版本**: 1.0
- **创建时间**: 2026-02-03
- **兼容性**: 与 `02_agent_invocation_protocol.md` v1.0 兼容

---

## 2. 错误分类体系

### 2.1 错误分类维度

```
错误分类体系
├── 按严重程度 (Severity)
│   ├── Critical (致命) - 系统无法继续执行
│   ├── Error (错误) - 功能失败，可能影响结果质量
│   ├── Warning (警告) - 潜在问题，不影响主要功能
│   └── Info (信息) - 状态提示，无需处理
│
├── 按可恢复性 (Recoverability)
│   ├── Recoverable (可恢复) - 自动重试可解决
│   ├── Partially Recoverable (部分可恢复) - 降级后可继续
│   └── Non-recoverable (不可恢复) - 需要人工干预
│
├── 按来源 (Source)
│   ├── System (系统) - 基础设施、资源问题
│   ├── Agent (智能体) - Agent执行错误
│   ├── Data (数据) - 数据质量或访问问题
│   └── External (外部) - 第三方服务问题
│
└── 按影响范围 (Scope)
    ├── Task (任务级) - 影响单个任务
    ├── Stage (阶段级) - 影响任务阶段
    ├── Plan (计划级) - 影响整个执行计划
    └── System (系统级) - 影响整个系统
```

### 2.2 错误码规范

错误码格式：`E{分类}{子类}{序号}`

```
E  X  YY  ZZZ
│  │  │   └── 具体错误序号 (001-999)
│  │  └────── 子类代码 (01-99)
│  └───────── 分类代码 (A-Z)
└──────────── 错误标识
```

**分类代码：**
- `S`: 系统错误 (System)
- `A`: Agent错误 (Agent)
- `D`: 数据错误 (Data)
- `V`: 验证错误 (Validation)
- `Q`: 质量错误 (Quality)
- `R`: 资源错误 (Resource)
- `E`: 外部错误 (External)

---

## 3. 错误码详细定义

### 3.1 系统错误 (E00x系列)

| 错误码 | 严重程度 | 可恢复性 | 说明 | 触发条件 |
|--------|----------|----------|------|----------|
| `E001` | Critical | Partially | 源文件/仓库无法访问 | 论文PDF、代码仓库不可达 |
| `E002` | Error | Recoverable | 输入解析失败 | JSON格式错误、字段缺失 |
| `E003` | Error | Recoverable | 输出验证失败 | 原子格式不符合规范 |
| `E004` | Warning | Recoverable | 执行超时 | 任务执行超过timeout |
| `E005` | Error | Partially | 质量低于阈值 | 原子质量分数低于threshold |
| `E006` | Critical | Non-recoverable | 依赖任务失败 | 前置任务失败导致当前任务无法执行 |
| `E007` | Critical | Non-recoverable | 检测到幻觉内容 | Agent产出与源材料严重不符 |
| `E008` | Critical | Partially | 系统内部错误 | 未知异常、内存溢出等 |

### 3.2 Agent错误 (EAxx系列)

| 错误码 | 严重程度 | 可恢复性 | 说明 | 触发条件 |
|--------|----------|----------|------|----------|
| `EA01` | Critical | Non-recoverable | Agent初始化失败 | System Prompt加载失败 |
| `EA02` | Error | Recoverable | Agent执行异常 | LLM API调用异常 |
| `EA03` | Warning | Recoverable | Agent响应格式错误 | 响应不符合预期格式 |
| `EA04` | Error | Partially | Agent技能不足 | 无法处理特定类型任务 |
| `EA05` | Critical | Non-recoverable | Agent配置错误 | 参数配置不合理 |

### 3.3 数据错误 (EDxx系列)

| 错误码 | 严重程度 | 可恢复性 | 说明 | 触发条件 |
|--------|----------|----------|------|----------|
| `ED01` | Error | Recoverable | 数据引用解析失败 | JSONPath解析错误 |
| `ED02` | Error | Partially | 数据格式不匹配 | 数据类型不符合预期 |
| `ED03` | Warning | Recoverable | 数据不完整 | 必需字段缺失 |
| `ED04` | Error | Partially | 数据一致性错误 | 跨源数据矛盾 |
| `ED05` | Critical | Non-recoverable | 数据损坏 | 文件损坏、数据库损坏 |

### 3.4 验证错误 (EVxx系列)

| 错误码 | 严重程度 | 可恢复性 | 说明 | 触发条件 |
|--------|----------|----------|------|----------|
| `EV01` | Error | Recoverable | 质量门检查失败 | 预执行门检查未通过 |
| `EV02` | Warning | Recoverable | 一致性检查失败 | 跨任务产出不一致 |
| `EV03` | Error | Partially | 完整性检查失败 | 产出数量不足 |
| `EV04` | Critical | Non-recoverable | 安全验证失败 | 内容违反安全策略 |
| `EV05` | Error | Recoverable | 合规性检查失败 | 不符合行业标准 |

### 3.5 质量错误 (EQxx系列)

| 错误码 | 严重程度 | 可恢复性 | 说明 | 触发条件 |
|--------|----------|----------|------|----------|
| `EQ01` | Warning | Recoverable | 质量评分低于预期 | 原子质量分低于阈值 |
| `EQ02` | Error | Partially | 质量维度缺失 | 关键质量维度未评估 |
| `EQ03` | Warning | Recoverable | 质量评估不一致 | 自评与交叉评差异常 |
| `EQ04` | Error | Partially | 质量改进失败 | 质量优化建议无法实施 |
| `EQ05` | Critical | Non-recoverable | 质量熔断触发 | 连续低质量产出 |

### 3.6 资源错误 (ERxx系列)

| 错误码 | 严重程度 | 可恢复性 | 说明 | 触发条件 |
|--------|----------|----------|------|----------|
| `ER01` | Error | Recoverable | 上下文窗口不足 | 输入数据超过token限制 |
| `ER02` | Critical | Partially | GPU内存不足 | 显存溢出 |
| `ER03` | Warning | Recoverable | API配额不足 | LLM API调用次数超限 |
| `ER04` | Error | Partially | 存储空间不足 | 磁盘空间不足 |
| `ER05` | Critical | Non-recoverable | 网络连接失败 | 无法访问外部服务 |

### 3.7 外部错误 (EExx系列)

| 错误码 | 严重程度 | 可恢复性 | 说明 | 触发条件 |
|--------|----------|----------|------|----------|
| `EE01` | Warning | Recoverable | 外部服务临时故障 | GitHub API限流 |
| `EE02` | Error | Partially | 外部数据源变更 | 论文版本更新 |
| `EE03` | Critical | Non-recoverable | 外部认证失败 | API密钥失效 |
| `EE04` | Warning | Recoverable | 外部依赖版本不兼容 | 依赖库版本冲突 |
| `EE05` | Error | Partially | 外部服务响应异常 | 返回数据格式异常 |

---

## 4. 错误处理策略

### 4.1 策略选择矩阵

| 错误类型 | 首选策略 | 备选策略 | 升级策略 |
|----------|----------|----------|----------|
| 可恢复错误 | 自动重试 | 参数调整 | 人工检查 |
| 部分可恢复错误 | 优雅降级 | 功能裁剪 | 人工介入 |
| 不可恢复错误 | 快速失败 | 状态保存 | 人工修复 |
| 系统性错误 | 熔断保护 | 服务降级 | 系统重启 |

### 4.2 重试策略 (Retry Strategy)

```json
{
  "retry_policy": {
    "max_attempts": 3,
    "backoff_strategy": "exponential",
    "base_delay_seconds": 5,
    "max_delay_seconds": 300,
    "jitter": true,
    "retryable_errors": ["E001", "E002", "E003", "E004", "EA02", "ED01"],
    "condition_check": {
      "pre_retry_check": "validate_input_still_valid",
      "post_retry_check": "validate_output_quality"
    }
  }
}
```

**退避策略类型：**
- `fixed`: 固定间隔重试
- `linear`: 线性增加间隔
- `exponential`: 指数增加间隔
- `random`: 随机间隔，避免惊群

### 4.3 降级策略 (Degradation Strategy)

```json
{
  "degradation_policy": {
    "trigger_conditions": ["E005", "EQ01", "ER01"],
    "degradation_levels": [
      {
        "level": "light",
        "actions": [
          "reduce_output_detail",
          "increase_timeout",
          "lower_quality_threshold"
        ],
        "applicable_errors": ["E004", "EQ01"]
      },
      {
        "level": "moderate",
        "actions": [
          "skip_optional_checks",
          "use_cached_data",
          "simplify_analysis"
        ],
        "applicable_errors": ["E001", "E005"]
      },
      {
        "level": "heavy",
        "actions": [
          "switch_to_backup_agent",
          "use_alternative_algorithm",
          "output_summary_only"
        ],
        "applicable_errors": ["EA04", "ER02"]
      }
    ],
    "recovery_condition": "resource_available || manual_override"
  }
}
```

### 4.4 熔断策略 (Circuit Breaker Strategy)

```json
{
  "circuit_breaker": {
    "failure_threshold": 3,
    "reset_timeout": "PT5M",
    "half_open_max_attempts": 1,
    "states": ["closed", "open", "half_open"],
    "metrics": {
      "failure_rate": 0.5,
      "slow_call_rate": 0.2,
      "minimum_number_of_calls": 10
    },
    "notification": {
      "on_open": ["slack", "email"],
      "on_close": ["slack"],
      "on_half_open": ["log"]
    }
  }
}
```

**熔断状态转换：**
```
正常状态 (closed)
    ↓ (连续失败 > threshold)
熔断状态 (open) → 定时器 (reset_timeout)
    ↓ (定时器到期)
半开状态 (half_open) → 尝试请求
    ↓ (成功)            ↓ (失败)
正常状态 (closed)     熔断状态 (open)
```

### 4.5 隔离策略 (Isolation Strategy)

```json
{
  "isolation_policy": {
    "bulkhead": {
      "max_concurrent_calls": 10,
      "max_wait_time_ms": 5000
    },
    "timeout": {
      "task_level": "PT10M",
      "stage_level": "PT30M",
      "plan_level": "PT2H"
    },
    "resource_limits": {
      "max_memory_mb": 4096,
      "max_cpu_percent": 80,
      "max_disk_gb": 50
    }
  }
}
```

---

## 5. 错误处理流程

### 5.1 错误检测与捕获

```
错误处理流程
├── 错误检测
│   ├── 主动检测: 质量门检查、验证规则
│   ├── 被动检测: 异常捕获、超时监控
│   └── 预测检测: 基于历史数据的异常预测
│
├── 错误分类
│   ├── 提取错误特征
│   ├── 匹配错误模式
│   └── 确定错误编码
│
├── 策略选择
│   ├── 根据错误编码选择处理策略
│   ├── 考虑上下文环境
│   └── 应用策略优先级
│
└── 执行处理
    ├── 执行选定策略
    ├── 监控处理效果
    └── 记录处理结果
```

### 5.2 错误恢复流程

```json
{
  "error_recovery_workflow": {
    "step1": {
      "action": "error_identification",
      "timeout": "PT30S",
      "fallback": "default_to_unknown_error"
    },
    "step2": {
      "action": "context_preservation",
      "save_state": ["input_data", "intermediate_results", "execution_context"],
      "storage": "persistent_storage"
    },
    "step3": {
      "action": "strategy_selection",
      "selection_criteria": ["error_code", "recoverability", "impact_scope"],
      "default_strategy": "retry"
    },
    "step4": {
      "action": "strategy_execution",
      "monitoring": ["progress", "resource_usage", "side_effects"],
      "timeout": "PT5M"
    },
    "step5": {
      "action": "recovery_validation",
      "validation_criteria": ["output_quality", "system_integrity", "data_consistency"],
      "failure_action": "escalate"
    }
  }
}
```

### 5.3 错误上报与通知

```json
{
  "error_reporting": {
    "report_levels": [
      {
        "level": "debug",
        "audience": ["developers"],
        "channels": ["log_file", "debug_console"],
        "content": "完整错误堆栈、上下文数据"
      },
      {
        "level": "operational",
        "audience": ["operators", "sre"],
        "channels": ["slack", "monitoring_dashboard"],
        "content": "错误摘要、影响范围、处理状态"
      },
      {
        "level": "user",
        "audience": ["end_users"],
        "channels": ["ui_notification", "email"],
        "content": "用户友好错误消息、预计恢复时间、建议操作"
      }
    ],
    "notification_rules": {
      "immediate": ["critical", "non_recoverable"],
      "delayed": ["error", "partially_recoverable"],
      "batch": ["warning", "info"]
    }
  }
}
```

---

## 6. 错误数据模型

### 6.1 错误对象结构

```json
{
  "error_record": {
    "error_id": "ERR_20260203_143052_001",
    "timestamp": "2026-02-03T14:30:52Z",
    "error_code": "E001",
    "severity": "critical",
    "recoverability": "partially_recoverable",
    "source": {
      "component": "Scholar_Internalizer",
      "task_id": "TASK_SCHOLAR_DIFF_01",
      "stage_id": "S1",
      "plan_id": "AVP_E2E_20260203_001"
    },
    "context": {
      "input_data": {...},
      "execution_state": "pre_execution",
      "resource_usage": {"memory_mb": 512, "cpu_percent": 30},
      "environment": {"python_version": "3.9", "os": "linux"}
    },
    "details": {
      "message": "无法访问指定的GitHub仓库",
      "root_cause": "网络连接超时",
      "stack_trace": "...",
      "additional_data": {"repo_url": "https://github.com/example/repo"}
    },
    "handling": {
      "strategy_applied": "retry",
      "attempts_made": 1,
      "current_status": "in_progress",
      "next_action": "wait_for_retry",
      "estimated_recovery_time": "PT5M"
    },
    "impact": {
      "affected_tasks": ["TASK_SCHOLAR_DIFF_01"],
      "blocked_tasks": ["TASK_VALIDATOR_DIFF_01"],
      "estimated_delay": "PT15M",
      "quality_impact": "medium"
    },
    "metadata": {
      "reported_by": "Dispatcher",
      "version": "1.0",
      "tags": ["network", "external_service"]
    }
  }
}
```

### 6.2 错误聚合视图

```json
{
  "error_aggregation": {
    "time_window": "PT1H",
    "summary": {
      "total_errors": 15,
      "by_severity": {"critical": 2, "error": 5, "warning": 8},
      "by_recoverability": {"recoverable": 10, "partially": 3, "non": 2},
      "by_component": {"Scholar": 8, "Code": 4, "System": 3}
    },
    "trends": {
      "error_rate": 2.5,
      "recovery_rate": 0.8,
      "mttr": "PT8M", // Mean Time To Recovery
      "mttf": "PT45M" // Mean Time To Failure
    },
    "top_errors": [
      {
        "error_code": "E001",
        "count": 5,
        "last_occurrence": "2026-02-03T14:25:00Z",
        "common_causes": ["network_issue", "repo_private"],
        "suggested_fixes": ["check_connectivity", "verify_permissions"]
      }
    ]
  }
}
```

---

## 7. 错误预防与改进

### 7.1 错误预防措施

| 预防措施 | 实施方法 | 目标错误类型 |
|----------|----------|--------------|
| **输入验证** | 执行前验证输入完整性、格式 | E002, E003 |
| **资源预留** | 预分配足够资源，设置使用上限 | ER01, ER02 |
| **健康检查** | 定期检查组件健康状态 | EA01, EE01 |
| **容错设计** | 设计冗余和备用路径 | E006, EE02 |
| **监控预警** | 实时监控关键指标，提前预警 | E004, EQ05 |

### 7.2 错误根因分析 (RCA)

```json
{
  "root_cause_analysis": {
    "analysis_framework": "5 Whys",
    "data_collection": [
      "error_records",
      "system_logs",
      "performance_metrics",
      "user_feedback"
    ],
    "analysis_steps": [
      {
        "step": "problem_definition",
        "questions": ["什么错误？", "何时发生？", "影响范围？"]
      },
      {
        "step": "data_collection",
        "sources": ["monitoring", "logging", "tracing"]
      },
      {
        "step": "cause_identification",
        "techniques": ["fishbone", "fault_tree", "timeline"]
      },
      {
        "step": "solution_development",
        "considerations": ["effectiveness", "cost", "timeline"]
      },
      {
        "step": "implementation",
        "phases": ["immediate", "short_term", "long_term"]
      }
    ],
    "output": {
      "rca_report": "根因分析报告",
      "action_items": ["具体改进措施"],
      "prevention_plan": ["预防计划"],
      "metrics": ["改进效果指标"]
    }
  }
}
```

### 7.3 持续改进循环

```
错误数据收集 → 模式识别 → 根因分析 → 改进实施 → 效果验证
     ↑                                           ↓
     └───────────────────────────────────────────┘
```

**改进措施类型：**
- **立即措施**: 热修复、配置调整
- **短期措施**: 代码修复、流程优化
- **长期措施**: 架构改进、技术升级

---

## 8. 完整示例

### 8.1 错误处理完整流程示例

**场景**: Scholar_Internalizer 无法访问论文PDF

```json
{
  "error_scenario": {
    "detection": {
      "timestamp": "2026-02-03T14:30:52Z",
      "detected_by": "pre_execution_gate",
      "detection_method": "source_accessibility_check",
      "error_signature": "file_not_found"
    },
    "classification": {
      "error_code": "E001",
      "severity": "critical",
      "recoverability": "partially_recoverable",
      "impact_scope": "task_level"
    },
    "handling": {
      "strategy_selected": "retry_with_fallback",
      "actions_taken": [
        {
          "attempt": 1,
          "action": "retry_original_source",
          "result": "failed",
          "duration": "PT30S"
        },
        {
          "attempt": 2,
          "action": "try_alternative_source",
          "parameters": {"source": "arxiv_cache"},
          "result": "success",
          "duration": "PT10S"
        }
      ],
      "final_outcome": "recovered_with_degradation",
      "degradation_level": "light"
    },
    "reporting": {
      "internal_report": {
        "level": "operational",
        "channels": ["slack", "dashboard"],
        "message": "E001: 论文源访问失败，已使用缓存版本"
      },
      "user_report": {
        "level": "user",
        "channels": ["ui_notification"],
        "message": "系统正在使用缓存的论文版本进行分析，部分最新更新可能未包含"
      }
    },
    "improvement": {
      "root_cause": "论文服务器临时维护",
      "preventive_action": "增强缓存机制，增加多个备用源",
      "action_priority": "medium",
      "assigned_to": "infrastructure_team"
    }
  }
}
```

### 8.2 错误配置示例

```json
{
  "error_handling_config": {
    "global_policies": {
      "retry_policy": {
        "max_attempts": 3,
        "backoff_strategy": "exponential",
        "base_delay_seconds": 5
      },
      "circuit_breaker": {
        "failure_threshold": 3,
        "reset_timeout": "PT5M"
      },
      "degradation_policy": {
        "levels": ["light", "moderate", "heavy"],
        "auto_escalate": true
      }
    },
    "error_specific_policies": {
      "E001": {
        "primary_strategy": "retry",
        "fallback_strategy": "use_cached",
        "notification": {"channels": ["slack"], "urgency": "high"}
      },
      "E005": {
        "primary_strategy": "degrade",
        "degrade_level": "light",
        "quality_threshold_adjustment": -0.1
      },
      "E007": {
        "primary_strategy": "abort",
        "requires_human_review": true,
        "notification": {"channels": ["slack", "email"], "urgency": "critical"}
      }
    },
    "agent_specific_policies": {
      "Scholar_Internalizer": {
        "common_errors": ["E001", "E005", "EQ01"],
        "suggested_actions": ["verify_sources", "adjust_temperature"]
      },
      "Code_Architect": {
        "common_errors": ["ER02", "ED02", "EV03"],
        "suggested_actions": ["optimize_memory", "validate_input_format"]
      }
    }
  }
}
```

### 8.3 错误仪表板示例

```json
{
  "error_dashboard": {
    "current_status": {
      "open_errors": 5,
      "critical_errors": 1,
      "recovery_in_progress": 2,
      "system_health": "degraded"
    },
    "recent_trends": {
      "error_rate_last_hour": 2.3,
      "recovery_rate": 0.85,
      "top_error_codes": ["E001", "E004", "EQ01"],
      "most_affected_components": ["Scholar", "Dispatcher"]
    },
    "impact_assessment": {
      "affected_tasks": 3,
      "delayed_plans": 1,
      "quality_impact": "low",
      "estimated_recovery_time": "PT20M"
    },
    "action_items": [
      {
        "priority": "high",
        "action": "investigate_E001_spike",
        "assigned_to": "sre_team",
        "due_by": "2026-02-03T15:30:00Z"
      },
      {
        "priority": "medium",
        "action": "optimize_context_window_usage",
        "assigned_to": "performance_team",
        "due_by": "2026-02-04T10:00:00Z"
      }
    ]
  }
}
```

---

## 9. 实施指南

### 9.1 错误处理集成点

| 组件 | 集成点 | 错误处理职责 |
|------|--------|--------------|
| **Orchestrator** | 计划生成时 | 预定义错误处理策略 |
| **Dispatcher** | 任务调度时 | 错误检测、策略执行 |
| **Agent** | 任务执行时 | 错误抛出、局部恢复 |
| **Quality Gate** | 质量检查时 | 质量错误检测 |
| **Monitor** | 系统运行时 | 错误聚合、预警 |

### 9.2 错误处理测试策略

| 测试类型 | 测试方法 | 验证目标 |
|----------|----------|----------|
| **错误注入测试** | 模拟各种错误场景 | 错误检测和分类准确性 |
| **恢复测试** | 触发错误后验证恢复 | 恢复策略有效性 |
| **压力测试** | 高错误率场景 | 系统稳定性 |
| **回归测试** | 修复后验证 | 错误不再重现 |

### 9.3 监控与告警配置

```yaml
监控指标:
  - error_rate: 错误发生频率
  - recovery_rate: 错误恢复成功率
  - mttr: 平均恢复时间
  - impact_score: 错误影响程度

告警规则:
  - critical_error_rate > 5/min: P1告警
  - recovery_rate < 0.7: P2告警  
  - mttr > PT30M: P3告警
  - system_health == "degraded": P2告警
```

---

## 10. 版本历史

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| 1.0 | 2026-02-03 | 初始版本 |

## 11. 相关文档

- `01_task_package_protocol.md` - 任务包协议
- `02_agent_invocation_protocol.md` - Agent调用协议（包含基础错误码）
- `03_data_transfer_protocol.md` - 数据传递协议
- `04_quality_gate_protocol.md` - 质量门协议
- `../common/common_quality_assessment.md` - 质量评估框架
- `../common/common_output.md` - 原子输出格式标准