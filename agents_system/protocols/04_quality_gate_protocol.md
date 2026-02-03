# 质量门协议 (Quality Gate Protocol)

## 1. 概述

### 1.1 协议目的

本协议定义了多智能体系统中的质量门控机制，确保：

- **过程质量控制**：在关键执行节点设置检查点，防止低质量产出进入下一阶段
- **风险早期识别**：通过预定义的质量标准，提前发现潜在问题
- **资源优化**：避免在低质量产出上继续投入计算资源
- **质量可追溯**：每个质量决策都有明确的标准和依据

### 1.2 质量门类型

| 门类型 | 触发时机 | 检查重点 | 决策影响 |
|--------|----------|----------|----------|
| **预执行门** | 任务执行前 | 输入完整性、依赖就绪、资源可用性 | 是否允许任务执行 |
| **执行中门** | 任务执行中 | 进度监控、中间结果质量、资源消耗 | 是否继续执行或调整参数 |
| **后执行门** | 任务完成后 | 产出质量、符合度、完整性 | 是否接受产出、触发重试或降级 |
| **集成门** | 阶段集成时 | 跨任务一致性、逻辑自洽性、整体质量 | 是否允许进入下一阶段 |

### 1.3 协议版本

- **版本**: 1.0
- **创建时间**: 2026-02-03
- **兼容性**: 与 `common_quality_assessment.md` v1.0 兼容

---

## 2. 质量门通用结构

### 2.1 质量门定义

```json
{
  "gate_id": "QG_PRE_SCHOLAR_01",
  "gate_type": "pre_execution | in_execution | post_execution | integration",
  "task_id": "TASK_SCHOLAR_DIFF_01",
  "stage_id": "S1",
  "checks": [
    {
      "check_id": "CHECK_001",
      "check_type": "source_accessibility | dependency_resolved | output_quality | ...",
      "target": "string - 检查目标",
      "condition": "string - 布尔条件表达式",
      "error_code": "string - 错误码",
      "severity": "critical | warning | info"
    }
  ],
  "failure_action": "block | retry | degrade | skip | abort",
  "action_config": {
    "retry_count": 3,
    "degrade_threshold": "C",
    "notification_channel": "slack | email"
  },
  "metadata": {
    "created_at": "2026-02-03T10:30:00Z",
    "created_by": "Orchestrator",
    "version": "1.0"
  }
}
```

### 2.2 检查类型定义

| 检查类型 | 适用门类型 | 检查内容 | 条件示例 |
|----------|------------|----------|----------|
| `source_accessibility` | pre_execution | 数据源可访问性 | `paper_accessible == true` |
| `dependency_resolved` | pre_execution | 依赖任务完成状态 | `dependency.status == "success"` |
| `context_window` | pre_execution | 上下文窗口是否充足 | `available_tokens > required_tokens` |
| `progress_check` | in_execution | 执行进度监控 | `elapsed_time < timeout * 0.5` |
| `intermediate_quality` | in_execution | 中间结果质量 | `partial_output.quality_score > 0.6` |
| `output_quality` | post_execution | 最终产出质量 | `output.quality_check.avg_quality_score >= 0.7` |
| `atom_count` | post_execution | 产出原子数量 | `output.atoms.length >= min_atoms` |
| `required_fields` | post_execution | 必需字段完整性 | `output.atoms[*].content.mathematical_expression exists` |
| `consistency_check` | integration | 跨任务一致性 | `math_atoms[*].content matches code_atoms[*].content` |
| `contradiction_check` | integration | 矛盾检测 | `contradiction_count == 0` |

---

## 3. 预执行质量门 (Pre-Execution Gates)

### 3.1 检查点设计

**输入完整性检查：**

```json
{
  "gate_id": "QG_PRE_SCHOLAR_INPUT_01",
  "gate_type": "pre_execution",
  "task_id": "TASK_SCHOLAR_DIFF_01",
  "checks": [
    {
      "check_id": "CHECK_PAPER_ACCESS",
      "check_type": "source_accessibility",
      "target": "paper_source",
      "condition": "paper_accessible == true",
      "error_code": "E001",
      "severity": "critical"
    },
    {
      "check_id": "CHECK_CODE_CLONED",
      "check_type": "source_accessibility", 
      "target": "code_repo",
      "condition": "code_repo_cloned == true",
      "error_code": "E001",
      "severity": "critical"
    },
    {
      "check_id": "CHECK_FOCUS_SECTIONS",
      "check_type": "content_availability",
      "target": "paper_focus_sections",
      "condition": "focus_sections.length > 0",
      "error_code": "E002",
      "severity": "warning"
    }
  ],
  "failure_action": "block",
  "action_config": {
    "notification_channel": "slack",
    "escalation_timeout": "PT10M"
  }
}
```

**依赖就绪检查：**

```json
{
  "gate_id": "QG_PRE_SCHOLAR_DEP_01",
  "gate_type": "pre_execution",
  "task_id": "TASK_SCHOLAR_DIFF_01",
  "checks": [
    {
      "check_id": "CHECK_VAULT_DEP",
      "check_type": "dependency_resolved",
      "target": "TASK_VAULT_CHECK_01",
      "condition": "dependency.status == 'success'",
      "error_code": "E006",
      "severity": "critical"
    },
    {
      "check_id": "CHECK_DEP_DATA",
      "check_type": "data_availability",
      "target": "existing_assets",
      "condition": "existing_assets.length > 0",
      "error_code": "E003",
      "severity": "warning"
    }
  ],
  "failure_action": "block",
  "action_config": {
    "wait_timeout": "PT30M",
    "polling_interval": "PT1M"
  }
}
```

### 3.2 上下文窗口检查

```json
{
  "gate_id": "QG_PRE_CONTEXT_01",
  "gate_type": "pre_execution",
  "task_id": "TASK_SCHOLAR_DIFF_01",
  "checks": [
    {
      "check_id": "CHECK_CONTEXT_SIZE",
      "check_type": "context_window",
      "target": "input_data",
      "condition": "estimated_input_tokens < available_tokens * 0.8",
      "error_code": "E004",
      "severity": "critical",
      "metrics": {
        "estimated_input_tokens": 45000,
        "available_tokens": 80000,
        "safety_margin": 0.2,
        "required_tokens": 36000
      }
    }
  ],
  "failure_action": "degrade",
  "action_config": {
    "degrade_strategy": "summarize_input",
    "summary_ratio": 0.5
  }
}
```

### 3.3 资源可用性检查

```json
{
  "gate_id": "QG_PRE_RESOURCE_01",
  "gate_type": "pre_execution",
  "task_id": "TASK_CODE_DIFF_01",
  "checks": [
    {
      "check_id": "CHECK_GPU_MEMORY",
      "check_type": "resource_availability",
      "target": "gpu_memory",
      "condition": "available_gpu_memory > estimated_memory_usage",
      "error_code": "E008",
      "severity": "critical",
      "metrics": {
        "available_gpu_memory": 8192,
        "estimated_memory_usage": 4096,
        "margin": 0.2
      }
    },
    {
      "check_id": "CHECK_API_QUOTA",
      "check_type": "resource_availability",
      "target": "llm_api_quota",
      "condition": "remaining_quota > estimated_quota_usage",
      "error_code": "E008",
      "severity": "critical"
    }
  ],
  "failure_action": "block",
  "action_config": {
    "retry_after": "PT1H",
    "alternative_resource": "backup_gpu"
  }
}
```

---

## 4. 执行中质量门 (In-Execution Gates)

### 4.1 进度监控检查

```json
{
  "gate_id": "QG_IN_PROGRESS_01",
  "gate_type": "in_execution",
  "task_id": "TASK_SCHOLAR_DIFF_01",
  "checks": [
    {
      "check_id": "CHECK_TIME_PROGRESS",
      "check_type": "progress_check",
      "target": "execution_time",
      "condition": "elapsed_time < timeout_seconds * 0.7",
      "error_code": "E004",
      "severity": "warning",
      "metrics": {
        "elapsed_time": 300,
        "timeout_seconds": 600,
        "progress_percentage": 50
      }
    },
    {
      "check_id": "CHECK_TOKEN_PROGRESS",
      "check_type": "progress_check",
      "target": "token_usage",
      "condition": "used_tokens < max_tokens * 0.8",
      "error_code": "E004",
      "severity": "warning"
    }
  ],
  "failure_action": "degrade",
  "action_config": {
    "degrade_strategy": "reduce_output_detail",
    "max_atoms": 3
  }
}
```

### 4.2 中间结果质量检查

```json
{
  "gate_id": "QG_IN_QUALITY_01",
  "gate_type": "in_execution",
  "task_id": "TASK_SCHOLAR_DIFF_01",
  "checks": [
    {
      "check_id": "CHECK_INTERMEDIATE_ATOMS",
      "check_type": "intermediate_quality",
      "target": "partial_atoms",
      "condition": "partial_atoms.avg_quality_score > 0.6",
      "error_code": "E005",
      "severity": "warning",
      "metrics": {
        "partial_atoms_count": 2,
        "avg_quality_score": 0.72,
        "min_quality_score": 0.65
      }
    },
    {
      "check_id": "CHECK_HALLUCINATION",
      "check_type": "hallucination_detection",
      "target": "partial_output",
      "condition": "hallucination_score < 0.3",
      "error_code": "E007",
      "severity": "critical"
    }
  ],
  "failure_action": "retry",
  "action_config": {
    "retry_count": 1,
    "adjust_temperature": 0.1
  }
}
```

### 4.3 流式输出检查

```json
{
  "gate_id": "QG_IN_STREAM_01",
  "gate_type": "in_execution",
  "task_id": "TASK_SCHOLAR_DIFF_01",
  "checks": [
    {
      "check_id": "CHECK_OUTPUT_FORMAT",
      "check_type": "format_compliance",
      "target": "streaming_output",
      "condition": "output_format == 'json'",
      "error_code": "E003",
      "severity": "critical"
    },
    {
      "check_id": "CHECK_REQUIRED_FIELDS",
      "check_type": "field_completeness",
      "target": "partial_atoms",
      "condition": "partial_atoms[*].has_all_required_fields == true",
      "error_code": "E003",
      "severity": "warning"
    }
  ],
  "failure_action": "retry",
  "action_config": {
    "retry_strategy": "partial_retry",
    "retry_from_checkpoint": true
  }
}
```

---

## 5. 后执行质量门 (Post-Execution Gates)

### 5.1 基于质量评估框架的检查

**引用 `common_quality_assessment.md` 的质量维度：**

```json
{
  "gate_id": "QG_POST_SCHOLAR_01",
  "gate_type": "post_execution",
  "task_id": "TASK_SCHOLAR_DIFF_01",
  "checks": [
    {
      "check_id": "CHECK_QUALITY_SCORE",
      "check_type": "output_quality",
      "target": "output.quality_check",
      "condition": "avg_quality_score >= 0.7",
      "error_code": "E005",
      "severity": "critical",
      "quality_dimensions": [
        {
          "dimension": "source_credibility",
          "min_score": 0.6,
          "weight": 0.25
        },
        {
          "dimension": "physical_intuition",
          "min_score": 0.7,
          "weight": 0.3
        },
        {
          "dimension": "traditional_mapping",
          "min_score": 0.5,
          "weight": 0.2
        },
        {
          "dimension": "boundary_audit",
          "min_score": 0.4,
          "weight": 0.15
        },
        {
          "dimension": "incremental_value",
          "min_score": 0.5,
          "weight": 0.1
        }
      ]
    },
    {
      "check_id": "CHECK_GRADE_DISTRIBUTION",
      "check_type": "grade_distribution",
      "target": "output.quality_check.grades",
      "condition": "grades.A + grades.B >= grades.total * 0.7",
      "error_code": "E005",
      "severity": "warning"
    }
  ],
  "failure_action": "retry",
  "action_config": {
    "max_retries": 2,
    "quality_threshold_adjustment": -0.1
  }
}
```

### 5.2 产出完整性检查

```json
{
  "gate_id": "QG_POST_COMPLETENESS_01",
  "gate_type": "post_execution",
  "task_id": "TASK_SCHOLAR_DIFF_01",
  "checks": [
    {
      "check_id": "CHECK_ATOM_COUNT",
      "check_type": "atom_count",
      "target": "output.atoms",
      "condition": "atoms.length >= 3 && atoms.length <= 10",
      "error_code": "E003",
      "severity": "critical",
      "expected_range": {
        "min": 3,
        "max": 10,
        "optimal": 5
      }
    },
    {
      "check_id": "CHECK_REQUIRED_FIELDS",
      "check_type": "required_fields",
      "target": "output.atoms[*].content",
      "condition": "all atoms have fields: ['mathematical_expression', 'physical_intuition', 'provenance.paper_location']",
      "error_code": "E003",
      "severity": "critical"
    },
    {
      "check_id": "CHECK_OUTPUT_SUMMARY",
      "check_type": "summary_completeness",
      "target": "output.summary",
      "condition": "summary.length > 50 && summary.length < 500",
      "error_code": "E003",
      "severity": "warning"
    }
  ],
  "failure_action": "retry",
  "action_config": {
    "retry_focus": "missing_fields",
    "supplement_instructions": "请补充缺失的字段"
  }
}
```

### 5.3 源锚定检查

```json
{
  "gate_id": "QG_POST_ANCHORING_01",
  "gate_type": "post_execution",
  "task_id": "TASK_SCHOLAR_DIFF_01",
  "checks": [
    {
      "check_id": "CHECK_PAPER_ANCHOR",
      "check_type": "source_anchoring",
      "target": "output.atoms[*].provenance",
      "condition": "all atoms have provenance.paper_location",
      "error_code": "E003",
      "severity": "critical"
    },
    {
      "check_id": "CHECK_CODE_ANCHOR",
      "check_type": "source_anchoring",
      "target": "output.atoms[*].provenance",
      "condition": "code_atoms have provenance.code_link",
      "error_code": "E003",
      "severity": "critical"
    },
    {
      "check_id": "CHECK_ANCHOR_ACCURACY",
      "check_type": "anchor_accuracy",
      "target": "provenance.paper_location",
      "condition": "paper_location matches pattern 'Page \\d+, Eq\\. \\d+'",
      "error_code": "E003",
      "severity": "warning"
    }
  ],
  "failure_action": "degrade",
  "action_config": {
    "degrade_action": "mark_as_partial",
    "quality_grade": "C"
  }
}
```

---

## 6. 集成质量门 (Integration Gates)

### 6.1 跨任务一致性检查

```json
{
  "gate_id": "QG_INTEG_CONSISTENCY_01",
  "gate_type": "integration",
  "stage_id": "S2",
  "checks": [
    {
      "check_id": "CHECK_MATH_CODE_ALIGN",
      "check_type": "consistency_check",
      "target": ["TASK_SCHOLAR_DIFF_01.output.atoms", "TASK_CODE_DIFF_01.output.atoms"],
      "condition": "math_atoms[*].content.mathematical_expression matches code_atoms[*].content.source_code",
      "error_code": "E009",
      "severity": "critical",
      "matching_strategy": "semantic_similarity",
      "similarity_threshold": 0.8
    },
    {
      "check_id": "CHECK_TERMINOLOGY_CONSISTENCY",
      "check_type": "consistency_check",
      "target": "all_tasks.output.atoms[*].content",
      "condition": "key_terms are used consistently across atoms",
      "error_code": "E009",
      "severity": "warning"
    }
  ],
  "failure_action": "block",
  "action_config": {
    "resolution_strategy": "reconcile_differences",
    "arbitrator": "Strategic_Critic"
  }
}
```

### 6.2 逻辑自洽性检查

```json
{
  "gate_id": "QG_INTEG_LOGICAL_01",
  "gate_type": "integration",
  "stage_id": "S2",
  "checks": [
    {
      "check_id": "CHECK_CONTRADICTION",
      "check_type": "contradiction_check",
      "target": "all_tasks.output.atoms[*].content",
      "condition": "contradiction_count == 0",
      "error_code": "E009",
      "severity": "critical",
      "contradiction_types": ["mathematical_contradiction", "physical_contradiction", "empirical_contradiction"]
    },
    {
      "check_id": "CHECK_COMPLETENESS",
      "check_type": "completeness_check",
      "target": "expected_outputs.atoms",
      "condition": "all expected atom types are produced",
      "error_code": "E009",
      "severity": "warning"
    }
  ],
  "failure_action": "degrade",
  "action_config": {
    "degrade_action": "flag_as_partial",
    "generate_reconciliation_report": true
  }
}
```

### 6.3 整体质量评估

```json
{
  "gate_id": "QG_INTEG_QUALITY_01",
  "gate_type": "integration",
  "stage_id": "S2",
  "checks": [
    {
      "check_id": "CHECK_OVERALL_QUALITY",
      "check_type": "overall_quality",
      "target": "all_tasks.output.quality_check",
      "condition": "weighted_avg_quality_score >= 0.75",
      "error_code": "E005",
      "severity": "critical",
      "weighting": {
        "TASK_SCHOLAR_DIFF_01": 0.4,
        "TASK_CODE_DIFF_01": 0.3,
        "TASK_VALIDATOR_DIFF_01": 0.3
      }
    },
    {
      "check_id": "CHECK_RISK_ASSESSMENT",
      "check_type": "risk_assessment",
      "target": "TASK_VALIDATOR_DIFF_01.output.atoms[*].content.risk_assessment",
      "condition": "max_severity <= 'medium'",
      "error_code": "E009",
      "severity": "warning"
    }
  ],
  "failure_action": "abort",
  "action_config": {
    "abort_reason": "overall_quality_below_threshold",
    "generate_post_mortem": true
  }
}
```

---

## 7. 质量门执行引擎

### 7.1 执行流程

```
┌─────────────────────────────────────────────────────────────┐
│                      质量门引擎                             │
├─────────────────────────────────────────────────────────────┤
│ 1. 接收任务状态变更事件                                    │
│ 2. 查找关联的质量门配置                                    │
│ 3. 执行所有检查项                                          │
│ 4. 聚合检查结果                                           │
│ 5. 根据失败动作执行相应操作                                │
│ 6. 记录质量门执行日志                                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        v
┌─────────────────────────────────────────────────────────────┐
│                    检查结果聚合                             │
│  • 所有检查通过 → 触发success_action                       │
│  • 有warning检查失败 → 触发warning_action                  │
│  • 有critical检查失败 → 触发failure_action                 │
│  • 混合结果 → 按最高严重级别处理                           │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        v
┌─────────────────────────────────────────────────────────────┐
│                    动作执行器                               │
│  • block: 设置任务状态为blocked，等待人工干预              │
│  • retry: 触发任务重试，可配置重试次数和退避策略           │
│  • degrade: 降低质量标准继续执行                          │
│  • skip: 跳过当前任务，标记为skipped                      │
│  • abort: 终止整个阶段或计划                              │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 质量门配置管理

```json
{
  "quality_gate_config": {
    "default_gates": {
      "pre_execution": ["source_accessibility", "dependency_resolved", "context_window"],
      "post_execution": ["output_quality", "atom_count", "required_fields"],
      "integration": ["consistency_check", "contradiction_check"]
    },
    "agent_specific_gates": {
      "Scholar_Internalizer": {
        "additional_checks": ["source_anchoring", "physical_intuition_quality"]
      },
      "Code_Architect": {
        "additional_checks": ["code_anchoring", "performance_analysis"]
      }
    },
    "mode_specific_gates": {
      "Deep_Internalization": {
        "strictness_level": "high",
        "quality_threshold": 0.7
      },
      "Quick_Assessment": {
        "strictness_level": "medium",
        "quality_threshold": 0.6
      }
    }
  }
}
```

### 7.3 质量门执行日志

```json
{
  "gate_execution_log": {
    "execution_id": "QG_EXEC_20260203_143052_001",
    "gate_id": "QG_POST_SCHOLAR_01",
    "task_id": "TASK_SCHOLAR_DIFF_01",
    "start_time": "2026-02-03T14:30:52Z",
    "end_time": "2026-02-03T14:31:15Z",
    "check_results": [
      {
        "check_id": "CHECK_QUALITY_SCORE",
        "status": "passed",
        "actual_value": 0.82,
        "threshold": 0.7,
        "severity": "critical",
        "details": "质量分数达到要求"
      },
      {
        "check_id": "CHECK_ATOM_COUNT",
        "status": "passed",
        "actual_value": 4,
        "expected_range": {"min": 3, "max": 10},
        "severity": "critical",
        "details": "原子数量在期望范围内"
      }
    ],
    "overall_status": "passed",
    "action_taken": "none",
    "metadata": {
      "executed_by": "Quality_Gate_Engine",
      "version": "1.0"
    }
  }
}
```

---

## 8. 完整示例

### 8.1 Diffusion Planner任务的质量门配置

```json
{
  "quality_gates": {
    "pre_execution": [
      {
        "gate_id": "QG_PRE_SCHOLAR_DIFF_01",
        "gate_type": "pre_execution",
        "task_id": "TASK_SCHOLAR_DIFF_01",
        "checks": [
          {
            "check_id": "CHECK_PAPER_ACCESS",
            "check_type": "source_accessibility",
            "target": "paper_source",
            "condition": "paper_accessible == true",
            "error_code": "E001",
            "severity": "critical"
          },
          {
            "check_id": "CHECK_VAULT_DEP",
            "check_type": "dependency_resolved",
            "target": "TASK_VAULT_CHECK_01",
            "condition": "dependency.status == 'success'",
            "error_code": "E006",
            "severity": "critical"
          }
        ],
        "failure_action": "block",
        "action_config": {
          "notification_channel": "slack",
          "escalation_timeout": "PT15M"
        }
      }
    ],
    "post_execution": [
      {
        "gate_id": "QG_POST_SCHOLAR_DIFF_01",
        "gate_type": "post_execution",
        "task_id": "TASK_SCHOLAR_DIFF_01",
        "checks": [
          {
            "check_id": "CHECK_QUALITY_SCORE",
            "check_type": "output_quality",
            "target": "output.quality_check",
            "condition": "avg_quality_score >= 0.7",
            "error_code": "E005",
            "severity": "critical"
          },
          {
            "check_id": "CHECK_ATOM_COUNT",
            "check_type": "atom_count",
            "target": "output.atoms",
            "condition": "atoms.length >= 3",
            "error_code": "E003",
            "severity": "critical"
          },
          {
            "check_id": "CHECK_SOURCE_ANCHOR",
            "check_type": "source_anchoring",
            "target": "output.atoms[*].provenance",
            "condition": "all atoms have provenance.paper_location",
            "error_code": "E003",
            "severity": "critical"
          }
        ],
        "failure_action": "retry",
        "action_config": {
          "max_retries": 2,
          "backoff_seconds": 30
        }
      }
    ],
    "integration": [
      {
        "gate_id": "QG_INTEG_DIFF_01",
        "gate_type": "integration",
        "stage_id": "S2",
        "checks": [
          {
            "check_id": "CHECK_CONSISTENCY",
            "check_type": "consistency_check",
            "target": ["TASK_SCHOLAR_DIFF_01.output", "TASK_CODE_DIFF_01.output"],
            "condition": "math_and_code_aligned == true",
            "error_code": "E009",
            "severity": "critical"
          },
          {
            "check_id": "CHECK_OVERALL_QUALITY",
            "check_type": "overall_quality",
            "target": "all_tasks.output.quality_check",
            "condition": "weighted_avg_quality_score >= 0.75",
            "error_code": "E005",
            "severity": "critical"
          }
        ],
        "failure_action": "abort",
        "action_config": {
          "generate_post_mortem": true,
          "notify_stakeholders": ["orchestrator", "user"]
        }
      }
    ]
  }
}
```

### 8.2 质量门执行场景

**场景1：预执行门失败**

```
事件: 任务 TASK_SCHOLAR_DIFF_01 准备执行
触发: QG_PRE_SCHOLAR_DIFF_01
检查: CHECK_PAPER_ACCESS (paper_accessible == false)
结果: 失败
动作: block
影响: 任务状态设为 blocked，发送Slack通知
处理: 人工检查论文可访问性，修复后手动触发重试
```

**场景2：后执行门部分失败**

```
事件: 任务 TASK_SCHOLAR_DIFF_01 完成执行
触发: QG_POST_SCHOLAR_DIFF_01
检查:
  - CHECK_QUALITY_SCORE: 通过 (0.72 >= 0.7)
  - CHECK_ATOM_COUNT: 通过 (4 >= 3)
  - CHECK_SOURCE_ANCHOR: 失败 (2/4个原子缺少paper_location)
结果: 部分失败
动作: retry
影响: 触发任务重试，重试时补充缺失的paper_location信息
```

**场景3：集成门成功**

```
事件: 阶段 S2 所有任务完成
触发: QG_INTEG_DIFF_01
检查:
  - CHECK_CONSISTENCY: 通过 (math_and_code_aligned == true)
  - CHECK_OVERALL_QUALITY: 通过 (0.78 >= 0.75)
结果: 成功
动作: 无
影响: 允许进入下一阶段，记录质量门通过日志
```

---

## 9. 版本历史

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| 1.0 | 2026-02-03 | 初始版本 |

## 10. 相关文档

- `01_task_package_protocol.md` - 任务包协议
- `02_agent_invocation_protocol.md` - Agent调用协议
- `03_data_transfer_protocol.md` - 数据传递协议
- `05_error_handling_protocol.md` - 错误处理协议
- `../common/common_quality_assessment.md` - 质量评估框架
- `../common/common_output.md` - 原子输出格式标准