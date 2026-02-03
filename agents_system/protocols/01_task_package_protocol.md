# 任务包协议 (Task Package Protocol)

## 1. 概述

### 1.1 协议目的

本协议定义了Orchestrator（中央规划器）输出的标准化任务包格式，确保：

- **结构化规划**：将复杂的端到端学习需求拆解为可执行的任务图
- **自动化调度**：调度器（Dispatcher）能够解析并执行任务包中的任务
- **质量可控**：内置质量门控检查点，确保每个任务达到预期质量标准
- **可追溯性**：完整的执行链路追踪，从原始请求到最终产出

### 1.2 适用范围

本协议适用于以下流程：

| 阶段 | 输入 | 输出 | 执行者 |
|------|------|------|--------|
| 需求分析 | 用户原始请求 + 知识库快照 | 任务包 | Orchestrator |
| 任务调度 | 任务包 | Agent调用请求序列 | Dispatcher |
| 执行监控 | Agent响应 | 质量评估结果 | Quality Gate |

### 1.3 协议版本

- **版本**: 1.0
- **创建时间**: 2026-02-03
- **兼容性**: 与 `02_agent_invocation_protocol.md` v1.0 兼容

---

## 2. 任务包完整结构

### 2.1 顶层结构

```json
{
  "plan_id": "AVP_E2E_20260203_001",
  "created_at": "2026-02-03T10:30:00Z",
  "meta": {
    "focus": "Diffusion Planner在AVP场景下的去噪采样机制",
    "mode": "Deep_Internalization",
    "paper_source": "ArXiv:2406.01234",
    "code_source": "github.com/example/diffusion-planner"
  },
  "context": {
    "atlas_snapshot": {
      "total_atoms": 1256,
      "domain_coverage": ["planning", "control", "perception"],
      "recent_contributions": ["MATH_VAD_ATTENTION_01", "CODE_DETR_QUERY_01"]
    },
    "pre_check_result": {
      "paper_accessible": true,
      "code_repo_cloned": true,
      "domain_overlap": 0.78,
      "risk_assessment": "low"
    }
  },
  "task_graph": {
    "stages": [
      {
        "stage_id": "S1",
        "stage_name": "parallel_analysis",
        "execution_mode": "parallel",
        "tasks": [...]
      },
      {
        "stage_id": "S2", 
        "stage_name": "integration_validation",
        "execution_mode": "sequential",
        "depends_on": ["S1"],
        "tasks": [...]
      }
    ],
    "dependencies": [...]
  },
  "quality_gates": {
    "pre_execution": [...],
    "post_execution": [...],
    "final_integration": [...]
  },
  "error_handling": {
    "retry_policy": {...},
    "fallback_strategies": {...},
    "escalation_rules": {...}
  },
  "expected_outputs": {
    "atoms": [...],
    "assessments": [...],
    "decisions": [...]
  }
}
```

### 2.2 字段概览

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `plan_id` | string | 是 | 计划唯一标识符 |
| `created_at` | string | 是 | ISO8601时间戳 |
| `meta` | object | 是 | 元数据信息 |
| `context` | object | 是 | 执行上下文 |
| `task_graph` | object | 是 | 任务图定义 |
| `quality_gates` | object | 是 | 质量门控配置 |
| `error_handling` | object | 是 | 错误处理配置 |
| `expected_outputs` | object | 是 | 期望产出描述 |

---

## 3. 详细字段说明

### 3.1 plan_id 生成规则

```
AVP_E2E_{YYYYMMDD}_{SEQ}_{MODE_SHORT}

示例: AVP_E2E_20260203_001_DI

其中:
- YYYYMMDD: 日期
- SEQ: 3位序号，从001开始
- MODE_SHORT: 执行模式简称
  - DI: Deep_Internalization
  - QA: Quick_Assessment
  - ER: Engineering_Reproduction
```

### 3.2 meta 对象

```json
{
  "focus": "string - 技术聚焦点，人类可读描述",
  "mode": "string - 执行模式，见模式定义",
  "paper_source": "string - 论文来源标识",
  "code_source": "string - 代码来源标识",
  "priority": "integer - 优先级，1-5，1最高",
  "deadline": "string - 可选，ISO8601截止时间",
  "tags": ["array", "of", "tags"]
}
```

**执行模式定义：**

| 模式 | 简称 | 说明 |
|------|------|------|
| Deep_Internalization | DI | 深度内化，全面分析论文和代码 |
| Quick_Assessment | QA | 快速评估，聚焦核心创新点 |
| Engineering_Reproduction | ER | 工程复现，侧重代码实现评估 |
| Validation_Focused | VF | 验证导向，重点评估算法可行性 |

### 3.3 context 对象

```json
{
  "atlas_snapshot": {
    "total_atoms": "integer - 知识库总原子数",
    "domain_coverage": ["array", "of", "domains"],
    "recent_contributions": ["array", "of", "atom_ids"],
    "quality_distribution": {
      "A": "integer",
      "B": "integer", 
      "C": "integer",
      "D": "integer"
    }
  },
  "pre_check_result": {
    "paper_accessible": "boolean - 论文是否可访问",
    "code_repo_cloned": "boolean - 代码仓库是否已克隆",
    "domain_overlap": "float - 与现有知识库领域重叠度",
    "risk_assessment": "string - 风险评估等级",
    "estimated_complexity": "integer - 预估复杂度分数"
  }
}
```

### 3.4 task_graph 对象

#### 3.4.1 stages 数组

```json
{
  "stage_id": "string - 阶段唯一标识",
  "stage_name": "string - 阶段名称",
  "execution_mode": "parallel | sequential",
  "depends_on": ["array", "of", "stage_ids"],
  "tasks": [
    {
      "task_id": "string - 任务唯一标识",
      "agent": "string - 负责的Agent",
      "input_spec": {...},
      "output_spec": {...},
      "timeout_seconds": "integer",
      "priority": "integer"
    }
  ]
}
```

**阶段类型定义：**

| 阶段名称 | 执行模式 | 典型任务 |
|----------|----------|----------|
| parallel_analysis | parallel | Scholar, Code并行分析 |
| integration_validation | sequential | Validator, Vault整合验证 |
| strategic_assessment | sequential | Critic最终评估 |
| knowledge_integration | parallel | Vault原子入库 |

#### 3.4.2 tasks 详细定义

```json
{
  "task_id": "TASK_SCHOLAR_DIFF_01",
  "agent": "Scholar_Internalizer",
  "input_spec": {
    "sources": [
      {
        "type": "paper",
        "paper_id": "ArXiv:2406.01234",
        "focus_sections": ["Section 3.2", "Section 4.1"]
      },
      {
        "type": "code",
        "repo_url": "https://github.com/example/diffusion-planner",
        "focus_files": ["models/diffusion.py"]
      }
    ],
    "dependencies": [
      {
        "task_id": "TASK_VAULT_CHECK_01",
        "output_ref": "existing_assets"
      }
    ]
  },
  "output_spec": {
    "atom_type": "Math_Atom",
    "required_fields": ["mathematical_expression", "physical_intuition"],
    "quality_threshold": "B",
    "expected_count": "3-5"
  },
  "timeout_seconds": 600,
  "priority": 1,
  "retry_config": {
    "max_attempts": 3,
    "backoff_factor": 2
  }
}
```

**Agent映射表：**

| Agent | 主要产出 | 典型任务ID前缀 |
|-------|----------|----------------|
| Scholar_Internalizer | Math_Atom | TASK_SCHOLAR |
| Code_Architect | Code_Atom | TASK_CODE |
| Scenario_Validator | Scenario_Atom | TASK_VALIDATOR |
| Knowledge_Vault | 资产索引更新 | TASK_VAULT |
| Strategic_Critic | Strategic_Decision | TASK_CRITIC |

#### 3.4.3 dependencies 定义

```json
{
  "dependencies": [
    {
      "from_task": "TASK_SCHOLAR_DIFF_01",
      "to_task": "TASK_VALIDATOR_DIFF_01",
      "data_flow": {
        "ref_type": "task_output",
        "path": "$.atoms",
        "transform": "summary"
      },
      "condition": "success | partial",
      "timeout": "PT5M"
    }
  ]
}
```

---

## 4. 质量门控 (Quality Gates)

### 4.1 质量门类型

| 门类型 | 触发时机 | 检查内容 | 通过标准 |
|--------|----------|----------|----------|
| pre_execution | 任务执行前 | 输入数据完整性 | 所有必需输入就绪 |
| post_execution | 任务执行后 | 输出质量评估 | 达到quality_threshold |
| final_integration | 所有任务完成后 | 整体一致性检查 | 无冲突，逻辑自洽 |

### 4.2 质量门配置

```json
{
  "quality_gates": {
    "pre_execution": [
      {
        "gate_id": "QG_PRE_SCHOLAR_01",
        "task_id": "TASK_SCHOLAR_DIFF_01",
        "checks": [
          {
            "check_type": "source_accessibility",
            "target": "paper",
            "condition": "accessible == true",
            "error_code": "E001"
          },
          {
            "check_type": "dependency_resolved",
            "target": "TASK_VAULT_CHECK_01",
            "condition": "status == success",
            "error_code": "E006"
          }
        ],
        "failure_action": "block"
      }
    ],
    "post_execution": [
      {
        "gate_id": "QG_POST_SCHOLAR_01",
        "task_id": "TASK_SCHOLAR_DIFF_01",
        "checks": [
          {
            "check_type": "output_quality",
            "metric": "avg_quality_score",
            "threshold": 0.7,
            "error_code": "E005"
          },
          {
            "check_type": "atom_count",
            "min": 1,
            "max": 10,
            "error_code": "E003"
          }
        ],
        "failure_action": "retry"
      }
    ]
  }
}
```

**失败处理动作：**

| 动作 | 说明 |
|------|------|
| block | 阻塞执行，需要人工干预 |
| retry | 触发重试机制 |
| degrade | 降低标准继续执行 |
| skip | 跳过当前任务 |
| abort | 终止整个计划 |

---

## 5. 错误处理配置

### 5.1 重试策略

```json
{
  "retry_policy": {
    "max_attempts": 3,
    "backoff_strategy": "exponential",
    "base_delay_seconds": 5,
    "max_delay_seconds": 300,
    "retryable_errors": ["E001", "E002", "E003", "E004"]
  }
}
```

### 5.2 降级策略

```json
{
  "fallback_strategies": {
    "on_quality_failure": {
      "action": "degrade_threshold",
      "new_threshold": "C",
      "notify": "orchestrator"
    },
    "on_source_unavailable": {
      "action": "use_cached",
      "max_age_days": 30,
      "notify": "user"
    },
    "on_agent_failure": {
      "action": "alternative_agent",
      "mapping": {
        "Scholar_Internalizer": "Code_Architect",
        "Code_Architect": "Strategic_Critic"
      }
    }
  }
}
```

### 5.3 升级规则

```json
{
  "escalation_rules": [
    {
      "condition": "consecutive_failures >= 3",
      "action": "notify_human",
      "level": "urgent",
      "channel": "slack"
    },
    {
      "condition": "critical_path_blocked",
      "action": "rollback",
      "level": "critical",
      "channel": "email"
    }
  ]
}
```

---

## 6. 期望产出描述

### 6.1 原子产出

```json
{
  "expected_outputs": {
    "atoms": [
      {
        "type": "Math_Atom",
        "count": "3-5",
        "coverage": ["denoising_formula", "conditional_guidance"],
        "quality_target": "B+"
      },
      {
        "type": "Code_Atom",
        "count": "2-4",
        "coverage": ["sampler_implementation", "performance_optimization"],
        "quality_target": "B"
      }
    ],
    "assessments": [
      {
        "type": "feasibility_assessment",
        "scope": "Orin_X_deployment",
        "required_fields": ["latency_estimate", "memory_footprint"]
      }
    ],
    "decisions": [
      {
        "type": "adoption_recommendation",
        "format": "structured_decision",
        "confidence_threshold": 0.8
      }
    ]
  }
}
```

---

## 7. 完整示例

### 7.1 Diffusion Planner分析任务包

```json
{
  "plan_id": "AVP_E2E_20260203_001_DI",
  "created_at": "2026-02-03T10:30:00Z",
  "meta": {
    "focus": "Diffusion Planner在AVP地库场景下的去噪采样机制与工程可行性",
    "mode": "Deep_Internalization",
    "paper_source": "ArXiv:2406.01234",
    "code_source": "github.com/example/diffusion-planner",
    "priority": 1,
    "tags": ["diffusion", "planning", "avp", "denoising"]
  },
  "context": {
    "atlas_snapshot": {
      "total_atoms": 1256,
      "domain_coverage": ["planning", "control", "perception"],
      "recent_contributions": ["MATH_VAD_ATTENTION_01", "CODE_DETR_QUERY_01"],
      "quality_distribution": {"A": 312, "B": 589, "C": 289, "D": 66}
    },
    "pre_check_result": {
      "paper_accessible": true,
      "code_repo_cloned": true,
      "domain_overlap": 0.78,
      "risk_assessment": "low",
      "estimated_complexity": 7
    }
  },
  "task_graph": {
    "stages": [
      {
        "stage_id": "S1",
        "stage_name": "parallel_analysis",
        "execution_mode": "parallel",
        "tasks": [
          {
            "task_id": "TASK_SCHOLAR_DIFF_01",
            "agent": "Scholar_Internalizer",
            "input_spec": {
              "sources": [
                {
                  "type": "paper",
                  "paper_id": "ArXiv:2406.01234",
                  "focus_sections": ["Section 3.2 Denoising Process", "Section 4.1 Conditional Guidance"]
                }
              ],
              "dependencies": [
                {
                  "task_id": "TASK_VAULT_CHECK_01",
                  "output_ref": "existing_assets"
                }
              ]
            },
            "output_spec": {
              "atom_type": "Math_Atom",
              "required_fields": ["mathematical_expression", "physical_intuition", "avp_relevance"],
              "quality_threshold": "B",
              "expected_count": "3-5"
            },
            "timeout_seconds": 600,
            "priority": 1
          },
          {
            "task_id": "TASK_CODE_DIFF_01",
            "agent": "Code_Architect",
            "input_spec": {
              "sources": [
                {
                  "type": "code",
                  "repo_url": "https://github.com/example/diffusion-planner",
                  "focus_files": ["models/diffusion.py", "models/sampler.py"]
                }
              ]
            },
            "output_spec": {
              "atom_type": "Code_Atom",
              "required_fields": ["source_code", "complexity_analysis", "deployment_suggestion"],
              "quality_threshold": "B",
              "expected_count": "2-4"
            },
            "timeout_seconds": 450,
            "priority": 1
          },
          {
            "task_id": "TASK_VAULT_CHECK_01",
            "agent": "Knowledge_Vault",
            "input_spec": {
              "sources": [
                {
                  "type": "db_query",
                  "table": "knowledge_atoms",
                  "conditions": [
                    {"field": "tags", "operator": "contains", "value": "diffusion"},
                    {"field": "tags", "operator": "contains", "value": "planning"}
                  ],
                  "limit": 50
                }
              ]
            },
            "output_spec": {
              "atom_type": "Asset_Index",
              "required_fields": ["existing_assets", "domain_gaps"],
              "quality_threshold": "C",
              "expected_count": "1"
            },
            "timeout_seconds": 120,
            "priority": 2
          }
        ]
      },
      {
        "stage_id": "S2",
        "stage_name": "integration_validation",
        "execution_mode": "sequential",
        "depends_on": ["S1"],
        "tasks": [
          {
            "task_id": "TASK_VALIDATOR_DIFF_01",
            "agent": "Scenario_Validator",
            "input_spec": {
              "sources": [],
              "dependencies": [
                {
                  "task_id": "TASK_SCHOLAR_DIFF_01",
                  "output_ref": "output.atoms"
                },
                {
                  "task_id": "TASK_CODE_DIFF_01",
                  "output_ref": "output.atoms"
                }
              ]
            },
            "output_spec": {
              "atom_type": "Scenario_Atom",
              "required_fields": ["challenge_scenario", "failure_mechanism", "risk_assessment"],
              "quality_threshold": "B",
              "expected_count": "3-5"
            },
            "timeout_seconds": 300,
            "priority": 1
          }
        ]
      }
    ],
    "dependencies": [
      {
        "from_task": "TASK_SCHOLAR_DIFF_01",
        "to_task": "TASK_VALIDATOR_DIFF_01",
        "data_flow": {
          "ref_type": "task_output",
          "path": "$.atoms",
          "transform": "summary"
        },
        "condition": "success",
        "timeout": "PT5M"
      }
    ]
  },
  "quality_gates": {
    "pre_execution": [
      {
        "gate_id": "QG_PRE_SCHOLAR_01",
        "task_id": "TASK_SCHOLAR_DIFF_01",
        "checks": [
          {
            "check_type": "source_accessibility",
            "target": "paper",
            "condition": "accessible == true"
          }
        ],
        "failure_action": "block"
      }
    ],
    "post_execution": [
      {
        "gate_id": "QG_POST_SCHOLAR_01",
        "task_id": "TASK_SCHOLAR_DIFF_01",
        "checks": [
          {
            "check_type": "output_quality",
            "metric": "avg_quality_score",
            "threshold": 0.7
          }
        ],
        "failure_action": "retry"
      }
    ]
  },
  "error_handling": {
    "retry_policy": {
      "max_attempts": 3,
      "backoff_strategy": "exponential",
      "base_delay_seconds": 5
    },
    "fallback_strategies": {
      "on_quality_failure": {
        "action": "degrade_threshold",
        "new_threshold": "C"
      }
    }
  },
  "expected_outputs": {
    "atoms": [
      {
        "type": "Math_Atom",
        "count": "3-5",
        "coverage": ["denoising_formula", "conditional_guidance", "noise_schedule"],
        "quality_target": "B+"
      },
      {
        "type": "Code_Atom", 
        "count": "2-4",
        "coverage": ["sampler_implementation", "performance_optimization"],
        "quality_target": "B"
      },
      {
        "type": "Scenario_Atom",
        "count": "3-5",
        "coverage": ["avp_scenario", "failure_mode", "mitigation"],
        "quality_target": "B"
      }
    ],
    "assessments": [
      {
        "type": "feasibility_assessment",
        "scope": "Orin_X_deployment",
        "required_fields": ["latency_estimate", "memory_footprint", "power_consumption"]
      }
    ]
  }
}
```

### 7.2 任务包执行流程

```
┌─────────────────────────────────────────────────────────────┐
│                      Orchestrator                           │
│  输入: 用户请求 + 知识库快照                               │
│  输出: 结构化任务包 (本协议定义)                           │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        v
┌─────────────────────────────────────────────────────────────┐
│                      Dispatcher                             │
│  1. 解析任务包，构建执行计划                               │
│  2. 检查质量门控条件                                       │
│  3. 按依赖关系调度任务                                     │
│  4. 监控执行状态，处理错误                                 │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        v
┌─────────────────────────────────────────────────────────────┐
│                      质量门控                               │
│  • pre_execution: 检查输入就绪                             │
│  • post_execution: 检查产出质量                            │
│  • final_integration: 检查整体一致性                       │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        v
┌─────────────────────────────────────────────────────────────┐
│                      执行引擎                               │
│  按02_agent_invocation_protocol调用各Agent                 │
│  按03_data_transfer_protocol传递数据                       │
│  按04_quality_gate_protocol评估质量                        │
│  按05_error_handling_protocol处理错误                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. 版本历史

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| 1.0 | 2026-02-03 | 初始版本 |

## 9. 相关文档

- `02_agent_invocation_protocol.md` - Agent调用协议
- `03_data_transfer_protocol.md` - 数据传递协议  
- `04_quality_gate_protocol.md` - 质量门协议
- `05_error_handling_protocol.md` - 错误处理协议
- `../common/common_output.md` - 原子输出格式标准
- `schemas/task_package.schema.json` - 任务包JSON Schema验证