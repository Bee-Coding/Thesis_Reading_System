# Agent调用协议 (Agent Invocation Protocol)

## 1. 概述

### 1.1 协议目的

本协议定义了自动化调度器（Dispatcher）与单个Agent之间的标准化通信格式，确保：

- **一致性**：所有Agent调用遵循统一的请求/响应格式
- **可追溯性**：每次调用都有唯一标识，支持全链路追踪
- **可扩展性**：支持不同Agent的特定参数配置
- **质量可控**：内置质量检查机制，确保产出符合标准

### 1.2 适用范围

本协议适用于以下Agent的调用：

| Agent | 职责 | 主要产出 |
|-------|------|----------|
| Scholar_Internalizer | 数学公式物理化内化 | Math_Atom |
| Code_Architect | 代码分析与工程落地评估 | Code_Atom |
| Scenario_Validator | AVP场景挑战与风险评估 | Scenario_Atom |
| Knowledge_Vault | 知识资产管理与检索 | 资产索引更新 |
| Strategic_Critic | 量产可行性审计与决策 | Strategic_Decision |

### 1.3 协议版本

- **版本**: 1.0
- **创建时间**: 2026-02-03
- **兼容性**: 与 `01_task_package_protocol.md` v1.0 兼容

---

## 2. 调用请求格式 (AgentRequest)

### 2.1 完整结构

```json
{
  "invocation_id": "INV_20260203_143052_SCHOLAR_001",
  "agent": "Scholar_Internalizer",
  "task_id": "TASK_SCHOLAR_DIFF_01",
  "plan_id": "AVP_E2E_20260203_001",
  "system_prompt": "完整的Agent System Prompt内容...",
  "user_prompt": {
    "instruction": "执行指令描述",
    "context": {
      "paper_info": {},
      "code_info": {},
      "dependencies": [],
      "existing_assets": []
    },
    "focus": "技术聚焦点",
    "output_requirements": {
      "atom_type": "Math_Atom",
      "required_fields": [],
      "quality_threshold": "B"
    }
  },
  "constraints": {
    "timeout_seconds": 300,
    "max_tokens": 8000,
    "temperature": 0.3
  }
}
```

### 2.2 字段详细说明

#### 2.2.1 顶层字段

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `invocation_id` | string | 是 | 唯一调用标识符，格式：`INV_{YYYYMMDD}_{HHMMSS}_{AGENT}_{SEQ}` |
| `agent` | string | 是 | Agent名称，必须是预定义的Agent之一 |
| `task_id` | string | 是 | 关联的任务ID，来自任务包 |
| `plan_id` | string | 是 | 关联的计划ID，来自Orchestrator |
| `system_prompt` | string | 是 | Agent的完整System Prompt |
| `user_prompt` | object | 是 | 用户提示词对象 |
| `constraints` | object | 是 | 执行约束参数 |

#### 2.2.2 user_prompt 对象

```json
{
  "instruction": "string - 具体执行指令",
  "context": {
    "paper_info": {
      "paper_id": "ArXiv:24xx.xxxx",
      "title": "论文标题",
      "pdf_path": "/raw_papers/paper.pdf",
      "focus_sections": ["Section 3", "Section 4.2"]
    },
    "code_info": {
      "repo_url": "https://github.com/user/repo",
      "branch": "main",
      "focus_files": ["models/planner.py", "utils/diffusion.py"]
    },
    "dependencies": [
      {
        "task_id": "TASK_VAULT_CHECK_01",
        "output_ref": "existing_assets"
      }
    ],
    "existing_assets": ["MATH_VAD_ATTENTION_01", "CODE_VAD_SAMPLER_01"]
  },
  "focus": "Diffusion Planner的去噪采样机制",
  "output_requirements": {
    "atom_type": "Math_Atom",
    "required_fields": [
      "mathematical_expression",
      "physical_intuition",
      "traditional_mapping",
      "provenance.paper_location"
    ],
    "quality_threshold": "B",
    "max_atoms": 5
  }
}
```

#### 2.2.3 constraints 对象

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `timeout_seconds` | integer | 300 | 执行超时时间（秒） |
| `max_tokens` | integer | 8000 | LLM最大输出token数 |
| `temperature` | float | 0.3 | LLM采样温度 |

---

## 3. 调用响应格式 (AgentResponse)

### 3.1 完整结构

```json
{
  "invocation_id": "INV_20260203_143052_SCHOLAR_001",
  "status": "success",
  "agent": "Scholar_Internalizer",
  "task_id": "TASK_SCHOLAR_DIFF_01",
  "execution_time_ms": 45230,
  "output": {
    "atoms": [
      {
        "asset_id": "MATH_PLAN_DIFF_01",
        "category": "Math_Ops",
        "data_status": "Verified_Source_Anchored",
        "metadata": {},
        "content": {},
        "provenance": {},
        "delta_audit": {}
      }
    ],
    "summary": "成功提取3个数学原子，覆盖Diffusion Planner的核心去噪公式。"
  },
  "quality_check": {
    "atoms_generated": 3,
    "avg_quality_score": 0.85,
    "grades": {
      "A": 1,
      "B": 2,
      "C": 0,
      "D": 0
    },
    "issues": []
  },
  "errors": []
}
```

### 3.2 字段详细说明

#### 3.2.1 顶层字段

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `invocation_id` | string | 是 | 与请求对应的调用ID |
| `status` | string | 是 | 执行状态，见状态说明 |
| `agent` | string | 是 | 执行的Agent名称 |
| `task_id` | string | 是 | 关联的任务ID |
| `execution_time_ms` | integer | 是 | 实际执行时间（毫秒） |
| `output` | object | 是 | 产出内容 |
| `quality_check` | object | 是 | 质量检查结果 |
| `errors` | array | 是 | 错误信息数组 |

#### 3.2.2 output 对象

```json
{
  "atoms": [
    // 标准原子数组，格式遵循 common_output.md
  ],
  "summary": "执行摘要，人类可读的简短描述"
}
```

#### 3.2.3 quality_check 对象

```json
{
  "atoms_generated": 3,
  "avg_quality_score": 0.85,
  "grades": {
    "A": 1,
    "B": 2,
    "C": 0,
    "D": 0
  },
  "issues": [
    {
      "atom_id": "MATH_PLAN_DIFF_02",
      "issue_type": "missing_field",
      "field": "provenance.paper_location",
      "severity": "warning",
      "message": "论文位置信息不完整"
    }
  ]
}
```

#### 3.2.4 errors 数组

```json
[
  {
    "error_code": "E001",
    "error_type": "source_not_found",
    "message": "无法访问指定的GitHub仓库",
    "recoverable": true,
    "suggested_action": "检查仓库URL或网络连接"
  }
]
```

---

## 4. Status状态说明

### 4.1 状态枚举

| 状态 | 说明 | 后续处理 |
|------|------|----------|
| `success` | 完全成功，所有原子均通过质量检查 | 直接使用产出 |
| `partial` | 部分成功，部分原子通过质量检查 | 使用有效原子，记录问题 |
| `failed` | 执行失败，无有效产出 | 触发重试或降级策略 |
| `timeout` | 执行超时 | 返回已获取数据，触发重试 |

### 4.2 状态转换规则

```
                    +-------------+
                    |   pending   |
                    +------+------+
                           | 开始执行
                           v
                    +-------------+
                    |  executing  |
                    +------+------+
                           |
           +---------------+---------------+
           |               |               |
           v               v               v
    +-------------+ +-------------+ +-------------+
    |   success   | |   partial   | |   failed    |
    +-------------+ +-------------+ +-------------+
                           |               |
                           |               v
                           |        +-------------+
                           +------->|   timeout   |
                                    +-------------+
```

### 4.3 状态判定条件

| 状态 | 判定条件 |
|------|----------|
| `success` | `atoms_generated > 0` 且 `avg_quality_score >= 0.7` 且 `errors.length == 0` |
| `partial` | `atoms_generated > 0` 且 (`avg_quality_score < 0.7` 或 `errors.length > 0`) |
| `failed` | `atoms_generated == 0` 或存在不可恢复错误 |
| `timeout` | `execution_time_ms > timeout_seconds * 1000` |

---

## 5. LLM调用参数建议

### 5.1 各Agent推荐参数

| Agent | temperature | max_tokens | 说明 |
|-------|-------------|------------|------|
| Scholar_Internalizer | 0.2 - 0.3 | 8000 | 数学推导需要高确定性 |
| Code_Architect | 0.1 - 0.2 | 10000 | 代码分析需要精确性 |
| Scenario_Validator | 0.3 - 0.4 | 6000 | 场景推演需要一定创造性 |
| Knowledge_Vault | 0.1 | 4000 | 检索任务需要高确定性 |
| Strategic_Critic | 0.3 - 0.5 | 8000 | 战略决策需要综合判断 |

### 5.2 执行模式参数调整

| 执行模式 | temperature调整 | max_tokens调整 | timeout调整 |
|----------|-----------------|----------------|-------------|
| Deep_Internalization | 基准值 | 基准值 x 1.5 | 基准值 x 2 |
| Quick_Assessment | 基准值 + 0.1 | 基准值 x 0.5 | 基准值 x 0.5 |
| Engineering_Reproduction | 基准值 - 0.1 | 基准值 x 1.2 | 基准值 x 1.5 |

### 5.3 参数配置示例

```json
{
  "Scholar_Internalizer": {
    "Deep_Internalization": {
      "temperature": 0.25,
      "max_tokens": 12000,
      "timeout_seconds": 600
    },
    "Quick_Assessment": {
      "temperature": 0.35,
      "max_tokens": 4000,
      "timeout_seconds": 150
    }
  },
  "Code_Architect": {
    "Deep_Internalization": {
      "temperature": 0.15,
      "max_tokens": 15000,
      "timeout_seconds": 600
    },
    "Engineering_Reproduction": {
      "temperature": 0.1,
      "max_tokens": 12000,
      "timeout_seconds": 450
    }
  }
}
```

---

## 6. user_prompt构建规则

### 6.1 instruction构建规则

instruction字段应包含以下要素：

1. **动作动词**：明确的执行动作（分析、提取、评估、检索等）
2. **目标对象**：具体的分析目标（公式、代码、场景等）
3. **输出期望**：期望的产出类型和数量
4. **约束条件**：特殊限制或要求

**模板**：
```
[动作] [目标对象]，重点关注 [聚焦点]，产出 [数量] 个 [原子类型]，确保 [约束条件]。
```

**示例**：
```
分析Diffusion Planner的去噪采样公式，重点关注条件引导机制，产出3-5个Math_Atom，确保每个公式都有明确的论文页码锚定。
```

### 6.2 context构建规则

#### 6.2.1 paper_info构建

```json
{
  "paper_info": {
    "paper_id": "必需 - 论文唯一标识",
    "title": "必需 - 论文标题",
    "pdf_path": "可选 - 本地PDF路径",
    "arxiv_url": "可选 - ArXiv链接",
    "focus_sections": "推荐 - 重点分析章节列表"
  }
}
```

#### 6.2.2 code_info构建

```json
{
  "code_info": {
    "repo_url": "必需 - GitHub仓库URL",
    "branch": "可选 - 分支名称，默认main",
    "commit_hash": "可选 - 特定commit",
    "focus_files": "推荐 - 重点分析文件列表",
    "entry_point": "可选 - 入口文件"
  }
}
```

#### 6.2.3 dependencies构建

```json
{
  "dependencies": [
    {
      "task_id": "依赖任务ID",
      "output_ref": "引用的输出字段路径",
      "required": true
    }
  ]
}
```

### 6.3 output_requirements构建规则

| 字段 | 说明 | 示例 |
|------|------|------|
| `atom_type` | 期望的原子类型 | `"Math_Atom"`, `"Code_Atom"` |
| `required_fields` | 必需字段列表 | `["content.physical_intuition"]` |
| `quality_threshold` | 最低质量等级 | `"B"` |
| `max_atoms` | 最大原子数量 | `5` |

---

## 7. Prompt组装示例

### 7.1 Scholar_Internalizer 调用示例

#### 请求

```json
{
  "invocation_id": "INV_20260203_143052_SCHOLAR_001",
  "agent": "Scholar_Internalizer",
  "task_id": "TASK_SCHOLAR_DIFF_01",
  "plan_id": "AVP_E2E_20260203_001",
  "system_prompt": "# Role: Scholar_Internalizer (首席研究员/理论内化专家)\n\n## Profile\n* **角色名称**: Scholar_Internalizer\n* **角色描述**: 深耕自动驾驶算法架构与随机过程的资深研究员...\n\n[完整System Prompt内容，来自 agents_system/Scholar_Internalizer.md]",
  "user_prompt": {
    "instruction": "深度分析Diffusion Planner论文中的去噪采样公式，提取核心数学原子，建立物理直觉映射。",
    "context": {
      "paper_info": {
        "paper_id": "ArXiv:2406.01234",
        "title": "Diffusion Planner: End-to-End Planning with Denoising Diffusion",
        "pdf_path": "/raw_papers/diffusion_planner.pdf",
        "focus_sections": ["Section 3.2 Denoising Process", "Section 4.1 Conditional Guidance"]
      },
      "code_info": {
        "repo_url": "https://github.com/example/diffusion-planner",
        "branch": "main",
        "focus_files": ["models/diffusion.py"]
      },
      "dependencies": [
        {
          "task_id": "TASK_VAULT_CHECK_01",
          "output_ref": "existing_assets"
        }
      ],
      "existing_assets": ["MATH_VAD_ATTENTION_01", "MATH_DETR_QUERY_01"]
    },
    "focus": "去噪采样公式的物理直觉与AVP场景映射",
    "output_requirements": {
      "atom_type": "Math_Atom",
      "required_fields": [
        "content.mathematical_expression",
        "content.physical_intuition",
        "content.traditional_mapping",
        "content.avp_relevance",
        "provenance.paper_location"
      ],
      "quality_threshold": "B",
      "max_atoms": 5
    }
  },
  "constraints": {
    "timeout_seconds": 600,
    "max_tokens": 12000,
    "temperature": 0.25
  }
}
```

#### 响应

```json
{
  "invocation_id": "INV_20260203_143052_SCHOLAR_001",
  "status": "success",
  "agent": "Scholar_Internalizer",
  "task_id": "TASK_SCHOLAR_DIFF_01",
  "execution_time_ms": 45230,
  "output": {
    "atoms": [
      {
        "asset_id": "MATH_DIFF_DENOISE_01",
        "category": "Math_Ops",
        "data_status": "Verified_Source_Anchored",
        "metadata": {
          "created_at": "2026-02-03T14:35:22Z",
          "created_by": "Scholar_Internalizer",
          "version": "1.0",
          "tags": ["#Diffusion", "#Denoising", "#Trajectory"]
        },
        "content": {
          "mathematical_expression": "x_{t-1} = \\frac{1}{\\sqrt{\\alpha_t}}(x_t - \\frac{1-\\alpha_t}{\\sqrt{1-\\bar{\\alpha}_t}}\\epsilon_\\theta(x_t, t, c))",
          "physical_intuition": "这是一个带条件引导的轨迹去噪过程。在AVP场景下，x_t代表当前时刻的噪声轨迹，c是地库环境条件（如车位边界、障碍物位过程逐步将随机轨迹细化为符合环境约束的可行轨迹。",
          "traditional_mapping": "类似于传统规控中的迭代优化器，每一步去噪相当于一次梯度下降，将轨迹向可行域推进。",
          "avp_relevance": "地库窄路场景下的轨迹细化，条件c编码了库位边界约束",
          "assumptions": ["高斯噪声假设", "马尔可夫链假设"],
          "boundary_conditions": "在动态障碍物频繁出现的场景下，条件c的实时更新可能滞后"
        },
        "provenance": {
          "paper_id": "ArXiv:2406.01234",
          "paper_location": "Page 5, Eq. 7",
          "code_link": "github.com/example/diffusion-planner/blob/main/models/diffusion.py#L142-L158",
          "atom_path": "/atoms/math_ops/MATH_DIFF_DENOISE_01.json"
        },
        "delta_audit": {
          "existing_assets": ["MATH_VAD_ATTENTION_01"],
          "incremental_value": "相比VAD的注意力机制，Diffusion去噪提供了显式的不确定性建模，能够生成多样化的候选轨迹。",
          "contradiction_marked": false
        }
      }
    ],
    "summary": "成功提取3个数学原子，覆盖Diffusion Planner的去噪采样核心公式、条件引导机制和噪声调度策略。"
  },
  "quality_check": {
    "atoms_generated": 3,
    "avg_quality_score": 0.88,
    "grades": {
      "A": 2,
      "B": 1,
      "C": 0,
      "D": 0
    },
    "issues": []
  },
  "errors": []
}
```

### 7.2 Code_Architect 调用示例

#### 请求

```json
{
  "invocation_id": "INV_20260203_144530_CODE_001",
  "agent": "Code_Architect",
  "task_id": "TASK_CODE_DIFF_01",
  "plan_id": "AVP_E2E_20260203_001",
  "system_prompt": "# Role: Code_Architect (代码架构师/工程落地专家)\n\n[完整System Prompt内容]",
  "user_prompt": {
    "instruction": "分析Diffusion Planner的采样器实现，评估在Orin-X平台的部署可行性，提取可复用的代码原子。",
    "context": {
      "paper_info": {
        "paper_id": "ArXiv:2406.01234",
        "title": "Diffusion Planner"
      },
   "code_info": {
        "repo_url": "https://github.com/example/diffusion-planner",
        "branch": "main",
        "focus_files": [
          "models/diffusion.py",
          "models/sampler.py",
          "utils/noise_schedule.py"
        ],
        "entry_point": "inference.py"
      },
      "dependencies": [
        {
          "task_id": "TASK_SCHOLAR_DIFF_01",
          "output_ref": "output.atoms"
        }
      ],
      "existing_assets": ["CODE_VAD_SAMPLER_01"]
    },
    "focus": "采样器的TensorRT加速可行性与性能瓶颈分析",
    "output_requirements": {
      "atom_type": "Code_Atom",
      "required_fields": [
        "content.source_code",
        "content.code_location",
        "content.performance_note.complexity",
        "content.performance_note.deployment_suggestion"
      ],
      "quality_threshold": "B",
      "max_atoms": 4
    }
  },
  "constraints": {
    "timeout_seconds": 450,
    "max_tokens": 12000,
    "temperature": 0.15
  }
}
```

### 7.3 Scenario_Validator 调用示例

#### 请求

```json
{
  "invocation_id": "INV_20260203_150000_VALIDATOR_001",
  "agent": "Scenario_Validator",
  "task_id": "TASK_VALIDATOR_DIFF_01",
  "plan_id": "AVP_E2E_20260203_001",
  "system_prompt": "# Role: Scenario_Validator\n\n[完整System Prompt内容]",
  "user_prompt": {
    "instruction": "基于Diffusion Planner的数学假设和代码实现，推演AVP地库场景下的潜在失效模式。",
    "context": {
      "paper_info": {
        "paper_id": "ArXiv:2406.01234"
      },
      "dependencies": [
        {
          "task_id": "TASK_SCHOLAR_DIFF_01",
          "output_ref": "output.atoms"
        },
        {
          "task_id": "TASK_CODE_DIFF_01",
          "output_ref": "output.atoms"
        }
      ],
      "existing_assets": ["SCENARIO_AVP_LIGHT_01", "SCENARIO_NARROW_PASSAGE_01"]
    },
    "focus": "去噪过程在动态环境下的失效风险",
    "output_requirements": {
      "atom_type": "Scenario_Atom",
      "required_fields": [
        "content.challenge_scenario",
        "content.failure_mechanism",
        "content.risk_assessment.severity"
      ],
      "quality_threshold": "B",
      "max_atoms": 5
    }
  },
  "constraints": {
    "timeout_seconds": 300,
    "max_tokens": 8000,
    "temperature": 0.35
  }
}
```

---

## 8. 错误码定义

### 8.1 错误码列表

| 错误码 | 类型 | 说明 | 可恢复 |
|--------|------|------|--------|
| E001 | source_not_found | 无法访问源文件/仓库 | 是 |
| E002 | parse_error | 输入解析失败 | 是 |
| E003 | validation_error | 输出验证失败 | 是 |
| E004 | timeout | 执行超时 | 是 |
| E005 | quality_below_threshold | 质量低于阈值 | 是 |
| E006 | dependency_failed | 依赖任务失败 | 否 |
| E007 | hallucination_detected | 检测到幻觉内容 | 否 |
| E008 | system_error | 系统内部错误 | 否 |

### 8.2 错误处理建议

| 错误码 | 建议处理 |
|--------|----------|
| E001 | 重试3次，间隔递增（5s, 15s, 30s） |
| E002 | 检查输入格式，修正后重试 |
| E003 | 降低quality_threshold后重试 |
| E004 | 增加timeout_seconds后重试 |
| E005 | 调整temperature或增加max_tokens后重试 |
| E006 | 终止当前任务，标记为blocked |
| E007 | 终止当前任务，触发人工审核 |
| E008 | 记录日志，通知运维 |

---

## 9. 附录

### 9.1 invocation_id生成规则

```
INV_{YYYYMMDD}_{HHMMSS}_{AGENT_SHORT}_{SEQ}

示例: INV_20260203_143052_SCHOLAR_001

其中:
- YYYYMMDD: 日期
- HHMMSS: 时间
- AGENT_SHORT: Agent简称
  - SCHOLAR: Scholar_Internalizer
  - CODE: Code_Architect
  - VALIDATOR: Scenario_Validator
  - VAULT: Knowledge_Vault
  - CRITIC: Strategic_Critic
- SEQ: 3位序号，从001开始
```

### 9.2 相关文档

- `01_task_package_protocol.md` - 任务包协议
- `03_data_transfer_protocol.md` - 数据传递协议
- `04_quality_gate_protocol.md` - 质量门协议
- `05_error_handling_protocol.md` - 错误处理协议
- `../common/common_output.md` - 原子输出格式标准
- `schemas/agent_request.schema.json` - 请求JSON Schema
- `schemas/agent_response.schema.json` - 响应JSON Schema

---

## 版本历史

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| 1.0 | 2026-02-03 | 初始版本 |
