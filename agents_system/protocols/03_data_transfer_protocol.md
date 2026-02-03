# 数据传递协议 (Data Transfer Protocol)

## 1. 概述

### 1.1 协议目的

本协议定义了多智能体系统中Agent之间的数据传递机制，确保：

- **数据一致性**：上游Agent的输出能被下游Agent正确解析和使用
- **引用透明性**：通过统一的引用格式，实现数据的延迟加载和按需获取
- **上下文效率**：在有限的上下文窗口内，最大化有效信息密度
- **可追溯性**：所有数据传递都有明确的来源和路径记录

### 1.2 适用范围

本协议适用于以下场景：

| 场景 | 说明 |
|------|------|
| 任务链传递 | 前置任务输出作为后续任务输入 |
| 并行任务汇聚 | 多个并行任务的输出汇聚到单一任务 |
| 跨阶段引用 | 后续阶段引用早期阶段的产出 |
| 外部数据接入 | 引用文件系统或数据库中的数据 |

---

## 2. 数据引用格式

### 2.1 引用类型概览

```json
{
  "data_reference": {
    "ref_type": "task_output | file | db_query",
    "...": "类型特定字段"
  }
}
```

### 2.2 任务输出引用 (task_output)

引用其他任务的执行结果。

```json
{
  "data_reference": {
    "ref_type": "task_output",
    "task_id": "task_scholar_001",
    "path": "$.atoms[*].content",
    "filter": {
      "field": "atom_type",
      "operator": "eq",
      "value": "methodology"
    },
    "transform": "none"
  }
}
```

**字段说明：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `ref_type` | string | 是 | 固定值 `"task_output"` |
| `task_id` | string | 是 | 被引用任务的唯一标识 |
| `path` | string | 否 | JSONPath表达式，定位具体数据位置 |
| `filter` | object | 否 | 过滤条件，筛选符合条件的数据 |
| `transform` | string | 否 | 转换方式：`none`/`summary`/`keys_only` |

**Path表达式示例：**

```
$.atoms                      # 获取所有原子
$.atoms[0]                   # 获取第一个原子
$.atoms[*].content           # 获取所有原子的content字段
$.metadata.quality_score     # 获取质量分数
$.atoms[?(@.confidence>0.8)] # 获取置信度>0.8的原子
```

**Filter操作符：**

| 操作符 | 说明 | 示例 |
|--------|------|------|
| `eq` | 等于 | `{"field": "type", "operator": "eq", "value": "claim"}` |
| `ne` | 不等于 | `{"field": "status", "operator": "ne", "value": "rejected"}` |
| `in` | 包含于 | `{"field": "tag", "operator": "in", "value": ["core", "key"]}` |
| `gt`/`gte` | 大于/大于等于 | `{"field": "score", "operator": "gte", "value": 0.7}` |
| `lt`/`lte` | 小于/小于等于 | `{"field": "priority", "operator": "lt", "value": 3}` |
| `contains` | 字符串包含 | `{"field": "text", "operator": "contains", "value": "attention"}` |

### 2.3 文件引用 (file)

引用文件系统中的文件。

```json
{
  "data_reference": {
    "ref_type": "file",
    "path": "/workspace/papers/attention_is_all_you_need.pdf",
    "format": "pdf",
    "encoding": "utf-8",
    "sections": ["abstract", "introduction", "methodology"]
  }
}
```

**字段说明：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `ref_type` | string | 是 | 固定值 `"file"` |
| `path` | string | 是 | 文件绝对路径或相对于工作目录的路径 |
| `format` | string | 否 | 文件格式：`pdf`/`md`/`json`/`txt` |
| `encoding` | string | 否 | 文件编码，默认 `utf-8` |
| `sections` | array | 否 | 仅提取指定章节（适用于结构化文档） |

### 2.4 数据库引用 (db_query)

引用数据库中的数据。

```json
{
  "data_reference": {
    "ref_type": "db_query",
    "table": "knowledge_atoms",
    "conditions": [
      {"field": "paper_id", "operator": "eq", "value": "paper_001"},
      {"field": "atom_type", "operator": "in", "value": ["claim", "evidence"]}
    ],
    "select": ["atom_id", "content", "confidence"],
    "order_by": {"field": "confidence", "direction": "desc"},
    "limit": 50
  }
}
```

**字段说明：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `ref_type` | string | 是 | 固定值 `"db_query"` |
| `table` | string | 是 | 目标表名 |
| `conditions` | array | 否 | 查询条件列表（AND关系） |
| `select` | array | 否 | 返回字段列表，默认返回全部 |
| `order_by` | object | 否 | 排序规则 |
| `limit` | integer | 否 | 返回记录数限制 |

**支持的表：**

| 表名 | 说明 |
|------|------|
| `papers` | 论文元数据 |
| `knowledge_atoms` | 知识原子 |
| `relations` | 原子间关系 |
| `quality_records` | 质量评估记录 |
| `execution_logs` | 执行日志 |

---

## 3. 引用解析规则

### 3.1 解析时机

```
+------------------+     +------------------+     +------------------+
|   任务调度前      |     |   任务执行时      |     |   任务完成后      |
+------------------+     +------------------+     +------------------+
        |                        |                   |
        v                        v                        v
  [静态引用解析]           [动态引用解析]           [结果引用注册]
  - 文件引用              - 任务输出引用            - 输出存储
  - 数据库引用            - 条件引用               - 引用索引更新
  - 已完成任务引用         - 流式数据引用            - 依赖通知
```

**解析时机规则：**

| 引用类型 | 解析时机 | 说明 |
|----------|----------|------|
| `file` | 任务调度前 | 文件内容在调度时加载 |
| `db_query` | 任务调度前 | 数据库查询在调度时执行 |
| `task_output` (已完成) | 任务调度前 | 已完成任务的输出直接获取 |
| `task_output` (未完成) | 任务执行时 | 等待依赖任务完成后解析 |

### 3.2 解析失败处理

```json
{
  "resolution_failure": {
    "ref_type": "task_output",
    "task_id": "task_scholar_001",
    "error_code": "REF_NOT_FOUND",
    "error_message": "Referenced task output not found",
    "fallback_strategy": "use_default",
    "fallback_value": null,
    "timestamp": "2026-02-03T10:30:00Z"
  }
}
```

**错误码与处理策略：**

| 错误码 | 说明 | 默认处理策略 |
|--------|------|--------------|
| `REF_NOT_FOUND` | 引用目标不存在 | 使用默认值或跳过 |
| `REF_TIMEOUT` | 引用解析超时 | 重试或降级 |
| `REF_PERMISSION_DENIED` | 无访问权限 | 报错终止 |
| `REF_FORMAT_ERROR` | 数据格式不匹配 | 尝试转换或报错 |
| `REF_PATH_INVALID` | JSONPath无效 | 返回完整数据 |
| `REF_FILTER_ERROR` | 过滤条件错误 | 忽略过滤条件 |

**Fallback策略配置：**

```json
{
  "fallback_config": {
    "strategy": "use_default | retry | skip | abort",
    "default_value": null,
    "retry_count": 3,
    "retry_delay_ms": 1000,
    "on_final_failure": "skip | abort"
  }
}
```

---

## 4. 数据格式转换

### 4.1 完整传递 (Full Transfer)

将数据完整传递给下游Agent，不做任何压缩或摘要。

```json
{
  "transfer_mode": "full",
  "data": {
    "atoms": [
      {
        "atom_id": "atom_001",
        "atom_type": "claim",
        "content": "Transformer架构通过自注意力机制实现了并行计算...",
        "confidence": 0.95,
        "evidence": ["evidence_001", "evidence_002"],
        "metadata": {
          "source_section": "3.1",
          "extraction_method": "llm_extraction"
        }
      }
    ],
    "relations": [],
    "metadata": {}
  }
}
```

**适用场景：**
- 数据量小于上下文窗口限制的20%
- 下游任务需要完整细节
- 数据不可压缩（如代码、公式）

### 4.2 摘要传递 (Summary Transfer)

对数据进行摘要后传递，保留关键信息。

```json
{
  "transfer_mode": "summary",
  "summary_config": {
    "max_length": 2000,
    "preserve_fields": ["atom_id", "atom_type", "confidence"],
    "summarize_fields": ["content"],
    "summary_ratio": 0.3
  },
  "data": {
    "atoms_summary": [
      {
        "atom_id": "atom_001",
        "atom_type": "claim",
        "content_summary": "Transformer通过自注意力实现并行计算",
        "confidence": 0.95
      }
    ],
    "total_atoms": 15,
    "summarized_atoms": 15,
    "omitted_fields": ["evidence", "metadata"]
  }
}
```

**摘要策略：**

| 策略 | 说明 | 压缩比 |
|------|------|--------|
| `extractive` | 提取关键句子 | 0.2-0.4 |
| `abstractive` | 生成式摘要 | 0.1-0.3 |
| `key_points` | 仅保留要点 | 0.1-0.2 |
| `truncate` | 截断超长内容 | 可配置 |

### 4.3 引用传递 (Reference Transfer)

仅传递数据引用，下游Agent按需获取。

```json
{
  "transfer_mode": "reference",
  "reference": {
    "ref_type": "task_output",
    "task_id": "task_scholar_001",
    "available_paths": [
      "$.atoms",
      "$.relations",
      "$.metadata"
    ],
    "data_stats": {
      "total_atoms": 45,
      "total_relations": 120,
      "estimated_size_bytes": 125000
    }
  },
  "inline_preview": {
    "atoms_preview": [
      {"atom_id": "atom_001", "atom_type": "claim", "content_preview": "Transformer架构..."}
    ],
    "preview_count": 3
  }
}
```

**适用场景：**
- 数据量超过上下文窗口限制
- 下游任务可能只需要部分数据
- 需要保持数据实时性

---

## 5. 传递策略选择表

### 5.1 基于数据量的策略选择

| 数据量 (tokens) | 推荐策略 | 说明 |
|-----------------|----------|------|
| < 2,000 | 完整传递 | 直接传递，无需压缩 |
| 2,000 - 10,000 | 摘要传递 | 压缩后传递，保留关键信息 |
| 10,000 - 50,000 | 引用传递 + 预览 | 传递引用和摘要预览 |
| > 50,000 | 分批引用传递 | 分批次按需获取 |

### 5.2 基于任务类型的策略选择

| 下游任务类型 | 推荐策略 | 原因 |
|--------------|----------|------|
| Scholar (文献分析) | 完整传递 | 需要原文细节 |
| Code (代码分析) | 完整传递 | 代码不可压缩 |
| Validator (验证) | 摘要传递 | 关注结论和证据 |
| Knowledge_Vault (整合) | 引用传递 | 按需获取各部分 |
| Strategic_Critic (评估) | 摘要传递 | 关注整体质量 |
| Orchestrator (规划) | 摘要传递 | 关注任务状态 |

### 5.3 基于数据类型的策略选择

| 数据类型 | 推荐策略 | 压缩方法 |
|----------|----------|----------|
| 文本内容 | 摘要传递 | 提取式/生成式摘要 |
| 代码片段 | 完整传递 | 不压缩 |
| 数学公式 | 完整传递 | 不压缩 |
| 表格数据 | 摘要传递 | 保留表头+统计信息 |
| 图表描述 | 摘要传递 | 保留关键数据点 |
| 元数据 | 完整传递 | 通常较小 |
| 关系图谱 | 引用传递 | 按需查询 |

### 5.4 策略选择决策树

```
                    [数据量评估]
                         |
          +--------------+--------------+
          |              |              |
      < 2K tokens   2K-50K tokens   > 50K tokens
          |              |              |
    [完整传递]     [检查任务类型]    [分批引用]
                         |
          +--------------+--------------+
          |              |              |
    需要完整细节    可接受摘要      仅需索引
          |              |              |
    [完整传递]     [摘要传递]     [引用传递]
```

---

## 6. 上下文窗口管理

### 6.1 单次传递数据量限制

```json
{
  "context_limits": {
    "max_input_tokens": 100000,
    "max_output_tokens": 16000,
    "reserved_for_system_prompt": 5000,
    "reserved_for_instructions": 3000,
    "available_for_data": 92000,
    "safety_margin": 0.9,
    "effective_data_limit": 82800
  }
}
```

**各Agent的上下文分配：**

| Agent | 输入限制 | 数据区域 | 说明 |
|-------|----------|----------|------|
| Orchestrator | 100K | 85K | 需要处理完整任务上下文 |
| Scholar | 100K | 80K | 需要处理长文档 |
| Code | 100K | 75K | 代码+文档 |
| Validator | 100K | 70K | 多源数据验证 |
| Knowledge_Vault | 100K | 85K | 大量原子整合 |
| Strategic_Critic | 100K | 60K | 摘要评估为主 |

### 6.2 超限时的分批策略

**分批配置：**

```json
{
  "batch_config": {
    "enabled": true,
    "batch_size_tokens": 30000,
    "overlap_tokens": 500,
    "max_batches": 10,
    "aggregation_strategy": "merge | vote | latest",
    "batch_priority": "chronological | importance | random"
  }
}
```

**分批处理流程：**

```
[原始数据 150K tokens]
         |
         v
    [数据分割器]
         |
    +----+----+----+----+----+
    |    |    |    |    |    |
  Batch1 Batch2 Batch3 Batch4 Batch5
  (30K)  (30K)  (30K)  (30K)  (30K)
    |    |    |    |    |    |
    v    v    v    v    v    v
  [Agent处理每个批次]
    |    |    |    |    |    |
    +----+----+----+----+----+
         |
         v
    [结果聚合器]
         |
         v
    [最终输出]
```

**聚合策略说明：**

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `merge` | 合并所有批次结果 | 提取类任务 |
| `vote` | 多数投票决定 | 分类/判断类任务 |
| `latest` | 使用最后批次结果 | 迭代优化类任务 |
| `weighted` | 加权合并 | 置信度差异大时 |

### 6.3 关键信息优先级

**优先级定义：**

```json
{
  "priority_config": {
    "levels": [
      {
        "level": 1,
        "name": "critical",
        "description": "必须包含，不可省略",
        "examples": ["核心论点", "主要结论", "关键证据"]
      },
      {
        "level": 2,
        "name": "important",
        "description": "应当包含，空间不足时可摘要",
        "examples": ["方法论细节", "实验设置", "相关工作"]
      },
      {
        "level": 3,
        "name": "supplementary",
        "description": "可选包含，空间不足时可省略",
        "examples": ["背景介绍", "未来工作", "致谢"]
      },
      {
        "level": 4,
        "name": "optional",
        "description": "仅在空间充足时包含",
        "examples": ["详细推导", "完整数据表", "附录内容"]
      }
    ]
  }
}
```

**优先级裁剪算法：**

```
输入: 数据列表 D, 上下文限制 L
输出: 裁剪后的数据列表 D'

1. 按优先级对 D 排序 (level 1 -> level 4)
2. 初始化 D' = [], current_size = 0
3. 对于 D 中的每个数据项 d:
   a. 如果 d.level == 1:
      - 必须添加，如超限则压缩
      - D'.append(compress(d) if needed)
   b. 如果 d.level == 2:
      - 如果 current_size + size(d) < L * 0.8:
        D'.append(d)
      - 否则: D'.append(summarize(d))
   c. 如果 d.level == 3:
      - 如果 current_size + size(d) < L * 0.9:
        D'.append(d)
      - 否则: 跳过
   d. 如果 d.level == 4:
      - 如果 current_size + size(d) < L * 0.95:
        D'.append(d)
      - 否则: 跳过
   e. 更新 current_size
4. 返回 D'
```

**各数据类型的默认优先级：**

| 数据类型 | 默认优先级 | 说明 |
|----------|------------|------|
| 任务指令 | 1 (critical) | 必须完整传递 |
| 核心原子 (claim) | 1 (critical) | 论文核心论点 |
| 证据原子 (evidence) | 2 (important) | 支撑论点的证据 |
| 方法原子 (methodology) | 2 (important) | 方法论描述 |
| 背景原子 (background) | 3 (supplementary) | 背景信息 |
| 元数据 | 3 (supplementary) | 辅助信息 |
| 完整引用 | 4 (optional) | 详细参考文献 |
| 原始文本 | 4 (optional) | 未处理的原文 |

---

## 7. 完整示例

### 7.1 任务链数据传递示例

```json
{
  "task_chain": [
    {
      "task_id": "task_scholar_001",
      "agent": "Scholar",
      "input": {
        "data_references": [
          {
            "ref_type": "file",
            "path": "/workspace/papers/transformer.pdf",
            "format": "pdf"
          }
        ]
      },
      "output_schema": "knowledge_atoms"
    },
    {
      "task_id": "task_validator_001",
      "agent": "Validator",
      "input": {
        "data_references": [
          {
            "ref_type": "task_output",
            "task_id": "task_scholar_001",
            "path": "$.atoms[?(@.atom_type=='claim')]",
            "transform": "summary"
          }
        ],
        "transfer_config": {
          "mode": "summary",
          "max_tokens": 5000,
          "priority_filter": [1, 2]
        }
      }
    },
    {
      "task_id": "task_vault_001",
      "agent": "Knowledge_Vault",
      "input": {
        "data_references": [
          {
            "ref_type": "task_output",
            "task_id": "task_scholar_001",
            "path": "$.atoms"
          },
          {
            "ref_type": "task_output",
            "task_id": "task_validator_001",
            "path": "$.validation_results"
          },
          {
            "ref_type": "db_query",
            "table": "knowledge_atoms",
            "conditions": [
              {"field": "domain", "operator": "eq", "value": "deep_learning"}
            ],
            "limit": 100
          }
        ],
        "transfer_config": {
          "mode": "reference",
          "inline_preview_count": 5
        }
      }
    }
  ]
}
```

### 7.2 上下文窗口管理示例

```json
{
  "context_management": {
    "task_id": "task_vault_001",
    "total_input_data": {
      "scholar_output": {"tokens": 45000, "priority": 1},
      "validator_output": {"tokens": 8000, "priority": 1},
      "db_query_result": {"tokens": 35000, "priority": 2},
      "system_prompt": {"tokens": 5000, "priority": 1},
      "task_instructions": {"tokens": 2000, "priority": 1}
    },
    "total_tokens": 95000,
    "context_limit": 82800,
    "overflow": 12200,
    "resolution": {
      "strategy": "priority_based_trimming",
      "actions": [
        {
          "data": "db_query_result",
          "action": "summarize",
          "original_tokens": 35000,
          "reduced_tokens": 15000
        },
        {
          "data": "scholar_output",
          "action": "filter_by_priority",
          "filter": "priority <= 2",
          "original_tokens": 45000,
          "reduced_tokens": 38000
        }
      ],
      "final_tokens": 68000,
      "within_limit": true
    }
  }
}
```

---

## 8. 版本信息

| 属性 | 值 |
|------|-----|
| 版本 | 1.0 |
| 创建时间 | 2026-02-03 |
| 适用范围 | Thesis_Reading_System Agent间数据传递 |
| 维护者 | System Architect |

## 9. 相关文档

- `01_task_package_protocol.md` - 任务包协议
- `02_agent_invocation_protocol.md` - Agent调用协议
- `04_quality_gate_protocol.md` - 质量门协议
- `05_error_handling_protocol.md` - 错误处理协议
- `../common/common_output.md` - 原子输出格式规范
