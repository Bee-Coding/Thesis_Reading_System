# 数据库设计

## 概述

本目录包含Thesis_Reading_System的数据库Schema设计，基于PostgreSQL 14+。数据库设计旨在支持多智能体系统的任务调度、知识原子存储、质量监控和错误处理。

## 设计原则

1. **JSONB优先**：灵活存储结构化数据，支持快速查询和更新
2. **事件溯源**：记录所有状态变化，支持调试和审计
3. **可扩展性**：支持未来新的原子类型和Agent类型
4. **性能优化**：为常用查询创建索引，分区大表
5. **数据完整性**：使用外键约束和CHECK约束确保数据一致性

## 核心表结构

### 1. 原子表 (atoms)

存储所有类型的知识原子（Math_Atom, Code_Atom, Scenario_Atom, Strategic_Decision）。

**关键字段：**
- `asset_id`: 资产唯一标识符（如 `MATH_PLAN_DIFF_01`）
- `category`: 原子类别（Math_Ops, Code_Snippets, Scenarios, Strategic）
- `data_status`: 数据验证状态
- `metadata`: 元数据（创建时间、创建者、版本、标签）
- `content`: 具体内容（JSON格式，类别特定）
- `provenance`: 来源追溯信息
- `delta_audit`: 增量审计信息

**索引：**
- 按类别、状态、质量分数、创建时间索引
- 资产ID唯一索引
- 全文搜索索引（内容搜索）

### 2. 原子关系表 (atom_relations)

存储原子之间的关联关系。

**关系类型：**
- `depends_on`: 依赖关系
- `contradicts`: 矛盾关系
- `supports`: 支持关系
- `extends`: 扩展关系
- `similar_to`: 相似关系
- `part_of`: 部分关系

### 3. 计划表 (plans)

存储Orchestrator生成的任务包。

**状态流转：**
```
created → scheduled → executing → (completed | failed | cancelled)
```

**存储内容：**
- 完整的任务包JSON（遵循 `01_task_package_protocol.md`）
- 执行统计信息
- 时间戳记录

### 4. 任务表 (tasks)

存储任务执行记录，与计划一对多关系。

**状态：**
- `pending`: 等待执行
- `waiting_dependencies`: 等待依赖就绪
- `ready`: 依赖就绪，等待调度
- `executing`: 执行中
- `success`: 执行成功
- `partial`: 部分成功
- `failed`: 执行失败
- `timeout`: 执行超时
- `blocked`: 被质量门阻塞
- `skipped`: 跳过执行

### 5. 执行记录表 (executions)

存储Agent调用的详细记录，包括请求和响应。

**记录信息：**
- 完整的Agent请求和响应JSON
- 执行指标（时间、token使用、成本估算）
- 质量指标
- 错误信息

### 6. 质量检查表 (quality_checks)

存储质量门控检查结果。

**门类型：**
- `pre_execution`: 预执行门
- `in_execution`: 执行中门
- `post_execution`: 后执行门
- `integration`: 集成门

### 7. 错误记录表 (error_records)

存储系统错误和处理记录。

**错误分类：**
- 按严重程度：critical, error, warning, info
- 按可恢复性：recoverable, partially_recoverable, non_recoverable
- 按来源：system, agent, data, validation, quality, resource, external

### 8. 资源使用表 (resource_usage)

存储系统资源使用情况，用于容量规划和监控。

**资源类型：**
- GPU内存、CPU使用率、磁盘空间、API配额、网络带宽

### 9. 知识库统计表 (knowledge_stats)

每日快照，存储知识库的聚合统计信息。

## 视图 (Views)

### 1. task_execution_status

任务执行状态视图，关联计划和任务信息。

### 2. quality_gate_status

质量门控状态视图，显示检查结果和关联任务状态。

### 3. error_statistics

错误统计视图，提供错误聚合分析。

## 函数和触发器

### 1. update_atom_quality_score()

在插入或更新原子时，自动从metadata中提取质量分数和等级。

### 2. update_plan_statistics()

在任务状态变化时，自动更新计划的统计信息（完成数、失败数等）。

### 3. log_error_handling()

在错误状态变化时，记录处理日志到 `error_handling_logs` 表。

## 数据模型关系图

```
┌─────────────┐     ┌───────────────┐     ┌──────────────┐
│    plans    │1───∞│    tasks      │1───∞│ executions   │
└─────────────┘     └───────────────┘     └──────────────┘
                         │                        │
                         │                        │
                    ┌─────────────┐          ┌─────────────┐
                    │   atoms     │∞─────∞   │atom_relations│
                    └─────────────┘          └─────────────┘
                         │
                         │
                    ┌─────────────┐
                    │quality_checks│
                    └─────────────┘
                         │
                         │
                    ┌─────────────┐
                    │error_records│
                    └─────────────┘
```

## 部署指南

### 1. 环境要求

- PostgreSQL 14+
- `uuid-ossp` 扩展
- `pgcrypto` 扩展
- 至少 10GB 存储空间（随知识库增长）

### 2. 初始化步骤

```bash
# 创建数据库
createdb thesis_reading_system

# 连接数据库
psql thesis_reading_system

# 执行Schema
\i agents_system/database/schema.sql
```

### 3. 备份策略

```sql
-- 每日全量备份
pg_dump thesis_reading_system > backup_$(date +%Y%m%d).sql

-- 原子表分区备份（按日期）
pg_dump -t atoms --data-only thesis_reading_system > atoms_backup_$(date +%Y%m%d).sql
```

### 4. 性能优化建议

```sql
-- 为大型表创建分区（按日期）
CREATE TABLE atoms_2026_02 PARTITION OF atoms
FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');

-- 创建复合索引（常用查询）
CREATE INDEX idx_atoms_category_status ON atoms(category, data_status);
CREATE INDEX idx_tasks_plan_status ON tasks(plan_id, status);

-- 定期维护
VACUUM ANALYZE atoms;
REINDEX TABLE atoms;
```

## 查询示例

### 1. 查找高质量数学原子

```sql
SELECT asset_id, content->>'mathematical_expression' as formula,
       quality_score, quality_grade
FROM atoms
WHERE category = 'Math_Ops'
  AND quality_grade IN ('A', 'B')
  AND data_status = 'Verified_Source_Anchored'
ORDER BY quality_score DESC
LIMIT 10;
```

### 2. 获取任务执行统计

```sql
SELECT plan_id, 
       COUNT(*) as total_tasks,
       SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as completed,
       SUM(CASE WHEN status IN ('failed', 'timeout') THEN 1 ELSE 0 END) as failed,
       AVG(execution_time_ms) as avg_execution_time
FROM tasks
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY plan_id
ORDER BY total_tasks DESC;
```

### 3. 质量门控失败分析

```sql
SELECT gate_type, check_status, failure_action, COUNT(*) as count
FROM quality_checks
WHERE check_status = 'failed'
  AND created_at >= NOW() - INTERVAL '24 hours'
GROUP BY gate_type, check_status, failure_action
ORDER BY count DESC;
```

### 4. 错误趋势分析

```sql
SELECT date_trunc('hour', detected_at) as hour,
       error_code,
       COUNT(*) as error_count,
       AVG(EXTRACT(EPOCH FROM (resolved_at - detected_at))) as avg_resolution_seconds
FROM error_records
WHERE detected_at >= NOW() - INTERVAL '24 hours'
GROUP BY hour, error_code
ORDER BY hour DESC, error_count DESC;
```

## 监控指标

### 关键性能指标 (KPIs)

1. **知识库增长**：每日新增原子数
2. **原子质量**：A/B级原子比例
3. **任务成功率**：成功任务比例
4. **平均执行时间**：任务执行时间
5. **错误率**：错误发生频率
6. **质量门通过率**：质量检查通过比例

### 监控查询

```sql
-- 实时系统状态
SELECT 
  (SELECT COUNT(*) FROM atoms WHERE created_at >= NOW() - INTERVAL '24 hours') as atoms_today,
  (SELECT COUNT(*) FROM tasks WHERE status = 'executing') as tasks_executing,
  (SELECT COUNT(*) FROM error_records WHERE severity = 'critical' AND resolved_at IS NULL) as critical_errors_open,
  (SELECT AVG(quality_score) FROM atoms WHERE created_at >= NOW() - INTERVAL '7 days') as avg_quality_week;
```

## 扩展指南

### 添加新的原子类型

1. 在 `atoms.category` 中添加新的枚举值
2. 更新 `common_output.md` 定义新类型的格式
3. 创建对应的内容验证规则
4. 更新统计查询以包含新类型

### 添加新的Agent类型

1. 在 `tasks.agent` 和 `executions.agent` 中添加新的枚举值
2. 更新Agent调用协议文档
3. 创建Agent特定的质量检查规则

### 数据迁移

```sql
-- 示例：添加新字段
ALTER TABLE atoms ADD COLUMN IF NOT EXISTS embedding_vector vector(768);

-- 示例：创建新索引
CREATE INDEX idx_atoms_embedding ON atoms USING ivfflat (embedding_vector);
```

## 故障排除

### 常见问题

1. **连接问题**：检查PostgreSQL服务状态和防火墙设置
2. **性能问题**：检查索引使用情况，考虑分区大表
3. **数据不一致**：运行数据完整性检查脚本
4. **备份失败**：检查磁盘空间和权限设置

### 维护脚本

维护脚本位于 `scripts/` 目录：
- `check_data_integrity.sql`：数据完整性检查
- `optimize_performance.sql`：性能优化
- `cleanup_old_data.sql`：清理旧数据

## 版本历史

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| 1.0 | 2026-02-03 | 初始版本，支持核心功能 |
| 1.1 | 待定 | 计划添加向量搜索支持 |

## 相关文档

- `../protocols/01_task_package_protocol.md` - 任务包协议
- `../protocols/02_agent_invocation_protocol.md` - Agent调用协议
- `../protocols/03_data_transfer_protocol.md` - 数据传递协议
- `../protocols/04_quality_gate_protocol.md` - 质量门协议
- `../protocols/05_error_handling_protocol.md` - 错误处理协议
- `../common/common_output.md` - 原子输出格式标准