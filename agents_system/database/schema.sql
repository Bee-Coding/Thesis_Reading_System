-- Thesis_Reading_System 数据库Schema
-- 版本: 1.0
-- 创建时间: 2026-02-03

-- 启用必要的扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- 原子表：存储所有类型的知识原子
-- ============================================================================
CREATE TABLE atoms (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    asset_id VARCHAR(50) UNIQUE NOT NULL,
    category VARCHAR(20) NOT NULL CHECK (category IN ('Math_Ops', 'Code_Snippets', 'Scenarios', 'Strategic')),
    data_status VARCHAR(30) NOT NULL CHECK (data_status IN (
        'Verified_Source_Anchored', 
        'Unverified_No_Source', 
        'Partially_Verified', 
        'Contradiction_Detected'
    )),
    metadata JSONB NOT NULL,
    content JSONB NOT NULL,
    provenance JSONB NOT NULL,
    delta_audit JSONB NOT NULL,
    quality_score DECIMAL(4,2) CHECK (quality_score >= 0 AND quality_score <= 100),
    quality_grade CHAR(1) CHECK (quality_grade IN ('A', 'B', 'C', 'D')),
    paper_id VARCHAR(100),
    code_repo_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ingested_at TIMESTAMP WITH TIME ZONE,
    CONSTRAINT valid_metadata CHECK (
        metadata ? 'created_at' AND 
        metadata ? 'created_by' AND 
        metadata ? 'version' AND 
        metadata ? 'tags'
    )
);

CREATE INDEX idx_atoms_category ON atoms(category);
CREATE INDEX idx_atoms_data_status ON atoms(data_status);
CREATE INDEX idx_atoms_quality_score ON atoms(quality_score);
CREATE INDEX idx_atoms_created_at ON atoms(created_at);
CREATE INDEX idx_atoms_paper_id ON atoms(paper_id);

-- ============================================================================
-- 原子关系表
-- ============================================================================
CREATE TABLE atom_relations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_atom_id VARCHAR(50) NOT NULL REFERENCES atoms(asset_id) ON DELETE CASCADE,
    target_atom_id VARCHAR(50) NOT NULL REFERENCES atoms(asset_id) ON DELETE CASCADE,
    relation_type VARCHAR(30) NOT NULL CHECK (relation_type IN (
        'depends_on', 'contradicts', 'supports', 'extends', 'similar_to', 'part_of'
    )),
    confidence DECIMAL(3,2) CHECK (confidence >= 0 AND confidence <= 1),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_atom_id, target_atom_id, relation_type)
);

CREATE INDEX idx_atom_relations_source ON atom_relations(source_atom_id);
CREATE INDEX idx_atom_relations_target ON atom_relations(target_atom_id);
CREATE INDEX idx_atom_relations_type ON atom_relations(relation_type);

-- ============================================================================
-- 计划表
-- ============================================================================
CREATE TABLE plans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    plan_id VARCHAR(50) UNIQUE NOT NULL,
    meta JSONB NOT NULL,
    context JSONB NOT NULL,
    task_graph JSONB NOT NULL,
    quality_gates JSONB NOT NULL,
    error_handling JSONB NOT NULL,
    expected_outputs JSONB NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN (
        'created', 'scheduled', 'executing', 'completed', 'failed', 'cancelled'
    )) DEFAULT 'created',
    total_tasks INTEGER DEFAULT 0,
    completed_tasks INTEGER DEFAULT 0,
    failed_tasks INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    scheduled_at TIMESTAMP WITH TIME ZONE,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_plans_plan_id ON plans(plan_id);
CREATE INDEX idx_plans_status ON plans(status);
CREATE INDEX idx_plans_created_at ON plans(created_at);

-- ============================================================================
-- 任务表
-- ============================================================================
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id VARCHAR(50) NOT NULL,
    plan_id VARCHAR(50) NOT NULL REFERENCES plans(plan_id) ON DELETE CASCADE,
    agent VARCHAR(50) NOT NULL CHECK (agent IN (
        'Scholar_Internalizer', 'Code_Architect', 'Scenario_Validator', 
        'Knowledge_Vault', 'Strategic_Critic'
    )),
    stage_id VARCHAR(10) NOT NULL,
    input_spec JSONB NOT NULL,
    output_spec JSONB NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN (
        'pending', 'waiting_dependencies', 'ready', 'executing', 'success',
        'partial', 'failed', 'timeout', 'blocked', 'skipped'
    )) DEFAULT 'pending',
    invocation_id VARCHAR(50),
    execution_time_ms INTEGER,
    output JSONB,
    quality_check JSONB,
    errors JSONB DEFAULT '[]',
    attempt_count INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    priority INTEGER CHECK (priority >= 1 AND priority <= 5),
    timeout_seconds INTEGER,
    dependencies JSONB DEFAULT '[]',
    depends_on_tasks TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ready_at TIMESTAMP WITH TIME ZONE,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    UNIQUE(plan_id, task_id)
);

CREATE INDEX idx_tasks_task_id ON tasks(task_id);
CREATE INDEX idx_tasks_plan_id ON tasks(plan_id);
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_agent ON tasks(agent);
CREATE INDEX idx_tasks_stage_id ON tasks(stage_id);
CREATE INDEX idx_tasks_priority ON tasks(priority);
CREATE INDEX idx_tasks_creatks(created_at);

-- ============================================================================
-- 执行记录表
-- ============================================================================
CREATE TABLE executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    invocation_id VARCHAR(50) UNIQUE NOT NULL,
    task_id VARCHAR(50) NOT NULL,
    agent_request JSONB NOT NULL,
    agent_response JSONB,
    status VARCHAR(20) NOT NULL CHECK (status IN (
        'pending', 'executing', 'success', 'partial', 'failed', 'timeout'
    )),
    execution_time_ms INTEGER,
    token_usage INTEGER,
    cost_estimate DECIMAL(10,4),
    quality_score DECIMAL(4,2),
    atoms_generated INTEGER,
    error_code VARCHAR(10),
    error_details TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_executions_invocation_id ON executions(invocation_id);
CREATE INDEX idx_executions_task_id ON executions(task_id);
CREATE INDEX idx_executions_status ON executions(status);
CREATE INDEX idx_executions_created_at ON executions(created_at);

-- ============================================================================
-- 质量检查表
-- ============================================================================
CREATE TABLE quality_checks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    gate_id VARCHAR(50) NOT NULL,
    task_id VARCHAR(50),
    stage_id VARCHAR(10),
    plan_id VARCHAR(50) REFERENCES plans(plan_id) ON DELETE CASCADE,
    gate_type VARCHAR(20) NOT NULL CHECK (gate_type IN (
        'pre_execution', 'in_execution', 'post_execution', 'integration'
    )),
    check_config JSONB NOT NULL,
    check_status VARCHAR(20) NOT NULL CHECK (check_status IN (
        'passed', 'failed', 'warning', 'skipped'
    )),
    check_results JSONB NOT NULL,
    failure_action VARCHAR(20),
    action_executed VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    executed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_quality_checks_gate_id ON quality_checks(gate_id);
CREATE INDEX idx_quality_checks_task_id ON quality_checks(task_id);
CREATE INDEX idx_quality_checks_plan_id ON quality_checks(plan_id);
CREATE INDEX idx_quality_checks_gate_type ON quality_checks(gate_type);
CREATE INDEX idx_quality_checks_check_status ON quality_checks(check_status);
CREATE INDEX idx_quality_checks_created_at ON quality_checks(created_at);

-- ===================================================================
-- 错误记录表
-- ============================================================================
CREATE TABLE error_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    error_id VARCHAR(50) UNIQUE NOT NULL,
    error_code VARCHAR(10) NOT NULL,
    error_type VARCHAR(30) NOT NULL CHECK (error_type IN (
        'system', 'agent', 'data', 'validation', 'quality', 'resource', 'external'
    )),
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('critical', 'error', 'warning', 'info')),
    recoverability VARCHAR(30) NOT NULL CHECK (recoverability IN (
        'recoverable', 'partially_recoverable', 'non_recoverable'
    )),
    component VARCHAR(50),
    task_id VARCHAR(50),
    plan_id VARCHAR(50) REFERENCES plans(plan_id) ON DELETE SET NULL,
    invocation_id VARCHAR(50),
    error_message TEXT NOT NULL,
    error_context JSONB,
    stack_trace TEXT,
    handling_strategy VARCHAR(30),
    handling_status VARCHAR(20) CHECK (handling_status IN (
        'detected', 'handling', 'resolved', 'escalated', 'ignored'
    )),
    resolution_notes TEXT,
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE
);

CREATE INDdx_error_records_error_id ON error_records(error_id);
CREATE INDEX idx_error_records_error_code ON error_records(error_code);
CREATE INDEX idx_error_records_severity ON error_records(severity);
CREATE INDEX idx_error_records_component ON error_records(component);
CREATE INDEX idx_error_records_task_id ON error_records(task_id);
CREATE INDEX idx_error_records_detected_at ON error_records(detected_at);

-- ============================================================================
-- 错误处理日志表
-- ============================================================================
CREATE TABLE error_handling_log    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    error_id VARCHAR(50) NOT NULL,
    old_status VARCHAR(20),
    new_status VARCHAR(20),
    action_taken VARCHAR(30),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_error_handling_logs_error_id ON error_handling_logs(error_id);
CREATE INDEX idx_error_handling_logs_timestamp ON error_handling_logs(timestamp);

-- ============================================================================
-- 资源使用表
-- ============================================================================
CREATE TABLE rese (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resource_type VARCHAR(30) NOT NULL CHECK (resource_type IN (
        'gpu_memory', 'cpu_usage', 'disk_space', 'api_quota', 'network_bandwidth'
    )),
    used_amount DECIMAL(12,2),
    total_amount DECIMAL(12,2),
    usage_percent DECIMAL(5,2),
    component VARCHAR(50),
    task_id VARCHAR(50)
);

CREATE INDEX idx_resource_usage_timestamp ON resource_usage(timestamp);
CREATE INDEX idx_resource_usage_resource_type ON resource_usage(resource_type);
CREATE INDEX idx_resource_usage_compone resource_usage(component);

-- ============================================================================
-- 知识库统计表
-- ============================================================================
CREATE TABLE knowledge_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    snapshot_date DATE UNIQUE NOT NULL DEFAULT CURRENT_DATE,
    total_atoms INTEGER NOT NULL DEFAULT 0,
    math_atoms INTEGER DEFAULT 0,
    code_atoms INTEGER DEFAULT 0,
    scenario_atoms INTEGER DEFAULT 0,
    strategic_atoms INTEGER DEFAULT 0,
    avg_quality_score DECIMAL(4,2),
    quality_distribution JSONB,
    domain_coverage JSONB,
    reontributions JSONB,
    atoms_added_today INTEGER DEFAULT 0,
    atoms_updated_today INTEGER DEFAULT 0
);

CREATE INDEX idx_knowledge_stats_snapshot_date ON knowledge_stats(snapshot_date);

-- ============================================================================
-- 视图
-- ============================================================================
CREATE VIEW task_execution_status AS
SELECT 
    p.plan_id, p.status as plan_status, t.task_id, t.agent, t.stage_id,
    t.status as task_status, t.priority, t.attempt_count, t.execution_time_ms,
    t.created_at, t.started_at, t.completed_at,
    CASE 
        WHEN t.status = 'success' THEN 'completed'
        WHEN t.status IN ('failed', 'timeout', 'blocked') THEN 'failed'
        ELSE 'in_progress'
    END as execution_phase
FROM tasks t JOIN plans p ON t.plan_id = p.plan_id;

CREATE VIEW quality_gate_status AS
SELECT 
    qc.gate_id, qc.gate_type, qc.task_id, qc.plan_id, qc.check_status,
    qc.failure_action, qc.action_executed, qc.created_at, qc.executed_at,
    t.status as task_status, p.status as plan_status
FROM quality_checks qc
LEFT JOIN tasks t ON qc.task_id = t.task_id
LEFT JOIN plans p ON qc.plan_id = p.plan_id;

CREATE VIEW error_statistics AS
SELECT 
    error_code, error_type, severity, component, COUNT(*) as error_count,
    AVG(EXTRACT(EPOCH FROM (resolved_at - detected_at))) as avg_resolution_time_seconds,
    MIN(detected_at) as first_occurrence, MAX(detected_at) as last_occurrence
FROM error_records
GROUP BY error_code, error_type, severity, component;

-- ============================================================================
-- 函数和触发器
-- ============================================================================
CREATE OR REPLACE FUNCTION update_atom_quality_score()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.metadata ? 'quality_assessment' THEN
        NEW.quality_score := (NEW.metadata->'quality_assessment'->>'score')::DECIMAL;
        NEW.quality_grade := (NEW.metadata->'quality_assessment'->>'grade')::CHAR;
    END IF;
    NEW.updated_at := CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_atom_quality_score
    BEFORE INSERT OR UPDATE ON atoms
    FOR EACH ROW EXECUTE FUNCTION update_atom_quality_score();

CREATE OR REPLACE FUNCTION update_plan_statistics()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'UPDATE' AND OLD.status != NEW.status THEN
        UPDATE plans SET 
            completed_tasks = (SELECT COUNT(*) FROM tasks WHERE plan_id = NEW.plan_id AND status IN ('success', 'partial')),
            failed_tasks = (SELECT COUNT(*) FROM tasks WHERE plan_id = NEW.plan_id AND status IN ('failed', 'timeout', 'blocked')),
            total_tasks = (SELECT COUNT(*) FROM tasks WHERE plan_id = NEW.plan_id)
        WHERE plan_id = NEW.plan_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_plan_statistics
    AFTER INSERT OR UPDATE ON tasks
    FOR EACH ROW EXECUTE FUNCTION update_plan_statistics();

CREATE OR REPLACE FUNCTION log_error_handling()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'UPDATE' AND OLD.handling_status IS DISTINCT FROM NEW.handling_status THEN
        INSERT INTO error_handling_logs (error_id, old_status, new_status, action_taken, timestamp)
        VALUES (NEW.error_id, OLD.handling_status, NEW.handling_status, NEW.handling_strategy, CURRENT_TIMESTAMP);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_log_error_handling
    AFTER UPDATE ON error_records
    FOR EACH ROW EXECUTE FUNCTION log_error_handling();

-- 初始数据
INSERT INTO knowledge_stats (snapshot_date, total_atoms, avg_quality_score, quality_distribution)
VALUES (CURRENT_DATE, 0, 0.0, '{"A": 0, "B": 0, "C": 0, "D": 0}')
ON CONFLICT (snapshot_date) DO NOTHING;
