-- Thesis Reading System 数据库Schema
-- PostgreSQL 14+

-- 创建扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- 知识原子表
CREATE TABLE IF NOT EXISTS atoms (
    atom_id VARCHAR(64) PRIMARY KEY,
    paper_id VARCHAR(64) NOT NULL,
    atom_type VARCHAR(32) NOT NULL,
    content JSONB NOT NULL,
    quality_grade CHAR(1) NOT NULL CHECK (quality_grade IN ('A', 'B', 'C', 'D')),
    quality_score DECIMAL(3,2) NOT NULL CHECK (quality_score >= 0 AND quality_score <= 1),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 执行计划表
CREATE TABLE IF NOT EXISTS plans (
    plan_id VARCHAR(64) PRIMARY KEY,
    paper_id VARCHAR(64) NOT NULL,
    execution_mode VARCHAR(32) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    task_graph JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 执行记录表
CREATE TABLE IF NOT EXISTS executions (
    execution_id VARCHAR(64) PRIMARY KEY,
    plan_id VARCHAR(64) NOT NULL REFERENCES plans(plan_id),
    stage_id VARCHAR(64) NOT NULL,
    task_id VARCHAR(64) NOT NULL,
    agent_name VARCHAR(64) NOT NULL,
    status VARCHAR(32) NOT NULL,
    input_data JSONB DEFAULT '{}',
    output_data JSONB DEFAULT '{}',
    errors JSONB DEFAULT '[]',
    execution_time_ms INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 论文元数据表
CREATE TABLE IF NOT EXISTS papers (
    paper_id VARCHAR(64) PRIMARY KEY,
    title TEXT NOT NULL,
    authors JSONB DEFAULT '[]',
    abstract TEXT,
    source_path TEXT,
    source_type VARCHAR(32),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Agent调用日志表
CREATE TABLE IF NOT EXISTS agent_invocations (
    invocation_id VARCHAR(64) PRIMARY KEY,
    execution_id VARCHAR(64) REFERENCES executions(execution_id),
    agent_name VARCHAR(64) NOT NULL,
    model_name VARCHAR(64),
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    latency_ms INTEGER DEFAULT 0,
    status VARCHAR(32) NOT NULL,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 质量检查记录表
CREATE TABLE IF NOT EXISTS quality_checks (
    check_id VARCHAR(64) PRIMARY KEY,
    execution_id VARCHAR(64) REFERENCES executions(execution_id),
    check_type VARCHAR(32) NOT NULL,
    gate_name VARCHAR(64),
    passed BOOLEAN NOT NULL,
    score DECIMAL(3,2),
    details JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_atoms_paper_id ON atoms(paper_id);
CREATE INDEX IF NOT EXISTS idx_atoms_type ON atoms(atom_type);
CREATE INDEX IF NOT EXISTS idx_atoms_quality ON atoms(quality_grade, quality_score DESC);
CREATE INDEX IF NOT EXISTS idx_atoms_content_gin ON atoms USING GIN (content);

CREATE INDEX IF NOT EXISTS idx_plans_paper_id ON plans(paper_id);
CREATE INDEX IF NOT EXISTS idx_plans_status ON plans(status);

CREATE INDEX IF NOT EXISTS idx_executions_plan_id ON executions(plan_id);
CREATE INDEX IF NOT EXISTS idx_executions_task_id ON executions(task_id);
CREATE INDEX IF NOT EXISTS idx_executions_status ON executions(status);

CREATE INDEX IF NOT EXISTS idx_agent_invocations_execution ON agent_invocations(execution_id);
CREATE INDEX IF NOT EXISTS idx_agent_invocations_agent ON agent_invocations(agent_name);

CREATE INDEX IF NOT EXISTS idx_quality_checks_execution ON quality_checks(execution_id);

-- 更新时间触发器函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 应用触发器
DROP TRIGGER IF EXISTS update_atoms_updated_at ON atoms;
CREATE TRIGGER update_atoms_updated_at
    BEFORE UPDATE ON atoms
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_plans_updated_at ON plans;
CREATE TRIGGER update_plans_updated_at
    BEFORE UPDATE ON plans
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_papers_updated_at ON papers;
CREATE TRIGGER update_papers_updated_at
    BEFORE UPDATE ON papers
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- 视图：执行摘要
CREATE OR REPLACE VIEW execution_summary AS
SELECT 
    p.plan_id,
    p.paper_id,
    p.execution_mode,
    p.status as plan_status,
    COUNT(e.execution_id) as total_tasks,
    COUNT(CASE WHEN e.status = 'success' THEN 1 END) as completed_tasks,
    COUNT(CASE WHEN e.status IN ('failed', 'timeout', 'blocked') THEN 1 END) as failed_tasks,
    SUM(e.execution_time_ms) as total_execution_time_ms,
    p.created_at,
    p.updated_at
FROM plans p
LEFT JOIN executions e ON p.plan_id = e.plan_id
GROUP BY p.plan_id;

-- 视图：原子质量统计
CREATE OR REPLACE VIEW atom_quality_stats AS
SELECT 
    paper_id,
    atom_type,
    COUNT(*) as atom_count,
    AVG(quality_score) as avg_quality_score,
    COUNT(CASE WHEN quality_grade = 'A' THEN 1 END) as grade_a_count,
    COUNT(CASE WHEN quality_grade = 'B' THEN 1 END) as grade_b_count,
    COUNT(CASE WHEN quality_grade = 'C' THEN 1 END) as grade_c_count,
    COUNT(CASE WHEN quality_grade = 'D' THEN 1 END) as grade_d_count
FROM atoms
GROUP BY paper_id, atom_type;

-- 授权（根据实际用户调整）
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO thesis_app;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO thesis_app;
