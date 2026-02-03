"""
数据库仓储层 - 提供数据访问抽象
"""
import json
from datetime import datetime
from typing import Optional, Any, Dict, List
from dataclasses import dataclass, asdict
import logging

from .connection import DatabaseConnection, get_connection

logger = logging.getLogger(__name__)


@dataclass
class AtomRecord:
    """知识原子记录"""
    atom_id: str
    paper_id: str
    atom_type: str
    content: Dict[str, Any]
    quality_grade: str
    quality_score: float
    created_at: datetime = None
    updated_at: datetime = None
    metadata: Dict[str, Any] = None


@dataclass
class PlanRecord:
    """执行计划记录"""
    plan_id: str
    paper_id: str
    execution_mode: str
    status: str
    task_graph: Dict[str, Any]
    created_at: datetime = None
    updated_at: datetime = None
    metadata: Dict[str, Any] = None


@dataclass
class ExecutionRecord:
    """执行记录"""
    execution_id: str
    plan_id: str
    stage_id: str
    task_id: str
    agent_name: str
    status: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    errors: List[Dict[str, Any]]
    execution_time_ms: int
    created_at: datetime = None


class AtomRepository:
    """知识原子仓储"""
    
    def __init__(self, connection: Optional[DatabaseConnection] = None):
        self.conn = connection or get_connection()
    
    async def create(self, atom: AtomRecord) -> str:
        """创建知识原子"""
        query = """
            INSERT INTO atoms (atom_id, paper_id, atom_type, content, quality_grade, 
                              quality_score, metadata, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), NOW())
            RETURNING atom_id
        """
        return await self.conn.fetchval(
            query,
            atom.atom_id,
            atom.paper_id,
            atom.atom_type,
            json.dumps(atom.content),
            atom.quality_grade,
            atom.quality_score,
            json.dumps(atom.metadata or {})
        )
    
    async def get_by_id(self, atom_id: str) -> Optional[AtomRecord]:
        """根据ID获取知识原子"""
        query = "SELECT * FROM atoms WHERE atom_id = $1"
        row = await self.conn.fetchrow(query, atom_id)
        if row:
            return self._row_to_record(row)
        return None
    
    async def get_by_paper(self, paper_id: str) -> List[AtomRecord]:
        """获取论文的所有知识原子"""
        query = "SELECT * FROM atoms WHERE paper_id = $1 ORDER BY created_at"
        rows = await self.conn.fetch(query, paper_id)
        return [self._row_to_record(row) for row in rows]
    
    async def get_by_type(self, paper_id: str, atom_type: str) -> List[AtomRecord]:
        """获取指定类型的知识原子"""
        query = "SELECT * FROM atoms WHERE paper_id = $1 AND atom_type = $2 ORDER BY created_at"
        rows = await self.conn.fetch(query, paper_id, atom_type)
        return [self._row_to_record(row) for row in rows]
    
    async def update(self, atom: AtomRecord) -> bool:
        """更新知识原子"""
        query = """
            UPDATE atoms SET 
                content = $2, quality_grade = $3, quality_score = $4, 
                metadata = $5, updated_at = NOW()
            WHERE atom_id = $1
        """
        result = await self.conn.execute(
            query,
            atom.atom_id,
            json.dumps(atom.content),
            atom.quality_grade,
            atom.quality_score,
            json.dumps(atom.metadata or {})
        )
        return "UPDATE 1" in result
    
    async def delete(self, atom_id: str) -> bool:
        """删除知识原子"""
        query = "DELETE FROM atoms WHERE atom_id = $1"
        result = await self.conn.execute(query, atom_id)
        return "DELETE 1" in result
    
    async def search(self, paper_id: str, keyword: str) -> List[AtomRecord]:
        """搜索知识原子"""
        query = """
            SELECT * FROM atoms 
            WHERE paper_id = $1 AND content::text ILIKE $2
            ORDER BY quality_score DESC
        """
        rows = await self.conn.fetch(query, paper_id, f"%{keyword}%")
        return [self._row_to_record(row) for row in rows]
    
    def _row_to_record(self, row: Dict[str, Any]) -> AtomRecord:
        """将数据库行转换为记录对象"""
        content = row.get("content")
        if isinstance(content, str):
            content = json.loads(content)
        
        metadata = row.get("metadata")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        
        return AtomRecord(
            atom_id=row["atom_id"],
            paper_id=row["paper_id"],
            atom_type=row["atom_type"],
            content=content,
            quality_grade=row["quality_grade"],
            quality_score=row["quality_score"],
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
            metadata=metadata
        )


class PlanRepository:
    """执行计划仓储"""
    
    def __init__(self, connection: Optional[DatabaseConnection] = None):
        self.conn = connection or get_connection()
    
    async def create(self, plan: PlanRecord) -> str:
        """创建执行计划"""
        query = """
            INSERT INTO plans (plan_id, paper_id, execution_mode, status, 
                              task_graph, metadata, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, NOW(), NOW())
            RETURNING plan_id
        """
        return await self.conn.fetchval(
            query,
            plan.plan_id,
            plan.paper_id,
            plan.execution_mode,
            plan.status,
            json.dumps(plan.task_graph),
            json.dumps(plan.metadata or {})
        )
    
    async def get_by_id(self, plan_id: str) -> Optional[PlanRecord]:
        """根据ID获取执行计划"""
        query = "SELECT * FROM plans WHERE plan_id = $1"
        row = await self.conn.fetchrow(query, plan_id)
        if row:
            return self._row_to_record(row)
        return None
    
    async def get_by_paper(self, paper_id: str) -> List[PlanRecord]:
        """获取论文的所有执行计划"""
        query = "SELECT * FROM plans WHERE paper_id = $1 ORDER BY created_at DESC"
        rows = await self.conn.fetch(query, paper_id)
        return [self._row_to_record(row) for row in rows]
    
    async def update_status(self, plan_id: str, status: str) -> bool:
        """更新执行计划状态"""
        query = "UPDATE plans SET status = $2, updated_at = NOW() WHERE plan_id = $1"
        result = await self.conn.execute(query, plan_id, status)
        return "UPDATE 1" in result
    
    def _row_to_record(self, row: Dict[str, Any]) -> PlanRecord:
        """将数据库行转换为记录对象"""
        task_graph = row.get("task_graph")
        if isinstance(task_graph, str):
            task_graph = json.loads(task_graph)
        
        metadata = row.get("metadata")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        
        return PlanRecord(
            plan_id=row["plan_id"],
            paper_id=row["paper_id"],
            execution_mode=row["execution_mode"],
            status=row["status"],
            task_graph=task_graph,
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
            metadata=metadata
        )


class ExecutionRepository:
    """执行记录仓储"""
    
    def __init__(self, connection: Optional[DatabaseConnection] = None):
        self.conn = connection or get_connection()
    
    async def create(self, record: ExecutionRecord) -> str:
        """创建执行记录"""
        query = """
            INSERT INTO executions (execution_id, plan_id, stage_id, task_id, 
                                   agent_name, status, input_data, output_data, 
                                   errors, execution_time_ms, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW())
            RETURNING execution_id
        """
        return await self.conn.fetchval(
            query,
            record.execution_id,
            record.plan_id,
            record.stage_id,
            record.task_id,
            record.agent_name,
            record.status,
            json.dumps(record.input_data),
            json.dumps(record.output_data),
            json.dumps(record.errors),
            record.execution_time_ms
        )
    
    async def get_by_plan(self, plan_id: str) -> List[ExecutionRecord]:
        """获取计划的所有执行记录"""
        query = "SELECT * FROM executions WHERE plan_id = $1 ORDER BY created_at"
        rows = await self.conn.fetch(query, plan_id)
        return [self._row_to_record(row) for row in rows]
    
    async def get_by_task(self, plan_id: str, task_id: str) -> Optional[ExecutionRecord]:
        """执行记录"""
        query = "SELECT * FROM executions WHERE plan_id = $1 AND task_id = $2"
        row = await self.conn.fetchrow(query, plan_id, task_id)
        if row:
            return self._row_to_record(row)
        return None
    
    async def get_failed_executions(self, plan_id: str) -> List[ExecutionRecord]:
        """获取失败的执行记录"""
        query = """
            SELECT * FROM executions 
            WHERE plan_id = $1 AND status IN ('failed', 'timeout', 'blocked')
            ORDER BY created_at
        """
        rows = await self.conn.fetch(query, plan_id)
        return [self._row_to_record(row) for row in rows]
    
    def _row_to_record(self, row: Dict[str, Any]) -> ExecutionRecord:
        """将数据库行转换为记录对象"""
        input_data = row.get("input_data")
        if isinstance(input_data, str):
            input_data = json.loads(input_data)
        
        output_data = row.get("output_data")
        if isinstance(output_data, str):
            output_data = json.loads(output_data)
        
        errors = row.get("errors")
        if isinstance(errors, str):
            errors = json.loads(errors)
        
        return ExecutionRecord(
            execution_id=row["execution_id"],
            plan_id=row["plan_id"],
            stage_id=row["stage_id"],
            task_id=row["task_id"],
            agent_name=row["agent_name"],
            status=row["status"],
            input_data=input_data,
            output_data=output_data,
            errors=errors or [],
            execution_time_ms=row["execution_time_ms"],
            created_at=row.get("created_at")
        )
