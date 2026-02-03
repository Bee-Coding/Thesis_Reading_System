"""
Dispatcher调度器模块 - 核心调度逻辑
"""
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    WAITING_DEPENDENCIES = "waiting_dependencies"
    READY = "ready"
    EXECUTING = "executing"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class StageStatus(Enum):
    """阶段状态"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskResult:
    """任务执行结果"""
    task_id: str
    status: TaskStatus
    output: Optional[Dict[str, Any]] = None
    quality_check: Optional[Dict[str, Any]] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    execution_time_ms: int = 0
    invocation_id: Optional[str] = None


@dataclass
class StageResult:
    """阶段执行结果"""
    stage_id: str
    status: StageStatus
    task_results: Dict[str, TaskResult] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class Dispatcher:
    """
    调度器 - 负责解析任务包并协调Agent执行
    """
    
    def __init__(
        self,
        agent_invoker: Any = None,
        quality_gate: Any = None,
        data_transfer: Any = None,
        error_handler: Any = None,
        db_repository: Any = None,
    ):
        self.agent_invoker = agent_invoker
        self.quality_gate = quality_gate
        self.data_transfer = data_transfer
        self.error_handler = error_handler
        self.db_repository = db_repository
        
        self.plan: Dict[str, Any] = {}
        self.plan_id: str = ""
        self.stage_results: Dict[str, StageResult] = {}
        self.task_outputs: Dict[str, Any] = {}
    
    def load_plan(self, plan_path: str) -> Dict[str, Any]:
        """加载任务包"""
        with open(plan_path, 'r', encoding='utf-8') as f:
            self.plan = json.load(f)
        self.plan_id = self.plan.get("plan_id", "")
        return self.plan
    
    def load_plan_from_dict(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """从字典加载任务包"""
        self.plan = plan
        self.plan_id = plan.get("plan_id", "")
        return self.plan
    
    async def execute(self) -> Dict[str, Any]:
        """执行任务包"""
        if not self.plan:
            raise ValueError("No plan loaded. Call load_plan() first.")
        
        print(f"[Dispatcher] Starting execution of plan: {self.plan_id}")
        
        task_graph = self.plan.get("task_graph", {})
        stages = task_graph.get("stages", [])
        
        for stage in stages:
            stage_id = stage.get("stage_id", "")
            depends_on = stage.get("depends_on", [])
            
            if not self._check_stage_dependencies(depends_on):
                print(f"[Dispatcher] Stage {stage_id} blocked due to failed dependencies")
                self.stage_results[stage_id] = StageResult(
                    stage_id=stage_id,
                    status=StageStatus.FAILED
                )
                continue
            
            stage_result = await self._execute_stage(stage)
            self.stage_results[stage_id] = stage_result
            
            if stage_result.status == StageStatus.FAILED:
                if not self._should_continue_on_failure(stage):
                t(f"[Dispatcher] Stopping execution due to stage {stage_id} failure")
                    break
        
        return self._generate_execution_summary()
    
    def _check_stage_dependencies(self, depends_on: List[str]) -> bool:
        """检查阶段依赖是否满足"""
        for stage_id in depends_on:
            if stage_id not in self.stage_results:
                return False
            if self.stage_results[stage_id].status != StageStatus.COMPLETED:
                return False
        return True
    
    def _should_continue_on_failure(self, stage: Dict[str, Any]) -> bool:
        """判断阶段失败后是否继续执行"""
        return False
    
 async def _execute_stage(self, stage: Dict[str, Any]) -> StageResult:
        """执行单个阶段"""
        stage_id = stage.get("stage_id", "")
        stage_name = stage.get("stage_name", "")
        execution_mode = stage.get("execution_mode", "sequential")
        tasks = stage.get("tasks", [])
        
        print(f"[Dispatcher] Executing stage: {stage_id} ({stage_name}) - mode: {execution_mode}")
        
        result = StageResult(
            stage_id=stage_id,
            status=StageStatus.EXECUTING,
            start_time=datetime.now()
       
        task_results: Dict[str, TaskResult] = {}
        
        try:
            if execution_mode == "parallel":
                task_results = await self._execute_tasks_parallel(tasks)
            else:
                task_results = await self._execute_tasks_sequential(tasks)
            
            result.task_results = task_results
            
            failed_count = sum(
                1 for r in task_results.values() 
                if r.status in [TaskStatus.FAILED, TaskStatus.TIMEOUT, TaskStatus.BLOCKED]
            )
            
            if failed_count == 0:
                result.status = StageStatus.COMPLETED
            elif failed_count < len(task_results):
                result.status = StageStatus.COMPLETED
            else:
                result.status = StageStatus.FAILED
                
        except Exception as e:
            print(f"[Dispatcher] Stage {stage_id} failed with error: {e}")
            result.status = StageStatus.FAILED
        
        result.end_time = datetime.now()
        return result
    
    async def _execute_tasks_parallel(self, tasks: List[Dict[str, Any]]) -> Dict[str, TaskResult]:
        """并行执行任务"""
        results = await asyncio.gather(
            *[self._execute_task(task) for task in tasks],
            return_exceptions=True
        )
        
        task_results: Dict[str, TaskResult] = {}
        for task, result in zip(tasks, results):
            task_id = task.get("task_id", "")
            if isinstance(result, Exception):
                task_results[task_id] = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    errors=[{"error_type": "exception", "message": str(result)}]
                )
            elif isinstance(result, TaskResult):
                task_results[task_id] = result
        
        returresults
    
    async def _execute_tasks_sequential(self, tasks: List[Dict[str, Any]]) -> Dict[str, TaskResult]:
        """顺序执行任务"""
        task_results: Dict[str, TaskResult] = {}
        
        for task in tasks:
            task_id = task.get("task_id", "")
            try:
                result = await self._execute_task(task)
                task_results[task_id] = result
            except Exception as e:
                task_results[task_id] = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    errors=[{"error_type": "exception", "message": str(e)}]
                )
        
        return task_results
    
    async def _execute_task(self, task: Dict[str, Any]) -> TaskResult:
        """执行单个任务"""
        task_id = task.get("task_id", "")
        agent_name = task.get("agent", "")
        
        print(f"[Dispatcher] Executing task: {task_id} with agent: {agent_name}")
        
        start_time = datetime.now()
        
        # 1. 预执行质量门检查
        if self.quality_gate:
            pre_check = await self._run_pre_execution_gate(task)
            if not pre_check.get("passed", True):
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.BLOCKED,
                    errors=[{"error_type": "quality_gate", "message": "Pre-execution check failed"}]
                )
        
        # 2. 解析依赖数据
        input_data = await self._resolve_task_dependencies(task)
        
        # 3. 调用Agent
        response: Dict[str, Any] = {}
        try:
            if self.agent_invoker:
                response = await self.agent_invoker.invoke(
                    agent_name=agent_name,
                    task=task,
                    input_data=input_data,
                    plan_context=self.plan.get("context", {})
                )
            else:
                response = self._mock_agent_response(task)
        except Exception as e:
            if self.error_handler:
                recovery = await self.error_handler.handle(e, task)
                if recovery.get("retry"):
                    return await self._execute_task(task)
            
            return TaskResult(
                task_id=task_id,
                status=TaAILED,
                errors=[{"error_type": "agent_error", "message": str(e)}]
            )
        
        # 4. 后执行质量门检查
        if self.quality_gate:
            post_check = await self._run_post_execution_gate(task, response)
            if not post_check.get("passed", True) and post_check.get("retry"):
                return await self._execute_task(task)
        
        # 5. 保存任务输出
        self.task_outputs[task_id] = response.get("output", {})
        
        end_time = datetime.now()
        execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        status = TaskStatus.     if response.get("status") == "partial":
            status = TaskStatus.PARTIAL
        elif response.get("status") == "failed":
            status = TaskStatus.FAILED
        
        return TaskResult(
            task_id=task_id,
            status=status,
            output=response.get("output"),
            quality_check=response.get("quality_check"),
            errors=response.get("errors", []),
            execution_time_ms=execution_time_ms,
            invocation_id=response.get("invocation_id")
        )
    
    async def _resolve_task_dependencies(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """据"""
        input_spec = task.get("input_spec", {})
        dependencies = input_spec.get("dependencies", [])
        
        resolved_data: Dict[str, Any] = {
            "sources": input_spec.get("sources", []),
            "dependencies": {}
        }
        
        for dep in dependencies:
            dep_task_id = dep.get("task_id", "")
            output_ref = dep.get("output_ref", "")
            
            if dep_task_id in self.task_outputs:
                if self.data_transfer:
                    resolved = await self.data_transfer.resolve_reference(
                        self.task_outputs[dep_task_id],
                        output_ref
                    )
                else:
                    resolved = self.task_outputs[dep_task_id]
                
                resolved_data["dependencies"][dep_task_id] = resolved
        
        return resolved_data
    
    async def _run_pre_execution_gate(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """运行预执行质量门"""
        task_id = task.get("task_id", "")
        gates = self.plan.get("quality_gates", {}).get("pre_execution", [])
        
        for gate in gates:
            if gate.get("task_id") == task_id:
                return await self.quality_gate.check(gate, task, None)
        
        return {"passed": True}
    
    async def _run_post_execution_gate(self, task: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
        """运行后执行质量门"""
        task_id = task.get("task_id", "")
        gates = self.plan.get("quality_gates", {}).get("post_execution", [])
        
        for gate in gates:
            if gate.get("task_id") == task_id:
                re self.quality_gate.check(gat response)
        
        return {"passed": True}
    
    def _mock_agent_response(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """模拟Agent响应（用于测试）"""
        return {
            "status": "success",
            "invocation_id": f"INV_MOCK_{task.get('task_id', '')}",
            "output": {
                "atoms": [],
                "summary": f"Mock response for task {task.get('task_id', '')}"
            },
            "quality_check": {
                "atoms_generated": 0,
                "avg_quality_score": 0.8,
                "grades": {"A": 0, "B": 0, "C": 0, "D": 0},
                "issues": []
            },
            "errors": []
        }
    
    def _generate_execution_summary(self) -> Dict[str, Any]:
        """生成执行摘要"""
        total_tasks = 0
        completed_tasks = 0
        failed_tasks = 0
        
        for stage_result in self.stage_results.values():
            for task_result in stage_result.task_results.values():
                total_tasks += 1
                if task_result.status == TaskStatus.SUCCESS:
                    completed_tasks += 1
                elif task_result.status in [TaskStatus.FAILED, TaskStatusEOUT, TaskStatus.BLOCKED]:
                    failed_tasks += 1
        
        overall_status = "completed"
        if failed_tasks > 0:
            overall_status = "partial" if completed_tasks > 0 else "failed"
        
        stage_summary = {}
        for stage_id, result in self.stage_results.items():
            stage_summary[stage_id] = {
                "status": result.status.value,
                "task_count": len(result.task_results),
                "start_time": result.start_time.isoformat() if result.start_time else None,
                "end_time": result.end_time.isoformat() if result.end_time else None,
            }
        
        return {
            "plan_id": self.plan_id,
            "status": overall_status,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "stage_results": stage_summary,
            "task_outputs": self.task_outputs
        }
