"""
上下文管理器模块 - 管理执行上下文和状态
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """执行上下文"""
    plan_id: str
    paper_id: str
    execution_mode: str
    paper_info: Dict[str, Any] = field(default_factory=dict)
    global_variables: Dict[str, Any] = field(default_factory=dict)
    task_outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class ContextManager:
    """上下文管理器 - 管理执行过程中的上下文状态"""
    
    def __init__(self):
        self._contexts: Dict[str, ExecutionContext] = {}
        self._current_context_id: Optional[str] = None
    
    def create_context(
        self,
        plan_id: str,
        paper_id: str,
        execution_mode: str,
        paper_info: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ExecutionContext:
        """创建新的执行上下文"""
        context = ExecutionContext(
            plan_id=plan_id,
            paper_id=paper_id,
            execution_mode=execution_mode,
            paper_info=paper_info or {},
            metadata=metadata or {}
        )
        self._contexts[plan_id] = context
        self._current_context_id = plan_id
        logger.info(f"Created execution context for plan: {plan_id}")
        return context
    
    def get_context(self, plan_id: Optional[str] = None) -> Optional[ExecutionContext]:
        """获取执行上下文"""
        context_id = plan_id or self._current_context_id
        if context_id:
            return self._contexts.get(context_id)
        return None
    
    def set_current_context(self, plan_id: str) -> bool:
        """设置当前上下文"""
        if plan_id in self._contexts:
            self._current_context_id = plan_id
            return True
        return False
    
    def store_task_output(
        self,
        plan_id: str,
        task_id: str,
        output: Dict[str, Any]
    ) -> None:
        """存储任务输出"""
        context = self._contexts.get(plan_id)
        if context:
            context.task_outputs[task_id] = output
            logger.debug(f"Stored output for task {task_id} in plan {plan_id}")
    
    def get_task_output(
        self,
        plan_id: str,
        task_id: str
    ) -> Optional[Dict[str, Any]]:
        """获取任务输出"""
        context = self._contexts.get(plan_id)
        if context:
            return context.task_outputs.get(task_id)
        return None
    
    def set_variable(
        self,
        plan_id: str,
        name: str,
        value: Any
    ) -> None:
        """设置全局变量"""
        context = self._contexts.get(plan_id)
        if context:
            context.global_variables[name] = value
    
    def get_variable(
        self,
        plan_id: str,
        name: str,
        default: Any = None
    ) -> Any:
        """获取全局变量"""
        context = self._contexts.get(plan_id)
        if context:
            return context.global_variables.get(name, default)
        return default
    
    def get_all_task_outputs(self, plan_id: str) -> Dict[str, Any]:
        """获取所有任务输出"""
        context = self._contexts.get(plan_id)
        if context:
            return context.task_outputs.copy()
        return {}
    
    def build_task_input(
        self,
        plan_id: str,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """为任务构建输入数据"""
        context = self._contexts.get(plan_id)
        if not context:
            return {"sources": [], "dependencies": {}}
        
        input_spec = task.get("input_spec", {})
        sources = input_spec.get("sources", [])
        dependencies = input_spec.get("dependencies", [])
        
        resolved_deps = {}
        for dep in dependencies:
            dep_task_id = dep.get("task_id", "")
            output_ref = dep.get("output_ref", "")
            
            if dep_task_id in context.task_outputs:
                dep_output = context.task_outputs[dep_task_id]
                if output_ref:
                    # 简单的引用解析
                    parts = output_ref.split(".")
                    value = dep_output
                    for part in parts:
                        if isinstance(value, dict):
                            value = value.get(part)
                        else:
                            value = None
                            break
                    resolved_deps[dep_task_id] = value
                else:
                    resolved_deps[dep_task_id] = dep_output
        
        return {
            "sources": sources,
            "dependencies": resolved_deps,
            "context": {
                "paper_info": context.paper_info,
                "execution_mode": context.execution_mode,
                "variables": context.global_variables
            }
        }
    
    def cleanup_context(self, plan_id: str) -> None:
        """清理执行上下文"""
        if plan_id in self._contexts:
            del self._contexts[plan_id]
            if self._current_context_id == plan_id:
                self._current_context_id = None
            logger.info(f"Cleaned up context for plan: {plan_id}")
    
    def get_context_summary(self, plan_id: str) -> Dict[str, Any]:
        """获取上下文摘要"""
        context = self._contexts.get(plan_id)
        if not context:
            return {}
        
        return {
            "plan_id": context.plan_id,
            "paper_id": context.paper_id,
            "execution_mode": context.execution_mode,
            "task_count": len(context.task_outputs),
            "variable_count": len(context.global_variables),
            "created_at": context.created_at.isoformat()
        }
