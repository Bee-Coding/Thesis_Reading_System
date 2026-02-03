"""
Agent调用器模块 - 协调Agent调用流程
"""
import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from .llm_client import LLMClient, LLMResponse, create_llm_client
from .prompt_builder import PromptBuilder
from ..config.settings import settings, AgentType, AGENT_LLM_PARAMS, MODE_ADJUSTMENTS

logger = logging.getLogger(__name__)


class AgentInvoker:
    """Agent调用器 - 负责调用Agent并处理响应"""
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        prompt_builder: Optional[PromptBuilder] = None
    ):
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder or PromptBuilder(settings.agents_dir)
        self._initialized = False
    
    async def initialize(self):
        """初始化调用器"""
        if self._initialized:
            return
        
        if self.llm_client is None:
            self.llm_client = create_llm_client(
                provider=settings.llm.provider,
                api_key=settings.llm.api_key,
                api_base=settings.llm.api_base,
                default_model=settings.llm.model,
                timeout=settings.llm.timeout
            )
        
        self._initialized = True
    
    async def invoke(
        self,
        agent_name: str,
        task: Dict[str, Any],
        input_data: Dict[str, Any],
        plan_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """调用Agent执行任务"""
        await self.initialize()
        
        invocation_id = f"INV_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()
        
        logger.info(f"[AgentInvoker] Invoking {agent_name} for task {task.get('task_id', '')}")
        
        try:
            # 1. 构建Prompt
            messages = self.prompt_builder.build_prompt(
                agent_name=agent_name,
                task=task,
                input_data=input_data,
                plan_context=plan_context
            )
            
            # 2. 获取Agent特定的LLM参数
            llm_params = self._get_llm_params(agent_name, plan_context)
            
            # 3. 调用LLM
            response = await self.llm_client.complete(
                messages=messages,
                temperature=llm_params.get("temperature", 0.3),
                max_tokens=llm_params.get("max_tokens", 8000)
            )
            
            # 4. 解析响应
            parsed_response = self._parse_response(response)
            
            # 5. 构建返回结果
            end_time = datetime.now()
            execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            result = {
                "invocation_id": invocation_id,
                "agent_name": agent_name,
                "task_id": task.get("task_id", ""),
                "status": parsed_response.get("status", "success"),
                "output": parsed_response.get("output", {}),
                "quality_check": self._extract_quality_info(parsed_response),
                "errors": parsed_response.get("errors", []),
                "metrics": {
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "total_tokens": response.total_tokens,
                    "latency_ms": response.latency_ms,
                    "execution_time_ms": execution_time_ms,
                },
                "timestamp": end_time.isoformat()
            }
            
            logger.info(f"[AgentInvoker] {agent_name} completed: {result['status']}")
            return result
            
        except Exception as e:
            logger.error(f"[AgentInvoker] {agent_name} failed: {e}")
            return {
                "invocation_id": invocation_id,
                "agent_name": agent_name,
                "task_id": task.get("task_id", ""),
                "status": "failed",
                "output": {},
                "quality_check": {},
                "errors": [{"error_type": "invocation_error", "message": str(e)}],
                "metrics": {},
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_llm_params(
        self,
        agent_name: str,
        plan_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """获取Agent特定的LLM参数"""
        # 基础参数
        params = {
            "temperature": settings.llm.temperature,
            "max_tokens": settings.llm.max_tokens,
            "timeout": settings.llm.timeout,
        }
        
        # Agent特定参数
        agent_type = self._get_agent_type(agent_name)
        if agent_type and agent_type in AGENT_LLM_PARAMS:
            agent_params = AGENT_LLM_PARAMS[agent_type]
            params.update(agent_params)
        
        # 执行模式调整
        execution_mode = plan_context.get("execution_mode", "")
        if execution_mode:
            from ..config.settings import ExecutionMode
            try:
                mode = ExecutionMode(execution_mode)
                if mode in MODE_ADJUSTMENTS:
                    adjustments = MODE_ADJUSTMENTS[mode]
                    params["temperature"] += adjustments.get("temperature_delta", 0)
                    params["max_tokens"] = int(params["max_tokens"] * adjustments.get("max_tokens_multiplier", 1))
                    params["timeout"] = int(params["timeout"] * adjustments.get("timeout_multiplier", 1))
            except ValueError:
                pass
        
        return params
    
    def _get_agent_type(self, agent_name: str) -> Optional[AgentType]:
        """根据名称获取Agent类型"""
        name_to_type = {
            "Scholar_Internalizer": AgentType.SCHOLAR,
            "Code_Architect": AgentType.CODE,
            "Scenario_Validator": AgentType.VALIDATOR,
            "Knowledge_Vault": AgentType.VAULT,
            "Strategic_Critic": AgentType.CRITIC,
            "E2E-Learning-Orchestrator": AgentType.ORCHESTRATOR,
        }
        return name_to_type.get(agent_name)
    
    def _parse_response(self, response: LLMResponse) -> Dict[str, Any]:
        """解析LLM响应"""
        content = response.content.strip()
        
        # 尝试提取JSON
        try:
            # 查找JSON块
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                if end > start:
                    content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                if end > start:
                    content = content[start:end].strip()
            
            parsed = json.loads(content)
            
            # 标准化输出格式
            return {
                "status": parsed.get("status", "success"),
                "output": {
                    "atoms": parsed.get("atoms", []),
                    "summary": parsed.get("summary", ""),
                },
                "errors": parsed.get("issues", []) or parsed.get("errors", []),
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # 返回原始内容作为摘要
            return {
                "status": "partial",
                "output": {
                    "atoms": [],
                    "summary": content[:1000],
                    "raw_content": content,
                },
                "errors": [{"error_type": "parse_error", "message": str(e)}],
            }
    
    def _extract_quality_info(self, parsed_response: Dict[str, Any]) -> Dict[str, Any]:
        """提取质量信息"""
        output = parsed_response.get("output", {})
        atoms = output.get("atoms", [])
        
        if not atoms:
            return {
                "atoms_generated": 0,
                "avg_quality_score": 0.0,
                "grades": {"A": 0, "B": 0, "C": 0, "D": 0},
                "issues": parsed_response.get("errors", [])
            }
        
        grades = {"A": 0, "B": 0, "C": 0, "D": 0}
        total_score = 0.0
        
        for atom in atoms:
            grade = atom.get("quality_grade", "C")
            if grade in grades:
                grades[grade] += 1
            score = atom.get("quality_score", 0.7)
            total_score += score
        
        return {
            "atoms_generated": len(atoms),
            "avg_quality_score": total_score / len(atoms) if atoms else 0.0,
            "grades": grades,
            "issues": parsed_response.get("errors", [])
        }
