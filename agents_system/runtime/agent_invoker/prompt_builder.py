"""
Prompt构建器模块 - 构建Agent的输入Prompt
"""
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Prompt构建器 - 为Agent构建输入Prompt"""
    
    def __init__(self, agents_dir: Optional[Path] = None):
        self.agents_dir = agents_dir
        self._prompt_cache: Dict[str, str] = {}
    
    def build_prompt(
        self,
        agent_name: str,
        task: Dict[str, Any],
        input_data: Dict[str, Any],
        plan_context: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """构建完整的Prompt消息列表"""
        messages = []
        
        # 1. 系统提示词
        system_prompt = self._get_system_prompt(agent_name)
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # 2. 构建用户消息
        user_message = self._build_user_message(task, input_data, plan_context)
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        return messages
    
    def _get_system_prompt(self, agent_name: str) -> str:
        """获取Agent的系统提示词"""
        if agent_name in self._prompt_cache:
            return self._prompt_cache[agent_name]
        
        if self.agents_dir:
            prompt_file = self.agents_dir / f"{agent_name}.md"
            if prompt_file.exists():
                prompt = prompt_file.read_text(encoding="utf-8")
                self._prompt_cache[agent_name] = prompt
                return prompt
        
        # 返回默认提示词
        return self._get_default_prompt(agent_name)
    
    def _get_default_prompt(self, agent_name: str) -> str:
        """获取默认系统提示词"""
        default_prompts = {
            "Scholar_Internalizer": """You are Scholar_Internalizer, an expert at deeply understanding academic papers.
Your task is to extract and internalize key concepts, methods, and findings from research papers.
Output your analysis as structured knowledge atoms in JSON format.""",
            
            "Code_Architect": """You are Code_Architect, an expert at understanding and reproducing code implementations.
Your task is to analyze code, understand algorithms, and create reproducible implementations.
Output your analysis as structured knowledge atoms in JSON format.""",
            
            "Scenario_Validator": """You are Scenario_Validator, an expert at validating research claims through scenarios.
Your task is to create validation scenarios and test the applicability of research findings.
Output your analysis as structured knowledge atoms in JSON format.""",
            
            "Knowledge_Vault": """You are Knowledge_Vault, responsible for organizing and storing knowledge.
Your task is to structure, index, and maintain knowledge atoms for easy retrieval.
Output your analysis as structured knowledge atoms in JSON format.""",
            
            "Strategic_Critic": """You are Strategic_Critic, an expert at critical analysis of research.
Your task is to identify limitations, assumptions, and potential issues in research.
Output your analysis as structured knowledge atoms in JSON format.""",
        }
        return default_prompts.get(agent_name, "You are a helpful research assistant.")
    
    def _build_user_message(
        self,
        task: Dict[str, Any],
        input_data: Dict[str, Any],
        plan_context: Dict[str, Any]
    ) -> str:
        """构建用户消息"""
        parts = []
        
        # 任务描述
        parts.append("## Task")
        parts.append(f"Task ID: {task.get('task_id', 'unknown')}")
        parts.append(f"Description: {task.get('description', 'No description')}")
        
        # 输出规范
        output_spec = task.get("output_spec", {})
        if output_spec:
            parts.append("")
            parts.append("## Expected Output")
            parts.append(f"Atom Types: {output_spec.get('atom_types', [])}")
            parts.append(f"Quality Threshold: {output_spec.get('quality_threshold', 'B')}")
        
        # 输入数据
        parts.append("")
        parts.append("## Input Data")
        
        sources = input_data.get("sources", [])
        if sources:
            parts.append("### Sources")
            for source in sources:
                parts.append(f"- {source}")
        
        dependencies = input_data.get("dependencies", {})
        if dependencies:
            parts.append("### Dependencies from Previous Tasks")
            for dep_id, dep_data in dependencies.items():
                parts.append(f"#### From {dep_id}:")
                if isinstance(dep_data, dict):
                    parts.append(json.dumps(dep_data, indent=2, ensure_ascii=False))
                else:
                    parts.append(str(dep_data))
        
        # 上下文信息
        if plan_context:
            parts.append("")
            parts.append("## Context")
            paper_info = plan_context.get("paper_info", {})
            if paper_info:
                parts.append(f"Paper: {paper_info.get('title', 'Unknown')}")
                parts.append(f"Authors: {paper_info.get('authors', [])}")
            
            execution_mode = plan_context.get("execution_mode", "")
            if execution_mode:
                parts.append(f"Execution Mode: {execution_mode}")
        
        # 输出格式要求
        parts.append("")
        parts.append("## Output Format")
        parts.append("Please provide your response in the following JSON format:")
        parts.append(self._get_output_template())
        
        return "\n".join(parts)
    
    def _get_output_template(self) -> str:
        """获取输出模板"""
        template = {
            "status": "success | partial | failed",
            "atoms": [
                {
                    "atom_id": "unique_id",
                    "atom_type": "concept | method | finding | code | question | insight",
                    "content": {
                        "field1": "value1",
                        "field2": "value2"
                    },
                    "quality_grade": "A | B | C | D",
                    "quality_score": 0.85,
                    "source_reference": "page/section reference"
                }
            ],
            "summary": "Brief summary of the analysis",
            "issues": []
        }
        return json.dumps(template, indent=2, ensure_ascii=False)
