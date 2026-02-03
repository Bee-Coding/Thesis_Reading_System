"""
引用解析器模块 - 解析任务间的数据引用
"""
from typing import Dict, List, Any, Optional
import re
import logging

logger = logging.getLogger(__name__)


class ReferenceResolver:
    """引用解析器 - 解析任务输出中的引用路径"""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
    
    async def resolve_reference(
        self,
        data: Dict[str, Any],
        reference: str
    ) -> Any:
        """解析引用路径获取数据"""
        if not reference:
            return data
        
        try:
            return self._resolve_path(data, reference)
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Failed to resolve reference '{reference}': {e}")
            return None
    
    def _resolve_path(self, data: Any, path: str) -> Any:
        """递归解析路径"""
        if not path:
            return data
        
        parts = path.split(".")
        current = data
        
        for part in parts:
            # 检查是否有索引
            match = re.match(r'^(\w+)\[(\d+|\*)\]$', part)
            if match:
                key, index = match.groups()
                if isinstance(current, dict):
                    current = current.get(key)
                else:
                    return None
                
                if index == '*':
                    if isinstance(current, list):
                        return current
                    return None
                else:
                    try:
                        current = current[int(index)]
                    except (IndexError, TypeError):
                        return None
            else:
                if isinstance(current, dict):
                    current = current.get(part)
                else:
                    return None
        
        return current
    
    def extract_atoms(
        self,
        task_output: Dict[str, Any],
        atom_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """从任务输出中提取原子"""
        output = task_output.get("output", {})
        atoms = output.get("atoms", [])
        
        if atom_types:
            atoms = [a for a in atoms if a.get("atom_type") in atom_types]
        
        return atoms
    
    def extract_summary(self, task_output: Dict[str, Any]) -> str:
        """从任务输出中提取摘要"""
        output = task_output.get("output", {})
        return output.get("summary", "")
    
    def merge_outputs(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """合并多个任务输出"""
        merged = {
            "atoms": [],
            "summaries": [],
            "errors": []
        }
        
        for output in outputs:
            atoms = self.extract_atoms(output)
            merged["atoms"].extend(atoms)
            
            summary = self.extract_summary(output)
            if summary:
                merged["summaries"].append(summary)
            
            errors = output.get("errors", [])
            merged["errors"].extend(errors)
        
        return merged
