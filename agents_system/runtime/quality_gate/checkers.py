"""
质量检查器模块 - 提供各种质量检查实现
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseChecker(ABC):
    """检查器基类"""
    
    @abstractmethod
    async def check(
        self,
        task: Dict[str, Any],
        response: Optional[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行检查"""
        pass
    
    def _create_result(
        self,
        result: str,
        score: float,
        details: Dict[str, Any] = None,
        issues: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """创建检查结果"""
        return {
            "result": result,
            "score": score,
            "details": details or {},
            "issues": issues or [],
        }


class CompletenessChecker(BaseChecker):
    """完整性检查器"""
    
    REQUIRED_FIELDS = {
        "concept": ["concept_name", "definition", "context"],
        "method": ["method_name", "steps", "purpose"],
        "finding": ["finding_statement", "evidence", "significance"],
        "code": ["code_snippet", "language", "description"],
    }
    
    async def check(
        self,
        task: Dict[str, Any],
        response: Optional[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        target = config.get("target", "output")
        if target == "input":
            return await self._check_input(task)
        return await self._check_output(response)
    
    async def _check_input(self, task: Dict[str, Any]) -> Dict[str, Any]:
        issues = []
        required = ["task_id", "agent", "input_spec"]
        missing = [f for f in required if not task.get(f)]
        if missing:
            issues.append({"type": "missing_field", "fields": missing})
        score = max(0.0, 1.0 - len(issues) * 0.3)
        result = "pass" if score >= 0.7 else "fail"
        return self._create_result(result, score, {"missing": missing}, issues)
    
    async def _check_output(self, response: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not response:
            return self._create_result("fail", 0.0, {}, [{"type": "missing_field"}])
        output = response.get("output", {})
        atoms = output.get("atoms", [])
        if not atoms:
            return self._create_result("fail", 0.3, {"atom_count": 0}, [])
        score = 1.0 - (len([a for a in atoms if not a.get("content")]) / len(atoms))
        result = "pass" if score >= 0.8 else "fail"
        return self._create_result(result, score, {"atom_count": len(atoms)}, [])


class AccuracyChecker(BaseChecker):
    """准确性检查器"""
    
    async def check(
        self,
        task: Dict[str, Any],
        response: Optional[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not response:
            return self._create_result("skip", 0.0, {}, [])
        quality_check = response.get("quality_check", {})
        avg_quality = quality_check.get("avg_quality_score", 0.8)
        errors = response.get("errors", [])
        score = max(0.0, avg_quality - len(errors) * 0.1)
        result = "pass" if score >= 0.7 else "fail"
        return self._create_result(result, score, {"avg_quality": avg_quality}, [])


class ConsistencyChecker(BaseChecker):
    """一致性检查器"""
    
    async def check(
        self,
        task: Dict[str, Any],
        response: Optional[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not response:
            return self._create_result("skip", 0.0, {}, [])
        output = response.get("output", {})
        atoms = output.get("atoms", [])
        atom_ids = [a.get("atom_id", "") for a in atoms]
        duplicates = len(atom_ids) - len(set(atom_ids))
        score = max(0.0, 1.0 - duplicates * 0.2)
        result = "pass" if score >= 0.8 else "fail"
        return self._create_result(result, score, {"atom_count": len(atoms)}, [])


class DepthChecker(BaseChecker):
    """深度检查器"""
    
    MIN_LENGTH = {"concept": 100, "method": 150, "finding": 80, "code": 50}
    
    async def check(
        self,
        task: Dict[str, Any],
        response: Optional[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not response:
            return self._create_result("skip", 0.0, {}, [])
        output = response.get("output", {})
        atoms = output.get("atoms", [])
        if not atoms:
            return self._create_result("fail", 0.0, {}, [])
        
        depth_scores = []
        for atom in atoms:
            atom_type = atom.get("atom_type", "")
            content = atom.get("content", {})
            total_len = sum(len(str(v)) for v in content.values() if isinstance(v, str))
            min_len = self.MIN_LENGTH.get(atom_type, 50)
            depth_scores.append(min(1.0, total_len / min_len))
        
        avg_depth = sum(depth_scores) / len(depth_scores) if depth_scores else 0.0
        result = "pass" if avg_depth >= 0.7 else "fail"
        return self._create_result(result, avg_depth, {"avg_depth": avg_depth}, [])
