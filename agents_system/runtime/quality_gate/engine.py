"""
质量门引擎 - 核心质量检查逻辑
"""
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..config.settings import settings, QualityGateConfig

logger = logging.getLogger(__name__)


class GateType(Enum):
    """质量门类型"""
    PRE_EXECUTION = "pre_execution"
    POST_EXECUTION = "post_execution"
    INTEGRATION = "integration"


class CheckResult(Enum):
    """检查结果"""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class GateCheckResult:
    """质量门检查结果"""
    gate_name: str
    gate_type: GateType
    passed: bool
    score: float = 0.0
    grade: str = "D"
    checks: List[Dict[str, Any]] = field(default_factory=list)
    issues: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    retry_suggested: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "gate_type": self.gate_type.value,
            "passed": self.passed,
            "score": self.score,
            "grade": self.grade,
            "checks": self.checks,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "retry_suggested": self.retry_suggested,
        }


class QualityGateEngine:
    """质量门引擎 - 执行质量检查"""
    
    def __init__(self, config: Optional[QualityGateConfig] = None):
        self.config = config or settings.quality_gate
        self.checkers: Dict[str, Any] = {}
        self._register_default_checkers()
    
    def _register_default_checkers(self):
        """注册默认检查器"""
        from .checkers import (
            CompletenessChecker,
            AccuracyChecker,
            ConsistencyChecker,
            DepthChecker,
        )
        
        self.checkers = {
            "completeness": CompletenessChecker(),
            "accuracy": AccuracyChecker(),
            "consistency": ConsistencyChecker(),
            "depth": DepthChecker(),
        }
    
    def register_checker(self, name: str, checker: Any):
        """注册自定义检查器"""
        self.checkers[name] = checker
    
    async def check(
        self,
        gate_config: Dict[str, Any],
        task: Dict[str, Any],
        response: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """执行质量门检查"""
        gate_name = gate_config.get("gate_name", "unnamed_gate")
        gate_type_str = gate_config.get("gate_type", "post_execution")
        gate_type = GateType(gate_type_str)
        
        logger.info(f"[QualityGate] Running gate: {gate_name} ({gate_type.value})")
        
        # 获取检查项配置
        checks_config = gate_config.get("checks", [])
        threshold = gate_config.get("threshold", self.config.default_threshold)
        min_score = gate_config.get("min_score", self.config.min_quality_score)
        
        # 执行所有检查
        check_results = []
        total_score = 0.0
        total_weight = 0.0
        all_issues = []
        
        for check_config in checks_config:
            checker_name = check_config.get("checker", "")
            weight = check_config.get("weight", 1.0)
            
            if checker_name in self.checkers:
                checker = self.checkers[checker_name]
                try:
                    result = await checker.check(task, response, check_config)
                    check_results.append({
                        "checker": checker_name,
                        "result": result.get("result", CheckResult.SKIP.value),
                        "score": result.get("score", 0.0),
                        "details": result.get("details", {}),
                    })
                    
                    total_score += result.get("score", 0.0) * weight
                    total_weight += weight
                    
                    if result.get("issues"):
                        all_issues.extend(result["issues"])
                        
                except Exception as e:
                    logger.error(f"[QualityGate] Checker {checker_name} failed: {e}")
                    check_results.append({
                        "checker": checker_name,
                        "result": CheckResult.SKIP.value,
                        "score": 0.0,
                        "error": str(e),
                    })
        
        # 计算最终分数
        final_score = total_score / total_weight if total_weight > 0 else 0.0
        grade = self._score_to_grade(final_score)
        passed = self._check_passed(grade, threshold, final_score, min_score)
        
        # 生成建议
        recommendations = self._generate_recommendations(all_issues, grade)
        
        result = GateCheckResult(
            gate_name=gate_name,
            gate_type=gate_type,
            passed=passed,
            score=final_score,
            grade=grade,
            checks=check_results,
            issues=all_issues,
            recommendations=recommendations,
            retry_suggested=not passed and grade in ["C", "D"],
        )
        
        logger.info(f"[QualityGate] {gate_name} result: passed={passed}, grade={grade}, score={final_score:.2f}")
        
        return result.to_dict()
    
    async def run_pre_execution_gate(
        self,
        task: Dict[str, Any],
        plan_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """运行预执行质量门"""
        if not self.config.enable_pre_execution:
            return {"passed": True, "skipped": True}
        
        gate_config = {
            "gate_name": f"pre_exec_{task.get('task_id', '')}",
            "gate_type": GateType.PRE_EXECUTION.value,
            "checks": [
                {"checker": "completeness", "weight": 1.0, "target": "input"},
            ],
            "threshold": "C",
            "min_score": 0.5,
        }
        
        return await self.check(gate_config, task, None)
    
    async def run_post_execution_gate(
        self,
        task: Dict[str, Any],
        response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """运行后执行质量门"""
        if not self.config.enable_post_execution:
            return {"passed": True, "skipped": True}
        
        gate_config = {
            "gate_name": f"post_exec_{task.get('task_id', '')}",
            "gate_type": GateType.POST_EXECUTION.value,
            "checks": [
                {"checker": "completeness", "weight": 0.3, "target": "output"},
                {"checker": "accuracy", "weight": 0.3},
                {"checker": "consistency", "weight": 0.2},
                {"checker": "depth", "weight": 0.2},
            ],
            "threshold": self.config.default_threshold,
            "min_score": self.config.min_quality_score,
        }
        
        return await self.check(gate_config, task, response)
    
    async def run_integration_gate(
        self,
        atoms: List[Dict[str, Any]],
        plan_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """运行集成质量门"""
        if not self.config.enable_integration:
            return {"passed": True, "skipped": True}
        
        gate_config = {
            "gate_name": "integration_gate",
            "gate_type": GateType.INTEGRATION.value,
            "checks": [
                {"checker": "consistency", "weight": 0.5, "target": "atoms"},
                {"checker": "completeness", "weight": 0.5, "target": "atoms"},
            ],
            "threshold": "B",
            "min_score": 0.75,
        }
        
        task = {"atoms": atoms, "context": plan_context}
        return await self.check(gate_config, task, {"atoms": atoms})
    
    def _score_to_grade(self, score: float) -> str:
        """将分数转换为等级"""
        if score >= 0.9:
            return "A"
        elif score >= 0.75:
            return "B"
        elif score >= 0.6:
            return "C"
        else:
            return "D"
    
    def _check_passed(
        self,
        grade: str,
        threshold: str,
        score: float,
        min_score: float
    ) -> bool:
        """检查是否通过"""
        grade_order = {"A": 4, "B": 3, "C": 2, "D": 1}
        return (
            grade_order.get(grade, 0) >= grade_order.get(threshold, 0)
            and score >= min_score
        )
    
    def _generate_recommendations(
        self,
        issues: List[Dict[str, Any]],
        grade: str
    ) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        issue_types = set(issue.get("type", "") for issue in issues)
        
        if "missing_field" in issue_types:
            recommendations.append("确保所有必需字段都已填充")
        
        if "low_depth" in issue_types:
            recommendations.append("增加分析深度，提供更详细的解释")
        
        if "inconsistency" in issue_types:
            recommendations.append("检查输出与输入的一致性")
        
        if "accuracy_issue" in issue_types:
            recommendations.append("验证输出的准确性，确保与原文一致")
        
        if grade == "D":
            recommendations.append("考虑重新执行任务以提高质量")
        
        return recommendations
