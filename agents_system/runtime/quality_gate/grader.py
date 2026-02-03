"""
质量评分器模块 - 提供知识原子的质量评分
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GradeResult:
    """评分结果"""
    grade: str
    score: float
    breakdown: Dict[str, float]
    feedback: List[str]


class QualityGrader:
    """质量评分器 - 对知识原子进行质量评分"""
    
    DIMENSION_WEIGHTS = {
        "accuracy": 0.25,
        "completeness": 0.25,
        "clarity": 0.20,
        "depth": 0.15,
        "relevance": 0.15,
    }
    
    GRADE_THRESHOLDS = {
        "A": 0.90,
        "B": 0.75,
        "C": 0.60,
        "D": 0.0,
    }
    
    def __init__(self, custom_weights: Optional[Dict[str, float]] = None):
        self.weights = custom_weights if custom_weights else self.DIMENSION_WEIGHTS.copy()
    
    def grade_atom(self, atom: Dict[str, Any]) -> GradeResult:
        """对单个原子进行评分"""
        content = atom.get("content", {})
        atom_type = atom.get("atom_type", "")
        
        breakdown = {
            "accuracy": self._score_accuracy(content),
            "completeness": self._score_completeness(content, atom_type),
            "clarity": self._score_clarity(content),
            "depth": self._score_depth(content, atom_type),
            "relevance": self._score_relevance(content),
        }
        
        total_score = sum(
            breakdown[dim] * self.weights.get(dim, 0)
            for dim in breakdown
        )
        
        grade = self._score_to_grade(total_score)
        feedback = self._generate_feedback(breakdown, grade)
        
        return GradeResult(
            grade=grade,
            score=total_score,
            breakdown=breakdown,
            feedback=feedback
        )
    
    def grade_atoms(self, atoms: List[Dict[str, Any]]) -> Dict[str, Any]:
        """对多个原子进行评分"""
        results = []
        grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        total_score = 0.0
        
        for atom in atoms:
            result = self.grade_atom(atom)
            results.append({
                "atom_id": atom.get("atom_id", ""),
                "grade": result.grade,
                "score": result.score,
                "breakdown": result.breakdown,
            })
            grade_counts[result.grade] += 1
            total_score += result.score
        
        avg_score = total_score / len(atoms) if atoms else 0.0
        
        return {
            "atom_results": results,
            "summary": {
                "total_atoms": len(atoms),
                "avg_score": avg_score,
                "avg_grade": self._score_to_grade(avg_score),
                "grade_distribution": grade_counts,
            }
        }
    
    def _score_accuracy(self, content: Dict[str, Any]) -> float:
        """评估准确性"""
        score = 0.8
        if content.get("source_reference"):
            score += 0.1
        if content.get("verified"):
            score += 0.1
        return min(1.0, score)
    
    def _score_completeness(self, content: Dict[str, Any], atom_type: str) -> float:
        """评估完整性"""
        required_fields = {
            "concept": ["concept_name", "definition", "context"],
            "method": ["method_name", "steps", "purpose"],
            "finding": ["finding_statement", "evidence"],
            "code": ["code_snippet", "description"],
        }
        fields = required_fields.get(atom_type, [])
        if not fields:
            return 0.8
        present = sum(1 for f in fields if content.get(f))
        return present / len(fields)
    
    def _score_clarity(self, content: Dict[str, Any]) -> float:
        """评估清晰度"""
        score = 0.7
        text_content = " ".join(str(v) for v in content.values() if isinstance(v, str))
        word_count = len(text_content.split())
        if 50 <= word_count <= 500:
            score += 0.2
        elif 20 <= word_count < 50 or 500 < word_count <= 1000:
            score += 0.1
        if any(isinstance(v, (list, dict)) for v in content.values()):
            score += 0.1
        return min(1.0, score)
    
    def _score_depth(self, content: Dict[str, Any], atom_type: str) -> float:
        """评估深度"""
        min_lengths = {"concept": 100, "method": 150, "finding": 80, "code": 50}
        min_len = min_lengths.get(atom_type, 50)
        text_content = " ".join(str(v) for v in content.values() if isinstance(v, str))
        actual_len = len(text_content)
        ratio = min(1.0, actual_len / min_len)
        bonus = 0.0
        if content.get("examples"):
            bonus += 0.1
        if content.get("related_concepts"):
            bonus += 0.05
        return min(1.0, ratio * 0.8 + bonus + 0.1)
    
    def _score_relevance(self, content: Dict[str, Any]) -> float:
        """评估相关性"""
        score = 0.75
        if content.get("context"):
            score += 0.1
        if content.get("related_to") or content.get("dependencies"):
            score += 0.1
        if content.get("applications"):
            score += 0.05
        return min(1.0, score)
    
    def _score_to_grade(self, score: float) -> str:
        """将分数转换为等级"""
        for grade, threshold in self.GRADE_THRESHOLDS.items():
            if score >= threshold:
                return grade
        return "D"
    
    def _generate_feedback(self, breakdown: Dict[str, float], grade: str) -> List[str]:
        """生成改进反馈"""
        feedback = []
        sorted_dims = sorted(breakdown.items(), key=lambda x: x[1])
        suggestions = {
            "accuracy": "添加来源引用以提高准确性",
            "completeness": "补充缺失的必要字段",
            "clarity": "优化表述使内容更清晰",
            "depth": "增加更多细节和示例",
            "relevance": "添加上下文信息",
        }
        for dim, score in sorted_dims[:2]:
            if score < 0.7:
                feedback.append(suggestions.get(dim, "请改进此维度"))
        if grade in ["C", "D"]:
            feedback.append("建议重新审视并补充内容")
        return feedback
