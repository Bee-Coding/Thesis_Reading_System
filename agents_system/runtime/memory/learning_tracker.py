"""
学习状态追踪器
追踪学习进度、知识盲区、理解程度等
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from .memory_manager import MemoryManager, MemoryType

logger = logging.getLogger(__name__)


class LearningTracker:
    """学习状态追踪器"""
    
    def __init__(self, memory_manager: MemoryManager):
        """
        初始化学习追踪器
        
        Args:
            memory_manager: 记忆管理器实例
        """
        self.memory_manager = memory_manager
        logger.info("Learning tracker initialized")
    
    def start_paper(self, paper_id: str, paper_title: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        开始学习一篇新论文
        
        Args:
            paper_id: 论文ID
            paper_title: 论文标题
            metadata: 额外元数据
        
        Returns:
            添加结果
        """
        content = f"开始学习论文: {paper_title}"
        full_metadata = {
            "paper_id": paper_id,
            "paper_title": paper_title,
            "status": "started",
            **(metadata or {})
        }
        
        result = self.memory_manager.add_memory(
            content=content,
            memory_type=MemoryType.PAPER_METADATA,
            metadata=full_metadata
        )
        
        logger.info(f"Started learning paper: {paper_id}")
        return result
    
    def update_progress(
        self,
        paper_id: str,
        task: str,
        status: str = "completed",
        understanding_level: str = "medium",
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        更新学习进度
        
        Args:
            paper_id: 论文ID
            task: 任务描述
            status: 状态 (pending/in_progress/completed)
            understanding_level: 理解程度 (basic/medium/advanced)
            notes: 备注
        
        Returns:
            添加结果
        """
        content = f"任务: {task} - 状态: {status} - 理解程度: {understanding_level}"
        if notes:
            content += f"\n备注: {notes}"
        
        metadata = {
            "paper_id": paper_id,
            "task": task,
            "status": status,
            "understanding_level": understanding_level
        }
        
        result = self.memory_manager.add_memory(
            content=content,
            memory_type=MemoryType.LEARNING_PROGRESS,
            metadata=metadata
        )
        
        logger.info(f"Updated progress for {paper_id}: {task} - {status}")
        return result
    
    def add_knowledge_gap(
        self,
        gap_id: str,
        description: str,
        priority: str = "medium",
        paper_id: Optional[str] = None,
        related_concepts: Optional[List[str]] = None,
        next_steps: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        添加知识盲区
        
        Args:
            gap_id: 盲区ID (如 GAP_FLOWMATCHING_01)
            description: 描述
            priority: 优先级 (low/medium/high)
            paper_id: 相关论文ID
            related_concepts: 相关概念列表
            next_steps: 下一步行动列表
        
        Returns:
            添加结果
        """
        content = f"知识盲区 [{gap_id}]: {description}"
        if next_steps:
            content += f"\n下一步: {', '.join(next_steps)}"
        
        metadata = {
            "gap_id": gap_id,
            "priority": priority,
            "status": "pending",
            "paper_id": paper_id,
            "related_concepts": related_concepts or [],
            "next_steps": next_steps or []
        }
        
        result = self.memory_manager.add_memory(
            content=content,
            memory_type=MemoryType.KNOWLEDGE_GAP,
            metadata=metadata
        )
        
        logger.info(f"Added knowledge gap: {gap_id} - {priority} priority")
        return result
    
    def resolve_knowledge_gap(
        self,
        gap_id: str,
        resolution: str,
        confidence: float = 0.8
    ) -> Dict[str, Any]:
        """
        解决知识盲区
        
        Args:
            gap_id: 盲区ID
            resolution: 解决方案/理解
            confidence: 信心程度 (0-1)
        
        Returns:
            添加结果
        """
        content = f"知识盲区 [{gap_id}] 已解决: {resolution}"
        
        metadata = {
            "gap_id": gap_id,
            "status": "resolved",
            "confidence": confidence,
            "resolved_at": datetime.now().isoformat()
        }
        
        result = self.memory_manager.add_memory(
            content=content,
            memory_type=MemoryType.KNOWLEDGE_GAP,
            metadata=metadata
        )
        
        logger.info(f"Resolved knowledge gap: {gap_id} (confidence: {confidence})")
        return result
    
    def add_insight(
        self,
        insight: str,
        paper_id: Optional[str] = None,
        confidence: float = 0.8,
        related_gaps: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        添加关键洞察
        
        Args:
            insight: 洞察内容
            paper_id: 相关论文ID
            confidence: 信心程度 (0-1)
            related_gaps: 相关的知识盲区ID列表
        
        Returns:
            添加结果
        """
        content = f"关键洞察: {insight}"
        
        metadata = {
            "paper_id": paper_id,
            "confidence": confidence,
            "related_gaps": related_gaps or []
        }
        
        result = self.memory_manager.add_memory(
            content=content,
            memory_type=MemoryType.INSIGHT,
            metadata=metadata
        )
        
        logger.info(f"Added insight for paper {paper_id}")
        return result
    
    def add_question(
        self,
        question: str,
        paper_id: Optional[str] = None,
        answered: bool = False,
        answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        添加问题
        
        Args:
            question: 问题内容
            paper_id: 相关论文ID
            answered: 是否已回答
            answer: 答案（如果已回答）
        
        Returns:
            添加结果
        """
        content = f"问题: {question}"
        if answered and answer:
            content += f"\n答案: {answer}"
        
        metadata = {
            "paper_id": paper_id,
            "answered": answered
        }
        
        result = self.memory_manager.add_memory(
            content=content,
            memory_type=MemoryType.QUESTION,
            metadata=metadata
        )
        
        logger.info(f"Added question for paper {paper_id}")
        return result
    
    def get_learning_summary(self, paper_id: str) -> Dict[str, Any]:
        """
        获取学习总结
        
        Args:
            paper_id: 论文ID
        
        Returns:
            学习总结字典
        """
        # 获取该论文的所有记忆
        memories = self.memory_manager.get_paper_memories(paper_id)
        
        # 按类型分类
        progress = []
        gaps = []
        insights = []
        questions = []
        
        for mem in memories:
            mem_type = mem.get("metadata", {}).get("type")
            if mem_type == MemoryType.LEARNING_PROGRESS.value:
                progress.append(mem)
            elif mem_type == MemoryType.KNOWLEDGE_GAP.value:
                gaps.append(mem)
            elif mem_type == MemoryType.INSIGHT.value:
                insights.append(mem)
            elif mem_type == MemoryType.QUESTION.value:
                questions.append(mem)
        
        # 统计知识盲区状态
        pending_gaps = [g for g in gaps if g.get("metadata", {}).get("status") == "pending"]
        resolved_gaps = [g for g in gaps if g.get("metadata", {}).get("status") == "resolved"]
        
        # 统计学习进度
        completed_tasks = [p for p in progress if p.get("metadata", {}).get("status") == "completed"]
        in_progress_tasks = [p for p in progress if p.get("metadata", {}).get("status") == "in_progress"]
        
        summary = {
            "paper_id": paper_id,
            "total_memories": len(memories),
            "progress": {
                "total_tasks": len(progress),
                "completed": len(completed_tasks),
                "in_progress": len(in_progress_tasks)
            },
            "knowledge_gaps": {
                "total": len(gaps),
                "pending": len(pending_gaps),
                "resolved": len(resolved_gaps)
            },
            "insights": len(insights),
            "questions": {
                "total": len(questions),
                "answered": len([q for q in questions if q.get("metadata", {}).get("answered")])
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Generated learning summary for {paper_id}")
        return summary
    
    def get_pending_gaps(self, paper_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取待解决的知识盲区
        
        Args:
            paper_id: 论文ID（可选）
        
        Returns:
            待解决的知识盲区列表
        """
        if paper_id:
            memories = self.memory_manager.get_paper_memories(paper_id)
        else:
            memories = self.memory_manager.get_by_type(MemoryType.KNOWLEDGE_GAP)
        
        # 过滤待解决的盲区
        pending = [
            mem for mem in memories
            if mem.get("metadata", {}).get("type") == MemoryType.KNOWLEDGE_GAP.value
            and mem.get("metadata", {}).get("status") == "pending"
        ]
        
        # 按优先级排序
        priority_order = {"high": 0, "medium": 1, "low": 2}
        sorted_pending = sorted(
            pending,
            key=lambda m: priority_order.get(m.get("metadata", {}).get("priority", "medium"), 1)
        )
        
        logger.info(f"Found {len(sorted_pending)} pending knowledge gaps")
        return sorted_pending
    
    def get_last_session(self) -> Optional[Dict[str, Any]]:
        """
        获取上次学习会话信息
        
        Returns:
            上次会话信息字典，如果没有则返回None
        """
        recent_memories = self.memory_manager.get_recent_memories(limit=20)
        
        if not recent_memories:
            return None
        
        # 提取最近的论文ID
        paper_ids = set()
        for mem in recent_memories:
            paper_id = mem.get("metadata", {}).get("paper_id")
            if paper_id:
                paper_ids.add(paper_id)
        
        if not paper_ids:
            return None
        
        # 假设最近的论文ID是当前会话
        current_paper_id = list(paper_ids)[0]
        
        # 获取该论文的学习总结
        summary = self.get_learning_summary(current_paper_id)
        
        # 获取待解决的盲区
        pending_gaps = self.get_pending_gaps(current_paper_id)
        
        session_info = {
            "paper_id": current_paper_id,
            "summary": summary,
            "pending_gaps": pending_gaps[:5],  # 只返回前5个
            "last_activity": recent_memories[0].get("metadata", {}).get("timestamp")
        }
        
        logger.info(f"Retrieved last session for paper {current_paper_id}")
        return session_info


# 便捷函数：创建学习追踪器
def create_learning_tracker(user_id: str = "zhn") -> LearningTracker:
    """
    创建学习追踪器实例
    
    Args:
        user_id: 用户ID
    
    Returns:
        LearningTracker实例
    """
    from .memory_manager import create_memory_manager
    memory_manager = create_memory_manager(user_id=user_id)
    return LearningTracker(memory_manager=memory_manager)
