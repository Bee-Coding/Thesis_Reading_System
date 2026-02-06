"""
记忆管理器
提供高级的记忆管理功能，包括分类、检索、更新等
"""
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import logging

from .mem0_client import Mem0Client

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """记忆类型枚举"""
    # 学习状态
    LEARNING_PROGRESS = "learning_progress"
    UNDERSTANDING_LEVEL = "understanding_level"
    KNOWLEDGE_GAP = "knowledge_gap"
    
    # 知识内容
    CONCEPT = "concept"
    METHOD = "method"
    INSIGHT = "insight"
    QUESTION = "question"
    
    # 论文相关
    PAPER_METADATA = "paper_metadata"
    PAPER_SECTION = "paper_section"
    CROSS_REFERENCE = "cross_reference"
    
    # 个人偏好
    LEARNING_STYLE = "learning_style"
    RESEARCH_INTEREST = "research_interest"
    DISCUSSION_HISTORY = "discussion_history"


class MemoryManager:
    """记忆管理器"""
    
    def __init__(self, client: Mem0Client):
        """
        初始化记忆管理器
        
        Args:
            client: Mem0客户端实例
        """
        self.client = client
        logger.info("Memory manager initialized")
    
    def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        添加记忆
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型
            metadata: 额外的元数据
        
        Returns:
            添加结果
        """
        # 构建完整的元数据
        full_metadata = {
            "type": memory_type.value,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {})
        }
        
        return self.client.add(content=content, metadata=full_metadata)
    
    def search_by_type(
        self,
        query: str,
        memory_types: List[MemoryType],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        按类型搜索记忆
        
        Args:
            query: 搜索查询
            memory_types: 记忆类型列表
            limit: 返回结果数量限制
        
        Returns:
            匹配的记忆列表
        """
        type_values = [mt.value for mt in memory_types]
        filters = {"type": type_values}
        
        return self.client.search(query=query, limit=limit, filters=filters)
    
    def get_by_type(self, memory_type: MemoryType) -> List[Dict[str, Any]]:
        """
        获取指定类型的所有记忆
        
        Args:
            memory_type: 记忆类型
        
        Returns:
            记忆列表
        """
        all_memories = self.client.get_all()
        
        # 过滤指定类型
        filtered = [
            mem for mem in all_memories
            if mem.get("metadata", {}).get("type") == memory_type.value
        ]
        
        logger.info(f"Retrieved {len(filtered)} memories of type {memory_type.value}")
        return filtered
    
    def get_recent_memories(
        self,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        获取最近的记忆
        
        Args:
            memory_type: 记忆类型（可选）
            limit: 返回结果数量限制
        
        Returns:
            记忆列表（按时间倒序）
        """
        if memory_type:
            memories = self.get_by_type(memory_type)
        else:
            memories = self.client.get_all()
        
        # 按时间戳排序
        sorted_memories = sorted(
            memories,
            key=lambda m: m.get("metadata", {}).get("timestamp", ""),
            reverse=True
        )
        
        return sorted_memories[:limit]
    
    def update_memory(
        self,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        更新记忆
        
        Args:
            memory_id: 记忆ID
            content: 新内容
            metadata: 新元数据（可选）
        
        Returns:
            更新结果
        """
        # 添加更新时间戳
        if metadata:
            metadata["updated_at"] = datetime.now().isoformat()
        
        return self.client.update(memory_id=memory_id, content=content)
    
    def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """
        删除记忆
        
        Args:
            memory_id: 记忆ID
        
        Returns:
            删除结果
        """
        return self.client.delete(memory_id=memory_id)
    
    def search_related(
        self,
        query: str,
        paper_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        搜索相关记忆
        
        Args:
            query: 搜索查询
            paper_id: 论文ID（可选），用于限制搜索范围
            limit: 返回结果数量限制
        
        Returns:
            相关记忆列表
        """
        filters = {}
        if paper_id:
            filters["paper_id"] = paper_id
        
        return self.client.search(query=query, limit=limit, filters=filters if filters else None)
    
    def get_paper_memories(self, paper_id: str) -> List[Dict[str, Any]]:
        """
        获取特定论文的所有记忆
        
        Args:
            paper_id: 论文ID
        
        Returns:
            记忆列表
        """
        all_memories = self.client.get_all()
        
        # 过滤特定论文
        filtered = [
            mem for mem in all_memories
            if mem.get("metadata", {}).get("paper_id") == paper_id
        ]
        
        logger.info(f"Retrieved {len(filtered)} memories for paper {paper_id}")
        return filtered
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取记忆统计信息
        
        Returns:
            统计信息字典
        """
        all_memories = self.client.get_all()
        
        # 按类型统计
        type_counts = {}
        for mem in all_memories:
            mem_type = mem.get("metadata", {}).get("type", "unknown")
            type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
        
        # 按论文统计
        paper_counts = {}
        for mem in all_memories:
            paper_id = mem.get("metadata", {}).get("paper_id")
            if paper_id:
                paper_counts[paper_id] = paper_counts.get(paper_id, 0) + 1
        
        stats = {
            "total_memories": len(all_memories),
            "by_type": type_counts,
            "by_paper": paper_counts,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Memory statistics: {stats['total_memories']} total memories")
        return stats


# 便捷函数：创建记忆管理器
def create_memory_manager(user_id: str = "zhn") -> MemoryManager:
    """
    创建记忆管理器实例
    
    Args:
        user_id: 用户ID
    
    Returns:
        MemoryManager实例
    """
    from .mem0_client import create_mem0_client
    client = create_mem0_client(user_id=user_id)
    return MemoryManager(client=client)
