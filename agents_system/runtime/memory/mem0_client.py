"""
Mem0 客户端封装
提供统一的记忆存储和检索接口
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from mem0 import Memory
import logging

logger = logging.getLogger(__name__)


class Mem0Client:
    """Mem0客户端封装类"""
    
    def __init__(self, api_key: Optional[str] = None, user_id: str = "default_user"):
        """
        初始化Mem0客户端
        
        Args:
            api_key: Mem0 API密钥，如果为None则从环境变量读取
            user_id: 用户ID，用于隔离不同用户的记忆
        """
        self.api_key = api_key or os.getenv("MEM0_API_KEY")
        self.user_id = user_id
        
        if not self.api_key:
            raise ValueError("MEM0_API_KEY not found. Please set it in environment or pass as parameter.")
        
        # 初始化Memory客户端
        try:
            # 配置使用 DeepSeek 模型
            from mem0 import Memory
            
            # 检查是否使用自定义 API
            openai_base_url = os.getenv("OPENAI_BASE_URL", "")
            
            if "ltcraft" in openai_base_url.lower() or "anthropic" in openai_base_url.lower():
                # 使用 Anthropic Claude API + HuggingFace Embedding
                config = {
                    "llm": {
                        "provider": "openai",
                        "config": {
                            "model": "claude-sonnet-4-5",
                            "temperature": 0.2,
                            "max_tokens": 1500,
                        }
                    },
                    "embedder": {
                        "provider": "huggingface",
                        "config": {
                            "model": "multi-qa-MiniLM-L6-cos-v1"
                        }
                    },
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {
                            "collection_name": "mem0_memories",
                            "embedding_model_dims": 384,  # MiniLM 的维度
                            "path": str(Path(__file__).parent.parent.parent.parent / "data" / "qdrant_mem0"),
                        }
                    }
                }
                self.memory = Memory.from_config(config_dict=config)
                logger.info(f"Mem0 client initialized with Claude + HuggingFace Embedding for user: {user_id}")
            elif "deepseek" in openai_base_url.lower():
                # 使用 DeepSeek LLM + Hugging Face Embedding + 本地 Qdrant
                config = {
                    "llm": {
               "provider": "openai",
                        "config": {
                            "model": "deepseek-chat",
                            "temperature": 0.2,
                            "max_tokens": 1500,
                        }
                    },
                    "embedder": {
                        "provider": "huggingface",
                        "config": {
                            "model": "multi-qa-MiniLM-L6-cos-v1"
                        }
                    },
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {
                            "collection_name": "mem0_memories",
                            "embedding_model_dims": 384,
                            "path": str(Path(__file__).parent.parent.parent.parent / "data" / "qdrant_mem0"),
                        }
                    }
                }
                self.memory = Memory.from_config(config_dict=config)
                logger.info(f"Mem0 client initialized with DeepSeek LLM + HuggingFace Embedding for user: {user_id}")
            else:
                # 使用默认配置
                self.memory = Memory()
                logger.info(f"Mem0 client initialized for user: {user_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Mem0 client: {e}")
            raise
    
    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        添加记忆
        
        Args:
            content: 记忆内容（文本）
            metadata: 元数据（可选），用于分类和过滤
        
        Returns:
            添加结果，包含记忆ID
        """
        try:
            messages = [{"role": "user", "content": content}]
            result = self.memory.add(
                messages=messages,
                user_id=self.user_id,
                metadata=metadata or {}
            )
            logger.info(f"Memory added: {content[:50]}...")
            return result
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise
    
    def search(
        self, 
        query: str, 
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索记忆
        
        Args:
            query: 搜索查询
            limit: 返回结果数量限制
            filters: 过滤条件（基于metadata）
        
        Returns:
            匹配的记忆列表
        """
        try:
            results = self.memory.search(
                query=query,
                user_id=self.user_id,
                limit=limit
            )
            
            # 如果有过滤条件，进行过滤
            if filters:
                results = self._apply_filters(results, filters)
            
            logger.info(f"Memory search: '{query}' returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Failed to search memory: {e}")
            raise
    
    def get_all(self) -> List[Dict[str, Any]]:
        """
        获取所有记忆
        
        Returns:
            所有记忆列表
        """
        try:
            results = self.memory.get_all(user_id=self.user_id)
            
            # Mem0 返回的可能是字典格式 {'results': [...]}
            if isinstance(results, dict):
                if 'results' in results:
                    results = results['results']
                else:
                    logger.warning(f"Unexpected dict format: {results.keys()}")
                    return []
            
            # 确保返回列表
            if not isinstance(results, list):
                logger.warning(f"Results is not a list: {type(results)}")
                return []
            
            logger.info(f"Retrieved {len(results)} memories")
            return results
        except Exception as e:
            logger.error(f"Failed to get all memories: {e}")
            raise
    
    def update(self, memory_id: str, content: str) -> Dict[str, Any]:
        """
        更新记忆
        
        Args:
            memory_id: 记忆ID
            content: 新的内容
        
        Returns:
            更新结果
        """
        try:
            result = self.memory.update(
                memory_id=memory_id,
                data=content
            )
            logger.info(f"Memory updated: {memory_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to update memory: {e}")
            raise
    
    def delete(self, memory_id: str) -> Dict[str, Any]:
        """
        删除记忆
        
        Args:
            memory_id: 记忆ID
        
        Returns:
            删除结果
        """
        try:
            result = self.memory.delete(memory_id=memory_id)
            logger.info(f"Memory deleted: {memory_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            raise
    
    def _apply_filters(
        self, 
        results: List[Dict[str, Any]], 
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        应用过滤条件
        
        Args:
            results: 原始结果列表
            filters: 过滤条件
        
        Returns:
            过滤后的结果列表
        """
        # 如果 results 不是列表，直接返回空列表
        if not isinstance(results, list):
            logger.warning(f"Results is not a list: {type(results)}")
            return []
        
        filtered = []
        for result in results:
            # 如果 result 不是字典，跳过
            if not isinstance(result, dict):
                logger.warning(f"Result is not a dict: {type(result)}")
                continue
                
            metadata = result.get("metadata", {})
            match = True
            
            for key, value in filters.items():
                if isinstance(value, list):
                    # 如果过滤值是列表，检查metadata中的值是否在列表中
                    if metadata.get(key) not in value:
                        match = False
                        break
                else:
                    # 精确匹配
                    if metadata.get(key) != value:
                        match = False
                        break
            
            if match:
                filtered.append(result)
        
        return filtered
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        获取记忆历史
        
        Returns:
            记忆历史列表
        """
        try:
            history = self.memory.history(user_id=self.user_id)
            logger.info(f"Retrieved memory history: {len(history)} entries")
            return history
        except Exception as e:
            logger.error(f"Failed to get memory history: {e}")
            raise


# 便捷函数：创建客户端实例
def create_mem0_client(user_id: str = "default_user") -> Mem0Client:
    """
    创建Mem0客户端实例
    
    Args:
        user_id: 用户ID
    
    Returns:
        Mem0Client实例
    """
    return Mem0Client(user_id=user_id)
