"""
数据传递模块 - 提供引用解析和上下文管理
"""
from .resolver import ReferenceResolver
from .context import ContextManager

__all__ = [
    "ReferenceResolver",
    "ContextManager",
]
