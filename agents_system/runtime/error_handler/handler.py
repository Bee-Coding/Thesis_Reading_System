"""
错误处理器模块 - 统一错误处理和恢复策略
"""
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

from .retry import RetryStrategy, ExponentialBackoff
from .circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from ..config.settings import settings

logger = logging.getLogger(__name__)


class ErrorCode(Enum):
    """错误代码"""
    E001 = "NETWORK_ERROR"
    E002 = "TIMEOUT_ERROR"
    E003 = "RATE_LIMIT_ERROR"
    E004 = "API_ERROR"
    E005 = "PARSE_ERROR"
    E006 = "VALIDATION_ERROR"
    E007 = "QUALITY_GATE_FAILED"
    E008 = "DEPENDENCY_ERROR"
    E009 = "CIRCUIT_BREAKER_OPEN"
    E010 = "UNKNOWN_ERROR"


@dataclass
class ErrorInfo:
    """错误信息"""
    code: ErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None
    recoverable: bool = True
    retry_suggested: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.name,
            "message": self.message,
            "details": self.details or {},
            "recoverable": self.recoverable,
            "retry_suggested": self.retry_suggested,
        }


class ErrorHandler:
    """错误处理器 - 统一处理各种错误"""
    
    def __init__(
        self,
        retry_strategy: Optional[RetryStrategy] = None,
        circuit_breakers: Optional[Dict[str, CircuitBreaker]] = None
    ):
        self.retry_strategy = retry_strategy or ExponentialBackoff(
            max_attempts=settings.retry.max_attempts,
            base_delay=settings.retry.base_delay_seconds,
            max_delay=settings.retry.max_delay_seconds
        )
        self.circuit_breakers = circuit_breakers or {}
    
    async def handle(
        self,
        error: Exception,
        task: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """处理错误并返回恢复建议"""
        error_info = self._classify_error(error)
        logger.error(f"[ErrorHandler] {error_info.code.name}: {error_info.message}")
        
        recovery = self._get_recovery_action(error_info)
        
        return {
            "error": error_info.to_dict(),
            "recovery": recovery,
            "retry": recovery.get("retry", False),
            "fallback": recovery.get("fallback"),
        }
    
    def _classify_error(self, error: Exception) -> ErrorInfo:
        """分类错误"""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        if isinstance(error, CircuitBreakerOpenError):
            return ErrorInfo(code=ErrorCode.E009, message=str(error), recoverable=False)
        
        if "timeout" in error_str:
            return ErrorInfo(code=ErrorCode.E002, message=str(error), retry_suggested=True)
        
        if "rate limit" in error_str or "429" in error_str:
            return ErrorInfo(code=ErrorCode.E003, message=str(error), retry_suggested=True)
        
        if "connection" in error_str or "network" in error_str:
            return ErrorInfo(code=ErrorCode.E001, message=str(error), retry_suggested=True)
        
        if "json" in error_str or "parse" in error_str:
            return ErrorInfo(code=ErrorCode.E005, message=str(error), retry_suggested=True)
        
        if "validation" in error_str:
            return ErrorInfo(code=ErrorCode.E006, message=str(error), recoverable=False)
        
        return ErrorInfo(code=ErrorCode.E010, message=str(error), details={"type": error_type})
    
    def _get_recovery_action(self, error_info: ErrorInfo) -> Dict[str, Any]:
        """获取恢复动作"""
        actions = {
            ErrorCode.E001: {"retry": True, "delay": 5, "message": "Network error, will retry"},
            ErrorCode.E002: {"retry": True, "delay": 10, "message": "Timeout error, will retry"},
            ErrorCode.E003: {"retry": True, "delay": 60, "message": "Rate limited, will retry"},
            ErrorCode.E004: {"retry": True, "delay": 5, "message": "API error, will retry"},
            ErrorCode.E005: {"retry": True, "delay": 0, "message": "Parse error, will retry"},
            ErrorCode.E006: {"retry": False, "message": "Validation error"},
            ErrorCode.E007: {"retry": True, "delay": 0, "message": "Quality gate failed"},
            ErrorCode.E008: {"retry": False, "message": "Dependency error"},
            ErrorCode.E009: {"retry": False, "message": "Circuit breaker open"},
            ErrorCode.E010: {"retry": False, "message": "Unknown error"},
        }
        return actions.get(error_info.code, {"retry": False, "message": "Unknown error"})
    
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """获取或创建熔断器"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=settings.circuit_breaker.failure_threshold,
                reset_timeout=settings.circuit_breaker.reset_timeout_seconds
            )
        return self.circuit_breakers[name]
