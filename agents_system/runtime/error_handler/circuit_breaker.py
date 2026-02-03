"""
熔断器模块 - 实现熔断器模式防止级联故障
"""
import asyncio
from enum import Enum
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitStats:
    """熔断器统计"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None


class CircuitBreaker:
    """熔断器 - 防止对故障服务的持续调用"""
    
    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_max_calls: int = 3,
        success_threshold: int = 2
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._half_open_calls = 0
        self._half_open_successes = 0
        self._state_changed_at = datetime.now()
        self._lock = asyncio.Lock()
    
    @property
    def state(self) -> CircuitState:
        return self._state
    
    @property
    def stats(self) -> CircuitStats:
        return self._stats
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """通过熔断器执行调用"""
        async with self._lock:
            if not self._can_execute():
                raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is open")
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure(e)
            raise
    
    def _can_execute(self) -> bool:
        if self._state == CircuitState.CLOSED:
            return True
        
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to(CircuitState.HALF_OPEN)
                return True
            return False
        
        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_calls < self.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False
        
        return False
    
    def _should_attempt_reset(self) -> bool:
        elapsed = datetime.now() - self._state_changed_at
        return elapsed.total_seconds() >= self.reset_timeout
    
    async def _on_success(self) -> None:
        async with self._lock:
            self._stats.total_calls += 1
            self._stats.successful_calls += 1
            self._stats.consecutive_failures = 0
            self._stats.last_success_time = datetime.now()
            
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
    
    async def _on_failure(self, error: Exception) -> None:
        async with self._lock:
            self._stats.total_calls += 1
            self._stats.failed_calls += 1
            self._stats.consecutive_failures += 1
            self._stats.last_failure_time = datetime.now()
            
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
    
    def _transition_to(self, new_state: CircuitState) -> None:
        old_state = self._state
        self._state = new_state
        self._state_changed_at = datetime.now()
        
        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._half_open_successes = 0
        elif new_state == CircuitState.CLOSED:
            self._stats.consecutive_failures = 0
        
        logger.info(f"Circuit breaker '{self.name}' transitioned from {old_state.value} to {new_state.value}")
    
    def reset(self) -> None:
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._half_open_calls = 0
        self._half_open_successes = 0
        self._state_changed_at = datetime.now()
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self._state.value,
            "stats": {
                "total_calls": self._stats.total_calls,
                "successful_calls": self._stats.successful_calls,
                "failed_calls": self._stats.failed_calls,
                "consecutive_failures": self._stats.consecutive_failures,
            }
        }


class CircuitBreakerOpenError(Exception):
    """熔断器打开异常"""
    pass
