"""
重试策略模块 - 提供各种重试策略实现
"""
import asyncio
import random
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetryResult:
    """重试结果"""
    success: bool
    result: Any = None
    attempts: int = 0
    last_error: Optional[Exception] = None


class RetryStrategy(ABC):
    """重试策略基类"""
    
    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """获取下次重试的延迟时间（秒）"""
        pass
    
    @abstractmethod
    def should_retry(self, attempt: int, error: Exception) -> bool:
        """判断是否应该重试"""
        pass


class ExponentialBackoff(RetryStrategy):
    """指数退避重试策略"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[tuple] = None
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or (Exception,)
    
    def get_delay(self, attempt: int) -> float:
        """计算指数退避延迟"""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            delay = delay * (0.5 + random.random())
        
        return delay
    
    def should_retry(self, attempt: int, error: Exception) -> bool:
        """判断是否应该重试"""
        if attempt >= self.max_attempts:
            return False
        return isinstance(error, self.retryable_exceptions)


class LinearBackoff(RetryStrategy):
    """线性退避重试策略"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        delay_increment: float = 5.0,
        max_delay: float = 60.0,
        retryable_exceptions: Optional[tuple] = None
    ):
        self.max_attempts = max_attempts
        self.delay_increment = delay_increment
        self.max_delay = max_delay
        self.retryable_exceptions = retryable_exceptions or (Exception,)
    
    def get_delay(self, attempt: int) -> float:
        """计算线性退避延迟"""
        delay = self.delay_increment * (attempt + 1)
        return min(delay, self.max_delay)
    
    def should_retry(self, attempt: int, error: Exception) -> bool:
        """判断是否应该重试"""
        if attempt >= self.max_attempts:
            return False
        return isinstance(error, self.retryable_exceptions)


class FixedDelay(RetryStrategy):
    """固定延迟重试策略"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        delay: float = 5.0,
        retryable_exceptions: Optional[tuple] = None
    ):
        self.max_attempts = max_attempts
        self.delay = delay
        self.retryable_exceptions = retryable_exceptions or (Exception,)
    
    def get_delay(self, attempt: int) -> float:
        """返回固定延迟"""
        return self.delay
    
    def should_retry(self, attempt: int, error: Exception) -> bool:
        """判断是否应该重试"""
        if attempt >= self.max_attempts:
            return False
        return isinstance(error, self.retryable_exceptions)


async def retry_async(
    func: Callable,
    strategy: RetryStrategy,
    *args,
    **kwargs
) -> RetryResult:
    """异步重试执行函数"""
    attempt = 0
    last_error = None
    
    while True:
        try:
            result = await func(*args, **kwargs)
            return RetryResult(
                success=True,
                result=result,
                attempts=attempt + 1
            )
        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            
            if not strategy.should_retry(attempt, e):
                logger.error(f"Max retries reached or non-retryable error")
                return RetryResult(
                    success=False,
                    attempts=attempt + 1,
                    last_error=e
                )
            
            delay = strategy.get_delay(attempt)
            logger.info(f"Retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)
            attempt += 1
