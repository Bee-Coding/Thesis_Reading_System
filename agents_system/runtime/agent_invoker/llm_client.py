"""
LLM客户端模块 - 封装各种LLM API调用，支持DeepSeek、Anthropic轮换
"""
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class RotationStrategy(Enum):
    """轮换策略"""
    ROUND_ROBIN = "round_robin"
    FALLBACK = "fallback"


@dataclass
class LLMResponse:
    """LLM响应"""
    content: str
    model: str
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    raw_response: Optional[Dict[str, Any]] = None


class LLMClient(ABC):
    """LLM客户端基类"""
    
    provider_name: str = "base"
    
    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4000,
        **kwargs
    ) -> LLMResponse:
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        pass


class DeepSeekClient(LLMClient):
    """DeepSeek API客户端 (兼容OpenAI接口)"""
    
    provider_name = "deepseek"
    
    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.deepseek.com/v1",
        default_model: str = "deepseek-reasoner",
        timeout: int = 600
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.default_model = default_model
        self.timeout = timeout
        self._client = None
    
    async def _get_client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base,
                    timeout=self.timeout
                )
            except ImportError:
                logger.error("openai package not installed")
                raise
        return self._client
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 8000,
        **kwargs
    ) -> LLMResponse:
        client = await self._get_client()
        model = model or self.default_model
        start_time = time.time()
        
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            content = response.choices[0].message.content or ""
            
            return LLMResponse(
                content=content,
                model=response.model,
                provider=self.provider_name,
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
                latency_ms=latency_ms,
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None
            )
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise
    
    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            await client.models.list()
            return True
        except Exception as e:
            logger.error(f"DeepSeek health check failed: {e}")
            return False


class AnthropicClient(LLMClient):
    """Anthropic API客户端 (Claude Opus 4.5)"""
    
    provider_name = "anthropic"
    
    def __init__(
        self,
        api_key: str,
        api_base: Optional[str] = None,
        default_model: str = "claude-opus-4-5-20250514",
        timeout: int = 600
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.default_model = default_model
        self.timeout = timeout
        self._client = None
    
    async def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                kwargs = {
                    "api_key": self.api_key,
                    "timeout": self.timeout
                }
                if self.api_base:
                    # 移除末尾的/v1，因为anthropic库会自动添加
                    base_url = self.api_base.rstrip('/')
                    if base_url.endswith('/v1'):
                        base_url = base_url[:-3]
                    kwargs["base_url"] = base_url
                self._client = anthropic.AsyncAnthropic(**kwargs)
            except ImportError:
                logger.error("anthropic package not installed")
                raise
        return self._client
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 8000,
        **kwargs
    ) -> LLMResponse:
        client = await self._get_client()
        model = model or self.default_model
        
        system_message = ""
        api_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg.get("content", "")
            else:
                api_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        start_time = time.time()
        
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_message,
                messages=api_messages,
                temperature=temperature,
                **kwargs
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            content = response.content[0].text if response.content else ""
            
            return LLMResponse(
                content=content,
                model=response.model,
                provider=self.provider_name,
                prompt_tokens=response.usage.input_tokens if response.usage else 0,
                completion_tokens=response.usage.output_tokens if response.usage else 0,
                total_tokens=(response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0,
                latency_ms=latency_ms,
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None
            )
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    async def health_check(self) -> bool:
        try:
            await self.complete(
                messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10
            )
            return True
        except Exception as e:
            logger.error(f"Anthropic health check failed: {e}")
            return False


class RotatingLLMClient(LLMClient):
    """轮换LLM客户端 - 支持DeepSeek和Anthropic轮换使用"""
    
    provider_name = "rotating"
    
    def __init__(
        self,
        clients: List[LLMClient],
        strategy: RotationStrategy = RotationStrategy.ROUND_ROBIN
    ):
        self.clients = clients
        self.strategy = strategy
        self._current_index = 0
        self._call_count = 0
    
    def _get_next_client(self) -> LLMClient:
        client = self.clients[self._current_index]
        self._current_index = (self._current_index + 1) % len(self.clients)
        return client
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4000,
        **kwargs
    ) -> LLMResponse:
        self._call_count += 1
        
        if self.strategy == RotationStrategy.ROUND_ROBIN:
            client = self._get_next_client()
            logger.info(f"[RotatingLLM] Using {client.provider_name} (call #{self._call_count})")
            return await client.complete(messages, model, temperature, max_tokens, **kwargs)
        
        elif self.strategy == RotationStrategy.FALLBACK:
            for i, client in enumerate(self.clients):
                try:
                    logger.info(f"[RotatingLLM] Trying {client.provider_name}")
                    return await client.complete(messages, model, temperature, max_tokens, **kwargs)
                except Exception as e:
                    logger.warning(f"[RotatingLLM] {client.provider_name} failed: {e}")
                    if i == len(self.clients) - 1:
                        raise
            raise RuntimeError("All LLM clients failed")
        
        return await self.clients[0].complete(messages, model, temperature, max_tokens, **kwargs)
    
    async def health_check(self) -> bool:
        for client in self.clients:
            if await client.health_check():
                return True
        return False


def create_llm_client(provider: str, api_key: str, **kwargs) -> LLMClient:
    """创建单个LLM客户端"""
    if provider == "deepseek":
        return DeepSeekClient(api_key=api_key, **kwargs)
    elif provider == "anthropic":
        return AnthropicClient(api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def create_rotating_client_from_env() -> LLMClient:
    """从环境变量创建轮换客户端"""
    clients = []
    
    # DeepSeek
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_key:
        clients.append(DeepSeekClient(
            api_key=deepseek_key,
            api_base=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
            default_model=os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
        ))
        logger.info("Added DeepSeek client to rotation")
    
    # Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        clients.append(AnthropicClient(
            api_key=anthropic_key,
            api_base=os.getenv("ANTHROPIC_API_BASE"),
            default_model=os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5-20250514")
        ))
        logger.info(f"Added Anthropic client to rotation (base: {os.getenv('ANTHROPIC_API_BASE', 'default')})")
    
    if not clients:
        raise ValueError("No LLM API keys configured. Set DEEPSEEK_API_KEY or ANTHROPIC_API_KEY in .env")
    
    strategy_str = os.getenv("LLM_ROTATION_STRATEGY", "round_robin")
    strategy = RotationStrategy(strategy_str)
    
    if len(clients) == 1:
        return clients[0]
    
    return RotatingLLMClient(clients=clients, strategy=strategy)
