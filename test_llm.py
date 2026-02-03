#!/usr/bin/env python3
"""
LLM API连接测试脚本
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents_system.runtime.config.settings import load_dotenv
load_dotenv()

from agents_system.runtime.agent_invoker.llm_client import create_rotating_client_from_env


async def test_llm_connection():
    """测试LLM连接"""
    print("=" * 60)
    print("LLM API 连接测试")
    print("=" * 60)
    
    try:
        client = create_rotating_client_from_env()
        print(f"\n客户端类型: {client.provider_name}")
        
        if hasattr(client, 'clients'):
            print(f"轮换客户端数量: {len(client.clients)}")
            for c in client.clients:
                print(f"  - {c.provider_name}: {c.default_model}")
        
        # 测试简单请求
        print("\n[测试1] 发送简单请求...")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Reply briefly."},
            {"role": "user", "content": "Say 'Hello, Thesis Reading System!' in one line."}
        ]
        
        response = await client.complete(
            messages=messages,
            max_tokens=50,
            temperature=0.1
        )
        
        print(f"  Provider: {response.provider}")
        print(f"  Model: {response.model}")
        print(f"  Response: {response.content.strip()}")
        print(f"  Tokens: {response.total_tokens}")
        print(f"  Latency: {response.latency_ms}ms")
        
        # 如果是轮换客户端，测试第二个
        if hasattr(client, 'clients') and len(client.clients) > 1:
            print("\n[测试2] 测试轮换到下一个提供商...")
            
            response2 = await client.complete(
                messages=messages,
                max_tokens=50,
                temperature=0.1
            )
            
            print(f"  Provider: {response2.provider}")
            print(f"  Model: {response2.model}")
            print(f"  Response: {response2.content.strip()}")
            print(f"  Latency: {response2.latency_ms}ms")
        
        print("\n" + "=" * 60)
        print("✓ LLM API 连接测试通过!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_llm_connection())
    sys.exit(0 if success else 1)
