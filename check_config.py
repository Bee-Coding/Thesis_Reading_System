#!/usr/bin/env python3
"""
配置验证脚本 - 检查环境配置是否正确
"""
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from agents_system.runtime.config.settings import settings


def check_config():
    """检查配置"""
    print("=" * 60)
    print("Thesis Reading System - 配置检查")
    print("=" * 60)
    
    # 检查目录
    print("\n[目录配置]")
    print(f"  基础目录: {settings.base_dir}")
    print(f"  Agents目录: {settings.agents_dir} {'✓' if settings.agents_dir.exists() else '✗'}")
    print(f"  Protocols目录: {settings.protocols_dir} {'✓' if settings.protocols_dir.exists() else '✗'}")
    print(f"  Raw Papers目录: {settings.raw_papers_dir} {'✓' if settings.raw_papers_dir.exists() else '✗'}")
    
    # 检查论文
    print("\n[论文文件]")
    if settings.raw_papers_dir.exists():
        papers = list(settings.raw_papers_dir.glob("*.pdf"))
        if papers:
            for i, paper in enumerate(papers, 1):
                size_mb = paper.stat().st_size / (1024 * 1024)
                print(f"  {i}. {paper.name} ({size_mb:.1f} MB)")
        else:
            print("  未找到PDF文件")
    else:
        print("  raw_papers目录不存在")
    
    # 检查LLM配置
    print("\n[LLM配置]")
    
    # DeepSeek
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    print(f"  DeepSeek API Key: {'已配置 ✓' if deepseek_key else '未配置'}")
    if deepseek_key:
        masked = deepseek_key[:8] + "..." if len(deepseek_key) > 8 else "***"
        print(f"    Key (masked): {masked}")
        print(f"    Model: {os.environ.get('DEEPSEEK_MODEL', 'deepseek-reasoner')}")
    
    # Anthropic
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    print(f"  Anthropic API Key: {'已配置 ✓' if anthropic_key else '未配置'}")
    if anthropic_key:
        masked = anthropic_key[:8] + "..." if len(anthropic_key) > 8 else "***"
        print(f"    Key (masked): {masked}")
        print(f"    Model: {os.environ.get('ANTHROPIC_MODEL', 'claude-opus-4-5-20250514')}")
    
    # 轮换策略
    rotation_enabled = os.environ.get("LLM_ROTATION_ENABLED", "false").lower() == "true"
    rotation_strategy = os.environ.get("LLM_ROTATION_STRATEGY", "round_robin")
    if deepseek_key and anthropic_key:
        print(f"  轮换模式: 启用 ({rotation_strategy})")
    elif deepseek_key or anthropic_key:
        print(f"  轮换模式: 单一提供商")
    else:
        print(f"  轮换模式: 未配置API密钥")
    
    # 检查数据库配置
    print("\n[数据库配置]")
    print(f"  Host: {settings.database.host}")
    print(f"  Port: {settings.database.port}")
    print(f"  Database: {settings.database.database}")
    print(f"  User: {settings.database.user}")
    print(f"  Password: {'已配置 ✓' if settings.database.password else '未配置 ✗'}")
    
    # 检查质量门配置
    print("\n[质量门配置]")
    print(f"  默认阈值: {settings.quality_gate.default_threshold}")
    print(f"  最低分数: {settings.quality_gate.min_quality_score}")
    print(f"  预执行检查: {'启用' if settings.quality_gate.enable_pre_execution else '禁用'}")
    print(f"  后执行检查: {'启用' if settings.quality_gate.enable_post_execution else '禁用'}")
    
    # 总结
    print("\n" + "=" * 60)
    issues = []
    if not os.environ.get("DEEPSEEK_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        issues.append("LLM API Key未配置 (需要DEEPSEEK_API_KEY或ANTHROPIC_API_KEY)")
    if not settings.raw_papers_dir.exists():
        issues.append("raw_papers目录不存在")
    elif not list(settings.raw_papers_dir.glob("*.pdf")):
        issues.append("未找到论文PDF文件")
    
    if issues:
        print("⚠ 发现以下问题:")
        for issue in issues:
            print(f"  - {issue}")
        print("\n请按照以下步骤配置:")
        print("  1. 复制 .env.example 为 .env")
        print("  2. 编辑 .env 填入你的API密钥")
    else:
        print("✓ 配置检查通过!")
    
    print("=" * 60)
    return len(issues) == 0


if __name__ == "__main__":
    success = check_config()
    sys.exit(0 if success else 1)
