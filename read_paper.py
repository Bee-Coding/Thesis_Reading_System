#!/usr/bin/env python3
"""
论文阅读系统 - 主入口脚本
"""
import asyncio
import json
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from agents_system.runtime.config.settings import load_dotenv, settings
load_dotenv()

from agents_system.runtime.pdf_parser import PDFParser, PaperContent
from agents_system.runtime.agent_invoker.llm_client import create_rotating_client_from_env


class ThesisReader:
    """论文阅读器"""
    
    def __init__(self):
        self.parser = PDFParser()
        self.llm_client = None
        self.paper_content = None
        self.output_dir = settings.base_dir / "outputs"
        self.output_dir.mkdir(exist_ok=True)
    
    async def initialize(self):
        """初始化"""
        self.llm_client = create_rotating_client_from_env()
        print(f"LLM客户端已初始化: {self.llm_client.provider_name}")
    
    def parse_paper(self, pdf_path: str) -> PaperContent:
        """解析论文PDF"""
        print(f"\n{'='*60}")
        print(f"解析论文: {Path(pdf_path).name}")
        print(f"{'='*60}")
        
        self.paper_content = self.parser.parse(pdf_path)
        
        print(f"\n论文信息:")
        print(f"  标题: {self.paper_content.title[:80]}...")
        print(f"  页数: {self.paper_content.page_count}")
        print(f"  章节数: {len(self.paper_content.sections)}")
        print(f"  全文长度: {len(self.paper_content.full_text)} 字符")
        
        if self.paper_content.abstract:
            print(f"\n摘要 (前300字):")
            print(f"  {self.paper_content.abstract[:300]}...")
        
        print(f"\n章节结构:")
        for i, section in enumerate(self.paper_content.sections[:10], 1):
            print(f"  {i}. {section.title[:50]}")
        
        return self.paper_content
    
    async def generate_study_plan(self, execution_mode: str = "Deep_Internalization") -> dict:
        """使用Orchestrator生成学习计划"""
        print(f"\n{'='*60}")
        print(f"生成学习计划 (模式: {execution_mode})")
        print(f"{'='*60}")
        
        # 构建Orchestrator的输入
        paper_summary = f"""
论文标题: {self.paper_content.title}

摘要:
{self.paper_content.abstract}

章节结构:
{chr(10).join(f'- {s.title}' for s in self.paper_content.sections[:15])}

全文长度: {len(self.paper_content.full_text)} 字符
"""
        
        # 读取Orchestrator的System Prompt
        orchestrator_prompt_path = settings.agents_dir / "E2E-Learning-Orchestrator.md"
        if orchestrator_prompt_path.exists():
            system_prompt = orchestrator_prompt_path.read_text(encoding='utf-8')
        else:
            system_prompt = self._get_default_orchestrator_prompt()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""请为以下论文生成学习计划。

执行模式: {execution_mode}

{paper_summary}

请生成一个结构化的学习计划，包含具体的任务分解和执行顺序。输出JSON格式的执行计划。"""}
        ]
        
        print("正在调用LLM生成计划...")
        response = await self.llm_client.complete(
            messages=messages,
            max_tokens=8000,  # 增加到8000以避免计划被截断
            temperature=0.3
        )
        
        print(f"  Provider: {response.provider}")
        print(f"  Latency: {response.latency_ms}ms")
        
        # 保存计划
        plan_file = self.output_dir / f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        plan_file.write_text(response.content, encoding='utf-8')
        print(f"  计划已保存: {plan_file.name}")
        
        return {"content": response.content, "file": str(plan_file)}
    
    async def analyze_section(self, section_name: str, agent_type: str = "Scholar") -> dict:
        """使用指定Agent分析特定章节"""
        # 找到对应章节
        section = None
        for s in self.paper_content.sections:
            if section_name.lower() in s.title.lower():
                section = s
                break
        
        if not section:
            return {"error": f"未找到章节: {section_name}"}
        
        print(f"\n{'='*60}")
        print(f"分析章节: {section.title}")
        print(f"使用Agent: {agent_type}")
        print(f"{'='*60}")
        
        # 获取Agent的System Prompt
        agent_prompts = {
            "Scholar": "Scholar_Internalizer.md",
            "Code": "Code_Architect.md",
            "Validator": "Scenario_Validator.md",
            "Critic": "Strategic_Critic.md",
        }
        
        prompt_file = settings.agents_dir / agent_prompts.get(agent_type, "Scholar_Internalizer.md")
        if prompt_file.exists():
            system_prompt = prompt_file.read_text(encoding='utf-8')
        else:
            system_prompt = f"You are a {agent_type} agent. Analyze the following content deeply."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""请深入分析以下论文章节内容，提取关键知识点。

论文标题: {self.paper_content.title}

章节: {section.title}

内容:
{section.content[:8000]}

请提取:
1. 核心概念和定义
2. 关键方法和技术
3. 重要发现和结论
4. 与其他工作的关系

输出结构化的知识原子(JSON格式)。"""}
        ]
        
        print("正在分析...")
        response = await self.llm_client.complete(
            messages=messages,
            max_tokens=6000,
            temperature=0.25
        )
        
        print(f"  Provider: {response.provider}")
        print(f"  Latency: {response.latency_ms}ms")
        
        # 保存分析结果
        result_file = self.output_dir / f"analysis_{section.title[:20].replace(' ', '_')}_{datetime.now().strftime('%H%M%S')}.md"
        result_file.write_text(response.content, encoding='utf-8')
        print(f"  结果已保存: {result_file.name}")
        
        return {"content": response.content, "file": str(result_file)}
    
    async def deep_read(self) -> dict:
        """深度阅读整篇论文"""
        print(f"\n{'='*60}")
        print(f"开始深度阅读: {self.paper_content.title[:50]}...")
        print(f"{'='*60}")
        
        results = {
            "paper_title": self.paper_content.title,
            "timestamp": datetime.now().isoformat(),
            "analyses": []
        }
        
        # 分析关键章节
        key_sections = ["Introduction", "Method", "Experiment", "Conclusion"]
        
        for section_name in key_sections:
            section = self.paper_content.get_section(section_name)
            if section:
                print(f"\n>>> 分析: {section.title}")
                result = await self.analyze_section(section_name)
                results["analyses"].append({
                    "section": section.title,
                    "result": result
                })
        
        # 保存完整结果
        output_file = self.output_dir / f"deep_read_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"结果已保存: {output_file}")
        
        return results
    
    async def build_knowledge_network(self) -> dict:
        """构建知识网络"""
        print(f"\n{'='*60}")
        print(f"构建知识网络 (Knowledge_Weaver)")
        print(f"{'='*60}")
        
        # 读取现有的 algorithm_atlas.md
        atlas_path = settings.base_dir / "manifests" / "algorithm_atlas.md"
        atlas_content = ""
        if atlas_path.exists():
            atlas_content = atlas_path.read_text(encoding='utf-8')
            print(f"  已加载现有技术图谱")
        
        # 读取 paper_index.json
        index_path = settings.base_dir / "manifests" / "paper_index.json"
        index_data = {}
        if index_path.exists():
            index_data = json.loads(index_path.read_text(encoding='utf-8'))
            print(f"  已加载论文索引 (共{index_data.get('total_papers', 0)}篇)")
        
        # 读取 Knowledge_Weaver 的 System Prompt
        weaver_prompt_path = settings.agents_dir / "Knowledge_Weaver.md"
        if weaver_prompt_path.exists():
            system_prompt = weaver_prompt_path.read_text(encoding='utf-8')
        else:
            system_prompt = self._get_default_weaver_prompt()
        
        # 构建输入
        user_prompt = f"""请分析以下新生成的原子与已有知识库的关系，并更新技术图谱。

## 新论文信息
论文标题: {self.paper_content.title}
论文ID: GoalFlow_2024

## 新生成的原子
1. CONCEPT_GOALFLOW_FRAMEWORK_01 - Goal-Driven Flow Matching框架
2. METHOD_FLOW_MATCHING_INFERENCE_01 - 基于流匹配的少步/单步推理
3. FINDING_FLOWMATCHING_EFFICIENCY_01 - Flow Matching效率优势

## 当前技术图谱摘要
{atlas_content[:2000]}...

## 任务
请输出以下内容：

### 1. 技术关系分析
识别新原子与已有技术的关系（继承/演进/互补/对立），并给出关系强度（0.0-1.0）。

### 2. 更新后的技术图谱片段
提供需要添加到 algorithm_atlas.md 的新内容（Markdown格式）。

### 3. 关系原子
生成 Relation_Atom 的 JSON 格式（如果发现了重要关系）。

### 4. 知识债务清单
从论文和原子中提取的知识债务项。

请按照 Knowledge_Weaver 的输出标准进行分析。
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        print("正在调用LLM分析知识关系...")
        response = await self.llm_client.complete(
            messages=messages,
            max_tokens=8000,
            temperature=0.2
        )
        
        print(f"  Provider: {response.provider}")
        print(f"  Latency: {response.latency_ms}ms")
        
        # 保存分析结果
        network_file = self.output_dir / f"knowledge_network_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        network_file.write_text(response.content, encoding='utf-8')
        print(f"  知识网络分析已保存: {network_file.name}")
        
        # 尝试更新 algorithm_atlas.md（追加新内容）
        try:
            # 简单追加策略：在文件末尾添加更新日志
            update_log = f"\n\n### {datetime.now().strftime('%Y-%m-%d')}\n"
            update_log += f"- **新增论文**: {self.paper_content.title}\n"
            update_log += f"- **新增原子**: 3个 (Concept, Method, Finding)\n"
            update_log += f"- **详细分析**: 参见 `{network_file.name}`\n"
            
            with open(atlas_path, 'a', encoding='utf-8') as f:
                f.write(update_log)
            
            print(f"  ✓ algorithm_atlas.md 已更新")
        except Exception as e:
            print(f"  ⚠ 更新 algorithm_atlas.md 失败: {e}")
        
        return {
            "content": response.content,
            "file": str(network_file)
        }
    
    def _get_default_weaver_prompt(self) -> str:
        return """You are Knowledge_Weaver, responsible for analyzing relationships between new knowledge atoms and existing knowledge base.

Your task is to:
1. Identify technical relationships (inheritance, evolution, complement, conflict)
2. Calculate relationship strength (0.0-1.0)
3. Extract knowledge gaps and assumptions
4. Update the technology atlas

Output structured analysis in Markdown format."""
    
    def _get_default_orchestrator_prompt(self) -> str:
        return """You are E2E-Learning-Orchestrator, responsible for creating comprehensive learning plans for academic papers.

Your task is to analyze the paper structure and create a detailed execution plan that includes:
1. Paper overview and key objectives
2. Task breakdown for each major section
3. Dependencies between tasks
4. Quality checkpoints
5. Expected outputs (knowledge atoms)

Output your plan in a structured format."""


async def main():
    """主函数"""
    print("="*60)
    print("论文阅读系统 - Thesis Reading System")
    print("="*60)
    
    # 初始化
    reader = ThesisReader()
    await reader.initialize()
    
    # 解析GoalFlow论文
    pdf_path = settings.raw_papers_dir / "GoalFlow_ Goal-Driven Flow Matching for Multimodal Trajectories Generation.pdf"
    
    if not pdf_path.exists():
        print(f"错误: 论文文件不存在 - {pdf_path}")
        return
    
    # 1. 解析论文
    reader.parse_paper(str(pdf_path))
    
    # 2. 生成学习计划
    plan = await reader.generate_study_plan("Deep_Internalization")
    
    print("\n" + "="*60)
    print("学习计划预览:")
    print("="*60)
    print(plan["content"][:2000])
    
    # 3. 执行深度分析
    print("\n" + "="*60)
    print("开始深度分析关键章节...")
    print("="*60)
    
    results = await reader.deep_read()
    
    print("\n" + "="*60)
    print("深度分析完成!")
    print(f"分析了 {len(results['analyses'])} 个章节")
    print("="*60)
    
    # 4. 构建知识网络
    print("\n" + "="*60)
    print("开始构建知识网络...")
    print("="*60)
    
    network_result = await reader.build_knowledge_network()
    
    print("\n" + "="*60)
    print("知识网络构建完成!")
    print(f"分析结果已保存")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
