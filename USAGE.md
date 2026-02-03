# Thesis Reading System - 使用指南

## 目录

1. [基本使用](#基本使用)
2. [高级功能](#高级功能)
3. [自定义配置](#自定义配置)
4. [API参考](#api参考)
5. [常见问题](#常见问题)

## 基本使用

### 分析单篇论文

```bash
# 1. 将PDF放入raw_papers目录
cp your_paper.pdf raw_papers/

# 2. 运行分析
python3 read_paper.py
```

### 查看分析结果

```bash
# 查看学习计划
cat outputs/plan_*.md

# 查看章节分析
cat outputs/analysis_*.md

# 查看知识原子
cat atoms/concepts/*.json
cat atoms/methods/*.json
cat atoms/findings/*.json
```

## 高级功能

### 1. 自定义分析章节

编辑 `read_paper.py`，修改 `deep_read()` 方法：

```python
async def deep_read(self):
    # 自定义要分析的章节
    key_sections = ["Introduction", "Method", "Related Work", "Conclusion"]
    
    for section_name in key_sections:
        section = self.paper_content.get_section(section_name)
        if section:
            result = await self.analyze_section(section_name, agent_type="Scholar")
            # 处理结果...
```

### 2. 使用不同的Agent

```python
# 使用Scholar Agent分析理论
await reader.analyze_section("Method", agent_type="Scholar")

# 使用Code Architect分析实现
await reader.analyze_section("Implementation", agent_type="Code")

# 使用Validator分析实验
await reader.analyze_section("Experiments", agent_type="Validator")

# 使用Critic进行评估
await reader.analyze_section("Results", agent_type="Critic")
```

### 3. 批量处理多篇论文

创建批处理脚本 `batch_process.py`：

```python
import asyncio
from pathlib import Path
from read_paper import ThesisReader

async def process_papers():
    reader = ThesisReader()
    await reader.initialize()
    
    papers = Path("raw_papers").glob("*.pdf")
    
    for paper in papers:
        print(f"Processing: {paper.name}")
        reader.parse_paper(str(paper))
        await reader.generate_study_plan()
        await reader.deep_read()

if __name__ == "__main__":
    asyncio.run(process_papers())
```

### 4. 导出知识图谱

```python
import json
from pathlib import Path

def export_knowledge_graph():
    atoms = []
    
    # 收集所有原子
    for atom_file in Path("atoms").rglob("*.json"):
        with open(atom_file) as f:
            atoms.append(json.load(f))
    
    # 构建关系图
    graph = {
        "nodes": atoms,
        "edges": []  # 可以添加原子之间的关系
    }
    
    # 导出
    with open("knowledge_graph.json", "w") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
```

## 自定义配置

### 执行模式

在 `.env` 文件中设置：

```bash
DEFAULT_EXECUTION_MODE=Deep_Internalization
```

可选值：
- `Deep_Internalization` - 深度内化（默认）
- `Quick_Assessment` - 快速评估
- `Engineering_Reproduction` - 工程复现
- `Validation_Focused` - 验证聚焦

### LLM参数调整

编辑 `agents_system/runtime/config/settings.py`：

```python
AGENT_LLM_PARAMS = {
    AgentType.SCHOLAR: {
        "temperature": 0.25,      # 降低随机性
        "maxs": 12000,      # 增加输出长度
        "timeout": 600,           # 超时时间（秒）
    },
}
```

### PDF解析器调整

如果遇到特殊格式的PDF，可以修改 `agents_system/runtime/pdf_parser/parser.py`：

```python
# 添加新的章节模式
SECTION_PATTERNS = [
    r'^(\d+)\.\s*(YourCustomSection)',
    # ... 其他模式
]
```

## API参考

### PDFParser

```python
from agents_system.runtime.pdf_parser import PDFParser

parser = PDFParser()
content = parser.parse("path/to/paper.pdf")

# 访问解析结果
print(content.title)           # 标题
print(content.abstract)        # 摘要
print(content.sections)        # 章节列表
print(content.references)      # 参考文献
print(content.page_count)      # 页数
```

### LLMClient

```python
from agents_system.runtime.agent_invoker.llm_client import create_rotating_client_from_env

# 创建客户端
client = create_rotating_client_from_env()

# 调用LLM
response = await client.complete(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Analyze this paper..."}
    ],
    max_tokens=4000,
    temperature=0.3
)

print(response.content)        # 响应内容
print(response.provider)       # 使用的提供商
print(response.latency_ms)     # 延迟（毫秒）
```

### ThesisReader

```python
from read_paper import ThesisReader

reader = ThesisReader()
await reader.initialize()

# 解析论文
reader.parse_paper("raw_papers/paper.pdf")

# 生成学习计划
plan = await reader.generate_study_plan("Deep_Internalization")

# 分析章节
result = await reader.analyze_section("Method", agent_type="Scholar")

# 深度阅读
results = await reader.deep_read()
```

## 常见问题

### Q: PDF解析失败怎么办？

A: 系统会自动尝试两种解析器（pdfplumber和PyMuPDF）。如果都失败：
1. 检查PDF是否损坏
2. 尝试用其他工具转换PDF格式
3. 手动提取文本并创建TXT文件

### Q: 章节识别不完整？

A: PDF解析器已优化支持多种格式，但某些特殊PDF可能需要调整正则表达式。
检查 `parser.py` 中的 `SECTION_PATTERNS`。

### Q: LLM API调用超时？

A: 增加超时时间：
```python
AGENT_LLM_PARAMS = {
    AgentType.SCHOLAR: {
        "timeout": 1200,  # 增加到20分钟
    },
}
``` 如何减少API成本？

A: 
1. 使用 `fallback` 策略优先使用DeepSeek
2. 减少 `max_tokens` 参数
3. 只分析关键章节而非全文

### Q: 知识原子如何复用？

A: 
1. 查看 `manifests/paper_index.json` 找到相关原子
2. 读取 `atoms/` 目录下的JSON文件
3. 在新的分析中引用已有原子的 `asset_id`

### Q: 如何添加新的Agent？

A: 
1. 在 `agents_system/` 创建新的Agent配置文件（.md）
2. 在 `settings.py` 的 `AgentType` 枚举中添加新类型
3. 在 `AGENT_LLM_PARAMS` 中配置参数
4. 在 `read_paper.py` 中调用新Agent

## 性能优化建议

### 1. 使用缓存

```python
# 缓存PDF解析结果
import pickle

def parse_with_cache(pdf_path):
    cache_file = f"{pdf_path}.cache"
    if Path(cache_file).exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    content = parser.parse(pdf_path)
    with open(cache_file, 'wb') as f:
        pickle.dump(content, f)
    
    return content
```

### 2. 并行处理

```python
import asyncio

async def parallel_analysis():
    tasks = [
        reader.analyze_section("Introduction"),
        reader.analyze_section("Method"),
        reader.analyze_section("Experiments"),
    ]
    results = await asyncio.gather(*tasks)
    return results
```

### 3. 增量更新

只分析新增的论文，避免重复处理：

```python
def get_unprocessed_papers():
    with open("manifests/paper_indexn") as f:
        processed = {p["title"] for p in json.load(f)["papers"]}
    
    all_papers = Path("raw_papers").glob("*.pdf")
    return [p for p in all_papers if p.stem not in processed]
```

## 更多资源

- Agent配置文档：`agents_system/*.md`
- 协议文档：`agents_system/protocols/*.md`
- 示例输出：`outputs/`

如有其他问题，请查看项目Issue或联系维护者。
