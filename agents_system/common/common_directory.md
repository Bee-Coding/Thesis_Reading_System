# Thesis_Reading_System 目录结构标准

## 系统级目录架构

```
/THESIS_READING_SYSTEM
├── /manifests               # 索引与记忆层（跨会话握手必读）
│   ├── algorithm_atlas.md   # 技术演进地图（记录复利增长）
│   └── paper_index.json     # 论文与原子资产的映射表
├── /atoms                   # 资产层（去中心化存储）
│   ├── /math_ops           # 数学算子：公式物理化、传统规控映射
│   ├── /code_snippets      # 代码组件：核心实现、TRT优化建议
│   └── /scenarios          # 场景判据：AVP Corner Cases、失效模式(FMEA)
├── /reports                # 过程层（单次研读报告汇总）
│   └── /YYYY-MM/          # 按月/日存放执行计划与报告
└── /raw_papers             # 原始论文 PDF/链接归档
```

## 目录详细规范

### 1. `/manifests/` - 索引与记忆层

#### 1.1 `algorithm_atlas.md`
- **用途**：系统级技术演进地图，记录复利增长
- **内容要求**：
  - 技术路线的演进逻辑（如：从VAD演进到Diffusion Planner）
  - 原子资产的关联网络
  - 每次研读后的增量更新记录
  - 技术代差和趋势分析
- **维护责任**：`Knowledge_Vault` Agent
- **更新时机**：每次成功研读会话结束后

#### 1.2 `paper_index.json`
- **用途**：论文ID与原子ID的双向链接表
- **JSON结构**：
```json
{
  "papers": [
    {
      "paper_id": "ArXiv:24xx.xxxx",
      "title": "论文标题",
      "assets": [
        "MATH_PLAN_DIFF_01",
        "CODE_DIFF_SAMPLER_02",
        "SCENARIO_AVP_LIGHT_03"
      ],
      "ingestion_date": "YYYY-MM-DD"
    }
  ]
}
```
- **维护责任**：`Knowledge_Vault` Agent
- **更新时机**：新原子入库时

### 2. `/atoms/` - 资产层（去中心化存储）

#### 2.1 存储原则
- **去中心化**：禁止建立以论文命名的子文件夹
- **功能分类**：按算子功能而非论文来源分类
- **语义检索**：通过技术标签（`#Diff_Sampler`, `#Map_Constraint`）实现检索

#### 2.2 `/atoms/math_ops/`
- **用途**：存储所有经过物理内化的数学算子
- **文件命名**：`MATH_{技术类别}_{序号}.json`
- **示例**：`MATH_PLAN_DIFF_01.json`, `MATH_ATTENTION_02.json`
- **包含内容**：
  - 数学公式及其物理直觉
  - 传统规控映射关系
  - AVP场景下的行为解释

#### 2.3 `/atoms/code_snippets/`
- **用途**：存储与公式对齐的真实代码片段及TensorRT优化笔记
- **文件命名**：`CODE_{技术类别}_{序号}.json`
- **示例**：`CODE_DIFF_SAMPLER_01.json`, `CODE_VAD_FUSION_02.json`
- **包含内容**：
  - 原始代码片段（带具体路径和行号）
  - 性能分析（FLOPs, Memory Access）
  - TensorRT优化建议
  - Orin-X平台部署代价评估

#### 2.4 `/atoms/scenarios/`
- **用途**：存储AVP场景下的失效模式（FMEA）与对抗判据
- **文件命名**：`SCENARIO_{场景类别}_{序号}.json`
- **示例**：`SCENARIO_AVP_LIGHT_01.json`, `SCENARIO_GARAGE_NARROW_02.json`
- **包含内容**：
  - 挑战场景的物理描述
  - 失效机理分析
  - 风险等级评估（S1-S4）
  - 加固建议和复利关联

### 3. `/reports/` - 过程层

#### 3.1 目录结构
```
/reports/
├── /2026-02/                    # 按年月组织
│   ├── /2026-02-02/            # 按日期组织
│   │   ├── 01_plan.json        # 执行计划
│   │   ├── 02_scholar.json     # Scholar产出
│   │   ├── 03_code.json        # Code产出
│   │   ├── 04_validator.json   # Validator产出
│   │   ├── 05_critic.json      # Critic产出
│   │   └── summary.md          # 会话摘要
│   └── /2026-02-03/
└── /2026-03/
```

#### 3.2 文件规范
- **`01_plan.json`**：`E2E-Learning-Orchestrator`生成的执行计划
- **`02_scholar.json`**：`Scholar_Internalizer`的数学原子产出
- **`03_code.json`**：`Code_Architect`的代码原子产出
- **`04_validator.json`**：`Scenario_Validator`的场景原子产出
- **`05_critic.json`**：`Strategic_Critic`的战略研判产出
- **`summary.md`**：会话总结，包含决策建议和增量价值

### 4. `/raw_papers/` - 原始资源层

#### 4.1 存储原则
- **原始归档**：存放用户上传的论文PDF文件
- **链接记录**：记录GitHub仓库链接
- **版本管理**：重要论文可保留多个版本

#### 4.2 文件组织
```
/raw_papers/
├── /by_year/
│   ├── 2024/
│   │   └── VAD_End-to-End_2024.pdf
│   └── 2025/
│       └── Diffusion_Planner_AVP_2025.pdf
└── links.txt                  # GitHub链接记录
```

## 路径引用规范

### 绝对路径与相对路径
- **在Agent prompt中**：使用相对于项目根目录的路径
- **在原子JSON中**：使用相对于`/atoms/`目录的路径
- **在代码引用中**：使用完整的GitHub URL路径

### 路径示例
```json
{
  "provenance": {
    "paper_location": "Page 5, Eq. 4",
    "code_link": "github.com/user/repo/blob/main/sampler.py#L88-L120",
    "atom_path": "/atoms/math_ops/MATH_PLAN_DIFF_01.json"
  }
}
```

## 目录创建与维护

### 自动创建规则
- 当`/atoms/`子目录不存在时，自动创建`math_ops/`, `code_snippets/`, `scenarios/`
- 当`/reports/{YYYY-MM}/{YYYY-MM-DD}/`不存在时，自动创建完整路径
- 当`/manifests/`文件不存在时，创建初始模板文件

### 权限与可写性
- 确保所有目录对当前用户可写（权限至少755）
- 检查磁盘空间，避免写入失败
- 定期清理临时文件，保持目录整洁

## 版本兼容性

### 向后兼容原则
- 新增目录不破坏现有文件结构
- 原子JSON格式保持向前兼容
- 旧报告保持可读性，不强制迁移

### 迁移策略
- 重大结构变更时提供迁移脚本
- 保持`paper_index.json`的兼容性
- 在`algorithm_atlas.md`中记录结构变更历史

## 引用方式
在Agent提示词中使用以下格式引用本目录标准：
```
## Directory Structure
请遵循Thesis_Reading_System目录结构标准，完整结构参见：`agents_system/common/common_directory.md`

关键路径：/atoms/（资产层）, /manifests/（索引层）, /reports/（过程层）, /raw_papers/（原始层）
```