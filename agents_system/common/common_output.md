# Thesis_Reading_System 原子输出格式标准

## 基础原子结构

所有Agent产出必须遵循以下基础原子JSON结构：

```json
{
  "asset_id": "资产唯一标识符",
  "category": "原子类别",
  "data_status": "数据验证状态",
  "metadata": {
    "created_at": "ISO 8601时间戳",
    "created_by": "创建Agent",
    "version": "原子版本号",
    "tags": ["技术标签1", "技术标签2"]
  },
  "content": {
    // 具体内容字段，根据原子类别变化
  },
  "provenance": {
    "paper_id": "论文标识",
    "paper_location": "论文位置",
    "code_link": "代码链接",
    "atom_path": "原子存储路径"
  },
  "delta_audit": {
    "existing_assets": ["已有资产ID"],
    "incremental_value": "增量价值描述",
    "contradiction_marked": true/false
  }
}
```

## 原子类别定义

### 1. 数学原子 (Math_Atom)
- **category**: `"Math_Ops"`
- **asset_id格式**: `MATH_{技术类别}_{序号}` (如：`MATH_PLAN_DIFF_01`)
- **用途**: 存储经过物理内化的数学公式

### 2. 代码原子 (Code_Atom)
- **category**: `"Code_Snippets"`
- **asset_id格式**: `CODE_{技术类别}_{序号}` (如：`CODE_DIFF_SAMPLER_01`)
- **用途**: 存储真实代码片段及优化建议

### 3. 场景原子 (Scenario_Atom)
- **category**: `"Scenarios"`
- **asset_id格式**: `SCENARIO_{场景类别}_{序号}` (如：`SCENARIO_AVP_LIGHT_01`)
- **用途**: 存储AVP挑战场景与失效模式

### 4. 战略决策 (Strategic_Decision)
- **category**: `"Strategic"`
- **asset_id格式**: `DECISION_{技术类别}_{序号}` (如：`DECISION_E2E_DIFF_01`)
- **用途**: 存储量产可行性审计意见

## 字段定义标准

### 1. 顶层字段
- **asset_id** (字符串): 资产唯一标识符，必须全局唯一
- **category** (字符串): 原子类别，必须是上述四种之一
- **data_status** (字符串): 数据验证状态
  - `"Verified_Source_Anchored"` - 来源已验证且锚定
  - `"Unverified_No_Source"` - 未验证，缺少来源
  - `"Partially_Verified"` - 部分验证
  - `"Contradiction_Detected"` - 检测到冲突

### 2. 元数据字段 (metadata)
- **created_at** (字符串): ISO 8601格式时间戳，如 `"2026-02-02T20:30:00Z"`
- **created_by** (字符串): 创建此原子的Agent名称
  - `"Scholar_Internalizer"` - 数学原子
  - `"Code_Architect"` - 代码原子  
  - `"Scenario_Validator"` - 场景原子
  - `"Strategic_Critic"` - 战略决策
- **version** (字符串): 原子版本号，格式 `"1.0"`, `"2.1"`等
- **tags** (数组): 技术标签列表，用于语义检索
  - 示例: `["#Diff_Sampler", "#Map_Constraint", "#AVP_Narrow"]`

### 3. 来源追溯字段 (provenance)
- **paper_id** (字符串): 论文标识，如 `"ArXiv:24xx.xxxx"` 或论文标题
- **paper_location** (字符串): 论文中的具体位置，格式 `"Page X, Eq. Y"`
- **code_link** (字符串): GitHub代码链接，格式 `"github.com/user/repo/blob/main/file.py#L88-L120"`
- **atom_path** (字符串): 原子在系统中的存储路径，如 `"/atoms/math_ops/MATH_PLAN_DIFF_01.json"`

### 4. 增量审计字段 (delta_audit)
- **existing_assets** (数组): 已存在的相似资产ID列表
- **incremental_value** (字符串): 增量价值描述，回答"本次更新相比已有资产提升了什么？"
- **contradiction_marked** (布尔): 是否检测到与已有资产的冲突

## 类别特定格式

### Math_Atom 完整格式
```json
{
  "asset_id": "MATH_PLAN_DIFF_01",
  "category": "Math_Ops",
  "data_status": "Verified_Source_Anchored",
  "metadata": {
    "created_at": "2026-02-02T20:30:00Z",
    "created_by": "Scholar_Internalizer",
    "version": "1.0",
    "tags": ["#Diffusion", "#Trajectory", "#AVP"]
  },
  "content": {
    "mathematical_expression": "\\nabla_x \\log p(x|y)",
    "physical_intuition": "这是一个带噪声抑制的轨迹细化过程。在 AVP 场景下，x 代表了模型对地库动态障碍物不确定性的容忍度。",
    "traditional_mapping": "类似于传统优化中的**带有动量项的迭代寻优**。",
    "avp_relevance": "地库窄路博弈引导力",
    "assumptions": ["高斯噪声假设", "环境静态假设"],
    "boundary_conditions": "在暗光、强遮挡环境下假设可能失效"
  },
  "provenance": {
    "paper_id": "ArXiv:24xx.xxxx",
    "paper_location": "Page 6, Eq. 8",
    "code_link": "",
    "atom_path": "/atoms/math_ops/MATH_PLAN_DIFF_01.json"
  },
  "delta_audit": {
    "existing_assets": ["MATH_VAD_ATTENTION_01"],
    "incremental_value": "相比于库中的 MATH_VAD_ATTENTION_01，该公式增加了一项环境特征引导，使得泊车轨迹更贴合库位边缘。",
    "contradiction_marked": false
  }
}
```

### Code_Atom 完整格式
```json
{
  "asset_id": "CODE_DIFF_SAMPLER_01",
  "category": "Code_Snippets",
  "data_status": "Verified_Source_Anchored",
  "metadata": {
    "created_at": "2026-02-02T20:30:00Z",
    "created_by": "Code_Architect",
    "version": "1.0",
    "tags": ["#Diffusion_Sampler", "#Trajectory", "#TensorRT"]
  },
  "content": {
    "technical_category": "Trajectory_Sampling",
    "source_code": "def diffusion_sampler(x, t, guidance_scale):\n    # 具体实现代码\n    return sampled_trajectory",
    "code_location": "projects/mmdet3d_plugin/models/utils/diffusion_utils.py (L142-L185)",
    "logic_description": "该实现采用了步长衰减策略，在 AVP 窄路博弈时能显著提高近端轨迹的一致性。",
    "performance_note": {
      "complexity": "O(N * D)，其中 N 为去噪步数，D 为轨迹维度",
      "deployment_suggestion": "建议将采样循环在 TensorRT 中做算子融合，预计在 Orin-X 上单次 Inference 延迟增加 12ms。",
      "memory_footprint": "显存占用约 256MB",
      "bottleneck_analysis": "采样循环是主要瓶颈，占总延迟的 85%"
    }
  },
  "provenance": {
    "paper_id": "ArXiv:24xx.xxxx",
    "paper_location": "Page 8, Algorithm 1",
    "code_link": "github.com/user/repo/blob/main/projects/mmdet3d_plugin/models/utils/diffusion_utils.py#L142-L185",
    "atom_path": "/atoms/code_snippets/CODE_DIFF_SAMPLER_01.json"
  },
  "delta_audit": {
    "existing_assets": ["CODE_VAD_SAMPLER_01"],
    "incremental_value": "相比 VAD 采样器，增加了步长衰减策略，在 AVP 窄路场景下轨迹一致性提升 15%。",
    "contradiction_marked": false
  }
}
```

### Scenario_Atom 完整格式
```json
{
  "asset_id": "SCENARIO_AVP_LIGHT_01",
  "category": "Scenarios",
  "data_status": "Verified_Source_Anchored",
  "metadata": {
    "created_at": "2026-02-02T20:30:00Z",
    "created_by": "Scenario_Validator",
    "version": "1.0",
    "tags": ["#Light_Transition", "#AVP_Entrance", "#Perception_Failure"]
  },
  "content": {
    "technical_association": "针对 Diffusion Planner 的时空特征融合模块。",
    "challenge_scenario": "地库入口处，从强阳光瞬间进入暗光的动态感知跳变（Lux 差值 > 10,000）。",
    "failure_mechanism": "模型在特征提取阶段由于归一化算子（Normalization）失效，导致 Trajectory Query 在隐含空间发生漂移，规控输出表现为'刹车点头'。",
    "risk_assessment": {
      "severity": "S3 (High) - 可能导致追尾",
      "exposure_probability": "Medium (在阳光明媚的日子常见)",
      "controllability": "Low (驾驶员反应时间不足)"
    },
    "audit_note": "基于代码 `vad.py#L254` 处的特征融合逻辑推演。",
    "reinforcement_suggestion": "建议在 `[Code_Atom: Feature_Fusion]` 中加入抗曝光干扰的预处理算子。"
  },
  "provenance": {
    "paper_id": "",
    "paper_location": "",
    "code_link": "github.com/user/repo/blob/main/vad.py#L254",
    "atom_path": "/atoms/scenarios/SCENARIO_AVP_LIGHT_01.json"
  },
  "delta_audit": {
    "existing_assets": ["SCENARIO_SENSOR_FAILURE_01"],
    "incremental_value": "新增光照跳变场景，补充了感知失效的多样性。",
    "contradiction_marked": false
  }
}
```

### Strategic_Decision 完整格式
```json
{
  "asset_id": "DECISION_E2E_DIFF_01",
  "category": "Strategic",
  "data_status": "Verified_Source_Anchored",
  "metadata": {
    "created_at": "2026-02-02T20:30:00Z",
    "created_by": "Strategic_Critic",
    "version": "1.0",
    "tags": ["#Diffusion_Planner", "#AVP", "#ROI_Audit"]
  },
  "content": {
    "strategic_tone": "Watch (暂缓复现，仅提取算子)",
    "core_reasoning": [
      "**性能赤字**: 根据 `[Code_Atom: DIFF_SAMPLER]` 测算，Orin-X 平台 Inference 延迟达 45ms，将使 AVP 规划频率降至 10Hz 以下，无法满足窄路博弈实时性。",
      "**数学依赖**: 算法高度依赖 `[Math_Atom: GAUSS_NOISE]` 假设，在 `[Scenario_Atom: RAINY_NIGHT]`（雨夜地库反射）中存在失效风险。"
    ],
    "compound_suggestion": "仅指令 **Knowledge_Vault** 归档其'多模态轨迹去重算子'，暂不启动全量工程复现。",
    "roi_analysis": {
      "computational_cost": "High (需要额外 15% 算力)",
      "safety_improvement": "Medium (在标准场景下提升有限)",
      "data_dependency": "High (需要大量长尾数据)",
      "overall_roi": "Low (当前阶段不推荐量产)"
    }
  },
  "provenance": {
    "paper_id": "ArXiv:24xx.xxxx",
    "paper_location": "Page 6-8",
    "code_link": "github.com/user/repo/blob/main/infer.py",
    "atom_path": "/reports/2026-02/2026-02-02/05_critic.json"
  },
  "delta_audit": {
    "existing_assets": ["DECISION_VAD_PLANNER_01"],
    "incremental_value": "相比 VAD 方案，Diffusion Planner 在轨迹多样性上有所提升，但算力代价过高。",
    "contradiction_marked": false
  }
}
```

## 输出文件命名规范

### 原子文件命名
```
/atoms/{category}/{asset_id}.json
```
- **示例**: `/atoms/math_ops/MATH_PLAN_DIFF_01.json`

### 报告文件命名
```
/reports/{YYYY-MM}/{YYYY-MM-DD}/{文件编号}_{类型}.json
```
- **`01_plan.json`** - 执行计划
- **`02_scholar.json`** - Scholar产出（Math_Atom数组）
- **`03_code.json`** - Code产出（Code_Atom数组）
- **`04_validator.json`** - Validator产出（Scenario_Atom数组）
- **`05_critic.json`** - Critic产出（Strategic_Decision数组）

## 编码与格式要求

### 编码标准
- **字符编码**: UTF-8
- **换行符**: LF (Unix风格)
- **BOM**: 禁止使用BOM

### JSON格式要求
- **缩进**: 2个空格（禁止使用Tab）
- **引号**: 双引号
- **逗号**: 不允许尾部逗号
- **排序**: 字段按上述定义顺序排列
- **空值**: 使用`null`而不是空字符串

### 内容长度限制
- **asset_id**: 不超过50字符
- **物理直觉描述**: 不超过300字符
- **增量价值描述**: 不超过200字符
- **标签**: 每个不超过30字符，总数不超过5个

## 验证规则

### 必需字段验证
每种原子类型必须包含以下必需字段：

| 类别 | 必需字段 |
|------|----------|
| Math_Atom | mathematical_expression, physical_intuition, paper_location |
| Code_Atom | source_code, code_location, performance_note.complexity |
| Scenario_Atom | challenge_scenario, failure_mechanism, risk_assessment.severity |
| Strategic_Decision | strategic_tone, core_reasoning, roi_analysis.overall_roi |

### 数据质量验证
1. **来源验证**: `paper_location` 或 `code_link` 至少一项非空
2. **状态验证**: `data_status` 必须是预定义值之一
3. **ID验证**: `asset_id` 必须符合命名规范
4. **路径验证**: `atom_path` 必须符合目录结构

### 完整性检查
- JSON必须可解析（通过`jq .`验证）
- 必需字段不能为`null`或空字符串
- 数组字段不能为`null`，可以为空数组`[]`
- 对象字段不能为`null`，可以为空对象`{}`

## 错误输出格式

### 空结果输出
```json
[]
```

### 错误原子输出（仅用于调试）
```json
{
  "asset_id": "ERROR_INVALID_SOURCE",
  "category": "Error",
  "data_status": "Unverified_No_Source",
  "metadata": {
    "created_at": "2026-02-02T20:30:00Z",
    "created_by": "Agent_Name",
    "version": "1.0",
    "tags": ["#Error"]
  },
  "content": {
    "error_type": "来源缺失",
    "error_message": "论文未提供公式具体位置，无法生成验证原子",
    "recovery_action": "请提供论文页码或公式编号",
    "partial_data": "已识别的公式内容（如有）"
  },
  "provenance": {
    "paper_id": "",
    "paper_location": "",
    "code_link": "",
    "atom_path": ""
  },
  "delta_audit": {
    "existing_assets": [],
    "incremental_value": "",
    "contradiction_marked": false
  }
}
```

## 引用方式
在Agent提示词中使用以下格式引用本输出标准：
```
## 输出格式
请遵循Thesis_Reading_System原子输出格式标准，完整格式参见：`agents_system/common/common_output.md`

输出要求：标准JSON原子结构、必需字段完整、来源锚定验证、增量价值明确
```