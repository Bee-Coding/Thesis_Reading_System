# GoalFlow 端到端推理脚本实现指南

**创建日期**：2026-02-14  
**目标文件**：`inference.py`  
**实现者**：你自己实现，我来指导

---

## 📋 **推理脚本功能概述**

`inference.py` 的作用是将三个训练好的模块组合起来，实现完整的 GoalFlow 推理流程。

### **完整推理流程**：

```
输入：BEV 场景特征
  ↓
Step 1: GoalPointScorer 选择目标点
  ↓
Step 2: GoalFlowMatcher 生成多条候选轨迹
  ↓
Step 3: TrajectorySelector 选择最优轨迹
  ↓
输出：最优轨迹 + 评估指标
```

---

## 🎯 **需要实现的核心函数**

### **1. load_models() - 加载训练好的模型**

**功能**：从 checkpoint 加载 Scorer 和 Matcher

**输入**：
- `scorer_checkpoint_path`: Scorer 模型路径
- `matcher_checkpoint_path`: Matcher 模型路径
- `device`: 设备（cuda 或 cpu）

**输出**：
- `scorer`: 加载好的 GoalPointScorer
- `matcher`: 加载好的 GoalFlowMatcher
- `selector`: 初始化的 TrajectorySelector
- `vocabulary`: 目标点词汇表

**实现提示**：
```python
def load_models(scorer_checkpoint_path, matcher_checkpoint_path, device):
    """
    加载训练好的模型
    
    步骤：
    1. 加载数据集以获取 vocabulary
    2. 初始化 GoalPointScorer 模型
    3. 加载 Scorer checkpoint
    4. 初始化 GoalFlowMatcher 模型
    5. 加载 Matcher checkpoint
    6. 初始化 TrajectorySelector（无需加载，规则基础）
    7. 将所有模型设为 eval 模式
    
    注意事项：
    - 模型参数要与训练时一致
    - 使用 model.load_state_dict() 加载权重
    - 记得 model.eval() 和 torch.no_grad()
    """
    pass  # 你来实现
```

---

### **2. inference_single_sample() - 单样本推理**

**功能**：对单个样本执行完整的推理流程

**输入**：
- `scorer`: GoalPointScorer 模型
- `matcher`: GoalFlowMatcher 模型
- `selector`: TrajectorySelector
- `vocabulary`: 目标点词汇表
- `sample`: 单个数据样本（包含 bev_feature, gt_trajectory 等）
- `num_candidates`: 生成的候选轨迹数量（默认 10）
- `device`: 设备

**输出**：
- `best_trajectory`: 最优轨迹 (T, 2)
- `selected_goal`: 选中的目标点 (2,)
- `all_trajectories`: 所有候选轨迹 (num_candidates, T, 2)
- `scores`: 所有轨迹的评分 (num_candidates,)

**实现提示**：
```python
def inference_single_sample(scorer, matcher, selector, vocabulary, 
                           sample, num_candidates=10, device='cpu'):
    """
    单样本推理
    
    步骤：
    1. 提取 BEV 特征
    2. 使用 Scorer 预测距离分数和 DAC 分数
    3. 选择得分最高的目标点
    4. 使用 Matcher 生成多条候选轨迹
    5. 使用 Selector 评分并选择最优轨迹
    
    关键代码：
    - pred_dis, pred_dac = scorer(vocabulary_expanded, bev_feature)
    - selected_goal_idx = pred_dis.argmax(dim=-1)
    - trajectories = matcher.generate_multiple(goal, scene, num_candidates)
    - best_traj, scores = selector(trajectories, goal, gt_traj)
    """
    pass  # 你来实现
```

---

### **3. evaluate_on_dataset() - 在数据集上评估**

**功能**：在整个测试集上运行推理并计算指标

**输入**：
- `scorer`, `matcher`, `selector`: 三个模块
- `vocabulary`: 词汇表
- `test_loader`: 测试数据加载器
- `device`: 设备

**输出**：
- `results`: 包含所有评估指标的字典
  - `avg_ade`: 平均 ADE
  - `avg_fde`: 平均 FDE
  - `min_ade`: 最小 ADE（多模态）
  - `min_fde`: 最小 FDE（多模态）

**实现提示**：
```python
def evaluate_on_dataset(scorer, matcher, selector, vocabulary, 
                       test_loader, device='cpu'):
    """
    在测试集上评估
    
    步骤：
    1. 遍历测试集
    2. 对每个样本执行推理
    3. 计算 ADE/FDE
    4. 累积统计
    5. 返回平均指标
    
    注意：
    - 使用 torch.no_grad() 禁用梯度
    - 使用 tqdm 显示进度
    - 记录每个样本的指标用于分析
    """
    pass  # 你来实现
```

---

### **4. visualize_sample() - 可视化单个样本**

**功能**：可视化推理结果

**输入**：
- `sample`: 数据样本
- `best_trajectory`: 最优轨迹
- `all_trajectories`: 所有候选轨迹
- `selected_goal`: 选中的目标点
- `vocabulary`: 词汇表
- `save_path`: 保存路径

**输出**：
- 保存可视化图片

**实现提示**：
```python
def visualize_sample(sample, best_trajectory, all_trajectories, 
                    selected_goal, vocabulary, save_path):
    """
    可视化推理结果
    
    绘制内容：
    1. 真实轨迹（绿色实线）
    2. 最优预测轨迹（红色实线）
    3. 所有候选轨迹（灰色虚线）
    4. 起始点（蓝色圆点）
    5. 真实目标点（绿色星号）
    6. 选中的目标点（红色星号）
    7. 词汇表点（灰色小点）
    
    使用 matplotlib 绘制
    """
    pass  # 你来实现
```

---

### **5. main() - 主函数**

**功能**：整合所有功能，执行完整的推理和评估

**实现提示**：
```python
def main():
    """
    主函数
    
    流程：
    1. 解析命令行参数（可选）
    2. 加载模型
    3. 加载测试数据
    4. 在测试集上评估
    5. 可视化部分样本
    6. 打印统计结果
    7. 保存结果到文件
    """
    pass  # 你来实现
```

---

## 📊 **数据流示意图**

```
sample = {
    'bev_feature': (1, 64, 32, 32),
    'trajectory': (1, 6, 2),
    'goal': (1, 2),
    'drivable_area': (1, 32, 32)
}

vocabulary: (128, 2)

↓ Step 1: Scorer

pred_dis: (1, 128)  # 距离分数
pred_dac: (1, 128)  # DAC 分数
selected_goal_idx: (1,)
selected_goal: (1, 2)

↓ Step 2: Matcher

all_trajectories: (1, 10, 6, 2)  # 10 条候选轨迹

↓ Step 3: Selector

scores: (1, 10)  # 每条轨迹的评分
best_trajectory: (1, 6, 2)  # 最优轨迹
```

---

## 🔧 **关键实现细节**

### **1. 加载 Checkpoint**n
# 加载 checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)

# 提取模型权重
model.load_state_dict(checkpoint['model_state_dict'])

# 设为评估模式
model.eval()
```

### **2. Vocabulary 扩展**

```python
# Scorer 需要 (B, N, 2) 形状的 vocabulary
B = bev_feature.shape[0]
vocabulary_expanded = vocabulary.unsqueeze(0).expand(B, -1, -1)
```

### **3. 多模态评估**

```python
# 对于多模态，计算所有候选轨迹的 ADE/FDE
all_ades = []
for traj in all_trajectories:
    ade = compute_ade(traj, gt_trajectory)
    all_ades.append(ade)

# 最小 ADE（最好的候选）
min_ade = min(all_ades)

# 平均 ADE（选中的轨迹）
avg_ade = compute_ade(best_trajectory, gt_trajectory)
```

### **4. 错误处理**

```python
try:
    # 推理代码
    pass
except Exception as e:
    print(f"Error during inference: {e}")
    import traceback
    traceback.print_exc()
    # 继续处理下一个样本
    continue
```

---

## 📁 **文件结构**

```python
# inference.py 结构

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

# 导入模型
from models.goal_point_scorer import GoalPointScorer
from models.goal_flow_matcher import GoalFlowMatcher
from models.trajectory_selector import TrajectorySelector
from data.toy_goalflow_dataset import ToyGoalFlowDataset
from config.scorer_config import ScorerConfig
from config.matcher_config import MatcherConfig


def models(...):
    pass

def inference_single_sample(...):
    pass

def evaluate_on_dataset(...):
    pass

def visualize_sample(...):
    pass

def compute_ade(...):
    pass

def compute_fde(...):
    pass

def main():
    pass


if __name__ == "__main__":
    main()
```

---

## ✅ **实现检查清单**

明天实现时，按照这个顺序：

- [ ] 1. 实现 `load_models()` - 先确保能加载模型
- [ ] 2. 实现 `compute_ade()` 和 `compute_fde()` - 简单的指标计算
- [ ] 3. 实现 `inference_single_sample()` - 核心推理逻辑
- [ ] 4. 测试单样本推理 - 确保流程正确
- [ ] 5. 实现 `evaluate_on_dataset()` - 批量评估
- [ ] 6. 实现 `visualize_sample()` - 可视化结果
- [ ] 7. 实现 `main()` - 整合所有功能
- [ ] 8. 完整测试 - 运行整个脚本

---

## 🎯 **预期输*

运行 `python inference.py` 后应该看到：

```
============================================================
GoalFlow Inference
============================================================

Loading models...
[OK] Loaded GoalPointScorer from checkpoints/scorer/best.pth
[OK] Loaded GoalFlowMatcher from checkpoints/matcher/best.pth
[OK] Initialized TrajectorySelector
[OK] Loaded vocabulary: 128 points

Loading test data...
[OK] Test samples: 400

Running inference on test set...
[Inference]: 100%|████████| 400/400 [02:30<00:00,  2.67it/s]

============================================================
Evaluation Results
============================================================
Average ADE: 0.85
Average FDE: 1.23
Min ADE (best candidate): 0.62
Min FDE (best candidate): 0.89

Visualizing samples...
[OK] Saved visualization to outputs/inference_sample_0.png
[OK] Saved visualization to outputs/inference_sample_1.png
...

============================================================
Inference completed!
============================================================
```

---

## 💡 **实现建议**

1. **先实现最简单的版本**
   - 只处理单个样本
   - 不考虑错误处理
   - 不做可视化

2. **逐步添加功能**
   - 添加批量处理
   - 添加错误处理
   - 添加可视化

3. **参考现有代码**
   - `test_train_matcher.py` 中的模型加载
   - `train_goal_scorer.py` 中的 Scorer 使用
   - `train_flow_matcher.py` 中的 Matcher 使用

4. **遇到问题时**
   - 先打印中间结果的形状
   - 检查数据类型和设备
   - 使用小数据集测试

---

## 📞 **明天开始时**

1. 告诉我训练结果（ADE/FDE 是多少）
2. 我会根据训练结果调整推理策略
3. 你开始实现，我提供指导
4. 遇到问题随时问我

---

**祝训练顺利！明天见！🚀**
