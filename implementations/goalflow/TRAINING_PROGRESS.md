# GoalFlow 训练脚本开发进度

**最后更新**：2026-02-12  
**当前状态**：GoalPointScorer 训练脚本完成，准备开发 GoalFlowMatcher 训练脚本

---

## ✅ 已完成工作

### 1. Toy 数据集生成（2026-02-12）

**文件**：
- `data/generate_toy_data.py` - 数据生成脚本
- `data/toy_goalflow_dataset.py` - PyTorch Dataset 类
- `data/test_dataset_simple.py` - 简单测试脚本
- `data/README.md` - 使用文档

**数据集特点**：
- 1000 条轨迹（训练集 800 + 验证集 200）
- 4 个多模态目标区域
- 128 个词汇点（K-means 聚类）
- BEV 特征：64 通道，32×32
- 可行驶区域：圆形 mask

**测试结果**：
```bash
✅ 数据加载成功
✅ DataLoader 正常工作
✅ 可视化生成成功（sample_visualization.png, batch_visualization.png）
```

---

### 2. GoalPointScorer 训练脚本（2026-02-12）

**文件**：
- `config/scorer_config.py` - 训练配置
- `train_goal_scorer.py` - 训练脚本
- `test_train_scorer.py` - 快速测试脚本

**实现的核心函数**：
```python
def compute_target_labels(vocabulary, gt_goals):
    """计算最近词汇点索引"""
    
def compute_accuracy(pred_scores, target_idx, k=1):
    """计算 Top-K 准确率"""
    
def train_one_epoch(model, train_loader, vocabulary, optimizer, device, config):
    """训练一个 epoch"""
    
def validate(model, val_loader, vocabulary, device, config):
    """验证模型（Top-1/Top-5）"""
```

**测试结果**（3 epochs, CPU）：
```
Epoch 1/3:
  Train Loss: 4.8535, Train Acc: 0.0063
  Val Loss: 4.8520, Top-1 Acc: 0.0000, Top-5 Acc: 0.0250
  ✅ 模型保存成功

✅ 训练流程完全正常
✅ 无任何错误
✅ 代码逻辑验证通过
```

**代码修复记录**：
1. ✅ 修复 `compute_accuracy()` 的 Top-K 计算逻辑
2. ✅ 修复 `train_one_epoch()` 的 return 位置错误
3. ✅ 修复模型调用参数（vocabulary 扩展）
4. ✅ 修复损失函数返回值处理（tuple）
5. ✅ 修复变量名拼写错误

**训练配置**：
```python
# config/scorer_config.py
vocab_size = 128
hidden_dim = 256
num_layers = 4
batch_size = 32
learning_rate = 1e-4
num_epochs = 100
lambda_dis = 1.0
lambda_dac = 0.5
```

---

## 🔄 下一步工作

### 1. 实现 GoalFlowMatcher 训练脚本

**需要创建**：
- `train_flow_matcher.py` - FlowMatcher 训练脚本

**核心功能**：
```python
def train_one_epoch(model, train_loader, vocabulary, scorer, optimizer, device, config):
    """
    训练一个 epoch
    
    关键步骤：
    1. 采样时间 t ~ U(0, 1)
    2. 采样噪声轨迹 x_0 ~ N(0, I)
    3. 插值得到 x_t = (1-t)*x_0 + t*x_1
    4. 预测速度场 v_pred = model(x_t, goal, scene, t)
    5. 计算损失 loss = ||v_pred - (x_1 - x_0)||²
    """

def validate(model, val_loader, vocabulary, scorer, device, config):
    """
    验证模型
    
    关键步骤：
    1. 使用 Scorer 选择目标点（或使用 gt_goal）
    2. 生成轨迹：model.generate()
    3. 计算 ADE/FDE 指标
    """
```

**训练策略**：
- 选项 1：使用 gt_goal（简化训练）
- 选项 2：使用 Scorer 选出的目标（更真实）

---

### 2. 实现端到端推理脚本

**需要创建**：
- `inference.py` - 完整推理流程

**推理流程**：
```python
# 1. 加载模型
scorer = load_model('checkpoints/scorer/best.pth')
matcher = load_model('checkpoints/matcher/best.pth')
selector = TrajectorySelector()

# 2. 推理
for batch in test_loader:
    # Step 1: 选择目标点
    pred_dis, pred_dac = scorer(vocabulary, batch['bev_feature'])
    selected_goal = vocabulary[pred_dis.argmax(dim=-1)]
    
    # Step 2: 生成多条候选轨迹
    trajectories = matcher.generate_multiple(
        goal=selected_goal,
        scene=batch['bev_feature'],
        num_candidates=10
    )
    
    # Step 3: 选择最优轨迹
    best_traj, scores = selector(
        trajectories=trajectories,
        goal=selected_goal,
        gt_trajectory=batch['trajectory']
    )
    
    # 计算指标
    ade = compute_ade(best_traj, batch['trajectory'])
    fde = compute_fde(best_traj, batch['trajectory'])
```

---

### 3. 完整训练计划

**阶段 1：训练 GoalPointScorer**
```bash
cd /home/zhn/work/text/Thesis_Reading_System/implementations/goalflow
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python train_goal_scorer.py
```
- 训练 100 epochs
- 预期 Top-1 准确率：60-80%
- 预期 Top-5 准确率：90%+

**阶段 2：训练 GoalFlowMatcher**
```bash
python train_flow_matcher.py
```
- 训练 200 epochs
- 使用 gt_goal 或 Scorer 选出的目标
- 验证 ADE/FDE 指标

**阶段 3：端到端测试**
```bash
python inference.py
```
- 完整推理流程
- 生成可视化结果
- 评估最终性能

---

## 📝 重要提醒

### Matplotlib 库依赖问题

**问题**：系统 libstdc++ 版本太旧（3.4.28），matplotlib 需要 3.4.29

**解决方案**：
```bash
# 方法 1：使用便捷脚本
./run_python.sh your_script.py

# 方法 2：手动设置环境变量
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python your_script.py

# 方法 3：永久设置（推荐）
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

详细说明见：`MATPLOTLIB_FIX.md`

---

## 📊 当前项目结构

```
implementations/goalflow/
├── models/
│   ├── goal_point_scorer.py      ✅ 100%
│   ├── goal_flow_matcher.py      ✅ 100%
│   └── trajectory_selector.py    ✅ 100%
├── test/
│   ├── test_goal_flow_matcher.py ✅
│   ├── test_trajectory_selector.py ✅
│   └── README.md
├── data/
│   ├── generate_toy_data.py      ✅ 100%
│   ├── toy_goalflow_dataset.py   ✅ 100%
│   ├── test_dataset_simple.py    ✅ 100%
│   ├── toy_data.npz              ✅ 已生成
│   ├── sample_visualization.png  ✅ 已生成
│   ├── batch_visualization.png   ✅ 已生成
│   └── README.md                 ✅ 100%
├── config/
│   ├── scorer_config.py          ✅ 100%
│   └── matcher_config.py         ✅ 100%
├── checkpoints/
│   ├── scorer/
│   │   └── best.pth              ✅ 测试模型已保存
│   └── matcher/
├── train_goal_scorer.py           ✅ 100%
├── test_train_scorer.py           ✅ 100%
├── train_flow_matcher.py          ⏳ 下一步
├── inference.py                   ⏳ 待创建
├── visualize_results.py           ⏳ 待创建
├── run_python.sh                  ✅ 便捷脚本
├── MATPLOTLIB_FIX.md              ✅ 问题解决文档
├── CODE_REVIEW.md                 ✅ GoalFlowMatcher 代码审查
├── TRAJECTORY_SELECTOR_REPORT.md  ✅ TrajectorySelector 报告
├── NEXT_STEPS.md                  ✅ 详细下一步指南
└── SUMMARY.md                     ✅ 快速总结
```

---

## 🎯 预期成果

### GoalPointScorer 训练后
- Top-1 准确率：60-80%
- Top-5 准确率：90%+
- 能够准确选择最接近 gt_goal 的词汇点

### GoalFlowMatcher 训练后
- ADE < 1.0（平均位移误差）
- FDE < 2.0（最终位移误差）
- 生成的轨迹平滑且符合物理约束

### 端到端系统
- 完整的推理流程
- 多模态轨迹生成
- 最优轨迹选择
- 可视化结果展示
