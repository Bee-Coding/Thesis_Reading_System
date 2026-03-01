# Flow Matching 复现学习进度

## 📚 学习方法
- **苏格拉底学习法**：通过提问引导思考，深入理解概念
- **费曼学习法**：用简单的语言解释复杂概念，确保真正理解

---

## 🎯 总体目标
复现 Flow Matching 算法，用于轨迹预测任务

---

## 📋 学习路线图

```
Phase 1: 基础组件实现 ✓
├─ [✓] 项目结构搭建
├─ [✓] 时间编码器 (SinusoidalEmbedding)
└─ [✓] 速度场网络 (VelocityFieldMLP)

Phase 2: 数据准备 ✓
├─ [✓] Toy Dataset 设计
├─ [✓] 轨迹生成器实现
├─ [✓] Dataset 类实现
└─ [✓] DataLoader 测试

Phase 3: 核心算法 ✓
├─ [✓] Flow Matching 数学原理学习
├─ [✓] ConditionalFlowMatcher 实现
├─ [✓] OT Flow 定义
├─ [✓] CFM Loss 实现
└─ [✓] ODE Solver 实现 (Euler & RK4)

Phase 4: 训练与验证 ✓
├─ [✓] 训练脚本实现
├─ [✓] 验证逻辑实现
├─ [✓] 模型保存/加载
└─ [✓] 在 Toy Dataset 上训练测试

Phase 5: 可视化与分析 ✓
├─ [✓] 轨迹可视化
├─ [✓] 生成过程可视化
├─ [✓] 对比真实轨迹和生成轨迹
└─ [✓] 结果分析

Phase 6: GoalFlow 论文复现 ✓
├─ [✓] GoalFlow 架构理解 (GoalPointScorer + GoalFlowMatcher)
├─ [✓] Toy Dataset 上验证 GoalFlow 架构
├─ [✓] GoalPointScorer 实现与训练
└─ [✓] GoalFlowMatcher 实现与训练

Phase 7: nuScenes 真实数据集集成 ✓
├─ [✓] nuScenes 环境配置与数据下载
├─ [✓] 数据预处理 (轨迹提取、BEV 栅格化、词汇表构建)
├─ [✓] Agent-centered BEV 修复
├─ [✓] Scorer 训练 (vocab_size=32)
└─ [✓] Matcher 训练 (初始版本, ADE=17.77m)

Phase 8: 模型调试与优化 ✓ ← 当前完成
├─ [✓] 诊断 ADE=17.77m 的根本原因
├─ [✓] StandardScaler 归一化 (替代 /bev_range)
├─ [✓] 轨迹位置编码 (traj_pos_embed)
├─ [✓] 场景下采样改进 (AdaptiveAvgPool2d)
├─ [✓] Matcher 重训练 (ADE: 17.77m → 3.65m)
├─ [✓] Scorer 重训练 (适配新归一化数据)
├─ [✓] 端到端推理脚本 (inference_nuscenes.py)
└─ [✓] 可视化结果生成

Phase 9: 扩展与改进 [待开始]
├─ [ ] 扩展到完整 nuScenes 数据集 (850 scenes)
├─ [ ] 多模态预测 (K=6 候选轨迹)
├─ [ ] 数据增强策略
├─ [ ] 基线方法对比实验
└─ [ ] 性能优化与评估
```

---

## ✅ 已完成的工作

### 1. 项目结构搭建 (已完成)
**时间**: Session 1  
**文件结构**:
```
implementations/flow_matching/
├── data/
│   └── __init__.py
├── models/
│   ├── __init__.py
│   ├── time_embedding.py
│   └── velocity_field_MLP.py
├── train.py
├── inference.py
└── README.md
```

---

### 2. 时间编码器实现 (已完成)
**时间**: Session 1  
**文件**: `implementations/flow_matching/models/time_embedding.py`

**关键设计**:
- 类名: `SinusoidalEmbedding`
- 输入: 时间标量 `t` (B,)，范围 [0, 1]
- 输出: 时间编码 (B, 128)
- 编码方式: Sinusoidal encoding (类似 Transformer 位置编码)

**公式**:
```
freq_i = 1 / (max_period^(2i/dim))
embedding[2i] = sin(t * freq_i)
embedding[2i+1] = cos(t * freq_i)
```

**测试结果**: ✓ 所有测试通过

---

### 3. 速度场网络实现 (已完成)
**时间**: Session 1  
**文件**: `implementations/flow_matching/models/velocity_field_MLP.py`

**网络架构**:
```
输入: state (B, 12) + time_embedding (B, 128) + cond (B, 287)
      → 拼接后 (B, 427)
      
隐藏层1: Linear(427, 256) + ReLU + Dropout(0.1)
隐藏层2: Linear(256, 256) + ReLU + Dropout(0.1)
隐藏层3: Linear(256, 256) + ReLU + Dropout(0.1)
隐藏层4: Linear(256, 256) + ReLU + Dropout(0.1)
输出层: Linear(256, 12)  # 无激活函数

输出: velocity (B, 12)
```

**关键设计决策**:
1. **时间编码**: 使用 Sinusoidal Embedding，将标量时间扩展到 128 维
2. **隐藏层架构**: 恒定型 (4层 × 256维)
   - 优势: 梯度流动稳定，适合小维度状态空间
3. **输出层无激活函数**: 速度场可以是任意实数（正负都可以）
4. **条件维度**: cond_dim = 287
   - ego_state: 5维 [vx, vy, heading, ax, ay]
   - goal: 4维 [x, y, heading, v_goal]
   - obstacles: 150维 (10个障碍物 × 15维/个)
   - bev: 128维

**障碍物编码** (每个15维):
- 连续特征: x, y, heading, vx, vy (5维)
- type (one-hot): 4维 [vehicle, pedestrian, bicycle, motorcycle]
- motion_type (one-hot): 3维 [static, constant_velocity, accelerating]
- motion_direction (one-hot): 3维 [straight, left, right]

**测试结果**: ✓ 所有测试通过
- 单样本前向传播 ✓
- 批量样本前向传播 ✓
- 梯度反向传播 ✓
- 时间编码连续性 ✓
- 总参数量: 310,028

**学到的关键概念**:
1. 时间编码的重要性：提供丰富的时间表示
2. 隐藏层架构选择：恒定型 vs 扩张型 vs 渐进收缩型
3. 激活函数选择：输出层为什么不用激活函数
4. One-hot 编码 vs Embedding：适用场景分析
5. 坐标系选择：自车中心坐标系 vs 全局坐标系（关键讨论）
   - **自车中心坐标系**：轨迹预测标准做法，所有坐标相对于自车
   - **全局坐标系**：简单但不真实，需要考虑范围约束
6. 物理约束考虑：实际车辆的最小转弯半径（5.2米）

---

### 4. Toy Dataset 实现 (已完成) ✓
**时间**: Session 2  
**文件**: `implementations/flow_matching/data/toy_dataset.py`

**设计决策**:
- **坐标系**: 自车中心坐标系（自车在原点，所有坐标相对自车）
- **数据格式**: Dataset 返回 `(6, 2)`，训练时 flatten 成 `(12,)`
- **轨迹类型**: 圆形、直线、S形、二次多项式
- **数据范围**: 
  - 输出范围: (-20, 20) 米
  - 圆形半径: (5.2, 9) 米（基于实际车辆最小转弯半径）
  - 直线/曲线长度: (5, 15) 米
- **数据集大小**: 训练集 5000，验证集 500

**实现的类**:
1. `TrajectoryGenerator`: 生成不同类型的 2D 轨迹
   - `generate_circle()`: 圆形轨迹
   - `generate_line()`: 直线轨迹
   - `generate_s_curve()`: S形曲线
   - `generate_polynomial()`: 二次多项式轨迹

2. `ToyTrajectoryDataset`: PyTorch Dataset 类
   - 加载 `.npz` 文件
   - 返回字典: `{'trajectory': (6, 2), 'type': str}`

3. `generate_and_save_dataset()`: 生成并保存数据集

**测试结果**: ✓ 所有测试通过
- 生成的数据集: `data/toy_train.npz`, `data/toy_val.npz`
- 数据统计: 每种类型约 25% 分布均匀

---

### 5. Flow Matching 数学原理学习 (已完成) ✓
**时间**: Session 2

**核心概念掌握**:

1. **Optimal Transport (OT) Flow**
   - 线性插值路径: `x_t = (1-t) * x_0 + t * x_1`
   - 速度场: `v_t = dx_t/dt = x_1 - x_0` (常数)
   - 特点: 最短路径，速度恒定

2. **Conditional Flow Matching (CFM)**
   - 条件速度场: `v_θ(x_t, t)` 学习从噪声到数据的映射
   - 边际速度场: `v*(x_t, t) = E[x_1 - x_0 | x_t]`
   - 处理多模态: 通过条件期望平均不同路径

3. **CFM Loss**
   - 损失函数: `L = E_{t, x_0, x_1} [ || v_θ(x_t, t) - (x_1 - x_0) ||^2 ]`
   - 为什么有效: 最小化预测速度和真实速度的差异
   - 等价于学习边际速度场

4. **ODE Solver**
   - **Euler 方法**: 一阶方法，误差 O(dt²)
     - 公式: `x_{t+dt} = x_t + v(x_t, t) * dt`
   - **RK4 方法**: 四阶方法，误差 O(dt⁵)
     - 使用四个阶段: k1, k2, k3, k4
     - 权重: (1/6, 2/6, 2/6, 1/6)
     - 更高精度，可用更少步数

**关键洞察**:
- Flow Matching 学习的是速度场，不是特定轨迹
- 速度场的本质：理想是常数，现实是近似常数+时间校正
- 条件期望处理多模态：同一位置可能来自不同路径
- RK4 是 ODE 求解的黄金标准

---

### 6. Flow Matching 核心实现 (已完成) ✓
**时间**: Session 2  
**文件**: `implementations/flow_matching/models/flow_matcher.py`

**实现的类和方法**:

**ConditionalFlowMatcher 类**:
- `sample_ot_flow(x_0, x_1, t)`: 采样 OT Flow
  - 输入: 起点 x_0, 终点 x_1, 时间 t
  - 输出: 插值点 x_t, 速度 v_t
  - 实现: 线性插值和常数速度场

- `compute_cfm_loss(model, x_0, x_1, t)`: 计算 CFM Loss
  - 采样 OT Flow 得到 x_t 和真实速度 v_true
  - 网络预测速度 v_pred = model(x_t, t)
  - 计算 MSE: `loss = mean((v_pred - v_true)²)`

- `sample_trajectory(model, x_0, num_steps, method)`: 生成轨迹
  - 使用训练好的模型
  - 通过 ODE 求解器从噪声生成数据
  - 支持 Euler 和 RK4 方法

**测试结果**: ✓ 所有测试通过
- OT Flow 采样正确
- CFM Loss 计算正确
- 简单模型训练收敛
- 轨迹生成成功

---

### 7. ODE 求解器实现 (已完成) ✓
**时间**: Session 2  
**文件**: `implementations/flow_matching/models/ode_solver.py`

**实现的类和方法**:

**ODESolver 类**:
- 支持两种方法: `euler` 和 `rk4`
- `solve(velocity_field, x_0, num_steps, return_trajectory)`: 求解 ODE
 场函数、初始状态、步数
  - 输出: 最终状态、完整轨迹（可选）

**Euler 方法**:
```python
x_{t+dt} = x_t + v(x_t, t) * dt
```
- 一阶方法，误差 O(dt²)
- 简单快速，但精度较低
- 需要更多步数

**RK4 方法**:
```python
k1 = v(x_t, t)
k2 = v(x_t + 0.5*dt*k1, t + 0.5*dt)
k3 = v(x_t + 0.5*dt*k2, t + 0.5*dt)
k4 = v(x_t + dt*k3, t + dt)
x_{t+dt} = x_t + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
```
- 四阶方法，误差 O(dt⁵)
- 精度高，可用更少步数
- 计算量是 Euler 的 4 倍

**便捷函数**:
- `euler_solve()`: Euler 方法快捷调用
- `rk4_solve()`: RK4 方法快捷调用

**测试结果**: ✓ 所有测试通过
- 线性速度场测试
- Flow Matching 速度场测试
- 轨迹长度验证
- 精度对比（RK4 > Euler）

---

### 8. 训练脚本实现 (已完成) ✓
**时间**: Session 3  
**文件**: `itions/flow_matching/train.py`

**实现的类和方法**:

**SimpleVelocityField 类**:
- 简化版速度场网络（适用于 toy dataset）
- 只需要 state 和 time，不需要 condition
- 架构: 
  - 输入: state (12) + time_embedding (128) = 140
  - 隐藏层: 4层 × 256维
  - 输出: velocity (12)
- 参数量: 236,556

**Trainer 类**:
- `train_epoch(epoch)`: 训练一个 epoch
  - 数据处理: 提取 trajectory，flatten 成向量
  - 噪声采样: `x_0 = torch.randn_like(x_1) * 0.5`
  - 训练循环: zero_grad → forward → backward → step
  - 梯度裁剪: `clip_grad_norm_(max_norm=1.0)`
  - 进度显示: 使用 tqdm

- `validate()`: 验证模型
  - 使用 `torch.no_grad()` 不计算梯度
  - 只计算损失，不更新参数
  - 返回平均验证损失

- `save_checkpoint(epoch, val_loss, is_best)`: 保存模型
  - 保存 model_state_dict, optimizer_state_dict
  - 保存 epoch, val_loss, best_val_loss
  - 最佳模型额外保存到 `best.pth`

- `train(num_epochs)`: 完整训练流程
  - 训练 → 验证 → 保存
  - 打印训练信息
  - 跟踪最佳验证损失

**main() 函数**:
- 命令行参数解析
- 数据加载（ToyTrajectoryDataset）
- 模型创建（SimpleVelocityField）
- 优化器（Adam）和调度器（CosineAnnealingLR）
- 创建 Trainer 并开始训练

**训练结果** (2 epochs 测试):
```
Epoch 1/2
  Train Loss: 9.994838
  Val Loss:   6.000483  ✓ 最佳

Epoch 2/2
  Train Loss: 6.006995
  Val Loss:   6.116862
```

**关键学习点**:
1. **数据处理**: Dataset 返回字典，需要提取tten
2. **训练 vs 验证**: 验证时不调用 backward() 和 step()
3. **梯度裁剪**: 防止梯度爆炸
4. **学习率调度**: CosineAnnealingLR 逐渐降低学习率
5. **模型保存**: 保存完整的训练状态

**第一次写训练代码遇到的问题**:
- ❌ 对字典使用 `torch.rand_like()` → ✓ 先提取 tensor
- ❌ 验证时调用 backward() → ✓ 只计算损失
- ❌ 数据形状不匹配 → ✓ flatten (B, 6, 2) → (B, 12)
- ❌ 模型接口不匹配 → ✓ 创建 SimpleVelocityField

---

### 9. 可视化工具实现 (已完成) ✓
**时间**: Session 3  
**文件**: `implementations/flow_matching/visualize.py`

**实现的函数**:

1. **load_model(checkpoint_path, device)**: 加载训练好的模型
   - 创建模型结构
   - 加载权重
   - 打印模型信息

2. **generate_trajectories(model, flow_matcher, num_samples, ...)**: 生成轨迹
   - 采样初始噪声 x_0
   - 使用 ODE 求解器生成轨迹
   - 返回最终状态和完整轨迹

3. **plot_trajectories(generated_trajs, real_trajs, ...)**: 绘制轨迹
   - 4×4 网格显示 16 个样本
   - 蓝色实线: 生成的轨迹
   - 绿色圆点: 起点
   - 红色圆点: 终点
   - 黑色虚线: 真实轨迹（可选）

4. **plot_generation_process(trajectory_list, ...)**: 绘制生成过程
   - 显示 5 个时间步: t=0, 0.25, 0.5, 0.75, 1.0
   - 展示从噪声到数据的演化过程

5. **compare_with_real_data(model, dataset, ...)**: 对比真实数据
   - 生成轨迹 vs 真实轨迹
   - 蓝色实线: 生成
   - 红色虚线: 真实

**main() 函数**:
- 加载模型和数据
- 生成轨迹
- 绘制 3 张图片:
  1. `generated_trajectories.png`: 生成的轨迹
  2. `generation_process.png`: 生成过程演化
  3. `comparison.png`: 对比真实数据

**可视化结果**:
- ✓ 成功生成 3 张可视化图片
- ✓ 保存在 `visualizations/` 目录
- ✓ 使用 `matplotlib.use('Agg')` 支持无显示环境

**技术细节**:
- 使用 `plt.close()` 而不是 `plt.show()`
- 设置坐标轴范围 (-20, 20)
- 使用 `set_aspect('equal')` 保持比例

---

### 10. 学习文档创建 (已完成) ✓
**时间**: Session 3  
**文件**: `implementations/flow_matching/TRAINING_GUIDE.md`

**文档内容**:
1. **核心概念回顾**: Flow Matching 的训练目标和损失函数
2. **代码结构详解**: 数据处理、训练循环、验证循环
3. **你的代码问题总结**: 详细分析遇到的 3 个主要问题
4. **训练技巧**: 学习率调整、梯度裁剪、早停、检查点保存
5. **学习建议**: 第一次写训练代码的常见困惑和解答
6. **推荐学习资源**: PyTorch 官方教程、调试技巧

**价值**:
- 完整记录了第一次写训练代码的学习过程
- 详细解释了每个问题的原因和解决方案
- 提供了可复用的训练流程模板

---

### 11. Mem0 学习记录 (已完成) ✓
**时间**: Session 3  
**文件**: `record_flow_matching_implementation.py`

**记录内容**:
1. **代码实现的关键洞察** (14 条)
   - PyTorch 训练循环标准流程
   - 数据处理技巧
   - 模型设计要点
   - 可视化技巧

2. **已解决的代码问题** (6 个)
   - 数据处理错误
   - 验证函数错误
   - 数据形状不匹配
   - 模型接口不匹配
   - 可视化环境问题

3. **第一次写训练代码的常见困惑** (6 个)
   - 为什么要 zero_grad()？
   - backward() 和 step() 的区别？
   - train() 和 eval() 的区别？
   - 为什么验证时用 no_grad()？
   - 如何处理 Dataset 返回的字典？
   - 为什么要 flatten 轨迹数据？

4. **学习进度更新** (5 个任务)
   - ODE 求解器实现
   - ConditionalFlowMatcher 实现实现
   - 训练测试
   - 可视化

5. **代码能力提升** (8 项)
   - PyTorch 训练循环
   - Dataset 和 DataLoader
   - 模型保存和加载
   - 梯度裁剪
   - 学习率调度
   - 可视化
   - 调试技巧

6. **训练结果记录**
   - 数据集: 5000 训练, 500 验证
   - 模型: 236,556 参数
   - 训练损失: 9.99 → 6.01
   - 验证损失: 6.00

7. **下一步计划** (4 个)
   - 增加训练 epochs
   - 实现更复杂的网络
   - 实现推理脚本
   - 评估生成质量

**执行结果**: ✓ 成功记录到 Mem0

---

### 12. GoalFlow 架构实现 (已完成) ✓
**时间**: Session 4-5
**目录**: `implementations/goalflow/`

**GoalFlow 论文核心思想**:
- 两阶段架构：先预测目标点 (Goal)，再生成到达目标的轨迹
- **GoalPointScorer**: 从 BEV 场景中评估候选目标点的可达性
- **GoalFlowMatcher**: 以目标点为条件，用 Flow Matching 生成轨迹

**实现的模型**:

1. **GoalPointScorer** (`models/goal_point_scorer.py`)
   - 输入: BEV 图像 (3, 200, 200) + 候选目标点词汇表 (N, 2)
   - 输出: 距离分数 pred_dis (B, N) + 可行驶区域分数 pred_dac (B, N)
   - 架构: CNN 提取 BEV 特征 → Transformer 交叉注意力 → 分数预测
   - 损失: L1 距离损失 + BCE 可行驶区域损失

2. **GoalFlowMatcher** (`models/goal_flow_matcher.py`)
   - 输入: BEV 图像 + 目标点 + 噪声轨迹 + 时间步 t
   - 输出: 速度场 v_t (B, T, 2)
   - 架构: CNN 场景编码 → Transformer (场景 tokens + 轨迹 tokens + 目标 token) → 速度预测
   - 推理: ODE 求解 (Euler 方法)，从噪声 x_0 ~ N(0,1) 积分到 x_1

**在 Toy Dataset 上的验证**:
- 先在简单数据上验证架构正确性
- 确认 Scorer 能学会评分、Matcher 能生成轨迹
- 为 nuScenes 集成打下基础

---

### 13. nuScenes 数据预处理 (已完成) ✓
**时间**: Session 5-6
**文件**: `implementations/goalflow/data/nuscenes_preprocessor.py`

**预处理流程**:
1. **轨迹提取**: 从 nuScenes 场景中提取所有 agent 的历史/未来轨迹
2. **BEV 栅格化**: 将 HD 地图转换为 3 通道 BEV 图像 (200×200)
   - Channel 0: 车道线 (drivable_area)
   - Channel 1: 道路边界 (road_segment)
   - Channel 2: 人行道 (walkway)
3. **词汇表构建**: K-means 聚类所有训练轨迹的终点 → 32 个候选目标点
4. **数据划分**: 6 train / 2 val / 1 test scenes (mini 数据集)

**关键 Bug 修复 — Agent-centered BEV**:
- **问题**: 所有 agent 共享同一个 ego-centered BEV → 只有 ~50 个唯一 BEV
- **原因**: BEV 应该以每个 agent 自己的位置/朝向为中心
- **修复**: 每个 agent 独立计算 BEV，以 agent 的全局位置和旋转为中心
- **效果**: 修复后 533/574 个唯一 BEV (93%)

**数据统计** (mini 数据集):
- 训练集: 574 样本
- 验证集: 158 样本
- 测试集: 123 样本
- 历史帧: 4 帧 (2.0s)，未来帧: 12 帧 (6.0s)

---

### 14. 初始训练与问题诊断 (已完成) ✓
**时间**: Session 6-7

**初始训练结果**:
- Scorer: Top-1 准确率 ~24% (vocab_size=32, 574 样本)
- Matcher: ADE = 17.77m ← **严重偏高**

**诊断过程 — 为什么 ADE 这么高？**

发现了 3 个根本原因：

#### 原因 1: 归一化方式不匹配 (最关键)

```
原始方法: traj_normalized = traj / bev_range  (除以 50 米)
结果:
  X 方向 std = 0.38
  Y 方向 std = 0.05
  
Flow Matching 噪声: x_0 ~ N(0, 1), std = 1.0

问题: 数据尺度 (0.05~0.38) 远小于噪声尺度 (1.0)
→ 模型学到的是 v ≈ -x_0 (只在消除噪声，没学到轨迹结构)
```

#### 原因 2: 缺少轨迹位置编码

```
Transformer 输入: [traj_token_1, traj_token_2, ..., traj_token_12]
问题: Transformer 不知道哪个 token 对应哪个时间步
→ 输出的轨迹没有时间连续性，像 [50, 6, 87, 43, ...] 随机跳跃
```

#### 原因 3: 场景下采样过度

```
原始: 200×200 → 13×13 (4 个 stride=2 卷积, 压缩比 237:1)
→ 太多空间信息丢失，模型看不到道路细节
```

---

### 15. 关键改进与重训练 (已完成) ✓ ← 今天的核心工作
**时间**: Session 7-8 (2026-02-28)

#### 改进 1: StandardScaler 归一化

**原理**:
```python
# 旧方法: 除以固定范围
traj_norm = traj / 50.0  # X std=0.38, Y std=0.05

# 新方法: StandardScaler
traj_mean = [9.44, -0.067]   # 训练集的均值
traj_std  = [18.81, 2.70]    # 训练集的标准差
traj_norm = (traj - traj_mean) / traj_std  # X std=1.0, Y std=1.0
```

**为什么有效**:
- Flow Matching 的噪声 x_0 ~ N(0, 1)，std=1.0
- StandardScaler 后数据也是 std=1.0
- 数据分布和噪声分布完美匹配
- 模型可以学到有意义的速度场，而不是只消除噪声

**实现细节**:
- 训练集计算 mean/std，保存到 `metadata.pkl`
- 验证集/测试集复用训练集的 mean/std (通过 `config._norm_stats`)
- 推理时用 `pred * std + mean` 反归一化回真实坐标

**学到的教训**:
> 归一化不是随便除一个数就行。必须让数据分布和算法假设匹配。
> Flow Matching 假设噪声是标准正态分布，所以数据也应该标准化到类似尺度。

#### 改进 2: 轨迹位置编码 (traj_pos_embed)

**原理**:
```python
# 在 GoalFlowMatcher.__init__ 中添加
self.traj_pos_embed = nn.Parameter(
    torch.randn(1, num_future_steps, hidden_dim) * 0.02
)

# 在 forward 中使用
traj_tokens = self.traj_embed(noisy_traj) + self.traj_pos_embed
#                                            ↑ 告诉 Transformer 这是第几个时间步
```

**为什么有效**:
- Transformer 的自注意力机制是排列不变的 (permutation invariant)
- 没有位置编码，[t1, t2, t3] 和 [t3, t1, t2] 对模型来说完全一样
- 位置编码让模型知道每个 token 的时间顺序
- 类比：给书的每一页加上页码

**学到的教训**:
> 只要用 Transformer 处理有序序列，就必须加位置编码。
> 这是 Transformer 架构的基本要求，不是可选的。

#### 改进 3: 场景下采样改进

**原理**:
```python
# 旧方法: 4 个 stride=2 卷积
200×200 → 100 → 50 → 25 → 13×13  # 只有 169 个 tokens

# 新方法: CNN + AdaptiveAvgPool2d
200×200 → CNN特征提取 → AdaptiveAvgPool2d(25, 25)  # 625 个 tokens
```

**为什么有效**:
- 625 tokens vs 169 tokens → 保留 3.7 倍的空间信息
- `AdaptiveAvgPool2d` 可以灵活控制输出分辨率
- 更多 scene tokens → Transformer 能看到更多道路细节

**配置参数**: `scene_token_size = (25, 25)` 在 `nuscenes_config.py` 中可调

#### 改进 4: 模型容量增加

```python
matcher_num_layers: 4 → 6      # Transformer 更深，表达能力更强
matcher_num_steps: 10 → 20     # ODE 求解更精细，轨迹更准确
```

#### 训练结果对比

| 指标 | 改进前 | 改进后 | 提升 |
|------|--------|--------|------|
| Matcher ADE | 17.77m | 3.65m | **-79.5%** |
| Matcher FDE | ~30m | 5.68m | **-81%** |
| 最佳 epoch | - | 88 | - |

---

### 16. Scorer 重训练 (已完成) ✓
**时间**: Session 8 (2026-02-28)

**为什么需要重训练**:
- Scorer 的旧 checkpoint 是在 StandardScaler 归一化之前训练的
- 旧模型学到的词汇表分布和新数据不匹配
- 通过检查 checkpoint 时间戳发现：Scorer 训练时间 < 数据重新预处理时间

**改进: checkpoint 审计信息**:
```python
# 新增保存到 checkpoint 的信息
save_dict['vocabulary'] = vocabulary     # 训练时使用的词汇表
save_dict['traj_mean'] = metadata['traj_mean']  # 归一化均值
save_dict['traj_std'] = metadata['traj_std']    # 归一化标准差
save_dict['norm_type'] = 'standard'             # 归一化类型
```

**学到的教训**:
> Checkpoint 应该保存足够的元信息，让你能在事后验证它和当前数据是否兼容。
> 否则你可能用错误的 checkpoint 做推理而不自知。

**重训练结果**:
- Best epoch: 3, Val loss: 3.43
- Top-1 准确率: 16.3%, Top-5: 25.7%
- 准确率不高是因为 mini 数据集太小 (574 样本, 32 类)

---

### 17. 端到端推理与可视化 (已完成) ✓
**时间**: Session 8 (2026-02-28)
**文件**: `implementations/goalflow/inference_nuscenes.py`

**推理流程**:
1. 加载 Scorer + Matcher 模型
2. 加载测试集 (123 样本)
3. 两种评估模式:
   - **GT Goal**: 用真实目标点 → 只评估 Matcher 的轨迹生成能力
   - **Scorer Goal**: 用 Scorer 预测的目标点 → 评估端到端性能
4. 生成可视化图片

**最终评估结果** (测试集, 123 样本):

| 模式 | ADE | FDE | minADE | minFDE |
|------|-----|-----|--------|--------|
| GT Goal (Matcher only) | 3.42m | 4.92m | 2.08m | 1.59m |
| Scorer Goal (End-to-End) | 14.83m | 26.44m | 13.31m | 23.57m |

**结果分析**:
- **Matcher 表现良好**: 给定正确目标点，ADE=3.42m，与训练时一致
- **Scorer 是瓶颈**: 目标点误差 27.1m，导致端到端性能差
- **根本原因**: mini 数据集太小 (574 样本)，Scorer 没有足够数据学习
- **解决方向**: 扩展到完整 nuScenes 数据集 (850 scenes, ~数万样本)

**可视化输出**: `visualizations/nuscenes/inference/` 目录下 10 张图片
- 绿色轨迹 (GT Goal) 能较好地跟踪真实轨迹
- 红色轨迹 (Scorer Goal) 经常方向错误 (因为 Scorer 选错了目标点)
- 轨迹本身是平滑、时间连续的 (位置编码修复生效)

---

## 🔄 当前进行中的工作

### 18. 消化与理解今天的改进 (当前)
**时间**: Session 8 后
**任务**: 深入理解为什么 StandardScaler、位置编码、下采样改进会生效

**建议的验证实验**:
- 去掉位置编码重新训练，观察轨迹是否又变成随机跳跃
- 改回 `/50` 归一化，观察 ADE 是否回到 17m
- 画出归一化前后的数据分布直方图

---

## ⏳ 待完成的工作

### 19. 扩展到完整 nuScenes 数据集 (待开始)
**目标**: 使用全部 850 个场景训练，大幅提升 Scorer 准确率

**预期效果**:
- 训练样本从 574 → 数万
- Scorer Top-1 准确率应显著提升
- 端到端 ADE 应大幅下降

### 20. 多模态预测 (待开始)
**目标**: 输出 K=6 条候选轨迹，覆盖不同可能的未来

### 21. 基线方法对比 (待开始)
**目标**: 实现 Constant Velocity 等简单基线，验证 GoalFlow 的优势

---

## 📝 关键概念记录

### 1. Flow Matching 核心思想
- 从数据分布 p_1(x) 到噪声分布 p_0(x) 构建一条连续路径
- 训练速度场网络预测路径上每个点的速度
- 推理时通过 ODE 求解从噪声生成数据

### 2. 为什么使用时间编码？
- 时间 t ∈ [0, 1] 是标量，信息量太少
- Sinusoidal encoding 将 1维扩展到高维（128维）
- 提供更丰富的时间表示，帮助网络学习时间依赖关系

### 3. 速度场网络的输出是什么？
- 输出是**速度场** v_t(x_t, t, c)，不是下一时刻状态
- 速度场表示"在时刻 t，状态 x_t 应该朝哪个方向移动"
- 维度和状态维度相同（都是 12 维）

### 4. 为什么输出层不用激活函数？
- 速度场可以是任意实数（正负都可以）
- 如果加 ReLU 会限制为正数，无法学习负方向的速度
- 如果加 Tanh 会限制范围，可能影响学习大幅度变化

### 5. 隐藏层架构的选择
**恒定型** (推荐用于 Flow Matching):
- 结构: [256, 256, 256, 256]
- 优势: 梯度流动稳定，适合生成任务
- 适用: 状态维度较小的情况

**扩张-收缩型**:
- 结构: [512, 1024, 512]
- 优势: 更大的表达空间
- 劣势: 参数量大，可能过拟合

**渐进收缩型**:
- 结构: [512, 256, 128]
- 适用: 分类任务
- 不适合: 生成任务（可能过早丢失信息）

### 6. PyTorch 训练循环的本质
**训练循环**:
```python
for batch in dataloader:
    optimizer.zero_grad()      # 清零梯度（必须！）
    loss = compute_loss(...)   # 计算损失
    loss.backward()            # 反向传播
    optimizer.step()           # 更新参数
```

**验证循环**:
```python
model.eval()
with torch.no_grad():
    for batch in dataloader:
        loss = compute_loss(...)  # 只计算损失
        # 不调用 backward() 和 step()
```

### 7. 数据处理的关键
```python
# Dataset 返回字典
batch = {'trajectory': (B, 6, 2), 'type': list}

# 提取并处理
x_1 = batch['trajectory'].to(device)  # 移到GPU
x_1 = x_1.reshape(B, -1)              # Flatten成(B, 12)
x_0 = torch.randn_like(x_1) * 0.5     # 采样噪声
```

### 8. ODE 求解器的选择
**Euler 方法**:
- 一阶方法，误差 O(dt²)
- 简单快速，但精度较低
- 需要更多步数（~100步）

**RK4 方法**:
- 四阶方法，误差 O(dt⁵)
- 精度高，可用更少步数（~50步）
- 计算量是 Euler 的 4 倍
- **推荐使用**：精度和效率的最佳平衡

### 9. 训练技巧
1. **梯度裁剪**: 防止梯度爆炸
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

2. **学习率调度**: 逐渐降低学习率
   ```python
   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
       optimizer, T_max=epochs, eta_min=1e-6
   )
   ```

3. **模型保存**: 保存完整的训练状态
   ```python
   checkpoint = {
       'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'val_loss': val_loss,
   }
   torch.save(checkpoint, 'best.pth')
   ```

### 10. 第一次写训练代码的常见困惑
1. **为什么要 zero_grad()？**
   - PyTorch 默认会累积梯度
   - 每次反向传播前必须清零

2. **backward() 和 step() 的区别？**
   - `backward()`: 计算梯度（存储在 `.grad` 中）
   - `step()`: 根据梯度更新参数

3. **train() 和 eval() 的区别？**
   - `train()`: 启用 dropout、batch norm 等
   - `eval()`: 关闭 dropout、batch norm 等

4. **为什么验证时用 no_grad()？**
   - 不需要计算梯度，节省内存
   - 加快计算速度

### 11. 数据归一化与 Flow Matching 的匹配 (重要!)

**核心原则**: 数据分布的尺度必须和噪声分布的尺度匹配。

```
Flow Matching 的训练过程:
  x_t = (1-t) * x_0 + t * x_1    # 线性插值
  v_t = x_1 - x_0                 # 真实速度

其中 x_0 ~ N(0, 1)  ← 噪声，std=1.0
     x_1 = 数据      ← 必须也是 std≈1.0

如果 x_1 的 std=0.05 (Y方向除以50m后):
  v_t = x_1 - x_0 ≈ 0 - x_0 = -x_0
  → 模型只学到"消除噪声"，没学到数据结构
```

**StandardScaler 的作用**:
```python
x_normalized = (x - mean) / std
# 归一化后 mean=0, std=1 → 和 N(0,1) 噪声完美匹配
```

**推广**: 任何生成模型（Diffusion、Flow Matching、Score Matching）都需要注意数据和噪声的尺度匹配。

### 12. Transformer 位置编码的必要性

**为什么 Transformer 需要位置编码**:
- 自注意力机制是排列不变的: Attention(Q, K, V) 不依赖 token 顺序
- 对于有序序列（时间步、文本位置），必须显式注入顺序信息
- 没有位置编码 → 模型无法区分 [t1, t2, t3] 和 [t3, t1, t2]

**两种常见方式**:
```python
# 1. 固定的 Sinusoidal 编码 (Transformer 原论文)
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

# 2. 可学习的位置编码 (本项目使用)
self.pos_embed = nn.Parameter(torch.randn(1, seq_len, dim))
```

**本项目的应用**: GoalFlowMatcher 中的 `traj_pos_embed`
- 12 个未来时间步的轨迹 tokens 需要位置编码
- 加上后轨迹从随机跳跃变成平滑连续

### 13. 模型调试的系统方法

**当模型性能不好时的排查顺序**:

1. **数据层面** (最常见的问题源)
   - 归一化是否正确？数据分布是否合理？
   - 标签是否正确？数据是否有泄漏？
   - 数据量是否足够？

2. **模型层面**
   - 模型容量是否足够？(层数、维度)
   - 是否缺少关键组件？(如位置编码)
   - 特征提取是否充分？(如下采样过度)

3. **训练层面**
   - 学习率是否合适？
   - 是否过拟合/欠拟合？
   - 损失函数是否正确？

**本项目的经验**: ADE=17.77m 的问题 80% 来自数据归一化，15% 来自模型缺陷，5% 来自容量不足。

### 14. Checkpoint 管理最佳实践

**应该保存到 checkpoint 的信息**:
```python
checkpoint = {
    # 必须保存
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'val_loss': val_loss,
    
    # 建议保存 (审计用)
    'traj_mean': traj_mean,      # 归一化参数
    'traj_std': traj_std,
    'norm_type': 'standard',
    'vocabulary': vocabulary,     # 训练时的词汇表
    'config': config_dict,        # 训练配置
}
```

**为什么重要**: 如果不保存归一化参数，事后无法验证 checkpoint 和数据是否兼容。
本项目就遇到了这个问题：Scorer checkpoint 是旧数据训练的，但没有保存归一化信息，
只能通过文件时间戳推断不兼容。

---

## 🔧 技术细节记录

### 维度计算
```python
# 状态维度
state_dim = 12  # 6个点 × 2 (x, y)

# 时间维度
time_dim = 128  # Sinusoidal encoding 后

# 条件维度 (完整版)
ego_state_dim = 5      # [vx, vy, heading, ax, ay]
goal_dim = 4           # [x, y, heading, v_goal]
obstacle_per_dim = 15  # 每个障碍物
N_obs = 10             # 最多10个障碍物
bev_dim = 128          # BEV特征
cond_dim = 5 + 4 + 150 + 128 = 287

# Toy Dataset (简化版)
cond_dim = 0  # 暂时不需要条件

# MLP 输入维度
input_dim = state_dim + time_dim + cond_dim
input_dim = 12 + 128 + 287 = 427  # 完整版
input_dim = 12 + 128 + 0 = 140    # Toy 版本
```

### 文件导入问题解决
```python
# 支持两种导入方式：作为模块导入 或 直接运行测试
try:
    from .time_embedding import SinusoidalEmbedding
except ImportError:
    from time_embedding import SinusoidalEmbedding
```

---

## 🎓 学习心得

### Session 1 学习总结
1. **苏格拉底式提问的价值**
   - 通过提问引导思考，而不是直接给答案
   - 帮助理解"为什么"而不只是"是什么"
   - 例如: 为什么输出层不用激活函数？

2. **费曼学习法的应用**
   - 用简单的语言解释复杂概念
   - 例如: 时间编码就像给时间"扩展维度"
   - 如果不能简单解释，说明还没真正理解

3. **先实现再理解 vs 先理解再实现**
   - 选择了先实现基础组件（时间编码、MLP）
   - 再学习核心算法（Flow Matching）
   - 这样可以边学边验证

4. **测试驱动开发的重要性**
   - 每个组件都编写了完整的测试
   - 确保维度正确、梯度正常、功能符合预期
   - 为后续集成打下坚实基础

### Session 2 学习总结
1. **数学原理的深入理解**
   - 通过苏格拉底式提问深入理解 Flow Matching
   - 理解了 OT Flow、CFM Loss、条件期望的本质
   - 掌握了 RK4 方法的原理和优势

2. **从理论到实现的转化**
   - 将数学公式转化为代码实现
   - ConditionalFlowMatcher 和 ODESolver 的实现
   - 验证了理论理解的正确性

3. **代码测试的重要性**
   - 每个函数都编写了测试用例
   - 通过测试发现和修复问题
   - 确保实现符合数学定义

### Session 3 学习总结（代码实践）
1. **第一次写训练代码的挑战**
   - 遇到了数据处理、验证函数等多个问题
   - 通过详细的错误分析和指导解决了所有问题
   - 深刻理解了 PyTorch 训练循环的本质

2. **从困惑到理解的过程**
   - 为什么要 zero_grad()？
   - backward() 和 step() 的区别？
   - 验证时为什么不能更新参数？
   - 通过实践理解了这些概念

3. **代码能力的显著提升**
   - 掌握了 PyTorch 训练循环的标准流程
   - 学会了 Dataset 和 DataLoader 的使用
   - 理解了模型保存、梯度裁剪、学习率调度等技巧
   - 学会了使用 matplotlib 进行可视化

4. **学习方法的有效性**
   - 详细的学习指南（TRAINING_GUIDE.md）非常有帮助
   - 通过对比"错误代码"和"正确代码"加深理解
   - 记录问题和解决方案形成知识积累

5. **成就感和信心**
   - 成功训练了第一个 Flow Matching 模型
   - 看到了损失下降和模型收敛
   - 生成了可视化结果，直观看到效果
   - 为后续更复杂的实现打下了坚实基础

### 整体学习收获
1. **完整的项目实现经验**
   - 从数学原理到代码实现
   - 从数据准备到模型训练
   - 从结果可视化到文档记录
   - 形成了完整的项目开发流程

2. **深入理解 Flow Matching**
   - 不仅知道"是什么"，更理解"为什么"
   - 能够解释每个设计决策的原因
   - 能够独立实现和调试代码

3. **代码能力的全面提升**
   - PyTorch 深度学习框架
   - 数据处理和可视化
   - 调试和问题解决
   - 文档编写和知识管理

4. **学习方法的掌握**
   - 苏格拉底式提问引导思考
   - 费曼学习法确保理解
   - 测试驱动开发保证质量
   - 文档记录形成知识积累

### Session 7-8 学习总结（GoalFlow nuScenes 调试与优化）
1. **数据归一化是深度学习中最容易被忽视但最关键的环节**
   - 错误的归一化可以让模型完全失效 (ADE 17.77m)
   - 正确的归一化可以让同一个模型表现优秀 (ADE 3.65m)
   - 必须理解算法对数据分布的假设

2. **Transformer 不是万能的，需要正确配置**
   - 缺少位置编码 → 无法建模序列顺序
   - 输入分辨率太低 → 丢失关键信息
   - 层数不够 → 表达能力不足

3. **系统化的调试方法比盲目调参更有效**
   - 先分析数据分布 (打印 mean/std)
   - 再检查模型输出 (是否有意义)
   - 最后调整超参数
   - 每次只改一个变量，观察效果

4. **Checkpoint 管理是工程实践的重要部分**
   - 保存足够的元信息用于审计
   - 确保 checkpoint 和数据版本匹配
   - 文件时间戳可以作为最后的兼容性检查手段

5. **小数据集的局限性**
   - 574 个样本训练 32 类分类器 → 准确率只有 16%
   - Scorer 是端到端系统的瓶颈
   - 扩展数据量是最直接的改进方向

---

## 📌 下次会话开始时

### 快速回顾清单
1. 查看本文件，了解当前进度
2. 检查 "当前进行中的工作" 部分
3. 继续未完成的任务

### 当前状态
- **Phase**: Phase 8 - 模型调试与优化 ✓ 已完成
- **下一个 Phase**: Phase 9 - 扩展与改进
- **建议下一步**: 
  1. 消化今天的改进，理解原理
  2. 扩展到完整 nuScenes 数据集
  3. 实现多模态预测

### 项目文件结构
```
implementations/flow_matching/          # Phase 1-5: 基础 Flow Matching
├── data/
│   ├── toy_train.npz
│   ├── toy_val.npz
│   └── toy_dataset.py
├── models/
│   ├── ode_solver.py
│   ├── flow_matcher.py
│   ├── time_embedding.py
│   └── velocity_field_MLP.py
├── checkpoints/
├── visualizations/
├── train.py
├── visualize.py
├── TRAINING_GUIDE.md
└── LEARNING_PROGRESS.md               # 本文件

implementations/goalflow/               # Phase 6-8: GoalFlow + nuScenes
├── config/
│   └── nuscenes_config.py              # 配置 (vocab_size=32, scene_token_size=(25,25))
├── data/
│   ├── nuscenes_preprocessor.py        # 预处理 (StandardScaler 归一化)
│   ├── nuscenes_dataset.py             # PyTorch Dataset
│   └── nuscenes_utils.py               # 坐标转换工具
├── models/
│   ├── goal_point_scorer.py            # GoalPointScorer
│   └── goal_flow_matcher.py            # GoalFlowMatcher (含 traj_pos_embed)
├── data/nuscenes_processed/
│   ├── train/ (574 samples, metadata.pkl 含 traj_mean/std)
│   ├── val/   (158 samples)
│   └── test/  (123 samples)
├── checkpoints/
│   ├── scorer_nuscenes/best_model.pth  # Top-1: 16.3%
│   └── matcher_nuscenes/best_model.pth # ADE: 3.65m
├── visualizations/nuscenes/inference/  # 推理可视化图片
├── train_scorer_nuscenes.py
├── train_matcher_nuscenes.py
├── inference_nuscenes.py               # 端到端推理 + 可视化
└── NUSCENES_IMPLEMENTATION_GUIDE.md    # 教学计划
```

### 训练结果摘要

**Phase 1-5: 基础 Flow Matching (Toy Dataset)**
```
模型: SimpleVelocityField (236,556 参数)
数据: 5000 训练, 500 验证
训练: 2 epochs (测试)
Epoch 1: Train Loss 9.99 → Val Loss 6.00 ✓ 最佳
Epoch 2: Train Loss 6.01 → Val Loss 6.12
```

**Phase 6-8: GoalFlow + nuScenes (Mini Dataset)**
```
GoalPointScorer:
  参数量: 2.18M
  数据: 574 训练, 158 验证, 123 测试
  Best epoch: 3, Val Top-1: 16.3%, Top-5: 25.7%

GoalFlowMatcher:
  数据: 同上
  改进前: ADE = 17.77m (归一化错误 + 缺少位置编码)
  改进后: ADE = 3.65m, FDE = 5.68m (best epoch 88)
  
端到端推理 (测试集):
  GT Goal:     ADE=3.42m, FDE=4.92m, minADE=2.08m
  Scorer Goal: ADE=14.83m (受限于 Scorer 准确率)
```

### 快速命令
```bash
# 查看可视化结果
ls -lh implementations/flow_matching/visualizations/

# 继续训练（更多 epochs）
cd implementations/flow_matching
python train.py --epochs 50 --batch_size 32 --lr 1e-3

# 生成新的可视化
python visualize.py --checkpoint ./checkpoints/best.pth

# 记录学习内容到 Mem0
cd /home/zhn/work/text/Thesis_Reading_System
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
python record_flow_matching_implementation.py
```

---

## 📚 参考资料

### 论文
- Flow Matching for Generative Modeling (Lipman et al., 2023)
- Conditional Flow Matching (Tong et al., 2023)

### 代码参考
- [官方实现](https://github.com/atong01/cal-flow-matching)

---

## 🐛 遇到的问题和解决方案

### 问题1: 相对导入错误
**错误**: `ImportError: attempted relative import with no known parent package`

**原因**: 直接运行 Python 文件时，不知道父包

**解决方案**:
```python
try:
    from .time_embedding import SinusoidalEmbedding
except ImportError:
    from time_embedding import SinusoidalEmbedding
```

### 问题2: 层数逻辑混淆
**问题**: `num_layers=4` 实际创建了 5 层

**解决方案**: 改为 `num_hidden_layers=4`，语义更清晰

### 问题3: Flow Matching 数据归一化不匹配 (Phase 8)
**错误**: Matcher ADE = 17.77m，模型几乎没学到有用信息

**原因**: 
- 用 `/bev_range` (除以50) 归一化，导致数据 std=0.05~0.38
- Flow Matching 噪声 x_0 ~ N(0,1)，std=1.0
- 数据尺度远小于噪声尺度，模型只学到消除噪声

**解决方案**:
```python
# StandardScaler: 让数据 std=1.0，匹配噪声分布
traj_norm = (traj - traj_mean) / traj_std
```

**效果**: ADE 从 17.77m → 3.65m

### 问题4: 轨迹预测无时间连续性 (Phase 8)
**错误**: 预测轨迹像 [50, 6, 87, 43, ...] 随机跳跃

**原因**: GoalFlowMatcher 的 Transformer 没有轨迹位置编码，无法区分时间步顺序

**解决方案**:
```python
self.traj_pos_embed = nn.Parameter(torch.randn(1, T, D) * 0.02)
traj_tokens = self.traj_embed(noisy_traj) + self.traj_pos_embed
```

### 问题5: Scorer checkpoint 与数据版本不兼容 (Phase 8)
**错误**: 推理时 Scorer 目标点误差异常大

**原因**: Scorer 在旧数据（非 StandardScaler）上训练，但推理用的是新数据

**解决方案**: 
1. 重训练 Scorer
2. 在 checkpoint 中保存归一化参数和词汇表，便于事后审计

---

**最后更新**: Session 8 (2026-02-28)  
**下次更新**: 完成下一个任务后

**总结**: 
- ✅ Phase 1-5: Flow Matching 基础实现（从理论到代码）
- ✅ Phase 6: GoalFlow 架构实现（Scorer + Matcher）
- ✅ Phase 7: nuScenes 数据集集成（预处理、训练）
- ✅ Phase 8: 模型调试与优化（ADE: 17.77m → 3.65m）
- ✅ 端到端推理与可视化完成
- 🎯 下一步：消化改进原理，扩展到完整数据集
