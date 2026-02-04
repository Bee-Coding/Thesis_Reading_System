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

Phase 2: 数据准备 [进行中]
├─ [•] Toy Dataset 设计
├─ [ ] 轨迹生成器实现
├─ [ ] Dataset 类实现
└─ [ ] DataLoader 测试

Phase 3: 核心算法 [待开始]
├─ [ ] Flow Matching 数学原理学习
├─ [ ] ConditionalFlowMatcher 实现
├─ [ ] OT Flow 定义
├─ [ ] CFM Loss 实现
└─ [ ] ODE Solver 实现

Phase 4: 训练与验证 [待开始]
├─ [ ] 训练脚本实现
├─ [ ] 验证逻辑实现
├─ [ ] 模型保存/加载
└─ [ ] 超参数调优

Phase 5: 可视化与分析 [待开始]
├─ [ ] 轨迹可视化
├─ [ ] 训练曲线绘制
├─ [ ] Flow 演化过程可视化
└─ [ ] 结果分析

Phase 6: 进阶实现 [待开始]
├─ [ ] Transformer 版本速度场网络
├─ [ ] 真实数据集集成
├─ [ ] 条件信息完整实现
└─ [ ] 性能优化
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

---

## 🔄 当前进行中的工作

### 4. Toy Dataset 设计 (进行中)
**时间**: Session 1 (当前)  
**目标文件**: `implementations/flow_matching/data/toy_dataset.py`

**设计目标**:
- 创建简单的 2D 轨迹数据集
- 验证 Flow Matching 算法的正确性
- 不包含复杂的条件信息（先验证核心逻辑）

**待确认的设计问题**:
1. **数据格式**: 
   - 轨迹形状: `(6, 2)` 还是 `(12,)`？
   - 建议: Dataset 返回 `(6, 2)`，训练时 flatten
   
2. **轨迹类型**:
   - 圆形轨迹 (circle)
   - 直线轨迹 (line)
   - S形轨迹 (s_curve)
   - 是否需要其他类型？
   
3. **数据范围**:
   - 坐标范围: [-5, 5]
   - 半径/长度范围: [1, 3] 和 [3, 8]
   
4. **数据集大小**:
   - 训练集: 1000-5000 个样本
   - 验证集: 200-500 个样本

**下一步**:
- 用户确认设计方案
- 实现轨迹生成器
- 实现 ToyTrajectoryDataset 类
- 测试 DataLoader

---

## ⏳ 待完成的工作

### 5. Flow Matching 数学原理学习 (待开始)
**计划**: 在 Toy Dataset 完成后进行

**学习内容**:
1. **Optimal Transport (OT) Flow**
   - 线性插值路径: `x_t = (1-t) * x_0 + t * x_1`
   - 速度场: `v_t = dx_t/dt = x_1 - x_0`
   
2. **Conditional Flow Matching (CFM)**
   - 条件概率路径
   - 边际概率路径
   
3. **CFM Loss**
   - 损失函数: `L = E[ || v_θ(x_t, t, c) - (x_1 - x_0) ||^2 ]`
   - 为什么这个损失函数有效？
   
4. **ODE Solver**
   - Euler 方法
   - Runge-Kutta 方法
   - 如何从噪声生成轨迹

**学习方式**: 费曼学习法讲解 + 苏格拉底式提问

---

### 6. Flow Matching 主模型实现 (待开始)
**目标文件**: `implementations/flow_matching/models/flow_matcher.py`

**需要实现的类/函数**:
- `ConditionalFlowMatcher` 类
- `optimal_transport_flow()` 函数
- `compute_cfm_loss()` 函数
- `ODESolver` 类 (Euler/RK4)

---

### 7. 训练脚本实现 (待开始)
**目标文件**: `implementations/flow_matching/train.py`

**需要实现**:
- 训练循环
- 验证逻辑
- 模型保存/加载
- 日志记录
- 超参数配置

---

### 8. 推理脚本实现 (待开始)
**目标文件**: `implementations/flow_matching/inference.py`

**需要实现**:
- 从噪声生成轨迹
- ODE 求解
- 结果保存

---

### 9. 可视化工具实现 (待开始)
**目标文件**: `implementations/flow_matching/visualize.py`

**需要实现**:
- 轨迹可视化
- 训练曲线绘制
- Flow 演化过程动画
- 对比真实轨迹和生成轨迹

---

### 10. Transformer 版本速度场网络 (待开始)
**目标文件**: `implementations/flow_matching/models/velocity_field_transformer.py`

**设计考虑**:
- 如何将轨迹点序列化
- 注意力机制的设计
- 位置编n
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

---

## 📌 下次会话开始时

### 快速回顾清单
1. 查看本文件，了解当前进度
2. 检查 "当前进行中的工作" 部分
3. 继续未完成的任务

### 当前状态
- **Phase**: Phase 2 - 数据准备
- **当前任务**: Toy Dataset 设计
- **下一步**: 用户确认设计方案后，实现轨迹生成器

### 待用户确认的问题
1. 数据格式: `(6, 2)` 还是 `(12,)`？
2. 轨迹类型: 圆形、直线、S形，是否够用？
3. 数据范围: 坐标 [-5, 5]，半径/长度 [1, 3] 和 [3, 8]
4. 数据集大小: 训练集多少？验证集多少？
5. 文件位置: `implementations/flow_matching/data/toy_dataset.py`？

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

---

**最后更新**: Session 1  
**下次更新**: 完成 Toy Dataset 后
