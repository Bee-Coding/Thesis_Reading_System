# Goal Flow Matching 复现与 AVP 应用路线图

## 🎯 总体目标

1. 深入理解 Flow Matching 算法原理
2. 复现 Goal Flow Matching 论文
3. 应用到 AVP 记忆泊车场景
4. 使用实际数据优化模型

---

## 📅 阶段 1：理论学习与分析（1-2周）

### ✅ 已完成
- [x] Flow Matching 基础实现（toy dataset）
- [x] 条件速度场网络（VelocityFieldMLP）
- [x] 训练和可视化流程

### 🔄 进行中

#### 任务 1.1：深入理解 Flow Matching 数学原理
**时间**：2-3天

**学习内容**：
1. **Optimal Transport 基础**
   - Monge 问题和 Kantorovich 问题
   - Wasserstein 距离
   - OT Flow 的几何意义

2. **Flow Matching 推导**
   - 从 ODE 到 Flow Matching
   - CFM Loss 的推导
   - 为什么 OT Flow 是最优的

3. **与其他生成模型的关系**
   - Diffusion Model vs Flow Matching
   - Score-based Models
   - Normalizing Flows

**学习资源**：
- 论文：Flow Matching for Generative Modeling (Lipman et al., 2023)
- 视频：YouTube 上的 Flow Matching 讲解
- 博客：Lil'Log 的生成模型系列

**输出**：
- [ ] 完成理论笔记（已创建 `flow_matching_theory.md`）
- [ ] 手推 CFM Loss 的数学推导
- [ ] 绘制 Flow Matching 的示意图

---

#### 任务 1.2：阅读 Goal Flow Matching 论文
**时间**：2-3天

**论文信息**：
- 标题：Goal Flow Matching (需要确认具体论文)
- 可能的相关论文：
  - "Motion Planning Diffusion" (ICLR 2023)
  - "Guided Diffusion for Trajectory Prediction" (CVPR 2023)
  - "Goal-Conditioned Trajectory Generation" (CoRL 2023)

**阅读重点**：
1. **核心创新点**
   - 与标准 Flow Matching 的区别
   - 如何引入 Goal 信息
   - 网络架构的改进

2. **技术细节**
   - 条件编码方式
   - 损失函数设计
   - 训练技巧

3. **实验设置**
   - 使用的数据集
   - 评估指标
   - 基线方法对比

**输出**：
- [ ] 论文精读笔记
- [ ] 关键公式和算法伪代码
- [ ] 与我们当前实现的对比分析

---

#### 任务 1.3：分析当前实现与论文的差异
**时间**：1天

**对比维度**：

| 维度 | 当前实现 | Goal Flow Matching | 需要改进 |
|------|----------|-------------------|----------|
| **数据** | Toy 2D轨迹 | 真实驾驶轨迹 | ✅ 需要 |
| **条件** | goal + type | goal + map + agents | ✅ 需要 |
| **网络** | MLP | Transformer/GNN | ✅ 需要 |
| **特征** | 简单拼接 | Cross-attention | ✅ 需要 |
| **评估** | 可视化 | FDE/ADE/CR | ✅ 需要 |

**输出**：
- [ ] 差异分析文档
- [ ] 改进优先级列表
- [ ] 实现计划

---

## 📊 阶段 2：论文复现（3-4周）

### 任务 2.1：选择和准备数据集
**时间**：3-5天

**候选数据集**：

#### 选项 1：nuScenes（推荐）
- **优点**：
  - 数据质量高，标注完整
  - 包含地图、障碍物、轨迹
  - 社区支持好，有很多参考代码
  - 适合自动驾驶场景
  
- **缺点**：
  - 数据量大（~350GB）
  - 需要申请下载

- **数据内容**：
  - 1000个场景，每个场景20秒
  - 6个相机 + 5个雷达 + 1个激光雷达
  - 高精地图（车道线、路口等）
  - 目标轨迹标注

#### 选项 2：Waymo Open Dataset
- **优点**：
  - 数据量最大
  - 场景多样性好
  
- **缺点**：
  - 数据量巨大（>1TB）
  - 处理复杂

#### 选项 3：Argoverse 2
- **优点**：
  - 专注于轨迹预测
  - 数据格式简单
  
- **缺点**：
  - 地图信息相对简单

**推荐**：从 **nuScenes** 开始

**数据处理流程**：
```python
# 1. 下载 nuScenes mini (10% 数据，用于快速验证)
# 2. 提取轨迹数据
#    - 历史轨迹：过去2秒（4个时间步）
#    - 未来轨迹：未来6秒（12个时间步）
# 3. 提取条件信息
#    - Goal: 未来轨迹的终点
#    - Map: 周围的车道线、路口
#    - Agents: 周围车辆的轨迹
# 4. 数据增强
#    - 旋转、平移
#    - 添加噪声
```

**输出**：
- [ ] 数据下载和预处理脚本
- [ ] 数据加载器（Dataset 类）
- [ ] 数据统计和可视化

---

### 任务 2.2：实现 Goal Flow Matching 模型架构
**时间**：1-2周

#### 2.2.1 改进网络架构

**从 MLP 升级到 Transformer**：

```python
class GoalFlowMatchingModel(nn.Module):
    """
    Goal Flow Matching 模型
    
    输入：
        - trajectory: (B, T, 2) 当前轨迹状态
        - goal: (B, 2) 目标点
        - map: (B, N_lane, 2) 车道线点
        - agents: (B, N_agent, T, 2) 周围车辆轨迹
        - time: (B,) 时间标量
    
    输出：
        - velocity: (B, T, 2) 速度场
    """
    def __init__(self):
        # 1. 编码器
        self.traj_encoder = TrajectoryEncoder()
        self.goal_encoder = GoalEncoder()
        self.map_encoder = MapEncoder()
        self.agent_encoder = AgentEncoder()
        self.time_encoder = TimeEncoder()
        
        # 2. 融合模块（Cross-Attention）
        self.fusion = CrossAttentionFusion()
        
        # 3. 解码器
        self.decoder = VelocityDecoder()
    
    def forward(self, trajectory, goal, map, agents, time):
        # 编码
        traj_feat = self.traj_encoder(trajectory)
        goal_feat = self.goal_encoder(goal)
        map_feat = self.map_encoder(map)
        agent_feat = self.agent_encoder(agents)
        time_feat = self.time_encoder(time)
        
        # 融合
        context = self.fusion(
            query=traj_feat,
            key_value=[goal_feat, map_feat, agent_feat],
            time=time_feat
        )
        
        # 解码
        velocity = self.decoder(context)
        return velocity
```

**关键模块**：

1. **Trajectory Encoder**
   - 输入：轨迹点序列
   - 输出：轨迹特征
   - 方法：1D CNN 或 Transformer

2. **Map Encoder**
   - 输入：车道线点云
   - 输出：地图特征
   - 方法：PointNet 或 VectorNet

3. **Agent Encoder**
   - 输入：周围车辆轨迹
   - 输出：交互特征
   - 方法：GNN 或 Transformer

4. **Cross-Attention Fusion**
   - 将轨迹特征与条件特征融合
   - 使用 Multi-Head Attention

**输出**：
- [ ] 实现各个编码器模块
- [ ] 实现融合模块
- [ ] 单元测试（测试每个模块的输入输出）

---

#### 2.2.2 改进训练流程

**新的训练目标**：

```python
# 1. CFM Loss（主要损失）
loss_cfm = E[ ||v_θ(x_t, cond, t) - (x_1 - x_0)||² ]

# 2. Goal Reaching Loss（辅助损失）
loss_goal = ||x_1[-1] - goal||²

# 3. Collision Loss（可选）
loss_collision = collision_penalty(x_1, obstacles)

# 总损失
loss = loss_cfm + λ_goal * loss_goal + λ_collision * loss_collision
```

**训练技巧**：
- Warmup learning rate
- Gradient clipping
- EMA (Exponential Moving Average)
- Mixed precision training

**输出**：
- [ ] 更新训练脚本
- [ ] 添加多个损失函数
- [ ] 实现训练监控（TensorBoard）

---

### 任务 2.3：在真实数据集上训练和评估
**时间**：3-5天

#### 评估指标

**1. 轨迹精度**：
- **ADE (Average Displacement Error)**：平均位移误差
  ```python
  ADE = mean(||pred_traj - gt_traj||)
  ```
  
- **FDE (Final Displacement Error)**：终点位移误差
  ```python
  FDE = ||pred_traj[-1] - gt_traj[-1]||
  ```

**2. 多模态评估**：
- **minADE / minFDE**：K条轨迹中最好的
- **MR (Miss Rate)**：终点误差 > 2m 的比例

**3. 场景理解**：
- **CR (Collision Rate)**：碰撞率
- **OR (Off-Road Rate)**：偏离道路率

**基线方法对比**：
- Constant Velocity
- LSTM
- Social GAN
- Trajectron++

**输出**：
- [ ] 实现评估脚本
- [ ] 在验证集上评估
- [ ] 生成评估报告和可视化

---

## 🚗 阶段 3：AVP 场景适配（2-3周）

### 任务 3.1：理解 AVP 记忆泊车场景
**时间**：2-3天

**AVP 场景特点**：
1. **低速场景**（<20 km/h）
2. **结构化环境**（停车场）
3. *）
4. **精确控制**（厘米级精度）

**与开放道路的区别**：

| 特性 | 开放道路 | AVP 停车场 |
|------|----------|-----------|
| 速度 | 高（>50km/h） | 低（<20km/h） |
| 环境 | 动态、复杂 | 结构化、静态 |
| 轨迹 | 多样 | 重复、固定 |
| 精度要求 | 米级 | 厘米级 |
| 障碍物 | 车辆、行人 | 车位、柱子 |

**记忆泊车的特殊需求**：
1. **路径记忆**：记住之前走过的路径
2. **路径复现**：在相似场景下复现路径
3. **动态避障**：遇到障碍物时调整路径
4. **精确停车**：准确停到目标车位

**输出**：
- [ ] AVP 场景需求文档
- [ ] 与通用场景的差异分析
- [ ] 模型适配方案

---

### 任务 3.2：数据格式适配
**时间**：2-3天

**AVP 数据格式**：

```python
# 输入数据
{
    'ego_state': {
        'position': (x, y, heading),  # 当前位置
        'velocity': (vx, vy),          # 当前速度
        'timestamp': t                 # 时间戳
    },
    
    'goal': {
        'position': (x_goal, y_goal, heading_goal),  # 目标车位
        'type': 'parking_slot'                       # 目标类型
    },
    
    'map': {
        'parking_slots': [...],   # 车位信息
        'lanes': [...],           # 车道线
        'obstacles': [...]        # 静态障碍物（柱子等）
    },
    
    'history_trajectory': {
        'positions': [(x, y), ...],  # 历史轨迹
        'timestamps': [t, ...]       # 时间戳
    },
    
    'reference_trajectory': {  # 记忆的参考轨迹
        'positions': [(x, y), ...],
        'valid': True/False     # 是否有参考轨迹
    }
}

# 输出数据
{
    'predicted_trajectory': {
        'positions': [(x, y), ...],  # 预测轨迹（未来6秒）
        'velocities': [(vx, vy), ...],
        'confidence': [0.0-1.0, ...]
    }
}
```

**数据预处理**：
1. 坐标系转换（世界坐标 → 车辆坐标）
2. 归一化（位置、速度）
3. 数据增强（旋转、平移、噪声）

**输出**：
- [ ] AVP 数据加载器
- [ ] 数据预处理脚本
- [ ] 数据可视化工具

---

### 任务 3.3：模型适配和优化
**时间**：1周

**模型改进**：

1. **添加参考轨迹编码器**
   ```python
   class ReferenceTrajectoryEncoder(nn.Module):
       """编码记忆的参考轨迹"""
       def forward(self, ref_traj, valid_mask):
           # 如果有参考轨迹，编码它
           # 如果没有，返回零向量
           pass
   ```

2. **添加精确控制模块**
   ```python
   # 在接近目标时，增加位置精度
   if distance_to_goal < threshold:
       velocity = velocity * precision_factor
   ```

3. **添加动力学约束**
   ```python
   # 低速场景的运动学约束
   max_velocity = 5.0  # m/s
   max_acceleration = 2.0  # m/s²
   max_steering = 30  # degrees
   ```

**训练策略**：
1. **预训练**：在 nuScenes 上预训练
2. **微调**：在 AVP 数据上微调
3. **在线学习**：收集新数据持续优化

**输出**：
- [ ] AVP 专用模型
- [ ] 训练脚本
- [ ] 评估脚本

---

### 任务 3.4：集成到工程系统
**时间**：3-5天

**系统架构**：

```
┌─────────────────────────────────────────────┐
│           AVP 记忆泊车系统                    │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────┐    ┌──────────────────────┐  │
│  │ 感知模块  │───>│  Goal Flow Matching  │  │
│  │          │    │   轨迹生成模块        │  │
│  │ - 定位   │    │                      │  │
│  │ - 地图   │    │  - 轨迹预测          │  │
│  │ - 障碍物 │    │  - 路径规划          │  │
│  └──────────┘    │  - 动态避障          │  │
│                  └──────────────────────┘  │
│                           │                 │
│                           ▼                 │
│                  ┌──────────────────────┐  │
│                  │   控制模块            │  │
│                  │                      │  │
│                  │  - 轨迹跟踪          │  │
│                  │  - 速度控制          │  │
│                  └──────────────────────┘  │
│                                             │
└─────────────────────────────────────────────┘
```

**接口设计**：

```python
class AVPTrajectoryGenerator:
    """AVP 轨迹生成器"""
    
    def __init__(self, model_path):
        self.model = load_model(model_path)
    
    def generate_trajectory(
        self,
        ego_state: dict,
        goal: dict,
        map_info: dict,
        reference_traj: dict = None
    ) -> dict:
        """
        生成泊车轨迹
        
        Returns:
            {
                'trajectory': np.ndarray,  # (T, 2)
                'velocity': np.ndarray,    # (T, 2)
                'confidence': float
            }
        """
        pass
    
    def update_with_feedback(self, trajectory, success):
        """根据执行反馈更新模型"""
        pass
```

**输出**：
- [ ] Python API 接口
- [ ] C++ 接口（如果需要）
- [ ] ROS 节点（如果使用 ROS）
- [ ] 集成测试

---

## 📈 阶段 4：数据收集与优化（持续）

### 任务 4.1：数据收集方案
**时间**：1周

**数据收集策略**：

1. **仿真数据**（初期）
   - 使用 CARLA/LGSVL 仿真器
   - 生成大量场景
   - 快速迭代

2. **实车数据**（中期）
   - 在测试场收集
   - 标注关键信息
   - 质量控制

3. **在线数据**（后期）
   - 实际运行中收集
   - 自动标注
   - 持续优化

**数据标注**：
```python
# 需要标注的信息
{
    'trajectory': [...],      # 实际行驶轨迹
    'goal': {...},           # 目标车位
    'map': {...},            # 地图信息
    'success': True/False,   # 是否成功泊车
    'failure_reason': '',    # 失败原因（如果失败）
    'quality_score': 0-10    # 轨迹质量评分
}
```

**输出**：
- [ ] 数据收集工具
- [ ] 数据标注工具
- [ ] 数据管理系统

---

### 任务 4.2：模型持续优化
**时间**：持续

**优化策略**：

1. **定期重训练**
   - 每周/每月用新数据重训练
   - 对比新旧模型性能
   - A/B 测试

2. **在线学习**
   - 收集失败案例
   - 针对性优化
   - 快速部署

3. **模型蒸馏**
   - 大模型 → 小模型
   - 保持性能，提升速度
   - 适配嵌入式设备

**监控指标**：
- 成功率
- 平均泊车时间
- 用户满意度
- 系统稳定性

**输出**：
- [ ] 模型版本管理
- [ ] 性能监控系统
- [ ] 自动化训练流程

---

## 📊 项目时间线

```
Week 1-2:  理论学习 + 论文阅读
Week 3-4:  数据准备 + 模型实现
Week 5-6:  模型训练 + 评估
Week 7-8:  AVP 适配 + 集成
Week 9+:   数据收集 + 持续优化
```

---

## 🛠️ 技术栈

### 深度学习框架
- PyTorch 2.0+
- PyTorch Lightning（训练框架）
- Weights & Biases（实验管理）

### 数据处理
- NumPy, Pandas
- nuScenes-devkit
化）

### 工程部署
- ONNX（模型转换）
- TensorRT（推理加速）
- Docker（容器化）
- ROS/ROS2（机器人系统）

---

## 📚 学习资源

### 论文
1. Flow Matching for Generative Modeling (ICLR 2023)
2. Conditional Flow Matching (ICML 2023)
3. Motion Planning Diffusion (ICLR 2023)
4. VectorNet (CVPR 2020)
5. TNT: Target-driven Trajectory Prediction (CoRL 2020)

### 代码参考
1. [torchdyn](https://github.com/DiffEqML/torchdyn) - Flow Matching 实现
2. [nuScenes-devkit](https://github.com/nutonomy/nuscenes-devkit)
3. [Trajectron++](https://github.com/StanfordASL/Trajectron-plus-plus)

### 课程
1. Stanford CS231n (Deep Learning for Computision)
2. Berkeley CS285 (Deep Reinforcement Learning)
3. Coursera: Self-Driving Cars Specialization

---

## ✅ 检查清单

### 阶段 1
- [ ] 完成 Flow Matching 理论学习
- [ ] 阅读 Goal Flow Matching 论文
- [ ] 分析当前实现与论文差异

### 阶段 2
- [ ] 下载并处理 nuScenes 数据
- [ ] 实现 Transformer 架构
- [ ] 训练并评估模型
- [ ] 达到论文中的性能指标

### 阶段 3
- [ ] 理解 AVP 场景需求
- [ ] 适配数据格式
- [ ] 优化模型
- [ ] 集成到工程系统

### 阶段 4
- [ ] 建立数据收集流程
- [ ] 实现持续优化机制
- [ ] 部署到实车

---

## 🎯 成功标准

### 短期目标（1-2个月）
- ✅ 在 nuScenes 上复现论文结果
- ✅ ADE < 1.0m, FDE < 2.0m
- ✅ 完成 AVP 场景适配

### 中期目标（3-6个月）
- ✅ 在实车上测试
- ✅ 泊车成功率 > 95%
- ✅ 收集 1000+ 实车数据

### 长期目标（6-12个月）
- ✅ 系统稳定运行
- ✅ 支持多种停车场景
- ✅ 用户满意度 > 90%

---

## 📞 需要帮助？

在实施过程中遇到问题，可以：
1. 查看论文和代码
2. 在 GitHub 上搜索相关实现
3. 在论坛（如 Reddit, Stack Overflow）提问
4. 联系论文作者

---

**最后更新**：2026-02-06
**版本**：v1.0
