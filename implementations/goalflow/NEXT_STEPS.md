# 🚀 GoalFlow 下一步计划

**当前状态**: ✅ GoalFlowMatcher 实现完成并测试通过  
**日期**: 2026-02-10

---

## 📊 当前进度

```
GoalFlow 实现进度：40% 完成

✅ GoalPointScorer      100% ✅ 已完成并测试
✅ GoalFlowMatcher      100% ✅ 已完成并测试
⏳ TrajectorySel ector    0% ⏳ 待实现
⏳ Toy Dataset           0% ⏳ 待实现
⏳ 端到端训练            0% ⏳ 待实现
⏳ 真实数据集适配        0% ⏳ 待实现
```

---

## 🎯 下一步选项（3选1）

### 选项 1：创建 Toy Dataset（推荐）⭐⭐⭐

**目标**: 创建简化的训练数据集，用于快速验证整个流程

**为什么推荐**:
- 可以快速验证 GoalPointScorer + GoalFlowMatcher 的联合工作
- 不需要下载大型数据集（nuScenes 35GB）
- 可以快速迭代和调试
- 为后续真实数据训练打基础

**工作内容**:
1. 生成模拟轨迹数据
   - 起点：随机分布
   - 终点：4个区域（模拟多模态）
   - 轨迹：平滑曲线连接起点和终点
   
2. 生成模拟 BEV 特征
   - 简化为随机噪声或固定模式
   - 形状：(B, 64, 32, 32)
   
3. 生成模拟可行驶区域
   - 简单的矩形或圆形区域
   - 形状：(B, 32, 32)
   
4. 构建 Goal Point Vocabulary
   - 对终点聚类（K-means，K=128）
   - 保存为 .npy 文件
   
5. 实现 DataLoader
   - 支持批量加载
   - 数据增强（可选）

**预计时间**: 1-2天  
**难度**: 中等

**输出**:
```
data/
├── toy_trajectories.npz      # 轨迹数据
├── toy_vocabulary.npy         # Goal Point Vocabulary
└── toy_goalflow_dataset.py   # DataLoader
```

---

### 选项 2：实现 Trajectory Selector（可选）⭐⭐

**目标**: 实现轨迹评分和选择模块

**为什么可选**:
- 这是 GoalFlow 的最后一个模块
- 需要先有生成的轨迹才能测试
- 可以先用简单的距离评分代替

**工作内容**:
1. 实现轨迹评分函数
   - Distance Score: 与真实轨迹的距离
   - Progress Score: 朝向目标的进度
   - 公式：`f(τ) = -λ1·Φ(f_dis) + λ2·Φ(f_pg)`
   
2. 实现 Shadow Trajectories 生成
   - Mask 部分目标点
   - 生成多条影子轨迹
   
3. 实现最优轨迹选择
   - 选择评分最高的轨迹
   
4. 编写测试

**预计时间**: 1-1.5天  
**难度**: 中等

**输出**:
```
models/
└── trajectory_selector.py    # 轨迹选择器
```

---

### 选项 3：端到端训练（推荐在有数据后）⭐

**目标**: 联合训练 GoalPointScorer + GoalFlowMatcher

**前置条件**:
- ✅ GoalPointScorer 已实现
- ✅ GoalFlowMatcher 已实现
- ⏳ 需要先有 Toy Dataset

**工作内容**:
1. 创建训练脚本
   - 联合训练两个模块
   - 或者分阶段训练
   
2. 实现训练循环
   - 数据加载
   - 前向传播
   - 损失计算
   - 反向传播
   - 优化器更新
   
3. 实现可视化
   - 生成轨迹可视化
   - 损失曲线
   - Goal Point 选择可视化
   
4. 超参数调整

**预计时间**: 2-3天  
**难度**: 中等

**输出**:
```
train_goalflow.py              # 训练脚本
visualize_results.py           # 可视化脚本
checkpoints/                   # 模型权重
logs/                          # 训练日志
```

---

## 📋 推荐的实施顺序

### 阶段 1：Toy Dataset + 端到端训练（3-5天）

```
Day 1-2: 创建 Toy Dataset
  ├─ 生成模拟轨迹数据
  ├─ 构建 Goal Point Vocabulary
  └─ 实现 DataLoader

Day 3-4: 端到端训练
  ├─ 创建训练脚本
  ├─ 联合训练两个模块
  └─ 可视化结果

Day 5: 调试和优化
  ├─ 调整超参数
  ├─ 分析生成质量
  └─ 修复问题
```

**里程碑**: 在 Toy Dataset 上成功训练，生成合理的轨迹

---

### 阶段 2：Trajectory Selector（1-2天）

```
Day 1: 实现 Trajectory Selector
  ├─ 实现评分函数
  ├─ 实现轨迹选择
  └─ 编写测试

Day 2: 集成到训练流程
  ├─ 修改训练脚本
  ├─ 生成多条候选轨迹
  └─ 选择最优轨迹
```

**里程碑**: 完整的 GoalFlow 流程可以运行

---

### 阶段 3：真实数据集适配（1-2周）

```
Week 1: 数据准备
  ├─ 下载 nuScenes mini (35GB)
  ├─ 提取轨迹数据
  ├─ 提取 BEV 特征
  └─ 构建 Goal Point Vocabulary

Week 2: 训练和评估
  ├─ 在真实数据上训练
  ├─ 实现评估指标 (ADE, FDE, DAC)
  ├─ 对比论文结果
  └─ 优化模型
```

**里程碑**: 在 nuScenes 上达到合理的性能

---

## 🎯 我的建议

**最佳路线**：

1. **先做选项 1（Toy Dataset）** ⭐⭐⭐
   - 快速验证整个流程
   - 发现潜在问题
   - 建立信心

2. **然后做选项 3（端到端训练）** ⭐⭐⭐
   - 在 Toy Dataset 上训练
   - 可视化结果
   - 调试和优化

3. **最后做选项 2（Trajectory Selector）** ⭐⭐
   - 完善整个系统
   - 提升性能

4. **准备真实数据集** ⭐
   - 下载 nuScenes
   - 适配数据格式
   - 大规模训练

---

## 📝 Toy Dataset 详细设计

### 数据生成策略

```python
# 1. 轨迹生成
def generate_toy_trajectories(num_samples=1000):
    """
    生成模拟轨迹数据
    
    策略：
    - 起点：在原点附近随机分布 N(0, 1)
    - 终点：4个区域（模拟多模态）
      - 区域1: (10, 10)
      - 区域2: (10, -10)
      - 区域3: (-10, 10)
      - 区域4: (-10, -10)
    - 轨迹：使用三次样条插值生成平滑曲线
    """
    trajectories = []
    goals = []
    
    for i in range(num_samples):
        # 随机选择一个目标区域
        region = np.random.choice(4)
        goal_centers = [(10, 10), (10, -10), (-10, 10), (-10, -10)]
        goal = goal_centers[region] + np.random.randn(2) * 2
        
        # 生成起点
        start = np.random.randn(2) * 1
        
        # 生成轨迹（6个点）
        t = np.linspace(0, 1, 6)
        trajectory = start + (goal - start) * t[:, None]
        
        # 添加一些噪声使轨迹更自然
        trajectory += np.random.randn(6, 2) * 0.5
        
        trajectories.append(trajectory)
        goals.append(goal)
    
    return np.array(trajectories), np.array(goals)

# 2. Goal Point Vocabulary 构建
def build_vocabulary(goals, n_clusters=128):
    """
    对终点聚类构建 Vocabulary
    """
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(goals)
    
    vocabulary = kmeans.cluster_centers_
    return vocabulary

# 3. BEV 特征生成（简化）
def generate_bev_features(num_samples=1000):
    """
    生成简化的 BEV 特征
    """
    # 简化为随机噪声
    bev_features = np.random.randn(num_samples, 64, 32, 32).astype(np.float32)
    return bev_features

# 4. 可行驶区域生成
def generate_drivable_area(num_samples=1000):
    """
    生成简化的可行驶区域
    """
    # 简化为中心的圆形区域
    drivable_areas = []
    for i in range(num_samples):
        area = np.zeros((32, 32), dtype=np.float32)
        center = (16, 16)
        radius = 12
        
        for y in range(32):
            for x in range(32):
                if (x - center[0])**2 + (y - center[1])**2 < radius**2:
                    area[y, x] = 1.0
        
        drivable_areas.append(area)
    
    return np.array(drivable_areas)
```

### DataLoader 实现

```python
class ToyGoalFlowDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, split='train'):
        """
        Toy GoalFlow Dataset
        
        Args:
            data_path: 数据文件路径
            split: 'train' 或 'val'
        """
        data = np.load(data_path)
        
        self.trajectories = torch.from_numpy(data['trajectories'])
        self.goals = torch.from_numpy(data['goals'])
        self.bev_features = torch.from_numpy(data['bev_features'])
        self.drivable_areas = torch.from_numpy(data['drivable_areas'])
        self.vocabulary = torch.from_numpy(data['vocabulary'])
        
        # 划分训练集和验证集
        n = len(self.trajectories)
        if split == 'train':
            self.indices = range(0, int(n * 0.8))
        else:
            self.indices = range(int(n * 0.8), n)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        idx = self.indices[idx]
        
        return {
            'trajectory': self.trajectories[idx],
            'goal': self.goals[idx],
            'bev_feature': self.bev_features[idx],
            'drivable_area': self.drivable_areas[idx],
            'vocabulary': self.vocabulary  # 共享的 vocabulary
        }
```

---

## 🔑 关键文件清单

### 需要创建的文件

```
implementations/goalflow/
├── data/
│   ├── generate_toy_data.py       # 生成 Toy 数据
│   ├── toy_goalflow_dataset.py    # DataLoader
│   └── toy_data.npz               # 生成的数据
├── train_goalflow.py              # 训练脚本
├── visualize_results.py           # 可视化脚本
└── configs/
    └── toy_config.yaml            # 配置文件
```

---

## 💡 快速开始命令

### 创建 Toy Dataset

```bash
cd /home/zhn/work/text/Thesis_Reading_System/implementations/goalflow

# 1. 生成数据
python data/generate_toy_data.py --num_samples 1000 --output data/toy_data.npz

# 2. 验证数据
python data/toy_goalflow_dataset.py --data_path data/toy_data.npz --visualize

# 3. 开始训练
python train_goalflow.py --config configs/toy_config.yaml --epochs 100
```

---

## 📞 遇到问题？

1. **数据生成问题** → 检查 `generate_toy_data.py`
2. **训练问题** → 检查 `train_goalflow.py`
3. **可视化问题** → 检查 `visualize_results.py`
4. **模型问题** → 回到 `test/` 目录运行测试

---

## 🎉 总结

你已经完成了 GoalFlow 的两个核心模块！

**已完成**:
- ✅ GoalPointScorer (100%)
- ✅ GoalFlowMatcher (100%)

**下一步**:
- 🎯 创建 Toy Dataset（推荐）
- 🎯 端到端训练
- 🎯 实现 Trajectory Selector

**预计剩余时间**: 1-2周（Toy Dataset + 训练）

加油！你正在做一件很酷的事情！🚀
