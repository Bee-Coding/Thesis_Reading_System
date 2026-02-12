# GoalFlow Toy Dataset

简化的轨迹数据集，用于快速验证 GoalFlow 模型的完整流程。

## 数据生成

### 生成数据
```bash
cd /home/zhn/work/text/Thesis_Reading_System/implementations/goalflow
python data/generate_toy_data.py --num_samples 1000 --output data/toy_data.npz
```

### 参数说明
- `--num_samples`: 生成的轨迹数量（默认：1000）
- `--num_points`: 每条轨迹的点数（默认：6）
- `--n_clusters`: 词汇表大小（默认：128）
- `--output`: 输出文件路径
- `--seed`: 随机种子（默认：42）

## 数据结构

生成的 `.npz` 文件包含以下数据：

```python
{
    'trajectories': (N, T, 2),      # N条轨迹，每条T个点
    'goals': (N, 2),                # 目标点
    'start_points': (N, 2),         # 起始点
    'vocabulary': (n_clusters, 2),  # 目标点词汇表（K-means聚类中心）
    'bev_features': (N, C, H, W),   # BEV特征 (C=64, H=W=32)
    'drivable_area': (N, H, W)      # 可行驶区域 mask
}
```

### 数据特点

1. **多模态目标分布**：目标点分布在4个区域
   - 区域1: (10, 10) - 右上
   - 区域2: (10, -10) - 右下
   - 区域3: (-10, 10) - 左上
   - 区域4: (-10, -10) - 左下

2. **平滑轨迹**：使用三次样条插值生成平滑曲线

3. **简化的 BEV 特征**：随机特征 + 空间结构模式

4. **圆形可行驶区域**：中心区域为可行驶区域

## 数据加载

### 使用 PyTorch Dataset

```python
from data.toy_goalflow_dataset import ToyGoalFlowDataset
from torch.utils.data import DataLoader

# 创建数据集
train_dataset = ToyGoalFlowDataset('data/toy_data.npz', split='train')
val_dataset = ToyGoalFlowDataset('data/toy_data.npz', split='val')

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 获取词汇表（所有样本共享）
vocabulary = train_dataset.get_vocabulary()  # (128, 2)

# 迭代数据
for batch in train_loader:
    trajectory = batch['trajectory']      # (B, T, 2)
    goal = batch['goal']                  # (B, 2)
    start_point = batch['start_point']    # (B, 2)
    bev_feature = batch['bev_feature']    # (B, C, H, W)
    drivable_area = batch['drivable_area']  # (B, H, W)
```

### 简单测试（不需要 matplotlib）

```bash
python data/test_dataset_simple.py
```

输出示例：
```
✅ Loaded train dataset: 800 samples
✅ Loaded val dataset: 200 samples
📊 Dataset statistics:
   - Train samples: 800
   - Val samples: 200
   - Train batches: 25
   - Val batches: 7
🔍 Testing batch loading...
   Batch shapes:
   - trajectory: torch.Size([32, 6, 2])
   - goal: torch.Size([32, 2])
   - bev_feature: torch.Size([32, 64, 32, 32])
   - drivable_area: torch.Size([32, 32, 32])
   - vocabulary: torch.Size([128, 2])
✅ All tests passed!
```

## 数据统计

- **训练集**: 800 samples (80%)
- **验证集**: 200 samples (20%)
- **轨迹范围**: 约 [-15, 15]
- **目标点范围**: 约 [-12, 13]
- **BEV 特征**: 64 通道，32x32 分辨率
- **可行驶区域**: 32x32 二值 mask

## 下一步

数据集准备完成后，可以开始：

1. **训练 GoalPointScorer**：学习目标点评分
2. **训练 GoalFlowMatcher**：学习轨迹生成
3. **端到端测试**：完整的 GoalFlow 流程
4. **迁移到真实数据**：nuScenes 数据集

## 文件结构

```
data/
├── generate_toy_data.py          # 数据生成脚本
├── toy_goalflow_dataset.py       # PyTorch Dataset 类（带可视化）
├── test_dataset_simple.py        # 简单测试脚本（无 matplotlib）
├── toy_data.npz                  # 生成的数据文件
└── README.md                     # 本文档
```

## 注意事项

1. 如果遇到 matplotlib 库依赖问题，使用 `test_dataset_simple.py` 进行测试
2. 词汇表是通过 K-means 聚类生成的，所有样本共享同一个词汇表
3. 数据坐标系：原点在中心，X轴向右，Y轴向上
4. 可行驶区域坐标转换：假设轨迹坐标范围 [-50, 50] 映射到图像 [0, 31]
