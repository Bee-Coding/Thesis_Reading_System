# nuScenes 快速开始指南

## 🎯 你需要做什么

我已经为你创建了完整的代码框架，你只需要实现 **3 个核心函数**即可！

## 📁 已创建的文件

### ✅ 完整实现（无需修改）
- `config/nuscenes_config.py` - 配置文件
- `data/nuscenes_utils.py` - 工具函数
- `data/nuscenes_dataset.py` - PyTorch 数据集类
- `scripts/test_nuscenes_installation.py` - 环境验证
- `scripts/preprocess_nuscenes.py` - 预处理脚本
- `data/test_nuscenes_data.py` - 数据测试
- `train_scorer_nuscenes.py` - 训练脚本

### ⚠️ 需要你实现（3 个函数）
- `data/nuscenes_preprocessor.py` - 数据预处理模块
  - `build_vocabulary()` - 第 240 行（⭐ 简单，1小时）
  - `extract_agent_trajectories()` - 第 40 行（⭐⭐⭐ 中等，2-3小时）
  - `rasterize_map()` - 第 207 行（⭐⭐⭐⭐ 较难，3-4小时）

## 🚀 快速开始

### 步骤 1: 验证环境（5 分钟）

```bash
cd implementations/goalflow
python scripts/test_nuscenes_installation.py
```

如果环境未准备好，按照提示安装：
```bash
pip install nuscenes-devkit opencv-python
```

### 步骤 2: 实现 3 个函数（6-8 小时）

打开 `data/nuscenes_preprocessor.py`，按顺序实现：

#### 2.1 `build_vocabulary()` - 第 240 行

最简单的函数，只需 5 行代码：

```python
from sklearn.cluster import KMeans

def build_vocabulary(goal_points, n_clusters=256, seed=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    kmeans.fit(goal_points)
    return kmeans.cluster_centers_
```

#### 2.2 `extract_agent_trajectories()` - 第 40 行

参考文件中的详细注释和代码提示，核心逻辑：
1. 遍历场景中的所有帧
2. 对每个车辆标注，提取历史和未来轨迹
3. 转换到 ego 坐标系
4. 过滤无效轨迹

#### 2.3 `rasterize_map()` - 第 207 行

最复杂的函数，核心逻辑：
1. 创建空白 BEV 图像
2. 获取附近的地图元素（车道线、道路、人行道）
3. 转换坐标并绘制到图像
4. 归一化并返回

### 步骤 3: 运行预处理（30 分钟）

```bash
python scripts/preprocess_nuscenes.py
```

### 步骤 4: 测试数据（5 分钟）

```bash
python data/test_nuscenes_data.py
```

### 步骤 5: 训练模型（2-3 小时）

```bash
python train_scorer_nuscenes.py
```

## 📖 详细文档

- **完整实施指南**: `NUSCENES_IMPLEMENTATION_GUIDE.md`
- **环境配置**: `NUSCENES_SETUP.md`
- **整体计划**: `GOALFLOW_NUSCENES_PLAN.md`

## 💡 实施建议

1. **先实现最简单的**: 从 `build_vocabulary()` 开始
2. **及时测试**: 每实现一个函数就测试
3. **参考注释**: 每个函数都有详细的实现步骤和代码提示
4. **遇到问题**: 查看文档或询问我

## 🎯 核心函数位置

```
data/nuscenes_preprocessor.py
├── build_vocabulary()              # 第 240 行 ⭐
├── extract_agent_trajectories()    # 第 40 行  ⭐⭐⭐
└── rasterize_map()                 # 第 207 行 ⭐⭐⭐⭐
```

每个函数都包含：
- 详细的文档字符串
- 清晰的实现步骤
- 具体的代码提示
- 使用示例

## ✅ 完成检查

- [ ] 环境验证通过
- [ ] 实现 `build_vocabulary()`
- [ ] 实现 `extract_agent_trajectories()`
- [ ] 实现 `rasterize_map()`
- [ ] 预处理成功
- [ ] 数据测试通过
- [ ] 模型训练成功

## 🆘 需要帮助？

如果遇到问题：
1. 查看函数内的详细注释
2. 参考 `NUSCENES_IMPLEMENTATION_GUIDE.md`
3. 查看 nuScenes 官方文档
4. 随时询问我

祝你实现顺利！🚀
