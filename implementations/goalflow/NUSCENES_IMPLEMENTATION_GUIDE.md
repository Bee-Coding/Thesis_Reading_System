# nuScenes 数据集集成 - 实施指南

## 📋 已创建的文件

### ✅ 核心实现文件（需要你填充）

1. **`data/nuscenes_preprocessor.py`** - 数据预处理模块
   - ⚠️ `extract_agent_trajectories()` - 轨迹提取（难度: ⭐⭐⭐）
   - ⚠️ `rasterize_map()` - 地图栅格化（难度: ⭐⭐⭐⭐）
   - ⚠️ `build_vocabulary()` - 词汇表构建（难度: ⭐）

2. **`data/nuscenes_dataset.py`** - PyTorch 数据集类
   - ✅ 已完整实现，无需修改

### ✅ 配置和工具文件（已完成）

3. **`config/nuscenes_config.py`** - 配置文件（已存在）
4. **`data/nuscenes_utils.py`** - 工具函数（已存在）

### ✅ 脚本文件（已完成）

5. **`scripts/test_nuscenes_installation.py`** - 环境验证脚本
6. **`scripts/preprocess_nuscenes.py`** - 预处理执行脚本
7. **`data/test_nuscenes_data.py`** - 数据测试脚本
8. **`train_scorer_nuscenes.py`** - Scorer 训练脚本

---

## 🎯 实施步骤

### 第 1 步：环境准备

#### 1.1 安装依赖包

```bash
cd implementations/goalflow

# 安装基础依赖
pip install torch torchvision numpy scipy scikit-learn tqdm matplotlib opencv-python

# 安装 nuScenes devkit
pip install nuscenes-devkit
```

#### 1.2 下载 nuScenes mini 数据集

1. 访问 https://www.nuscenes.org/nuscenes
2. 注册账号（免费）
3. 下载以下文件：
   - `v1.0-mini.tgz` (~4GB)
   - `nuScenes-map-expansion-v1.3.zip` (~500MB)

4. 解压到项目目录：

```bash
# 创建数据目录
mkdir -p data/nuscenes

# 解压数据
tar -xzf v1.0-mini.tgz -C data/nuscenes/
unzip nuScenes-map-expansion-v1.3.zip -d data/nuscenes/maps/
```

#### 1.3 验证环境

```bash
python scripts/test_nuscenes_installation.py
```

预期输出：
```
✅ 所有测试通过！环境配置正确。
```

---

### 第 2 步：实现核心函数

你需要实现 `data/nuscenes_preprocessor.py` 中的 3 个核心函数。

#### 建议实现顺序：

##### 2.1 先实现 `build_vocabulary()` （最简单，热身）

**位置**: `data/nuscenes_preprocessor.py:240`

**任务**: 使用 K-means 聚类构建 goal 词汇表

**代码提示**:
```python
from sklearn.cluster import KMeans

def build_vocabulary(goal_points, n_clusters=256, seed=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    kmeans.fit(goal_points)
    vocabulary = kmeans.cluster_centers_
    return vocabulary
```

**测试**: 实现后，可以单独测试这个函数：
```python
import numpy as np
from data.nuscenes_preprocessor import build_vocabulary

# 测试数据
goals = np.random.randn(1000, 2) * 10
vocab = build_vocabulary(goals, n_clusters=256)
print(f"词汇表形状: {vocab.shape}")  # 应该是 (256, 2)
```

---

##### 2.2 再实现 `extract_agent_trajectories()` （中等难度）

**位置**: `data/nuscenes_preprocessor.py:40`

**任务**: 从 nuScenes 场景中提取车辆轨迹

**实现步骤**:

1. 获取场景信息
```python
scene = nusc.get('scene', scene_token)
sample_token = scene['first_sample_token']
```

2. 遍历所有帧
```python
while sample_token:
    sample = nusc.get('sample', sample_token)
    
    # 获取 ego 位姿
    ego_pose_token = sample['data']['LIDAR_TOP']
    sample_data = nusc.get('sample_data', ego_pose_token)
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    
    # 处理该帧的标注...
    
    sample_token = sample['next']
```

3. 提取轨迹
```python
for ann_token in sample['anns']:
    ann = nusc.get('sample_annotation', ann_token)
    
    # 检查类别
    if ann['category_name'] not in vehicle_categories:
        continue
    
    # 提取历史轨迹（向前查找）
    history_traj = []
    current_ann = ann
    for i in range(history_frames):
        if current_ann['prev'] == '':
            break
        current_ann = nusc.get('sample_annotation', current_ann['prev'])
        position = current_ann['translation'][:2]
        history_traj.append(position)
    
    # 提取未来轨迹（向后查找）
    future_traj = []
    current_ann = ann
    for i in range(future_frames):
        if current_ann['next'] == '':
            break
        current_ann = nusc.get('sample_annotation', current_ann['next'])
        position = current_ann['translation'][:2]
        future_traj.append(position)
    
    # 检查轨迹完整性
    if len(history_traj) == history_frames and len(future_traj) == future_frames:
        # 转换到 ego 坐标系
        from data import nuscenes_utils
        ego_translation = np.array(ego_pose['translation'])
        ego_rotation = Quaternion(ego_pose['rotation'])
        
        history_ego = nuscenes_utils.global_to_ego(
            np.array(history_traj), ego_translation, ego_rotation
        )
        future_ego = nuscenes_utils.global_to_ego(
            np.array(future_traj), ego_translation, ego_rotation
        )
        
        # 保存轨迹
        all_history.append(history_ego)
        all_future.append(future_ego)
        all_goals.append(future_ego[-1])  # 终点作为 goal
```

**测试**: 实现后测试：
```python
from nuscenes.nuscenes import NuScenes
from data.nuscenes_preprocessor import extract_agent_trajectories

nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes')
scene_token = nusc.scene[0]['token']

trajs = extract_agent_trajectories(nusc, scene_token)
print(f"提取了 {len(trajs['goals'])} 条轨迹")
print(f"History shape: {trajs['history'].shape}")
print(f"Future shape: {trajs['future'].shape}")
```

---

##### 2.3 最后实现 `rasterize_map()` （最复杂）

**位置**: `data/nuscenes_preprocessor.py:207`

**任务**: 将 HD 地图栅格化为 BEV 图像

**实现步骤**:

1. 创建空白图像
```python
bev_image = np.zeros((map_size[0], map_size[1], 3), dtype=np.uint8)
```

2. 获取 ego 位置和地图元素
```python
ego_x = ego_pose['translation'][0]
ego_y = ego_pose['translation'][1]
ego_translation = np.array(ego_pose['translation'])
ego_rotation = Quaternion(ego_pose['rotation'])

# 获取附近的车道线
lane_records = nusc_map.get_records_in_radius(
    ego_x, ego_y, map_range, ['lane', 'lane_connector']
)
```

3. 处理车道线（Channel 0）
```python
import cv2
from data import nuscenes_utils

for record_token in lane_records['lane']:
    lane_record = nusc_map.get('lane', record_token)
    
    # 获取车道线多边形
    polygon_token = lane_record['polygon_token']
    polygon = nusc_map.extract_polygon(polygon_token)
    exterior_coords = np.array(polygon.exterior.coords)[:, :2]  # 只取 x, y
    
    # 转换到 ego 坐标系
    coords_ego = nuscenes_utils.global_to_ego(
        exterior_coords, ego_translation, ego_rotation
    )
    
    # 转换到像素坐标
    coords_pixel = nuscenes_utils.ego_to_pixel(
        coords_ego, map_range, map_size
    )
    
    # 绘制到图像
    coords_pixel = coords_pixel.astype(np.int32)
    cv2.polylines(
        bev_image[:, :, 0],  # Channel 0: 车道线
        [coords_pixel],
        isClosed=True,
        color=255,
        thickness=2
    )
```

4. 类似地处理道路边界（Channel 1）和人行道（Channel 2）

5. 归一化和转换格式
```python
# 归一化到 [0, 1]
bev_image = bev_image.astype(np.float32) / 255.0

# 转换为 (C, H, W) 格式
bev_image = bev_image.transpose(2, 0, 1)

return bev_image
```

**测试**: 实现后测试：
```python
from nuscenes.map_expansion.map_api import NuScenesMap
from data.nuscenes_preprocessor import rasterize_map

nusc_map = NuScenesMap(dataroot='data/nuscenes', map_name='boston-seaport')
ego_pose = {'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0]}

bev = rasterize_map(nusc_map, ego_pose)
print(f"BEV shape: {bev.shape}")  # 应该是 (3, 200, 200)

# 可视化
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(bev[i], cmap='gray')
    plt.title(['Lane', 'Road', 'Walkway'][i])
plt.savefig('bev_test.png')
```

---

### 第 3 步：运行预处理

实现完 3 个核心函数后，运行预处理脚本：

```bash
python scripts/preprocess_nuscenes.py
```

预期输出：
```
✅ 预处理完成！
数据已保存到: data/nuscenes_processed
```

生成的文件：
```
data/nuscenes_processed/
├── train/
│   ├── history.npy
│   ├── future.npy
│   ├── goals.npy
│   ├── bev_features.npy
│   ├── vocabulary.npy
│   └── metadata.pkl
├── val/
│   └── (同上)
└── test/
    └── (同上)
```

---

### 第 4 步：测试数据

```bash
python data/test_nuscenes_data.py
```

预期输出：
```
✅ 所有测试通过！数据准备就绪。
```

---

### 第 5 步：训练模型

#### 5.1 训练 GoalPointScorer

```bash
python train_scorer_nuscenes.py
```

预期结果：
- Top-1 准确率: 30-50%
- Top-5 准确率: 60-80%
- 训练时间: ~2-3 小时（mini 数据集）

#### 5.2 训练 GoalFlowMatcher

```bash
python train_matcher_nuscenes.py
```

（此脚本待创建，类似 train_scorer_nuscenes.py）

---

## 🐛 常见问题

### Q1: 导入 nuscenes 时报错

**错误**: `ModuleNotFoundError: No module named 'nuscenes'`

**解决**:
```bash
pip install nuscenes-devkit
```

### Q2: 数据目录不存在

**错误**: `❌ 数据目录不存在: data/nuscenes`

**解决**: 请按照第 1 步下载并解压数据

### Q3: GPU 内存不足

**错误**: `RuntimeError: CUDA out of memory`

**解决**: 修改 `config/nuscenes_config.py`:
```python
scorer_batch_size = 4  # 从 16 改为 4
bev_height = 128  # 从 200 改为 128
bev_width = 128
```

### Q4: 预处理很慢

**原因**: 地图栅格化比较耗时

**解决**: 
- 减少 BEV 分辨率
- 使用多进程（需要修改代码）
- 先用少量场景测试

---

## 📊 预期性能指标

### nuScenes mini 数据集

| 指标 | 预期值 |
|------|--------|
| 训练样本数 | ~1500-2000 |
| Scorer Top-1 Acc | 30-50% |
| Scorer Top-5 Acc | 60-80% |
| Matcher ADE (with gt_goal) | < 2.0m |
| Matcher FDE (with gt_goal) | < 3.0m |
| 端到端 ADE | 2.0-4.0m |
| 端到端 FDE | 3.0-6.0m |

---

## 📝 实施检查清单

- [ ] 安装所有依赖包
- [ ] 下载 nuScenes mini 数据集
- [ ] 运行环境验证脚本
- [ ] 实现 `build_vocabulary()`
- [ ] 实现 `extract_agent_trajectories()`
- [ ] 实现 `rasterize_map()`
- [ ] 运行预处理脚本
- [ ] 运行数据测试脚本
- [ ] 训练 GoalPointScorer
- [ ] 训练 GoalFlowMatcher
- [ ] 端到端推理和评估

---

## 💡 实施建议

1. **逐步实现**: 先实现最简单的 `build_vocabulary()`，再实现其他函数
2. **及时测试**: 每实现一个函数，立即测试，不要等到全部完成
3. **参考文档**: 查看 nuScenes 官方文档和示例代码
4. **可视化调试**: 使用 matplotlib 可视化中间结果
5. **寻求帮助**: 遇到问题随时询问

---

## 📚 参考资源

- **nuScenes 官网**: https://www.nuscenes.org/
- **nuScenes devkit**: https://github.com/nutonomy/nuscenes-devkit
- **nuScenes 教程**: https://www.nuscenes.org/nuscenes#tutorials
- **项目文档**: 
  - `NUSCENES_SETUP.md` - 环境配置指南
  - `GOALFLOW_NUSCENES_PLAN.md` - 整体实施计划

---

## 🎉 完成后

恭喜！你已经成功将 GoalFlow 模型适配到 nuScenes 数据集。

下一步可以：
1. 使用完整的 nuScenes trainval 数据集（~350GB）
2. 调优超参数提升性能
3. 添加数据增强
4. 尝试其他轨迹预测数据集（Argoverse, Waymo 等）

祝你实现顺利！🚀
