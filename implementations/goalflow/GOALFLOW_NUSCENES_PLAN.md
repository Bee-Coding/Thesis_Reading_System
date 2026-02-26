# GoalFlow + nuScenes 实现计划

**目标**: 使用 nuScenes mini 数据集训练和验证 GoalFlow 轨迹预测模型

**日期**: 2026-02-26  
**预计完成时间**: 2-4 天  
**实现方式**: 引导式实现（框架 + 关键部分填充）

---

## 📋 目录

1. [总体架构](#总体架构)
2. [数据集信息](#数据集信息)
3. [实施阶段](#实施阶段)
4. [文件结构](#文件结构)
5. [关键实现点](#关键实现点)
6. [评估指标](#评估指标)
7. [时间线](#时间线)

---

## 总体架构

```
nuScenes 数据
    ↓
数据预处理
    ├── 轨迹提取
    ├── HD 地图栅格化 (BEV)
    └── Goal 词汇表构建
    ↓
训练数据集
    ├── Trajectories: (N, 12, 2)  # 6秒 @ 2Hz
    ├── Goals: (N, 2)
    ├── BEV Features: (N, 3, 200, 200)
    └── Vocabulary: (256, 2)
    ↓
模型训练
    ├── GoalPointScorer (Stage 1)
    └── GoalFlowMatcher (Stage 2)
    ↓
端到端推理
    └── 评估 + 可视化
```

---

## 数据集信息

### nuScenes Mini

- **大小**: ~4GB
- **场景数**: 10 个场景
- **帧数**: ~400 帧
- **采样率**: 2Hz (每 0.5 秒一帧)
- **传感器**:
  - 6 个相机 (前、后、左前、右前、左后、右后)
  - 1 个 LiDAR
  - 5 个 Radar
  - GPS/IMU
- **标注**:
  - 3D bounding boxes (23 类物体)
  - 轨迹跟踪 ID
  - HD 地图 (车道线、道路边界、人行道等)

### 数据下载

```bash
# 1. 注册账号
https://www.nuscenes.org/nuscenes

# 2. 下载 nuScenes mini
wget https://www.nuscenes.org/data/v1.0-mini.tgz

# 3. 解压
tar -xzf v1.0-mini.tgz -C data/nuscenes/
```

**目录结构**:
```
data/nuscenes/
├── maps/                    # HD 地图
├── samples/                 # 关键帧数据
├── sweeps/                  # 中间帧数据
└── v1.0-mini/              # 标注文件
    ├── attribute.json
    ├── calibrated_sensor.json
    ├── category.json
    ├── ego_pose.json
    ├── instance.json
    ├── log.json
    ├── map.json
    ├── sample.json
    ├── sample_annotation.json
    ├── sample_data.json
    ├── scene.json
    └── sensor.json
```

---

## 实施阶段

### 阶段 0: 环境准备 (1-2 小时)

**任务**:
- [x] 下载 nuScenes mini 数据集
- [x] 安装 nuscenes-devkit
- [x] 验证数据完整性
- [x] 运行官方示例代码

**验证**:
```bash
cd implementations/goalflow
python scripts/test_nuscenes_installation.py
```

---

### 阶段 1: 数据预处理 (6-9 小时)

#### 1.1 轨迹提取 (2-3 小时)

**目标**: 从 nuScenes 标注中提取 agent 轨迹

**输入**:
- nuScenes 场景数据
- 标注文件 (sample_annotation.json)

**输出**:
- 历史轨迹: (N, 4, 2) - 2秒历史 @ 2Hz
- 未来轨迹: (N, 12, 2) - 6秒未来 @ 2Hz
- Goal 点: (N, 2) - 未来轨迹的终点

**关键函数** (需要你实现):
```python
def extract_agent_trajectories(
    nusc: NuScenes,
    scene_token: str,
    history_frames: int = 4,
    future_frames: int = 12
) -> Dict[str, np.ndarray]:
    """
    从 nuScenes 场景中提取 agent 轨迹
    
    TODO: 实现以下步骤
    1. 遍历场景中的所有帧
    2. 对每个 agent (车辆):
       - 提取历史轨迹 (4 帧)
       - 提取未来轨迹 (12 帧)
       - 转换到 ego 车坐标系
    3. 过滤无效轨迹 (轨迹不完整、静止等)
    4. 返回轨迹数据
    
    提示:
    - 使用 nusc.get('sample_annotation', token) 获取标注
    - 使用 nusc.get('ego_pose', token) 获取 ego 车位姿
    - 坐标转换: global -> ego
    """
    pass
```

**实现提示**:
- 参考 `nuscenes_utils.py` 中的坐标转换函数
- 过滤条件:
  - 轨迹长度 >= history_frames + future_frames
  - 移动距离 > 1.0m (过滤静止物体)
  - 在 ego 车前方 50m 范围内
- 坐标系: ego 车为原点，前方为 x 正方向，左侧为 y 正方向

#### 1.2 HD 地图栅格化 (3-4 小时)

**目标**: 将 HD 地图转换为 BEV 栅格图

**输入**:
- nuScenes 地图数据
- Ego 车位姿

**输出**:
- BEV 特征: (3, 200, 200)
  - Channel 0: 车道线 (lane)
  - Channel 1: 道路边界 (road boundary)
  - Channel 2: 人行道 (walkway)

**关键函数** (需要你实现):
```python
def rasterize_map(
    nusc_map: NuScenesMap,
    ego_pose: Dict,
    map_size: Tuple[int, int] = (200, 200),
    map_range: float = 50.0
) -> np.ndarray:
    """
    将 HD 地图栅格化为 BEV 图像
    
    TODO: 实现以下步骤
    1. 获取 ego 车周围的地图元素
       - 车道线 (lane)
       - 道路边界 (road_segment)
       - 人行道 (walkway)
    2. 将地图元素转换到 ego 车坐标系
    3. 栅格化到 BEV 图像
       - 分辨率: map_range * 2 / map_size = 0.5m/pixel
       - 范围: [-50m, 50m] x [-50m, 50m]
    4. 返回 (3, H, W) 的 BEV 特征
    
    提示:
    - 使用 nusc_map.get_records_in_radius() 获取附近地图元素
    - 使用 cv2.polylines() 绘制线段
    - 使用 cv2.fillPoly() 填充区域
    """
    pass
```

**实现提示**:
- 参考 nuScenes devkit 的 `render_map()` 函数
- 栅格化步骤:
  1. 创建空白图像 (200, 200, 3)
  2. 遍历地图元素，绘制到对应通道
  3. 归一化到 [0, 1]
- 坐标转换: global -> ego -> pixel

#### 1.3 Goal 词汇表构建 (1-2 小时)

**目标**: 从训练集构建 goal 点词汇表

**输入**:
- 所有训练轨迹的终点

**输出**:
- Vocabulary: (256, 2) - K-means 聚类中心

**关键函数** (需要你实现):
```python
def build_vocabulary(
    goal_points: np.ndarray,
    n_clusters: int = 256,
    seed: int = 42
) -> np.ndarray:
    """
    使用 K-means 构建 goal 词汇表
    
    TODO: 实现以下步骤
    1. 收集所有训练轨迹的终点
    2. 使用 K-means 聚类
    3. 返回聚类中心作为词汇表
    
    提示:
    - 使用 sklearn.cluster.KMeans
    - 设置 random_state=seed 保证可复现
    """
    pass
```

**实现提示**:
- 参考 `generate_toy_data.py` 中的 `build_vocabulary()` 函数
- 词汇表大小: 256 (可调整)
- 保存为 `data/nuscenes/vocabulary.npy`

---

### 阶段 2: 模型适配 (2-3 小时)

#### 2.1 创建 nuScenes 数据集类

**文件**: `data/nuscenes_dataset.py`

**已实现**:
- PyTorch Dataset 基本结构
- 数据加载和缓存
- Batch collation

**需要你验证**:
- 数据形状是否正确
- 坐标系是否一致
- 数据范围是否合理

#### 2.2 修改配置文件

**文件**: `config/nuscenes_config.py`

**已实现**:
- 所有超参数配置
- 路径配置
- 训练参数

**需要你调整**:
- `batch_size`: 根据 GPU 内存调整
- `num_workers`: 根据 CPU 核心数调整
- `data_root`: nuScenes 数据路径

---

### 阶段 3: 训练 (6-8 小时)

#### 3.1 训练 GoalPointScorer (2-3 小时)

**命令**:
```bash
cd implementations/goalflow
python train_scorer_nuscenes.py
```

**预期结果**:
- Top-1 准确率: 30-50%
- Top-5 准确率: 60-80%
- 训练时间: ~2-3 小时 (mini 数据集)

**监控指标**:
- Loss 曲线
- Top-1/Top-5 准确率
- Goal Error (m)

**调试技巧**:
- 如果 loss 不下降: 检查学习率、数据归一化
- 如果准确率很低: 检查 BEV 特征、词汇表覆盖
- 如果过拟合: 增加 dropout、减少模型容量

#### 3.2 训练 GoalFlowMatcher (3-4 小时)

**命令**:
```bash
python train_matcher_nuscenes.py
```

**预期结果**:
- ADE (with gt_goal): < 2.0m
- FDE (with gt_goal): < 3.0m
- 训练时间: ~3-4 小时

**监控指标**:
- ADE/FDE 曲线
- Loss 曲线

**调试技巧**:
- 如果 ADE 很高: 增加 ODE 步数、调整噪声
- 如果训练慢: 减少 ODE 步数、减少候选数量

---

### 阶段 4: 评估和可视化 (2-3 小时)

#### 4.1 端到端推理

**命令**:
```bash
python inference_nuscenes.py
```

**评估指标**:
- **ADE** (Average Displacement Error): 平均位移误差
- **FDE** (Final Displacement Error): 终点位移误差
- **Miss Rate @ 2m**: 终点误差 > 2m 的比例
- **DAC** (Drivable Area Compliance): 可行驶区域符合率

**预期结果** (nuScenes mini):
- ADE: 2.0-4.0m
- FDE: 3.0-6.0m
- Miss Rate @ 2m: 30-50%

#### 4.2 可视化

**命令**:
```bash
python scripts/visualize_nuscenes.py --sample_idx 0
```

**可视化内容**:
- BEV 地图 (HD map)
- 历史轨迹 (蓝色)
- 真实未来轨迹 (绿色)
- 预测轨迹 (红色)
- 候选轨迹 (灰色)
- Goal 点 (星形标记)

---

### 阶段 5: 调优 (可选, 4-6 小时)

#### 5.1 超参数调优

**Scorer 调优**:
- `lambda_dis`: 距离损失权重 (默认 1.0)
- `lambda_dac`: DAC 损失权重 (默认 0.005)
- `hidden_dim`: 隐藏层维度 (默认 256)
- `num_layers`: Transformer 层数 (默认 4)

**Matcher 调优**:
- `num_steps`: ODE 步数 (默认 10)
- `noise_std`: 初始噪声标准差 (默认 1.0)
- `hidden_dim`: 隐藏层维度 (默认 256)

#### 5.2 数据增强

**可选增强**:
- 旋转: ±30°
- 平移: ±5m
- 缩放: 0.9-1.1x
- 翻转: 左右翻转

#### 5.3 模型改进

**可选改进**:
- 增加 BEV 特征通道 (LiDAR, 动态物体)
- 使用更大的词汇表 (512, 1024)
- 增加网络容量
- 使用预训练模型

---

## 文件结构

```
implementations/goalflow/
├── GOALFLOW_NUSCENES_PLAN.md       # 本文档
├── NUSCENES_SETUP.md               # 环境配置指南
│
├── config/
│   ├── scorer_config.py            # Scorer 配置 (toy data)
│   ├── matcher_config.py           # Matcher 配置 (toy data)
│   └── nuscenes_config.py          # ✅ nuScenes 配置 (新)
│
├── data/
│   ├── toy_goalflow_dataset.py     # Toy 数据集
│   ├── nuscenes_dataset.py         # ✅ nuScenes 数据集 (新)
│   ├── nuscenes_preprocessor.py    # ⚠️ 数据预处理 (需要你实现)
│   ├── nuscenes_utils.py           # ✅ 工具函数 (新)
│   └── test_nuscenes_data.py       # ✅ 测试脚本 (新)
│
├── scripts/
│   ├── download_nuscenes.sh        # ✅ 下载脚本 (新)
│   ├── preprocess_nuscenes.py      # ✅ 预处理脚本 (新)
│   ├── visualize_nuscenes.py       # ✅ 可视化脚本 (新)
│   └── test_nuscenes_installation.py # ✅ 安装测试 (新)
│
├── train_scorer_nuscenes.py        # ⚠️ Scorer 训练 (基于现有代码)
├── train_matcher_nuscenes.py       # ⚠️ Matcher 训练 (基于现有代码)
├── inference_nuscenes.py           # ⚠️ 推理脚本 (基于现有代码)
│
└── models/                         # 模型代码 (无需修改)
    ├── goal_point_scorer.py
    ├── goal_flow_matcher.py
    └── trajectory_selector.py
```

**图例**:
- ✅ 完整实现 (我提供)
- ⚠️ 框架 + 关键部分 (你填充)
- 无标记: 已存在，无需修改

---

## 关键实现点

### 你需要实现的函数

#### 1. `extract_agent_trajectories()` (nuscenes_preprocessor.py)

**难度**: ⭐⭐⭐  
**预计时间**: 2-3 小时  
**关键点**:
- nuScenes API 使用
- 坐标系转换
- 轨迹过滤

**提示**:
```python
# 获取场景中的所有帧
scene = nusc.get('scene', scene_token)
sample_token = scene['first_sample_token']

while sample_token:
    sample = nusc.get('sample', sample_token)
    
    # 获取该帧的所有标注
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        
        # 提取轨迹...
        
    sample_token = sample['next']
```

#### 2. `rasterize_map()` (nuscenes_preprocessor.py)

**难度**: ⭐⭐⭐⭐  
**预计时间**: 3-4 小时  
**关键点**:
- HD 地图 API 使用
- 栅格化算法
- 坐标转换

**提示**:
```python
# 获取附近的地图元素
lane_records = nusc_map.get_records_in_radius(
    ego_pose['translation'][0],
    ego_pose['translation'][1],
    map_range,
    ['lane', 'lane_connector']
)

# 栅格化
for record in lane_records:
    polygon = nusc_map.extract_polygon(record)
    # 转换到 ego 坐标系
    # 绘制到 BEV 图像
```

#### 3. `build_vocabulary()` (nuscenes_preprocessor.py)

**难度**: ⭐  
**预计时间**: 1 小时  
**关键点**:
- K-means 聚类

**提示**:
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
kmeans.fit(goal_points)
vocabulary = kmeans.cluster_centers_
```

---

## 评估指标

### 主要指标

1. **ADE (Average Displacement Error)**
   ```python
   ade = np.mean(np.linalg.norm(pred_traj - gt_traj, axis=-1))
   ```
   - 预测轨迹与真实轨迹的平均距离
   - 越小越好
   - nuScenes 典型值: 1.5-3.0m

2. **FDE (Final Displacement Error)**
   ```python
   fde = np.linalg.norm(pred_traj[-1] - gt_traj[-1])
   ```
   - 预测终点与真实终点的距离
   - 越小越好
   - nuScenes 典型值: 2.0-5.0m

3. **Miss Rate @ 2m**
   ```python
   miss_rate = (fde > 2.0).mean()
   ```
   - 终点误差 > 2m 的比例
   - 越小越好
   - nuScenes 典型值: 20-40%

### 辅助指标

4. **Goal Error**
   ```python
   goal_error = np.linalg.norm(selected_goal - gt_goal)
   ```
   - 选中的 goal 与真实 goal 的距离
   - 评估 Scorer 性能

5. **Top-K Accuracy**
   ```python
   top_k_acc = (gt_goal_idx in top_k_indices).mean()
   ```
   - 真实 goal 是否在 Top-K 预测中
   - 评估 Scorer 性能

6. **DAC (Drivable Area Compliance)**
   ```python
   dac = (pred_traj_in_drivable_area).mean()
   ```
   - 预测轨迹在可行驶区域内的比例
   - 越高越好

---

## 时间线

### 第 1 天: 环境准备 + 数据预处理

| 时间 | 任务 | 预计时长 |
|------|------|----------|
| 09:00-10:00 | 下载 nuScenes mini | 1h |
| 10:00-11:00 | 安装环境、验证数据 | 1h |
| 11:00-13:00 | 实现 `extract_agent_trajectories()` | 2h |
| 14:00-17:00 | 实现 `rasterize_map()` | 3h |
| 17:00-18:00 | 实现 `build_vocabulary()` | 1h |
| 18:00-19:00 | 运行预处理、验证数据 | 1h |

**里程碑**: 完成数据预处理，生成训练数据

---

### 第 2 天: 模型训练

| 时间 | 任务 | 预计时长 |
|------|------|----------|
| 09:00-10:00 | 验证数据集、配置训练参数 | 1h |
| 10:00-13:00 | 训练 GoalPointScorer | 3h |
| 14:00-17:00 | 训练 GoalFlowMatcher | 3h |
| 17:00-18:00 | 检查训练结果、保存模型 | 1h |

**里程碑**: 完成两个模型的训练

---

### 第 3 天: 评估和调优

| 时间 | 任务 | 预计时长 |
|------|------|----------|
| 09:00-11:00 | 端到端推理、计算指标 | 2h |
| 11:00-13:00 | 可视化结果、分析问题 | 2h |
| 14:00-17:00 | 超参数调优 | 3h |
| 17:00-18:00 | 最终评估、生成报告 | 1h |

**里程碑**: 完成评估和调优，生成最终报告

---

### 第 4 天: 文档和总结 (可选)

| 时间 | 任务 | 预计时长 |
|------|------|----------|
| 09:00-11:00 | 整理代码、添加注释 | 2h |
| 11:00-13:00 | 编写实验报告 | 2h |
| 14:00-16:00 | 准备演示材料 | 2h |

**里程碑**: 完成项目文档和总结

---

## 常见问题

### Q1: nuScenes mini 数据集够用吗？

**A**: 对于快速验证 GoalFlow 流程，mini 数据集足够。但如果要获得好的性能指标，建议使用完整的 nuScenes trainval 数据集 (~350GB)。

### Q2: 我的 GPU 内存不够怎么办？

**A**: 
- 减小 `batch_size` (默认 16 → 8 或 4)
- 减小 BEV 分辨率 (200x200 → 128x128)
- 使用 CPU 训练 (会很慢)

### Q3: 训练时间太长怎么办？

**A**:
- 减少训练 epoch (50 → 20)
- 减少 ODE 步数 (10 → 5)
- 使用更小的模型 (hidden_dim 256 → 128)

### Q4: 如何判断模型训练是否成功？

**A**:
- **Scorer**: Top-1 准确率 > 30%
- **Matcher**: ADE (with gt_goal) < 2.0m
- **整体**: ADE < 4.0m, FDE < 6.0m

### Q5: 如果性能不好怎么办？

**A**:
1. 先检查数据: 可视化 BEV 特征、轨迹
2. 再检查模型: 运行诊断脚本
3. 最后调优: 调整超参数

---

## 参考资源

### nuScenes 官方资源

- **官网**: https://www.nuscenes.org/
- **论文**: https://arxiv.org/abs/1903.11027
- **GitHub**: https://github.com/nutonomy/nuscenes-devkit
- **教程**: https://www.nuscenes.org/nuscenes

### GoalFlow 相关

- **论文**: (请补充 GoalFlow 论文链接)
- **相关工作**:
  - MultiPath++: https://arxiv.org/abs/2111.14973
  - QCNet: https://arxiv.org/abs/2306.15195
  - PGP: https://arxiv.org/abs/2103.09122

### 代码参考

- **nuScenes 轨迹预测**: https://github.com/nutonomy/nuscenes-devkit/tree/master/python-sdk/nuscenes/prediction
- **BEV 地图渲染**: https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/map_expansion/map_api.py

---

## 下一步

1. ✅ 阅读 `NUSCENES_SETUP.md` 配置环境
2. ⏳ 下载 nuScenes mini 数据集
3. ⏳ 运行 `test_nuscenes_installation.py` 验证安装
4. ⏳ 开始实现 `nuscenes_preprocessor.py` 中的关键函数
5. ⏳ 运行预处理脚本生成训练数据
6. ⏳ 训练模型
7. ⏳ 评估和可视化

---

**祝你实现顺利！如有问题，随时查看文档或寻求帮助。** 🚀
