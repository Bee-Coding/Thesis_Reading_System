# nuScenes 环境配置指南

本文档指导你完成 nuScenes 数据集的下载、安装和验证。

---

## 📋 目录

1. [系统要求](#系统要求)
2. [安装步骤](#安装步骤)
3. [数据下载](#数据下载)
4. [验证安装](#验证安装)
5. [常见问题](#常见问题)

---

## 系统要求

### 硬件要求

- **存储空间**: 
  - nuScenes mini: ~10GB (数据 4GB + 预处理缓存 6GB)
  - nuScenes trainval (可选): ~400GB
- **内存**: 
  - 最低 8GB RAM
  - 推荐 16GB+ RAM
- **GPU** (可选):
  - 训练推荐: NVIDIA GPU with 6GB+ VRAM
  - CPU 也可以训练，但会很慢

### 软件要求

- **操作系统**: Linux / macOS / Windows (推荐 Linux)
- **Python**: 3.8+
- **PyTorch**: 1.10+
- **CUDA** (可选): 11.0+ (如果使用 GPU)

---

## 安装步骤

### Step 1: 创建 Python 环境

推荐使用 conda 创建独立环境：

```bash
# 创建新环境
conda create -n goalflow python=3.10
conda activate goalflow

# 或使用现有环境
conda activate your_env
```

### Step 2: 安装 PyTorch

根据你的 CUDA 版本安装 PyTorch：

```bash
# CUDA 11.8 (推荐)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

验证安装：
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 3: 安装 nuScenes devkit

```bash
# 安装 nuscenes-devkit
pip install nuscenes-devkit

# 验证安装
python -c "from nuscenes.nuscenes import NuScenes; print('nuScenes devkit installed successfully!')"
```

### Step 4: 安装其他依赖

```bash
cd implementations/goalflow

# 安装项目依赖
pip install -r requirements.txt
```

如果没有 `requirements.txt`，手动安装：

```bash
pip install numpy scipy scikit-learn tqdm matplotlib opencv-python pillow
```

---

## 数据下载

### 方式 1: 官网下载 (推荐)

#### 1.1 注册账号

访问 https://www.nuscenes.org/nuscenes 并注册账号（免费）

#### 1.2 下载 nuScenes mini

登录后，进入 [Download 页面](https://www.nuscenes.org/download)

下载以下文件：
- **v1.0-mini.tgz** (~4GB) - 包含 10 个场景的数据
- **nuScenes-map-expansion-v1.3.zip** (~500MB) - HD 地图数据

#### 1.3 解压数据

```bash
# 创建数据目录
mkdir -p data/nuscenes

# 解压 mini 数据集
tar -xzf v1.0-mini.tgz -C data/nuscenes/

# 解压地图数据
unzip nuScenes-map-expansion-v1.3.zip -d data/nuscenes/
```

**最终目录结构**:
```
data/nuscenes/
├── maps/                           # HD 地图
│   ├── basemap/
│   ├── expansion/
│   ├── prediction/
│   ├── 36092f0b03a857c6a3403e25b4b7aab3.png
│   ├── 37819e65e09e5547b8a3ceaefba56bb2.png
│   ├── 53992ee3023e5494b90c316c183be829.png
│   └── 93406b464a165eaba6d9de76ca09f5da.png
├── samples/                        # 关键帧数据
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   ├── CAM_BACK_RIGHT/
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   ├── LIDAR_TOP/
│   ├── RADAR_BACK_LEFT/
│   ├── RADAR_BACK_RIGHT/
│   ├── RADAR_FRONT/
│   ├── RADAR_FRONT_LEFT/
│   └── RADAR_FRONT_RIGHT/
├── sweeps/                         # 中间帧数据
│   └── (同上)
└── v1.0-mini/                      # 标注文件
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
    ├── sensor.json
    └── visibility.json
```

### 方式 2: 使用脚本下载 (备选)

如果官网下载慢，可以使用我们提供的脚本：

```bash
cd implementations/goalflow
bash scripts/download_nuscenes.sh
```

**注意**: 脚本会自动下载并解压数据到 `data/nuscenes/` 目录。

---

## 验证安装

### Step 1: 测试 nuScenes devkit

运行官方示例：

```bash
cd implementations/goalflow
python scripts/test_nuscenes_installation.py
```

**预期输出**:
```
✅ nuScenes devkit installed successfully!
✅ Data path exists: data/nuscenes
✅ Loading nuScenes mini dataset...
✅ Loaded 10 scenes, 404 samples
✅ Scene 0: scene-0061
   - Description: Parked truck, construction, intersectio...
   - Location: boston-seaport
   - # Samples: 39
✅ All checks passed!
```

### Step 2: 可视化样本数据

```bash
python scripts/visualize_nuscenes.py --scene_idx 0 --sample_idx 0
```

这会生成一张可视化图片，包含：
- 6 个相机视角
- BEV 地图
- 3D bounding boxes

### Step 3: 测试数据预处理

```bash
python data/test_nuscenes_data.py
```

**预期输出**:
```
Testing nuScenes data extraction...
✅ Extracted 150 trajectories from scene 0
✅ Trajectory shape: (150, 12, 2)
✅ Goal shape: (150, 2)
✅ BEV feature shape: (150, 3, 200, 200)
✅ All tests passed!
```

---

## 常见问题

### Q1: 下载速度很慢怎么办？

**A**: 
1. 使用代理或 VPN
2. 使用国内镜像（如果有）
3. 分段下载，使用断点续传工具（如 `wget -c`）

### Q2: 解压时提示空间不足

**A**:
1. 检查磁盘空间: `df -h`
2. 清理不必要的文件
3. 使用外接硬盘

### Q3: 导入 nuscenes 时报错

**错误信息**:
```
ModuleNotFoundError: No module named 'nuscenes'
```

**解决方案**:
```bash
# 确认环境激活
conda activate goalflow

# 重新安装
pip install nuscenes-devkit

# 验证
python -c "from nuscenes.nuscenes import NuScenes"
```

### Q4: 加载数据时报错 "Data path not found"

**错误信息**:
```
AssertionError: Database version not found: data/nuscenes/v1.0-mini
```

**解决方案**:
1. 检查数据路径是否正确
2. 确认 `v1.0-mini` 目录存在
3. 检查目录权限

```bash
# 检查目录结构
ls -la data/nuscenes/
ls -la data/nuscenes/v1.0-mini/

# 如果路径不对，修改配置文件
# config/nuscenes_config.py
data_root = '/your/correct/path/to/nuscenes'
```

### Q5: GPU 内存不足

**错误信息**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:
1. 减小 batch_size
   ```python
   # config/nuscenes_config.py
   batch_size = 8  # 改为 4 或 2
   ```

2. 减小 BEV 分辨率
   ```python
   bev_height = 128  # 从 200 改为 128
   bev_width = 128
   ```

3. 使用 CPU 训练
   ```python
   device = 'cpu'
   ```

### Q6: 地图加载失败

**错误信息**:
```
AssertionError: Map not found: boston-seaport
```

**解决方案**:
1. 确认地图文件已下载
   ```bash
   ls data/nuscenes/maps/
   ```

2. 重新下载地图数据
   ```bash
   # 下载 nuScenes-map-expansion-v1.3.zip
   # 解压到 data/nuscenes/maps/
   ```

### Q7: OpenCV 相关错误

**错误信息**:
```
ImportError: libGL.so.1: cannot open shared object file
```

**解决方案** (Linux):
```bash
# Ubuntu/Debian
sudo apt-get install libgl1-mesa-glx

# CentOS/RHEL
sudo yum install mesa-libGL
```

---

## 数据集统计信息

### nuScenes mini

- **场景数**: 10
- **样本数**: ~400
- **轨迹数**: ~2000 (预处理后)
- **地图**: 4 个 (Boston Seaport, Singapore)
- **物体类别**: 23 类
  - 车辆: car, truck, bus, trailer, construction_vehicle, ...
  - 行人: pedestrian, ...
  - 其他: bicycle, motorcycle, traffic_cone, barrier, ...

### 场景分布

| 场景 ID | 描述 | 位置 | 样本数 |
|---------|------|------|--------|
| scene-0061 | Parked truck, construction | Boston | 39 |
| scene-0103 | Many peds right, wait for turning car | Boston | 39 |
| scene-0655 | Parking lot, parked cars | Boston | 39 |
| scene-0757 | Highway, many cars | Boston | 40 |
| scene-0796 | Rainy, highway | Boston | 40 |
| scene-0916 | Intersection, many cars | Boston | 40 |
| scene-1077 | Highway, construction | Singapore | 40 |
| scene-1094 | Intersection, traffic light | Singapore | 40 |
| scene-1100 | Highway, many cars | Singapore | 40 |

---

## 下一步

环境配置完成后，请继续：

1. ✅ 阅读 `GOALFLOW_NUSCENES_PLAN.md` 了解整体计划
2. ⏳ 开始实现数据预处理代码
3. ⏳ 运行预处理脚本生成训练数据
4. ⏳ 训练模型

---

## 参考资源

- **nuScenes 官网**: https://www.nuscenes.org/
- **nuScenes devkit**: https://github.com/nutonomy/nuscenes-devkit
- **nuScenes 论文**: https://arxiv.org/abs/1903.11027
- **nuScenes 教程**: https://www.nuscenes.org/nuscenes#tutorials

---

**如有问题，请查看常见问题部分或寻求帮助。** 🚀
