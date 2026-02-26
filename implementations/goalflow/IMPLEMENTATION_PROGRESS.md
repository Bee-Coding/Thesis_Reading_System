# GoalFlow + nuScenes 实现进度

**更新时间**: 2026-02-26  
**状态**: 框架搭建完成，等待用户实现关键函数

---

## ✅ 已完成的文件

### 1. 文档

- ✅ `GOALFLOW_NUSCENES_PLAN.md` - 完整实现计划
- ✅ `NUSCENES_SETUP.md` - 环境配置指南
- ✅ `IMPLEMENTATION_PROGRESS.md` - 本文档

### 2. 配置文件

- ✅ `config/nuscenes_config.py` - nuScenes 配置（完整）

### 3. 工具函数

- ✅ `data/nuscenes_utils.py` - 坐标转换等工具函数（完整）

---

## ⏳ 待创建的文件

### 高优先级（核心功能）

1. **data/nuscenes_preprocessor.py** ⚠️
   - 状态: 需要用户实现 3 个关键函数
   - 函数:
     - `extract_agent_trajectories()` - 从 nuScenes 提取轨迹
     - `rasterize_map()` - HD 地图栅格化
     - `build_vocabulary()` - 构建 goal 词汇表
   - 预计时间: 6-9 小时

2. **data/nuscenes_dataset.py** ✅
   - 状态: 将提供完整实现
   - 功能: PyTorch Dataset 类

3. **scripts/preprocess_nuscenes.py** ✅
   - 状态: 将提供完整实现
   - 功能: 运行预处理流程

4. **scripts/test_nuscenes_installation.py** ✅
   - 状态: 将提供完整实现
   - 功能: 测试环境安装

### 中优先级（训练和评估）

5. **train_scorer_nuscenes.py** ⚠️
   - 状态: 基于现有 `train_goal_scorer.py` 修改
   - 修改: 导入 nuScenes 配置和数据集

6. **train_matcher_nuscenes.py** ⚠️
   - 状态: 基于现有 `train_flow_matcher.py` 修改
   - 修改: 导入 nuScenes 配置和数据集

7. **inference_nuscenes.py** ⚠️
   - 状态: 基于现有 `inference.py` 修改
   - 修改: 导入 nuScenes 配置和数据集

### 低优先级（辅助功能）

8. **scripts/visualize_nuscenes.py** ✅
   - 状态: 将提供完整实现
   - 功能: 可视化预测结果

9. **scripts/download_nuscenes.sh** ✅
   - 状态: 将提供完整实现
   - 功能: 自动下载数据集

10. **data/test_nuscenes_data.py** ✅
    - 状态: 将提供完整实现
    - 功能: 测试数据预处理

---

## 📝 下一步行动

### 立即执行（用户）

1. **阅读文档**
   - ✅ `GOALFLOW_NUSCENES_PLAN.md` - 了解整体计划
   - ✅ `NUSCENES_SETUP.md` - 配置环境

2. **等待框架代码**
   - ⏳ 我将继续创建剩余的框架代码
   - ⏳ 你将收到所有文件和详细的实现指导

### 后续执行（用户）

3. **下载数据**
   - 注册 nuScenes 账号
   - 下载 mini 数据集
   - 验证安装

4. **实现关键函数**
   - `extract_agent_trajectories()`
   - `rasterize_map()`
   - `build_vocabulary()`

5. **运行预处理**
   - 生成训练数据
   - 验证数据质量

6. **训练模型**
   - 训练 Scorer
   - 训练 Matcher

7. **评估和调优**
   - 端到端推理
   - 可视化结果
   - 超参数调优

---

## 🎯 关键实现点

### 你需要实现的 3 个函数

#### 1. extract_agent_trajectories()

**位置**: `data/nuscenes_preprocessor.py`  
**难度**: ⭐⭐⭐  
**时间**: 2-3 小时

**任务**:
- 从 nuScenes 场景中提取所有 agent 的轨迹
- 转换到 ego 车坐标系
- 过滤无效轨迹

**提供的帮助**:
- 函数签名和文档
- nuScenes API 使用示例
- 坐标转换工具函数（已实现）
- 详细的实现步骤

#### 2. rasterize_map()

**位置**: `data/nuscenes_preprocessor.py`  
**难度**: ⭐⭐⭐⭐  
**时间**: 3-4 小时

**任务**:
- 将 HD 地图转换为 BEV 栅格图
- 包含车道线、道路边界、人行道
- 输出 (3, 200, 200) 的特征图

**提供的帮助**:
- 函数签名和文档
- nuScenes 地图 API 使用示例
- OpenCV 绘图示例
- 详细的实现步骤

#### 3. build_vocabulary()

**位置**: `data/nuscenes_preprocessor.py`  
**难度**: ⭐  
**时间**: 1 小时

**任务**:
- 使用 K-means 聚类构建 goal 词汇表
- 从所有训练轨迹的终点提取

**提供的帮助**:
- 函数签名和文档
- sklearn K-means 使用示例
- 参考 toy data 的实现

---

## 📊 预期时间线

| 阶段 | 任务 | 负责人 | 状态 | 预计时间 |
|------|------|--------|------|----------|
| 0 | 创建框架代码 | AI | ⏳ 进行中 | 1-2h |
| 1 | 下载数据集 | 用户 | ⏳ 待开始 | 1-2h |
| 2 | 实现关键函数 | 用户 | ⏳ 待开始 | 6-9h |
| 3 | 运行预处理 | 用户 | ⏳ 待开始 | 1-2h |
| 4 | 训练模型 | 用户 | ⏳ 待开始 | 6-8h |
| 5 | 评估调优 | 用户 | ⏳ 待开始 | 2-4h |

**总计**: 约 17-27 小时（2-4 天）

---

## 💡 实现建议

### 分步实现策略

1. **先实现最简单的函数**
   - 从 `build_vocabulary()` 开始
   - 验证 K-means 聚类是否正常工作
   - 可视化词汇表分布

2. **再实现轨迹提取**
   - 实现 `extract_agent_trajectories()`
   - 先处理单个场景
   - 可视化提取的轨迹
   - 验证坐标转换是否正确

3. **最后实现地图栅格化**
   - 实现 `rasterize_map()`
   - 先绘制单个地图层
   - 逐步添加其他层
   - 可视化 BEV 特征图

### 调试技巧

1. **使用可视化**
   - 每个函数都应该有可视化输出
   - 检查中间结果是否合理

2. **从小数据开始**
   - 先处理 1 个场景
   - 验证正确后再处理所有场景

3. **保存中间结果**
   - 缓存预处理数据
   - 避免重复计算

4. **单元测试**
   - 为每个函数编写测试
   - 验证边界情况

---

## 📚 参考资源

### nuScenes 官方

- **API 文档**: https://github.com/nutonomy/nuscenes-devkit
- **教程**: https://www.nuscenes.org/nuscenes#tutorials
- **论文**: https://arxiv.org/abs/1903.11027

### 代码示例

- **轨迹提取**: nuScenes devkit 的 `prediction` 模块
- **地图渲染**: nuScenes devkit 的 `map_expansion` 模块
- **K-means**: sklearn 官方文档

---

## ❓ 常见问题

### Q: 我需要修改现有的模型代码吗？

**A**: 不需要。`models/` 目录下的代码（GoalPointScorer, GoalFlowMatcher）无需修改，可以直接使用。

### Q: 如果我的实现有问题怎么办？

**A**: 
1. 检查函数文档中的提示
2. 运行测试脚本验证
3. 可视化中间结果
4. 查看参考代码示例

### Q: 数据预处理需要多长时间？

**A**: 
- nuScenes mini: ~10-20 分钟
- nuScenes trainval: ~2-4 小时

### Q: 我可以跳过某些步骤吗？

**A**: 不建议。每个步骤都是必要的，跳过会导致后续问题。

---

## 🚀 准备好了吗？

当我完成所有框架代码后，你将收到：

1. ✅ 完整的代码框架
2. ✅ 详细的实现指导
3. ✅ 测试和验证脚本
4. ✅ 可视化工具

然后你就可以开始实现关键函数了！

---

**更新**: 正在创建剩余的框架代码...
