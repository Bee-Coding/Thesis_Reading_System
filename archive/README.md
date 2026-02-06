# 归档文件说明

本目录包含已归档的学习记录脚本。

## 📁 文件列表

### 1. `record_flow_matching_learning.py`
**用途**：记录 Flow Matching 理论学习内容到 Mem0

**记录时间**：2026-02-05（早期学习阶段）

**记录内容**：
- Flow Matching 基础理论
- OT Flow、CFM Loss、RK4 方法
- 关键洞察和知识盲区
- 理论问题和答案

**状态**：已归档，作为学习历史保留

---

### 2. `record_flow_matching_implementation.py`
**用途**：记录 Flow Matching 代码实现学习内容到 Mem0

**记录时间**：2026-02-05（toy dataset 训练完成时）

**记录内容**：
- PyTorch 训练循环实现
- 代码问题和解决方案
- 训练结果（2 epochs）
- 第一次写训练代码的经验

**状态**：已归档，作为学习历史保留

---

## 🔄 替代文件

这些归档文件已被以下新文件替代：

- `../record_goalflow_learning.py` - 记录 GoalFlow 论文学习
- `../record_goalflow_implementation.py` - 记录 GoalFlow 实现学习

---

## 📝 为什么归档？

1. **内容过时**：记录的是早期学习内容（toy dataset 阶段）
2. **进度更新**：现在已经进入 GoalFlow 论文复现阶段
3. **保留历史**：作为学习轨迹的一部分保留

---

## 🎯 如何使用归档文件

### 查看学习历史
```bash
# 查看早期的理论学习记录
python archive/record_flow_matching_learning.py

# 查看早期的实现学习记录
python archive/record_flow_matching_implementation.py
```

### 对比学习进步
可以对比归档文件和新文件，看到学习进步：
- 从 Flow Matching 基础 → GoalFlow 论文复现
- 从 toy dataset → 真实数据集
- 从 2 epochs → 477 epochs

---

**归档时间**：2026-02-06  
**归档原因**：进入 GoalFlow 复现阶段，创建新的记录脚本
