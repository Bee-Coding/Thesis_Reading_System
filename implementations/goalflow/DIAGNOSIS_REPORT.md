# GoalFlow 诊断报告

**日期**: 2026-02-25  
**问题**: 推理 ADE 为 9.0+，远高于预期（应 < 1.0）

---

## 🔍 诊断结果总结

### 1. Scorer 诊断（目标点选择）

**运行命令**: `python diagnose_scorer.py`

**关键指标**:
- **Goal Error**: 16.78 ± 9.28 m（轨迹长度的 97.6%）
- **Top-1 准确率**: 0.00%
- **Top-5 准确率**: 2.50%

**结论**: ❌ **Scorer 训练完全失败**

**详细分析**:
1. Scorer 只训练了 **3 个 epoch**（应该训练 100 个）
2. 目标点选择完全随机，Top-1 准确率为 0%
3. 选中的目标点与真实目标点的距离平均为 16.78m，几乎等于整个轨迹的长度
4. pred_dis 和 pred_dac 都没有学到有效的特征

**组件分析**:
- dis only: 16.78 m
- dac only: 16.71 m  
- combined: 16.78 m
- 结论：两个组件都没有学到任何有用信息

---

### 2. Matcher 诊断（轨迹生成）

**运行命令**: `python diagnose_matcher.py`

**部分结果**（使用真实目标点）:
- **ADE**: ~2.8-3.5 m（从部分输出推断）
- **训练 epoch**: 100

**结论**: ⚠️ **Matcher 部分工作，但性能不佳**

**详细分析**:
1. Matcher 训练了完整的 100 个 epoch
2. 即使使用真实目标点，ADE 仍然在 2.8-3.5 m 左右
3. 这个 ADE 相对于轨迹长度（17.19m）约为 16-20%
4. 虽然比随机选择的 9.0+ 好很多，但仍然不够理想（应该 < 1.0）

---

## 🎯 根本原因分析

### 主要问题：Scorer 训练失败

**证据**:
1. Checkpoint 显示只训练了 3 个 epoch
2. Top-1 准确率为 0%，完全随机选择
3. Goal Error 为 16.78m，接近轨迹长度

**为什么 Scorer 只训练了 3 个 epoch？**
- 需要检查 `train_goal_scorer.py` 的训练日志
- 可能原因：
  - 训练脚本提前终止
  - 验证集上的指标没有改善，触发了 early stopping
  - 训练过程中出现错误

### 次要问题：Matcher 性能不佳

**证据**:
1. 即使用真实目标点，ADE 仍为 2.8-3.5m
2. 这个误差相对于轨迹长度约为 16-20%

**可能原因**:
1. **ODE 步数太少**（当前为 10 步）
2. **初始噪声太大**（当前 noise_std = 1.0）
3. **网络容量不足**
4. **训练数据太简单**（toy data 可能不够复杂）

---

## 📊 数据上下文

- **轨迹数量**: 1000（800 train, 200 val）
- **轨迹长度**: 平均 17.19 m
- **轨迹点数**: 6 点
- **坐标范围**: [-15.55, 16.93]
- **词汇表大小**: 128 个目标点

---

## 🔧 解决方案

### 优先级 1：重新训练 Scorer（必须）

**问题**: Scorer 只训练了 3 个 epoch，完全没有学到任何东西

**解决方案**:
1. **检查训练日志**
   ```bash
   # 查看是否有训练日志文件
   ls -la implementations/goalflow/logs/
   ```

2. **重新训练 Scorer**
   ```bash
   cd implementations/goalflow
   python train_goal_scorer.py
   ```
   
3. **监控训练过程**
   - 确保训练运行完整的 100 个 epoch
   - 观察 loss 是否下降
   - 观察 Top-1 准确率是否提升

4. **预期结果**
   - 训练完成后，Top-1 准确率应该 > 50%
   - Goal Error 应该 < 5.0 m
   - 如果达到这个标准，重新运行 `inference.py`，ADE 应该会显著降低

### 优先级 2：改进 Matcher（可选）

**问题**: 即使用真实目标点，ADE 仍为 2.8-3.5m

**解决方案**（在 Scorer 重新训练后再考虑）:

1. **增加 ODE 步数**
   - 修改 `config/matcher_config.py`
   - 将 `num_steps` 从 10 增加到 20 或 50
   - 重新运行 `diagnose_matcher.py` 测试

2. **减小初始噪声**
   - 修改 `config/matcher_config.py`
   - 将 `noise_std` 从 1.0 减小到 0.5 或 0.3
   - 重新训练 Matcher

3. **增加网络容量**
   - 增加 `hidden_dim` 或 `num_layers`
   - 重新训练 Matcher

---

## 📝 下一步行动

### 立即执行

1. ✅ **运行 Scorer 诊断** - 已完成
2. ✅ **运行 Matcher 诊断** - 已完成（部分）
3. ⏳ **检查 Scorer 训练日志** - 待执行
4. ⏳ **重新训练 Scorer** - 待执行
5. ⏳ **验证 Scorer 性能** - 待执行
6. ⏳ **重新运行完整推理** - 待执行

### 后续优化（如果需要）

7. ⏳ **调整 Matcher 超参数** - 待执行
8. ⏳ **重新训练 Matcher** - 待执行

---

## 🎓 学习要点

### 诊断方法论

1. **分而治之**
   - 将复杂系统分解为独立模块
   - 分别测试每个模块的性能
   - 定位问题的根源

2. **使用 Ground Truth 测试**
   - 用真实目标点测试 Matcher，排除 Scorer 的影响
   - 这种方法可以快速定位问题所在

3. **关注关键指标**
   - Top-K 准确率：衡量分类/选择任务
   - ADE/FDE：衡量回归任务
   - 相对误差：将绝对误差与数据尺度对比

### 训练监控的重要性

1. **检查 checkpoint 信息**
   - epoch 数量
   - 验证集指标
   - 训练时间

2. **保存训练日志**
   - Loss 曲线
   - 验证指标曲线
   - 便于事后分析

---

## 📂 相关文件

### 诊断脚本
- `implementations/goalflow/diagnose_scorer.py` - Scorer 诊断
- `implementations/goalflow/diagnose_matcher.py` - Matcher 诊断

### 训练脚本
- `implementations/goalflow/train_goal_scorer.py` - Scorer 训练
- `implementations/goalflow/train_flow_matcher.py` - Matcher 训练

### Checkpoint
- `implementations/goalflow/checkpoints/scorer/best.pth` - Scorer（3 epochs）
- `implementations/goalflow/checkpoints/matcher/best.pth` - Matcher（100 epochs）

### 配置文件
- `implementations/goalflow/config/scorer_config.py`
- `implementations/goalflow/config/matcher_config.py`

---

## 💡 总结

**核心问题**: Scorer 只训练了 3 个 epoch，导致目标点选择完全随机

**解决方案**: 重新训练 Scorer 至 100 个 epoch

**预期改善**: 
- Scorer Top-1 准确率：0% → 50%+
- 推理 ADE：9.0+ → 3.0-4.0（如果 Matcher 保持当前性能）
- 如果进一步优化 Matcher，ADE 可能降至 1.0-2.0

**时间估计**:
- 重新训练 Scorer：~30-60 分钟
- 验证和测试：~10 分钟
- 总计：~1 小时

---

**诊断完成时间**: 2026-02-25  
**诊断工具**: `diagnose_scorer.py`, `diagnose_matcher.py`  
**下一步**: 重新训练 Scorer
