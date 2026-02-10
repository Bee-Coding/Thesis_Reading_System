# GoalFlowMatcher 测试套件

## 📁 文件结构

```
test/
├── __init__.py
├── test_goal_flow_matcher.py    # 完整测试套件
└── README.md                     # 本文件
```

---

## 🧪 测试覆盖

### 完整测试列表

1. **test_1_model_creation** - 模型创建测试
   - 验证模型能否正确初始化
   - 统计参数量
   - 检查配置参数

2. **test_2_forward_pass** - 前向传播测试
   - 验证输入输出形状
   - 检查是否有 NaN/Inf
   - 验证速度场范围

3. **test_3_loss_computation** - 损失计算测试
   - 验证损失函数正确性
   - 检查损失是否为标量
   - 验证损失非负

4. **test_4_backward_pass** - 反向传播测试
   - 验证梯度计算
   - 检查梯度是否有 NaN
   - 统计梯度范数

5. **test_5_generation_euler** - Euler 方法生成测试
   - 测试不同推理步数 (1, 5, 10)
   - 验证生成轨迹形状
   - 检查轨迹有效性

6. **test_6_generation_rk4** - RK4 方法生成测试
   - 测试 RK4 高精度求解
   - 验证生成轨迹质量

7. **test_7_multiple_generation** - 多轨迹生成测试
   - 测试批量生成多条轨迹
   - 验证轨迹多样性
   - 检查方差指标

8. **test_8_training_loop** - 训练循环测试
   - 模拟简单训练过程
   - 验证优化器工作正常
   - 检查损失变化

9. **test_9_different_configs** - 不同配置测试
   - 小模型 (d_model=128)
   - 中等模型 (d_model=256)
   - 大模型 (d_model=512)

10. **test_10_edge_cases** - 边界情况测试
    - batch_size=1
    - t=0 和 t=1
    - 不同轨迹点数 (3, 6, 12)

---

## 🚀 运行测试

### 方法 1：直接运行测试文件

```bash
cd /home/zhn/work/text/Thesis_Reading_System/implementations/goalflow/test
python test_goal_flow_matcher.py
```

### 方法 2：从项目根目录运行

```bash
cd /home/zhn/work/text/Thesis_Reading_System
python -m implementations.goalflow.test.test_goal_flow_matcher
```

### 方法 3：使用 pytest（如果安装）

```bash
cd /home/zhn/work/text/Thesis_Reading_System/implementations/goalflow
pytest test/test_goal_flow_matcher.py -v
```

---

## ✅ 预期输出

成功运行后，你应该看到：

```
======================================================================
开始运行 GoalFlowMatcher 完整测试套件
======================================================================
======================================================================
测试1：模型创建
======================================================================
✓ 模型创建成功
  总参数量: X,XXX,XXX
  d_model: 256
  轨迹点数: 6

======================================================================
测试2：前向传播
======================================================================
输入形状:
  x_t: torch.Size([4, 6, 2])
  goal: torch.Size([4, 2])
  scene: torch.Size([4, 64, 32, 32])
  t: torch.Size([4])

输出形状:
  v_pred: torch.Size([4, 6, 2])
✓ 前向传播测试通过
  速度范围: [-X.XXXX, X.XXXX]

... (更多测试输出)

======================================================================
✓✓✓ 所有测试通过！✓✓✓
======================================================================

恭喜！GoalFlowMatcher 实现正确，可以进行下一步了！
```

---

## ❌ 常见错误及解决方案

### 错误 1：导入失败

```
ModuleNotFoundError: No module named 'models.goal_flow_matcher'
```

**解决方案**：
- 确保在正确的目录运行
- 检查 `goal_flow_matcher.py` 是否存在于 `models/` 目录

### 错误 2：形状不匹配

```
AssertionError: 输出形状错误！期望 (4, 6, 2), 得到 (4, 6, 256)
```

**解决方案**：
- 检查 `velocity_decoder` 的输出维度
- 确保最后一层输出是 `traj_dim` (2)

### 错误 3：NaN 输出

```
AssertionError: 输出包含 NaN！
```

**解决方案**：
- 检查权重初始化
- 检查 BatchNorm 是否正确使用
- 检查是否有除零操作
- 降低学习率

### 错误 4：梯度消失/爆炸

```
梯度范数: min=0.000000, max=1000000.000000
```

**解决方案**：
- 使用梯度裁剪：`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- 检查权重初始化
- 使用 Pre-LN Transformer (`norm_first=True`)

---

## 🔧 调试技巧

### 1. 单独运行某个测试

修改 `test_goal_flow_matcher.py` 的 `run_all_tests()` 函数：

```python
def run_all_tests():
    model = test_1_model_creation()
    test_2_forward_pass(model)  # 只运行这一个测试
    # 注释掉其他测试
```

### 2. 打印中间结果

在你的 `goal_flow_matcher.py` 中添加调试输出：

```python
def forward(self, x_t, goal, scene, t):
    print(f"[DEBUG] x_t shape: {x_t.shape}")
    
    traj_tokens = self.traj_encoder(x_t)
    print(f"[DEBUG] traj_tokens shape: {traj_tokens.shape}")
    
    # ... 更多调试输出
```

### 3. 使用 PyTorch 的调试工具

```python
# 检测 NaN
torch.autograd.set_detect_anomaly(True)

# 打印梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm().item():.6f}")
```

---

## 📊 性能基准

### 参考参数量

| 配置 | d_model | nhead | layers | 参数量 |
|------|---------|-------|--------|--------|
| 小模型 | 128 | 4 | 2 | ~500K |
| 中等模型 | 256 | 8 | 4 | ~2M |
| 大模型 | 512 | 8 | 6 | ~8M |

### 参考运行时间

- **前向传播** (batch_size=4): ~10-50ms
- **反向传播** (batch_size=4): ~20-100ms
- **生成轨迹** (num_steps=1): ~5-20ms
- **完整测试套件**: ~30-60秒

---

## 📝 下一步

测试通过后，你可以：

1. **创建 Toy Dataset**
   - 生成模拟轨迹数据
   - 构建 Goal Point Vocabulary
   - 实现 DataLoader

2. **端到端训练**
   - 在 toy dataset 上训练
   - 可视化生成结果
   - 调整超参数

3. **集成 GoalPointScorer**
   - 联合训练两个模块
   - 实现完整的 GoalFlow 流程

4. **准备真实数据集**
   - 下载 nuScenes mini
   - 提取轨迹和 BEV 特征
   - 适配数据格式

---

## 💡 提示

- 如果测试失败，不要慌张，仔细阅读错误信息
- 从简单的测试开始调试（test_1, test_2）
- 使用小的 batch_size 和模型配置进行快速迭代
- 保存测试通过时的模型权重

---

**祝你测试顺利！** 🚀

如果遇到问题，请检查：
1. 模型实现是否完整
2. 所有方法的输入输出形状
3. 是否正确导入了依赖模块
