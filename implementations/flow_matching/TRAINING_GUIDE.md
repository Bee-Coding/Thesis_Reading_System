# Flow Matching 训练流程学习指南

## 🎯 核心概念回顾

### Flow Matching 的训练目标
训练一个神经网络 `v_θ(x, t)` 来预测速度场，使得：
- 从噪声 `x_0 ~ N(0, I)` 出发
- 通过求解 ODE: `dx/dt = v_θ(x, t)`
- 最终到达数据分布 `x_1 ~ p_data`

### 损失函数
```
L = E_{t, x_0, x_1} [ ||v_θ(x_t, t) - (x_1 - x_0)||² ]
```

其中：
- `x_t = (1-t) * x_0 + t * x_1` (线性插值)
- `x_1 - x_0` 是真实的速度场（OT Flow 的常数速度）

---

## 📝 代码结构详解

### 1. 数据处理流程

```python
# Dataset 返回的数据格式
batch = {
    'trajectory': torch.Tensor,  # shape: (B, 6, 2)
    'type': list                 # ['circle', 'line', ...]
}

# 训练时的处理步骤
x_1 = batch['trajectory']        # (B, 6, 2) - 真实轨迹
x_1 = x_1.reshape(B, -1)         # (B, 12) - Flatten
x_0 = torch.randn_like(x_1) * 0.5 # (B, 12) - 采样噪声
```

**为什么要 Flatten？**
- 原始数据：6个点，每个点2维 → (6, 2)
- 网络输入：需要一个向量 → (12,)
- 这样网络可以学习整条轨迹的速度场

**为什么 x_0 用 randn？**
- Flow Matching 从噪声分布开始
- `randn` 采样标准正态分布 N(0, 1)
- 乘以 0.5 是为了减小初始噪声的方差

---

### 2. 训练循环 (train_epoch)

```python
def train_epoch(self, epoch: int) -> float:
    self.model.train()  # 设置为训练模式（启用 dropout 等）
    
    for batch in self.train_loader:
        # Step 1: 准备数据
        x_1 = batch['trajectory'].to(device)  # 移到 GPU
        x_1 = x_1.reshape(B, -1)              # Flatten
        x_0 = torch.randn_like(x_1) * 0.5     # 采样噪声
        
        # Step 2: 清零梯度（重要！）
        self.optimizer.zero_grad()
        
        # Step 3: 计算损失
        loss = self.flow_matcher.compute_cfm_loss(model, x_0, x_1)
        
        # Step 4: 反向传播
        loss.backward()
        
        # Step 5: 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Step 6: 更新参数
        self.optimizer.step()
    
    return avg_loss
```

**关键点**：
1. **zero_grad()**: 必须清零，否则梯度会累积
2. **backward()**: 计算梯度
3. **step()**: 根据梯度更新参数
4. **梯度裁剪**: 防止梯度过大导致训练不稳定

---

### 3. 验证循环 (validate)

```python
def validate(self) -> float:
    self.model.eval()  # 设置为评估模式（关闭 dropout）
    
    with torch.no_grad():  # 不计算梯度，节省内存
        for batch in self.val_loader:
            # 只计算损失，不更新参数
            loss = self.flow_matcher.compute_cfm_loss(model, x_0, x_1)
    
    return avg_loss
```

**与训练的区别**：
- ❌ 不调用 `zero_grad()`
- ❌ 不调用 `backward()`
- ❌ 不调用 `step()`
- ✅ 使用 `torch.no_grad()` 节省内存

---

### 4. 完整训练流程 (train)

```python
def train(self, num_epochs: int):
    for epoch in range(num_epochs):
        # 1. 训练一个 epoch
        train_loss = self.train_epoch(epoch)
        
        # 2. 验证
        val_loss = self.validate()
        
        # 3. 保存最佳模型
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_checkpoint(epoch, val_loss, is_best=True)
```

---

## 🔧 你的代码问题总结

### ❌ 问题 1: 数据处理错误
```python
# 你的代码
batch_x1s = batch  # batch 是字典
batch_x0s = torch.rand_like(batch_x1s)  # ❌ 不能对字典用 rand_like
```

**正确做法**：
```python
x_1 = batch['trajectory'].to(device)  # 提取 trajectory
x_1 = x_1.reshape(batch_size, -1)     # Flatten
x_0 = torch.randn_like(x_1) * 0.5     # 采样噪声
```

---

### ❌ 问题 2: 验证时更新了参数
```python
# 你的代码（在 validate 函数中）
self.optimizer.zero_grad()  # ❌ 验证时不需要
loss.backward()             # ❌ 验证时不需要
self.optimizer.step()       # ❌ 验证时不需要
```

**正确做法**：
```python
with torch.no_grad():
    loss = self.flow_matcher.compute_cfm_loss(model, x_0, x_1)
    # 只计算损失，不更新参数
```

---

### ❌ 问题 3: 模型接口不匹配
你的 `VelocityFieldMLP` 需要 3 个参数：
```python
def forward(self, state, cond, t):  # 需要 condition
```

但 toy dataset 不需要条件，所以我创建了 `SimpleVelocityField`：
```python
def forward(self, x, t):  # 只需要 state 和 time
```

---

## 🚀 如何运行训练

### 1. 生成数据集（如果还没有）
```bash
cd implementations/flow_matching/data
python toy_dataset.py
```

### 2. 开始训练
```bash
cd implementations/flow_matching
python train.py --epochs 50 --batch_size 32 --lr 1e-3
```

### 3. 查看训练进度
训练时会显示：
```
Epoch 1 [Train]: 100%|████████| 157/157 [00:10<00:00, loss=0.234567]
Validating: 100%|████████| 16/16 [00:01<00:00, loss=0.123456]

Epoch 1/50
  Train Loss: 0.234567
  Val Loss:   0.123456
  ✓ 新的最佳验证损失!
```

---

## 📊 训练技巧

### 1. 学习率调整
```python
# 使用余弦退火
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-6
)
```

### 2. 梯度裁剪
```python
# 防止梯度爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 3. 早停 (Early Stopping)
如果验证损失不再下降，可以提前停止训练。

### 4. 保存检查点
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_loss': val_loss,
}
torch.save(checkpoint, 'best.pth')
```

---

## 🎓 学习建议

### 第一次写训练代码的常见困惑

1. **为什么要 zero_grad()？**
   - PyTorch 默认会累积梯度
   - 每次反向传播前必须清零

2. **backward() 和 step() 的区别？**
   - `backward()`: 计算梯度（存储在 `.grad` 中）
   - `step()`: 根据梯度更新参数

3. **train() 和 eval() 的区别？**
   - `train()`: 启用 dropout、batch norm 等
   - `eval()`: 关闭 dropout、batch norm 等

4. **为什么验证时用 no_grad()？**
   - 不需要计算梯度，节省内存
   - 加快计算速度

---

## 📖 推荐学习资源

1. **PyTorch 官方教程**
   - Training a Classifier: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

2. **理解反向传播**
   - 3Blue1Brown 的视频系列

3. **调试技巧**
   - 打印张量形状：`print(x.shape)`
   - 检查梯度：`print(model.parameters()[0].grad)`
   - 使用 `pdb` 调试器

---

## ✅ 下一步

完成训练后，你需要：
1. ✅ 运行训练脚本
2. ✅ 观察损失曲线
3. ✅ 使用训练好的模型生成轨迹
4. ✅ 可视化结果（我来帮你完成）

加油！🚀
