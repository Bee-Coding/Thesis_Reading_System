# Flow Matching: 训练 vs 推理详解

## 核心区别总结

| 阶段 | 输入 | 输出 | 关键点 |
|------|------|------|--------|
| **训练** | 真实轨迹 x₁ (已知) | 网络参数 θ | 有监督学习，知道答案 |
| **推理** | 随机噪声 x₀ | 生成轨迹 x₁ | 使用学到的映射生成新数据 |

---

## 第一部分：训练过程（有监督学习）

### 训练时我们有什么？

✅ **真实轨迹数据** x₁ (从数据集采样)
✅ **随机噪声** x₀ (随机生成)
✅ **真实速度标签** v_true = x₁ - x₀

### 训练流程（7个步骤）

```python
# Step 1: 从数据集采样真实轨迹
x_1 = dataset.sample()  # [6, 2] - 6个点，每个点(x,y)

# Step 2: 采样随机噪声
x_0 = torch.randn_like(x_1)  # [6, 2]

# Step 3: 随机采样时间 t ∈ [0, 1]
t = torch.rand(1)  # 例如 t = 0.6

# Step 4: 线性插值得到中间状态
x_t = (1 - t) * x_0 + t * x_1
# t=0: 纯噪声
# t=0.6: 60%真实数据 + 40%噪声
# t=1: 纯真实数据

# Step 5: 计算真实速度（这是标签！）
v_true = x_1 - x_0  # [6, 2]

# Step 6: 网络预测速度
v_pred = model(x_t, t, goal, bev_feat)

# Step 7: 计算损失并更新参数
loss = MSE(v_pred, v_true)
loss.backward()
optimizer.step()
```

### 关键理解

**训练的本质**：教网络学习"从噪声到数据的方向"

- 输入：(x_t, t, goal, bev_feat)
- 标签：v_true = x₁ - x₀ (已知！)
- 目标：让 v_pred ≈ v_true

---

## 第二部分：推理过程（生成新数据）

### 推理时我们有什么？

✅ **训练好的网络** v_θ
✅ **随机噪声** x₀
✅ **条件信息** (Goal Point, BEV特征)

❌ **没有真实数据** x₁ (这正是我们要生成的！)

### 推理流程

```python
# Step 1: 采样初始噪声
x_0 = torch.randn(1, 6, 2)

# Step 2: 获取条件信息
bev_feat = perception_module(images, lidar)
ego_state = get_ego_state()
goal = goal_point_constructor(bev_feat, ego_state)

# Step 3: 网络预测速度（从t=0开始）
t = torch.tensor([0.0])
v = model(x_0, t, goal, bev_feat)

# Step 4: 生成数据（单步）
x_1 = x_0 + v  # 这就是生成的轨迹！
```

### 多步推理（更精确）

```python
x = x_0
n_steps = 20
dt = 1.0 / n_steps

for i in range(n_steps):
    t = torch.tensor([i * dt])
    v = model(x, t, goal, bev_feat)
    x = x + v * dt  # 小步前进

x_1 = x  # 最终轨迹
```

### 关键理解

**推理的本质**：使用学到的"方向"从噪声生成数据

- 训练时：网络学会了 v_θ(x_t, t) ≈ x₁ - x₀
- 推理时：给定新噪声 x₀，预测 v ≈ x₁ - x₀
- 结果：x₁ = x₀ + v

---

## 第三部分：为什么推理时网络能工作？

### 数学解释

训练时，网络学习的是**条件分布**：

```
v_θ(x₀, 0, g, c) ≈ E[x₁ - x₀ | x₀, g, c]
```

意思是：给定噪声 x₀ 和条件 (g, c)，预测最可能的数据点的期望位置。

### 直观理解

**训练阶段**：
```
老师（数据集）：这是真实轨迹 x₁
学生（网络）：我记住了从噪声 x₀ 到 x₁ 的方向
```

**推理阶段**：
```
新噪声 x₀'：我在这里，应该往哪走？
网络：根据我学到的知识，你应该往这个方向走 v
结果：x₁' = x₀' + v
```

---

## 第四部分：训练 vs 推理对比表

| 维度 | 训练 | 推理 |
|------|------|------|
| **真实数据 x₁** | ✅ 有（从数据集） | ❌ 无（要生成的） |
| **噪声 x₀** | ✅ 随机采样 | ✅ 随机采样 |
| **时间 t** | 随机 t~U(0,1) | 固定从0开始 |
| **真实速度** | ✅ 已知 v=x₁-x₀ | ❌ 未知 |
| **网络作用** | 学习预测速度 | 使用学到的速度 |
| **目标** | 最小化预测误差 | 生成合理轨迹 |

---

## 第五部分：多模态生成

### 方法1：不同噪声初始化

```python
trajectories = []
for i in range(5):
    x_0 = torch.randn(1, 6, 2)  # 不同噪声
    x_1 = generate(x_0, goal, bev_feat)
    trajectories.append(x_1)

# 结果：5条不同的轨迹
```

### 方法2：不同Goal Point

```python
goals = get_top_k_goals(bev_feat, k=3)  # 3个候选目标

trajectories = []
for goal in goals:
    x_0 = torch.randn(1, 6, 2)
    x_1 = generate(x_0, goal, bev_feat)
    trajectories.append(x_1)

# 结果：3条对应不同目标的轨迹
```

### 方法

```python
# 3个Goal × 5个噪声 = 15条轨迹
goals = get_top_k_goals(bev_feat, k=3)

all_trajectories = []
for goal in goals:
    for i in range(5):
        x_0 = torch.randn(1, 6, 2)
        x_1 = generate(x_0, goal, bev_feat)
        all_trajectories.append(x_1)

# 从15条中选择最优
best = select_best(all_trajectories)
```

---

## 第六部分：常见疑问解答

### Q1: 训练时为什么要随机采样 t？

**答**：为了让网络学习整个路径上的速度场。

- 只训练 t=0：只学会第一步
- 只训练 t=1：只学会最后一步
- 随机采样 t：学会路径上每一点的速度

### Q2: 推理时为什么可以单步？

**答**：因为网络学习的是全局速度场。

- 训练时：在所有 t∈[0,1] 上训练
- 推理时：t=0 的速度已经"知道"整条路径
- 单步 = 直接跳到终点（近似）
- 多步 = 沿着路径走（更精确）

### Q3: 为什么不同噪声生成不同轨迹？

**答**：网络学习的是一对一映射。

- 训练时：每个 (x₀, x₁) 对都不同
- 网络学习：f(x₀) ≈ x₁
- 推理时：不同 x₀ → 不同 f(x₀)

### Q4: Goal Point如何影响生成？

**答**：Goal Point作为条件输入网络。

- 网络学习：v_θ(x_t, t, g, c)
- 不同的 g → 不同的速度场
- g_left → 向左的速度，g_right → 向右的速度

---

## 总结

### 训练的本质
- **有监督学习**：知道真实数据 x₁
- **学习目标**：预测速度 v = x₁ - x₀
- **训练数据**：(x₀, x₁, t) 三元组

### 推理的本质
- **生成过程**：从噪声 x₀ 生成数据 x₁
- **使用知识**：网络学到的速度场 v_θ
- **生成方式**：x₁ = x₀ + v_θ(x₀, 0, g, c)

### 关键理解
1. **训练时**：我们有真实答案，教网络"从噪声到数据的方向"
2. **推理时**：我们没有答案，但网络已经学会了"方向"
3. **多模态**：不同噪声 + 不同Goal Point → 多样化的轨迹
---

## 可视化对比

```
训练流程：
数据集 → x₁ (已知) ──┐
                      ├→ v_true = x₁ - x₀ (标签)
随机噪声 → x₀ ────────┘
         ↓
      插值 x_t
         ↓
    网络预测 v_pred
         ↓
    Loss = ||v_pred - v_true||²
         ↓
    更新参数 θ


推理流程：
随机噪声 → x₀
         ↓
    网络预测 v = v_θ(x₀, 0, g, c)
         ↓
    生成数据 x₁ = x₀ + v
         ↓
    输出轨迹
```
