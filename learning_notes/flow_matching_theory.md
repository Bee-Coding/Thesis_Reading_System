# Flow Matching 理论深入理解

## 1. 核心概念

### 1.1 什么是 Flow Matching？

Flow Matching 是一种生成模型，通过学习一个**连续时间的向量场**，将简单分布（如高斯噪声）变换到复杂的数据分布。

**关键思想**：
- 不是直接学习数据分布 p(x)
- 而是学习一个时间依赖的速度场 v(x, t)
- 通过求解 ODE 从噪声生成数据

### 1.2 数学形式

#### ODE 形式
```
dx/dt = v(x, t),  t ∈ [0, 1]
x(0) ~ p_0 (噪声分布)
x(1) ~ p_1 (数据分布)
```

#### 目标
学习速度场 v_θ(x, t)，使得：
- 从 x(0) 出发
- 沿着速度场积分
- 到达 x(1) 符合数据分布

---

## 2. Optimal Transport (OT) Flow

### 2.1 OT Flow 定义

最简单的 Flow：**线性插值**

```
x_t = (1-t) * x_0 + t * x_1
v_t = x_1 - x_0  (常数速度)
```

**特点**：
- 最短路径（直线）
- 速度恒定
- 计算简单

### 2.2 为什么叫 Optimal Transport？

- 在所有可能的传输路径中，直线路径的"传输成本"最小
- 对应 Optimal Transport 理论中的 Monge 问题

---

## 3. Conditional Flow Matching (CFM)

### 3.1 训练目标

**CFM Loss**：
```
L = E_{t, x_0, x_1} [ ||v_θ(x_t, t) - (x_1 - x_0)||² ]
```

**解释**：
- 采样时间 t ~ Uniform[0, 1]
- 采样噪声 x_0 ~ N(0, I)
- 采样数据 x_1 ~ p_data
- 计算插值点 x_t = (1-t)*x_0 + t*x_1
- 让网络预测的速度接近真实速度 (x_1 - x_0)

### 3.2 为什么这样训练有效？

**关键洞察**：
1. 如果网络能在所有 (x_t, t) 处预测正确的速度
2. 那么从任意 x_0 出发，沿着预测的速度场积分
3. 就能到达对应的 x_1

---

## 4. 与 Diffusion Model 的对比

| 特性 | Flow Matching | Diffusion Model |
|------|---------------|-----------------|
| 训练目标 | 速度场 v(x,t) | 噪声 ε(x,t) |
| 采样方式 | ODE 求解 | SDE/ODE 求解 |
| 采样步数 | 少（10-50步） | 多（100-1000步） |
| 训练稳定性 | 高 | 中等 |
| 理论基础 | Optimal Transport | Score Matching |

**Flow Matching 的优势**：
- 更快的采样速度
- 更简单的训练目标
- 更直观的几何解释

---

## 5. 实现细节

### 5.1 网络架构

**输入**：
- x_t: 当前状态 (B, D)
- t: 时间标量 (B,)
- cond: 条件信息 (B, C)

**输出**：
- v: 速度场 (B, D)

**常用架构**：
- MLP（简单场景）
- Transformer（复杂场景）
- U-Net（图像生成）

### 5.2 时间编码

使用 Sinusoidal Embedding：
```python
freq_i = 1 / (10000^(2i/d))
emb = [sin(t*freq_0), cos(t*freq_0), ..., sin(t*freq_{d/2}), cos(t*freq_{d/2})]
```

**作用**：
- 将标量 t 映射到高维空间
- 保留时间的连续性
- 类似 Transformer 的位置编码

### 5.3 ODE 求解

**Euler 方法**（一阶）：
```python
x_{t+dt} = x_t + v(x_t, t) * dt
```

**RK4 方法**（四阶，更精确）：
```python
k1 = v(x_t, t)
k2 = v(x_t + 0.5*dt*k1, t + 0.5*dt)
k3 = v(x_t + 0.5*dt*k2, t + 0.5*dt)
k4 = v(x_t + dt*k3, t + dt)
x_{t+dt} = x_t + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
```

---

## 6. 条件生成

### 6.1 为什么需要条件？

在轨迹预测中，我们需要：
- 给定目标点 → 生成到达该点的轨迹
- 给定场景信息 → 生成符合场景的轨迹
- 给定轨迹类型 → 生成特定类型的轨迹

### 6.2 如何添加条件？

**方法 1：拼接**（我们当前的实现）
```python
x_input = concat([x_t, cond, time_emb])
v = MLP(x_input)
```

**方法 2：Cross-Attention**（更强大）
```python
v = Transformer(x_t, context=cond, time=t)
```

**方法 3：Adaptive Layer Norm**
```python
scale, shift = MLP(cond, t)
x_normalized = LayerNorm(x)
x_modulated = x_normalized * scale + shift
```

---

## 7. 关键问题和解答

### Q1: 为什么 Flow Matching 比 Diffusion 快？

**A**: 
- Diffusion 需要逐步去噪，每步只能去除一点噪声
- Flow Matching 直接学习最优路径，可以用更少的步数
- OT Flow 是直线路径，最短距离

### Q2: 如何保证生成的轨迹合理？

**A**:
- 通过条件控制（目标点、场景信息）
- 训练数据的质量（只用合理的轨迹训练）
- 添加约束（碰撞检测、动力学约束）

### Q3: 如何处理多模态？

**A**:
- 噪声 x_0 的随机性提供多样性
- 不同的 x_0 → 不同的轨迹
- 可以采样多次生成多条轨迹

---

## 8. 下一步学习

- [ ] 阅读 Flow Matching 原始论文
- [ ] 理解 Optimal Transport 理论
- [ ] 学习 Goal Flow Matching 的改进
- [ ] 实现更复杂的条件控制
- [ ] 在真实数据集上验证

---

## 参考资料

1. **Flow Matching for Generative Modeling** (Lipman et al., 2023)
2. **Conditional Flow Matching** (Tong et al., 2023)
3. **Optimal Transport** (Villani, 2009)
4. **Score-Based Generative Models** (Song et al., 2021)

