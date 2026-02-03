# GoalFlow 深度学习笔记

## 第一部分：从扩散模型到Flow Matching

### 1.1 扩散模型（Diffusion Models）基础

#### 核心思想
扩散模型通过**逐步添加噪声**将数据分布转换为简单的高斯分布，然后学习**逆向去噪过程**来生成新数据。

#### 前向过程（加噪）
给定数据点 $x_0 \sim q(x_0)$，逐步添加高斯噪声：

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

其中：
- $t \in [0, T]$ 是时间步
- $\beta_t$ 是噪声调度参数
- 当 $t=T$ 时，$x_T \approx \mathcal{N}(0, I)$（纯噪声）

#### 逆向过程（去噪）
学习一个神经网络 $\epsilon_\theta(x_t, t)$ 来预测噪声，从而恢复数据：

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

#### 训练目标
最小化预测噪声与真实噪声的差异：

$$\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]$$

#### 推理过程
从纯噪声 $x_T \sim \mathcal{N}(0, I)$ 开始，逐步去噪：

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$

**问题**：需要 **T=1000** 步迭代，推理速度慢！

---

### 1.2 Flow Matching - 更高效的生成方法

#### 核心区别
| 维度 | 扩散模型 | Flow Matching |
|------|---------|---------------|
| 路径 | 曲折的马尔可夫链 | 直线路径（ODE） |
| 推理步数 | 1000步 | 1-20步 |
| 训练目标 | 预测噪声 $\epsilon$ | 预测速度场 $v$ |
| 数学框架 | SDE（随机微分方程） | ODE（常微分方程） |

#### Flow Matching的数学原理

**1. 概率路径定义**

给定：
- $\pi_0 = \mathcal{N}(0, I)$ - 初始噪声分布
- $\pi_1 = q(x)$ - 目标数据分布
- $t \in [0, 1]$ - 归一化时间

构造**直线路径**（Rectified Flow）：

$$x_t = (1-t) x_0 + t x_1$$

其中：
- $x_0 \sim \pi_0$ （噪声）
- $x_1 \sim \pi_1$ （真实数据）

**物理直觉**：想象从噪声点 $x_0$ 到数据点 $x_1$ 画一条直线，$t$ 控制在这条线上的位置。

**2. 速度场（Vector Field）**

沿着路径的瞬时速度：

$$\frac{dx_t}{dt} = x_1 - x_0$$

这是一个**常数速度**！（这就是为什么叫"直线路径"）

**3. 训练目标**

学习一个神经网络 $v_\theta(x_t, t)$ 来预测这个速度场：

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_0, x_1} \left[ \|v_\theta(x_t, t) - (x_1 - x_0)\|^2 \right]$$

**关键优势**：
- 目标简单：直接预测 $(x_1 - x_0)$
- 路径直：不需要多步迭代
- 速度快：单步推理即可

**4. 推理过程（采样）**

从噪声 $x_0 \sim \mathcal{N}(0, I)$ 开始，沿着学到的速度场积分：

$$x_1 = x_0 + \int_0^1 v_\theta(x_t, t) dt$$

实践中使用**欧拉法**（Euler method）：

$$x_{t+\Delta t} = x_t + v_\theta(x_t, t) \cdot \Delta t$$

**单步推理**：直接设 $\Delta t = 1$：

$$x_1 = x_0 + v_\theta(x_0, 0)$$

---

### 1.3 为什么Flow Matching更快？

#### 数学解释

**扩散模型**：
- 路径是**曲折的**马尔可夫链
- 每一步只能走一小步（$\beta_t$ 很小）
- 需要 $T \approx 1000$ 步才能从噪声到数据

**Flow Matching**：
- 路径是**直线**
- 可以大步长甚至单步到达
- 因为学习的是**全局速度场**，而非局部噪声

#### 可视化对比

```
扩散模型路径：
噪声 ~~~曲折路径~~~ 数据
      (1000步)

Flow Matching路径：
噪声 ————直线———— 数据
      (1步)
```

---

## 第二部分：GoalFlow的核心创新

### 2.1 问题：为什么需要Goal Point？

#### 多模态轨迹生成的挑战

在自动驾驶中，给定当前场景，可能有**多条合理轨迹**：
- 左转进入车位A
- 右转进入车位B
- 直行到车位C

**传统Flow Matching的问题**：
- 直接生成轨迹 $\tau$，容易产生**模糊的混合**
- 例如：生成一条"既不左也不右"的中间轨迹（无意义）

**GoalFlow的解决方案**：
- 先生成**目标点** $g$n- 再以 $g$ 为条件生成轨迹 $\tau$
- 不同的 $g$ 对应不同的模态

---

### 2.2 GoalFlow架构

#### 整体流程

```
输入: BEV特征 F_bev, Ego状态 s_ego
  ↓
[Goal Point Constructor]
  ↓
候选目标点: {g_1, g_2, ..., g_K}
  ↓
[Goal Point Scorer] → 选择最优 g*
  ↓
[Conditional Flow Matching]
  条件: g*, F_bev
  ↓
生成轨迹: {τ_1, τ_2, ..., τ_M}
  ↓
[Trajectory Scorer] → 选择最优 τ*
  ↓
输出: 最终轨迹 τ*
```

---

### 2.3 Goal Point生成机制

#### 2.3.1 Goal Point Vocabulary（目标点词汇表）

**构建方法**：
在BEV空间中预定义一个**密集的候选点集合** $\mathcal{V} = \{g_1, g_2, ..., g_K\}$

- 通常 $K = 128$ 或 $256$
- 覆盖车辆前方可达区域
- 例如：前方50米范围内，每隔1米设置一个点

**为什么不直接回归目标点？**
- 回归容易产生**模糊的平均值**
- 分类（从词汇表选择）更适合**多模态**场景

#### 2.3.2 Goal Point Scoring（目标点评分）

**评分网络**：

$$s_i = \text{MLP}([F_{\text{bev}}, g_i, s_{\text{ego}}])$$

其中：
- $F_{\text{bev}}$ - BEV特征（场景信息）
- $g_i$ - 候选目标点坐标
- $s_{\text{ego}}$ - 自车状态（位置、速度、航向）
- $s_i \in [0, 1]$ - 目标点 $g_i$ 的得分

**评分标准**（论文中的创新）：

$$\hat{\delta}_i^{\text{final}} = \hat{\delta}_i^{\text{dac}} \cdot \hat{\delta}_i^{\text{dis}}$$

1. **可达性得分** $\hat{\delta}_i^{\text{dac}}$（Drivable Area Constraint）
   - 目标点是否在可行驶区域内？
   - 使用BEV特征中的道路分割信息

2. **距离得分** $\hat{\delta}_i^{\text{dis}}$（Distance to Ground Truth）
   - 训练时：与真实目标点的距离
   - 推理时：与场景语义的匹配度

**训练损失**：

$$\mathcal{L}_{\text{goal}} = \text{BCE}(\hat{\delta}_i^{\text{final}}, \delta_i^{\text{gt}})$$

其中 $\delta_i^{\text{gt}}$ 是真实标签（最接近GT终点的候选点为1，其他为0）

---

### 2.4 条件Flow Matching（核心数学）

#### 2.4.1 条件生成公式

给定目标点 $g^*$ 和场景特征 $c = [F_{\text{bev}}, s_{\text{ego}}]$，生成轨迹 $\tau$。

**轨迹表示**：
$$\tau = \{(x_1, y_1), (x_2, y_2), ..., (x_N, y_N)\}$$

通常 $N=6$（未来3秒，每0.5秒一个点）

**条件概率路径**：

$$\tau_t = (1-t) \tau_0 + t \tau_1$$

其中：
- $\tau_0 \sim \mathcal{N}(0, I)$ - 噪声轨迹
- $\tau_1$ - 真实轨迹
- $t \in [0, 1]$

**条件速度场**：

$$v_\theta(\tau_t, t, g^*, c) = \tau_1 - \tau_0$$

#### 2.4.2 网络架构

**输入编码**：
```python
# 伪代码
def encode_condition(g_star, F_bev, s_ego):
    # 目标点编码
    g_embed = MLP(g_star)  # [2] -> [256]
    
    # BEV特征提取
    bev_embed = CNN(F_bev)  # [H, W, C] -> [256]
    
    # 自车状态编码
    ego_embed = MLP(s_ego)  # [6] -> [256]
    
    # 融合
    condition = concat([g_embed, bev_embed, ego_embed])  # [768]
    return condition
```

**速度场网络**：
```python
def velocity_network(tau_t, t, condition):
    # 时间编码（Sinusoidal）
    t_embed = sinusoidal_embedding(t)  # [128]
    
    # 轨迹编码
    tau_embed = MLP(tau_t.flatten())  # [N*2] -> [256]
    
    # 拼接所有条件
    x = concat([tau_embed, t_embed, condition])  # [1152]
    
    # Transformer或MLP
    v = Transformer(x)  # [1152] -> [N*2]
    
    return v.reshape(N, 2)  # 输出速度场
```

#### 2.4.3 训练目标

$$\mathcal{L}_{\text{traj}} = \mathbb{E}_{t, \tau_0, \tau_1, g^*, c} \left[ \|v_\theta(\tau_t, t, g^*, c) - (\tau_1 - \tau_0)\|^2 \right]$$

**采采样 $t \sim \mathcal{U}(0, 1)$
- 从数据集采样真实轨迹 $\tau_1$
- 采样噪声 $\tau_0 \sim \mathcal{N}(0, I)$
- 计算中间状态 $\tau_t = (1-t)\tau_0 + t\tau_1$

---

### 2.5 推理过程（单步vs多步）

#### 多步推理（标准）

```python
def generate_trajectory(g_star, F_bev, s_ego, n_steps=20):
    # 初始化噪声
    tau = torch.randn(N, 2)  # 噪声轨迹
    
    # 时间步
    dt = 1.0 / n_steps
    
    for i in range(n_steps):
        t = i * dt
        # 预测速度
        v = velocity_network(tau, t, [g_star, F_bev, s_ego])
        # 欧拉积分
        tau = tau + v * dt
    
    return tau
```

#### 单步推理（GoalFlow的优势）

```python
def generate_trajectory_fast(g_star, F_bev, s_ego):
    # 初始化噪声
    tau_0 = torch.randn(N, 2)
    
    # 单步预测
    v = velocity_network(tau_0, t=0, [g_star, F_bev, s_ego])
    tau_1 = tau_0 + v  # dt=1
    
    return tau_1
```

**论文实验结果**：
- 20步推理：PDMS = 90.3
- 1步推理：PDMS = 88.7（仅下降1.6%！）

---

### 2.6 轨迹选择机制

#### Shadow Trajectory（影子轨迹）

**问题**：即使选择了最优目标点 $g^*$，生成的轨迹仍可能有偏差。

**解决方案**：
1. 对于选中的 $g^*$，生成 $M$ 条轨迹（通过不同的噪声初始化）
2. 对每条轨迹评分
3. 选择得分最高的轨迹

**轨迹评分公式**：

$$\text{score}(\tau) = \alpha \cdot d(\tau_{\text{end}}, g^*) + \beta \cdot \text{collision}(\tau) + \gamma \cdot \text{smoothness}(\tau)$$

其中：
- $d(\tau_{\text{end}}, g^*迹终点与目标点的距离
- $\text{collision}(\tau)$ - 碰撞检测（与障碍物的距离）
- $\text{smoothness}(\tau)$ - 轨迹平滑度（曲率）

---

## 第三部分：复现要点

### 3.1 数据准备

**输入数据**：
```python
{
    'bev_features': torch.Tensor,  # [B, C, H, W] - BEV特征图
    'ego_state': torch.Tensor,     # [B, 6] - (x, y, vx, vy, yaw, yaw_rate)
    'gt_trajectory': torch.Tensor, # [B, N, 2] - 真实轨迹
    'gt_goal': torch.Tensor,       # [B, 2] - 真实目标点
}
```

**BEV特征提取**：
- 使用现有的BEV编码器（如LSS、BEVFormer）
- 或简化版：直接用俯视图投影

### 3.2 网络实现

#### Goal Point Constructor

```python
class GoalPointConstructor(nn.Module):
    def __init__(self, K=128):
        super().__init__()
        self.K = K
        # 构建目标点词汇表
        self.vocabulary = self.build_vocabulary()
        # 评分网络
        self.scorer = nn.Sequential(
            nn.Linear(256 + 2 + 6, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def build_vocabulary(self):
        # 在前方50m范围内均匀采样
        x = torch.linspace(0, 50, 16)
        y = torch.linspace(-10, 10, 8)
        xx, yy = torch.meshgrid(x, y)
        vocab = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        return vocab  # [K, 2]
    
    def forward(self, bev_feat, ego_state):
        B = bev_feat.shape[0]
        K = self.K
        
        # BEV特征池化
        bev_pooled = F.adaptive_avg_pool2d(bev_feat, 1).squeeze()  # [B, 256]
        
        # 扩展到所有候选点
        bev_expanded = bev_pooled.unsqueeze(1).expand(B, K, -1)  # [B, K, 256]
        ego_expanded = ego_state.unsqueeze(1).expand(B, K, -1)   # [B, K, 6]
        vocab_expanded = self.vocabulary.unsqueeze(0).expand(B, K, -1)  # [B, K, 2]
        
        # 拼接并评分
        x = torch.cat([nded, vocab_expanded, ego_expanded], dim=-1)
        scores = self.scorer(x).squeeze(-1)  # [B, K]
        
        # 选择最高分的目标点
        best_idx = scores.argmax(dim=-1)  # [B]
        best_goal = self.vocabulary[best_idx]  # [B, 2]
        
        return best_goal, scores
```

#### Flow Matching Trajectory Generator

```python
class FlowMatchingGenerator(nn.Module):
    def __init__(self, N=6):
        super().__init__()
        self.N = N  # 轨迹点数
        
        # 时间编码
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(128),
            nn.Linear(128, 256)
        )
        
      # 条件编码
        self.goal_embed = nn.Linear(2, 256)
        self.bev_encoder = nn.Linear(256, 256)
        self.ego_embed = nn.Linear(6, 256)
        
        # 速度场网络
        self.velocity_net = nn.Sequential(
            nn.Linear(N*2 + 256*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, N*2)
        )
    
    def forward(self, tau_t, t, goal, bev_feat, ego_state):
        B = tau_t.shape[0]
        
        # 编码所有条件
        t_emb = self.time_embed(t)  # [B, 256]
        g_emb = self.goal_embed(goal)  # [B, 256]
        bev_emb = self.bev_encoder(F.adaptive_avg_pool2d(bev_feat, 1).squeeze())
        ego_emb = self.ego_embed(ego_state)  # [B, 256]
        
        # 轨迹展平
        tau_flat = tau_t.reshape(B, -1)  # [B, N*2]
        
        # 拼接
        x = torch.cat([tau_flat, t_emb, g_emb, bev_emb, ego_emb], dim=-1)
        
        # 预测速度
        v = self.velocity_net(x)  # [B, N*2]
        v = v.reshape(B, self.N, 2)
        
        return v
    
    @torch.no_grad()
    def sample(self, goal, bev_feat, ego_state, n_steps=1):
        B = goal.shape[0]
        
        # 初始化噪声
        tau = torch.randn(B, self.N, 2, device=goal.device)
        
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=goal.device)
            v = self.forward(tau, t, goal, bev_feat, ego_state)
            tau = tau + v * dt
        
        return tau
```

### 3.3 训练流程

```python
def train_step(model, batch, optimizer):
    # 解包数据
    bev_feat = batch['bev_features']
    ego_state = batch['ego_state']
    gt_traj = batch['gt_trajectory']
    gt_goal = batch['gt_goal']
    
    B = bev_feat.shape[0]
    
    # 1. Goal Point预测
    pred_goal, goal_scoredel.goal_constructor(bev_feat, ego_state)
       # Goal Point损失
    goal_loss = F.mse_loss(pred_goal, gt_goal)
    
    # 2. Flow Matching训练
    # 随机采样时间
    t = torch.rand(B, device=bev_feat.device)
    
    # 采样噪声
    tau_0 = torch.randn_like(gt_traj)
    tau_1 = gt_traj
    
    # 插值
    tau_t = (1 - t.view(B, 1, 1)) * tau_0 + t.view(B, 1, 1) * tau_1
    
    # 真实速度
    v_true = tau_1 - tau_0
    
    # 预测速度
    v_pred = model.flow_generator(tau_t, t, gt_goal, bev_feat, ego_state)
    
    # Flow Matching损失
    fm_loss = F.mse_loss(v_pred, v_true)
    
    # 总损失
    loss = goal_loss + fm_loss
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return {
        'loss': loss.item(),
        'goal_loss': goal_loss.item(),
        'fm_loss': fm_loss.item()
    }
```

### 3.4 推理流程

```python
@torch.no_grad()
def inference(model, bev_feat, ego_state, n_candidates=5):
    # 1. 生成目标点
    goal, scores = model.goal_constructor(bev_feat, ego_state)
    
    # 2. 生成多条候选轨迹
    trajectories = []
    for _ in range(n_candidates):
        tau = model.flow_generator.sample(
            goal, bev_feat, ego_state, n_steps=1
        )
        trajectories.u)
    
    trajectories = torch.stack(trajectories, dim=1)  # [B, M, N, 2]
    
    # 3. 轨迹评分和选择
    scores = score_trajectories(trajectories, goal)
    best_idx = scores.argmax(dim=1)
    best_traj = trajectories[torch.arange(B), best_idx]
    
    return best_traj, goal
```

---

## 第四部分：关键技巧和注意事项

### 4.1 训练技巧

1. **时间采样策略**
   - 均匀采样 $t \sim \mathcal{U}(0, 1)$
   - 或重要性采样：更多采样 $t \in [0.3, 0.7]$

2. **数据增强**
   - 轨迹旋转
   - 轨迹平移
   - 速度扰动

3. **损失权重**
   ```python
   loss = λ_goal * goal_loss + λ_fm * fm_loss
   # 推荐: λ_goal=1.0, λ_fm=10.0
   ```

### 4.2 推理优化

1. **单步vs多步**
   - 训练时：多步（n_steps=20）
   - 推理时：单步（n_steps=1）
   - 性能下降<2%，速度提升20倍

2. **批量生成**
   - 并行生成多条轨迹
   - GPU利用率更高

### 4.3 常见问题

**Q: 为什么需要Goal Point Vocabulary？**
A: 直接回归会产生模糊的平均值，分类更适合多模态。

**Q: Flow Matching比Diffusion快多少？**
A: 理论上20倍（1步 vs 20步），实际约10-15倍（考虑网络开销）。

**Q: 如何处理动态障碍物？**
A: 在轨迹评分阶段加入碰撞检测项。

---

## 总结

**GoalFlow的核心贡献**：
1. **Goal Point机制** - 解决多模态轨迹的模糊问题
2. **Flow Matching** - 比扩散模型快20倍
3. **两阶段生成** - 先目标后轨迹，更符合人类规划逻辑

**数学精髓**：
- 用**直线路径**替代**曲折路径**
- 用**速度场**替代**噪声预测**
- 用**ODE**替代**SDE**

**复现关键**：
- BEV特征提取
- Goal Point Vocabulary设计
- 条件Flow Matching网络
- 单步推理优化
