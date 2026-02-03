# 脱困倒车功能实现方案

## 场景需求

**目标**：让模型在必要的脱困场景学会倒车，其他情况正常前行。

**关键挑战**：
1. 如何让模型识别"需要倒车"的场景？
2. 如何生成合理的倒车轨迹？
3. 如何确保只在必要时倒车？

---

## 方案一：扩展Goal Point Vocabulary（推荐）

### 1.1 构建包含后方的Goal Point词汇表

```python
class GoalPointVocabulary:
    def __init__(self):
        self.forward_goals = self.build_forward_goals()
        self.backward_goals = self.build_backward_goals()
        self.vocabulary = torch.cat([self.forward_goals, self.backward_goals])
    
    def build_forward_goals(self):
        """前方目标点：0-50米"""
        x = torch.linspace(0, 50, 20)  # 前方0-50米
        y = torch.linspace(-10, 10, 10)  # 左右±10米
        xx, yy = torch.meshgrid(x, y)
        forward = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        return forward  # [200, 2]
    
    def build_backward_goals(self):
        """后方目标点：-5到0米"""
        x = torch.linspace(-5, 0, 5)  # 后方0-5米
        y = torch.linspace(-5, 5, 5)  # 左右±5米
        xx, yy = torch.meshgrid(x, y)
        backward = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        return backward  # [25, 2]
    
    def get_vocabulary(self):
        """返回完整词汇表：225个候选点"""
        return self.vocabulary  # [225, 2]
```

**关键点**：
- 前方点：200个（主要区域）
- 后方点：25个（脱困用）
- 总共：225个候选Goal Point

### 1.2 训练数据准备

**必须包含倒车场景的数据**：

```python
# 数据集示例
dataset = [
    # 正常前行场景
    {
        'bev_features': ...,
        'ego_state': [x, y, vx, vy, yaw, yaw_rate],
        'trajectory': [[0,0], [1,0], [2,0], ...],  # 前行轨迹
        'goal_point': [10, 0],  # 前方目标点
        'scenario_type': 'forward'
    },
    
    # 脱困倒车场景
    {
        'bev_features': ...,  # BEV显示前方被堵
        'ego_state': [x, y, 0, 0, yaw, 0],  # 车辆静止
        'trajectory': [[0,0], [-0.5,0], [-1,0], ...],  # 倒车轨迹
        'goal_point': [-3, 0],  # 后方目标点
        'scenario_type': 'reverse'
    },
    
    # 更多场景...
]
```

**数据收集策略**：
1. **正常数据**：80-90%（前行场景）
2. **倒车数据**：10-20%（脱困场景）
3. **关键**：倒车数据必须包含"为什么需要倒车"的场景信息

### 1.3 Goal Point选择器训练

```python
class GoalPointSelector(nn.Module):
    def __init__(self, vocab_size=225):
        super().__init__()
        self.vocabulary = GoalPointVocabulary()
        
        # 场景编码器
        self.scene_encoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # 评分网络
        self.scorer = nn.Sequential(
            nn.Linear(128 + 6 + 2, 512),  # scene + ego + goal
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, bev_feat, ego_state):
        B = bev_feat.shape[0]
        vocab = self.vot_vocabulary()  # [225, 2]
        K = vocab.shape[0]
        
        # 编码场景
        scene_feat = self.scene_encoder(bev_feat)  # [B, 128]
        
        # 扩展到所有候选点
        scene_expanded = scene_feat.unsqueeze(1).expand(B, K, -1)
        ego_expanded = ego_state.unsqueeze(1).expand(B, K, -1)
        vocab_expanded = vocab.unsqueeze(0).expand(B, K, -1)
        
        # 拼接并评分
        x = torch.cat([scene_expanded, ego_expanded, vocab_expanded], dim=-1)
        scores = self.scorer(x).squeeze(-1)  # [B, K]
        
        # 选择最高分的Goal Point
        best_idx = scores.am=-1)
        best_goal = vocab[best_idx]
        
        return best_goal, scores
```

**训练目标**：

```python
def train_goal_selector(model, batch):
    bev_feat = batch['bev_features']
    ego_state = batch['ego_state']
    gt_goal = batch['goal_point']  # 可能在前方或后方！
    
    # 预测Goal Point
    pred_goal, scores = model(bev_feat, ego_state)
    
    # 损失1：Goal Point回归损失
    goal_loss = F.mse_loss(pred_goal, gt_goal)
    
    # 损失2：分类损失（哪个候选点最接近GT）
    vocab = model.vocabulary.get_vocabulary()
    distances = torch.norm(vocab.unsqueeze(0) - gt_goal.unsqueeze(1))
    gt_label = distances.argmin(dim=-1)
    cls_loss = F.cross_entropy(scores, gt_label)
    
    loss = goal_loss + cls_loss
    return loss
```

---

## 方案二：显式的场景分类器（更可控）

### 2.1 两阶段决策

```python
class ReverseAwareGoalSelector(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 阶段1：场景分类器
        self.scenario_classifier = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128 + 6, 256),
            nn.ReLU(),
            nn.Linear(2 # 2类：forward / reverse
            nn.Softmax(dim=-1)
        )
        
        # 阶段2：Goal Point选择器（分别为前进和倒车）
        self.forward_goal_selector = GoalPointSelector(
            vocabulary=build_forward_goals()
        )
        self.reverse_goal_selector = GoalPointSelector(
            vocabulary=build_backward_goals()
        )
    
    def forward(self, bev_feat, ego_state):
        # 1. 判断场景类型
        scene_input = torch.cat([
            F.adaptive_avg_pool2d(bev_feat, 1).flatten(1),
            ego_state
        ], dim=-1)
        scenario_probs = self.scenario_classifier(scene_input)  # [B, 2]
        
        # 2. 根据场景类型选择Goal Point
        forward_prob = scenario_probs[:, 0]
        reverse_prob = scenario_probs[:, 1]
        
        # 前进Goal
        forward_goal, forward_scores = self.forward_goal_selector(
            bev_feat, ego_state
        )
        
        # 倒车Goal
        reverse_goal, reverse_scores = self.reverse_goal_selector(
            bev_feat, ego_state
        )
        
        # 加权融合（或硬选择）
        if self.training:
            # 训练时：软融合
            goal = forward_prob.unsqueeze(-1) * forward_goal + \
                   reverse_prob.unsqueeze(-1) * reverse_goal
        else:
            # 推理时：硬选择
            is_reverse = (reverse_prob > 0.5).float()
            goal = is_reverse.unsqueeze(-1) * reverse_goal + \
                   (1 - is_reverse.unsqueeze(-1)) * forward_goal
        
        return goal, scenario_probs
```

### 2.2 场景分类器训练

```python
def train_scenario_classifier(model, batch):
    bev_feat = batch['bev_features']
    ego_state = batch['ego_state']
    scenario_type = batch['scenario_type']  # 'forward' or 'reverse'
    
    # 转换为标签
    labels = (scenario_type == 'reverse').long()  # 0=forward, 1=reverse
    
    # 预测
    _, scenario_probs = model(bev_feat, ego_state)
    
    # 分类损失
    loss = F.cross_entropy(scenario_probs, labels)
    
    return loss
```

### 2.3 脱困场景的识别特征

**BEV特征中的关键信号**：

```python
def extract_reverse_indicators(bev_feat):
    """
    提取需要倒车的场景特征
    """
    indicators = {}
    
    # 1. 前方障碍物密度
    front_region = bev_feat[:, :, :30, :]  # 前方30米
    indicators['front_obstacle_density'] = front_region.mean()
    
    # 2. 后方可行驶空间
    rear_region = bev_feat[:, :, 30:, :]  # 后方区域
    indicators['rear_drivable_space'] = rear_region.mean()
    
    # 3. 车辆速度（静止或低速）
    indicators['is_stopped'] = (ego_state[:, 2:4].norm(dim=-1) < 0.1)
    
    # 4. 前方通道宽度
    indicators['front_passage_width'] = compute_passage_width(front_region)
    
    return indicators

def should_reverse(indicators):
    """
    判断是否需要倒车
    """
    return (
        indicators['front_obstacle_density'] > 0.8 and  # 前方堵塞
        indicators['rear_drivable_space'] > 0.3 and     # 后方有空间
        indicators['is_stopped'] and                     # 车辆静止
        indicators['front_passage_width'] < 2.0          # 前方通道太窄
    )
```

---

## 方案三：条件Goal Point生成（最灵活）

### 3.1 直接回归Goal Point（不用词汇表）

```python
class ConditionalGoalGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 场景编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Goal Point生成器
        self.goal_generator = nn.Sequential(
            nn.Linear(128 + 6, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # 直接输出(x, y)
        )
        
        # 方向约束（可选）
        self.direction_head = nn.Sequential(
            nn.Linear(128 + 6, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0=forward, 1=reverse
        )
    
    def forward(self, bev_feat, ego_state):
        # 编码场景
        scene_feat = self.encoder(bev_feat)
        x = torch.cat([scene_feat, ego_state], dim=-1)
        
        # 预测方向
        direction = self.direction_head(x)  # [B, 1]
        
        # 生成Goal Point
        goal_raw = self.goal_generator(x)  # [B, 2]
        
        # 根据方向约束Goal Point的x坐标
        goal_x = goal_raw[:, 0]
        goal_y = goal_raw[:, 1]
        
        # 如果是倒车，强制x为负
        goal_x = torch.where(
            direction.squeeze() > 0.5,
            -torch.abs(goal_x),  # 倒车：x < 0
            torch.abs(goal_x)    # 前进：x > 0
        )
        
        goal = torch.stack([goal_x, goal_y], dim=-1)
        
        return goal, direction
```

---

## 数据增强策略

### 4.1 合成倒车场景

如果倒车数据不足，可以通过数据增强生成：

```python
def augment_reverse_scenario(forward_sample):
    """
    从前进场景合成倒车场景
    """
    # 1. 翻转BEV特征（前后对调）
    bev_feat_reversed = torch.flip(forward_sample['bev_features'], di2])
    
    # 2. 翻转轨迹（x坐标取反）
    trajectory = forward_sample['trajectory']
    trajectory_reversed = trajectory.clone()
    trajectory_reversed[:, 0] = -trajectory_reversed[:, 0]
    
    # 3. 翻转Goal Point
    goal = forward_sample['goal_point']
    goal_reversed = goal.clone()
    goal_reversed[0] = -goal_reversed[0]
    
    # 4. 调整自车状态（速度取反）
    ego_state = forward_sample['ego_state'].clone()
    ego_state[2:4] = -ego_state[2:4]  # vx, vy取反
    
    return {
        'bev_features': bev_feat_reversed,
        'ego_state': ego_state,
        'trajectory': trajectory_reversed,
        'goal_point': goal_reversed,
        'scenario_type': 'reverse'
    }
```

### 4.2 添加"前方堵塞"标记

```python
def add_blockage_to_bev(bev_feat, blockage_region='front'):
    """
    在BEV特征中添加障碍物，模拟堵塞场景
    """
    bev_modified = bev_feat.clone()
    
    if blockage_region == 'front':
        # 在前方5-10米区域添加高密度障碍物
        bev_modified[:, :, 10:20, :] = 1.0  # 障碍物通道
    
    return bev_modified
```

---

## 完整训练流程

```python
def train_reverse_aware_model(model, dataloader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
or epoch in range(epochs):
        for batch in dataloader:
            # 1. 获取数据
            bev_feat = batch['bev_features']
            ego_state = batch['ego_state']
            gt_trajectory = batch['trajectory']
            gt_goal = batch['goal_point']
            scenario_type = batch['scenario_type']
            
            # 2. Goal Point预测
            pred_goal, scenario_probs = model.goal_selector(
                bev_feat, ego_state
            )
            
            # 3. 轨迹生成
            x_0 = torch.randn_like(gt_trajectory)
            t = torch.rand(bev_feat.shape[0])
            x_t = (1 - t.view(-1, 1, 1)) * x_0 + t.view(-1, 1, 1) * gt_trajectory
            
            v_true = gt_trajectory - x_0
            v_pred = model.flow_generator(x_t, t, pred_goal, bev_feat)
            
            # 4. 计算损失
            # Goal Point损失
            goal_loss = F.mse_loss(pred_goal, gt_goal)
            
            # 场景分类损失
            scenario_labels = (scenario_type == 'reverse').long()
            scenario_loss = F.cross_entropy(scenario_probs, scenario_labels)
            
            # Flow Matching损失
            fm_loss = F.mse_loss(v_pred, v_true)
            
            # 总损失
            loss = goal_loss + 0.5 * scenario_loss + 10.0 * fm_loss
            
            # 5. 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 6. 日志
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}")
                print(f"  Goal Loss: {goal_loss.item():.4f}")
                print(f"  Scenario Loss: {scenario_loss.item():.4f}")
                print(f"  FM Loss: {fm_loss.item():.4f}")
                
                # 统计倒车预测准确率
                pred_reverse = (scenario_probs[:, 1] > 0.5)
                gt_reverse = (scenario_type == 'reverse')
                acc = (pred_reverse == gt_reverse).float().mean()
                print(f"  Reverse Detection Acc: {acc.item():.2%}")
```

---

## 推理时的使用

```python
@torch.no_grad()
def inference_with_reverse(model, bev_feat, ego_state):
    """
    支持倒车的推理
    """
    # 1. 预测Goal Point和场景类型
    goal, scenario_probs = model.goal_selector(bev_feat, ego_state)
    
    # 2. 判断是否倒车
    is_reverse = scenario_probs[0, 1] > 0.5
    
    print(f"Goal Point: {goal}")
    print(f"Scenario: {'REVERSE' if is_reverse else 'FORWARD'}")
    print(f"Confidence: {scenario_probs[0, 1 if is_reverse else 0]:.2%}")
    
    # 3. 生成轨迹
    x_0 = torch.randn(1, 6, 2)
    trajectory = model.flow_generator.sample(x_0, goal, bev_feat)
    
    # 4. 验证轨迹方向
    if is_reverse:
        # 倒车轨迹：x坐标应该递减
        assert trajectory[0, -1, 0] < trajectory[0, 0, 0], "倒车轨迹方向错误！"
    else:
        # 前进轨迹：x坐标应该递增
        assert trajectory[0, -1, 0] > trajectory[0, 0, 0], "前进轨迹方向错误！"
    
    return trajectory, goal, is_reverse
```

---

## 安全约束

### 倒车安全检查

```python
def safe_reverse_check(trajectory, bev_feat, ego_state):
    """
    倒车安全检查
    """
    checks = {}
    
    # 1. 后方障碍物检测
ear_region = bev_feat[:, :, 30:, :]  # 后方区域
    checks['rear_clear'] = (rear_region.max() < 0.5)
    
    # 2. 倒车距离限制
    reverse_distance = -trajectory[:, -1, 0].item()  # 负数表示后退
    checks['distance_safe'] = (reverse_distance < 5.0)  # 最多倒5米
    
    # 3. 倒车速度限制
    max_reverse_speed = 2.0  # m/s
    trajectory_speed = torch.norm(
        trajectory[:, 1:] - trajectory[:, :-1], dim=-1
    ).max()
    checks['speed_safe'] = (trajectory_speed < max_reverse_speed)
    
    # 4. 综合判断
    is_safe = all(checks.values())
    
    return is_safe, checks
```

---

## 总结

### 关键要点

1. **Goal Poocabulary必须包含后方点**
   - 前方点：主要区域（80-90%）
   - 后方点：脱困区域（10-20%）

2. **训练数据必须包含倒车场景**
   - 真实倒车数据
   - 或通过数据增强合成

3. **场景识别是关键**
   - 方案1：隐式学习（Goal Point选择器自动学会）
   - 方案2：显式分类器（更可控）
   - 方案3：条件生成（最灵活）

4. **安全约束不可少**
   - 后方障碍物检测
   - 倒车距离限制
   - 速度限制

### 推荐方案

**方案二（显式场景分类器）** 最适合您的需求：
- ✅ 可解释性强
- ✅ 可控性好
- ✅ 容易调试
- ✅ 可以设置阈值控制倒车触发条件

### 下一步

您想：
1. 实现其中一个方案的代码？
2. 讨论如何收集/标注倒车数据？
3. 设计更复杂的脱困策略（如三点掉头）？
