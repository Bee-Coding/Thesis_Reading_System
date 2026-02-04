import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

# 支持两种导入方式：作为模块导入 或 直接运行测试
try:
    from .time_embedding import SinusoidalEmbedding
except ImportError:
    from time_embedding import SinusoidalEmbedding

class VelocityFieldMLP(nn.Module):
    def __init__(self, 
                 state_dim: int, 
                 cond_dim: int, 
                 time_dim: int=128, 
                 hidden_dim: int=256,
                 num_hidden_layers: int=4,
                 activation: str='relu',
                 dropout: float=0.1):
        super().__init__()
        self.state_dim = state_dim      # 12维
        self.cond_dim = cond_dim        # ego_state_dim + goal_dim + obstacle_dim + bev_dim = 5 + 4 + 10*15 + 128
        self.time_dim = time_dim        # 128 
        self.hidden_dim = hidden_dim    # 256
        self.num_hidden_layers = num_hidden_layers
        self.activation = activation
        self.dropout = dropout

        self.time_embedding = SinusoidalEmbedding(embedding_dim=time_dim)
        # 速度场可以是任意实数（有正有负），所以输出层不能使用激活函数
        self.MLP = nn.Sequential(
            nn.Linear(state_dim + cond_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout))
        for i in range(num_hidden_layers - 1):
            self.MLP.add_module(f"layer_{i+1}", nn.Linear(hidden_dim, hidden_dim))
            self.MLP.add_module(f"activation_{i+1}", nn.ReLU())
            self.MLP.add_module(f"dropout_{i+1}", nn.Dropout(dropout))
        self.MLP.add_module("output", nn.Linear(hidden_dim, state_dim))

    def forward(self, state:torch.Tensor, cond:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, state_dim] 当前状态
            cond: [B, cond_dim] 条件
            t: [B] 时间标量，范围[0, 1]
        Returns:
            [B, state_dim] 速度场
        """
        time_embedding = self.time_embedding(t)
        x = torch.cat([state, cond, time_embedding], dim=-1)
        x = self.MLP(x)
        return x


if __name__ == "__main__":
    print("=" * 60)
    print("VelocityFieldMLP 测试")
    print("=" * 60)
    
    # 维度配置
    state_dim = 12  # 6个点 × 2 (x, y)
    ego_state_dim = 5  # [vx, vy, heading, ax, ay]
    goal_dim = 4  # [x, y, heading, v_goal]
    obstacle_per_dim = 15  # 每个障碍物15维
    N_obs = 10  # 最多10个障碍物
    bev_dim = 128  # BEV特征
    
    cond_dim = ego_state_dim + goal_dim + (obstacle_per_dim * N_obs) + bev_dim
    print(f"\n维度配置:")
    print(f"  state_dim: {state_dim}")
    print(f"  cond_dim: {cond_dim} = {ego_state_dim}(ego) + {goal_dim}(goal) + {obstacle_per_dim * N_obs}(obstacles) + {bev_dim}(bev)")
    print(f"  time_dim: 128 (编码后)")
    
    # 创建模型
    model = VelocityFieldMLP(
        state_dim=state_dim,
        cond_dim=cond_dim,
        time_dim=128,
        hidden_dim=256,
        num_hidden_layers=4,
        dropout=0.1
    )
    
    print(f"\n模型结构:")
    print(f"  隐藏层数量: 4")
    print(f"  每层神经元: 256")
    print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试1: 单样本
    print(f"\n{'='*60}")
    print("测试1: 单样本前向传播")
    print(f"{'='*60}")
    
    batch_size = 1
    state = torch.randn(batch_size, state_dim)
    cond = torch.randn(batch_size, cond_dim)
    t = torch.rand(batch_size)  # [0, 1]
    
    print(f"输入:")
    print(f"  state shape: {state.shape}")
    print(f"  cond shape: {cond.shape}")
    print(f"  t shape: {t.shape}, value: {t.item():.4f}")
    
    output = model(state, cond, t)
    
    print(f"输出:")
    print(f"  velocity shape: {output.shape}")
    print(f"  velocity range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    assert output.shape == (batch_size, state_dim), f"输出维度错误! 期望 {(batch_size, state_dim)}, 得到 {output.shape}"
    print(f"✓ 维度检查通过!")
    
    # 测试2: 批量样本
    print(f"\n{'='*60}")
    print("测试2: 批量样本前向传播")
    print(f"{'='*60}")
    
    batch_size = 8
    state = torch.randn(batch_size, state_dim)
    cond = torch.randn(batch_size, cond_dim)
    t = torch.rand(batch_size)
    
    print(f"输入:")
    print(f"  batch_size: {batch_size}")
    print(f"  state shape: {state.shape}")
    print(f"  cond shape: {cond.shape}")
    print(f"  t shape: {t.shape}")
    
    output = model(state, cond, t)
    
    print(f"输出:")
    print(f"  velocity shape: {output.shape}")
    
    assert output.shape == (batch_size, state_dim), f"输出维度错误!"
    print(f"✓ 批量维度检查通过!")
    
    # 测试3: 梯度反向传播
    print(f"\n{'='*60}")
    print("测试3: 梯度反向传播")
    print(f"{'='*60}")
    
    output = model(state, cond, t)
    loss = output.sum()
    loss.backward()
    
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"所有参数都有梯度: {has_grad}")
    
    # 检查梯度是否为 NaN
    has_nan = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
    print(f"梯度包含 NaN: {has_nan}")
    
    assert has_grad, "部分参数没有梯度!"
    assert not has_nan, "梯度包含 NaN!"
    print(f"✓ 梯度检查通过!")
    
    # 测试4: 时间编码的连续性
    print(f"\n{'='*60}")
    print("测试4: 时间编码的连续性")
    print(f"{'='*60}")
    
    t_values = torch.linspace(0, 1, 5)
    state_fixed = torch.randn(1, state_dim).repeat(5, 1)
    cond_fixed = torch.randn(1, cond_dim).repeat(5, 1)
    
    outputs = []
    for i, t_val in enumerate(t_values):
        t_input = torch.tensor([t_val])
        out = model(state_fixed[i:i+1], cond_fixed[i:i+1], t_input)
        outputs.append(out)
        print(f"  t={t_val:.2f}: velocity norm = {out.norm().item():.4f}")
    
    # 检查输出是否随时间变化
    outputs_tensor = torch.cat(outputs, dim=0)
    variance = outputs_tensor.var(dim=0).mean()
    print(f"输出方差 (应该 > 0): {variance.item():.6f}")
    
    assert variance > 0, "输出没有随时间变化!"
    print(f"✓ 时间编码连续性检查通过!")
    
    print(f"\n{'='*60}")
    print("所有测试通过! ✓")
    print(f"{'='*60}")