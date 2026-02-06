"""
Conditional Flow Matcher
实现 Flow Matching 的核心算法
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


class ConditionalFlowMatcher:
    """
    条件流匹配器
    
    实现 Optimal Transport (OT) Flow 和 Conditional Flow Matching (CFM) Loss
    """
    
    def __init__(self, sigma: float = 0.0):
        """
        初始化 Flow Matcher
        
        Args:
            sigma: 可选的噪声标准差（用于 Conditional Flow Matching）
                   sigma=0 对应 Optimal Transport Flow
        """
        self.sigma = sigma
    
    def sample_ot_flow(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样 Optimal Transport Flow
        
        OT Flow 定义：
            x_t = (1-t) * x_0 + t * x_1
            v_t = x_1 - x_0  (常数速度场)
        
        Args:
            x_0: 起点（噪声） (B, D)
            x_1: 终点（数据） (B, D)
            t: 时间 (B,) 或标量，如果为 None 则随机采样
        
        Returns:
            x_t: 插值点 (B, D)
            v_t: 速度场 (B, D)
        """
        batch_size = x_0.shape[0]
        
        # 如果没有提供 t，随机采样
        if t is None:
            t = torch.rand(batch_size, device=x_0.device, dtype=x_0.dtype)
        
        # 确保 t 的形状正确
        if t.dim() == 0:  # 标量
            t = t.expand(batch_size)
        
        # 扩展 t 的维度以便广播
        t_expanded = t.view(-1, *([1] * (x_0.dim() - 1)))
        
        # OT Flow: 线性插值
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        
        # 速度场：常数
        v_t = x_1 - x_0
        
        return x_t, v_t
    
    def compute_cfm_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算 Conditional Flow Matching Loss
        
        Loss = E_{t, x_0, x_1} [ ||v_θ(x_t, t) - (x_1 - x_0)||² ]
        
        Args:
            model: 速度场网络 v_θ(x, t)
            x_0: 起点（噪声） (B, D)
            x_1: 终点（数据） (B, D)
            t: 时间 (B,) 或 None
        
        Returns:
            loss: 标量损失
        """
        # 采样 OT Flow
        x_t, v_true = self.sample_ot_flow(x_0, x_1, t)
        
        # 获取时间张量
        if t is None:
            batch_size = x_0.shape[0]
            t = torch.rand(batch_size, device=x_0.device, dtype=x_0.dtype)
        
        # 网络预测速度
        v_pred = model(x_t, t)
        
        # MSE Loss
        loss = torch.mean((v_pred - v_true) ** 2)
        
        return loss
    
    def sample_trajectory(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        num_steps: int = 50,
        method: str = 'rk4'
    ) -> Tuple[torch.Tensor, list]:
        """
        使用训练好的模型生成轨迹
        
        Args:
            model: 训练好的速度场网络
            x_0: 初始噪声 (B, D)
            num_steps: ODE 求解步数
            method: 'euler' 或 'rk4'
        
        Returns:
            x_1: 生成的数据 (B, D)
            trajectory: 完整轨迹
        """
        try:
            from .ode_solver import ODESolver
        except ImportError:
            from ode_solver import ODESolver
        
        # 创建速度场函数
        def velocity_field(x, t):
            return model(x, t)
        
        # 求解 ODE
        solver = ODESolver(method=method)
        x_1, trajectory = solver.solve(
            velocity_field,
            x_0,
            num_steps=num_steps,
            return_trajectory=True
        )
        
        return x_1, trajectory


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("测试 Conditional Flow Matcher")
    print("=" * 60)
    
    # 创建简单的速度场网络（用于测试）
    class SimpleVelocityNet(nn.Module):
        """简单的速度场网络（用于测试）"""
        def __init__(self, dim=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim + 1, 64),  # x + t
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, dim)
            )
        
        def forward(self, x, t):
            # 拼接 x 和 t
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            xt = torch.cat([x, t], dim=-1)
            return self.net(xt)
    
    # 测试1：OT Flow 采样
    print("\n[测试1] OT Flow 采样")
    flow_matcher = ConditionalFlowMatcher()
    
    x_0 = torch.tensor([[0., 0.], [1., 1.]])
    x_1 = torch.tensor([[10., 10.], [5., 5.]])
    t = torch.tensor([0.5, 0.5])
    
    x_t, v_t = flow_matcher.sample_ot_flow(x_0, x_1, t)
    
    print(f"x_0: {x_0}")
    print(f"x_1: {x_1}")
    print(f"t: {t}")
    print(f"x_t (插值点): {x_t}")
    print(f"v_t (速度): {v_t}")
    
    # 验证
    expected_x_t = 0.5 * x_0 + 0.5 * x_1
    expected_v_t = x_1 - x_0
    print(f"\n验证:")
    print(f"  x_t 正确: {torch.allclose(x_t, expected_x_t)}")
    print(f"  v_t 正确: {torch.allclose(v_t, expected_v_t)}")
    
    # 测试2：CFM Loss 计算
    print("\n[测试2] CFM Loss 计算")
    model = SimpleVelocityNet(dim=2)
    
    # 随机数据
    x_0 = torch.randn(4, 2)
    x_1 = torch.randn(4, 2)
    
    loss = flow_matcher.compute_cfm_loss(model, x_0, x_1)
    print(f"Loss: {loss.item():.6f}")
    print(f"Loss 形状: {loss.shape}")
    print(f"Loss 是标量: {loss.dim() == 0}")
    
    # 测试3：训练一个简单的模型
    print("\n[测试3] 训练简单模型")
    
    # 创建简单的数据：从原点到目标点
    def create_simple_data(batch_size=32):
        x_0 = torch.randn(batch_size, 2) * 0.1  # 接近原点的噪声
        x_1 = torch.randn(batch_size, 2) * 2 + 5  # 目标分布
        return x_0, x_1
    
    model = SimpleVelocityNet(dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("训练 100 步...")
    for step in range(100):
        x_0, x_1 = create_simple_data()
        
        optimizer.zero_grad()
        loss = flow_matcher.compute_cfm_loss(model, x_0, x_1)
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 20 == 0:
            print(f"  Step {step+1}: Loss = {loss.item():.6f}")
    
    # 测试4：生成轨迹
    print("\n[测试4] 生成轨迹")
    model.eval()
    
    with torch.no_grad():
        x_0 = torch.zeros(2, 2)  # 从原点开始
        x_1, trajectory = flow_matcher.sample_trajectory(
            model, x_0, num_steps=20, method='rk4'
        )
    
    print(f"起点: {x_0[0].tolist()}")
    print(f"终点: {x_1[0].tolist()}")
    print(f"轨迹长度: {len(trajectory)}")
    print(f"轨迹前3个点:")
    for i in range(min(3, len(trajectory))):
        print(f"  t={i/20:.2f}: {trajectory[i][0].tolist()}")
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)
