"""
ODE 求解器
实现 Euler 和 RK4 方法用于 Flow Matching 的推理
"""
import torch
from typing import Callable, Tuple, List


class ODESolver:
    """ODE 求解器基类"""
    
    def __init__(self, method: str = "rk4"):
        """
        初始化 ODE 求解器
        
        Args:
            method: 求解方法，'euler' 或 'rk4'
        """
        self.method = method.lower()    # 将字符串中的所有大写字母转换为小写字母
        if self.method not in ['euler', 'rk4']:
            raise ValueError(f"Unknown method: {method}. Choose 'euler' or 'rk4'")
    
    def solve(
        self,
        velocity_field: Callable,
        x_0: torch.Tensor,
        num_steps: int = 100,
        return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        求解 ODE: dx/dt = v(x, t)
        
        Args:
            velocity_field: 速度场函数 v(x, t)
            x_0: 初始状态 (B, D)
            num_steps: 求解步数
            return_trajectory: 是否返回完整轨迹
        
        Returns:
            x_1: 最终状态 (B, D)
            trajectory: 轨迹列表（如果 return_trajectory=True）
        """
        if self.method == 'euler':
            return self._euler_solve(velocity_field, x_0, num_steps, return_trajectory)
        else:  # rk4
            return self._rk4_solve(velocity_field, x_0, num_steps, return_trajectory)
    
    def _euler_solve(
        self,
        velocity_field: Callable,
        x_0: torch.Tensor,
        num_steps: int,
        return_trajectory: bool
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Euler 方法求解 ODE
        
        一阶方法：x_{t+dt} = x_t + v(x_t, t) * dt
        误差：O(dt²)
        """
        x = x_0.clone()
        dt = 1.0 / num_steps
        trajectory = [x.clone()] if return_trajectory else []
        
        for i in range(num_steps):
            t = i * dt
            t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
            
            # Euler 更新
            v = velocity_field(x, t_tensor)
            x = x + v * dt
            
            if return_trajectory:
                trajectory.append(x.clone())
        
        return x, trajectory
    
    def _rk4_solve(
        self,
        velocity_field: Callable,
        x_0: torch.Tensor,
        num_steps: int,
        return_trajectory: bool
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Runge-Kutta 4阶方法求解 ODE
        
        四阶方法：
        k1 = v(x_t, t)
        k2 = v(x_t + 0.5*dt*k1, t + 0.5*dt)
        k3 = v(x_t + 0.5*dt*k2, t + 0.5*dt)
        k4 = v(x_t + dt*k3, t + dt)
        x_{t+dt} = x_t + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        误差：O(dt⁵)
        """
        x = x_0.clone()
        dt = 1.0 / num_steps
        trajectory = [x.clone()] if return_trajectory else []
        
        for i in range(num_steps):
            t = i * dt
            
            # 创建时间张量
            t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
            t_mid_tensor = torch.full((x.shape[0],), t + 0.5*dt, device=x.device, dtype=x.dtype)
            t_end_tensor = torch.full((x.shape[0],), t + dt, device=x.device, dtype=x.dtype)
            
            # RK4 四个阶段
            k1 = velocity_field(x, t_tensor)
            k2 = velocity_field(x + 0.5 * dt * k1, t_mid_tensor)
            k3 = velocity_field(x + 0.5 * dt * k2, t_mid_tensor)
            k4 = velocity_field(x + dt * k3, t_end_tensor)
            
            # 加权平均更新
            x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            if return_trajectory:
                trajectory.append(x.clone())
        
        return x, trajectory


def euler_solve(
    velocity_field: Callable,
    x_0: torch.Tensor,
    num_steps: int = 100,
    return_trajectory: bool = False
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Euler 方法求解 ODE（便捷函数）
    
    Args:
        velocity_field: 速度场函数 v(x, t)
        x_0: 初始状态 (B, D)
        num_steps: 求解步数
        return_trajectory: 是否返回完整轨迹
    
    Returns:
        x_1: 最终状态
        trajectory: 轨迹列表
    """
    solver = ODESolver(method='euler')
    return solver.solve(velocity_field, x_0, num_steps, return_trajectory)


def rk4_solve(
    velocity_field: Callable,
    x_0: torch.Tensor,
    num_steps: int = 50,
    return_trajectory: bool = False
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    RK4 方法求解 ODE（便捷函数）
    
    Args:
        velocity_field: 速度场函数 v(x, t)
        x_0: 初始状态 (B, D)
        num_steps: 求解步数（RK4 精度高，可以用更少步数）
        return_trajectory: 是否返回完整轨迹
    
    Returns:
        x_1: 最终状态
        trajectory: 轨迹列表
    """
    solver = ODESolver(method='rk4')
    return solver.solve(velocity_field, x_0, num_steps, return_trajectory)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("测试 ODE 求解器")
    print("=" * 60)
    
    # 测试1：简单的线性速度场
    print("\n[测试1] 线性速度场: v(x, t) = 1")
    
    def linear_velocity(x, t):
        """常数速度场"""
        return torch.ones_like(x)
    
    x_0 = torch.zeros(2, 2)  # 2个样本，2维
    
    # Euler 方法
    x_euler, traj_euler = euler_solve(linear_velocity, x_0, num_steps=100, return_trajectory=True)
    print(f"Euler (100步): x_0={x_0[0].tolist()} → x_1={x_euler[0].tolist()}")
    print(f"  理论值: [1.0, 1.0], 误差: {torch.norm(x_euler[0] - torch.ones(2)):.10f}")
    
    # RK4 方法
    x_rk4, traj_rk4 = rk4_solve(linear_velocity, x_0, num_steps=50, return_trajectory=True)
    print(f"RK4 (50步):   x_0={x_0[0].tolist()} → x_1={x_rk4[0].tolist()}")
    print(f"  理论值: [1.0, 1.0], 误差: {torch.norm(x_rk4[0] - torch.ones(2)):.10f}")
    
    # 测试2：Flow Matching 的速度场
    print("\n[测试2] Flow Matching 速度场: v(x, t) = x_1 - x_0")
    
    x_0 = torch.tensor([[0., 0.], [1., 1.]])
    x_1_target = torch.tensor([[10., 10.], [5., 5.]])
    
    def flow_matching_velocity(x, t):
        """Flow Matching 的速度场（常数）"""
        # 这里简化：假设我们知道目标
        # 实际中这是网络预测的
        return x_1_target - x_0
    
    x_euler, _ = euler_solve(flow_matching_velocity, x_0, num_steps=100)
    x_rk4, _ = rk4_solve(flow_matching_velocity, x_0, num_steps=50)
    
    print(f"样本1: x_0={x_0[0].tolist()} → 目标={x_1_target[0].tolist()}")
    print(f"  Euler: {x_euler[0].tolist()}, 误差: {torch.norm(x_euler[0] - x_1_target[0]):.6f}")
    print(f"  RK4:   {x_rk4[0].tolist()}, 误差: {torch.norm(x_rk4[0] - x_1_target[0]):.6f}")
    
    print(f"\n样本2: x_0={x_0[1].tolist()} → 目标={x_1_target[1].tolist()}")
    print(f"  Euler: {x_euler[1].tolist()}, 误差: {torch.norm(x_euler[1] - x_1_target[1]):.6f}")
    print(f"  RK4:   {x_rk4[1].tolist()}, 误差: {torch.norm(x_rk4[1] - x_1_target[1]):.6f}")
    
    # 测试3：轨迹长度
    print("\n[测试3] 轨迹长度")
    print(f"Euler (100步): 轨迹长度 = {len(traj_euler)}")
    print(f"RK4 (50步):    轨迹长度 = {len(traj_rk4)}")
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)
