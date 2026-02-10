"""
GoalFlowMatcher 完整测试套件

测试覆盖：
1. 模型创建
2. 前向传播
3. 损失计算
4. 反向传播
5. 轨迹生成（Euler）
6. 轨迹生成（RK4）
7. 多轨迹生成
8. 简单训练循环
9. 不同配置测试
10. 边界情况测试
"""

import torch
import torch.nn as nn
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.goal_flow_matcher import GoalFlowMatcher


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_1_model_creation():
    """测试1：模型创建"""
    print("=" * 70)
    print("测试1：模型创建")
    print("=" * 70)
    
    model = GoalFlowMatcher(
        traj_dim=2,
        num_traj_points=6,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        scene_channels=64
    )
    
    print(f"✓ 模型创建成功")
    print(f"  总参数量: {count_parameters(model):,}")
    print(f"  d_model: {model.d_model}")
    print(f"  轨迹点数: {model.num_traj_points}")
    
    return model


def test_2_forward_pass(model):
    """测试2：前向传播"""
    print("\n" + "=" * 70)
    print("测试2：前向传播")
    print("=" * 70)
    
    B, T, C, H, W = 4, 6, 64, 32, 32
    
    x_t = torch.randn(B, T, 2)
    goal = torch.randn(B, 2)
    scene = torch.randn(B, C, H, W)
    t = torch.rand(B)
    
    print(f"输入形状:")
    print(f"  x_t: {x_t.shape}")
    print(f"  goal: {goal.shape}")
    print(f"  scene: {scene.shape}")
    print(f"  t: {t.shape}")
    
    v_pred = model(x_t, goal, scene, t)
    
    print(f"\n输出形状:")
    print(f"  v_pred: {v_pred.shape}")
    
    assert v_pred.shape == (B, T, 2), f"输出形状错误！期望 {(B, T, 2)}, 得到 {v_pred.shape}"
    assert not torch.isnan(v_pred).any(), "输出包含 NaN！"
    assert not torch.isinf(v_pred).any(), "输出包含 Inf！"
    
    print(f"✓ 前向传播测试通过")
    print(f"  速度范围: [{v_pred.min().item():.4f}, {v_pred.max().item():.4f}]")


def test_3_loss_computation(model):
    """测试3：损失计算"""
    print("\n" + "=" * 70)
    print("测试3：损失计算")
    print("=" * 70)
    
    B, T, C, H, W = 4, 6, 64, 32, 32
    
    x_0 = torch.randn(B, T, 2)
    x_1 = torch.randn(B, T, 2)
    goal = torch.randn(B, 2)
    scene = torch.randn(B, C, H, W)
    
    print(f"输入形状:")
    print(f"  x_0 (噪声): {x_0.shape}")
    print(f"  x_1 (目标): {x_1.shape}")
    print(f"  goal: {goal.shape}")
    print(f"  scene: {scene.shape}")
    
    loss = model.compute_loss(x_0, x_1, goal, scene)
    
    print(f"\n损失:")
    print(f"  loss: {loss.item():.6f}")
    print(f"  loss shape: {loss.shape}")
    
    assert loss.dim() == 0, "损失应该是标量！"
    assert loss.item() >= 0, "损失应该非负！"
    assert not torch.isnan(loss), "损失是 NaN！"
    
    print(f"✓ 损失计算测试通过")


def test_4_backward_pass(model):
    """测试4：反向传播"""
    print("\n" + "=" * 70)
    print("测试4：反向传播")
    print("=" * 70)
    
    B, T, C, H, W = 2, 6, 64, 32, 32
    
    x_0 = torch.randn(B, T, 2)
    x_1 = torch.randn(B, T, 2)
    goal = torch.randn(B, 2)
    scene = torch.randn(B, C, H, W)
    
    loss = model.compute_loss(x_0, x_1, goal, scene)
    loss.backward()
    
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    has_nan = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
    
    print(f"梯度检查:")
    print(f"  所有参数都有梯度: {has_grad}")
    print(f"  梯度包含 NaN: {has_nan}")
    
    assert has_grad, "部分参数没有梯度！"
    assert not has_nan, "梯度包含 NaN！"
    
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"  梯度范数统计:")
    print(f"    最小: {min(grad_norms):.6f}")
    print(f"    最大: {max(grad_norms):.6f}")
    print(f"    平均: {sum(grad_norms)/len(grad_norms):.6f}")
    
    print(f"✓ 反向传播测试通过")


def test_5_generation_euler(model):
    """测试5：生成轨迹（Euler方法）"""
    print("\n" + "=" * 70)
    print("测试5：生成轨迹（Euler方法）")
    print("=" * 70)
    
    model.eval()
    
    B, C, H, W = 2, 64, 32, 32
    goal = torch.randn(B, 2)
    scene = torch.randn(B, C, H, W)
    
    for num_steps in [1, 5, 10]:
        trajectory = model.generate(goal, scene, num_steps=num_steps, method='euler')
        
        print(f"\nnum_steps={num_steps}:")
        print(f"  输出形状: {trajectory.shape}")
        print(f"  轨迹范围: [{trajectory.min().item():.4f}, {trajectory.max().item():.4f}]")
        
        assert trajectory.shape == (B, 6, 2), f"输出形状错误！期望 {(B, 6, 2)}, 得到 {trajectory.shape}"
        assert not torch.isnan(trajectory).any(), "轨迹包含 NaN！"
    
    print(f"\n✓ Euler 生成测试通过")
    
    model.train()  # 恢复训练模式


def test_6_generation_rk4(model):
    """测试6：生成轨迹（RK4方法）"""
    print("\n" + "=" * 70)
    print("测试6：生成轨迹（RK4方法）")
    print("=" * 70)
    
    model.eval()
    
    B, C, H, W = 2, 64, 32, 32
    goal = torch.randn(B, 2)
    scene = torch.randn(B, C, H, W)
    
    trajectory = model.generate(goal, scene, num_steps=5, method='rk4')
    
    print(f"输出形状: {trajectory.shape}")
    print(f"轨迹范围: [{trajectory.min().item():.4f}, {trajectory.max().item():.4f}]")
    
    assert trajectory.shape == (B, 6, 2), f"输出形状错误！期望 {(B, 6, 2)}, 得到 {trajectory.shape}"
    assert not torch.isnan(trajectory).any(), "轨迹包含 NaN！"
    
    print(f"✓ RK4 生成测试通过")
    
    model.train()  # 恢复训练模式


def test_7_multiple_generation(model):
    """测试7：生成多条候选轨迹"""
    print("\n" + "=" * 70)
    print("测试7：生成多条候选轨迹")
    print("=" * 70)
    
    model.eval()
    
    B, C, H, W = 2, 64, 32, 32
    num_samples = 10
    
    goal = torch.randn(B, 2)
    scene = torch.randn(B, C, H, W)
    
    trajectories = model.generate_multiple(goal, scene, num_samples=num_samples, num_steps=1)
    
    print(f"输出形状: {trajectories.shape}")
    print(f"  期望: (B={B}, num_samples={num_samples}, T=6, D=2)")
    
    assert trajectories.shape == (B, num_samples, 6, 2), f"输出形状错误！期望 {(B, num_samples, 6, 2)}, 得到 {trajectories.shape}"
    assert not torch.isnan(trajectories).any(), "轨迹包含 NaN！"
    
    # 检查多样性
    variance = trajectories.var(dim=1).mean()
    print(f"  轨迹方差（多样性指标）: {variance.item():.6f}")
    assert variance > 0, "生成的轨迹没有多样性！"
    
    print(f"✓ 多轨迹生成测试通过")
    
    model.train()  # 恢复训练模式


def test_8_training_loop(model):
    """测试8：简单训练循环"""
    print("\n" + "=" * 70)
    print("测试8：简单训练循环")
    print("=" * 70)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    B, T, C, H, W = 4, 6, 64, 32, 32
    
    print("训练 10 步...")
    losses = []
    
    for step in range(10):
        x_0 = torch.randn(B, T, 2)
        x_1 = torch.randn(B, T, 2)
        goal = torch.randn(B, 2)
        scene = torch.randn(B, C, H, W)
        
        optimizer.zero_grad()
        loss = model.compute_loss(x_0, x_1, goal, scene)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (step + 1) % 5 == 0:
            print(f"  Step {step+1}: Loss = {loss.item():.6f}")
    
    print(f"\n损失统计:")
    print(f"  初始损失: {losses[0]:.6f}")
    print(f"  最终损失: {losses[-1]:.6f}")
    print(f"  平均损失: {sum(losses)/len(losses):.6f}")
    
    print(f"✓ 训练循环测试通过")


def test_9_different_configs():
    """测试9：不同配置"""
    print("\n" + "=" * 70)
    print("测试9：不同配置")
    print("=" * 70)
    
    configs = [
        {"d_model": 128, "nhead": 4, "num_encoder_layers": 2, "name": "小模型"},
        {"d_model": 256, "nhead": 8, "num_encoder_layers": 4, "name": "中等模型"},
        {"d_model": 512, "nhead": 8, "num_encoder_layers": 6, "name": "大模型"},
    ]
    
    for i, config in enumerate(configs):
        name = config.pop("name")
        print(f"\n配置 {i+1} - {name}:")
        print(f"  d_model={config['d_model']}, nhead={config['nhead']}, layers={config['num_encoder_layers']}")
        
        model = GoalFlowMatcher(**config, scene_channels=64)
        print(f"  参数量: {count_parameters(model):,}")
        
        # 快速测试
        x_t = torch.randn(2, 6, 2)
        goal = torch.randn(2, 2)
        scene = torch.randn(2, 64, 32, 32)
        t = torch.rand(2)
        
        v_pred = model(x_t, goal, scene, t)
        assert v_pred.shape == (2, 6, 2), f"配置 {i+1} 输出形状错误！"
        assert not torch.isnan(v_pred).any(), f"配置 {i+1} 输出包含 NaN！"
        
        print(f"  ✓ 配置 {i+1} 测试通过")
    
    print(f"\n✓ 不同配置测试通过")


def test_10_edge_cases():
    """测试10：边界情况"""
    print("\n" + "=" * 70)
    print("测试10：边界情况")
    print("=" * 70)
    
    model = GoalFlowMatcher(scene_channels=64)
    model.eval()
    
    # 测试 batch_size=1
    print("\n测试 batch_size=1:")
    x_t = torch.randn(1, 6, 2)
    goal = torch.randn(1, 2)
    scene = torch.randn(1, 64, 32, 32)
    t = torch.rand(1)
    
    v_pred = model(x_t, goal, scene, t)
    assert v_pred.shape == (1, 6, 2), "batch_size=1 输出形状错误！"
    assert not torch.isnan(v_pred).any(), "batch_size=1 输出包含 NaN！"
    print("  ✓ batch_size=1 通过")
    
    # 测试 t=0
    print("\n测试 t=0:")
    t = torch.zeros(2)
    v_pred = model(torch.randn(2, 6, 2), torch.randn(2, 2), torch.randn(2, 64, 32, 32), t)
    assert not torch.isnan(v_pred).any(), "t=0 输出包含 NaN！"
    print("  ✓ t=0 通过")
    
    # 测试 t=1
    print("\n测试 t=1:")
    t = torch.ones(2)
    v_pred = model(torch.randn(2, 6, 2), torch.randn(2, 2), torch.randn(2, 64, 32, 32), t)
    assert not torch.isnan(v_pred).any(), "t=1 输出包含 NaN！"
    print("  ✓ t=1 通过")
    
    # 测试不同轨迹点数
    print("\n测试不同轨迹点数:")
    for T in [3, 6, 12]:
        x_t = torch.randn(2, T, 2)
        goal = torch.randn(2, 2)
        scene = torch.randn(2, 64, 32, 32)
        t = torch.rand(2)
        
        v_pred = model(x_t, goal, scene, t)
        assert v_pred.shape == (2, T, 2), f"T={T} 输出形状错误！"
        print(f"  ✓ T={T} 通过")
    
    print(f"\n✓ 边界情况测试通过")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("开始运行 GoalFlowMatcher 完整测试套件")
    print("=" * 70)
    
    try:
        # 创建模型
        model = test_1_model_creation()
        
        # 基础功能测试
        test_2_forward_pass(model)
        test_3_loss_computation(model)
        test_4_backward_pass(model)
        
        # 生成测试
        test_5_generation_euler(model)
        test_6_generation_rk4(model)
        test_7_multiple_generation(model)
        
        # 训练测试
        test_8_training_loop(model)
        
        # 配置和边界测试
        test_9_different_configs()
        test_10_edge_cases()
        
        print("\n" + "=" * 70)
        print("✓✓✓ 所有测试通过！✓✓✓")
        print("=" * 70)
        print("\n恭喜！GoalFlowMatcher 实现正确，可以进行下一步了！")
        
    except Exception as e:
        print(f"\n" + "=" * 70)
        print(f"❌ 测试失败: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
