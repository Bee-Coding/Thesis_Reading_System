"""
Trajectory Selector 测试套件

测试覆盖：
1. 距离评分计算
2. 进度评分计算
3. 评分归一化
4. 最终评分计算
5. 最优轨迹选择
6. Shadow Trajectories 生成（可选）
7. 端到端测试
8. ADE/FDE 计算
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.trajectory_selector import TrajectorySelector


def test_1_model_creation():
    """测试1：模型创建"""
    print("=" * 70)
    print("测试1：模型创建")
    print("=" * 70)
    
    selector = TrajectorySelector(
        lambda_dis=1.0,
        lambda_pg=1.0,
        use_shadow=False,
        normalize=True
    )
    
    print(f"✓ 模型创建成功")
    print(f"  lambda_dis: {selector.lambda_dis}")
    print(f"  lambda_pg: {selector.lambda_pg}")
    
    return selector


def test_2_distance_score(selector):
    """测试2：距离评分计算"""
    print("\n" + "=" * 70)
    print("测试2：距离评分计算")
    print("=" * 70)
    
    B, N, T = 4, 10, 6
    
    trajectories = torch.randn(B, N, T, 2)
    gt_trajectory = torch.randn(B, T, 2)
    
    print(f"输入形状:")
    print(f"  trajectories: {trajectories.shape}")
    print(f"  gt_trajectory: {gt_trajectory.shape}")
    
    f_dis = selector.compute_distance_score(trajectories, gt_trajectory)
    
    print(f"\n输出形状:")
    print(f"  f_dis: {f_dis.shape}")
    
    assert f_dis.shape == (B, N), f"输出形状错误！期望 {(B, N)}, 得到 {f_dis.shape}"
    assert not torch.isnan(f_dis).any(), "输出包含 NaN！"
    assert (f_dis >= 0).all(), "距离评分应该非负！"
    
    print(f"✓ 距离评分测试通过")
    print(f"  评分范围: [{f_dis.min().item():.4f}, {f_dis.max().item():.4f}]")


def test_3_progress_score(selector):
    """测试3：进度评分计算"""
    print("\n" + "=" * 70)
    print("测试3：进度评分计算")
    print("=" * 70)
    
    B, N, T = 4, 10, 6
    
    trajectories = torch.randn(B, N, T, 2)
    goal = torch.randn(B, 2)
    
    f_pg = selector.compute_progress_score(trajectories, goal)
    
    print(f"输出形状: {f_pg.shape}")
    
    assert f_pg.shape == (B, N), f"输出形状错误！期望 {(B, N)}, 得到 {f_pg.shape}"
    assert not torch.isnan(f_pg).any(), "输出包含 NaN！"
    assert (f_pg >= 0).all(), "进度评分应该非负！"
    
    print(f"✓ 进度评分测试通过")
    print(f"  评分范围: [{f_pg.min().item():.4f}, {f_pg.max().item():.4f}]")


def test_4_normalize_scores(selector):
    """测试4：评分归一化"""
    print("\n" + "=" * 70)
    print("测试4：评分归一化")
    print("=" * 70)
    
    B, N = 4, 10
    
    scores = torch.randn(B, N) * 10  # 随机评分
    
    normalized = selector.normalize_scores(scores)
    
    print(f"原始评分范围: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
    print(f"归一化后范围: [{normalized.min().item():.4f}, {normalized.max().item():.4f}]")
    
    assert normalized.shape == (B, N), f"输出形状错误！"
    assert (normalized >= 0).all() and (normalized <= 1).all(), "归一化后应该在 [0, 1] 范围！"
    
    print(f"✓ 归一化测试通过")


def test_5_final_score(selector):
    """测试5：最终评分计算"""
    print("\n" + "=" * 70)
    print("测试5：最终评分计算")
    print("=" * 70)
    
    B, N, T = 4, 10, 6
    
    trajectories = torch.randn(B, N, T, 2)
    goal = torch.randn(B, 2)
    gt_trajectory = torch.randn(B, T, 2)
    
    scores = selector.compute_final_score(trajectories, goal, gt_trajectory)
    
    print(f"输出形状: {scores.shape}")
    print(f"评分范围: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
    
    assert scores.shape == (B, N), f"输出形状错误！"
    assert not torch.isnan(scores).any(), "输出包含 NaN！"
    
    print(f"✓ 最终评分测试通过")


def test_6_select_best(selector):
    """测试6：最优轨迹选择"""
    print("\n" + "=" * 70)
    print("测试6：最优轨迹选择")
    print("=" * 70)
    
    B, N, T = 4, 10, 6
    
    trajectories = torch.randn(B, N, T, 2)
    scores = torch.randn(B, N)
    
    best_traj, best_idx = selector.select_best_trajectory(trajectories, scores)
    
    print(f"输出形状:")
    print(f"  best_trajectory: {best_traj.shape}")
    print(f"  best_indices: {best_idx.shape}")
    
    assert best_traj.shape == (B, T, 2), f"最优轨迹形状错误！期望 {(B, T, 2)}, 得到 {best_traj.shape}"
    assert best_idx.shape == (B,), f"索引形状错误！"
    assert (best_idx >= 0).all() and (best_idx < N).all(), "索引超出范围！"
    
    print(f"✓ 最优轨迹选择测试通过")


def test_7_forward_pass(selector):
    """测试7：前向传播"""
    print("\n" + "=" * 70)
    print("测试7：前向传播")
    print("=" * 70)
    
    B, N, T = 4, 10, 6
    
    trajectories = torch.randn(B, N, T, 2)
    goal = torch.randn(B, 2)
    gt_trajectory = torch.randn(B, T, 2)
    
    best_traj, scores = selector(trajectories, goal, gt_trajectory, return_scores=True)
    
    print(f"输出形状:")
    print(f"  best_trajectory: {best_traj.shape}")
    print(f"  scores: {scores.shape if scores is not None else None}")
    
    assert best_traj.shape == (B, T, 2), f"最优轨迹形状错误！"
    assert scores is not None, "return_scores=True 时应该返回 scores！"
    assert scores.shape == (B, N), f"评分形状错误！"
    
    print(f"✓ 前向传播测试通过")


def test_8_ade_fde(selector):
    """测试8：ADE/FDE 计算"""
    print("\n" + "=" * 70)
    print("测试8：ADE/FDE 计算")
    print("=" * 70)
    
    B, N, T = 4, 10, 6
    
    pred_traj = torch.randn(B, N, T, 2)
    gt_traj = torch.randn(B, T, 2)
    
    ade = selector.compute_ade(pred_traj, gt_traj)
    fde = selector.compute_fde(pred_traj, gt_traj)
    
    print(f"ADE 形状: {ade.shape}")
    print(f"FDE 形状: {fde.shape}")
    print(f"ADE 范围: [{ade.min().item():.4f}, {ade.max().item():.4f}]")
    print(f"FDE 范围: [{fde.min().item():.4f}, {fde.max().item():.4f}]")
    
    assert ade.shape == (B, N), f"ADE 形状错误！"
    assert fde.shape == (B, N), f"FDE 形状错误！"
    assert (ade >= 0).all(), "ADE 应该非负！"
    assert (fde >= 0).all(), "FDE 应该非负！"
    
    print(f"✓ ADE/FDE 测试通过")


def test_9_collision_score(selector):
    """测试9：碰撞评分计算"""
    print("\n" + "=" * 70)
    print("测试9：碰撞评分计算")
    print("=" * 70)
    
    B, N, T, M = 4, 10, 6, 5
    
    trajectories = torch.randn(B, N, T, 2)
    obstacles = torch.randn(B, M, 2)
    
    f_col = selector.compute_collision_score(trajectories, obstacles)
    
    print(f"输出形状: {f_col.shape}")
    print(f"评分范围: [{f_col.min().item():.4f}, {f_col.max().item():.4f}]")
    
    assert f_col.shape == (B, N), f"输出形状错误！"
    assert not torch.isnan(f_col).any(), "输出包含 NaN！"
    
    print(f"✓ 碰撞评分测试通过")


def test_10_dac_score(selector):
    """测试10：DAC评分计算"""
    print("\n" + "=" * 70)
    print("测试10：DAC评分计算")
    print("=" * 70)
    
    B, N, T = 4, 10, 6
    H, W = 32, 32
    
    trajectories = torch.randn(B, N, T, 2) * 10  # 缩放到合理范围
    drivable_area = torch.randint(0, 2, (B, H, W)).float()
    
    f_dac = selector.compute_dac_score(trajectories, drivable_area)
    
    print(f"输出形状: {f_dac.shape}")
    print(f"评分范围: [{f_dac.min().item():.4f}, {f_dac.max().item():.4f}]")
    
    assert f_dac.shape == (B, N), f"输出形状错误！"
    assert not torch.isnan(f_dac).any(), "输出包含 NaN！"
    assert (f_dac >= 0).all() and (f_dac <= 1).all(), "DAC评分应该在 [0, 1] 范围！"
    
    print(f"✓ DAC评分测试通过")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("开始运行 Trajectory Selector 完整测试套件")
    print("=" * 70)
    
    try:
        selector = test_1_model_creation()
        test_2_distance_score(selector)
        test_3_progress_score(selector)
        test_4_normalize_scores(selector)
        test_5_final_score(selector)
        test_6_select_best(selector)
        test_7_forward_pass(selector)
        test_8_ade_fde(selector)
        test_9_collision_score(selector)
        test_10_dac_score(selector)
        
        print("\n" + "=" * 70)
        print("✓✓✓ 所有测试通过！✓✓✓")
        print("=" * 70)
        print("\n恭喜！Trajectory Selector 实现正确，可以进行下一步了！")
        
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
