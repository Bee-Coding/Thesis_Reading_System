"""
Matcher 诊断脚本
测试 Matcher 在使用真实目标点（gt_goal）时的性能

目的：
- 如果用 gt_goal 的 ADE 很低（< 1.0），说明 Matcher 训练成功，问题在 Scorer
- 如果用 gt_goal 的 ADE 还是很高（> 5.0），说明 Matcher 训练失败
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.goal_flow_matcher import GoalFlowMatcher
from models.trajectory_selector import TrajectorySelector
from data.toy_goalflow_dataset import ToyGoalFlowDataset
from config.matcher_config import MatcherConfig


def compute_ade(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
    """计算 Average Displacement Error"""
    distances = torch.norm(pred_traj - gt_traj, dim=-1)
    ade = distances.mean(dim=-1)
    return ade


def compute_fde(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
    """计算 Final Displacement Error"""
    pred_end = pred_traj[..., -1, :]
    gt_end = gt_traj[..., -1, :]
    fde = torch.norm(pred_end - gt_end, dim=-1)
    return fde


def diagnose_matcher_with_gt_goal(matcher_checkpoint_path: str,
                                   data_path: str,
                                   num_candidates: int = 10,
                                   device: str = 'cpu'):
    """
    使用真实目标点测试 Matcher
    
    Args:
        matcher_checkpoint_path: Matcher checkpoint 路径
        data_path: 数据路径
        num_candidates: 候选轨迹数量
        device: 设备
    
    Returns:
        results: 诊断结果字典
    """
    print("=" * 70)
    print("Matcher Diagnosis: Testing with Ground Truth Goals")
    print("=" * 70)
    
    # 1. 加载 Matcher 模型
    print("\n[1/4] Loading Matcher model...")
    matcher_config = MatcherConfig()
    matcher = GoalFlowMatcher(
        traj_dim=matcher_config.traj_dim,
        num_traj_points=matcher_config.num_traj_points,
        d_model=matcher_config.hidden_dim,
        nhead=matcher_config.num_heads,
        num_encoder_layers=matcher_config.num_layers,
        dim_feedforward=matcher_config.hidden_dim * 4,
        dropout=matcher_config.dropout
    ).to(device)
    
    checkpoint = torch.load(matcher_checkpoint_path, map_location=device)
    matcher.load_state_dict(checkpoint['model_state_dict'])
    matcher.eval()
    print(f"   ✓ Matcher loaded from: {matcher_checkpoint_path}")
    print(f"   ✓ Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    
    # 2. 加载测试数据
    print("\n[2/4] Loading test dataset...")
    test_dataset = ToyGoalFlowDataset(data_path, split='val')
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    print(f"   ✓ Test dataset loaded: {len(test_dataset)} samples")
    
    # 3. 初始化 Selector
    selector = TrajectorySelector()
    
    # 4. 测试 Matcher（使用 gt_goal）
    print("\n[3/4] Testing Matcher with ground truth goals...")
    all_ade = []
    all_fde = []
    all_ade_multimodal = []
    all_fde_multimodal = []
    
    pbar = tqdm(test_loader, desc='Testing')
    
    max_samples = 50  # Limit to 50 samples for faster diagnosis
    sample_count = 0
    
    with torch.no_grad():
        for batch in pbar:
            scene_feature = batch['bev_feature'].to(device)  # (B, C, H, W)
            gt_goal = batch['goal'].to(device)  # (B, 2)
            gt_traj = batch['trajectory'].to(device)  # (B, T, 2)
            drivable_area = batch['drivable_area'].to(device)  # (B, H, W)
            B = scene_feature.shape[0]
            
            for i in range(B):
                if sample_count >= max_samples:
                    break
                sample_count += 1
                # 使用真实目标点生成轨迹
                goal = gt_goal[i:i+1]  # (1, 2)
                scene = scene_feature[i:i+1]  # (1, C, H, W)
                gt_trajectory = gt_traj[i:i+1]  # (1, T, 2)
                drivable = drivable_area[i:i+1]  # (1, H, W)
                
                # 生成多条候选轨迹
                trajectories = matcher.generate_multiple(
                    goal, scene, 
                    num_samples=num_candidates, 
                    num_steps=10
                )  # (1, num_candidates, T, 2)
                
                # 使用 Selector 选择最优轨迹
                scores = selector.compute_final_score(
                    trajectories, goal, gt_trajectory,
                    obstacle=None, drivable_area=drivable
                )  # (1, num_candidates)
                
                best_trajectory, _ = selector.select_best_trajectory(trajectories, scores)
                # (1, T, 2)
                
                # 计算单模态指标（最优轨迹）
                ade = compute_ade(best_trajectory, gt_trajectory)
                fde = compute_fde(best_trajectory, gt_trajectory)
                all_ade.append(ade.item())
                all_fde.append(fde.item())
                
                # 计算多模态指标（所有候选轨迹）
                gt_expanded = gt_trajectory.expand(1, num_candidates, -1, -1)  # (1, N, T, 2)
                ade_all = compute_ade(trajectories, gt_expanded)  # (1, N)
                fde_all = compute_fde(trajectories, gt_expanded)  # (1, N)
                
                min_ade = ade_all.min().item()
                min_fde = fde_all.min().item()
                all_ade_multimodal.append(min_ade)
                all_fde_multimodal.append(min_fde)
                
                pbar.set_postfix({
                    'ADE': f'{np.mean(all_ade):.3f}',
                    'FDE': f'{np.mean(all_fde):.3f}'
                })
            
            if sample_count >= max_samples:
                break
    
    # 5. 计算统计结果
    print("\n[4/4] Computing statistics...")
    results = {
        'avg_ade': np.mean(all_ade),
        'avg_fde': np.mean(all_fde),
        'min_ade': np.mean(all_ade_multimodal),
        'min_fde': np.mean(all_fde_multimodal),
        'std_ade': np.std(all_ade),
        'std_fde': np.std(all_fde),
    }
    
    # 6. 打印诊断结果
    print("\n" + "=" * 70)
    print("DIAGNOSIS RESULTS (Using Ground Truth Goals)")
    print("=" * 70)
    print(f"Average ADE (Single-modal):     {results['avg_ade']:.4f} ± {results['std_ade']:.4f} m")
    print(f"Average FDE (Single-modal):     {results['avg_fde']:.4f} ± {results['std_fde']:.4f} m")
    print(f"Min ADE (Multi-modal):          {results['min_ade']:.4f} m")
    print(f"Min FDE (Multi-modal):          {results['min_fde']:.4f} m")
    print("=" * 70)
    
    # 7. 诊断结论
    print("\n" + "=" * 70)
    print("DIAGNOSIS CONCLUSION")
    print("=" * 70)
    
    # 加载数据统计信息
    data = np.load(data_path)
    trajectories = data['trajectories']
    traj_lengths = np.linalg.norm(np.diff(trajectories, axis=1), axis=-1).sum(axis=1)
    avg_traj_length = np.mean(traj_lengths)
    
    print(f"\nData Context:")
    print(f"  - Average trajectory length: {avg_traj_length:.2f} m")
    print(f"  - Trajectory range: [{trajectories.min():.2f}, {trajectories.max():.2f}]")
    
    print(f"\nMatcher Performance (with GT goals):")
    print(f"  - ADE: {results['avg_ade']:.4f} m ({results['avg_ade']/avg_traj_length*100:.1f}% of traj length)")
    print(f"  - FDE: {results['avg_fde']:.4f} m")
    
    # 判断标准
    if results['avg_ade'] < 1.0:
        print("\n✅ MATCHER IS WORKING WELL!")
        print("   → The problem is likely in the Scorer (goal selection)")
        print("   → Next step: Run diagnose_scorer.py to check goal selection accuracy")
    elif results['avg_ade'] < 3.0:
        print("\n⚠️  MATCHER IS PARTIALLY WORKING")
        print("   → Matcher learned something but not optimal")
        print("   → Consider retraining with better hyperparameters")
        print("   → Also check Scorer performance")
    else:
        print("\n❌ MATCHER TRAINING FAILED!")
        print("   → Even with ground truth goals, ADE is too high")
        print("   → Matcher did not learn to generate trajectories properly")
        print("\n   Possible causes:")
        print("   1. Training loss did not converge")
        print("   2. Learning rate too high/low")
        print("   3. Network capacity insufficient")
        print("   4. ODE solver steps too few")
        print("   5. Initial noise std too large")
        print("\n   Recommended actions:")
        print("   1. Check training logs for loss curves")
        print("   2. Visualize some predictions to see if they're random")
        print("   3. Try retraining with adjusted hyperparameters")
    
    print("=" * 70)
    
    return results


def main():
    """主函数"""
    # 参数设置
    matcher_checkpoint_path = 'checkpoints/matcher/best.pth'
    data_path = 'data/toy_data.npz'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_candidates = 10
    
    print(f"\nDevice: {device}")
    print(f"Matcher checkpoint: {matcher_checkpoint_path}")
    print(f"Data path: {data_path}")
    print(f"Number of candidates: {num_candidates}\n")
    
    # 检查文件
    if not os.path.exists(matcher_checkpoint_path):
        print(f"❌ ERROR: Matcher checkpoint not found: {matcher_checkpoint_path}")
        return
    if not os.path.exists(data_path):
        print(f"❌ ERROR: Data file not found: {data_path}")
        return
    
    # 运行诊断
    results = diagnose_matcher_with_gt_goal(
        matcher_checkpoint_path=matcher_checkpoint_path,
        data_path=data_path,
        num_candidates=num_candidates,
        device=device
    )
    
    print("\n✓ Diagnosis completed!")


if __name__ == "__main__":
    main()
