"""
GoalFlow Inference Script
端到端推理脚本，包含模型加载、推理、评估和可视化
"""

import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 添加路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm
from models.goal_point_scorer import GoalPointScorer
from models.goal_flow_matcher import GoalFlowMatcher
from models.trajectory_selector import TrajectorySelector
from data.toy_goalflow_dataset import ToyGoalFlowDataset
from config.scorer_config import ScorerConfig
from config.matcher_config import MatcherConfig

# ============== 辅助函数 ========================
def compute_ade(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
    """
    计算 Average Displacement Error

    Args:
        pred_traj: (B, T, 2) 或 (T, 2)
        gt_traj: (B, T, 2) 或 (T, 2)

    Returns:
        ade: (B,) 或 标量
    """
    distances = torch.norm(pred_traj - gt_traj, dim=-1)
    ade = distances.mean(dim=-1)
    return ade


def compute_fde(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
    """
    计算 Final Displacement Error
    
    Args:
        pred_traj: (B, T, 2) 或 (T, 2)
        gt_traj: (B, T, 2) 或 (T, 2)

    Returns:
        ade: (B,) 或 标量
    """
    pred_end = pred_traj[:, -1, :]
    gt_end = gt_traj[:, -1, :]
    fde = torch.norm(pred_end - gt_end, dim=-1)
    return fde


# ================== 核心函数 =====================

def load_models(data_path: str,
                scorer_checkpoint_path: str="",
                matcher_checkpoint_path: str="",
                device: str='cpu'):
    """
    加载训练好的模型

    Args:
        score_checkpoint_path: score_checkpoint 路径
        matcher_checkpoint_path: matcher_checkpoint 路径
        data_path: 数据路径（用于获取 vocabulary ）
        device: 设备

    Returns:
        scorer: GoalPointScorer 模型
        matcher: GoalFlowMatcher 模型
        selector: TrajectorySelector
        vocabulary: 目标点词汇表(N, 2)
    """
    print("=" * 60)
    print("Inference Process Starting")
    print("=" * 60)

    scorer_config = ScorerConfig()
    matcher_config = MatcherConfig()

    # 1. 加载数据集获取vocabulary
    print("Getting vocabulary...")
    train_dataset = ToyGoalFlowDataset(data_path, split='train')
    vocabulary = train_dataset.get_vocabulary().to(device)  # (N, 2)

    # 2. 初始化 Scorer 模型
    print("Initializing score model...")
    scorer = GoalPointScorer(
                        vocabulary_size=scorer_config.vocab_size,
                        feature_dim=scorer_config.hidden_dim,  # 使用 hidden_dim 作为 feature_dim
                        hidden_dim=scorer_config.hidden_dim,
                        num_heads=scorer_config.num_heads,
                        num_layers=scorer_config.num_layers,
                        scene_in_channels=scorer_config.scene_channels,
                        kernel_size=3,
                        stride=1,
                        dropout=scorer_config.dropout).to(device)

    # 3. 加载Scorer checkpoint
    scorer_checkpoint = torch.load(scorer_checkpoint_path, map_location=device)
    scorer.load_state_dict(scorer_checkpoint['model_state_dict'])
    scorer.eval()

    # 4. 初始化 Matcher 模型
    print("Initializing matcher model...")
    matcher = GoalFlowMatcher(
                        traj_dim=matcher_config.traj_dim,
                        num_traj_points=matcher_config.num_traj_points,

                        # Transformer 配置
                        d_model=matcher_config.hidden_dim,
                        nhead=matcher_config.num_heads,
                        num_encoder_layers=matcher_config.num_layers,
                        dim_feedforward=matcher_config.hidden_dim*4,
                        dropout=matcher_config.dropout).to(device)
    
    # 5. 加载Matcher checkpoint
    matcher_checkpoint = torch.load(matcher_checkpoint_path, map_location=device)
    matcher.load_state_dict(matcher_checkpoint['model_state_dict'])
    matcher.eval()

    # 6. 初始化 Selector
    selector = TrajectorySelector()

    print("[OK] All models loaded successfully!")

    return scorer, matcher, selector, vocabulary



def inference_single_sample(scorer: nn.Module,
                            matcher: nn.Module,
                            selector: TrajectorySelector,
                            vocabulary: torch.Tensor,
                            sample: dict,
                            num_candidates: int=10,
                            device: str='cpu'):
    """
    单样本推理

    Args:
        scorer: GoalPointScorer
        matcher: GoalFlowMatcher
        selector: TrajectorySelector
        vocabulary: (N, 2)
        sample: 数据样本
        num_candidates: 候选轨迹数量
        device: 设备
    
    Returns:
        results: 包含推理结果的字典
            - best_trajectory: (T, 2)
            - selected_goal: (2,)
            - all_trajectories: (num_candidates, T, 2)
            - scores: (num_candidates,)
    """
    
    # step 0: 准备数据（添加 batch 维度）
    scene_feature = sample["bev_feature"].unsqueeze(0).to(device)    # (1, C, H, W)
    gt_goal = sample["goal"].unsqueeze(0).to(device)    # (1, 2)
    gt_traj = sample["trajectory"].unsqueeze(0).to(device)  # (1, T, 2)
    drivable_area = sample["drivable_area"].unsqueeze(0).to(device) # (1, H, W)
    B = scene_feature.shape[0]
    vocab_expend = vocabulary.unsqueeze(0).expand(B, -1, -1)  # (1, N, 2)
    # step 1: 使用 Scorer 选择目标点
    with torch.no_grad():
        pred_dis, pred_dac = scorer(vocab_expend, scene_feature)
        
        # 计算最终评分（论文公式）：δ_final = w1 * log(δ_dis) + w2 * log(δ_dac)
        # 转换到对数概率空间
        log_pred_dis = F.log_softmax(pred_dis, dim=-1)  # (B, N) - 对数概率
        log_pred_dac = torch.log(pred_dac + 1e-8)  # (B, N) - 加 epsilon 避免 log(0)
        
        # 加权组合（与训练时的损失权重一致）
        w1 = 1.0
        w2 = 0.005
        final_scores = w1 * log_pred_dis + w2 * log_pred_dac  # (B, N)
        
        best_goal_idx = final_scores.argmax(dim=-1)    # (B,)
        selected_goal = vocabulary[best_goal_idx]
    # step 2: 使用 Matcher 生成多条候选轨迹
    with torch.no_grad():
        trajectories = matcher.generate_multiple(selected_goal, 
                                                 scene_feature, 
                                                 num_samples=num_candidates, 
                                                 num_steps=10)
    # step 3: 使用 Selector 选择最优轨迹
    with torch.no_grad():
        scores = selector.compute_final_score(trajectories, 
                                          selected_goal, 
                                          gt_traj,
                                          obstacle=None,
                                          drivable_area=drivable_area)  # return (1, num_candidates)
        best_trajectory, best_indices = selector.select_best_trajectory(trajectories, scores)
    # step 4: 返回结果
    results = {
        'best_trajectory': best_trajectory.squeeze(0),
        'selected_goal': selected_goal.squeeze(0),
        'all_trajectories': trajectories.squeeze(0),
        'scores': scores.squeeze(0),
        'gt_trajectory': gt_traj.squeeze(0),
        'gt_goal': gt_goal.squeeze(0)
    }
    return results

def evaluate_on_dataset(scorer,
                        matcher,
                        selector,
                        vocabulary,
                        test_loader,
                        num_candidates,
                        device):
    """
    在测试集上评估
    Args:
        scorer, matcher, selector 三个模块
        vocabulary: (N, 2)
        test_loader: 测试数据加载器
        device: 设备
        num_candidates: 候选轨迹数据量
    
    Returns:
        results: 评估结果字典
            - avg_ade: 平均 ADE
            - avg_fde: 平均 FDE
            - min_ade: 最小 ADE(多模态)
            - min_fde: 最小 FDE(多模态)
    """
    all_ade = []
    all_fde = []
    all_ade_multimodal = [] # 用于计算 min_ade (多模态)
    all_fde_multimodal = [] # 用于计算 min_fde (多模态)
    pbar = tqdm(test_loader, desc='Test')

    # 2. 遍历测试集
    print("Evaluating on test set...")
    for batch_idx, batch in enumerate(pbar):
        B = batch['trajectory'].shape[0]
        for i in range(B):
            sample = {
                'trajectory': batch['trajectory'][i],   # (T, 2)
                'goal': batch['goal'][i],               # (2,)
                'bev_feature': batch['bev_feature'][i],     # (C, H, W)
                'drivable_area': batch['drivable_area'][i]  # (H, W)
            }
            results = inference_single_sample(scorer,
                                            matcher,
                                            selector,
                                            vocabulary,
                                            sample,
                                            num_candidates,
                                            device)
            # 3.1 单模态指标（最优轨迹 vs 真实轨迹）
            ade = compute_ade(results['best_trajectory'].unsqueeze(0), 
                                            results['gt_trajectory'].unsqueeze(0))
            fde = compute_fde(results['best_trajectory'].unsqueeze(0), 
                                            results['gt_trajectory'].unsqueeze(0))
            all_ade.append(ade.item())
            all_fde.append(fde.item())
            # 3.2 多模态指标（所有候选轨迹 vs 真实轨迹）
            # 计算每条候选轨迹的 ADE/FDE，取最小值
            gt_traj_expanded = results['gt_trajectory'].unsqueeze(0).expand(num_candidates, -1, -1)
            ade_all = compute_ade(results['all_trajectories'], gt_traj_expanded)
            fde_all = compute_fde(results['all_trajectories'], gt_traj_expanded)

            min_ade = ade_all.min().item()
            min_fde = fde_all.min().item()
            pbar.set_postfix({'ADE': f'{np.mean(all_ade):.3f}',
                              'FDE': f'{np.mean(all_fde):.3f}'})
            all_ade_multimodal.append(min_ade)
            all_fde_multimodal.append(min_fde)
        
    # 4. 计算平均值
    results = {
        'avg_ade': np.mean(all_ade),
        'avg_fde': np.mean(all_fde),
        'min_ade': np.mean(all_ade_multimodal),
        'min_fde': np.mean(all_fde_multimodal)
    }

    return results


def visualize_sample(sample,
                     best_trajectory,
                     all_trajectories,
                     selected_goal,
                     vocabulary,
                     save_path):
    """
    可视化推理结果

    Args:
        sample: 数据样本
        best_trajectory: (T, 2)
        all_trajectories: (num_candidates, T, 2)
        selected_goal: (2,)
        vocabulary: (N, 2)
        save_path: 保存路径
    """
    # 1. 提取数据并转为 numpy
    gt_trajectory = sample['trajectory'].cpu().numpy() if torch.is_tensor(sample['trajectory']) else sample['trajectory']  # (T, 2)
    gt_goal = sample['goal'].cpu().numpy() if torch.is_tensor(sample['goal']) else sample['goal']  # (2,)
    drivable_area = sample['drivable_area'].cpu().numpy() if torch.is_tensor(sample['drivable_area']) else sample['drivable_area']  # (H, W)
    
    best_trajectory = best_trajectory.cpu().numpy() if torch.is_tensor(best_trajectory) else best_trajectory  # (T, 2)
    all_trajectories = all_trajectories.cpu().numpy() if torch.is_tensor(all_trajectories) else all_trajectories  # (N, T, 2)
    selected_goal = selected_goal.cpu().numpy() if torch.is_tensor(selected_goal) else selected_goal  # (2,)
    vocabulary = vocabulary.cpu().numpy() if torch.is_tensor(vocabulary) else vocabulary  # (N, 2)
    
    num_candidates = all_trajectories.shape[0]
    
    # 2. 创建图形
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('GoalFlow Inference Result', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 3. 绘制可行驶区域（背景）
    ax.imshow(drivable_area, 
              cmap='gray', 
              alpha=0.3, 
              extent=[-50, 50, -50, 50],
              origin='lower',
              zorder=0)
    
    # 4. 绘制词汇表所有点（小灰点）
    ax.scatter(vocabulary[:, 0], vocabulary[:, 1], 
               c='gray', s=10, alpha=0.3, label='Vocabulary', zorder=1)
    
    # 5. 绘制所有候选轨迹（灰色半透明）
    for i in range(num_candidates):
        if i == 0:
            ax.plot(all_trajectories[i, :, 0], 
                   all_trajectories[i, :, 1], 
                   c='gray', alpha=0.2, linewidth=1, 
                   label=f'Candidates (N={num_candidates})', zorder=2)
        else:
            ax.plot(all_trajectories[i, :, 0], 
                   all_trajectories[i, :, 1], 
                   c='gray', alpha=0.2, linewidth=1, zorder=2)
    
    # 6. 绘制真实轨迹（绿色粗线）
    ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 
            c='green', linewidth=3, label='GT Trajectory', zorder=5, linestyle='-')
    
    # 7. 绘制最优预测轨迹（红色粗线）
    ax.plot(best_trajectory[:, 0], best_trajectory[:, 1], 
            c='red', linewidth=3, label='Best Trajectory', zorder=5, linestyle='-')
    
    # 8. 绘制起点和终点
    ax.scatter(gt_trajectory[0, 0], gt_trajectory[0, 1], 
               c='blue', s=150, marker='o', 
               edgecolors='black', linewidths=2,
               label='Start', zorder=6)
    ax.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], 
               c='orange', s=150, marker='s', 
               edgecolors='black', linewidths=2,
               label='End', zorder=6)
    
    # 9. 绘制真实目标点（绿色星星）
    ax.scatter(gt_goal[0], gt_goal[1], 
               c='green', s=300, marker='*', 
               edgecolors='black', linewidths=2, 
               label='GT Goal', zorder=10)
    
    # 10. 绘制选中的目标点（红色星星）
    ax.scatter(selected_goal[0], selected_goal[1], 
               c='red', s=300, marker='*', 
               edgecolors='black', linewidths=2, 
               label='Selected Goal', zorder=10)
    
    # 11. 计算并显示指标
    ade = np.mean(np.linalg.norm(best_trajectory - gt_trajectory, axis=1))
    fde = np.linalg.norm(best_trajectory[-1] - gt_trajectory[-1])
    goal_error = np.linalg.norm(selected_goal - gt_goal)
    
    text_str = f'ADE: {ade:.3f} m\nFDE: {fde:.3f} m\nGoal Error: {goal_error:.3f} m'
    ax.text(0.02, 0.98, text_str,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'),
            fontsize=11,
            family='monospace',
            zorder=15)
    
    # 12. 添加图例
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    # 13. 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Visualization saved to {save_path}")


def main():
    """
    主函数
    """
    # 1. 设置参数
    data_path = 'data/toy_data.npz'
    scorer_checkpoint_path = 'checkpoints/scorer/best.pth'
    matcher_checkpoint_path = 'checkpoints/matcher/best.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_candidates = 10
    
    print("="*60)
    print("GoalFlow Inference Pipeline")
    print("="*60)
    print(f"Device: {device}")
    print(f"Data path: {data_path}")
    print(f"Scorer checkpoint: {scorer_checkpoint_path}")
    print(f"Matcher checkpoint: {matcher_checkpoint_path}")
    print(f"Number of candidates: {num_candidates}")
    print("="*60)
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"[ERROR] Data file not found: {data_path}")
        return
    if not os.path.exists(scorer_checkpoint_path):
        print(f"[ERROR] Scorer checkpoint not found: {scorer_checkpoint_path}")
        return
    if not os.path.exists(matcher_checkpoint_path):
        print(f"[ERROR] Matcher checkpoint not found: {matcher_checkpoint_path}")
        return

    # 2. 加载模型
    scorer, matcher, selector, vocabulary = load_models(
        data_path=data_path,
        scorer_checkpoint_path=scorer_checkpoint_path,
        matcher_checkpoint_path=matcher_checkpoint_path,
        device=device
    )

    # 3. 加载测试数据
    print("\nLoading test dataset...")
    test_dataset = ToyGoalFlowDataset(data_path, split='val')
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    print(f"[OK] Test dataset loaded: {len(test_dataset)} samples")

    # 4. 评估模型
    print("\n" + "="*60)
    print("Starting Evaluation...")
    print("="*60)

    eval_results = evaluate_on_dataset(
        scorer, matcher, selector, vocabulary,
        test_loader, num_candidates, device
    )

    # 5. 打印评估结果
    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    print(f"Average ADE (Single-modal):     {eval_results['avg_ade']:.4f} m")
    print(f"Average FDE (Single-modal):     {eval_results['avg_fde']:.4f} m")
    print(f"Min ADE (Multi-modal):          {eval_results['min_ade']:.4f} m")
    print(f"Min FDE (Multi-modal):          {eval_results['min_fde']:.4f} m")
    print("="*60)
    
    # 6. 可视化样本
    print("\nGenerating visualizations...")
    os.makedirs('visualizations', exist_ok=True)

    num_vis_samples = min(5, len(test_dataset))  # 最多可视化5个样本
    for i in range(num_vis_samples):
        print(f"  Visualizing sample {i+1}/{num_vis_samples}...")
        sample = test_dataset[i]
        inference_results = inference_single_sample(
            scorer, matcher, selector, vocabulary,
            sample, num_candidates, device
        )
        visualize_sample(
            sample=sample,
            best_trajectory=inference_results['best_trajectory'],
            all_trajectories=inference_results['all_trajectories'],
            selected_goal=inference_results['selected_goal'],
            vocabulary=vocabulary,
            save_path=f'visualizations/sample_{i}.png'
        )

    print("\n" + "="*60)
    print("[OK] Inference completed!")
    print(f"Visualizations saved to: visualizations/")
    print("="*60)


if __name__ == "__main__":
    main()



