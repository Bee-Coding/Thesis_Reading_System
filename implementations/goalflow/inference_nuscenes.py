"""
GoalFlow nuScenes 端到端推理 + 可视化脚本

推理流程:
    1. GoalPointScorer 从词汇表中选择目标点
    2. GoalFlowMatcher 以目标点为条件生成多条候选轨迹
    3. 选择最优轨迹（距离 goal 最近）

评估模式:
    - GT Goal:  用真实目标点驱动 Matcher（评估 Matcher 本身的能力）
    - Pred Goal: 用 Scorer 预测的目标点驱动 Matcher（评估端到端效果）

使用方法:
    python inference_nuscenes.py
"""

import os
import sys
import pickle

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无头模式，不需要显示器
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from models.goal_point_scorer import GoalPointScorer
from models.goal_flow_matcher import GoalFlowMatcher
from data.nuscenes_dataset import NuScenesDataset, collate_fn
from config.nuscenes_config import NuScenesConfig


# ==================== 指标计算 ====================

def compute_ade(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """ADE: 所有时间步的平均位移误差。pred/gt: (B, T, 2) → (B,)"""
    return torch.norm(pred - gt, dim=-1).mean(dim=-1)


def compute_fde(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """FDE: 最后一个时间步的位移误差。pred/gt: (B, T, 2) → (B,)"""
    return torch.norm(pred[:, -1] - gt[:, -1], dim=-1)


# ==================== 模型加载 ====================

def load_models(config, device):
    """加载 Scorer 和 Matcher，返回模型 + 标准化参数"""

    # --- 标准化参数（从 train metadata 读取）---
    meta_path = os.path.join(config.processed_data_dir, 'train', 'metadata.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    traj_mean = meta['traj_mean']  # (2,) ndarray
    traj_std = meta['traj_std']    # (2,)

    # --- 词汇表（从 train 数据读取，已标准化）---
    vocab_path = os.path.join(config.processed_data_dir, 'train', 'vocabulary.npy')
    vocabulary = torch.from_numpy(np.load(vocab_path)).float().to(device)  # (K, 2)

    # --- Scorer ---
    scorer_path = os.path.join(config.scorer_checkpoint_dir, 'best_model.pth')
    scorer = GoalPointScorer(
        vocabulary_size=len(vocabulary),
        feature_dim=config.scorer_hidden_dim,
        hidden_dim=config.scorer_hidden_dim,
        num_heads=config.scorer_num_heads,
        num_layers=config.scorer_num_layers,
        scene_in_channels=config.bev_channels,
        kernel_size=3,
        stride=2,
        dropout=config.scorer_dropout,
    ).to(device)
    ckpt = torch.load(scorer_path, map_location=device, weights_only=False)
    scorer.load_state_dict(ckpt['model_state_dict'])
    scorer.eval()
    print(f"[OK] Scorer loaded: {scorer_path}  (Top-1 acc: {ckpt.get('val_top1_acc', '?'):.2%})")

    # --- Matcher ---
    matcher_path = os.path.join(config.matcher_checkpoint_dir, 'best_model.pth')
    matcher = GoalFlowMatcher(
        traj_dim=2,
        num_traj_points=config.future_frames,
        d_model=config.matcher_hidden_dim,
        nhead=config.matcher_num_heads,
        num_encoder_layers=config.matcher_num_layers,
        dim_feedforward=config.matcher_hidden_dim * 4,
        dropout=config.matcher_dropout,
        scene_channels=config.bev_channels,
        scene_size=(config.bev_height, config.bev_width),
        scene_token_size=config.scene_token_size,
    ).to(device)
    ckpt_m = torch.load(matcher_path, map_location=device, weights_only=False)
    matcher.load_state_dict(ckpt_m['model_state_dict'])
    matcher.eval()
    print(f"[OK] Matcher loaded: {matcher_path}  (ADE: {ckpt_m.get('ade', '?'):.4f}m)")

    return scorer, matcher, vocabulary, traj_mean, traj_std


# ==================== 评估 ====================

@torch.no_grad()
def evaluate(scorer, matcher, vocabulary, data_loader,
             traj_mean, traj_std, config, device):
    """
    在数据集上评估两种模式:
      1. GT Goal → Matcher（纯 Matcher 能力）
      2. Scorer Goal → Matcher（端到端）

    返回米制 ADE/FDE。
    """
    t_mean = torch.tensor(traj_mean, dtype=torch.float32, device=device)
    t_std = torch.tensor(traj_std, dtype=torch.float32, device=device)

    # 累积指标
    metrics = {
        'gt_ade': [], 'gt_fde': [],           # GT goal 模式
        'gt_min_ade': [], 'gt_min_fde': [],   # GT goal 多模态 best-of-N
        'pred_ade': [], 'pred_fde': [],       # Scorer goal 模式
        'pred_min_ade': [], 'pred_min_fde': [],
        'goal_error': [],                      # Scorer goal 与 GT goal 的距离
    }

    num_steps = config.matcher_num_steps
    num_candidates = config.num_candidates
    vocab_K = vocabulary.shape[0]

    pbar = tqdm(data_loader, desc="[Inference]")
    for batch in pbar:
        gt_traj = batch['future'].to(device)   # (B, 12, 2) 标准化
        gt_goal = batch['goal'].to(device)     # (B, 2) 标准化
        scene = batch['bev'].to(device)        # (B, 3, 200, 200)
        B = gt_traj.shape[0]

        # ---- 模式1: GT Goal ----
        # 单条轨迹
        pred_single = matcher.generate(gt_goal, scene, num_steps=num_steps, method='euler')
        # 多条候选
        pred_multi = matcher.generate_multiple(gt_goal, scene,
                                               num_samples=num_candidates,
                                               num_steps=num_steps, method='euler')  # (B, N, T, 2)

        # 反标准化到米制
        pred_s_m = pred_single * t_std + t_mean
        gt_m = gt_traj * t_std + t_mean
        pred_multi_m = pred_multi * t_std + t_mean

        ade = compute_ade(pred_s_m, gt_m)  # (B,)
        fde = compute_fde(pred_s_m, gt_m)
        metrics['gt_ade'].extend(ade.cpu().tolist())
        metrics['gt_fde'].extend(fde.cpu().tolist())

        # best-of-N
        gt_exp = gt_m.unsqueeze(1).expand_as(pred_multi_m)  # (B, N, T, 2)
        ade_all = torch.norm(pred_multi_m - gt_exp, dim=-1).mean(dim=-1)  # (B, N)
        fde_all = torch.norm(pred_multi_m[:, :, -1] - gt_exp[:, :, -1], dim=-1)  # (B, N)
        metrics['gt_min_ade'].extend(ade_all.min(dim=1).values.cpu().tolist())
        metrics['gt_min_fde'].extend(fde_all.min(dim=1).values.cpu().tolist())

        # ---- 模式2: Scorer Goal ----
        vocab_exp = vocabulary.unsqueeze(0).expand(B, -1, -1)  # (B, K, 2)
        pred_dis, pred_dac = scorer(vocab_exp, scene)
        # 选择得分最高的 goal
        log_dis = F.log_softmax(pred_dis, dim=-1)
        log_dac = torch.log(pred_dac + 1e-8)
        final_scores = log_dis + 0.01 * log_dac  # 与训练时 lambda_dac 一致
        best_idx = final_scores.argmax(dim=-1)  # (B,)
        scorer_goal = vocabulary[best_idx]  # (B, 2)

        # 生成轨迹
        pred_scorer = matcher.generate(scorer_goal, scene, num_steps=num_steps, method='euler')
        pred_scorer_multi = matcher.generate_multiple(scorer_goal, scene,
                                                      num_samples=num_candidates,
                                                      num_steps=num_steps, method='euler')

        pred_scorer_m = pred_scorer * t_std + t_mean
        pred_scorer_multi_m = pred_scorer_multi * t_std + t_mean

        ade2 = compute_ade(pred_scorer_m, gt_m)
        fde2 = compute_fde(pred_scorer_m, gt_m)
        metrics['pred_ade'].extend(ade2.cpu().tolist())
        metrics['pred_fde'].extend(fde2.cpu().tolist())

        ade_all2 = torch.norm(pred_scorer_multi_m - gt_exp, dim=-1).mean(dim=-1)
        fde_all2 = torch.norm(pred_scorer_multi_m[:, :, -1] - gt_exp[:, :, -1], dim=-1)
        metrics['pred_min_ade'].extend(ade_all2.min(dim=1).values.cpu().tolist())
        metrics['pred_min_fde'].extend(fde_all2.min(dim=1).values.cpu().tolist())

        # goal 误差
        scorer_goal_m = scorer_goal * t_std + t_mean
        gt_goal_m = gt_goal * t_std + t_mean
        ge = torch.norm(scorer_goal_m - gt_goal_m, dim=-1)
        metrics['goal_error'].extend(ge.cpu().tolist())

        pbar.set_postfix({
            'GT_ADE': f"{np.mean(metrics['gt_ade']):.2f}",
            'Pred_ADE': f"{np.mean(metrics['pred_ade']):.2f}",
        })

    # 汇总
    results = {k: np.mean(v) for k, v in metrics.items()}
    return results, metrics


# ==================== 可视化 ====================

@torch.no_grad()
def visualize_samples(scorer, matcher, vocabulary, dataset,
                      traj_mean, traj_std, config, device,
                      save_dir, num_samples=10):
    """可视化若干样本的推理结果，每张图包含 BEV + 轨迹"""

    os.makedirs(save_dir, exist_ok=True)
    t_mean = torch.tensor(traj_mean, dtype=torch.float32, device=device)
    t_std = torch.tensor(traj_std, dtype=torch.float32, device=device)

    num_samples = min(num_samples, len(dataset))
    # 均匀采样
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    for idx_i, idx in enumerate(indices):
        sample = dataset[idx]
        gt_traj = sample['future'].unsqueeze(0).to(device)  # (1, 12, 2)
        gt_goal = sample['goal'].unsqueeze(0).to(device)    # (1, 2)
        scene = sample['bev'].unsqueeze(0).to(device)       # (1, 3, 200, 200)
        bev_np = sample['bev'].numpy()                       # (3, 200, 200)

        # --- Scorer 预测 goal ---
        vocab_exp = vocabulary.unsqueeze(0)  # (1, K, 2)
        pred_dis, pred_dac = scorer(vocab_exp, scene)
        log_dis = F.log_softmax(pred_dis, dim=-1)
        log_dac = torch.log(pred_dac + 1e-8)
        final_scores = log_dis + 0.01 * log_dac
        best_idx = final_scores.argmax(dim=-1)
        scorer_goal = vocabulary[best_idx]  # (1, 2)

        # --- Matcher 生成轨迹 ---
        # GT goal 模式
        pred_gt = matcher.generate(gt_goal, scene,
                                   num_steps=config.matcher_num_steps, method='euler')
        pred_gt_multi = matcher.generate_multiple(gt_goal, scene,
                                                  num_samples=config.num_candidates,
                                                  num_steps=config.matcher_num_steps, method='euler')
        # Scorer goal 模式
        pred_sc = matcher.generate(scorer_goal, scene,
                                   num_steps=config.matcher_num_steps, method='euler')

        # 反标准化到米制
        gt_m = (gt_traj[0] * t_std + t_mean).cpu().numpy()          # (12, 2)
        gt_goal_m = (gt_goal[0] * t_std + t_mean).cpu().numpy()     # (2,)
        pred_gt_m = (pred_gt[0] * t_std + t_mean).cpu().numpy()     # (12, 2)
        pred_sc_m = (pred_sc[0] * t_std + t_mean).cpu().numpy()     # (12, 2)
        scorer_goal_m = (scorer_goal[0] * t_std + t_mean).cpu().numpy()
        pred_gt_multi_m = (pred_gt_multi[0] * t_std + t_mean).cpu().numpy()  # (N, 12, 2)
        vocab_m = (vocabulary * t_std + t_mean).cpu().numpy()        # (K, 2)

        # --- 绘图 ---
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        for ax_i, (ax, title, pred_m, goal_m, multi_m) in enumerate(zip(
            axes,
            ['GT Goal → Matcher', 'Scorer Goal → Matcher'],
            [pred_gt_m, pred_sc_m],
            [gt_goal_m, scorer_goal_m],
            [pred_gt_multi_m, None],
        )):
            # BEV 背景（合并3通道为灰度）
            bev_gray = bev_np.max(axis=0)  # (200, 200)
            bev_range = config.bev_range
            ax.imshow(bev_gray, cmap='gray', alpha=0.4,
                      extent=[-bev_range, bev_range, -bev_range, bev_range],
                      origin='lower', zorder=0)

            # 词汇表点
            ax.scatter(vocab_m[:, 0], vocab_m[:, 1],
                       c='lightgray', s=15, alpha=0.5, zorder=1, label='Vocabulary')

            # 候选轨迹（仅 GT goal 模式）
            if multi_m is not None:
                for k in range(multi_m.shape[0]):
                    lbl = f'Candidates (N={multi_m.shape[0]})' if k == 0 else None
                    ax.plot(multi_m[k, :, 0], multi_m[k, :, 1],
                            c='silver', alpha=0.3, linewidth=0.8, zorder=2, label=lbl)

            # GT 轨迹
            ax.plot(gt_m[:, 0], gt_m[:, 1],
                    'g-', linewidth=2.5, label='GT Trajectory', zorder=5)
            ax.plot(gt_m[0, 0], gt_m[0, 1], 'bo', markersize=8, zorder=6)  # 起点

            # 预测轨迹
            ax.plot(pred_m[:, 0], pred_m[:, 1],
                    'r-', linewidth=2.5, label='Predicted', zorder=5)

            # Goal 标记
            ax.scatter(*gt_goal_m, c='green', s=200, marker='*',
                       edgecolors='k', linewidths=1, zorder=10, label='GT Goal')
            ax.scatter(*goal_m, c='red', s=200, marker='*',
                       edgecolors='k', linewidths=1, zorder=10, label='Used Goal')

            # 指标
            ade_val = np.linalg.norm(pred_m - gt_m, axis=1).mean()
            fde_val = np.linalg.norm(pred_m[-1] - gt_m[-1])
            goal_err = np.linalg.norm(goal_m - gt_goal_m)
            info = f'ADE: {ade_val:.2f}m\nFDE: {fde_val:.2f}m\nGoal Err: {goal_err:.2f}m'
            ax.text(0.02, 0.98, info, transform=ax.transAxes, va='top',
                    fontsize=10, family='monospace',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.9), zorder=15)

            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_aspect('equal')
            ax.legend(loc='lower right', fontsize=8, framealpha=0.8)
            ax.grid(True, alpha=0.2)

            # 自动缩放到轨迹范围
            all_x = np.concatenate([gt_m[:, 0], pred_m[:, 0]])
            all_y = np.concatenate([gt_m[:, 1], pred_m[:, 1]])
            margin = 10
            ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
            ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

        fig.suptitle(f'Sample {idx}', fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'sample_{idx:04d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [{idx_i+1}/{num_samples}] Saved: {save_path}")


# ==================== 主函数 ====================

def main():
    config = NuScenesConfig()
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("GoalFlow nuScenes Inference")
    print("=" * 70)
    print(f"Device: {device}")

    # 1. 加载模型
    print("\n--- Loading models ---")
    scorer, matcher, vocabulary, traj_mean, traj_std = load_models(config, device)

    # 2. 加载测试集
    print("\n--- Loading test data ---")
    test_dir = os.path.join(config.processed_data_dir, 'test')
    test_dataset = NuScenesDataset(test_dir, split='test')
    test_loader = DataLoader(test_dataset, batch_size=config.matcher_batch_size,
                             shuffle=False, collate_fn=collate_fn, num_workers=0)
    print(f"Test samples: {len(test_dataset)}")

    # 3. 评估
    print("\n--- Evaluating ---")
    results, raw_metrics = evaluate(scorer, matcher, vocabulary, test_loader,
                                    traj_mean, traj_std, config, device)

    print("\n" + "=" * 70)
    print("Evaluation Results (meters)")
    print("=" * 70)
    print(f"{'Mode':<25} {'ADE':>8} {'FDE':>8} {'minADE':>8} {'minFDE':>8}")
    print("-" * 70)
    print(f"{'GT Goal (Matcher only)':<25} {results['gt_ade']:>8.3f} {results['gt_fde']:>8.3f} "
          f"{results['gt_min_ade']:>8.3f} {results['gt_min_fde']:>8.3f}")
    print(f"{'Scorer Goal (End-to-End)':<25} {results['pred_ade']:>8.3f} {results['pred_fde']:>8.3f} "
          f"{results['pred_min_ade']:>8.3f} {results['pred_min_fde']:>8.3f}")
    print(f"\nScorer Goal Error: {results['goal_error']:.3f}m")
    print("=" * 70)

    # 4. 可视化
    print("\n--- Generating visualizations ---")
    vis_dir = os.path.join(config.vis_dir, 'inference')
    visualize_samples(scorer, matcher, vocabulary, test_dataset,
                      traj_mean, traj_std, config, device,
                      save_dir=vis_dir, num_samples=10)

    print(f"\nVisualizations saved to: {vis_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
