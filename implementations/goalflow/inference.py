"""

"""

import os
import sys

import torch
import torch.nn as nn
import torch.utils.data import DataLoader
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

def load_models(scorer_checkpoint_path: str="",
                matcher_checkpoint_path: str="",
                data_path: str,
                device='cpu'):
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
    scorer_checkpoint = torch.load(score_checkpoint_path, map_location=device)
    scorer.load_state_dict(score_checkpoint['model_state_dict'])
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
                            sample: int,
                            num_candidates: int=10,
                            device = 'cpu'):
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
    scorer_config = ScorerConfig()
    matcher_config = MatcherConfig()
    scorer.
    


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
    pass


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
    pass


def main():
    """
    主函数
    """
    pass


if __name__ == "__main__":
    main()



