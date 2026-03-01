"""
GoalFlowMatcher nuScenes 训练脚本

基于 nuScenes 数据集训练 GoalFlowMatcher 模型。
训练时使用 ground truth goal，不依赖 Scorer 的准确率。

训练流程 (Conditional Flow Matching):
    1. 采样噪声: x_0 ~ N(0, 1)
    2. 采样时间: t ~ U(0, 1)
    3. 插值: x_t = (1-t)*x_0 + t*x_1
    4. 预测速度场: v_pred = model(x_t, goal, scene, t)
    5. 损失: loss = ||v_pred - (x_1 - x_0)||^2

使用方法:
    python train_matcher_nuscenes.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import pickle
from tqdm import tqdm
from typing import Tuple

# 添加路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from models.goal_flow_matcher import GoalFlowMatcher
from data.nuscenes_dataset import NuScenesDataset, collate_fn
from config.nuscenes_config import NuScenesConfig


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def compute_ade(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
    """
    Average Displacement Error
    Args:
        pred_traj: (B, T, 2)
        gt_traj: (B, T, 2)
    Returns:
        ade: (B,)
    """
    distances = torch.norm(pred_traj - gt_traj, dim=-1)  # (B, T)
    return distances.mean(dim=-1)  # (B,)


def compute_fde(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
    """
    Final Displacement Error
    Args:
        pred_traj: (B, T, 2)
        gt_traj: (B, T, 2)
    Returns:
        fde: (B,)
    """
    return torch.norm(pred_traj[:, -1, :] - gt_traj[:, -1, :], dim=-1)  # (B,)


def train_one_epoch(model, train_loader, optimizer, device, config):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="[Train Matcher]")
    for batch in pbar:
        # 字段映射: NuScenesDataset → Matcher
        x_1 = batch['future'].to(device)       # (B, 12, 2) 真实未来轨迹
        goal = batch['goal'].to(device)        # (B, 2) 真实目标点
        scene = batch['bev'].to(device)        # (B, 3, 200, 200) BEV 特征

        # 采样噪声和时间
        x_0 = torch.randn_like(x_1)           # (B, 12, 2) 高斯噪声
        B = x_1.shape[0]
        t = torch.rand(B, device=device)       # (B,) 均匀分布 [0, 1]

        # 计算 Flow Matching 损失
        loss = model.compute_loss(x_0, x_1, goal, scene, t)

        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止训练不稳定
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches


def validate(model, val_loader, device, config, traj_mean, traj_std):
    """验证模型：生成轨迹并计算 ADE/FDE（输出真实米制误差）"""
    model.eval()
    total_ade = 0.0
    total_fde = 0.0
    num_batches = 0
    
    # 反标准化用的 tensor: x_real = x_norm * std + mean
    t_mean = torch.tensor(traj_mean, dtype=torch.float32, device=device)  # (2,)
    t_std = torch.tensor(traj_std, dtype=torch.float32, device=device)    # (2,)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="[Val Matcher]")
        for batch in pbar:
            gt_traj = batch['future'].to(device)   # (B, 12, 2) 标准化坐标
            goal = batch['goal'].to(device)        # (B, 2) 标准化坐标
            scene = batch['bev'].to(device)        # (B, 3, 200, 200)

            # 用 ODE 求解器生成轨迹（标准化空间）
            pred_traj = model.generate(
                goal, scene,
                num_steps=config.matcher_num_steps,  # ODE 步数
                method='euler'
            )

            # 反标准化到米制，计算真实误差
            pred_traj_m = pred_traj * t_std + t_mean   # 标准化 → 米
            gt_traj_m = gt_traj * t_std + t_mean       # 标准化 → 米
            
            ade = compute_ade(pred_traj_m, gt_traj_m).mean().item()
            fde = compute_fde(pred_traj_m, gt_traj_m).mean().item()

            total_ade += ade
            total_fde += fde
            num_batches += 1
            pbar.set_postfix({'ade': f'{ade:.4f}', 'fde': f'{fde:.4f}'})

    return total_ade / num_batches, total_fde / num_batches


def main():
    """主函数"""
    print("=" * 70)
    print("GoalFlowMatcher nuScenes 训练")
    print("=" * 70)

    # 加载配置
    config = NuScenesConfig()
    config.print_config()

    # 设置随机种子
    set_seed(config.seed)

    # 设置设备
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 加载数据集
    print("\n加载数据集...")
    train_dir = os.path.join(config.processed_data_dir, 'train')
    val_dir = os.path.join(config.processed_data_dir, 'val')

    if not os.path.exists(train_dir):
        print(f"❌ 训练数据不存在: {train_dir}")
        print("请先运行预处理脚本: python scripts/preprocess_nuscenes.py")
        return

    # 加载训练集 metadata（包含标准化统计量）
    with open(os.path.join(train_dir, 'metadata.pkl'), 'rb') as f:
        train_metadata = pickle.load(f)
    traj_mean = train_metadata['traj_mean']  # (2,) 反标准化用
    traj_std = train_metadata['traj_std']    # (2,)
    print(f"标准化统计量: mean={traj_mean}, std={traj_std}")

    train_dataset = NuScenesDataset(train_dir, split='train')
    val_dataset = NuScenesDataset(val_dir, split='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.matcher_batch_size,   # 默认 8
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.matcher_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers
    )

    # 创建模型
    print("\n创建 GoalFlowMatcher 模型...")
    model = GoalFlowMatcher(
        traj_dim=2,                                    # 轨迹维度 (x, y)
        num_traj_points=config.future_frames,          # 12 个未来轨迹点
        d_model=config.matcher_hidden_dim,             # 256
        nhead=config.matcher_num_heads,                # 8
        num_encoder_layers=config.matcher_num_layers,  # 6
        dim_feedforward=config.matcher_hidden_dim * 4, # 1024
        dropout=config.matcher_dropout,                # 0.1
        scene_channels=config.bev_channels,            # 3 (lane, road, walkway)
        scene_size=(config.bev_height, config.bev_width),  # (200, 200) 输入尺寸
        scene_token_size=config.scene_token_size,      # (25, 25) 场景 token 目标尺寸
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {num_params:.2f}M")
    print(f"场景下采样后尺寸: {model._scene_out_h}x{model._scene_out_w} "
          f"= {model._scene_out_h * model._scene_out_w} tokens")

    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.matcher_learning_rate,       # 1e-4
        weight_decay=config.matcher_weight_decay  # 1e-5
    )

    # 学习率调度器
    scheduler = None
    if config.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience
        )

    # 创建 checkpoint 目录
    os.makedirs(config.matcher_checkpoint_dir, exist_ok=True)

    # 训练循环
    print("\n开始训练...")
    best_ade = float('inf')
    num_epochs = config.matcher_num_epochs  # 默认 50

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 70)

        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, device, config)
        print(f"Train Loss: {train_loss:.4f}")

        # 验证（每 5 个 epoch 验证一次）
        if epoch % config.eval_interval == 0:
            ade, fde = validate(model, val_loader, device, config, traj_mean, traj_std)
            print(f"Val ADE: {ade:.4f}m, Val FDE: {fde:.4f}m")

            # 学习率调度
            if scheduler is not None:
                scheduler.step(ade)

            # 保存最佳模型
            if ade < best_ade:
                best_ade = ade
                checkpoint_path = os.path.join(config.matcher_checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'ade': ade,
                    'fde': fde,
                    'traj_mean': traj_mean,  # 推理时反标准化用
                    'traj_std': traj_std,
                }, checkpoint_path)
                print(f"✅ 保存最佳模型: {checkpoint_path} (ADE: {ade:.4f}m)")

        # 定期保存
        if epoch % config.save_interval == 0:
            checkpoint_path = os.path.join(config.matcher_checkpoint_dir, f'epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

    print("\n" + "=" * 70)
    print("✅ 训练完成！")
    print("=" * 70)
    print(f"最佳验证 ADE: {best_ade:.4f}")
    print(f"模型保存在: {config.matcher_checkpoint_dir}")


if __name__ == "__main__":
    main()
