"""
GoalPointScorer nuScenes 训练脚本

基于 nuScenes 数据集训练 GoalPointScorer 模型

使用方法:
    python train_scorer_nuscenes.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
from tqdm import tqdm

# 添加路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from models.goal_point_scorer import GoalPointScorer
from data.nuscenes_dataset import NuScenesDataset, collate_fn
from config.nuscenes_config import NuScenesConfig


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def compute_target_labels(vocabulary: torch.Tensor, gt_goals: torch.Tensor) -> torch.Tensor:
    """
    计算训练标签：找到最接近 gt_goal 的 vocabulary 索引
    
    Args:
        vocabulary: (N, 2) 词汇表
        gt_goals: (B, 2) 真实目标点
    
    Returns:
        target_idx: (B,) 最近的 vocabulary 索引
    """
    diff = vocabulary.unsqueeze(0) - gt_goals.unsqueeze(1)  # (B, N, 2)
    dis = torch.norm(diff, dim=-1)  # (B, N)
    _, target_idx = torch.min(dis, dim=-1)  # (B,)
    return target_idx


def compute_accuracy(pred_scores: torch.Tensor, target_idx: torch.Tensor, k: int = 1) -> float:
    """
    计算 Top-K 准确率
    
    Args:
        pred_scores: (B, N) 预测分数
        target_idx: (B,) 目标索引
        k: Top-K
    
    Returns:
        accuracy: float
    """
    _, topk_indices = pred_scores.topk(k, dim=-1)  # (B, k)
    target_idx_expanded = target_idx.unsqueeze(-1)  # (B, 1)
    correct = (topk_indices == target_idx_expanded).any(dim=-1)  # (B,)
    accuracy = correct.float().mean().item()
    return accuracy


def train_one_epoch(model, train_loader, vocabulary, optimizer, device, config):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="[Train]")
    for batch in pbar:
        # 数据移到设备
        bev = batch['bev'].to(device)  # (B, 3, H, W)
        gt_goal = batch['goal'].to(device)  # (B, 2)

        # 计算目标标签
        target_idx = compute_target_labels(vocabulary, gt_goal)

        # 前向传播
        B = bev.shape[0]
        vocab_expanded = vocabulary.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)
        
        # 注意: GoalPointScorer 需要 (B, N, 2) 的 goal_points 和 (B, C, H, W) 的 bev_feature
        pred_scores = model(vocab_expanded, bev)  # (B, N)

        # 计算损失 (简化版，只使用分类损失)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(pred_scores, target_idx)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        acc = compute_accuracy(pred_scores, target_idx, k=1)
        total_acc += acc
        num_batches += 1

        pbar.set_postfix({'loss': loss.item(), 'acc': acc})

    return total_loss / num_batches, total_acc / num_batches


def validate(model, val_loader, vocabulary, device, config):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    total_top1_acc = 0.0
    total_top5_acc = 0.0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="[Val]")
        for batch in pbar:
            bev = batch['bev'].to(device)
            gt_goal = batch['goal'].to(device)

            target_idx = compute_target_labels(vocabulary, gt_goal)

            B = bev.shape[0]
            vocab_expanded = vocabulary.unsqueeze(0).expand(B, -1, -1)
            pred_scores = model(vocab_expanded, bev)

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(pred_scores, target_idx)

            total_loss += loss.item()
            total_top1_acc += compute_accuracy(pred_scores, target_idx, k=1)
            total_top5_acc += compute_accuracy(pred_scores, target_idx, k=5)
            num_batches += 1

            pbar.set_postfix({'loss': loss.item()})

    return (total_loss / num_batches, 
            total_top1_acc / num_batches, 
            total_top5_acc / num_batches)


def main():
    """主函数"""
    print("=" * 70)
    print("GoalPointScorer nuScenes 训练")
    print("=" * 70)

    # 加载配置
    config = NuScenesConfig()
    config.print_config()

    # 设置随机种子
    set_seed(config.seed)

    # 设置设备
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 创建数据集
    print("\n加载数据集...")
    train_dir = os.path.join(config.processed_data_dir, 'train')
    val_dir = os.path.join(config.processed_data_dir, 'val')

    if not os.path.exists(train_dir):
        print(f"❌ 训练数据不存在: {train_dir}")
        print("请先运行预处理脚本: python scripts/preprocess_nuscenes.py")
        return

    train_dataset = NuScenesDataset(train_dir, split='train')
    val_dataset = NuScenesDataset(val_dir, split='val')

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.scorer_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.scorer_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers
    )

    # 获取词汇表
    vocabulary = torch.from_numpy(train_dataset.get_vocabulary()).float().to(device)
    print(f"词汇表大小: {vocabulary.shape}")

    # 创建模型
    print("\n创建模型...")
    model = GoalPointScorer(
        bev_channels=config.bev_channels,
        hidden_dim=config.scorer_hidden_dim,
        num_heads=config.scorer_num_heads,
        num_layers=config.scorer_num_layers,
        dropout=config.scorer_dropout
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.scorer_learning_rate,
        weight_decay=config.scorer_weight_decay
    )

    # 学习率调度器
    if config.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            verbose=True
        )

    # 创建 checkpoint 目录
    os.makedirs(config.scorer_checkpoint_dir, exist_ok=True)

    # 训练循环
    print("\n开始训练...")
    best_val_loss = float('inf')

    for epoch in range(config.scorer_num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.scorer_num_epochs}")
        print("-" * 70)

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, vocabulary, optimizer, device, config
        )

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # 验证
        if (epoch + 1) % config.eval_interval == 0:
            val_loss, val_top1_acc, val_top5_acc = validate(
                model, val_loader, vocabulary, device, config
            )

            print(f"Val Loss: {val_loss:.4f}, Top-1 Acc: {val_top1_acc:.4f}, Top-5 Acc: {val_top5_acc:.4f}")

            # 学习率调度
            if config.use_scheduler:
                scheduler.step(val_loss)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(config.scorer_checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_top1_acc': val_top1_acc,
                    'val_top5_acc': val_top5_acc,
                }, checkpoint_path)
                print(f"✅ 保存最佳模型: {checkpoint_path}")

        # 定期保存
        if (epoch + 1) % config.save_interval == 0:
            checkpoint_path = os.path.join(config.scorer_checkpoint_dir, f'epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

    print("\n" + "=" * 70)
    print("✅ 训练完成！")
    print("=" * 70)
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存在: {config.scorer_checkpoint_dir}")


if __name__ == "__main__":
    main()
