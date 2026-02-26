"""
GoalPointScorer 训练脚本

训练流程：
1. 加载数据集和词汇表
2. 初始化模型、优化器
3. 训练循环：
   - 前向传播：预测 distance score 和 DAC score
   - 计算损失：CrossEntropy + BCE
   - 反向传播和优化
4. 验证：计算 Top-1/Top-5 准确率
5. 保存最佳模型

TODO: 你需要实现以下函数
- train_one_epoch()
- validate()
- compute_accuracy()
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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.goal_point_scorer import GoalPointScorer
from data.toy_goalflow_dataset import ToyGoalFlowDataset
from config.scorer_config import ScorerConfig


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def compute_target_labels(vocabulary: torch.Tensor, 
                          gt_goals: torch.Tensor
                          ) -> torch.Tensor:
    """
    计算训练标签：找到最接近 gt_goal 的 vocabulary 索引
    
    Args:
        vocabulary: (N, 2) 词汇表
        gt_goals: (B, 2) 真实目标点
    
    Returns:
        target_idx: (B,) 最近的 vocabulary 索引
    
    TODO: 实现这个函数
    提示：
    1. 计算 gt_goals 到每个 vocabulary 点的距离
    2. 找到距离最小的索引
    """
    diff = vocabulary.unsqueeze(0) - gt_goals.unsqueeze(1)  # (B, N, 2)
    dis = torch.norm(diff, dim=-1)    # (B, N)
    _, target_idx = torch.min(dis, dim=-1)     # (B,)

    return target_idx


def compute_accuracy(pred_scores: torch.Tensor,
                     target_idx: torch.Tensor, 
                     k: int=1) -> float:
    """
    计算 Top-K 准确率
    
    Args:
        pred_scores: (B, N) 预测分数
        target_idx: (B,) 目标索引
        k: Top-K
    
    Returns:
        accuracy: float
    """
    # 获取 Top-K 预测索引
    _, topk_indices = pred_scores.topk(k, dim=-1)  # (B, k)
    
    # 检查 target_idx 是否在 Top-K 中
    target_idx_expanded = target_idx.unsqueeze(-1)  # (B, 1)
    correct = (topk_indices == target_idx_expanded).any(dim=-1)  # (B,)
    
    # 计算准确率
    accuracy = correct.float().mean().item()
    
    return accuracy


def train_one_epoch(model: nn.Module, 
                    train_loader: DataLoader, 
                    vocabulary: torch.Tensor, 
                    optimizer: torch.optim.Optimizer, 
                    device, 
                    config):
    """
    训练一个 epoch
    
    Args:
        model: GoalPointScorer 模型
        train_loader: 训练数据加载器
        vocabulary: (N, 2) 词汇表
        optimizer: 优化器
        device: 设备
        config: 配置
    
    Returns:
        avg_loss: 平均损失
        avg_acc: 平均准确率
    
    TODO: 实现训练循环
    提示：
    1. 遍历 train_loader
    2. 前向传播：model.forward()
    3. 计算损失：model.compute_loss()
    4. 反向传播：loss.backward()
    5. 优化器更新：optimizer.step()
    6. 计算准确率
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="[Train]")  
    for batch in pbar:
        # 1. 数据移到设备
        bev_feature = batch['bev_feature'].to(device)
        gt_goal = batch['goal'].to(device)
        drivable_area = batch['drivable_area'].to(device)

        # 2. 计算目标标签
        target_idx = compute_target_labels(vocabulary, gt_goal)

        # 3. 前向传播
        # 注意：vocabulary 需要扩展为 (B, N, 2)
        B = bev_feature.shape[0]
        vocab_expanded = vocabulary.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)
        pred_dis, pred_dac = model(vocab_expanded, bev_feature)

        # 4. 计算真实标签
        true_dis = model.compute_distance_score(vocabulary, gt_goal)  # (B, N)
        true_dac = model.compute_dac_score(vocabulary, drivable_area)  # (B, N)

        # 5. 计算损失（使用配置文件中的权重）
        loss, loss_dict = model.compute_loss(
            pred_dis, pred_dac, true_dis, true_dac,
            lambda_dis=config.lambda_dis,
            lambda_dac=config.lambda_dac
        )
        
        # 6. 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 7. 统计
        total_loss += loss_dict['loss']
        acc = compute_accuracy(pred_dis, target_idx, k=1)
        total_acc += acc
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({'loss': loss.item(), 'acc': acc})

    return total_loss / num_batches, total_acc / num_batches


def validate(model, 
             val_loader, 
             vocabulary, 
             device, 
             config):
    """
    验证模型
    
    Args:
        model: GoalPointScorer 模型
        val_loader: 验证数据加载器
        vocabulary: (N, 2) 词汇表
        device: 设备
        config: 配置
    
    Returns:
        avg_loss: 平均损失
        top1_acc: Top-1 准确率
        top5_acc: Top-5 准确率
    
    TODO: 实现验证循环
    提示：
    1. 使用 torch.no_grad()
    2. 计算 Top-1 和 Top-5 准确率
    """
    model.eval()
    total_loss = 0.0
    total_top1_acc = 0.0
    total_top5_acc = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="[Validate]")  
        for batch in pbar:
            # 1. 数据移到设备
            bev_feature = batch['bev_feature'].to(device)
            gt_goal = batch['goal'].to(device)
            drivable_area = batch['drivable_area'].to(device)

            # 2. 计算目标标签
            target_idx = compute_target_labels(vocabulary, gt_goal)

            # 3. 前向传播
            # 注意：vocabulary 需要扩展为 (B, N, 2)
            B = bev_feature.shape[0]
            vocab_expanded = vocabulary.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)
            pred_dis, pred_dac = model(vocab_expanded, bev_feature)

            # 4. 计算真实标签
            true_dis = model.compute_distance_score(vocabulary, gt_goal)  # (B, N)
            true_dac = model.compute_dac_score(vocabulary, drivable_area)  # (B, N)

            # 5. 计算损失（使用配置文件中的权重）
            loss, loss_dict = model.compute_loss(
                pred_dis, pred_dac, true_dis, true_dac,
                lambda_dis=config.lambda_dis,
                lambda_dac=config.lambda_dac
            )

            # 6. 统计
            total_loss += loss_dict['loss']
            total_top1_acc += compute_accuracy(pred_dis, target_idx, k=1)
            total_top5_acc += compute_accuracy(pred_dis, target_idx, k=5)
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({'loss': loss.item()})
    
    return (total_loss / num_batches, 
            total_top1_acc / num_batches, 
            total_top5_acc / num_batches)


def main():
    # 加载配置
    config = ScorerConfig()
    
    # 设置随机种子
    set_seed(config.seed)
    
    # 设置设备
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建保存目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # ==================== 加载数据 ====================
    print("Loading dataset...")
    train_dataset = ToyGoalFlowDataset(config.data_path, split='train')
    val_dataset = ToyGoalFlowDataset(config.data_path, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    # 获取词汇表
    vocabulary = train_dataset.get_vocabulary().to(device)  # (N, 2)
    print(f"Vocabulary size: {vocabulary.shape[0]}")
    
    # ==================== 初始化模型 ====================
    print("Initializing model...")
    model = GoalPointScorer(
        vocabulary_size=config.vocab_size,
        feature_dim=config.hidden_dim,  # 使用 hidden_dim 作为 feature_dim
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        scene_in_channels=config.scene_channels,
        kernel_size=3,
        stride=1,
        dropout=config.dropout
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # ==================== 优化器和调度器 ====================
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = None
    if config.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience
        )
    
    # ==================== 训练循环 ====================
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, config.num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config.num_epochs}")
        print(f"{'='*50}")
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, vocabulary, optimizer, device, config
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # 验证
        if epoch % config.eval_interval == 0:
            val_loss, top1_acc, top5_acc = validate(
                model, val_loader, vocabulary, device, config
            )
            print(f"Val Loss: {val_loss:.4f}, Top-1 Acc: {top1_acc:.4f}, Top-5 Acc: {top5_acc:.4f}")
            
            # 学习率调度
            if scheduler is not None:
                scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'top1_acc': top1_acc,
                    'top5_acc': top5_acc,
                }, os.path.join(config.checkpoint_dir, 'best.pth'))
                print(f"✅ Saved best model (val_loss: {val_loss:.4f})")
        
        # 定期保存
        if epoch % config.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(config.checkpoint_dir, f'epoch_{epoch}.pth'))
    
    print("\n🎉 Training completed!")


if __name__ == "__main__":
    main()
