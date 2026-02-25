"""
GoalPointScorer 训练脚本

训练流程：
1. 加载数据集
2. 初始化 GoalFlowMatcher 模型
3. 训练循环：
   - 采样时间：t ~ U(0, 1)
   - 随机噪声：x_0 ~ N(0, 1)
   - 插值x_t: x_t = (1-t)*x_0 + t*x_1
   - 预测速度场：v_pred = model(x_t, goal, scene, t)
   - 计算损失：loss = ||v_pred - (x_1 - x_0)||^2
4. 验证：生成轨迹，计算 ADE/FDE
5. 保存最佳模型

TODO: 你需要实现以下函数
- train_one_epoch()
- validate()
- compute_ade()
- compute_fde()
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# 添加路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from tqdm import tqdm
from typing import Tuple
from models.goal_flow_matcher import GoalFlowMatcher
from config.matcher_config import MatcherConfig
from data.toy_goalflow_dataset import ToyGoalFlowDataset


def set_seed(seed):
    """
    设置随机种子

    作用：
    - 保证每次运行结果一致
    - 便于调试和复现实验
    - 控制权重初始化、数据打乱、Dropout 等随机操作
    """
    torch.manual_seed(seed)                 # Pytorch CPU 随机数种子
    torch.cuda.manual_seed_all(seed)        # Pytorch GPU 随机数种子
    np.random.seed(seed)                    # Numpy 随机数种子


def train_one_epoch(model: nn.Module,
                    train_loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device,
                    config
                    ) -> float:
    """
    训练模型
    训练一个epoch
    关键步骤：
        - 遍历train_loader
        - 提取数据 trajectory(x_1), goal, bev_feature(scene)
        - 采样噪声 x_0 = torch.randn_like(x_1)
        - 采样时间步 t = torch.rand(B)
        - 调用 model.compute_loss(x_t, goal, scene, t)
        - 反向传播: loss.backward()
        - 优化器更新: optimieze.step()
        - 统计平均损失

    Args:
        model: GoalFlowMatcher
        train_loader: DataLoader
        optimizer: torch.nn.optimizer
        device
        config

    Returns:
        avg_loss: 平均损失
    """
    model.train()

    total_loss = 0.0

    pbar =  tqdm(train_loader, desc="[FlowMatcherTrainer]")

    for batch in pbar:
        # 1. 提取数据
        x_1 = batch["trajectory"].to(device)    # (B, T, 2)
        goal = batch["goal"].to(device)         # (B, 2)
        bev_feature = batch["bev_feature"].to(device)   # (B, C, H, W)

        # 2. 采样时间
        x_0 = torch.randn_like(x_1).to(device)
        B = x_1.shape[0]
        t = torch.rand(B, device=device)
        
        # 计算loss
        loss = model.compute_loss(x_0, x_1, goal, bev_feature, t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()

        # 更新进度条
        pbar.set_postfix({'loss': loss.item()})

    return (total_loss / len(train_loader))


def validate(model: nn.Module,
             val_loader: DataLoader,
             device,
             config
             ) -> Tuple[float, float]:
    
    model.eval()        # 设置为评估模式
    ade = 0.0
    fde = 0.0

    pbar = tqdm(val_loader, desc="[Validate FlowMatcher]")
    with torch.no_grad():
        for batch in pbar:
            # 1. 提取数据
            gt_traj = batch["trajectory"].to(device)
            goal = batch["goal"].to(device)
            bev_feature = batch["bev_feature"].to(device)

            pred_traj = model.generate(goal, 
                                       bev_feature, 
                                       num_steps=config.val_num_steps, 
                                       method=config.val_method)

            ade += compute_ade(pred_traj, gt_traj).mean().item()
            fde += compute_fde(pred_traj, gt_traj).mean().item()

    return (ade / len(val_loader), fde / len(val_loader))




def compute_ade(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
    """
    计算 Average Displacement Error (ADE)
    
    Args:
        pred_traj: (B, T, 2) 预测轨迹
        gt_traj: (B, T, 2) 真实轨迹
    
    Returns:
        ade: (B, N) ADE 值
    
    TODO: 实现 ADE 计算
    提示：
    ADE = mean(||pred - gt||) over all time steps
    """
    distances = torch.norm(pred_traj - gt_traj, dim=-1)  # 计算欧氏距离用torch.norm
    ade = distances.mean(dim=-1)  # (B, N)

    return ade


def compute_fde(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
    """
    计算 Final Displacement Error (FDE)
    
    Args:
        pred_traj: (B, T, 2) 预测轨迹
        gt_traj: (B, T, 2) 真实轨迹
    
    Returns:
        fde: (B, N) FDE 值
    
    TODO: 实现 FDE 计算
    提示：
    FDE = ||pred_end - gt_end||
    """
    pred_end = pred_traj[:, -1, :]   # (B, 2)
    gt_end = gt_traj[:, -1, :]   # (B, 2)
    fde = torch.norm(pred_end - gt_end, dim=-1)  # (B, N)

    return fde
        

def main():
    """
    主训练流程：

    步骤：
    1. 加载配置： config = MatcherConfig()
    2. 设置随机种子和设备
    3. 加载数据集
    4. 初始化模型
    5. 初始化优化器
    6. 训练循环：
        for epoch in range(num_epochs):
            train_loss = train_one_epoch(…)
            if epoch % eval_interval == 0
                ade, fde = validate()
                保存最佳模型
    """
    print("=" * 60)
    print("GoalFlowMatcher Training")
    print("=" * 60)

    config = MatcherConfig()
    
    # 设置随机种子
    set_seed(config.seed)
    
    # 设置设备
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建保存目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # ==================== 加载数据 ====================
    print("\n" + "=" * 60)
    print("Loading dataset...")
    print("=" * 60)
    train_dataset = ToyGoalFlowDataset(config.data_path, split='train')
    val_dataset = ToyGoalFlowDataset(config.data_path, split='val')

    # 打印更多配置信息
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

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

    # ==================== 初始化模型 ====================
    print("Initializing model...")
    model = GoalFlowMatcher(
        traj_dim=config.traj_dim,
        num_traj_points=config.num_traj_points,

        # Transformer 配置
        d_model=config.hidden_dim,
        nhead=config.num_heads,
        num_encoder_layers=config.num_layers,
        dim_feedforward=config.hidden_dim*4,
        dropout=config.dropout

    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 优化器初始化！！！
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # 定义scheduler
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
    best_ade = float('inf')

    num_epochs = config.num_epochs
    for epoch in range(1, num_epochs+1):
        train_loss = train_one_epoch(model,
                                    train_loader,
                                    optimizer,
                                    device,
                                    config)

        print(f"train_loss: {train_loss}")
        
        # 验证
        if epoch % config.eval_interval == 0:
            ade, fde = validate(model,
                                val_loader,
                                device,
                                config)
            print(f"ADE: {ade:.4f}, FDE: {fde:.4f}")

            # 学习率调度
            if scheduler is not None:
                scheduler.step(ade)

            if ade < best_ade:
                # 保存最优模型
                best_ade = ade
                torch.save(
                    {'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'ade': ade,
                    'fde': fde},
                    os.path.join(config.checkpoint_dir, 'best.pth')
                )
                print(f"✅ Saved best model (ade: {ade:.4f})")

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
                