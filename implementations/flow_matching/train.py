"""
Flow Matching 训练脚本
用于在 Toy Dataset 上训练速度场网络
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import Optional, Tuple, Dict

from models.flow_matcher import ConditionalFlowMatcher
from models.time_embedding import SinusoidalEmbedding
from data.toy_dataset import ToyTrajectoryDataset
from models.velocity_field_MLP import VelocityFieldMLP


# ============================================================================
# 简化版速度场网络（用于 Toy Dataset，不需要条件输入）
# ============================================================================

class SimpleVelocityField(nn.Module):
    """
    简化版速度场网络
    只接受 state 和 time，不需要条件输入（适用于 toy dataset）
    """
    def __init__(
        self, 
        state_dim: int = 12,
        time_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.state_dim = state_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        
        # 时间编码
        self.time_embedding = SinusoidalEmbedding(embedding_dim=time_dim)
        
        # MLP 网络
        layers = []
        # 输入层
        layers.append(nn.Linear(state_dim + time_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # 隐藏层
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # 输出层（不使用激活函数，因为速度场可以是任意实数）
        layers.append(nn.Linear(hidden_dim, state_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, state_dim) 当前状态
            t: (B,) 时间标量，范围 [0, 1]
        
        Returns:
            (B, state_dim) 速度场
        """
        # 时间编码
        t_emb = self.time_embedding(t)  # (B, time_dim)
        
        # 拼接 state 和 time
        xt = torch.cat([x, t_emb], dim=-1)  # (B, state_dim + time_dim)
        
        # 通过 MLP
        v = self.mlp(xt)  # (B, state_dim)
        
        return v


# ============================================================================
# 训练器类
# ============================================================================

class Trainer:
    """Flow Matching 训练器"""
    
    def __init__(
        self, 
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        flow_matcher: ConditionalFlowMatcher,
        device: torch.device,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        save_dir: str = "./checkpoints"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.flow_matcher = flow_matcher
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float("inf")
    
    def train_epoch(self, epoch: int) -> float:
        """
        训练一个 epoch
        
        Args:
            epoch: 当前 epoch 编号
        
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # **使用 tqdm 显示进度条**
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch in pbar:
            # 1. 提取数据
            # batch 是字典: {'trajectory': (B, 6, 2), 'type': list, 'condition': (B, 8)}
            x_1 = batch['trajectory'].to(self.device)  # (B, 6, 2)
            cond = batch['condition'].to(self.device)  # (B, 8)
            
            # 2. Flatten 轨迹: (B, 6, 2) -> (B, 12)
            batch_size = x_1.shape[0]
            x_1 = x_1.reshape(batch_size, -1)  # (B, 12)
            
            # 3. 采样噪声 x_0（从标准正态分布）
            x_0 = torch.randn_like(x_1) * 0.5  # (B, 12)，可以调整标准差
            
            # 4. 清零梯度
            self.optimizer.zero_grad()
            
            # 5. 计算 CFM Loss
            loss = self.flow_matcher.compute_cfm_loss(self.model, x_0, x_1, cond)
            
            # 6. 反向传播
            loss.backward()
            
            # 7. 梯度裁剪（可选，防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 8. 更新参数
            self.optimizer.step()
            
            # 9. 累计损失
            total_loss += loss.item()
            num_batches += 1
            
            # 10. 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        
        # 更新学习率
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        return avg_loss
    
    def validate(self) -> float:
        """
        验证模型
        
        Returns:
            平均验证损失
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():  # 不计算梯度
            pbar = tqdm(self.val_loader, desc="Validating")
            
            for batch in pbar:
                # 1. 提取数据
                x_1 = batch['trajectory'].to(self.device)  # (B, 6, 2)
                cond = batch['condition'].to(self.device)  # (B, 8)
                
                # 2. Flatten
                batch_size = x_1.shape[0]
                x_1 = x_1.reshape(batch_size, -1)  # (B, 12)
                
                # 3. 采样噪声
                x_0 = torch.randn_like(x_1) * 0.5
                
                # 4. 计算损失（不需要反向传播）
                loss = self.flow_matcher.compute_cfm_loss(self.model, x_0, x_1, cond)
                
                # 5. 累计损失
                total_loss += loss.item()
                num_batches += 1
                
                # 6. 更新进度条
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
        }
        
        if self.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        # 保存最新的检查点
        latest_path = self.save_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # 如果是最佳模型，额外保存
        if is_best:
            best_path = self.save_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ 保存最佳模型到 {best_path}")
    
    def train(self, num_epochs: int):
        """
        完整训练流程
        
        Args:
            num_epochs: 训练轮数
        """
        print("=" * 60)
        print("开始训练 Flow Matching 模型")
        print("=" * 60)
        print(f"训练集大小: {len(self.train_loader.dataset)}")
        print(f"验证集大小: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"总 epochs: {num_epochs}")
        print(f"设备: {self.device}")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss = self.validate()
            
            # 打印信息
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            
            # 保存检查点
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"  ✓ 新的最佳验证损失!")
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            print("-" * 60)
        
        print("\n" + "=" * 60)
        print("训练完成!")
        print(f"最佳验证损失: {self.best_val_loss:.6f}")
        print("=" * 60)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练 Flow Matching 模型')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=4, help='网络层数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 加载数据集
    data_dir = Path(args.data_dir)
    train_dataset = ToyTrajectoryDataset(str(data_dir / 'toy_train.npz'))
    val_dataset = ToyTrajectoryDataset(str(data_dir / 'toy_val.npz'))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,                                               # 每个 epoch 开始前随机打乱数据顺序
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False         # 将数据固定在 页锁定内存（pinned memory） 中
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 创建模型
    state_dim = 12  # 6个点 × 2维
    cond_dim = 8  # goal(2) + direction(2) + type_onehot(4)
    model = VelocityFieldMLP(
        state_dim=state_dim,
        cond_dim=cond_dim,
        time_dim=128,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_layers,  # 注意：VelocityFieldMLP 使用 num_hidden_layers
        dropout=0.1
    ).to(device)
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 创建学习率调度器（可选）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # 创建 Flow Matcher
    flow_matcher = ConditionalFlowMatcher(sigma=0.0)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        flow_matcher=flow_matcher,
        device=device,
        lr_scheduler=scheduler,
        save_dir=args.save_dir
    )
    
    # 开始训练
    trainer.train(num_epochs=args.epochs)


if __name__ == "__main__":
    main()