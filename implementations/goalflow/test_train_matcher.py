"""
GoalFlowMatcher 训练脚本快速测试

目的：
- 验证训练脚本逻辑是否正确
- 只训练 3 个 epoch，快速发现问题
- 检查数据加载、模型前向传播、损失计算、验证流程是否正常

运行：
    python test_train_matcher.py
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Tuple

# 添加路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.goal_flow_matcher import GoalFlowMatcher
from config.matcher_config import MatcherConfig
from data.toy_goalflow_dataset import ToyGoalFlowDataset


# ==================== 测试配置 ====================
class TestConfig(MatcherConfig):
    """快速测试配置"""
    # 训练参数
    num_epochs = 3          # 只训练 3 个 epoch
    eval_interval = 1       # 每个 epoch 都验证
    save_interval = 3       # 最后保存一次
    
    # 设备设置
    device = 'cpu'          # 使用 CPU（如果有 GPU 可以改为 'cuda'）
    num_workers = 0         # Windows 上设为 0，Linux 可以设为 2
    
    # 其他参数保持不变
    batch_size = 8          # 减小 batch size 加快测试
    
    # 验证参数
    val_num_steps = 5       # 减少 ODE 求解步数加快验证


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_one_epoch(model: nn.Module,
                    train_loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    config: TestConfig
                    ) -> float:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="[Train]")

    for batch in pbar:
        # 1. 数据准备
        x_1 = batch["trajectory"].to(device)
        goal = batch["goal"].to(device)
        bev_feature = batch["bev_feature"].to(device)

        # 2. 采样噪声和时间
        x_0 = torch.randn_like(x_1)
        B = x_1.shape[0]
        t = torch.rand(B, device=device)
        
        # 3. 计算损失
        loss = model.compute_loss(x_0, x_1, goal, bev_feature, t)

        # 4. 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 5. 统计
        total_loss += loss.item()
        num_batches += 1

        # 更新进度条
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches


def validate(model: nn.Module,
             val_loader: DataLoader,
             device: torch.device,
             config: TestConfig
             ) -> Tuple[float, float]:
    """验证模型"""
    model.eval()
    total_ade = 0.0
    total_fde = 0.0
    num_batches = 0

    pbar = tqdm(val_loader, desc="[Validate]")
    
    with torch.no_grad():
        for batch in pbar:
            # 1. 数据准备
            gt_traj = batch["trajectory"].to(device)
            goal = batch["goal"].to(device)
            bev_feature = batch["bev_feature"].to(device)

            # 2. 生成轨迹
            pred_traj = model.generate(
                goal=goal,
                scene=bev_feature,
                num_steps=config.val_num_steps,
                method=config.val_method
            )

            # 3. 计算指标
            ade = compute_ade(pred_traj, gt_traj)
            fde = compute_fde(pred_traj, gt_traj)
            
            total_ade += ade.mean().item()
            total_fde += fde.mean().item()
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({
                'ADE': f'{ade.mean().item():.4f}',
                'FDE': f'{fde.mean().item():.4f}'
            })

    return total_ade / num_batches, total_fde / num_batches


def compute_ade(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
    """计算 Average Displacement Error"""
    distances = torch.norm(pred_traj - gt_traj, dim=-1)
    ade = distances.mean(dim=-1)
    return ade


def compute_fde(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
    """计算 Final Displacement Error"""
    pred_end = pred_traj[:, -1, :]
    gt_end = gt_traj[:, -1, :]
    fde = torch.norm(pred_end - gt_end, dim=-1)
    return fde


def main():
    """主测试流程"""
    print("=" * 70)
    print("GoalFlowMatcher Training - Quick Test (3 epochs)")
    print("=" * 70)
    
    # ==================== 1. 加载配置 ====================
    config = TestConfig()
    print(f"\n[Config] Test Configuration:")
    print(f"  - Epochs: {config.num_epochs}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Device: {config.device}")
    print(f"  - Eval interval: {config.eval_interval}")
    print(f"  - Val ODE steps: {config.val_num_steps}")
    
    # ==================== 2. 设置随机种子 ====================
    set_seed(config.seed)
    print(f"  - Random seed: {config.seed}")
    
    # ==================== 3. 设置设备 ====================
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"  - Using device: {device}")

    # ==================== 4. 创建保存目录 ====================
    test_checkpoint_dir = os.path.join(config.checkpoint_dir, 'test')
    os.makedirs(test_checkpoint_dir, exist_ok=True)
    print(f"  - Checkpoint dir: {test_checkpoint_dir}")

    # ==================== 5. 加载数据 ====================
    print("\n" + "=" * 70)
    print("Loading dataset...")
    print("=" * 70)
    
    try:
        train_dataset = ToyGoalFlowDataset(config.data_path, split='train')
        val_dataset = ToyGoalFlowDataset(config.data_path, split='val')
        print(f"[OK] Train samples: {len(train_dataset)}")
        print(f"[OK] Val samples: {len(val_dataset)}")
    except Exception as e:
        print(f"[ERROR] Error loading dataset: {e}")
        print(f"   Please check if data file exists: {config.data_path}")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False
    )
    
    print(f"[OK] Train batches: {len(train_loader)}")
    print(f"[OK] Val batches: {len(val_loader)}")

    # ==================== 6. 初始化模型 ====================
    print("\n" + "=" * 70)
    print("Initializing model...")
    print("=" * 70)
    
    try:
        model = GoalFlowMatcher(
            traj_dim=config.traj_dim,
            num_traj_points=config.num_traj_points,
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            num_encoder_layers=config.num_layers,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            scene_channels=config.scene_channels,
        ).to(device)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"[OK] Model initialized successfully")
        print(f"[OK] Model parameters: {num_params / 1e6:.2f}M")
    except Exception as e:
        print(f"[ERROR] Error initializing model: {e}")
        return

    # ==================== 7. 初始化优化器 ====================
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    print(f"[OK] Optimizer: Adam (lr={config.learning_rate})")

    # ==================== 8. 训练循环 ====================
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    best_ade = float('inf')
    train_losses = []
    val_ades = []
    val_fdes = []

    try:
        for epoch in range(1, config.num_epochs + 1):
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch}/{config.num_epochs}")
            print(f"{'=' * 70}")
            
            # 训练
            train_loss = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                device=device,
                config=config
            )
            train_losses.append(train_loss)
            print(f"[TRAIN] Train Loss: {train_loss:.4f}")
            
            # 验证
            if epoch % config.eval_interval == 0:
                ade, fde = validate(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    config=config
                )
                val_ades.append(ade)
                val_fdes.append(fde)
                print(f"[VAL] Val ADE: {ade:.4f}, Val FDE: {fde:.4f}")

                # 保存最佳模型
                if ade < best_ade:
                    best_ade = ade
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ade': ade,
                        'fde': fde,
                        'config': config
                    }
                    torch.save(
                        checkpoint,
                        os.path.join(test_checkpoint_dir, 'best.pth')
                    )
                    print(f"[SAVE] Saved best model (ADE: {ade:.4f})")

    except Exception as e:
        print(f"\n[ERROR] Error during training: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==================== 9. 测试总结 ====================
    print("\n" + "=" * 70)
    print("[SUCCESS] Test completed!")
    print("=" * 70)
    
    print(f"\n[SUMMARY] Training Summary:")
    print(f"  - Initial train loss: {train_losses[0]:.4f}")
    print(f"  - Final train loss: {train_losses[-1]:.4f}")
    print(f"  - Loss reduction: {train_losses[0] - train_losses[-1]:.4f}")
    
    if val_ades:
        print(f"\n[SUMMARY] Validation Summary:")
        print(f"  - Best ADE: {best_ade:.4f}")
        print(f"  - Final ADE: {val_ades[-1]:.4f}")
        print(f"  - Final FDE: {val_fdes[-1]:.4f}")
    
    print(f"\n[SAVE] Model saved at: {os.path.join(test_checkpoint_dir, 'best.pth')}")
    
    # ==================== 10. 验证检查 ====================
    print("\n" + "=" * 70)
    print("Verification Checks:")
    print("=" * 70)
    
    checks_passed = 0
    total_checks = 4
    
    # Check 1: 训练损失是否下降
    if train_losses[0] > train_losses[-1]:
        print("[OK] Check 1: Training loss decreased")
    else:
        print("[WARN] Check 1: Training loss did not decrease (might need more epochs)")
    
    # Check 2: 损失是否为有效数值
    if all(not np.isnan(loss) and not np.isinf(loss) for loss in train_losses):
        print("[OK] Check 2: No NaN/Inf in training loss")
        checks_passed += 1
    else:
        print("[ERROR] Check 2: NaN/Inf detected in training loss")
    
    # Check 3: 验证指标是否合理
    if val_ades and all(0 < ade < 100 for ade in val_ades):
        print("[OK] Check 3: Validation metrics are reasonable")
        checks_passed += 1
    else:
        print("[WARN] Check 3: Validation metrics might be unreasonable")
    
    # Check 4: 模型是否保存成功
    if os.path.exists(os.path.join(test_checkpoint_dir, 'best.pth')):
        print("[OK] Check 4: Model checkpoint saved successfully")
        checks_passed += 1
    else:
        print("[ERROR] Check 4: Model checkpoint not found")
    
    print(f"\n{'=' * 70}")
    print(f"Checks passed: {checks_passed}/{total_checks}")
    print(f"{'=' * 70}")
    
    if checks_passed == total_checks:
        print("\n[SUCCESS] All checks passed! Ready for full training.")
    elif checks_passed >= 2:
        print("\n[WARN] Some checks failed, but basic functionality works.")
        print("   You can proceed with caution or investigate the warnings.")
    else:
        print("\n[ERROR] Multiple checks failed. Please investigate before full training.")


if __name__ == "__main__":
    main()
