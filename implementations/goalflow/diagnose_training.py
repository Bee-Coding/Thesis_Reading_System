"""
诊断训练卡住问题的测试脚本
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.goal_flow_matcher import GoalFlowMatcher
from config.matcher_config import MatcherConfig
from data.toy_goalflow_dataset import ToyGoalFlowDataset

print("=" * 60)
print("Diagnostic Test")
print("=" * 60)

# 1. 加载配置
config = MatcherConfig()
config.num_workers = 0  # 确保为 0
device = torch.device('cpu')
print(f"Device: {device}")
print(f"Num workers: {config.num_workers}")

# 2. 加载数据
print("\nLoading data...")
train_dataset = ToyGoalFlowDataset(config.data_path, split='train')
train_loader = DataLoader(
    train_dataset,
    batch_size=2,  # 使用很小的 batch size
    shuffle=False,
    num_workers=0  # 强制为 0
)
print(f"Dataset loaded: {len(train_dataset)} samples")
print(f"Batches: {len(train_loader)}")

# 3. 测试数据加载
print("\nTesting data loading...")
try:
    batch = next(iter(train_loader))
    print(f"[OK] First batch loaded successfully")
    print(f"  - trajectory shape: {batch['trajectory'].shape}")
    print(f"  - goal shape: {batch['goal'].shape}")
    print(f"  - bev_feature shape: {batch['bev_feature'].shape}")
except Exception as e:
    print(f"[ERROR] Error loading batch: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 初始化模型
print("\nInitializing model...")
try:
    model = GoalFlowMatcher(
        traj_dim=2,
        num_traj_points=6,
        d_model=128,  # 减小模型
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        scene_channels=64,
    ).to(device)
    print(f"[OK] Model initialized")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
except Exception as e:
    print(f"[ERROR] Error initializing model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. 测试前向传播
print("\nTesting forward pass...")
try:
    x_1 = batch["trajectory"].to(device)
    goal = batch["goal"].to(device)
    bev_feature = batch["bev_feature"].to(device)
    
    print(f"  - x_1: {x_1.shape}")
    print(f"  - goal: {goal.shape}")
    print(f"  - bev_feature: {bev_feature.shape}")
    
    # 采样噪声和时间
    x_0 = torch.randn_like(x_1)
    B = x_1.shape[0]
    t = torch.rand(B, device=device)
    
    print(f"  - x_0: {x_0.shape}")
    print(f"  - t: {t.shape}")
    
    # 前向传播
    print("\n  Computing loss...")
    loss = model.compute_loss(x_0, x_1, goal, bev_feature, t)
    print(f"[OK] Forward pass successful")
    print(f"  - Loss: {loss.item():.4f}")
    
except Exception as e:
    print(f"[ERROR] Error in forward pass: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 6. 测试反向传播
print("\nTesting backward pass...")
try:
    loss.backward()
    print(f"[OK] Backward pass successful")
except Exception as e:
    print(f"[ERROR] Error in backward pass: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 7. 测试完整训练循环
print("\nTesting training loop (3 batches)...")
try:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for i, batch in enumerate(train_loader):
        if i >= 3:  # 只测试 3 个 batch
            break
            
        x_1 = batch["trajectory"].to(device)
        goal = batch["goal"].to(device)
        bev_feature = batch["bev_feature"].to(device)
        
        x_0 = torch.randn_like(x_1)
        B = x_1.shape[0]
        t = torch.rand(B, device=device)
        
        loss = model.compute_loss(x_0, x_1, goal, bev_feature, t)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  Batch {i+1}: loss = {loss.item():.4f}")
    
    print(f"[OK] Training loop successful")
    
except Exception as e:
    print(f"[ERROR] Error in training loop: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
