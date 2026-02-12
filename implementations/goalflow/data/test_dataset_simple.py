"""
Simple test script for ToyGoalFlowDataset without matplotlib dependency
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import sys
import os

# Add path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import dataset class (without matplotlib)
class ToyGoalFlowDataset(torch.utils.data.Dataset):
    """Simplified dataset class without visualization"""
    
    def __init__(self, data_path: str, split: str = 'train', train_ratio: float = 0.8):
        super().__init__()
        
        # Load data
        data = np.load(data_path)
        
        trajectories = data['trajectories']
        goals = data['goals']
        start_points = data['start_points']
        self.vocabulary = data['vocabulary']
        bev_features = data['bev_features']
        drivable_area = data['drivable_area']
        
        # Split train/val
        num_samples = len(trajectories)
        num_train = int(num_samples * train_ratio)
        
        if split == 'train':
            self.trajectories = trajectories[:num_train]
            self.goals = goals[:num_train]
            self.start_points = start_points[:num_train]
            self.bev_features = bev_features[:num_train]
            self.drivable_area = drivable_area[:num_train]
        else:
            self.trajectories = trajectories[num_train:]
            self.goals = goals[num_train:]
            self.start_points = start_points[num_train:]
            self.bev_features = bev_features[num_train:]
            self.drivable_area = drivable_area[num_train:]
        
        print(f"✅ Loaded {split} dataset: {len(self)} samples")
        print(f"   - Trajectory shape: {self.trajectories.shape}")
        print(f"   - Goal shape: {self.goals.shape}")
        print(f"   - BEV feature shape: {self.bev_features.shape}")
        print(f"   - Drivable area shape: {self.drivable_area.shape}")
        print(f"   - Vocabulary shape: {self.vocabulary.shape}")
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        return {
            'trajectory': torch.from_numpy(self.trajectories[idx]).float(),
            'goal': torch.from_numpy(self.goals[idx]).float(),
            'start_point': torch.from_numpy(self.start_points[idx]).float(),
            'bev_feature': torch.from_numpy(self.bev_features[idx]).float(),
            'drivable_area': torch.from_numpy(self.drivable_area[idx]).float(),
        }
    
    def get_vocabulary(self):
        return torch.from_numpy(self.vocabulary).float()


def test_dataset(data_path: str):
    """Test dataset loading and batching"""
    print("🧪 Testing ToyGoalFlowDataset...\n")
    
    # Create datasets
    train_dataset = ToyGoalFlowDataset(data_path, split='train')
    print()
    val_dataset = ToyGoalFlowDataset(data_path, split='val')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"\n📊 Dataset statistics:")
    print(f"   - Train samples: {len(train_dataset)}")
    print(f"   - Val samples: {len(val_dataset)}")
    print(f"   - Train batches: {len(train_loader)}")
    print(f"   - Val batches: {len(val_loader)}")
    
    # Test a single batch
    print(f"\n🔍 Testing batch loading...")
    batch = next(iter(train_loader))
    
    print(f"   Batch shapes:")
    for key, value in batch.items():
        print(f"   - {key}: {value.shape}")
    
    vocabulary = train_dataset.get_vocabulary()
    print(f"   - vocabulary: {vocabulary.shape}")
    
    # Verify data ranges
    print(f"\n📈 Data statistics:")
    print(f"   Trajectory range: [{batch['trajectory'].min():.2f}, {batch['trajectory'].max():.2f}]")
    print(f"   Goal range: [{batch['goal'].min():.2f}, {batch['goal'].max():.2f}]")
    print(f"   BEV feature range: [{batch['bev_feature'].min():.2f}, {batch['bev_feature'].max():.2f}]")
    print(f"   Drivable area range: [{batch['drivable_area'].min():.2f}, {batch['drivable_area'].max():.2f}]")
    
    # Check goal distribution (should be in 4 regions)
    goals = batch['goal'].numpy()
    print(f"\n🎯 Goal distribution:")
    print(f"   Mean: ({goals[:, 0].mean():.2f}, {goals[:, 1].mean():.2f})")
    print(f"   Std: ({goals[:, 0].std():.2f}, {goals[:, 1].std():.2f})")
    
    # Count goals in each quadrant
    q1 = ((goals[:, 0] > 0) & (goals[:, 1] > 0)).sum()
    q2 = ((goals[:, 0] < 0) & (goals[:, 1] > 0)).sum()
    q3 = ((goals[:, 0] < 0) & (goals[:, 1] < 0)).sum()
    q4 = ((goals[:, 0] > 0) & (goals[:, 1] < 0)).sum()
    print(f"   Quadrant distribution: Q1={q1}, Q2={q2}, Q3={q3}, Q4={q4}")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    data_path = "data/toy_data.npz"
    test_dataset(data_path)
