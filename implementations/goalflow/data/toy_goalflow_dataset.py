"""
Toy GoalFlow Dataset

PyTorch Dataset class for loading and batching toy trajectory data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse

# Set matplotlib backend before importing pyplot to avoid display issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


class ToyGoalFlowDataset(Dataset):
    """
    PyTorch Dataset for GoalFlow toy data.
    
    Data structure:
    {
        'trajectory': (B, T, 2),      # Trajectory points
        'goal': (B, 2),               # Goal point
        'start_point': (B, 2),        # Start point
        'bev_feature': (B, C, H, W),  # BEV features
        'drivable_area': (B, H, W),   # Drivable area mask
        'vocabulary': (N, 2)          # Shared vocabulary (not batched)
    }
    """
    
    def __init__(self, data_path: str, split: str = 'train', train_ratio: float = 0.8):
        """
        Args:
            data_path: Path to .npz file
            split: 'train' or 'val'
            train_ratio: Ratio of training data (default: 0.8)
        """
        super().__init__()
        
        # Load data
        data = np.load(data_path)
        
        trajectories = data['trajectories']  # (N, T, 2)
        goals = data['goals']  # (N, 2)
        start_points = data['start_points']  # (N, 2)
        self.vocabulary = data['vocabulary']  # (n_clusters, 2)
        bev_features = data['bev_features']  # (N, C, H, W)
        drivable_area = data['drivable_area']  # (N, H, W)
        
        # Split train/val
        num_samples = len(trajectories)
        num_train = int(num_samples * train_ratio)
        
        if split == 'train':
            self.trajectories = trajectories[:num_train]
            self.goals = goals[:num_train]
            self.start_points = start_points[:num_train]
            self.bev_features = bev_features[:num_train]
            self.drivable_area = drivable_area[:num_train]
        else:  # val
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
        """
        Returns a single sample as a dictionary.
        """
        return {
            'trajectory': torch.from_numpy(self.trajectories[idx]).float(),  # (T, 2)
            'goal': torch.from_numpy(self.goals[idx]).float(),  # (2,)
            'start_point': torch.from_numpy(self.start_points[idx]).float(),  # (2,)
            'bev_feature': torch.from_numpy(self.bev_features[idx]).float(),  # (C, H, W)
            'drivable_area': torch.from_numpy(self.drivable_area[idx]).float(),  # (H, W)
        }
    
    def get_vocabulary(self):
        """
        Returns the vocabulary as a torch tensor.
        This is shared across all samples.
        """
        return torch.from_numpy(self.vocabulary).float()  # (N, 2)


def visualize_sample(dataset: ToyGoalFlowDataset, idx: int = 0, save_path: str = None):
    """
    Visualize a single sample from the dataset.
    
    Args:
        dataset: ToyGoalFlowDataset instance
        idx: Sample index to visualize
        save_path: Optional path to save the figure
    """
    sample = dataset[idx]
    vocabulary = dataset.get_vocabulary()
    
    trajectory = sample['trajectory'].numpy()  # (T, 2)
    goal = sample['goal'].numpy()  # (2,)
    start_point = sample['start_point'].numpy()  # (2,)
    bev_feature = sample['bev_feature'].numpy()  # (C, H, W)
    drivable_area = sample['drivable_area'].numpy()  # (H, W)
    vocabulary_np = vocabulary.numpy()  # (N, 2)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Trajectory and vocabulary
    ax = axes[0]
    ax.scatter(vocabulary_np[:, 0], vocabulary_np[:, 1], 
               c='gray', s=10, alpha=0.3, label='Vocabulary')
    ax.plot(trajectory[:, 0], trajectory[:, 1], 
            'b-o', linewidth=2, markersize=6, label='Trajectory')
    ax.scatter(start_point[0], start_point[1], 
               c='green', s=100, marker='s', label='Start', zorder=5)
    ax.scatter(goal[0], goal[1], 
               c='red', s=100, marker='*', label='Goal', zorder=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Trajectory and Goal Vocabulary')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 2: BEV feature (first channel)
    ax = axes[1]
    im = ax.imshow(bev_feature[0], cmap='viridis', origin='lower')
    ax.set_title('BEV Feature (Channel 0)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax)
    
    # Plot 3: Drivable area
    ax = axes[2]
    im = ax.imshow(drivable_area, cmap='RdYlGn', origin='lower', vmin=0, vmax=1)
    ax.set_title('Drivable Area')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_batch(dataloader: DataLoader, save_path: str = None):
    """
    Visualize a batch of samples.
    
    Args:
        dataloader: DataLoader instance
        save_path: Optional path to save the figure
    """
    batch = next(iter(dataloader))
    
    trajectory = batch['trajectory']  # (B, T, 2)
    goal = batch['goal']  # (B, 2)
    start_point = batch['start_point']  # (B, 2)
    
    B = trajectory.shape[0]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot all trajectories in the batch
    for i in range(B):
        traj = trajectory[i].numpy()
        g = goal[i].numpy()
        s = start_point[i].numpy()
        
        ax.plot(traj[:, 0], traj[:, 1], '-o', linewidth=1, markersize=3, alpha=0.6)
        ax.scatter(s[0], s[1], c='green', s=30, marker='s', alpha=0.6)
        ax.scatter(g[0], g[1], c='red', s=50, marker='*', alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Batch of {B} Trajectories')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved batch visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def test_dataset(data_path: str):
    """
    Test the dataset loading and batching.
    """
    print("🧪 Testing ToyGoalFlowDataset...")
    
    # Create datasets
    train_dataset = ToyGoalFlowDataset(data_path, split='train')
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
    
    # Visualize
    print(f"\n🎨 Generating visualizations...")
    visualize_sample(train_dataset, idx=0, save_path='data/sample_visualization.png')
    visualize_batch(train_loader, save_path='data/batch_visualization.png')
    
    print("\n✅ All tests passed!")


def main():
    parser = argparse.ArgumentParser(description="Test ToyGoalFlowDataset")
    parser.add_argument("--data_path", type=str, default="data/toy_data.npz",
                        help="Path to .npz data file")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations")
    
    args = parser.parse_args()
    
    if args.visualize:
        test_dataset(args.data_path)
    else:
        # Just load and print info
        train_dataset = ToyGoalFlowDataset(args.data_path, split='train')
        val_dataset = ToyGoalFlowDataset(args.data_path, split='val')
        print(f"\n✅ Dataset loaded successfully!")


if __name__ == "__main__":
    main()
