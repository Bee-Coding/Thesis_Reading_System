"""
Toy Data Generator for GoalFlow

Generates simplified trajectory data for quick validation of the GoalFlow pipeline.

Strategy:
- Start points: Random near origin N(0, 1)
- End points: 4 regions (multimodal simulation)
- Trajectories: Smooth curves using cubic spline
- BEV features: Simplified random patterns
- Drivable area: Circular region in center
"""

import numpy as np
import argparse
from scipy.interpolate import CubicSpline
from sklearn.cluster import KMeans


def generate_toy_trajectories(num_samples=1000, num_points=6, seed=42):
    """
    Generate simulated trajectory data with multimodal goals.
    
    Args:
        num_samples: Number of trajectories to generate
        num_points: Number of points per trajectory (default: 6)
        seed: Random seed for reproducibility
    
    Returns:
        trajectories: (N, T, 2) - N trajectories with T points
        goals: (N, 2) - Goal points (endpoints)
        start_points: (N, 2) - Start points
    """
    np.random.seed(seed)
    
    # Define 4 goal regions (multimodal)
    goal_regions = np.array([
        [10.0, 10.0],   # Region 1: top-right
        [10.0, -10.0],  # Region 2: bottom-right
        [-10.0, 10.0],  # Region 3: top-left
        [-10.0, -10.0]  # Region 4: bottom-left
    ])
    
    trajectories = []
    goals = []
    start_points = []
    
    for i in range(num_samples):
        # Random start point near origin
        start = np.random.randn(2) * 1.0
        
        # Select random goal region
        region_idx = np.random.randint(0, 4)
        goal = goal_regions[region_idx] + np.random.randn(2) * 1.5  # Add noise
        
        # Generate smooth trajectory using cubic spline
        # Create intermediate control points
        t_control = np.array([0.0, 0.33, 0.67, 1.0])
        
        # Linear interpolation with some curvature
        mid1 = start + (goal - start) * 0.33 + np.random.randn(2) * 2.0
        mid2 = start + (goal - start) * 0.67 + np.random.randn(2) * 2.0
        
        control_points = np.array([start, mid1, mid2, goal])
        
        # Create cubic spline
        cs_x = CubicSpline(t_control, control_points[:, 0])
        cs_y = CubicSpline(t_control, control_points[:, 1])
        
        # Sample trajectory points
        t_sample = np.linspace(0, 1, num_points)
        traj_x = cs_x(t_sample)
        traj_y = cs_y(t_sample)
        
        trajectory = np.stack([traj_x, traj_y], axis=-1)  # (T, 2)
        
        # Add small noise for realism
        trajectory += np.random.randn(num_points, 2) * 0.1
        
        trajectories.append(trajectory)
        goals.append(goal)
        start_points.append(start)
    
    trajectories = np.array(trajectories)  # (N, T, 2)
    goals = np.array(goals)  # (N, 2)
    start_points = np.array(start_points)  # (N, 2)
    
    return trajectories, goals, start_points


def build_vocabulary(goals, n_clusters=128, seed=42):
    """
    Build goal point vocabulary using K-means clustering.
    
    Args:
        goals: (N, 2) - All goal points
        n_clusters: Number of vocabulary points (default: 128)
        seed: Random seed
    
    Returns:
        vocabulary: (n_clusters, 2) - Cluster centers
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    kmeans.fit(goals)
    vocabulary = kmeans.cluster_centers_  # (n_clusters, 2)
    
    return vocabulary


def generate_bev_features(num_samples=1000, channels=64, height=32, width=32, seed=42):
    """
    Generate simplified BEV features.
    
    For toy data, we use random patterns or simple geometric features.
    
    Args:
        num_samples: Number of samples
        channels: Number of feature channels (default: 64)
        height: BEV height (default: 32)
        width: BEV width (default: 32)
        seed: Random seed
    
    Returns:
        bev_features: (N, C, H, W) - BEV feature maps
    """
    np.random.seed(seed)
    
    # Generate random features with some structure
    bev_features = np.random.randn(num_samples, channels, height, width).astype(np.float32)
    
    # Add some spatial structure (e.g., gradient patterns)
    x_grid = np.linspace(-1, 1, width)
    y_grid = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(x_grid, y_grid)
    
    # Add circular pattern to first few channels
    for i in range(min(4, channels)):
        pattern = np.exp(-(xx**2 + yy**2) / 0.5)
        bev_features[:, i, :, :] += pattern[None, :, :]
    
    return bev_features


def generate_drivable_area(num_samples=1000, height=32, width=32, seed=42):
    """
    Generate simplified drivable area masks.
    
    For toy data, we use a circular region in the center.
    
    Args:
        num_samples: Number of samples
        height: Mask height (default: 32)
        width: Mask width (default: 32)
        seed: Random seed
    
    Returns:
        drivable_area: (N, H, W) - Binary masks (1=drivable, 0=not drivable)
    """
    np.random.seed(seed)
    
    # Create circular drivable region
    x_grid = np.linspace(-1, 1, width)
    y_grid = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(x_grid, y_grid)
    
    # Circular mask with radius ~0.8
    radius = 0.8 + np.random.randn(num_samples, 1, 1) * 0.1  # Slight variation
    distance = np.sqrt(xx**2 + yy**2)
    
    drivable_area = (distance[None, :, :] < radius).astype(np.float32)  # (N, H, W)
    
    return drivable_area


def save_toy_data(output_path, trajectories, goals, start_points, vocabulary, 
                  bev_features, drivable_area):
    """
    Save all generated data to a single .npz file.
    
    Args:
        output_path: Path to save the .npz file
        trajectories: (N, T, 2)
        goals: (N, 2)
        start_points: (N, 2)
        vocabulary: (n_clusters, 2)
        bev_features: (N, C, H, W)
        drivable_area: (N, H, W)
    """
    np.savez(
        output_path,
        trajectories=trajectories,
        goals=goals,
        start_points=start_points,
        vocabulary=vocabulary,
        bev_features=bev_features,
        drivable_area=drivable_area
    )
    print(f"✅ Saved toy data to {output_path}")
    print(f"   - Trajectories: {trajectories.shape}")
    print(f"   - Goals: {goals.shape}")
    print(f"   - Start points: {start_points.shape}")
    print(f"   - Vocabulary: {vocabulary.shape}")
    print(f"   - BEV features: {bev_features.shape}")
    print(f"   - Drivable area: {drivable_area.shape}")


def main():
    parser = argparse.ArgumentParser(description="Generate toy data for GoalFlow")
    parser.add_argument("--num_samples", type=int, default=1000, 
                        help="Number of trajectories to generate")
    parser.add_argument("--num_points", type=int, default=6,
                        help="Number of points per trajectory")
    parser.add_argument("--n_clusters", type=int, default=128,
                        help="Number of vocabulary points")
    parser.add_argument("--output", type=str, default="data/toy_data.npz",
                        help="Output path for .npz file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    print(f"🚀 Generating toy data with {args.num_samples} samples...")
    
    # Generate trajectories
    print("📍 Generating trajectories...")
    trajectories, goals, start_points = generate_toy_trajectories(
        num_samples=args.num_samples,
        num_points=args.num_points,
        seed=args.seed
    )
    
    # Build vocabulary
    print("📚 Building vocabulary...")
    vocabulary = build_vocabulary(goals, n_clusters=args.n_clusters, seed=args.seed)
    
    # Generate BEV features
    print("🗺️  Generating BEV features...")
    bev_features = generate_bev_features(num_samples=args.num_samples, seed=args.seed)
    
    # Generate drivable area
    print("🛣️  Generating drivable area...")
    drivable_area = generate_drivable_area(num_samples=args.num_samples, seed=args.seed)
    
    # Save data
    print("💾 Saving data...")
    save_toy_data(
        output_path=args.output,
        trajectories=trajectories,
        goals=goals,
        start_points=start_points,
        vocabulary=vocabulary,
        bev_features=bev_features,
        drivable_area=drivable_area
    )
    
    print("\n✨ Done! You can now use this data with ToyGoalFlowDataset.")


if __name__ == "__main__":
    main()
