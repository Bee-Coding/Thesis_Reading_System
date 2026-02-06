"""
Toy Dataset for Flow Matching

生成简单的2D轨迹数据用于验证Flow Matching算法
包含四种轨迹类型：圆形、直线、S形、二次多项式
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Literal, Tuple, Optional


class TrajectoryGenerator:
    """轨迹生成器：生成不同类型的2D轨迹"""
    
    def __init__(
        self,
        num_points: int = 6,
        output_range: Tuple[float, float] = (-20, 20),
        radius_range: Tuple[float, float] = (5.2, 9),
        length_range: Tuple[float, float] = (5, 15)
    ):
        """
        Args:
            num_points: 轨迹点数量
            output_range: s输出范围 (min, max)
            radius_range: 圆形半径范围 (min, max)
            length_range: 直线/曲线长度范围 (min, max)
        """
        self.num_points = num_points
        self.output_range = output_range
        self.radius_range = radius_range
        self.length_range = length_range
    
    def generate_circle(self) -> np.ndarray:
        """
        生成圆形轨迹（自车中心坐标系）
        自车在原点 (0, 0)，沿圆弧轨迹移动
        
        Returns:
            trajectory: shape (num_points, 2)
        """
        # 随机半径
        radius = np.random.uniform(*self.radius_range)
        
        # 随机起始角度
        start_angle = np.random.uniform(0, 2 * np.pi)

        # 随机圆弧角度
        arc_angle = np.random.uniform(np.pi / 4, np.pi)

        # 圆心坐标（使自车在圆周上）
        center_x = 0.0 - radius * np.cos(start_angle)
        center_y = 0.0 - radius * np.sin(start_angle)
        
        # 生成圆周上的点
        angles = np.linspace(start_angle, start_angle + arc_angle, 
                            self.num_points, endpoint=False)
        x = center_x + radius * np.cos(angles)
        y = center_y + radius * np.sin(angles)

        trajectory = np.stack([x, y], axis=1)  # shape: (num_points, 2)
        
        # 裁剪到output_range范围内
        if self.output_range:
            trajectory = np.clip(trajectory, 
                               self.output_range[0], 
                               self.output_range[1])
        
        return trajectory
    
    def generate_line(self) -> np.ndarray:
        """
        生成直线轨迹（自车中心坐标系）
        自车在原点 (0, 0)，沿x轴方向直线前进
        
        Returns:
            trajectory: shape (num_points, 2)
        """
        start_x = 0.0
        start_y = 0.0
        
        # 随机长度
        length = np.random.uniform(*self.length_range)
        
        # 生成直线上的点（沿x轴方向）
        t = np.linspace(0, 1, self.num_points)
        x = start_x + length * t
        y = np.full(self.num_points, start_y)  # y保持为0，符合车辆运动学

        trajectory = np.stack([x, y], axis=1)  # shape: (num_points, 2)
        
        # 裁剪到output_range范围内
        if self.output_range:
            trajectory = np.clip(trajectory, 
                               self.output_range[0], 
                               self.output_range[1])
        
        return trajectory 
    
    def generate_s_curve(self) -> np.ndarray:
        """
        生成S形曲线轨迹（自车中心坐标系）
        自车在原点 (0, 0)，沿S形曲线移动
        
        Returns:
            trajectory: shape (num_points, 2)
        """
        # 随机振幅和长度
        amplitude = np.random.uniform(2, 5)
        length = np.random.uniform(*self.length_range)
        
        # 生成S形曲线
        t = np.linspace(0, 1, self.num_points)
        x = length * t
        y = amplitude * np.sin(2 * np.pi * t)
        
        trajectory = np.stack([x, y], axis=1)  # shape: (num_points, 2)
        
        # 裁剪到output_range范围内
        if self.output_range:
            trajectory = np.clip(trajectory, 
                               self.output_range[0], 
                               self.output_range[1])
        
        return trajectory 
    
    def generate_polynomial(self) -> np.ndarray:
        """
        生成二次多项式（抛物线）轨迹（自车中心坐标系）
        自车在原点 (0, 0)，沿抛物线轨迹移动
        
        Returns:
            trajectory: shape (num_points, 2)
        """
        # 随机二次多项式系数
        # y = a*x^2 + b*x (相对于起点)
        a = np.random.uniform(-0.5, 0.5)
        b = np.random.uniform(-2, 2)
        
        # 随机x方向的长度
        length = np.random.uniform(*self.length_range)
        
        # 生成抛物线上的点
        t = np.linspace(0, 1, self.num_points)
        x = length * t  # x坐标
        y = a * x**2 + b * x  # y坐标
        
        trajectory = np.stack([x, y], axis=1)  # shape: (num_points, 2)
        
        # 裁剪到output_range范围内
        if self.output_range:
            trajectory = np.clip(trajectory, 
                               self.output_range[0], 
                               self.output_range[1])
        
        return trajectory 
    

    def generate(self, traj_type: str) -> np.ndarray:
        """     根据类型生成轨迹
        
        Args:
            traj_type: 轨迹类型 ('circle', 'line', 's_curve', 'polynomial')
            
        Returns:
            trajectory: shape (num_points, 2)
        """
        if traj_type == 'circle':
            return self.generate_circle()
        elif traj_type == 'line':
            return self.generate_line()
        elif traj_type == 's_curve':
            return self.generate_s_curve()
        elif traj_type == 'polynomial':
            return self.generate_polynomial()
        else:
            raise ValueError(f"Unknown trajectory type: {traj_type}")


class ToyTrajectoryDataset(Dataset):
    """Toy轨迹数据集"""
    
    def __init__(
        self,
        data_path: str,
        transform: Optional[callable] = None
    ):
        """
        Args:
            data_path: npz文件路径
            transform: 可选的数据变换
        """
        # 加载npz文件
        data = np.load(data_path)
        self.trajectories = data['trajectories']  # shape: (N, 6, 2)
        self.types = data['types']  # shape: (N,)
        self.transform = transform
        
        print(f"Loaded {len(self)} trajectories from {data_path}")
        print(f"Trajectory shape: {self.trajectories.shape}")
        print(f"Types: {np.unique(self.types)}")
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            {
                'trajectory': torch.Tensor, shape (6, 2)
                'type': str
            }
        """
        trajectory = self.trajectories[idx]  # shape: (6, 2)
        traj_type = str(self.types[idx])
        cond = self._generate_condition(trajectory, traj_type)
        
        if self.transform:
            trajectory = self.transform(trajectory)
        
        return {
            'trajectory': torch.from_numpy(trajectory).float(),
            'type': traj_type,
            'condition': cond
        }

    def _generate_condition(self, trajectory: np.ndarray, traj_type: str) -> torch.Tensor:
        """
        生成条件向量
        
        Args:
            trajectory: numpy array, shape (6, 2)
            traj_type: 轨迹类型字符串
            
        Returns:
            condition: torch.Tensor, shape (8,) = goal(2) + direction(2) + type_onehot(4)
        """
        # Goal Point: 终点坐标作为goal point
        goal_point = trajectory[-1]  # shape: (2,)
        
        # Ego State： 起点速度方向
        direction = trajectory[1] - trajectory[0]  # shape: (2,)
        
        # Trajectory Type: one-hot编码
        type_map = {'circle': 0, 'line': 1, 's_curve': 2, 'polynomial': 3}
        type_onehot = np.zeros(4, dtype=np.float32)
        type_onehot[type_map[traj_type]] = 1.0

        # 拼接所有条件 (全部使用 numpy)
        cond = np.concatenate([goal_point, direction, type_onehot])  # shape: (8,)
        return torch.from_numpy(cond).float()


def generate_and_save_dataset(
    save_dir: str,
    train_size: int = 5000,
    val_size: int = 500,
    num_points: int = 6,
    output_range: Tuple[float, float] = (-20, 20),
    radius_range: Tuple[float, float] = (5.2, 9),
    length_range: Tuple[float, float] = (5, 15),
    seed: int = 42
):
    """
    生成并保存toy dataset
    
    Args:
        save_dir: 保存目录
        train_size: 训练集大小
        val_size: 验证集大小
        num_points: 每条轨迹的点数
        output_range: 坐标范围
        radius_range: 圆形半径范围
        length_range: 直线/曲线长度范围
        seed: 随机种子
    """
    np.random.seed(seed)
    
    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建生成器
    generator = TrajectoryGenerator(
        num_points=num_points,
        output_range=output_range,
        radius_range=radius_range,
        length_range=length_range
    )
    
    # 轨迹类型
    traj_types = ['circle', 'line', 's_curve', 'polynomial']
    
    def generate_split(size: int, split_name: str):
        """生成一个数据集分割"""
        trajectories = []
        types = []
        
        print(f"\nGenerating {split_name} set ({size} samples)...")
        
        for i in range(size):
            # 随机选择轨迹类型
            traj_type = np.random.choice(traj_types)
            
            # 生成轨迹
            trajectory = generator.generate(traj_type)
            
            trajectories.append(trajectory)
            types.append(traj_type)
            
            if (i + 1) % 1000 == 0:
                print(f"  Generated {i + 1}/{size} trajectories")
        
        # 转换为numpy数组
        trajectories = np.array(trajectories)  # shape: (size, num_points, 2)
        types = np.array(types)  # shape: (size,)
        
        # 统计每种类型的数量
        print(f"\n{split_name} set statistics:")
        for traj_type in traj_types:
            count = np.sum(types == traj_type)
            print(f"  {traj_type}: {count} ({count/size*100:.1f}%)")
        
        return trajectories, types
    
    # 生成训练集
    train_trajectories, train_types = generate_split(train_size, "Train")
    train_path = save_dir / "toy_train.npz"
    np.savez(train_path, trajectories=train_trajectories, types=train_types)
    print(f"\nSaved train set to {train_path}")
    
    # 生成验证集
    val_trajectories, val_types = generate_split(val_size, "Validation")
    val_path = save_dir / "toy_val.npz"
    np.savez(val_path, trajectories=val_trajectories, types=val_types)
    print(f"Saved validation set to {val_path}")
    
    print("\n" + "="*60)
    print("Dataset generation completed!")
    print("="*60)


if __name__ == "__main__":
    # 测试轨迹生成器
    print("="*60)
    print("Testing Trajectory Generator")
    print("="*60)
    
    generator = TrajectoryGenerator(
        num_points=6,
        output_range=(-20, 20),
        radius_range=(5.2, 9),
        length_range=(5, 15)
    )
    
    # 测试每种类型
    traj_types = ['circle', 'line', 's_curve', 'polynomial']
    for traj_type in traj_types:
        trajectory = generator.generate(traj_type)
        print(f"\n{traj_type} trajectory:")
        print(f"  Shape: {trajectory.shape}")
        print(f"  X range: [{trajectory[:, 0].min():.2f}, {trajectory[:, 0].max():.2f}]")
        print(f"  Y range: [{trajectory[:, 1].min():.2f}, {trajectory[:, 1].max():.2f}]")
        print(f"  First point: ({trajectory[0, 0]:.2f}, {trajectory[0, 1]:.2f})")
        print(f"  Last point: ({trajectory[-1, 0]:.2f}, {trajectory[-1, 1]:.2f})")
    
    # 生成并保存数据集
    print("\n" + "="*60)
    print("Generating and Saving Dataset")
    print("="*60)
    
    save_dir = Path(__file__).parent
    print(f"save_path is {save_dir}")
    generate_and_save_dataset(
        save_dir=str(save_dir),
        train_size=5000,
        val_size=500,
        num_points=6,
        output_range=(-20, 20),
        radius_range=(5.2, 9),
        length_range=(5, 15),
        seed=42
    )
    
    # 测试数据加载
    print("\n" + "="*60)
    print("Testing Dataset Loading")
    print("="*60)
    
    train_dataset = ToyTrajectoryDataset(str(save_dir / "toy_train.npz"))
    val_dataset = ToyTrajectoryDataset(str(save_dir / "toy_val.npz"))
    
    # 测试获取样本
    sample = train_dataset[0]
    print(f"\nSample from train set:")
    print(f"  Trajectory shape: {sample['trajectory'].shape}")
    print(f"  Trajectory dtype: {sample['trajectory'].dtype}")
    print(f"  Type: {sample['type']}")
    
    # 测试DataLoader
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    batch = next(iter(train_loader))
    
    print(f"\nBatch from DataLoader:")
    print(f"  Trajectory shape: {batch['trajectory'].shape}")
    print(f"  Types: {batch['type'][:5]}")  # 显示前5个
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
