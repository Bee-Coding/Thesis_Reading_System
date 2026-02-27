"""
nuScenes 数据集类

PyTorch Dataset 类，用于加载预处理后的 nuScenes 数据
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from typing import Dict, Optional


class NuScenesDataset(Dataset):
    """
    nuScenes 轨迹预测数据集
    
    加载预处理后的数据，包括：
    - 历史轨迹
    - 未来轨迹
    - Goal 点
    - BEV 特征
    - Goal 词汇表
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[callable] = None
    ):
        """
        初始化数据集
        
        Args:
            data_dir: 预处理数据目录
            split: 数据集划分 ('train', 'val', 'test')
            transform: 数据增强函数（可选）
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # 加载数据
        print(f"加载 {split} 数据集...")
        self.history = np.load(os.path.join(data_dir, 'history.npy'))
        self.future = np.load(os.path.join(data_dir, 'future.npy'))
        self.goals = np.load(os.path.join(data_dir, 'goals.npy'))
        self.bev_features = np.load(os.path.join(data_dir, 'bev_features.npy'))
        self.vocabulary = np.load(os.path.join(data_dir, 'vocabulary.npy'))
        
        print(f"  加载了 {len(self.goals)} 个样本")
        print(f"  History shape: {self.history.shape}")
        print(f"  Future shape: {self.future.shape}")
        print(f"  Goals shape: {self.goals.shape}")
        print(f"  BEV shape: {self.bev_features.shape}")
        print(f"  Vocabulary shape: {self.vocabulary.shape}")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.goals)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
        
        Returns:
            sample: 字典，包含：
                - 'history': (history_frames, 2) 历史轨迹
                - 'future': (future_frames, 2) 未来轨迹
                - 'goal': (2,) 目标点
                - 'bev': (3, H, W) BEV 特征
                - 'vocabulary': (vocab_size, 2) 词汇表
        """
        # 获取数据
        history = self.history[idx]      # (history_frames, 2)
        future = self.future[idx]        # (future_frames, 2)
        goal = self.goals[idx]           # (2,)
        bev = self.bev_features[idx]     # (3, H, W)
        vocabulary = self.vocabulary     # (vocab_size, 2)
        
        # 数据增强（如果提供）
        if self.transform is not None:
            history, future, goal, bev = self.transform(history, future, goal, bev)
        
        # 转换为 Tensor
        sample = {
            'history': torch.from_numpy(history).float(),
            'future': torch.from_numpy(future).float(),
            'goal': torch.from_numpy(goal).float(),
            'bev': torch.from_numpy(bev).float(),
            'vocabulary': torch.from_numpy(vocabulary).float()
        }
        
        return sample
    
    def get_vocabulary(self) -> np.ndarray:
        """获取词汇表"""
        return self.vocabulary


def collate_fn(batch):
    """
    自定义 collate 函数，用于 DataLoader
    
    Args:
        batch: List[Dict]，每个元素是 __getitem__ 返回的字典
    
    Returns:
        batched_data: 字典，包含批次数据
    """
    # 提取所有键
    keys = batch[0].keys()
    
    # 对每个键进行批处理
    batched_data = {}
    for key in keys:
        if key == 'vocabulary':
            # 词汇表对所有样本都相同，只取第一个
            batched_data[key] = batch[0][key]
        else:
            # 其他数据进行堆叠
            batched_data[key] = torch.stack([item[key] for item in batch], dim=0)
    
    return batched_data


# ==================== 数据增强（可选） ====================

class RandomRotation:
    """随机旋转数据增强"""
    
    def __init__(self, max_angle: float = 30.0):
        """
        Args:
            max_angle: 最大旋转角度（度）
        """
        self.max_angle = max_angle
    
    def __call__(self, history, future, goal, bev):
        """
        应用随机旋转
        
        Args:
            history: (T1, 2) 历史轨迹
            future: (T2, 2) 未来轨迹
            goal: (2,) 目标点
            bev: (3, H, W) BEV 特征
        
        Returns:
            旋转后的数据
        """
        # 随机旋转角度
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        angle_rad = np.deg2rad(angle)
        
        # 旋转矩阵
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # 旋转轨迹
        history = history @ R.T
        future = future @ R.T
        goal = goal @ R
        
        # 旋转 BEV（使用 OpenCV）
        import cv2
        h, w = bev.shape[1:]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        bev_rotated = np.zeros_like(bev)
        for i in range(3):
            bev_rotated[i] = cv2.warpAffine(bev[i], M, (w, h))
        
        return history, future, goal, bev_rotated


class RandomFlip:
    """随机翻转数据增强"""
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: 翻转概率
        """
        self.p = p
    
    def __call__(self, history, future, goal, bev):
        """
        应用随机左右翻转
        
        Args:
            history: (T1, 2) 历史轨迹
            future: (T2, 2) 未来轨迹
            goal: (2,) 目标点
            bev: (3, H, W) BEV 特征
        
        Returns:
            翻转后的数据
        """
        if np.random.rand() < self.p:
            # 翻转 y 坐标
            history[:, 1] = -history[:, 1]
            future[:, 1] = -future[:, 1]
            goal[1] = -goal[1]
            
            # 翻转 BEV
            bev = bev[:, :, ::-1].copy()
        
        return history, future, goal, bev


class Compose:
    """组合多个数据增强"""
    
    def __init__(self, transforms):
        """
        Args:
            transforms: List[callable]，数据增强函数列表
        """
        self.transforms = transforms
    
    def __call__(self, history, future, goal, bev):
        """依次应用所有数据增强"""
        for t in self.transforms:
            history, future, goal, bev = t(history, future, goal, bev)
        return history, future, goal, bev


# ==================== 测试代码 ====================

if __name__ == "__main__":
    # 测试数据集加载
    print("测试 NuScenesDataset...")
    
    # 假设数据已经预处理
    data_dir = 'data/nuscenes_processed'
    
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        print("请先运行预处理脚本: python scripts/preprocess_nuscenes.py")
    else:
        # 创建数据集
        dataset = NuScenesDataset(data_dir, split='train')
        
        # 测试 __getitem__
        sample = dataset[0]
        print("\n样本数据:")
        for key, value in sample.items():
            if key != 'vocabulary':
                print(f"  {key}: {value.shape}")
        
        # 测试 DataLoader
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        batch = next(iter(dataloader))
        print("\n批次数据:")
        for key, value in batch.items():
            print(f"  {key}: {value.shape}")
        
        print("\n✅ 数据集测试通过！")
