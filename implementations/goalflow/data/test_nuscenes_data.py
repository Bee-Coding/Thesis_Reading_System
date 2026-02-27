"""
nuScenes 数据测试脚本

测试预处理后的数据是否正确

使用方法:
    python data/test_nuscenes_data.py
"""

import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.nuscenes_dataset import NuScenesDataset, collate_fn
from config.nuscenes_config import NuScenesConfig


def test_data_files():
    """测试数据文件是否存在"""
    print("=" * 60)
    print("测试数据文件")
    print("=" * 60)
    
    config = NuScenesConfig()
    
    splits = ['train', 'val', 'test']
    required_files = [
        'history.npy',
        'future.npy',
        'goals.npy',
        'bev_features.npy',
        'vocabulary.npy',
        'metadata.pkl'
    ]
    
    all_ok = True
    for split in splits:
        print(f"\n{split.upper()} 数据集:")
        split_dir = os.path.join(config.processed_data_dir, split)
        
        if not os.path.exists(split_dir):
            print(f"  ❌ 目录不存在: {split_dir}")
            all_ok = False
            continue
        
        for filename in required_files:
            filepath = os.path.join(split_dir, filename)
            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  ✅ {filename:20s} ({size_mb:.2f} MB)")
            else:
                print(f"  ❌ {filename:20s} (不存在)")
                all_ok = False
    
    return all_ok


def test_data_shapes():
    """测试数据形状"""
    print("\n" + "=" * 60)
    print("测试数据形状")
    print("=" * 60)
    
    config = NuScenesConfig()
    train_dir = os.path.join(config.processed_data_dir, 'train')
    
    if not os.path.exists(train_dir):
        print(f"❌ 训练数据目录不存在: {train_dir}")
        return False
    
    try:
        # 加载数据
        history = np.load(os.path.join(train_dir, 'history.npy'))
        future = np.load(os.path.join(train_dir, 'future.npy'))
        goals = np.load(os.path.join(train_dir, 'goals.npy'))
        bev_features = np.load(os.path.join(train_dir, 'bev_features.npy'))
        vocabulary = np.load(os.path.join(train_dir, 'vocabulary.npy'))
        
        n_samples = len(goals)
        
        print(f"\n样本数: {n_samples}")
        print(f"\nHistory shape: {history.shape}")
        print(f"  预期: ({n_samples}, {config.history_frames}, 2)")
        
        print(f"\nFuture shape: {future.shape}")
        print(f"  预期: ({n_samples}, {config.future_frames}, 2)")
        
        print(f"\nGoals shape: {goals.shape}")
        print(f"  预期: ({n_samples}, 2)")
        
        print(f"\nBEV features shape: {bev_features.shape}")
        print(f"  预期: ({n_samples}, 3, {config.bev_height}, {config.bev_width})")
        
        print(f"\nVocabulary shape: {vocabulary.shape}")
        print(f"  预期: ({config.vocab_size}, 2)")
        
        # 检查形状
        checks = [
            history.shape == (n_samples, config.history_frames, 2),
            future.shape == (n_samples, config.future_frames, 2),
            goals.shape == (n_samples, 2),
            bev_features.shape == (n_samples, 3, config.bev_height, config.bev_width),
            vocabulary.shape == (config.vocab_size, 2)
        ]
        
        if all(checks):
            print("\n✅ 所有形状正确")
            return True
        else:
            print("\n❌ 部分形状不正确")
            return False
        
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return False


def test_data_values():
    """测试数据值范围"""
    print("\n" + "=" * 60)
    print("测试数据值范围")
    print("=" * 60)
    
    config = NuScenesConfig()
    train_dir = os.path.join(config.processed_data_dir, 'train')
    
    try:
        # 加载数据
        history = np.load(os.path.join(train_dir, 'history.npy'))
        future = np.load(os.path.join(train_dir, 'future.npy'))
        goals = np.load(os.path.join(train_dir, 'goals.npy'))
        bev_features = np.load(os.path.join(train_dir, 'bev_features.npy'))
        
        print("\nHistory 统计:")
        print(f"  范围: [{history.min():.2f}, {history.max():.2f}]")
        print(f"  均值: {history.mean():.2f}")
        print(f"  标准差: {history.std():.2f}")
        
        print("\nFuture 统计:")
        print(f"  范围: [{future.min():.2f}, {future.max():.2f}]")
        print(f"  均值: {future.mean():.2f}")
        print(f"  标准差: {future.std():.2f}")
        
        print("\nGoals 统计:")
        print(f"  范围: [{goals.min():.2f}, {goals.max():.2f}]")
        print(f"  均值: {goals.mean():.2f}")
        print(f"  标准差: {goals.std():.2f}")
        
        print("\nBEV features 统计:")
        print(f"  范围: [{bev_features.min():.2f}, {bev_features.max():.2f}]")
        print(f"  均值: {bev_features.mean():.2f}")
        print(f"  非零比例: {(bev_features > 0).mean():.2%}")
        
        # 检查是否有 NaN 或 Inf
        has_nan = np.isnan(history).any() or np.isnan(future).any() or np.isnan(goals).any()
        has_inf = np.isinf(history).any() or np.isinf(future).any() or np.isinf(goals).any()
        
        if has_nan:
            print("\n❌ 数据包含 NaN 值")
            return False
        
        if has_inf:
            print("\n❌ 数据包含 Inf 值")
            return False
        
        print("\n✅ 数据值范围正常")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_dataset_loading():
    """测试 Dataset 类加载"""
    print("\n" + "=" * 60)
    print("测试 Dataset 类")
    print("=" * 60)
    
    config = NuScenesConfig()
    train_dir = os.path.join(config.processed_data_dir, 'train')
    
    try:
        # 创建数据集
        dataset = NuScenesDataset(train_dir, split='train')
        
        print(f"\n数据集大小: {len(dataset)}")
        
        # 测试 __getitem__
        sample = dataset[0]
        
        print("\n样本数据:")
        for key, value in sample.items():
            if key != 'vocabulary':
                print(f"  {key:15s}: {value.shape} ({value.dtype})")
        
        print(f"  {'vocabulary':15s}: {sample['vocabulary'].shape} ({sample['vocabulary'].dtype})")
        
        print("\n✅ Dataset 加载成功")
        return True
        
    except Exception as e:
        print(f"❌ Dataset 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader():
    """测试 DataLoader"""
    print("\n" + "=" * 60)
    print("测试 DataLoader")
    print("=" * 60)
    
    config = NuScenesConfig()
    train_dir = os.path.join(config.processed_data_dir, 'train')
    
    try:
        # 创建数据集
        dataset = NuScenesDataset(train_dir, split='train')
        
        # 创建 DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # 测试一个 batch
        batch = next(iter(dataloader))
        
        print("\n批次数据:")
        for key, value in batch.items():
            print(f"  {key:15s}: {value.shape} ({value.dtype})")
        
        print("\n✅ DataLoader 测试成功")
        return True
        
    except Exception as e:
        print(f"❌ DataLoader 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("nuScenes 数据测试")
    print("=" * 60)
    
    results = []
    
    # 1. 测试数据文件
    results.append(("数据文件", test_data_files()))
    
    # 2. 测试数据形状
    results.append(("数据形状", test_data_shapes()))
    
    # 3. 测试数据值
    results.append(("数据值", test_data_values()))
    
    # 4. 测试 Dataset
    results.append(("Dataset", test_dataset_loading()))
    
    # 5. 测试 DataLoader
    results.append(("DataLoader", test_dataloader()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:20s}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！数据准备就绪。")
        print("=" * 60)
        print("\n下一步:")
        print("  1. 训练 Scorer: python train_scorer_nuscenes.py")
        print("  2. 训练 Matcher: python train_matcher_nuscenes.py")
    else:
        print("\n" + "=" * 60)
        print("❌ 部分测试失败，请检查数据预处理。")
        print("=" * 60)
        print("\n建议:")
        print("  1. 重新运行预处理: python scripts/preprocess_nuscenes.py")
        print("  2. 检查预处理函数实现是否正确")


if __name__ == "__main__":
    main()
