"""
nuScenes 数据预处理脚本

运行此脚本来预处理 nuScenes 数据集，生成训练所需的数据文件。

使用方法:
    python scripts/preprocess_nuscenes.py

输出:
    - data/nuscenes_processed/train/
        - history.npy
        - future.npy
        - goals.npy
        - bev_features.npy
        - vocabulary.npy
        - metadata.pkl
    - data/nuscenes_processed/val/
        - (同上)
    - data/nuscenes_processed/test/
        - (同上)
"""

import sys
import os
import pickle

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.nuscenes_preprocessor import preprocess_nuscenes
from config.nuscenes_config import NuScenesConfig


def main():
    """主函数"""
    print("=" * 70)
    print("nuScenes 数据预处理")
    print("=" * 70)
    
    # 加载配置
    config = NuScenesConfig()
    config.print_config()
    
    # 创建输出目录
    config.create_dirs()
    
    # ========== 预处理训练集 ==========
    print("\n" + "=" * 70)
    print("预处理训练集")
    print("=" * 70)
    train_output_dir = os.path.join(config.processed_data_dir, 'train')
    os.makedirs(train_output_dir, exist_ok=True)
    
    try:
        preprocess_nuscenes(
            dataroot=config.data_root,
            version=config.version,
            output_dir=train_output_dir,
            config=config,
            scenes=config.train_scenes
        )
    except NotImplementedError as e:
        print(f"\n❌ 错误: {e}")
        print("\n请先实现 nuscenes_preprocessor.py 中的核心函数：")
        print("  1. extract_agent_trajectories()")
        print("  2. rasterize_map()")
        print("  3. build_vocabulary()")
        return
    except Exception as e:
        print(f"\n❌ 预处理失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 读取 train 的标准化统计量，供 val/test 复用
    train_metadata_path = os.path.join(train_output_dir, 'metadata.pkl')
    with open(train_metadata_path, 'rb') as f:
        train_metadata = pickle.load(f)
    
    train_norm_stats = {
        'mean': train_metadata['traj_mean'],
        'std': train_metadata['traj_std'],
    }
    print(f"\n训练集标准化统计量: mean={train_norm_stats['mean']}, std={train_norm_stats['std']}")
    print("val/test 将复用此统计量")
    
    # ========== 预处理验证集（复用 train 的 mean/std）==========
    print("\n" + "=" * 70)
    print("预处理验证集")
    print("=" * 70)
    val_output_dir = os.path.join(config.processed_data_dir, 'val')
    os.makedirs(val_output_dir, exist_ok=True)
    
    # 将 train 的统计量挂到 config 上，让 preprocess_nuscenes 复用
    config._norm_stats = train_norm_stats
    
    try:
        preprocess_nuscenes(
            dataroot=config.data_root,
            version=config.version,
            output_dir=val_output_dir,
            config=config,
            scenes=config.val_scenes
        )
    except Exception as e:
        print(f"\n❌ 验证集预处理失败: {e}")
    
    # ========== 预处理测试集（复用 train 的 mean/std）==========
    print("\n" + "=" * 70)
    print("预处理测试集")
    print("=" * 70)
    test_output_dir = os.path.join(config.processed_data_dir, 'test')
    os.makedirs(test_output_dir, exist_ok=True)
    
    try:
        preprocess_nuscenes(
            dataroot=config.data_root,
            version=config.version,
            output_dir=test_output_dir,
            config=config,
            scenes=config.test_scenes
        )
    except Exception as e:
        print(f"\n❌ 测试集预处理失败: {e}")
    
    print("\n" + "=" * 70)
    print("✅ 预处理完成！")
    print("=" * 70)
    print(f"\n数据已保存到: {config.processed_data_dir}")
    print("\n下一步:")
    print("  1. 运行数据测试: python data/test_nuscenes_data.py")
    print("  2. 训练 Scorer: python train_scorer_nuscenes.py")
    print("  3. 训练 Matcher: python train_matcher_nuscenes.py")


if __name__ == "__main__":
    main()
