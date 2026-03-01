"""
nuScenes Configuration for GoalFlow

配置 nuScenes 数据集的所有参数
"""

import os


class NuScenesConfig:
    """nuScenes 数据集配置"""
    
    # ==================== 数据路径 ====================
    data_root = 'data/nuscenes'  # nuScenes 数据根目录
    version = 'v1.0-mini'  # 数据集版本: 'v1.0-mini' 或 'v1.0-trainval'
    
    # 预处理数据保存路径
    processed_data_dir = 'data/nuscenes_processed'
    vocabulary_path = os.path.join(processed_data_dir, 'vocabulary.npy')
    
    # ==================== 轨迹参数 ====================
    # 采样率
    sample_rate = 2  # Hz (nuScenes 原始采样率为 2Hz)
    
    # 历史和未来帧数
    history_frames = 4  # 2秒历史 (4 frames @ 2Hz)
    future_frames = 12  # 6秒未来 (12 frames @ 2Hz)
    
    # 轨迹过滤条件
    min_movement = 0.5  # 最小移动距离 (m)，过滤静止物体
    max_distance = 60.0  # 最大距离 (m)，只考虑 ego 车附近的 agent
    
    # ==================== BEV 特征参数 ====================
    # BEV 图像尺寸
    bev_height = 200  # pixels
    bev_width = 200   # pixels
    bev_channels = 3  # HD map channels: [lane, road_boundary, walkway]
    
    # BEV 范围
    bev_range = 50.0  # meters ([-50m, 50m] x [-50m, 50m])
    bev_resolution = bev_range * 2 / bev_height  # 0.5 m/pixel
    
    # 地图层
    map_layers = ['lane', 'road_segment', 'walkway']
    
    # ==================== Goal 词汇表参数 ====================
    vocab_size = 32  # K-means 聚类数量
    vocab_seed = 42   # 随机种子
    
    # ==================== 模型参数 ====================
    # GoalPointScorer
    scorer_hidden_dim = 256
    scorer_num_heads = 8
    scorer_num_layers = 4
    scorer_dropout = 0.1
    
    # GoalFlowMatcher
    matcher_hidden_dim = 256
    matcher_num_heads = 8
    matcher_num_layers = 6       # 增加到6层（原4层），提升模型容量
    matcher_dropout = 0.1
    matcher_num_steps = 20       # ODE solver steps（原10步），更精细的积分
    matcher_noise_std = 1.0      # Initial noise std
    
    # 场景 token 目标空间尺寸（控制 BEV 下采样程度）
    # (25, 25) = 625 tokens — 平衡信息保留和计算量
    # (50, 50) = 2500 tokens — 更多空间信息但更慢
    # None = 旧逻辑（下采样到 ≤16x16）
    scene_token_size = (25, 25)  # 625 scene tokens
    
    # ==================== 训练参数 ====================
    # 通用
    seed = 42
    device = 'cuda'  # 'cuda' or 'cpu'
    num_workers = 4  # DataLoader workers
    
    # Scorer 训练
    scorer_batch_size = 16
    scorer_num_epochs = 100
    scorer_learning_rate = 1e-4
    scorer_weight_decay = 1e-5
    scorer_lambda_dis = 1.0  # Distance loss weight
    scorer_lambda_dac = 0.01  # DAC loss weight
    
    # Matcher 训练
    matcher_batch_size = 32
    matcher_num_epochs = 100
    matcher_learning_rate = 1e-4
    matcher_weight_decay = 1e-5
    
    # 学习率调度
    use_scheduler = True
    scheduler_factor = 0.5
    scheduler_patience = 5
    
    # ==================== 评估参数 ====================
    eval_interval = 1  # 每 N 个 epoch 评估一次
    save_interval = 10  # 每 N 个 epoch 保存一次
    
    # 推理参数
    num_candidates = 10  # 候选轨迹数量
    
    # ==================== 路径配置 ====================
    # Checkpoint 保存路径
    scorer_checkpoint_dir = 'checkpoints/scorer_nuscenes'
    matcher_checkpoint_dir = 'checkpoints/matcher_nuscenes'
    
    # 日志保存路径
    log_dir = 'logs/nuscenes'
    
    # 可视化保存路径
    vis_dir = 'visualizations/nuscenes'
    
    # ==================== 数据集划分 ====================
    train_scenes = [
        'scene-0061', 'scene-0103', 'scene-0655', 
        'scene-0757', 'scene-0796', 'scene-0916'
    ]  # 6 scenes for training
    
    val_scenes = [
        'scene-1077', 'scene-1094'
    ]  # 2 scenes for validation
    
    test_scenes = [
        'scene-1100'
    ]  # 1 scene for testing
    
    # ==================== 物体类别 ====================
    # 只考虑车辆类别
    vehicle_categories = [
        'vehicle.car',
        'vehicle.truck',
        'vehicle.bus',
        'vehicle.trailer',
        'vehicle.construction',
        'vehicle.emergency.ambulance',
        'vehicle.emergency.police'
    ]
    
    # ==================== 辅助函数 ====================
    @classmethod
    def create_dirs(cls):
        """创建所有必要的目录"""
        dirs = [
            cls.processed_data_dir,
            cls.scorer_checkpoint_dir,
            cls.matcher_checkpoint_dir,
            cls.log_dir,
            cls.vis_dir
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        print(f"✅ Created directories: {', '.join(dirs)}")
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("=" * 60)
        print("nuScenes Configuration")
        print("=" * 60)
        print(f"Data root: {cls.data_root}")
        print(f"Version: {cls.version}")
        print(f"History frames: {cls.history_frames} ({cls.history_frames / cls.sample_rate:.1f}s)")
        print(f"Future frames: {cls.future_frames} ({cls.future_frames / cls.sample_rate:.1f}s)")
        print(f"BEV size: {cls.bev_height}x{cls.bev_width}")
        print(f"BEV range: ±{cls.bev_range}m")
        print(f"BEV resolution: {cls.bev_resolution:.2f}m/pixel")
        print(f"Vocabulary size: {cls.vocab_size}")
        print(f"Device: {cls.device}")
        print(f"Train scenes: {len(cls.train_scenes)}")
        print(f"Val scenes: {len(cls.val_scenes)}")
        print(f"Test scenes: {len(cls.test_scenes)}")
        print("=" * 60)


# 创建配置实例
config = NuScenesConfig()


if __name__ == "__main__":
    # 测试配置
    config.print_config()
    config.create_dirs()
