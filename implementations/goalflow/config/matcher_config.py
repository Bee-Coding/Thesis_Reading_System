"""
GoalFlowMatcher 训练配置
"""

class MatcherConfig:
    # ==================== 模型参数 ====================
    traj_dim = 2              # 轨迹维度 (x, y)
    num_traj_points = 6       # 轨迹点数
    goal_dim = 2              # 目标点维度
    scene_channels = 64       # BEV 特征通道数
    scene_height = 32         # BEV 特征高度
    scene_width = 32          # BEV 特征宽度
    
    hidden_dim = 256          # Transformer 隐藏层维度
    num_heads = 8             # 注意力头数
    num_layers = 6            # Transformer 层数
    dropout = 0.1             # Dropout 比例
    
    # ==================== 训练参数 ====================
    batch_size = 32           # 批次大小
    learning_rate = 1e-4      # 学习率
    num_epochs = 200          # 训练轮数
    weight_decay = 1e-5       # 权重衰减
    
    # 学习率调度
    use_scheduler = True      # 是否使用学习率调度器
    scheduler_patience = 15   # ReduceLROnPlateau 的 patience
    scheduler_factor = 0.5    # 学习率衰减因子
    
    # ==================== Flow Matching 参数 ====================
    use_gt_goal = True        # 是否使用 gt_goal（False 则使用 Scorer 选出的目标）
    noise_std = 1.0           # x_0 噪声标准差
    
    # ==================== 数据路径 ====================
    data_path = 'data/toy_data.npz'
    scorer_checkpoint = 'checkpoints/scorer/best.pth'  # 预训练的 Scorer（如果 use_gt_goal=False）
    checkpoint_dir = 'checkpoints/matcher'
    log_dir = 'logs/matcher'
    
    # ==================== 训练设置 ====================
    device = 'cuda'           # 'cuda' 或 'cpu'
    num_workers = 4           # DataLoader 工作线程数
    save_interval = 20        # 每隔多少 epoch 保存一次
    eval_interval = 5         # 每隔多少 epoch 验证一次
    
    # 验证时生成参数
    val_num_steps = 10        # ODE 求解步数
    val_method = 'euler'      # 'euler' 或 'rk4'
    
    # ==================== 其他 ====================
    seed = 42                 # 随机种子
    resume = None             # 恢复训练的 checkpoint 路径（None 表示从头训练）
