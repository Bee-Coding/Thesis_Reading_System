"""
GoalPointScorer 训练配置
"""

class ScorerConfig:
    # ==================== 模型参数 ====================
    vocab_size = 128          # 词汇表大小
    vocab_dim = 2             # 词汇表维度 (x, y)
    scene_channels = 64       # BEV 特征通道数
    scene_height = 32         # BEV 特征高度
    scene_width = 32          # BEV 特征宽度
    
    hidden_dim = 256          # Transformer 隐藏层维度
    num_heads = 8             # 注意力头数
    num_layers = 4            # Transformer 层数
    dropout = 0.1             # Dropout 比例
    
    # ==================== 训练参数 ====================
    batch_size = 32           # 批次大小
    learning_rate = 1e-4      # 学习率
    num_epochs = 100          # 训练轮数
    weight_decay = 1e-5       # 权重衰减
    
    # 学习率调度
    use_scheduler = True      # 是否使用学习率调度器
    scheduler_patience = 10   # ReduceLROnPlateau 的 patience
    scheduler_factor = 0.5    # 学习率衰减因子
    
    # ==================== 损失权重 ====================
    lambda_dis = 1.0          # 距离评分损失权重
    lambda_dac = 0.5          # DAC 评分损失权重
    
    # ==================== 数据路径 ====================
    data_path = 'data/toy_data.npz'
    checkpoint_dir = 'checkpoints/scorer'
    log_dir = 'logs/scorer'
    
    # ==================== 训练设置 ====================
    device = 'cpu'           # 'cuda' 或 'cpu'
    num_workers = 4           # DataLoader 工作线程数
    save_interval = 10        # 每隔多少 epoch 保存一次
    eval_interval = 1         # 每隔多少 epoch 验证一次
    
    # ==================== 其他 ====================
    seed = 42                 # 随机种子
    resume = None             # 恢复训练的 checkpoint 路径（None 表示从头训练）
