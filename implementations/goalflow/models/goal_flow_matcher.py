import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from implementations.flow_matching.models.time_embedding import SinusoidalEmbedding
from implementations.flow_matching.models.ode_solver import ODESolver

class GoalFlowMatcher(nn.Module):
    """
    Goal-Driven Flow Matching 轨迹生成器
    使用Transformer架构，融合多个条件：
      - Goal Point:目标点坐标
      - Scene Feature: BEV场景特征
      - Time: 时间步

    训练：预测速度场 v_theta(x_t, t, goal, scene)
    推理：多步ODE求解生成轨迹
    """
    def __init__(self,
                 # 基础维度
                 traj_dim: int = 2,
                 num_traj_points: int = 6,

                 # Transformer 配置
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int=6,
                 dim_feedforward: int=1024,
                 dropout: float=0.1,

                 # 条件编码器配置 
                 goal_hidden_dim: int=128,
                 scene_channels: int=64,
                 scene_hidden_dim: int=256,
                 t_dim: int=128,

                 # 其他
                 activation: str='gelu'
                 ):
        super().__init__()

        # 1.输入编码器
        # 1.1 轨迹编码器
        self.traj_encoder = nn.Sequential(nn.Linear(traj_dim, d_model//2),
                                          nn.Linear(d_model//2, d_model))
        # 1.2 Goal编码器
        self.goal_encoder = nn.Sequential(nn.Linear(traj_dim, goal_hidden_dim),
                                          nn.Linear(goal_hidden_dim, d_model))
        # 1.3 场景编码器
        self.scene_conv = nn.Sequential(nn.Conv2d(scene_channels, scene_hidden_dim, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(scene_hidden_dim),
                                        nn.GELU(),
                                        nn.BatchNorm2d(scene_hidden_dim),
                                        nn.GELU())
        # 1.4 时间编码器
        self.time_embedding = SinusoidalEmbedding(t_dim)
        self.time_proj = nn.Linear(t_dim, d_model)

        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation=activation,
                                                   batch_first=True,
                                                   norm_first=True)         # ?
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=num_encoder_layers)
        
        # 3. 输出解码器
        self.velocity_decoder = nn.Sequential(nn.Linear(d_model, d_model//2),
                                              nn.GELU(),
                                              nn.Dropout(dropout),
                                              nn.Linear(d_model//2, traj_dim))
        
        # 4. Token类型嵌入(帮助区分不同类型的token)
        self.token_type_embed = nn.Embedding(4, d_model)

        # 初始化
        self._init_weights()
        
        
        
        
        
        
        
        

    def sample_ot_flow(self):
        pass

    def compute_cfm_loss(self,
                         model: nn.Module,
                         x_0: torch.Tensor, 
                         x_1: torch.Tensor, 
                         condition: Optional[torch.Tensor]=None, 
                         t: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        batch_size = x_1.shape[0]
        
        if t is None:
            t = torch.rand(batch_size, device=x_0.device, dtype=x_0.dtype)
        


