import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import sys
import os

# 添加父目录到路径以支持导入
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

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
        self.traj_dim = traj_dim
        self.num_traj_points = num_traj_points
        self.d_model = d_model
        # 1.输入编码器
        # 1.1 轨迹编码器
        self.traj_encoder = nn.Sequential(nn.Linear(traj_dim, d_model//2),
                                          nn.GELU(),
                                          nn.Linear(d_model//2, d_model))
        # 1.2 Goal编码器
        self.goal_encoder = nn.Sequential(nn.Linear(traj_dim, goal_hidden_dim),
                                          nn.GELU(),
                                          nn.Linear(goal_hidden_dim, d_model))
        # 1.3 场景编码器
        self.scene_conv = nn.Sequential(nn.Conv2d(scene_channels, scene_hidden_dim, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(scene_hidden_dim),
                                        nn.GELU(),
                                        nn.Conv2d(scene_hidden_dim, d_model, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(d_model),
                                        nn.GELU())
        # 位置编码（用于Scene tokens）
        self.scene_pos_embed = nn.Parameter(torch.randn(1, 32*32, d_model) * 0.02)

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
        
    def _init_weights(self):
        """
        初始化权重        
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

    def encode_conditions(self,
                          goal: torch.Tensor,
                          scene: torch.Tensor,
                          t: torch.Tensor) ->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        """
        编码所有条件
        Args:
            goal: (B, 2) 目标点
            scene: (B, C, H, W) BEV场景特征
            t: (B,) 时间标量
        Returns:
            goal_tokens: (B, 1, d_model)
            scene_tokens: (B, HW, d_model)
            time_tokens: (B, 1, d_model)
        """
        B = goal.shape[0]
        
        # 1. Goal tokens
        goal_feat = self.goal_encoder(goal)     # (B, d_model)
        goal_tokens = goal_feat.unsqueeze(1)    # (B, 1, d_model)

        # 2. Scene tokens
        scene_feat = self.scene_conv(scene)     # (B, d_model, H, W)
        scene_tokens = scene_feat.flatten(2).transpose(1, 2) # (B, HW, d_model)
        
        # 添加位置编码
        if scene_tokens.shape[1] == self.scene_pos_embed.shape[1]:
            scene_tokens = scene_tokens + self.scene_pos_embed

        # 3. Time tokens
        time_feat = self.time_embedding(t)      # (B, time_dim)
        time_feat = self.time_proj(time_feat)   # (B, d_model)
        time_tokens = time_feat.unsqueeze(1)    # (B, 1, d_model)

        return goal_tokens, scene_tokens, time_tokens

        
        
    def forward(self,
                x_t: torch.Tensor,
                goal: torch.Tensor,
                scene: torch.Tensor,
                time: torch.Tensor) -> torch.Tensor:
        """
        前向传播： 预测速度场
        Args:
            x_t: (B, T, 2)  当前轨迹状态
            goal: (B, 2)    目标点
            scene: (B, C, H, W) BEV场景特征
            time: (B,)      时间标量[0,1]

        Returns:
            v_pred: (B, T, 2)   预测的速度场
        """
        B, T, _ = x_t.shape
        # 1. 轨迹编码
        traj_tokens = self.traj_encoder(x_t)        # (B, T, d_model)
        # 2. 条件编码
        goal_tokens, scene_tokens, time_tokens = self.encode_conditions(goal, scene, time)
        # 3. 拼接所有token
        all_tokens = torch.cat([traj_tokens, goal_tokens, scene_tokens, time_tokens], dim=1) # (B, T+1+HW+1, d_model)
        # 4. 添加token类型嵌入
        num_scene_tokens = scene_tokens.shape[1]
        token_types = torch.cat([
            torch.zeros(B, T, dtype=torch.long, device=x_t.device),                     # traj
            torch.ones(B, 1, dtype=torch.long, device=x_t.device),                      # goal
            torch.full((B, num_scene_tokens), 3, dtype=torch.long, device=x_t.device),  # scene
            torch.full((B, 1), 3, dtype=torch.long, device=x_t.device)                  # time
        ], dim=1)   # (B, B, T+1+HW+1)
        type_embed = self.token_type_embed(token_types) # (B, T+1+HW+1, d_model)

        all_tokens = all_tokens + type_embed
        # 5. Transformer编码
        encoded = self.transformer_encoder(all_tokens)  # (B, T+1+HW+1, d_model)

        # 6. 提取轨迹tokens 并解码
        traj_encoded = encoded[:, :T, :]    # (B, T, d_model)
        v_pred = self.velocity_decoder(traj_encoded)    # (B, T, 2)

        return v_pred
    
    def compute_loss(self,
                     x_0: torch.Tensor,
                     x_1: torch.Tensor,
                     goal: torch.Tensor,
                     scene: torch.Tensor,
                     t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算Conditional Flow Matching Loss
        Loss = E_{t, x_0, x_1}[||v_theta(x_t, t, goal, scene) - (x_1 - x_0)||^2]

        Args: 
            x_0: (B, T, 2) 噪声
            x_1: (B, T, 2) 真实轨迹
            goal: (B, 2) 目标点
            scene: (B, C, H, W) BEV场景特征
            t: (B,) 时间标量[0,1]

        Returns:
            loss: 标量损失
        """
        B = x_1.shape[0]

        # 1. 随机采样时间（如果未提供）
        if t is None:
            t = torch.rand(B, device=x_0.device, dtype=x_0.dtype)
        
        # 2. OT Flow 插值
        t_expended = t.view(B, 1, 1)    # (B, 1, 1)
        x_t = (1 - t_expended) * x_0 + t_expended * x_1

        # 3. 计算真实速度
        v_true  = x_1 - x_0

        # 4. 网络预测速度
        v_pred = self.forward(x_t, goal, scene, t)

        # 5. MSE Loss
        loss = F.mse_loss(v_pred, v_true)

        return loss
    
    @torch.no_grad()
    def generate(self,
                 goal: torch.Tensor,
                 scene: torch.Tensor,
                 num_steps: int=1,
                 num_traj_points: Optional[int]=None,
                 method: str='euler') -> torch.Tensor:
        """
        生成轨迹（推理模式）
        
        Args:
            goal: (B, 2)
            scene: (B, C, H, W)
            num_steps: 推理步数(n=1单步, n>1多步)
            num_traj_points: 轨迹点数量
            method: 'euler'或'rk4'
        Returns:
            trajectory: (B, T, 2) 生成的轨迹
        """
        B = goal.shape[0]
        T = num_traj_points if num_traj_points is not None else self.num_traj_points

        x_0 = torch.randn(B, T, self.traj_dim, device=goal.device)
        
        if method == 'euler':
            # "Euler一阶龙格"
            return self._ode_euler_slover(x_0, goal, scene, num_steps)
        elif method == 'rk4':
            # "RK4四阶龙格"
            return self._ode_rk4_slover(x_0, goal, scene, num_steps)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _ode_euler_slover(self,
                          x_0: torch.Tensor,
                          goal: torch.Tensor,
                          scene: torch.Tensor,
                          num_steps: int) -> torch.Tensor:
        """
        Euler解ODE方程
        """
        B = x_0.shape[0]
        x = x_0.clone()
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t_i = (i + 0.5) * dt
            t_tensor = torch.full((B,), t_i, device=x_0.device)
            v_pred = self.forward(x, goal, scene, t_tensor)
            x = x + v_pred * dt

        return x
        

        
    def _ode_rk4_slover(self,
                        x_0: torch.Tensor,
                        goal: torch.Tensor,
                        scene: torch.Tensor,
                        num_steps: int) -> torch.Tensor:
        """
        RK4解ODE方程
        """
        B = x_0.shape[0]
        x = x_0.clone()
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = i * dt
            t1 = torch.full((B,), t, device=x_0.device)
            t2 = torch.full((B,), t + 0.5*dt, device=x_0.device)
            t3 = torch.full((B,), t + dt, device=x_0.device)

            k1 = self.forward(x, goal, scene, t1)
            k2 = self.forward(x + 0.5*dt*k1, goal, scene, t2)
            k3 = self.forward(x + 0.5*dt*k2, goal, scene, t2)
            k4 = self.forward(x + dt*k3, goal, scene, t3)

            x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return x
    

    @torch.no_grad()
    def generate_multiple(self,
                          goal: torch.Tensor,
                          scene: torch.Tensor,
                          num_samples: int=10,
                          num_steps: int=1,
                          method: str='euler') -> torch.Tensor:
        """
        生成多条候选轨迹（用于后续的 Trajectory Selector）
        
        Args:
            goal: (B, 2) 目标点
            scene: (B, C, H, W) BEV场景特征
            num_samples: 每个样本生成的轨迹数量
            num_steps: 推理步数
            method: 'euler'或'rk4'
        Returns:
            trajectories: (B, num_samples, T, 2) 多条候选轨迹
        """
        B = goal.shape[0]

        # 扩展输入以生成多条轨迹
        goal_expanded = goal.unsqueeze(1).expand(-1, num_samples, -1).reshape(B*num_samples, -1)
        scene_expanded = scene.unsqueeze(1).expand(-1, num_samples, -1, -1, -1).reshape(B*num_samples, *scene.shape[1:])
        
        # 生成轨迹
        trajectories = self.generate(goal_expanded, scene_expanded, num_steps=num_steps, method=method)
        
        # 重塑轨迹
        trajectories = trajectories.reshape(B, num_samples, *trajectories.shape[1:])

        return trajectories

def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

    
        

