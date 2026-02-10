import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class TrajectorySelector(nn.Module):
    """
    轨迹选择器

    输入：
    - trajectories: (B, N, T, 2) 多条候选轨迹
    - goal: (B, 2) 目标点
    - gt_trajectory: (B, T, 2) 真实轨迹
    
    输出：
    - best_trajectory: (B, T, 2) 最优轨迹
    - scores: (B, N) 所有轨迹的评分
    """

    def __init__(self,
                 # 评分权重
                 lambda_dis: float = 1.0,   # 距离评分权重
                 lambda_pg: float = 1.0,    # 进度评分权重
                 lambda_col: float = 0.0,   # 碰撞评分权重
                 lambda_dac: float = 0.0,   # DAC评分权重

                 # Shadow Trajectories 配置
                 use_shadow: bool = True,           # 是否使用Shadow Trajectories
                 shadow_mask_ratio: float = 0.3,    # Mask 比例
                 num_shadow: int = 5,               # Shadow 轨迹数量
                 # 其他
                 normalize: bool = True):
        super().__init__()
        self.lambda_dis = lambda_dis
        self.lambda_pg = lambda_pg
        self.lambda_col = lambda_col
        self.lambda_dac = lambda_dac

    def compute_distance_score(self,
                              trajectories: torch.Tensor,
                              gt_trajectory: torch.Tensor
                              ) -> torch.Tensor:
        """
        计算距离评分：与真实轨迹的距离

        Args:
            trajectories: (B, N, T, 2)  候选轨迹
            gt_trajectory: (B, T, 2)    真实轨迹

        Returns:
            f_dis: (B, N) 距离评分（越小越好）
        """
        B, N, T, _= trajectories.shape

        f_dis = torch.zeros((B, N), device=trajectories.device)

        expanded_gt = gt_trajectory.unsqueeze(1).expand(-1, N, -1)

        distances = torch.norm(trajectories - expanded_gt, dim=-1)  # 计算欧氏距离用torch.norm
        f_dis = distances.mean(dim=-1)  # (B, N)
                
        return f_dis



    def compute_progess_score(self,
                              trajectories: torch.Tensor,
                              goal: torch.Tensor
                              ) -> torch.Tensor:
        """
        计算进度评分：朝向目标的进度

        Args:
            trajectories: (B, N, T, 2) 候选轨迹
            goal: (B, 2)    目标点

        Returns:
            f_pg: (B, N) 进度评分（越小越好）
        """
        pass


    def compute_collision_score(self,
                                trajectories: torch.Tensor,
                                obstacle: Optional[torch.Tensor] = None
                                ) -> torch.Tensor:
        """
        计算进度评分：朝向目标的进度

        Args:
            trajectories: (B, N, T, 2) 候选轨迹
            obstacle: (B, M, 2)    障碍物位置

        Returns:
            f_col: (B, N) 碰撞评分（越小越好）
        """
        pass


    def compute_dac_score(self,
                          trajectories: torch.Tensor,
                          dirvable_area: Optional[torch.Tensor] = None
                          ) -> torch.Tensor:
        """
        计算进度评分：朝向目标的进度

        Args:
            trajectories: (B, N, T, 2) 候选轨迹
            dirvable_area: (B, H, W)    可行驶区域 mask

        Returns:
            f_col: (B, N) DAC评分（越小越好）
        """
        pass


    def normalize_scores(self,
                         scores: torch.Tensor
                         ) -> torch.Tensor:
        """
        Min-Max 归一化评分
        Args:
            scores: (B, N) 原始评分

        Returns:
            normalized_scores: (B, N) 归一化后的评分[0,1]
        """
        pass


    def compute_final_score(self,
                            trajectories: torch.Tensor,
                            goal: torch.Tensor,
                            gt_trajectory: torch.Tensor,
                            obstacle: Optional[torch.Tensor] = None,
                            drivable_area: Optional[torch.Tensor] = None
                            ) -> torch.Tensor:
        """
        计算最终评分
        Args:
            trajectories: (B, N, T, 2)  候选轨迹
            goal: (B, 2)    目标点
            gt_trajectory: (B, T, 2)    真实轨迹
            obstacle: (B, M, 2)    障碍物位置
            dirvable_area: (B, H, W)    可行驶区域 mask

        Returns:
            scores: (B, N) 最终评分（越高越好）
        """
        pass


    def generate_shadow_trajectories(self,
                                     trajectories: torch.Tensor,
                                     goal: torch.Tensor,
                                     model: nn.Module
                                     ) -> torch.Tensor:
        """
        生成 Shadow Trajectories

        Args:
            trajectories: (B, N, T, 2)  候选轨迹
            goal: (B, 2)    目标点
            model: GoalFlowMatcher 模型

        Returns:
            shadow_trajectories: (B, N*num_shadow, T, 2)
        """
        pass


    def select_best_trajectory(self,
                               trajectories: torch.Tensor,
                               scores: torch.Tensor
                               ) -> torch.Tensor:
        """
        选择最优轨迹

        Args:
            trajectories: (B, N, T, 2)  候选轨迹
            scores: (B, N) 评分

        Returns:
            best_trajectory: (B, T, 2) 最优轨迹
            best_indices: (B,) 最优轨迹的索引
        """
        pass


    def forward(self,
                trajectories: torch.Tensor,
                goal: torch.Tensor,
                gt_trajectory: torch.Tensor,
                obstacle: Optional[torch.Tensor] = None,
                drivable_area: Optional[torch.Tensor] = None,
                return_scores: bool = False
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播：评分并选择最优轨迹

        Args:
            trajectories: (B, N, T, 2)  候选轨迹
            goal: (B, 2)    目标点
            gt_trajectory: (B, T, 2)    真实轨迹
            obstacle: (B, M, 2)    障碍物位置
            dirvable_area: (B, H, W)    可行驶区域 mask
            return_scores: 是否返回所有评分

        Returns:
            best_trajectory: (B, T, 2) 最优轨迹
            scores: (B, N) 所有评分
        """
        pass


    def compute_ade(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
        """
        计算 Average Displacement Error (ADE)
        
        Args:
            pred_traj: (B, N, T, 2) 预测轨迹
            gt_traj: (B, T, 2) 真实轨迹
        
        Returns:
            ade: (B, N) ADE 值
        
        TODO: 实现 ADE 计算
        提示：
        ADE = mean(||pred - gt||) over all time steps
        """
        pass


    def compute_fde(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
        """
        计算 Final Displacement Error (FDE)
        
        Args:
            pred_traj: (B, N, T, 2) 预测轨迹
            gt_traj: (B, T, 2) 真实轨迹
        
        Returns:
            fde: (B, N) FDE 值
        
        TODO: 实现 FDE 计算
        提示：
        FDE = ||pred_end - gt_end||
        """
        pass

        



