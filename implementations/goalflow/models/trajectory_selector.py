import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import sys
import os

# 添加路径以支持导入
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# 这些导入是可选的，仅用于类型提示
# from implementations.goalflow.models.goal_point_scorer import GoalPointScorer
# from implementations.goalflow.models.goal_flow_matcher import GoalFlowMatcher

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

        self.collision_dis_threshold = 0.5
        self.use_shadow = use_shadow
        self.num_shadow = num_shadow
        self.shadow_mask_ratio = shadow_mask_ratio


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

        expanded_gt = gt_trajectory.unsqueeze(1)

        distances = torch.norm(trajectories - expanded_gt, dim=-1)  # 计算欧氏距离用torch.norm
        f_dis = distances.mean(dim=-1)  # (B, N)
                
        return f_dis



    def compute_progress_score(self,
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
        B, N, T, _ = trajectories.shape

        end_points = trajectories[:, :, -1, :]    # (B, N, 2)
        # goals = goal.unsqueeze(1).expand(-1, N, -1)     # (B, N, 2) expand需要完整形状
        goals = goal.unsqueeze(1)   # 利用自动广播

        f_pg = torch.norm(end_points- goals, dim=-1)    # (B, N)
        
        return f_pg


    def compute_collision_score(self,
                                trajectories: torch.Tensor,
                                obstacles: Optional[torch.Tensor] = None
                                ) -> torch.Tensor:
        """
        计算进度评分：朝向目标的进度

        Args:
            trajectories: (B, N, T, 2) 候选轨迹
            obstacles: (B, M, 2)    障碍物位置

        Returns:
            f_col: (B, N) 碰撞评分（越小越好）
        """
        B, N, T, _ = trajectories.shape
        f_col = torch.zeros(B, N, device=trajectories.device)

        if obstacles is not None:
            # 计算轨迹点与障碍物中心的距离
            # (B, N, T, 2) vs (B, M, 2) -> (B, N, T, M)
            diff = trajectories.unsqueeze(3) - obstacles.unsqueeze(1).unsqueeze(2)  # (B, N, T, M, 2)
            col_dis = torch.norm(diff, dim=-1)  # (B, N, T, M)
            # 取每条轨迹最小的碰撞距离
            min_col_dis = col_dis.min(dim=-1)[0].min(dim=-1)[0]  # (B, N)
            col_dis_threshold = getattr(self, 'collision_dis_threshold', 2.0)

            f_col = torch.exp(-(min_col_dis - col_dis_threshold) ** 2)
            
        return f_col


    def compute_dac_score(self,
                          trajectories: torch.Tensor,
                          drivable_area: Optional[torch.Tensor] = None
                          ) -> torch.Tensor:
        """
        计算DAC评分：可行驶区域合规性

        Args:
            trajectories: (B, N, T, 2) 候选轨迹
            drivable_area: (B, H, W)    可行驶区域 mask

        Returns:
            f_dac: (B, N) DAC评分（越小越好）
        """
        B, N, T, _ = trajectories.shape
        
        # 如果没有提供 drivable_area，返回零分
        if drivable_area is None:
            return torch.zeros(B, N, device=trajectories.device)
        
        H, W = drivable_area.shape[-2:]

        # 将轨迹点坐标转换为图像坐标
        # 假设轨迹坐标范围是 [-50, 50]
        x = trajectories[..., 0]  # (B, N, T)
        y = trajectories[..., 1]  # (B, N, T)

        x_img = ((x + 50) / 100 * (W - 1)).long().clamp(0, W-1)
        y_img = ((y + 50) / 100 * (H - 1)).long().clamp(0, H-1)

        # 检查每个点是否在可行驶区域内
        violations = torch.zeros(B, N, T, device=trajectories.device)
        for b in range(B):
            for n in range(N):
                for t in range(T):
                    violations[b, n, t] = 1.0 - drivable_area[b, y_img[b, n, t], x_img[b, n, t]].float()

        # 计算违规比例
        f_dac = violations.mean(dim=-1)  # (B, N)

        return f_dac


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
        max_score = scores.max(dim=1, keepdim=True)[0]  # 取 values
        min_score = scores.min(dim=1, keepdim=True)[0]  # 取 values

        # 防止除以0
        denominator = max_score - min_score
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)

        return (scores - min_score) / denominator


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
        B, N, T, _ = trajectories.shape
        scores = torch.zeros(B, N, device=trajectories.device)

        dis_scores = self.normalize_scores(self.compute_distance_score(trajectories, gt_trajectory))
        pg_scores = self.normalize_scores(self.compute_progress_score(trajectories, goal))
        col_scores = self.normalize_scores(self.compute_collision_score(trajectories, obstacle))
        dac_scores = self.normalize_scores(self.compute_dac_score(trajectories, drivable_area))

        scores = -(self.lambda_dis*dis_scores + 
                   self.lambda_pg*pg_scores + 
                   self.lambda_col*col_scores + 
                   self.lambda_dac*dac_scores)

        return scores


    def generate_shadow_trajectories(self,
                                     goal: torch.Tensor,
                                     scene: torch.Tensor,
                                     model: nn.Module,
                                     num_traj_points: int = 6
                                     ) -> torch.Tensor:
        """
        生成 Shadow Trajectories
        
        策略：对目标点添加噪声，使用 GoalFlowMatcher 生成多条轨迹

        Args:
            goal: (B, 2) 目标点
            scene: (B, C, H, W) BEV 场景特征
            model: GoalFlowMatcher 模型
            num_traj_points: 轨迹点数量

        Returns:
            shadow_trajectories: (B, num_shadow, T, 2) Shadow 轨迹
        """
        B = goal.shape[0]
        T = num_traj_points
        
        # 1. 生成带噪声的目标点
        # 扩展 goal: (B, 2) → (B, num_shadow, 2)
        goal_expanded = goal.unsqueeze(1).expand(-1, self.num_shadow, -1)  # (B, num_shadow, 2)
        
        # 添加高斯噪声
        noise = torch.randn_like(goal_expanded) * 0.5  # 标准差 0.5
        noisy_goals = goal_expanded + noise  # (B, num_shadow, 2)
        
        # 2. 重塑为 (B*num_shadow, 2) 以便批量生成
        noisy_goals_flat = noisy_goals.reshape(B * self.num_shadow, 2)
        
        # 3. 扩展 scene 特征
        scene_expanded = scene.unsqueeze(1).expand(-1, self.num_shadow, -1, -1, -1)
        scene_flat = scene_expanded.reshape(B * self.num_shadow, *scene.shape[1:])
        
        # 4. 使用 model 生成轨迹
        with torch.no_grad():
            shadow_traj_flat = model.generate(
                goal=noisy_goals_flat,
                scene=scene_flat,
                num_steps=1,
                num_traj_points=T,
                method='euler'
            )  # (B*num_shadow, T, 2)
        
        # 5. 重塑回 (B, num_shadow, T, 2)
        shadow_trajectories = shadow_traj_flat.reshape(B, self.num_shadow, T, 2)
        
        return shadow_trajectories


    def select_best_trajectory(self,
                               trajectories: torch.Tensor,
                               scores: torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        选择最优轨迹

        Args:
            trajectories: (B, N, T, 2)  候选轨迹
            scores: (B, N) 评分

        Returns:
            best_trajectory: (B, T, 2) 最优轨迹
            best_indices: (B,) 最优轨迹的索引
        """
        B = trajectories.shape[0]
        best_score, best_indices = scores.max(dim=1)

        best_trajectory = trajectories[torch.arange(B), best_indices]

        return best_trajectory, best_indices


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
            drivable_area: (B, H, W)    可行驶区域 mask
            return_scores: 是否返回所有评分

        Returns:
            best_trajectory: (B, T, 2) 最优轨迹
            scores: (B, N) 所有评分
        """
        
        scores = self.compute_final_score(trajectories, 
                                          goal, 
                                          gt_trajectory,
                                          obstacle,
                                          drivable_area)
        best_trajectory, best_indices = self.select_best_trajectory(trajectories, scores)

        if return_scores:
            return best_trajectory, scores
        else:
            return best_trajectory, None


    def compute_ade(self, pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
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
        gt_expanded = gt_traj.unsqueeze(1)
        distances = torch.norm(pred_traj - gt_expanded, dim=-1)  # 计算欧氏距离用torch.norm
        ade = distances.mean(dim=-1)  # (B, N)

        return ade


    def compute_fde(self, pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
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
        pred_end = pred_traj[:, :, -1, :]   # (B, N, 2)
        gt_end = gt_traj[:, -1, :]   # (B, 2)
        gt_end_expanded = gt_end.unsqueeze(1)   # (B, 1, 2)
        fde = torch.norm(pred_end - gt_end_expanded, dim=-1)  # (B, N)

        return fde

        



