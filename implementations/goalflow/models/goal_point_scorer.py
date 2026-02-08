import torch
import torch.nn as nn
import torch.nn.functional as F


class GoalPointScorer(nn.Module):
    def __init__(self, 
                 vocabulary_size: int, 
                 feature_dim: int, 
                 hidden_dim: int,
                 nhead: int,
                 num_layers: int,
                 dropout: float=0.1):
        super().__init__()
        self.vocab_size = vocabulary_size
        self.feat = feature_dim
        vocab_encoder_layer = nn.TransformerEncoderLayer()
        self.vocab_encoder = nn.TransformerEncoder(vocab_encoder_layer, num_layers)
        self.scene_encoder = nn.Conv2d()
        self.softmax = nn.Softmax(dim=1)
        hidden_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                                      nn.ReLU())
        self.dist_MLP = nn.Sequential(nn.Linear(vocabulary_size*3, hidden_dim), 
                                      nn.ReLU(),
                                      hidden_layer*num_layers,
                                      nn.Linear(hidden_dim, feature_dim),
                                      nn.ReLU())
        self.dac_MLP = nn.Sequential(nn.Linear(vocabulary_size*3, hidden_dim), 
                                    nn.ReLU(),
                                    hidden_layer*num_layers,
                                    nn.Linear(hidden_dim, feature_dim),
                                    nn.ReLU())
        
        def compute_distance_score(self, 
                                   vocabulary: torch.Tensor,
                                   gt_goal: torch.Tensor) -> float:
            """
            计算真实的距离番薯分布（用作训练标签）
            Args:
                vocabulary:(B,N,D)
                gt_goal:(D,)或(B,D)
            Return:
                d_dis: (N,)或(B,N)
            """
            if gt_goal.dim() == 1:
                # 单样本
                diff = vocabulary - gt_goal     # 利用广播机制,gt_goal广播到(128,2)
                distance_sq = torch.sum(diff**2, dim=1)
                # # 计算exp(-d^2)
                # exp_neg_dist = torch.exp(-distance_sq)
                # # 计算sum(exp(-d^2))
                # sum_exp = torch.sum(exp_neg_dist)
                # # softmax
                # d_dis = exp_neg_dist / sum_exp
                d_dis = F.softmax(-distance_sq, dim=0)
            else:
                # 批量样本
                diff = vocabulary.unsqueeze(0) - gt_goal.unsqueeze(1)     # 利用广播机制,gt_goal广播到(32,1,2)
                distance_sq = torch.sum(diff**2, dim=1)
                # # 计算exp(-d^2)
                # exp_neg_dist = torch.exp(-distance_sq)
                # # 计算sum(exp(-d^2))
                # sum_exp = torch.sum(exp_neg_dist)
                # # softmax
                # d_dis = exp_neg_dist / sum_exp
                d_dis = F.softmax(-distance_sq, dim=1)
            
            return d_dis
        
        def compute_dac_score(self, 
                              vocabulary: torch.Tensor,
                              drivable_area: torch.Tensor) -> float:
            pass
        
        def forward(self, 
                    vocabulary: torch.Tensor,
                    scene_feature: torch.Tensor) -> float:
            pass
        
        def compute_loss(self,
                         pred_dis: float,
                         pred_dac: float,
                         true_dis: float,
                         true_dac: float):
            pass
        