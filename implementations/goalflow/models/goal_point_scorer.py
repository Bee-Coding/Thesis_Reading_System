import torch
import torch.nn as nn
import torch.nn.functional as F


class GoalPointScorer(nn.Module):
    def __init__(self, 
                 vocabulary_size: int, 
                 feature_dim: int, 
                 hidden_dim: int,
                 num_heads: int,
                 num_layers: int,
                 scene_in_channels: int,
                 kernel_size: int,
                 stride: int,
                 dropout: float=0.1):
        super().__init__()
        self.vocab_size = vocabulary_size
        self.feat = feature_dim
        self.vocab_projection = nn.Linear(2, feature_dim)
        vocab_encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.vocab_encoder = nn.TransformerEncoder(vocab_encoder_layer, num_layers)

        self.scene_encoder = nn.Conv2d(
            in_channels=scene_in_channels,
            out_channels=feature_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=1
        )
        self.scene_pool = nn.AdaptiveAvgPool2d(1)

        hidden_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                                      nn.ReLU())
        self.dist_MLP = nn.Sequential(nn.Linear(feature_dim*2, hidden_dim), 
                                      nn.ReLU(),
                                      *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                                      nn.ReLU()) for _ in range(num_layers)],
                                      nn.Linear(hidden_dim, 1),
                                      nn.ReLU())
        self.dac_MLP = nn.Sequential(nn.Linear(feature_dim*2, hidden_dim), 
                                    nn.ReLU(),
                                    *[hidden_layer for _ in range(num_layers)],
                                    nn.Linear(hidden_dim, 1),
                                    nn.Sigmoid())
        
    def compute_distance_score(self,
                               vocabulary: torch.Tensor,
                               gt_goal: torch.Tensor) -> float:
        """
        计算真实的距离分数分布（用作训练标签）
        Args:
            vocabulary:(B,N,D)或(N,D)
            gt_goal:(D,)或(B,D)
        Return:
            d_dis: (N,)或(B,N)
        """
        if gt_goal.dim() == 1:
            # 单样本
            diff = vocabulary - gt_goal     # 利用广播机制,gt_goal广播到(128,2)=(N,D)
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
            diff = vocabulary - gt_goal.unsqueeze(1)     # 利用广播机制,gt_goal广播到(32,1,2)=(B,1,D)
            distance_sq = torch.sum(diff**2, dim=-1)
            # # 计算exp(-d^2)
            # exp_neg_dist = torch.exp(-distance_sq)
            # # 计算sum(exp(-d^2))
            # sum_exp = torch.sum(exp_neg_dist)
            # # softmax
            # d_dis = exp_neg_dist / sum_exp
            d_dis = F.softmax(-distance_sq, dim=-1)
        
        return d_dis
    
    def compute_dac_score(self,
                          vocabulary: torch.Tensor,
                          drivable_area: torch.Tensor) -> float:
        """
        计算真实的DAC分数（用作训练标签）
        Args:
            vocabulary:(B,N,D)或(N,D)
            drivable_area: (B,H,W)
        Return:
            d_dac: (N,)或(B,N)
        """
        if vocabulary.dim() == 2:
            # 单个vocabulary，多个batch
            B, H, W = drivable_area.shape
            N = vocabulary.shape[0]

            # 将坐标归一化到[0, H-1]和[0,W-1]
            # 假设vocabulary坐标范围是[-50,50], 简化只考虑中心点
            x = vocabulary[:,0]
            y = vocabulary[:,1]

            # 归一化到图像坐标
            # BEV 感知中空间转换的标准操作
            x_img = ((x+50) / 100*(W-1)).long().clamp(0,W-1)        # long()取整函数 clamp()防止越界，确保在有效范围内min(max(xint​,0),W−1)
            y_img = ((y+50) / 100*(H-1)).long().clamp(0,H-1)

            # 检查每个batch中的每个点
            d_dac = torch.zeros(B, N, device=vocabulary.device)
            for b in range(B):
                d_dac[b] = drivable_area[b, y_img, x_img].float()
        else:
            B, H, W = drivable_area.shape
            N = vocabulary.shape[1]
            d_dac = torch.zeros(B, N, device=vocabulary.device)
            for b in range(B):
                x = vocabulary[b,:,0]
                y = vocabulary[b,:,1]

                x_img = ((x+50) / 100*(W-1)).long().clamp(0,W-1)        # long()取整函数 clamp()防止越界，确保在有效范围内min(max(xint​,0),W−1)
                y_img = ((y+50) / 100*(H-1)).long().clamp(0,H-1)

                d_dac[b] = drivable_area[b, y_img, x_img].float()

        return d_dac
            
    def forward(self, 
                vocabulary: torch.Tensor,
                scene_feature: torch.Tensor) -> float:
        """
        前向传播：预测距离分数和DAC分数
        Arg:
            vocabulary: (B,N,D) - 候选点集
            scene_feature: (B,C,H,W) - BEV场景特征

        Returns:
            pred_dis: (B,N) - 预测的距离分数（未归一化）
            pred_dac: (B,N) - 预测的DAC分数（0-1）
        """
        B, N, D = vocabulary.shape
        
        # 1. 编码vocabulary
        vocab_feat = self.vocab_projection(vocabulary)
        vocab_feat = self.vocab_encoder(vocab_feat)     # (B,N,feature_dim)

        # 2. 编码scene_feature
        scene_feat = self.scene_encoder(scene_feature)  # (B,feature_dim,H,W)
        scene_feat = self.scene_pool(scene_feat).squeeze(-1).squeeze(-1)     # (B,feature_dim)

        # 3. 扩展scene特征以匹配vocabulary
        scene_feat = scene_feat.unsqueeze(1).expand(-1,N,-1)

        # 4. 拼接特征
        combined_feat = torch.cat([vocab_feat, scene_feat], dim=-1)     #(B, N, feature_dim*2)

        # 5. 预测分数
        pred_dis = self.dist_MLP(combined_feat).squeeze(-1)
        pred_dac = self.dac_MLP(combined_feat).squeeze(-1)

        return pred_dis, pred_dac
    
    def compute_loss(self,
                     pred_dis: float,
                     pred_dac: float,
                     true_dis: float,
                     true_dac: float) -> float:
        """
        损失函数
        """

        # 1.Distance loss:交叉熵
        log_pred_dis = F.log_softmax(pred_dis, dim=-1)
        loss_dis = -torch.sum(true_dis*log_pred_dis, dim=-1).mean() # 交叉熵

        # 2.DAC loss: 二元交叉熵
        loss_dac = F.binary_cross_entropy(pred_dac, true_dac, reduction="mean")     # 二元交叉熵

        # 3.总损失（加权）
        w1 = 1.0
        w2 = 0.005
        loss = w1 * loss_dis + w2 * loss_dac

        loss_dict = {
            'loss': loss.item(),
            'loss_dis': loss_dis.item(),
            'loss_dac': loss_dac.item()
        }

        return loss, loss_dict
    
if __name__ == "__main__":
    # test_goal_point_scorer.py

    # 创建模型
    model = GoalPointScorer(
        vocabulary_size=128,
        feature_dim=256,
        hidden_dim=512,
        nhead=8,
        num_layers=2,
        in_channels=64,
        kernel_size=3,
        stride=1
    )

    # 测试数据
    B, N, D = 4, 128, 2
    vocabulary = torch.randn(B, N, D)
    scene_feature = torch.randn(B, 64, 32, 32)
    gt_goal = torch.randn(B, D)
    drivable_area = torch.randint(0, 2, (B, 32, 32)).float()

    # 前向传播
    pred_dis, pred_dac = model(vocabulary, scene_feature)
    print(f"pred_dis shape: {pred_dis.shape}")  # 应该是 (4, 128)
    print(f"pred_dac shape: {pred_dac.shape}")  # 应该是 (4, 128)

    # 计算标签
    true_dis = model.compute_distance_score(vocabulary, gt_goal)
    true_dac = model.compute_dac_score(vocabulary, drivable_area)
    print(f"true_dis shape: {true_dis.shape}")  # 应该是 (4, 128)
    print(f"true_dac shape: {true_dac.shape}")  # 应该是 (4, 128)

    # 计算损失
    loss, loss_dict = model.compute_loss(pred_dis, pred_dac, true_dis, true_dac)
    print(f"Loss: {loss_dict}")
        