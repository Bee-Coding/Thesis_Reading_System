import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_dim=128, max_period=10000):
        """
        Args: 
            embedding_dim: 编码维度（必须是偶数）
            max_period: 最大周期
        """
        super().__init__()
        self.dim = embedding_dim
        self.max_period = max_period

    def forward(self, t):
        """
        Args:
            t: [B]时间标量，范围[0, 1]
        Returns:
            [B, embedding_dim] 时间编码向量
        """
        device = t.device
        half_dim = self.dim // 2
        # 计算频率
        # freq_i = 1 / (max_period^(2i/dim))
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(half_dim, device=device) / half_dim)
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding
    
if __name__ == "__main__":
    embedding = SinusoidalEmbedding(embedding_dim=128, max_period=10000)
    
    t = torch.tensor([0.5])
    out = embedding(t)
    print(f"input shape: {t.shape}")
    print(f"output shape: {out.shape}")
    assert out.shape == (1, 128)

    t_batch = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    out_batch = embedding(t_batch)
    print(f"_batch input shape: {t_batch.shape}")
    print(f"_batch output shape: {out_batch.shape}")
    assert out_batch.shape == (5, 128)

    print("Success!")

