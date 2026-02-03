# 可视化代码
import matplotlib.pyplot as plt
import numpy as np
import time

# 数据分布（两个高斯混合）
def data_distribution():
    if np.random.rand() > 0.5:
        return np.random.randn(2) + [2, 0]
    else:
        return np.random.randn(2) + [-2, 0]

# 加噪过程
x0 = data_distribution()
noise_levels = [0, 0.5, 1.0, 2.0, 5.0]

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, sigma in enumerate(noise_levels):
    # 从x0加噪到xt
    samples = [x0 + np.random.randn(2) * sigma for _ in range(100)]
    samples = np.array(samples)
    
    axes[i].scatter(samples[:, 0], samples[:, 1], alpha=0.3)
    axes[i].scatter(x0[0], x0[1], c='red', s=100, marker='*')
    axes[i].set_title(f't={i}, σ={sigma}')
    axes[i].set_xlim(-8, 8)
    axes[i].set_ylim(-8, 8)
    time.sleep(0.5)
    

plt.suptitle('Noise Adding Process: From Data Point (Red Star) to Noise Cloud')
plt.tight_layout()
plt.show()