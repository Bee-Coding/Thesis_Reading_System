"""
快速测试训练脚本 - 只训练 3 个 epoch 验证流程
"""

import sys
import os

# 修改配置为快速测试
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.scorer_config import ScorerConfig

# 覆盖配置
class TestConfig(ScorerConfig):
    num_epochs = 3           # 只训练 3 个 epoch
    batch_size = 8           # 小批次
    eval_interval = 1        # 每个 epoch 都验证
    save_interval = 3        # 最后保存
    num_workers = 0          # 避免多进程问题
    device = 'cpu'           # 使用 CPU 测试

# 替换配置
import train_goal_scorer
train_goal_scorer.ScorerConfig = TestConfig

# 运行训练
if __name__ == "__main__":
    train_goal_scorer.main()
