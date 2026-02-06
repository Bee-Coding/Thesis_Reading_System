#!/usr/bin/env python3
"""
记录 GoalFlow 论文学习内容到 Mem0
更新时间：2026-02-06
当前阶段：阶段 1 - 深入理解 GoalFlow 论文（80% 完成）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents_system.runtime.config.settings import load_dotenv
load_dotenv()

from agents_system.runtime.memory import create_learning_tracker

def record_goalflow_learning():
    """记录 GoalFlow 论文学习内容"""
    print("=" * 60)
    print("记录 GoalFlow 论文学习内容到 Mem0")
    print("=" * 60)
    
    tracker = create_learning_tracker(user_id="zhn")
    print("\n✓ 学习追踪器创建成功")
    
    # 记录论文信息
    print("\n[1] 记录论文信息...")
    tracker.start_paper(
        paper_id="goalflow_2025",
        paper_title="GoalFlow: Goal-Driven Flow Matching for Multimodal Trajectories Generation",
        metadata={
            "authors": "Zebin Xing, Xingyu Zhang, Yang Hu, et al.",
            "year": 2025,
            "venue": "arXiv",
            "institution": "Horizon Robotics",
            "code": "https://github.com/YvanYin/GoalFlow"
        }
    )
    print("✓ 论文信息已记录")
    
    print("\n✓ GoalFlow 学习内容已成功记录到 Mem0！")

if __name__ == "__main__":
    record_goalflow_learning()
