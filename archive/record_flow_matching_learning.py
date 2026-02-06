#!/usr/bin/env python3
"""
记录 Flow Matching 学习内容到 Mem0
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents_system.runtime.config.settings import load_dotenv
load_dotenv()

from agents_system.runtime.memory import create_learning_tracker

def record_learning():
    """记录今天的学习内容"""
    print("=" * 60)
    print("记录 Flow Matching 学习内容到 Mem0")
    print("=" * 60)
    
    # 创建追踪器
    tracker = create_learning_tracker(user_id="zhn")
    print("\n✓ 学习追踪器创建成功")
    
    # 1. 开始学习论文
    print("\n[1] 记录论文信息...")
    tracker.start_paper(
        paper_id="flow_matching_2023",
        paper_title="Flow Matching for Generative Modeling",
        metadata={
            "authors": "Lipman et al.",
            "year": 2023,
            "venue": "ICLR"
        }
    )
    print("✓ 论文信息已记录")
    
    # 2. 记录关键洞察
    print("\n[2] 记录关键洞察...")
    insights = [
        "Flow Matching 学习的是条件速度场 v_θ(x_t, t, g, c)，而非特定轨迹",
        "速度场的本质：理想是常数，现实是近似常数+时间校正",
        "Goal Point 全局作用，直接编码方向信息",
        "OT Flow 使用线性插值：x_t = (1-t)x_0 + tx_1，速度场为常数 v = x_1 - x_0",
        "CFM Loss 学习的是边际速度场：v*(x_t,t) = E[x_1-x_0 | x_t]",
        "条件期望处理多模态：同一位置可能来自不同路径，学习平均方向",
        "平均速度为零不会出问题：表示概率流平衡，不是不动",
        "RK4 (Runge-Kutta 4th Order) 是 ODE 求解的黄金标准，误差 O(dt^5)",
        "RK4 使用四个阶段：起点、中点1、中点2、终点，权重 (1/6, 2/6, 2/6, 1/6)",
        "训练目标：最小化 E[||v_θ(x_t,t) - (x_1-x_0)||²] 等价于学习真实速度场"
    ]
    
    for insight in insights:
        tracker.add_insight(
            insight=insight,
            paper_id="flow_matching_2023",
            confidence=0.95
        )
    print(f"✓ 已记录 {len(insights)} 条关键洞察")
    
    # 3. 记录已解决的知识盲区
    print("\n[3] 记录已解决的知识盲区...")
    resolved_gaps = [
        ("GAP_GOALFLOW_05", "速度场的时间依赖性：网络学习近似常数速度场，t用于误差校正"),
        ("GAP_GOALFLOW_06", "条件信息的作用机制：Goal Point全局作用，直接编码方向"),
        ("GAP_GOALFLOW_07", "学习目标的精确定义：学习条件速度场，预测期望方向"),
        ("GAP_FLOWMATCHING_03", "CFM Loss 期望公式的含义：两层期望处理多模态，学习边际速度场"),
        ("GAP_FLOWMATCHING_04", "RK4 方法的原理：四阶精度，通过多点采样和加权平均提高精度")
    ]
    
    for gap_id, resolution in resolved_gaps:
        tracker.resolve_knowledge_gap(
            gap_id=gap_id,
            resolution=resolution,
            confidence=0.95
        )
    print(f"✓ 已解决 {len(resolved_gaps)} 个知识盲区")
    
    # 4. 记录待解决的知识盲区
    print("\n[4] 记录待解决的知识盲区...")
    pending_gaps = [
        {
            "gap_id": "GAP_FLOWMATCHING_01",
            "description": "Flow Matching 的理论收敛性证明",
            "priority": "medium",
            "next_steps": ["查阅 Rectified Flow 论文", "理解 ODE 收敛性理论"]
        },
        {
            "gap_id": "GAP_FLOWMATCHING_02",
            "description": "Goal Point Vocabulary 密度优化",
            "priority": "high",
            "next_steps": ["实验验证不同 K 值的影响", "分析密度与性能的关系"]
        },
        {
            "gap_id": "GAP_FLOWMATCHING_05",
            "description": "自适应 ODE 求解器的实现",
            "priority": "low",
            "next_steps": ["学习 Dopri5 方法", "实现自适应步长控制"]
        }
    ]
    
    for gap in pending_gaps:
        tracker.add_knowledge_gap(
            gap_id=gap["gap_id"],
            description=gap["description"],
            priority=gap["priority"],
            paper_id="flow_matching_2023",
            next_steps=gap["next_steps"]
        )
    print(f"✓ 已记录 {len(pending_gaps)} 个待解决的知识盲区")
    
    # 5. 记录学习进度
    print("\n[5] 更新学习进度...")
    tracker.update_progress(
        paper_id="flow_matching_2023",
        task="理解 Flow Matching 数学原理",
        status="completed",
        understanding_level="advanced",
        notes="已深入理解 OT Flow、CFM Loss、条件期望、RK4 求解器"
    )
    
    tracker.update_progress(
        paper_id="flow_matching_2023",
        task="理解 CFM Loss 的条件期望公式",
        status="completed",
        understanding_level="advanced",
        notes="理解了两层期望的含义和边际速度场的概念"
    )
    
    tracker.update_progress(
        paper_id="flow_matching_2023",
        task="深入理解 RK4 方法",
        status="completed",
        understanding_level="advanced",
        notes="理解了四阶精度的原理和泰勒展开匹配"
    )
    
    tracker.update_progress(
        paper_id="flow_matching_2023",
        task="实现 Flow Matching 代码",
        status="in_progress",
        understanding_level="medium",
        notes="准备开始实现 ConditionalFlowMatcher 和 RK4 求解器"
    )
    print("✓ 学习进度已更新")
    
    # 6. 记录重要问题和答案
    print("\n[6] 记录重要问题...")
    questions = [
        {
            "question": "为什么 OT Flow 的速度场是常数？",
            "answer": "因为 x_t = (1-t)x_0 + tx_1 是线性插值，对 t 求导得 v = x_1 - x_0，与 t 无关"
        },
        {
            "question": "平均速度为零会导致网络无法前进吗？",
            "answer": "不会。平均速度为零表示概率流平衡（流入=流出），不是不动。且只在特定时刻发生，其他时刻速度不为零"
        },
        {
            "question": "RK4 的全称是什么？",
            "answer": "Runge-Kutta 4th Order Method（四阶龙格-库塔方法），由 Carl Runge 和 Wilhelm Kutta 在 1895-1901 年发明"
        },
        {
            "question": "为什么 RK4 的权重是 (1/6, 2/6, 2/6, 1/6)？",
            "answer": "通过泰勒展开匹配，这个权重组合能精确匹配到 dt^4 项，使误差为 O(dt^5)"
        }
    ]
    
    for q in questions:
        tracker.add_question(
            question=q["question"],
            paper_id="flow_matching_2023",
            answered=True,
            answer=q["answer"]
        )
    print(f"✓ 已记录 {len(questions)} 个问题和答案")
    
    # 7. 获取学习总结
    print("\n[7] 生成学习总结...")
    summary = tracker.get_learning_summary("flow_matching_2023")
    print("\n" + "=" * 60)
    print("学习总结")
    print("=" * 60)
    print(f"总记忆数: {summary['total_memories']}")
    print(f"完成任务: {summary['progress']['completed']}/{summary['progress']['total_tasks']}")
    print(f"知识盲区: {summary['knowledge_gaps']['pending']} 待解决, {summary['knowledge_gaps']['resolved']} 已解决")
    print(f"关键洞察: {summary['insights']}")
    print(f"问题: {summary['questions']['answered']}/{summary['questions']['total']} 已回答")
    print("=" * 60)
    
    print("\n✓ 所有学习内容已成功记录到 Mem0！")

if __name__ == "__main__":
    record_learning()
