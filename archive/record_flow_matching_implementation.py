#!/usr/bin/env python3
"""
记录 Flow Matching 代码实现学习内容到 Mem0
"""
# ========== 强制清除代理环境变量 ==========
import os
for key in ['ALL_PROXY', 'all_proxy', 'HTTP_PROXY', 'http_proxy', 
            'HTTPS_PROXY', 'https_proxy', 'NO_PROXY', 'no_proxy']:
    if key in os.environ:
        print(f"清除代理环境变量: {key}={os.environ[key]}")
        del os.environ[key]
# =========================================
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents_system.runtime.config.settings import load_dotenv
load_dotenv()

from agents_system.runtime.memory import create_learning_tracker

def record_implementation_learning():
    """记录代码实现的学习内容"""
    print("=" * 60)
    print("记录 Flow Matching 代码实现学习内容到 Mem0")
    print("=" * 60)
    
    # 创建追踪器
    tracker = create_learning_tracker(user_id="zhn")
    print("\n✓ 学习追踪器创建成功")
    
    # 1. 记录代码实现的关键洞察
    print("\n[1] 记录代码实现的关键洞察...")
    implementation_insights = [
        "PyTorch 训练循环的标准流程：zero_grad() -> forward -> loss.backward() -> optimizer.step()",
        "验证时必须使用 torch.no_grad() 且不调用 backward() 和 step()",
        "数据处理：Dataset 返回字典，需要提取 'trajectory' 并 flatten 成向量",
        "轨迹数据形状转换：(B, 6, 2) -> (B, 12) 用于网络输入",
        "噪声采样：x_0 = torch.randn_like(x_1) * 0.5，标准差可调整",
        "梯度裁剪防止梯度爆炸：torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)",
        "学习率调度器：CosineAnnealingLR 实现余弦退火，逐渐降低学习率",
        "模型保存：保存 state_dict、optimizer、epoch、loss 等信息到 checkpoint",
        "ODE 求解器实现：Euler 是一阶方法 O(dt²)，RK4 是四阶方法 O(dt⁵)",
        "ConditionalFlowMatcher 实现：sample_ot_flow() 采样插值点，compute_cfm_loss() 计算损失",
        "SimpleVelocityField 设计：只需要 state 和 time，不需要 condition（适用于 toy dataset）",
        "时间编码：使用 SinusoidalEmbedding 将标量 t 编码为高维向量",
        "训练监控：使用 tqdm 显示进度条，实时显示 loss",
        "可视化技巧：matplotlib.use('Agg') 用于无显示环境，plt.close() 释放内存"
    ]
    
    for insight in implementation_insights:
        tracker.add_insight(
            insight=insight,
            paper_id="flow_matching_implementation",
            confidence=0.95
        )
    print(f"✓ 已记录 {len(implementation_insights)} 条实现洞察")
    
    # 2. 记录已解决的代码问题
    print("\n[2] 记录已解决的代码问题...")
    resolved_issues = [
        ("ISSUE_TRAIN_01", "数据处理错误：不能对字典使用 torch.rand_like()，需要先提取 tensor"),
        ("ISSUE_TRAIN_02", "验证函数错误：验证时不应该调用 zero_grad()、backward()、step()"),
        ("ISSUE_TRAIN_03", "数据形状不匹配：需要将 (B, 6, 2) flatten 成 (B, 12)"),
        ("ISSUE_TRAIN_04", "模型接口不匹配：VelocityFieldMLP 需要 3 个参数，toy dataset 只需要 2 个"),
        ("ISSUE_VIS_01", "可视化卡住：需要使用 matplotlib.use('Agg') 和 plt.close()"),
        ("ISSUE_TRAIN_05", "训练损失下降：从 9.99 降到 6.01，验证损失稳定在 6.0 左右")
    ]
    
    for issue_id, resolution in resolved_issues:
        tracker.resolve_knowledge_gap(
            gap_id=issue_id,
            resolution=resolution,
            confidence=0.95
        )
    print(f"✓ 已解决 {len(resolved_issues)} 个代码问题")
    
    # 3. 记录第一次写训练代码的常见困惑
    print("\n[3] 记录第一次写训练代码的常见困惑...")
    beginner_confusions = [
        {
            "question": "为什么要 zero_grad()？",
            "answer": "PyTorch 默认会累积梯度，每次反向传播前必须清零，否则梯度会叠加"
        },
        {
            "question": "backward() 和 step() 的区别？",
            "answer": "backward() 计算梯度并存储在 .grad 中，step() 根据梯度更新参数"
        },
        {
            "question": "train() 和 eval() 的区别？",
            "answer": "train() 启用 dropout 和 batch norm，eval() 关闭它们，用于验证和推理"
        },
        {
            "question": "为什么验证时用 no_grad()？",
            "answer": "不需要计算梯度，节省内存和计算时间，加快验证速度"
        },
        {
            "question": "如何处理 Dataset 返回的字典？",
            "answer": "使用 batch['key'] 提取需要的数据，然后 .to(device) 移到 GPU"
        },
        {
            "question": "为什么要 flatten 轨迹数据？",
            "answer": "网络需要向量输入，(6, 2) 的轨迹需要 flatten 成 (12,) 的向量"
        }
    ]
    
    for q in beginner_confusions:
        tracker.add_question(
            question=q["question"],
            paper_id="flow_matching_implementation",
            answered=True,
            answer=q["answer"]
        )
    print(f"✓ 已记录 {len(beginner_confusions)} 个常见困惑")
    
    # 4. 记录学习进度
    print("\n[4] 更新学习进度...")
    
    tracker.update_progress(
        paper_id="flow_matching_implementation",
        task="实现 ODE 求解器（Euler 和 RK4）",
        status="completed",
        understanding_level="advanced",
        notes="成功实现了 Euler 和 RK4 方法，理解了误差阶数的区别"
    )
    
    tracker.update_progress(
        paper_id="flow_matching_implementation",
        task="实现 ConditionalFlowMatcher 类",
        status="completed",
        understanding_level="advanced",
        notes="实现了 sample_ot_flow() 和 compute_cfm_loss() 方法"
    )
    
    tracker.update_progress(
        paper_id="flow_matching_implementation",
        task="实现训练脚本",
        status="completed",
        understanding_level="medium",
        notes="第一次写训练代码，遇到了数据处理和验证函数的问题，已解决"
    )
    
    tracker.update_progress(
        paper_id="flow_matching_implementation",
        task="在 Toy Dataset 上测试训练",
        status="completed",
        understanding_level="medium",
        notes="训练成功，损失从 9.99 降到 6.01，模型收敛"
    )
    
    tracker.update_progress(
        paper_id="flow_matching_implementation",
        task="可视化生成结果",
        status="completed",
        understanding_level="medium",
        notes="成功生成了 3 张可视化图片：生成轨迹、生成过程、对比真实数据"
    )
    
    print("✓ 学习进度已更新")
    
    # 5. 记录代码能力提升
    print("\n[5] 记录代码能力提升...")
    skill_improvements = [
        "掌握了 PyTorch 训练循环的标准流程",
        "理解了 Dataset 和 DataLoader 的使用方法",
        "学会了使用 tqdm 显示训练进度",
        "掌握了模型保存和加载的方法",
        "学会了使用 matplotlib 进行可视化",
        "理解了梯度裁剪的作用和使用方法",
        "掌握了学习率调度器的使用",
        "学会了调试训练代码的方法"
    ]
    
    for skill in skill_improvements:
        tracker.add_insight(
            insight=f"代码能力提升：{skill}",
            paper_id="flow_matching_implementation",
            confidence=0.9
        )
    print(f"✓ 已记录 {len(skill_improvements)} 项能力提升")
    
    # 6. 记录训练结果
    print("\n[6] 记录训练结果...")
    training_results = {
        "dataset": "Toy Trajectory Dataset (5000 train, 500 val)",
        "model": "SimpleVelocityField (236,556 parameters)",
        "epochs": 2,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
        "train_loss_epoch1": 9.994838,
        "val_loss_epoch1": 6.000483,
        "train_loss_epoch2": 6.006995,
        "val_loss_epoch2": 6.116862,
        "best_val_loss": 6.000483,
        "training_time": "~1 second per epoch (CPU)",
        "visualizations": [
            "generated_trajectories.png",
            "generation_process.png",
            "comparison.png"
        ]
    }
    
    tracker.add_insight(
        insight=f"训练结果：{training_results}",
        paper_id="flow_matching_implementation",
        confidence=1.0
    )
    print("✓ 训练结果已记录")
    
    # 7. 记录下一步计划
    print("\n[7] 记录下一步计划...")
    next_steps = [
        {
            "gap_id": "NEXT_STEP_01",
            "description": "增加训练 epochs，观察损失是否继续下降",
            "priority": "high",
            "next_steps": ["训练 50-100 epochs", "绘制损失曲线", "分析收敛情况"]
        },
        {
            "gap_id": "NEXT_STEP_02",
            "description": "实现更复杂的速度场网络（加入条件输入）",
            "priority": "high",
            "next_steps": ["设计条件编码器", "实现 VelocityFieldMLP", "在真实数据上测试"]
        },
        {
            "gap_id": "NEXT_STEP_03",
            "description": "实现推理脚本，生成新的轨迹",
            "priority": "medium",
            "next_steps": ["加载训练好的模型", "采样噪声", "使用 ODE 求解器生成轨迹"]
        },
        {
            "gap_id": "NEXT_STEP_04",
            "description": "评估生成质量（FID、多样性等指标）",
            "priority": "medium",
            "next_steps": ["实现 FID 计算", "计算轨迹多样性", "对比不同方法"]
        }
    ]
    
    for step in next_steps:
        tracker.add_knowledge_gap(
            gap_id=step["gap_id"],
            description=step["description"],
            priority=step["priority"],
            paper_id="flow_matching_implementation",
            next_steps=step["next_steps"]
        )
    print(f"✓ 已记录 {len(next_steps)} 个下一步计划")
    
    # 8. 获取学习总结
    print("\n[8] 生成学习总结...")
    summary = tracker.get_learning_summary("flow_matching_implementation")
    print("\n" + "=" * 60)
    print("代码实现学习总结")
    print("=" * 60)
    print(f"总记忆数: {summary['total_memories']}")
    print(f"完成任务: {summary['progress']['completed']}/{summary['progress']['total_tasks']}")
    print(f"已解决问题: {summary['knowledge_gaps']['resolved']}")
    print(f"下一步计划: {summary['knowledge_gaps']['pending']}")
    print(f"关键洞察: {summary['insights']}")
    print(f"问题: {summary['questions']['answered']}/{summary['questions']['total']} 已回答")
    print("=" * 60)
    
    print("\n✓ 所有代码实现学习内容已成功记录到 Mem0！")

if __name__ == "__main__":
    record_implementation_learning()
