"""
Scorer 诊断脚本
测试 Scorer 的目标点选择准确性

目的：
- 计算 Goal Error（选中的目标点 vs 真实目标点的距离）
- 计算 Top-1/Top-5 准确率（真实目标点是否在 Top-K 中）
- 分析 pred_dis 和 pred_dac 的贡献
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.goal_point_scorer import GoalPointScorer
from data.toy_goalflow_dataset import ToyGoalFlowDataset
from config.scorer_config import ScorerConfig


def diagnose_scorer(scorer_checkpoint_path: str,
                    data_path: str,
                    device: str = 'cpu'):
    """
    诊断 Scorer 的目标点选择性能
    
    Args:
        scorer_checkpoint_path: Scorer checkpoint 路径
        data_path: 数据路径
        device: 设备
    
    Returns:
        results: 诊断结果字典
    """
    print("=" * 70)
    print("Scorer Diagnosis: Goal Selection Accuracy")
    print("=" * 70)
    
    # 1. 加载 Scorer 模型
    print("\n[1/4] Loading Scorer model...")
    scorer_config = ScorerConfig()
    
    # 加载数据集获取 vocabulary
    train_dataset = ToyGoalFlowDataset(data_path, split='train')
    vocabulary = train_dataset.get_vocabulary().to(device)  # (N, 2)
    vocab_size = vocabulary.shape[0]
    
    scorer = GoalPointScorer(
        vocabulary_size=vocab_size,
        feature_dim=scorer_config.hidden_dim,
        hidden_dim=scorer_config.hidden_dim,
        num_heads=scorer_config.num_heads,
        num_layers=scorer_config.num_layers,
        scene_in_channels=scorer_config.scene_channels,
        kernel_size=3,
        stride=1,
        dropout=scorer_config.dropout
    ).to(device)
    
    checkpoint = torch.load(scorer_checkpoint_path, map_location=device)
    scorer.load_state_dict(checkpoint['model_state_dict'])
    scorer.eval()
    print(f"   ✓ Scorer loaded from: {scorer_checkpoint_path}")
    print(f"   ✓ Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   ✓ Vocabulary size: {vocab_size}")
    
    # 2. 加载测试数据
    print("\n[2/4] Loading test dataset...")
    test_dataset = ToyGoalFlowDataset(data_path, split='val')
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    print(f"   ✓ Test dataset loaded: {len(test_dataset)} samples")
    
    # 3. 测试 Scorer
    print("\n[3/4] Testing Scorer goal selection...")
    
    all_goal_errors = []
    top1_correct = 0
    top5_correct = 0
    total_samples = 0
    
    # 用于分析 pred_dis 和 pred_dac 的贡献
    dis_only_errors = []
    dac_only_errors = []
    combined_errors = []
    
    pbar = tqdm(test_loader, desc='Testing')
    
    with torch.no_grad():
        for batch in pbar:
            scene_feature = batch['bev_feature'].to(device)  # (B, C, H, W)
            gt_goal = batch['goal'].to(device)  # (B, 2)
            B = scene_feature.shape[0]
            
            # 扩展 vocabulary
            vocab_expanded = vocabulary.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)
            
            # 前向传播
            pred_dis, pred_dac = scorer(vocab_expanded, scene_feature)
            # pred_dis: (B, N) - logits
            # pred_dac: (B, N) - probabilities
            
            # 计算最终评分（与 inference.py 一致）
            log_pred_dis = F.log_softmax(pred_dis, dim=-1)  # (B, N)
            log_pred_dac = torch.log(pred_dac + 1e-8)  # (B, N)
            
            w1 = 1.0
            w2 = 0.005
            final_scores = w1 * log_pred_dis + w2 * log_pred_dac  # (B, N)
            
            # 分别计算三种选择方式
            dis_only_idx = log_pred_dis.argmax(dim=-1)  # (B,)
            dac_only_idx = log_pred_dac.argmax(dim=-1)  # (B,)
            combined_idx = final_scores.argmax(dim=-1)  # (B,)
            
            # 获取选中的目标点
            dis_only_goal = vocabulary[dis_only_idx]  # (B, 2)
            dac_only_goal = vocabulary[dac_only_idx]  # (B, 2)
            combined_goal = vocabulary[combined_idx]  # (B, 2)
            
            # 计算 Goal Error
            dis_error = torch.norm(dis_only_goal - gt_goal, dim=-1)  # (B,)
            dac_error = torch.norm(dac_only_goal - gt_goal, dim=-1)  # (B,)
            combined_error = torch.norm(combined_goal - gt_goal, dim=-1)  # (B,)
            
            dis_only_errors.extend(dis_error.cpu().numpy())
            dac_only_errors.extend(dac_error.cpu().numpy())
            combined_errors.extend(combined_error.cpu().numpy())
            all_goal_errors.extend(combined_error.cpu().numpy())
            
            # 计算 Top-K 准确率
            # 找到与 gt_goal 最近的 vocabulary 点
            distances = torch.norm(vocabulary.unsqueeze(0) - gt_goal.unsqueeze(1), dim=-1)  # (B, N)
            closest_idx = distances.argmin(dim=-1)  # (B,)
            
            # Top-1 准确率
            top1_correct += (combined_idx == closest_idx).sum().item()
            
            # Top-5 准确率
            top5_indices = final_scores.topk(5, dim=-1).indices  # (B, 5)
            for i in range(B):
                if closest_idx[i] in top5_indices[i]:
                    top5_correct += 1
            
            total_samples += B
            
            pbar.set_postfix({
                'Goal Error': f'{np.mean(all_goal_errors):.3f}',
                'Top-1 Acc': f'{top1_correct/total_samples*100:.1f}%'
            })
    
    # 4. 计算统计结果
    print("\n[4/4] Computing statistics...")
    results = {
        'avg_goal_error': np.mean(all_goal_errors),
        'std_goal_error': np.std(all_goal_errors),
        'median_goal_error': np.median(all_goal_errors),
        'top1_accuracy': top1_correct / total_samples * 100,
        'top5_accuracy': top5_correct / total_samples * 100,
        'dis_only_error': np.mean(dis_only_errors),
        'dac_only_error': np.mean(dac_only_errors),
        'combined_error': np.mean(combined_errors),
    }
    
    # 5. 打印诊断结果
    print("\n" + "=" * 70)
    print("DIAGNOSIS RESULTS (Goal Selection)")
    print("=" * 70)
    print(f"Goal Error (Combined):          {results['avg_goal_error']:.4f} ± {results['std_goal_error']:.4f} m")
    print(f"Goal Error (Median):            {results['median_goal_error']:.4f} m")
    print(f"Top-1 Accuracy:                 {results['top1_accuracy']:.2f}%")
    print(f"Top-5 Accuracy:                 {results['top5_accuracy']:.2f}%")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("Component Analysis (pred_dis vs pred_dac)")
    print("=" * 70)
    print(f"Goal Error (dis only):          {results['dis_only_error']:.4f} m")
    print(f"Goal Error (dac only):          {results['dac_only_error']:.4f} m")
    print(f"Goal Error (combined):          {results['combined_error']:.4f} m")
    print("=" * 70)
    
    # 6. 诊断结论
    print("\n" + "=" * 70)
    print("DIAGNOSIS CONCLUSION")
    print("=" * 70)
    
    # 加载数据统计信息
    data = np.load(data_path)
    trajectories = data['trajectories']
    traj_lengths = np.linalg.norm(np.diff(trajectories, axis=1), axis=-1).sum(axis=1)
    avg_traj_length = np.mean(traj_lengths)
    
    print(f"\nData Context:")
    print(f"  - Average trajectory length: {avg_traj_length:.2f} m")
    print(f"  - Vocabulary size: {vocab_size}")
    
    print(f"\nScorer Performance:")
    print(f"  - Goal Error: {results['avg_goal_error']:.4f} m ({results['avg_goal_error']/avg_traj_length*100:.1f}% of traj length)")
    print(f"  - Top-1 Accuracy: {results['top1_accuracy']:.2f}%")
    print(f"  - Top-5 Accuracy: {results['top5_accuracy']:.2f}%")
    
    # 判断标准
    if results['top1_accuracy'] > 80:
        print("\n✅ SCORER IS WORKING WELL!")
        print("   → Goal selection is accurate")
        print("   → If overall ADE is still high, the problem is in Matcher")
    elif results['top1_accuracy'] > 50:
        print("\n⚠️  SCORER IS PARTIALLY WORKING")
        print("   → Goal selection has moderate accuracy")
        print("   → Consider retraining Scorer with:")
        print("     - Better feature extraction")
        print("     - Adjusted loss weights")
        print("     - More training epochs")
    else:
        print("\n❌ SCORER TRAINING FAILED!")
        print("   → Goal selection is poor (< 50% accuracy)")
        print("   → Scorer is essentially guessing randomly")
        print("\n   Possible causes:")
        print("   1. Training loss did not converge")
        print("   2. Feature extraction is not effective")
        print("   3. Loss weights (lambda_dis, lambda_dac) are imbalanced")
        print("   4. Vocabulary does not cover goal distribution well")
        print("\n   Recommended actions:")
        print("   1. Check training logs for loss curves")
        print("   2. Visualize vocabulary coverage")
        print("   3. Try retraining with adjusted hyperparameters")
    
    # 分析 pred_dis vs pred_dac
    print("\n" + "-" * 70)
    print("Component Analysis:")
    if results['dis_only_error'] < results['combined_error']:
        print("   ⚠️  pred_dac is HURTING performance!")
        print(f"      - Using dis only: {results['dis_only_error']:.4f} m")
        print(f"      - Using combined:  {results['combined_error']:.4f} m")
        print("   → Consider reducing lambda_dac weight or removing it")
    elif results['dac_only_error'] < results['combined_error']:
        print("   ⚠️  pred_dis is HURTING performance!")
        print(f"      - Using dac only: {results['dac_only_error']:.4f} m")
        print(f"      - Using combined:  {results['combined_error']:.4f} m")
        print("   → Consider increasing lambda_dac weight")
    else:
        print("   ✓ Both components are contributing positively")
        print(f"      - dis only:  {results['dis_only_error']:.4f} m")
        print(f"      - dac only:  {results['dac_only_error']:.4f} m")
        print(f"      - combined:  {results['combined_error']:.4f} m")
    
    print("=" * 70)
    
    return results


def main():
    """主函数"""
    # 参数设置
    scorer_checkpoint_path = 'checkpoints/scorer/best.pth'
    data_path = 'data/toy_data.npz'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nDevice: {device}")
    print(f"Scorer checkpoint: {scorer_checkpoint_path}")
    print(f"Data path: {data_path}\n")
    
    # 检查文件
    if not os.path.exists(scorer_checkpoint_path):
        print(f"❌ ERROR: Scorer checkpoint not found: {scorer_checkpoint_path}")
        return
    if not os.path.exists(data_path):
        print(f"❌ ERROR: Data file not found: {data_path}")
        return
    
    # 运行诊断
    results = diagnose_scorer(
        scorer_checkpoint_path=scorer_checkpoint_path,
        data_path=data_path,
        device=device
    )
    
    print("\n✓ Diagnosis completed!")


if __name__ == "__main__":
    main()
