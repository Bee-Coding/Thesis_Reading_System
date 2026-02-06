"""
Flow Matching 可视化脚本
用于可视化训练好的模型生成的轨迹
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from models.flow_matcher import ConditionalFlowMatcher
from train import SimpleVelocityField
from data.toy_dataset import ToyTrajectoryDataset


def load_model(checkpoint_path: str, device: torch.device):
    """加载训练好的模型"""
    # 创建模型
    model = SimpleVelocityField(
        state_dim=12,
        time_dim=128,
        hidden_dim=256,
        num_layers=4,
        dropout=0.1
    ).to(device)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ 加载模型: {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch'] + 1}")
    print(f"  Val Loss: {checkpoint['val_loss']:.6f}")
    
    return model


def generate_trajectories(
    model: torch.nn.Module,
    flow_matcher: ConditionalFlowMatcher,
    num_samples: int = 16,
    num_steps: int = 50,
    method: str = 'rk4',
    device: torch.device = torch.device('cpu')
):
    """生成轨迹"""
    model.eval()
    
    with torch.no_grad():
        # 采样初始噪声
        x_0 = torch.randn(num_samples, 12, device=device) * 0.5
        
        # 生成轨迹
        x_1, trajectory = flow_matcher.sample_trajectory(
            model, x_0, num_steps=num_steps, method=method
        )
        
        # 转换为 numpy 并 reshape
        trajectory_np = [t.cpu().numpy().reshape(-1, 6, 2) for t in trajectory]
        x_1_np = x_1.cpu().numpy().reshape(-1, 6, 2)
    
    return x_1_np, trajectory_np


def plot_trajectories(
    generated_trajs: np.ndarray,
    real_trajs: np.ndarray = None,
    save_path: str = None,
    title: str = "Generated Trajectories"
):
    """
    绘制轨迹
    
    Args:
        generated_trajs: 生成的轨迹 (N, 6, 2)
        real_trajs: 真实轨迹 (N, 6, 2)，可选
        save_path: 保存路径
        title: 图表标题
    """
    num_samples = min(16, len(generated_trajs))
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for idx in range(num_samples):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]
        
        # 绘制生成的轨迹
        traj = generated_trajs[idx]  # (6, 2)
        ax.plot(traj[:, 0], traj[:, 1], 'b-o', linewidth=2, 
                markersize=6, label='Generated', alpha=0.8)
        
        # 标记起点和终点
        ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Start')
        ax.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=10, label='End')
        
        # 如果有真实轨迹，也绘制出来
        if real_trajs is not None and idx < len(real_trajs):
            real_traj = real_trajs[idx]
            ax.plot(real_traj[:, 0], real_traj[:, 1], 'k--', 
                   linewidth=1.5, alpha=0.5, label='Real')
        
        # 设置坐标轴
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Sample {idx + 1}')
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 保存图片: {save_path}")
    
    plt.close()


def plot_generation_process(
    trajectory_list: list,
    sample_idx: int = 0,
    save_path: str = None
):
    """
    绘制生成过程（从噪声到数据的演化）
    
    Args:
        trajectory_list: 轨迹列表，每个元素是 (N, 6, 2)
        sample_idx: 要可视化的样本索引
        save_path: 保存路径
    """
    # 选择要可视化的时间步
    num_steps = len(trajectory_list)
    step_indices = [0, num_steps//4, num_steps//2, 3*num_steps//4, num_steps-1]
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(f'Generation Process (Sample {sample_idx + 1})', 
                 fontsize=16, fontweight='bold')
    
    for i, step_idx in enumerate(step_indices):
        ax = axes[i]
        traj = trajectory_list[step_idx][sample_idx]  # (6, 2)
        
        # 绘制轨迹
        ax.plot(traj[:, 0], traj[:, 1], 'b-o', linewidth=2, markersize=6)
        ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=10)
        ax.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=10)
        
        # 设置坐标轴
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f't = {step_idx / (num_steps - 1):.2f}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 保存图片: {save_path}")
    
    plt.close()


def compare_with_real_data(
    model: torch.nn.Module,
    flow_matcher: ConditionalFlowMatcher,
    dataset: ToyTrajectoryDataset,
    num_samples: int = 16,
    device: torch.device = torch.device('cpu'),
    save_path: str = None
):
    """对比生成数据和真实数据"""
    # 生成轨迹
    generated_trajs, _ = generate_trajectories(
        model, flow_matcher, num_samples, device=device
    )
    
    # 获取真实轨迹
    real_trajs = []
    for i in range(num_samples):
        sample = dataset[i]
        traj = sample['trajectory'].numpy().reshape(6, 2)
        real_trajs.append(traj)
    real_trajs = np.array(real_trajs)
    
    # 绘制对比图
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Generated vs Real Trajectories', fontsize=16, fontweight='bold')
    
    for idx in range(num_samples):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]
        
        # 绘制生成的轨迹
        gen_traj = generated_trajs[idx]
        ax.plot(gen_traj[:, 0], gen_traj[:, 1], 'b-o', 
               linewidth=2, markersize=6, label='Generated', alpha=0.8)
        
        # 绘制真实轨迹
        real_traj = real_trajs[idx]
        ax.plot(real_traj[:, 0], real_traj[:, 1], 'r--s', 
               linewidth=2, markersize=6, label='Real', alpha=0.8)
        
        # 标记起点
        ax.plot(gen_traj[0, 0], gen_traj[0, 1], 'go', markersize=10)
        ax.plot(real_traj[0, 0], real_traj[0, 1], 'go', markersize=10)
        
        # 设置坐标轴
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Sample {idx + 1}')
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 保存图片: {save_path}")
    
    plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='可视化 Flow Matching 生成结果')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best.pth',
                       help='模型检查点路径')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='数据目录')
    parser.add_argument('--save_dir', type=str, default='./visualizations',
                       help='可视化结果保存目录')
    parser.add_argument('--num_samples', type=int, default=16,
                       help='生成样本数量')
    parser.add_argument('--num_steps', type=int, default=50,
                       help='ODE 求解步数')
    parser.add_argument('--method', type=str, default='rk4',
                       choices=['euler', 'rk4'], help='ODE 求解方法')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备')
    
    args = parser.parse_args()
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 加载模型
    model = load_model(args.checkpoint, device)
    
    # 创建 Flow Matcher
    flow_matcher = ConditionalFlowMatcher(sigma=0.0)
    
    # 加载验证数据集
    data_dir = Path(args.data_dir)
    val_dataset = ToyTrajectoryDataset(str(data_dir / 'toy_val.npz'))
    
    print("\n" + "="*60)
    print("开始生成和可视化")
    print("="*60)
    
    # 1. 生成轨迹
    print("\n[1/3] 生成轨迹...")
    generated_trajs, trajectory_list = generate_trajectories(
        model, flow_matcher, 
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        method=args.method,
        device=device
    )
    
    # 2. 绘制生成的轨迹
    print("\n[2/3] 绘制生成的轨迹...")
    plot_trajectories(
        generated_trajs,
        save_path=str(save_dir / 'generated_trajectories.png'),
        title=f'Generated Trajectories ({args.method.upper()}, {args.num_steps} steps)'
    )
    
    # 3. 绘制生成过程
    print("\n[3/3] 绘制生成过程...")
    plot_generation_process(
        trajectory_list,
        sample_idx=0,
        save_path=str(save_dir / 'generation_process.png')
    )
    
    # 4. 对比真实数据
    print("\n[4/4] 对比真实数据...")
    compare_with_real_data(
        model, flow_matcher, val_dataset,
        num_samples=args.num_samples,
        device=device,
        save_path=str(save_dir / 'comparison.png')
    )
    
    print("\n" + "="*60)
    print("可视化完成!")
    print(f"结果保存在: {save_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
