"""
nuScenes 工具函数

提供坐标转换、数据处理等辅助功能
"""

import numpy as np
from typing import Dict, List, Tuple
from pyquaternion import Quaternion


def global_to_ego(
    points: np.ndarray,
    ego_translation: np.ndarray,
    ego_rotation: Quaternion
) -> np.ndarray:
    """
    将全局坐标转换到 ego 车坐标系
    
    Args:
        points: (N, 2) or (N, 3) 全局坐标点
        ego_translation: (3,) ego 车位置
        ego_rotation: Quaternion ego 车旋转
    
    Returns:
        points_ego: (N, 2) or (N, 3) ego 坐标系下的点
    """
    # 确保 points 是 3D
    if points.shape[-1] == 2:
        points_3d = np.concatenate([points, np.zeros((len(points), 1))], axis=-1)
    else:
        points_3d = points.copy()
    
    # 平移
    points_translated = points_3d - ego_translation
    
    # 旋转（使用逆旋转）
    points_rotated = np.array([
        ego_rotation.inverse.rotate(p) for p in points_translated
    ])
    
    # 返回 2D 或 3D
    if points.shape[-1] == 2:
        return points_rotated[:, :2]
    else:
        return points_rotated


def ego_to_pixel(
    points_ego: np.ndarray,
    bev_range: float,
    bev_size: Tuple[int, int]
) -> np.ndarray:
    """
    将 ego 坐标转换到 BEV 像素坐标
    
    Args:
        points_ego: (N, 2) ego 坐标系下的点
        bev_range: BEV 范围 (meters)
        bev_size: (height, width) BEV 图像尺寸
    
    Returns:
        points_pixel: (N, 2) 像素坐标 (row, col)
    """
    height, width = bev_size
    
    # ego 坐标范围: [-bev_range, bev_range]
    # 像素坐标范围: [0, height/width]
    
    # x (前方) -> col
    # y (左侧) -> row
    
    # 归一化到 [0, 1]
    x_norm = (points_ego[:, 0] + bev_range) / (2 * bev_range)
    y_norm = (points_ego[:, 1] + bev_range) / (2 * bev_range)
    
    # 转换到像素坐标
    col = x_norm * width
    row = (1 - y_norm) * height  # y 轴翻转（图像坐标系）
    
    points_pixel = np.stack([row, col], axis=-1)
    
    return points_pixel


def pixel_to_ego(
    points_pixel: np.ndarray,
    bev_range: float,
    bev_size: Tuple[int, int]
) -> np.ndarray:
    """
    将 BEV 像素坐标转换到 ego 坐标
    
    Args:
        points_pixel: (N, 2) 像素坐标 (row, col)
        bev_range: BEV 范围 (meters)
        bev_size: (height, width) BEV 图像尺寸
    
    Returns:
        points_ego: (N, 2) ego 坐标系下的点
    """
    height, width = bev_size
    
    row = points_pixel[:, 0]
    col = points_pixel[:, 1]
    
    # 归一化到 [0, 1]
    x_norm = col / width
    y_norm = 1 - (row / height)  # y 轴翻转
    
    # 转换到 ego 坐标
    x = x_norm * (2 * bev_range) - bev_range
    y = y_norm * (2 * bev_range) - bev_range
    
    points_ego = np.stack([x, y], axis=-1)
    
    return points_ego


def filter_points_in_range(
    points: np.ndarray,
    max_distance: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    过滤范围内的点
    
    Args:
        points: (N, 2) 点坐标
        max_distance: 最大距离
    
    Returns:
        filtered_points: (M, 2) 过滤后的点
        mask: (N,) bool mask
    """
    distances = np.linalg.norm(points, axis=-1)
    mask = distances <= max_distance
    filtered_points = points[mask]
    
    return filtered_points, mask


def compute_trajectory_length(trajectory: np.ndarray) -> float:
    """
    计算轨迹长度
    
    Args:
        trajectory: (T, 2) 轨迹点
    
    Returns:
        length: 轨迹总长度
    """
    if len(trajectory) < 2:
        return 0.0
    
    diffs = np.diff(trajectory, axis=0)
    distances = np.linalg.norm(diffs, axis=-1)
    length = distances.sum()
    
    return length


def interpolate_trajectory(
    trajectory: np.ndarray,
    num_points: int
) -> np.ndarray:
    """
    插值轨迹到指定点数
    
    Args:
        trajectory: (T, 2) 原始轨迹
        num_points: 目标点数
    
    Returns:
        interpolated: (num_points, 2) 插值后的轨迹
    """
    if len(trajectory) == num_points:
        return trajectory
    
    # 使用线性插值
    t_old = np.linspace(0, 1, len(trajectory))
    t_new = np.linspace(0, 1, num_points)
    
    x_new = np.interp(t_new, t_old, trajectory[:, 0])
    y_new = np.interp(t_new, t_old, trajectory[:, 1])
    
    interpolated = np.stack([x_new, y_new], axis=-1)
    
    return interpolated


def get_rotation_matrix(angle: float) -> np.ndarray:
    """
    获取 2D 旋转矩阵
    
    Args:
        angle: 旋转角度 (radians)
    
    Returns:
        R: (2, 2) 旋转矩阵
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    R = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    return R


def rotate_points(points: np.ndarray, angle: float) -> np.ndarray:
    """
    旋转点
    
    Args:
        points: (N, 2) 点坐标
        angle: 旋转角度 (radians)
    
    Returns:
        rotated: (N, 2) 旋转后的点
    """
    R = get_rotation_matrix(angle)
    rotated = points @ R.T
    
    return rotated


def get_agent_velocity(
    trajectory: np.ndarray,
    dt: float = 0.5
) -> np.ndarray:
    """
    计算 agent 速度
    
    Args:
        trajectory: (T, 2) 轨迹点
        dt: 时间间隔 (seconds)
    
    Returns:
        velocity: (T-1, 2) 速度向量
    """
    if len(trajectory) < 2:
        return np.zeros((0, 2))
    
    diffs = np.diff(trajectory, axis=0)
    velocity = diffs / dt
    
    return velocity


def get_agent_heading(trajectory: np.ndarray) -> np.ndarray:
    """
    计算 agent 朝向角度
    
    Args:
        trajectory: (T, 2) 轨迹点
    
    Returns:
        heading: (T-1,) 朝向角度 (radians)
    """
    velocity = get_agent_velocity(trajectory)
    
    if len(velocity) == 0:
        return np.array([])
    
    heading = np.arctan2(velocity[:, 1], velocity[:, 0])
    
    return heading


def is_trajectory_valid(
    trajectory: np.ndarray,
    min_length: int = 2,
    min_movement: float = 1.0
) -> bool:
    """
    检查轨迹是否有效
    
    Args:
        trajectory: (T, 2) 轨迹点
        min_length: 最小点数
        min_movement: 最小移动距离
    
    Returns:
        valid: bool
    """
    # 检查长度
    if len(trajectory) < min_length:
        return False
    
    # 检查移动距离
    total_movement = compute_trajectory_length(trajectory)
    if total_movement < min_movement:
        return False
    
    # 检查 NaN
    if np.isnan(trajectory).any():
        return False
    
    return True


def normalize_angle(angle: float) -> float:
    """
    归一化角度到 [-pi, pi]
    
    Args:
        angle: 角度 (radians)
    
    Returns:
        normalized: 归一化后的角度
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    
    return angle


def compute_ade(pred_traj: np.ndarray, gt_traj: np.ndarray) -> float:
    """
    计算 Average Displacement Error
    
    Args:
        pred_traj: (T, 2) 预测轨迹
        gt_traj: (T, 2) 真实轨迹
    
    Returns:
        ade: float
    """
    distances = np.linalg.norm(pred_traj - gt_traj, axis=-1)
    ade = distances.mean()
    
    return ade


def compute_fde(pred_traj: np.ndarray, gt_traj: np.ndarray) -> float:
    """
    计算 Final Displacement Error
    
    Args:
        pred_traj: (T, 2) 预测轨迹
        gt_traj: (T, 2) 真实轨迹
    
    Returns:
        fde: float
    """
    fde = np.linalg.norm(pred_traj[-1] - gt_traj[-1])
    
    return fde


if __name__ == "__main__":
    # 测试工具函数
    print("Testing nuScenes utils...")
    
    # 测试坐标转换
    points_global = np.array([[10.0, 5.0], [20.0, 10.0]])
    ego_translation = np.array([0.0, 0.0, 0.0])
    ego_rotation = Quaternion(axis=[0, 0, 1], angle=np.pi/4)
    
    points_ego = global_to_ego(points_global, ego_translation, ego_rotation)
    print(f"Global to ego: {points_global} -> {points_ego}")
    
    # 测试像素转换
    bev_range = 50.0
    bev_size = (200, 200)
    points_pixel = ego_to_pixel(points_ego, bev_range, bev_size)
    print(f"Ego to pixel: {points_ego} -> {points_pixel}")
    
    # 测试轨迹长度
    trajectory = np.array([[0, 0], [1, 0], [2, 1], [3, 2]])
    length = compute_trajectory_length(trajectory)
    print(f"Trajectory length: {length:.2f}")
    
    # 测试轨迹有效性
    valid = is_trajectory_valid(trajectory, min_length=2, min_movement=1.0)
    print(f"Trajectory valid: {valid}")
    
    print("✅ All tests passed!")
