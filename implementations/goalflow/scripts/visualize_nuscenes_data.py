"""
nuScenes 数据可视化脚本

可视化轨迹、地图和 BEV 特征，帮助理解数据

使用方法:
    python scripts/visualize_nuscenes_data.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import cv2

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion
from data import nuscenes_utils


def visualize_single_trajectory(nusc):
    """可视化单条轨迹"""
    print("=" * 70)
    print("可视化 1: 单条车辆轨迹")
    print("=" * 70)
    
    # 获取第一个场景的第一个车辆标注
    scene = nusc.scene[0]
    sample = nusc.get('sample', scene['first_sample_token'])
    
    # 找到一个车辆标注
    vehicle_ann = None
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        if ann['category_name'].startswith('vehicle.car'):
            vehicle_ann = ann
            break
    
    if not vehicle_ann:
        print("未找到车辆标注")
        return
    
    print(f"车辆类别: {vehicle_ann['category_name']}")
    
    # 提取完整轨迹
    trajectory = []
    
    # 向前追溯
    current = vehicle_ann
    history = []
    while current['prev']:
        current = nusc.get('sample_annotation', current['prev'])
        history.insert(0, current['translation'][:2])
    
    # 当前点
    trajectory = history + [vehicle_ann['translation'][:2]]
    
    # 向后追溯
    current = vehicle_ann
    future = []
    while current['next']:
        current = nusc.get('sample_annotation', current['next'])
        future.append(current['translation'][:2])
    
    trajectory = trajectory + future
    trajectory = np.array(trajectory)
    
    print(f"轨迹长度: {len(trajectory)} 帧")
    print(f"历史帧数: {len(history)}")
    print(f"未来帧数: {len(future)}")
    
    # 绘制
    plt.figure(figsize=(12, 10))
    
    # 绘制完整轨迹
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, alpha=0.5, label='Full Trajectory')
    
    # 标记历史部分
    if len(history) > 0:
        history_arr = np.array(history)
        plt.plot(history_arr[:, 0], history_arr[:, 1], 'go-', linewidth=3, markersize=8, label='History')
    
    # 标记当前点
    current_pos = vehicle_ann['translation'][:2]
    plt.plot(current_pos[0], current_pos[1], 'yo', markersize=15, label='Current', zorder=5)
    
    # 标记未来部分
    if len(future) > 0:
        future_arr = np.array(future)
        plt.plot(future_arr[:, 0], future_arr[:, 1], 'ro-', linewidth=3, markersize=8, label='Future')
        
        # 标记目标点（未来轨迹终点）
        goal = future_arr[-1]
        plt.plot(goal[0], goal[1], 'r*', markersize=20, label='Goal', zorder=5)
    
    # 标记起点
    plt.plot(trajectory[0, 0], trajectory[0, 1], 'bs', markersize=12, label='Start')
    
    plt.xlabel('X (meters)', fontsize=12)
    plt.ylabel('Y (meters)', fontsize=12)
    plt.title(f'Vehicle Trajectory - {vehicle_ann["category_name"]}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 保存
    output_path = 'trajectory_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 保存到: {output_path}")
    plt.close()


def visualize_ego_coordinate_transform(nusc):
    """可视化坐标转换"""
    print("\n" + "=" * 70)
    print("可视化 2: 全局坐标 vs Ego 坐标")
    print("=" * 70)
    
    # 获取数据
    scene = nusc.scene[0]
    sample = nusc.get('sample', scene['first_sample_token'])
    sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    
    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = Quaternion(ego_pose['rotation'])
    
    # 收集所有车辆位置
    global_positions = []
    ego_positions = []
    
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        if ann['category_name'].startswith('vehicle'):
            global_pos = np.array(ann['translation'][:2])
            global_positions.append(global_pos)
            
            # 转换到 ego 坐标
            ego_pos = nuscenes_utils.global_to_ego(
                global_pos.reshape(1, 2),
                ego_translation,
                ego_rotation
            )[0]
            ego_positions.append(ego_pos)
    
    global_positions = np.array(global_positions)
    ego_positions = np.array(ego_positions)
    
    print(f"找到 {len(global_positions)} 辆车")
    
    # 绘制
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 左图：全局坐标系
    ax1.scatter(global_positions[:, 0], global_positions[:, 1], 
                c='blue', s=100, alpha=0.6, label='Vehicles')
    ax1.plot(ego_translation[0], ego_translation[1], 
             'r^', markersize=15, label='Ego Vehicle')
    ax1.set_xlabel('X (meters)', fontsize=12)
    ax1.set_ylabel('Y (meters)', fontsize=12)
    ax1.set_title('Global Coordinate System', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 右图：Ego 坐标系
    ax2.scatter(ego_positions[:, 0], ego_positions[:, 1], 
                c='blue', s=100, alpha=0.6, label='Vehicles')
    ax2.plot(0, 0, 'r^', markersize=15, label='Ego Vehicle (Origin)')
    
    # 绘制坐标轴
    ax2.arrow(0, 0, 20, 0, head_width=2, head_length=2, fc='red', ec='red', linewidth=2)
    ax2.text(22, 0, 'X (Forward)', fontsize=10, color='red')
    ax2.arrow(0, 0, 0, 20, head_width=2, head_length=2, fc='green', ec='green', linewidth=2)
    ax2.text(0, 22, 'Y (Left)', fontsize=10, color='green')
    
    ax2.set_xlabel('X (meters) - Forward', fontsize=12)
    ax2.set_ylabel('Y (meters) - Left', fontsize=12)
    ax2.set_title('Ego Coordinate System', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    output_path = 'coordinate_transform.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 保存到: {output_path}")
    plt.close()
    
    print("\n💡 理解:")
    print("  - 左图: 所有物体在地图的全局坐标系中")
    print("  - 右图: 转换到以 ego 车为中心的坐标系")
    print("  - Ego 坐标系中，x 轴指向车辆前方，y 轴指向左侧")
    print("  - 预测模型使用 ego 坐标系进行训练")


def visualize_map_layers(nusc):
    """可视化地图层"""
    print("\n" + "=" * 70)
    print("可视化 3: HD 地图层")
    print("=" * 70)
    
    # 获取数据
    scene = nusc.scene[0]
    sample = nusc.get('sample', scene['first_sample_token'])
    log = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])
    map_name = log['location']
    
    print(f"地图: {map_name}")
    
    try:
        nusc_map = NuScenesMap(dataroot='data/nuscenes', map_name=map_name)
    except:
        print("❌ 地图加载失败，请先修复地图路径问题")
        return
    
    # 获取 ego 位置
    sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    ego_x = ego_pose['translation'][0]
    ego_y = ego_pose['translation'][1]
    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = Quaternion(ego_pose['rotation'])
    
    map_range = 50.0
    
    # 获取地图元素
    lane_records = nusc_map.get_records_in_radius(ego_x, ego_y, map_range, ['lane'])
    road_records = nusc_map.get_records_in_radius(ego_x, ego_y, map_range, ['road_segment'])
    walkway_records = nusc_map.get_records_in_radius(ego_x, ego_y, map_range, ['walkway'])
    
    print(f"\n找到的地图元素:")
    print(f"  - 车道线: {len(lane_records['lane'])} 条")
    print(f"  - 道路: {len(road_records['road_segment'])} 个")
    print(f"  - 人行道: {len(walkway_records['walkway'])} 个")
    
    # 绘制
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # 子图 1: 车道线
    ax = axes[0, 0]
    for record_token in lane_records['lane'][:20]:  # 只绘制前 20 条
        try:
            lane = nusc_map.get('lane', record_token)
        except KeyError:
            continue
        
        if 'polygon_token' in lane and lane['polygon_token']:
            polygon = nusc_map.extract_polygon(lane['polygon_token'])
            coords = np.array(polygon.exterior.coords)[:, :2]
            
            # 转换到 ego 坐标
            coords_ego = nuscenes_utils.global_to_ego(coords, ego_translation, ego_rotation)
            ax.plot(coords_ego[:, 0], coords_ego[:, 1], 'b-', linewidth=2, alpha=0.5)
    
    ax.plot(0, 0, 'r^', markersize=15, label='Ego Vehicle')
    ax.set_xlim(-map_range, map_range)
    ax.set_ylim(-map_range, map_range)
    ax.set_xlabel('X (meters)', fontsize=10)
    ax.set_ylabel('Y (meters)', fontsize=10)
    ax.set_title('Lane', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')
    
    # 子图 2: 道路
    ax = axes[0, 1]
    for record_token in road_records['road_segment'][:20]:
        try:
            road = nusc_map.get('road_segment', record_token)
        except KeyError:
            continue
        
        if 'polygon_token' in road and road['polygon_token']:
            polygon = nusc_map.extract_polygon(road['polygon_token'])
            coords = np.array(polygon.exterior.coords)[:, :2]
            
            coords_ego = nuscenes_utils.global_to_ego(coords, ego_translation, ego_rotation)
            ax.fill(coords_ego[:, 0], coords_ego[:, 1], 'gray', alpha=0.3, edgecolor='black')
    
    ax.plot(0, 0, 'r^', markersize=15, label='Ego Vehicle')
    ax.set_xlim(-map_range, map_range)
    ax.set_ylim(-map_range, map_range)
    ax.set_xlabel('X (meters)', fontsize=10)
    ax.set_ylabel('Y (meters)', fontsize=10)
    ax.set_title('Road Segment', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')
    
    # 子图 3: 人行道
    ax = axes[1, 0]
    for record_token in walkway_records['walkway'][:20]:
        try:
            walkway = nusc_map.get('walkway', record_token)
        except KeyError:
            continue
        
        if 'polygon_token' in walkway and walkway['polygon_token']:
            polygon = nusc_map.extract_polygon(walkway['polygon_token'])
            coords = np.array(polygon.exterior.coords)[:, :2]
            
            coords_ego = nuscenes_utils.global_to_ego(coords, ego_translation, ego_rotation)
            ax.fill(coords_ego[:, 0], coords_ego[:, 1], 'orange', alpha=0.3, edgecolor='darkorange')
    
    ax.plot(0, 0, 'r^', markersize=15, label='Ego Vehicle')
    ax.set_xlim(-map_range, map_range)
    ax.set_ylim(-map_range, map_range)
    ax.set_xlabel('X (meters)', fontsize=10)
    ax.set_ylabel('Y (meters)', fontsize=10)
    ax.set_title('Walkway', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')
    
    # 子图 4: 组合
    ax = axes[1, 1]
    
    # 道路
    for record_token in road_records['road_segment'][:20]:
        try:
            road = nusc_map.get('road_segment', record_token)
        except KeyError:
            continue
        
        if 'polygon_token' in road and road['polygon_token']:
            polygon = nusc_map.extract_polygon(road['polygon_token'])
            coords = np.array(polygon.exterior.coords)[:, :2]
            coords_ego = nuscenes_utils.global_to_ego(coords, ego_translation, ego_rotation)
            ax.fill(coords_ego[:, 0], coords_ego[:, 1], 'gray', alpha=0.3, edgecolor='black', linewidth=0.5)
    
    # 车道线
    for record_token in lane_records['lane'][:20]:
        try:
            lane = nusc_map.get('lane', record_token)
        except KeyError:
            continue
        
        if 'polygon_token' in lane and lane['polygon_token']:
            polygon = nusc_map.extract_polygon(lane['polygon_token'])
            coords = np.array(polygon.exterior.coords)[:, :2]
            coords_ego = nuscenes_utils.global_to_ego(coords, ego_translation, ego_rotation)
            ax.plot(coords_ego[:, 0], coords_ego[:, 1], 'b-', linewidth=2, alpha=0.7)
    
    # 人行道
    for record_token in walkway_records['walkway'][:20]:
        try:
            walkway = nusc_map.get('walkway', record_token)
        except KeyError:
            continue
        
        if 'polygon_token' in walkway and walkway['polygon_token']:
            polygon = nusc_map.extract_polygon(walkway['polygon_token'])
            coords = np.array(polygon.exterior.coords)[:, :2]
            coords_ego = nuscenes_utils.global_to_ego(coords, ego_translation, ego_rotation)
            ax.fill(coords_ego[:, 0], coords_ego[:, 1], 'orange', alpha=0.3, edgecolor='darkorange', linewidth=0.5)
    
    ax.plot(0, 0, 'r^', markersize=15, label='Ego Vehicle', zorder=10)
    ax.set_xlim(-map_range, map_range)
    ax.set_ylim(-map_range, map_range)
    ax.set_xlabel('X (meters)', fontsize=10)
    ax.set_ylabel('Y (meters)', fontsize=10)
    ax.set_title('Combined Map', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')
    
    plt.tight_layout()
    output_path = 'map_layers.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 保存到: {output_path}")
    plt.close()


def visualize_bev_rasterization(nusc):
    """可视化 BEV 栅格化"""
    print("\n" + "=" * 70)
    print("可视化 4: BEV 栅格化示例")
    print("=" * 70)
    
    # 获取数据
    scene = nusc.scene[0]
    sample = nusc.get('sample', scene['first_sample_token'])
    log = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])
    map_name = log['location']
    
    try:
        nusc_map = NuScenesMap(dataroot='data/nuscenes', map_name=map_name)
    except:
        print("❌ 地图加载失败")
        return
    
    # 获取 ego 位置
    sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = Quaternion(ego_pose['rotation'])
    
    # 创建 BEV 图像
    map_size = (200, 200)
    map_range = 50.0
    bev_image = np.zeros((map_size[0], map_size[1], 3), dtype=np.uint8)
    
    ego_x = ego_pose['translation'][0]
    ego_y = ego_pose['translation'][1]
    
    # 绘制车道线到 Channel 0
    lane_records = nusc_map.get_records_in_radius(ego_x, ego_y, map_range, ['lane'])
    for record_token in lane_records['lane']:
        try:
            lane = nusc_map.get('lane', record_token)
        except KeyError:
            continue
        
        if 'polygon_token' in lane and lane['polygon_token']:
            polygon = nusc_map.extract_polygon(lane['polygon_token'])
            coords = np.array(polygon.exterior.coords)[:, :2]
            
            # 转换到 ego 坐标
            coords_ego = nuscenes_utils.global_to_ego(coords, ego_translation, ego_rotation)
            
            # 转换到像素坐标
            coords_pixel = nuscenes_utils.ego_to_pixel(coords_ego, map_range, map_size)
            coords_pixel = coords_pixel.astype(np.int32)
            
            # 绘制（确保数组连续性）
            channel = np.ascontiguousarray(bev_image[:, :, 0])
            cv2.polylines(channel, [coords_pixel], isClosed=True, color=255, thickness=2)
            bev_image[:, :, 0] = channel
    
    # 绘制道路到 Channel 1
    road_records = nusc_map.get_records_in_radius(ego_x, ego_y, map_range, ['road_segment'])
    for record_token in road_records['road_segment']:
        try:
            road = nusc_map.get('road_segment', record_token)
        except KeyError:
            continue
        
        if 'polygon_token' in road and road['polygon_token']:
            polygon = nusc_map.extract_polygon(road['polygon_token'])
            coords = np.array(polygon.exterior.coords)[:, :2]
            coords_ego = nuscenes_utils.global_to_ego(coords, ego_translation, ego_rotation)
            coords_pixel = nuscenes_utils.ego_to_pixel(coords_ego, map_range, map_size)
            coords_pixel = coords_pixel.astype(np.int32)
            
            # 绘制（确保数组连续性）
            channel = np.ascontiguousarray(bev_image[:, :, 1])
            cv2.fillPoly(channel, [coords_pixel], color=255)
            bev_image[:, :, 1] = channel
    
    # 绘制人行道到 Channel 2
    walkway_records = nusc_map.get_records_in_radius(ego_x, ego_y, map_range, ['walkway'])
    for record_token in walkway_records['walkway']:
        try:
            walkway = nusc_map.get('walkway', record_token)
        except KeyError:
            continue
        
        if 'polygon_token' in walkway and walkway['polygon_token']:
            polygon = nusc_map.extract_polygon(walkway['polygon_token'])
            coords = np.array(polygon.exterior.coords)[:, :2]
            coords_ego = nuscenes_utils.global_to_ego(coords, ego_translation, ego_rotation)
            coords_pixel = nuscenes_utils.ego_to_pixel(coords_ego, map_range, map_size)
            coords_pixel = coords_pixel.astype(np.int32)
            
            # 绘制（确保数组连续性）
            channel = np.ascontiguousarray(bev_image[:, :, 2])
            cv2.fillPoly(channel, [coords_pixel], color=255)
            bev_image[:, :, 2] = channel
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Channel 0: 车道线
    axes[0, 0].imshow(bev_image[:, :, 0], cmap='gray', origin='lower')
    axes[0, 0].set_title('Channel 0: Lane', fontsize=12)
    axes[0, 0].axis('off')
    
    # Channel 1: 道路
    axes[0, 1].imshow(bev_image[:, :, 1], cmap='gray', origin='lower')
    axes[0, 1].set_title('Channel 1: Road', fontsize=12)
    axes[0, 1].axis('off')
    
    # Channel 2: 人行道
    axes[1, 0].imshow(bev_image[:, :, 2], cmap='gray', origin='lower')
    axes[1, 0].set_title('Channel 2: Walkway', fontsize=12)
    axes[1, 0].axis('off')
    
    # RGB 组合
    axes[1, 1].imshow(bev_image, origin='lower')
    axes[1, 1].set_title('RGB Combined', fontsize=12)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    output_path = 'bev_rasterization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 保存到: {output_path}")
    plt.close()
    
    print(f"\nBEV 图像形状: {bev_image.shape}")
    print(f"  - 尺寸: {map_size[0]} x {map_size[1]} 像素")
    print(f"  - 范围: {map_range} 米 (每个方向)")
    print(f"  - 分辨率: {2*map_range/map_size[0]:.2f} 米/像素")


def main():
    """主函数"""
    print("=" * 70)
    print("nuScenes 数据可视化")
    print("=" * 70)
    
    # 加载数据
    print("\n加载 nuScenes 数据集...")
    try:
        nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=False)
        print(f"✅ 成功加载 {len(nusc.scene)} 个场景")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return
    
    # 1. 可视化单条轨迹
    try:
        visualize_single_trajectory(nusc)
    except Exception as e:
        print(f"❌ 轨迹可视化失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. 可视化坐标转换
    try:
        visualize_ego_coordinate_transform(nusc)
    except Exception as e:
        print(f"❌ 坐标转换可视化失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. 可视化地图层
    try:
        visualize_map_layers(nusc)
    except Exception as e:
        print(f"❌ 地图可视化失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. 可视化 BEV 栅格化
    try:
        visualize_bev_rasterization(nusc)
    except Exception as e:
        print(f"❌ BEV 栅格化可视化失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ 可视化完成！")
    print("=" * 70)
    print("\n生成的图像:")
    print("  1. trajectory_visualization.png - 单条轨迹")
    print("  2. coordinate_transform.png - 坐标转换")
    print("  3. map_layers.png - 地图层")
    print("  4. bev_rasterization.png - BEV 栅格化")


if __name__ == "__main__":
    main()
