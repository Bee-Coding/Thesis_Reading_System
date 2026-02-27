"""
nuScenes 数据集交互式探索脚本

这个脚本帮助你理解 nuScenes 数据集的结构，包括：
1. 场景 (Scene) 结构
2. 样本 (Sample) 结构
3. 标注 (Annotation) 结构
4. 轨迹链接
5. Ego 位姿
6. 地图结构

使用方法:
    python scripts/explore_nuscenes.py
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Add project path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure font
from utils.chinese_font import setup_font
setup_font()

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def explore_scene_structure(nusc):
    """探索场景结构"""
    print_section("1. 场景 (Scene) 结构")
    
    scene = nusc.scene[0]
    
    print(f"\n总场景数: {len(nusc.scene)}")
    print(f"\n第一个场景的完整信息:")
    print(json.dumps(scene, indent=2))
    
    print(f"\n关键字段解释:")
    print(f"  - token: 场景的唯一标识符")
    print(f"  - name: 场景名称 ({scene['name']})")
    print(f"  - description: 场景描述")
    print(f"  - nbr_samples: 该场景包含的样本数 ({scene['nbr_samples']})")
    print(f"  - first_sample_token: 第一帧的 token")
    print(f"  - last_sample_token: 最后一帧的 token")
    
    return scene


def explore_sample_structure(nusc, scene):
    """探索样本结构"""
    print_section("2. 样本 (Sample) 结构")
    
    sample = nusc.get('sample', scene['first_sample_token'])
    
    print(f"\n样本的完整信息:")
    print(json.dumps(sample, indent=2))
    
    print(f"\n关键字段解释:")
    print(f"  - token: 样本的唯一标识符")
    print(f"  - timestamp: 时间戳")
    print(f"  - scene_token: 所属场景的 token")
    print(f"  - data: 传感器数据的 token 字典")
    print(f"    - LIDAR_TOP: 激光雷达数据")
    print(f"    - CAM_FRONT: 前置摄像头数据")
    print(f"    - 等等...")
    print(f"  - anns: 该帧所有标注的 token 列表 (共 {len(sample['anns'])} 个)")
    print(f"  - prev: 前一帧的 token")
    print(f"  - next: 后一帧的 token")
    
    return sample


def explore_annotation_structure(nusc, sample):
    """探索标注结构"""
    print_section("3. 标注 (Annotation) 结构")
    
    if not sample['anns']:
        print("该样本没有标注")
        return None
    
    ann_token = sample['anns'][0]
    ann = nusc.get('sample_annotation', ann_token)
    
    print(f"\n第一个标注的完整信息:")
    print(json.dumps(ann, indent=2))
    
    print(f"\n关键字段解释:")
    print(f"  - token: 标注的唯一标识符")
    print(f"  - sample_token: 所属样本的 token")
    print(f"  - instance_token: 物体实例的 token (用于跨帧追踪)")
    print(f"  - category_name: 物体类别 ({ann['category_name']})")
    print(f"  - translation: 3D 位置 [x, y, z] = {ann['translation']}")
    print(f"  - size: 3D 尺寸 [width, length, height] = {ann['size']}")
    print(f"  - rotation: 四元数旋转 = {ann['rotation']}")
    print(f"  - prev: 前一帧该物体的标注 token")
    print(f"  - next: 后一帧该物体的标注 token")
    
    return ann


def explore_trajectory_chain(nusc, ann):
    """探索轨迹链接"""
    print_section("4. 轨迹链接 - 理解如何提取轨迹")
    
    print(f"\n当前标注信息:")
    print(f"  - 类别: {ann['category_name']}")
    print(f"  - 位置: {ann['translation']}")
    print(f"  - 实例 ID: {ann['instance_token'][:8]}...")
    
    # 向前追溯历史
    print(f"\n向前追溯历史轨迹:")
    history_positions = []
    current = ann
    count = 0
    while current['prev'] and count < 5:  # 只追溯 5 帧作为示例
        current = nusc.get('sample_annotation', current['prev'])
        history_positions.append(current['translation'][:2])
        print(f"  第 {count+1} 帧前: 位置 = {current['translation'][:2]}")
        count += 1
    
    if not current['prev']:
        print(f"  (已到达轨迹起点)")
    
    # 向后追溯未来
    print(f"\n向后追溯未来轨迹:")
    future_positions = []
    current = ann
    count = 0
    while current['next'] and count < 5:  # 只追溯 5 帧作为示例
        current = nusc.get('sample_annotation', current['next'])
        future_positions.append(current['translation'][:2])
        print(f"  第 {count+1} 帧后: 位置 = {current['translation'][:2]}")
        count += 1
    
    if not current['next']:
        print(f"  (已到达轨迹终点)")
    
    print(f"\n💡 关键理解:")
    print(f"  - 使用 ann['prev'] 可以向前追溯历史轨迹")
    print(f"  - 使用 ann['next'] 可以向后追溯未来轨迹")
    print(f"  - 当 prev 或 next 为空字符串时，表示到达轨迹端点")
    print(f"  - translation[:2] 提取 [x, y] 坐标（忽略 z）")
    
    return history_positions, future_positions


def explore_ego_pose(nusc, sample):
    """探索 ego 位姿"""
    print_section("5. Ego 车位姿 - 理解坐标转换")
    
    # 获取 ego 位姿
    sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    
    print(f"\nEgo 位姿完整信息:")
    print(json.dumps(ego_pose, indent=2))
    
    print(f"\n关键字段解释:")
    print(f"  - translation: Ego 车在全局坐标系中的位置")
    print(f"    [x, y, z] = {ego_pose['translation']}")
    print(f"  - rotation: Ego 车的旋转（四元数）")
    print(f"    [w, x, y, z] = {ego_pose['rotation']}")
    
    print(f"\n💡 坐标系理解:")
    print(f"  - 全局坐标系: 整个地图的固定坐标系")
    print(f"  - Ego 坐标系: 以 ego 车为原点的坐标系")
    print(f"    - x 轴: 车辆前方")
    print(f"    - y 轴: 车辆左侧")
    print(f"    - z 轴: 车辆上方")
    print(f"  - 需要将全局坐标转换为 ego 坐标进行预测")
    
    return ego_pose


def explore_map_structure(nusc, sample):
    """探索地图结构"""
    print_section("6. 地图结构 - 理解 HD 地图")
    
    # 获取场景对应的地图
    scene_token = sample['scene_token']
    scene = nusc.get('scene', scene_token)
    log = nusc.get('log', scene['log_token'])
    map_name = log['location']
    
    print(f"\n当前场景的地图: {map_name}")
    
    try:
        nusc_map = NuScenesMap(dataroot='data/nuscenes', map_name=map_name)
        
        # 获取 ego 位置
        sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
        ego_x = ego_pose['translation'][0]
        ego_y = ego_pose['translation'][1]
        
        print(f"\nEgo 车位置: ({ego_x:.2f}, {ego_y:.2f})")
        
        # 获取附近的地图元素
        print(f"\n获取 ego 车周围 50m 的地图元素:")
        
        # 车道线
        lane_records = nusc_map.get_records_in_radius(
            ego_x, ego_y, 50, ['lane', 'lane_connector']
        )
        print(f"  - 车道线 (lane): {len(lane_records['lane'])} 条")
        print(f"  - 车道连接 (lane_connector): {len(lane_records['lane_connector'])} 条")
        
        # 道路
        road_records = nusc_map.get_records_in_radius(
            ego_x, ego_y, 50, ['road_segment']
        )
        print(f"  - 道路段 (road_segment): {len(road_records['road_segment'])} 个")
        
        # 人行道
        walkway_records = nusc_map.get_records_in_radius(
            ego_x, ego_y, 50, ['walkway']
        )
        print(f"  - 人行道 (walkway): {len(walkway_records['walkway'])} 个")
        
        # 查看一条车道线的详细信息
        if lane_records['lane']:
            lane_token = lane_records['lane'][0]
            lane = nusc_map.get('lane', lane_token)
            
            print(f"\n第一条车道线的详细信息:")
            print(json.dumps(lane, indent=2))
            
            print(f"\n关键字段解释:")
            print(f"  - token: 车道线的唯一标识符")
            print(f"  - polygon_token: 车道线多边形的 token")
            print(f"  - from_edge_line_token: 起始边界线")
            print(f"  - to_edge_line_token: 结束边界线")
            
            # 获取车道线的几何形状
            polygon = nusc_map.extract_polygon(lane['polygon_token'])
            coords = np.array(polygon.exterior.coords)
            
            print(f"\n车道线几何信息:")
            print(f"  - 多边形顶点数: {len(coords)}")
            print(f"  - 前 3 个顶点: {coords[:3]}")
            
            print(f"\n💡 地图栅格化理解:")
            print(f"  1. 使用 get_records_in_radius() 获取附近的地图元素")
            print(f"  2. 使用 extract_polygon() 获取几何形状")
            print(f"  3. 将全局坐标转换为 ego 坐标")
            print(f"  4. 将 ego 坐标转换为像素坐标")
            print(f"  5. 使用 cv2.polylines() 或 cv2.fillPoly() 绘制")
        
        return nusc_map, ego_pose
        
    except Exception as e:
        print(f"\n❌ 地图加载失败: {e}")
        print(f"请确保地图文件在正确位置: data/nuscenes/maps/expansion/")
        return None, None


def visualize_trajectory(nusc, ann):
    """可视化一条完整轨迹"""
    print_section("7. 可视化轨迹")
    
    print(f"\n正在提取完整轨迹...")
    
    # 提取完整轨迹
    trajectory = []
    
    # 向前追溯到起点
    current = ann
    while current['prev']:
        current = nusc.get('sample_annotation', current['prev'])
    
    # 从起点向后遍历
    start_ann = current
    while True:
        trajectory.append(current['translation'][:2])
        if not current['next']:
            break
        current = nusc.get('sample_annotation', current['next'])
    
    trajectory = np.array(trajectory)
    
    print(f"  - 轨迹总长度: {len(trajectory)} 帧")
    print(f"  - 起点: {trajectory[0]}")
    print(f"  - 终点: {trajectory[-1]}")
    print(f"  - 总移动距离: {np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)):.2f} 米")
    
    # 绘制轨迹
    plt.figure(figsize=(12, 10))
    
    # 绘制轨迹
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Trajectory')
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b.', markersize=8)
    
    # 标记起点和终点
    plt.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=15, label='Start', zorder=5)
    plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=15, label='End', zorder=5)
    
    # 标记当前帧
    current_idx = 0
    for i, pos in enumerate(trajectory):
        if np.allclose(pos, ann['translation'][:2]):
            current_idx = i
            break
    
    plt.plot(trajectory[current_idx, 0], trajectory[current_idx, 1], 
             'yo', markersize=15, label='Current Frame', zorder=5)
    
    # 添加箭头显示方向
    for i in range(0, len(trajectory)-1, max(1, len(trajectory)//10)):
        dx = trajectory[i+1, 0] - trajectory[i, 0]
        dy = trajectory[i+1, 1] - trajectory[i, 1]
        plt.arrow(trajectory[i, 0], trajectory[i, 1], dx, dy,
                 head_width=2, head_length=2, fc='blue', ec='blue', alpha=0.5)
    
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlabel('X (meters)', fontsize=12)
    plt.ylabel('Y (meters)', fontsize=12)
    plt.title(f'Agent Trajectory - {ann["category_name"]}', fontsize=14)
    
    output_path = 'trajectory_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 轨迹可视化已保存到: {output_path}")
    
    return trajectory


def visualize_bev_map(nusc_map, ego_pose):
    """可视化 BEV 地图"""
    print_section("8. 可视化 BEV 地图")
    
    if nusc_map is None or ego_pose is None:
        print("地图不可用，跳过可视化")
        return
    
    print(f"\n正在生成 BEV 地图...")
    
    ego_x = ego_pose['translation'][0]
    ego_y = ego_pose['translation'][1]
    map_range = 50.0
    
    # 创建图像
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 获取地图元素
    lane_records = nusc_map.get_records_in_radius(
        ego_x, ego_y, map_range, ['lane', 'lane_connector']
    )
    road_records = nusc_map.get_records_in_radius(
        ego_x, ego_y, map_range, ['road_segment']
    )
    walkway_records = nusc_map.get_records_in_radius(
        ego_x, ego_y, map_range, ['walkway']
    )
    
    # 绘制车道线
    ax = axes[0]
    for record_token in lane_records['lane'] + lane_records['lane_connector']:
        try:
            lane = nusc_map.get('lane', record_token)
        except KeyError:
            # 跳过不在当前地图中的记录
            continue
        
        polygon = nusc_map.extract_polygon(lane['polygon_token'])
        coords = np.array(polygon.exterior.coords)
        ax.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=1)
        ax.fill(coords[:, 0], coords[:, 1], 'blue', alpha=0.3)
    
    ax.plot(ego_x, ego_y, 'ro', markersize=10, label='Ego')
    ax.set_xlim(ego_x - map_range, ego_x + map_range)
    ax.set_ylim(ego_y - map_range, ego_y + map_range)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Lane', fontsize=12)
    ax.legend()
    
    # 绘制道路
    ax = axes[1]
    for record_token in road_records['road_segment']:
        try:
            road = nusc_map.get('road_segment', record_token)
        except KeyError:
            # 跳过不在当前地图中的记录
            continue
        
        polygon = nusc_map.extract_polygon(road['polygon_token'])
        coords = np.array(polygon.exterior.coords)
        ax.fill(coords[:, 0], coords[:, 1], 'gray', alpha=0.5)
    
    ax.plot(ego_x, ego_y, 'ro', markersize=10, label='Ego')
    ax.set_xlim(ego_x - map_range, ego_x + map_range)
    ax.set_ylim(ego_y - map_range, ego_y + map_range)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Road Segment', fontsize=12)
    ax.legend()
    
    # 绘制人行道
    ax = axes[2]
    for record_token in walkway_records['walkway']:
        try:
            walkway = nusc_map.get('walkway', record_token)
        except KeyError:
            # 跳过不在当前地图中的记录
            continue
        
        polygon = nusc_map.extract_polygon(walkway['polygon_token'])
        coords = np.array(polygon.exterior.coords)
        ax.fill(coords[:, 0], coords[:, 1], 'green', alpha=0.5)
    
    ax.plot(ego_x, ego_y, 'ro', markersize=10, label='Ego')
    ax.set_xlim(ego_x - map_range, ego_x + map_range)
    ax.set_ylim(ego_y - map_range, ego_y + map_range)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Walkway', fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    output_path = 'bev_map_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ BEV 地图可视化已保存到: {output_path}")


def print_summary():
    """打印总结"""
    print_section("总结 - 关键概念")
    
    print("""
📚 nuScenes 数据结构层次:

Scene (场景)
  └── Sample (样本/帧)
        ├── data: 传感器数据 (LIDAR, Camera, etc.)
        └── anns: 标注列表
              └── Annotation (标注)
                    ├── translation: 3D 位置
                    ├── category_name: 物体类别
                    ├── prev: 前一帧的标注 (用于追踪)
                    └── next: 后一帧的标注 (用于追踪)

🔑 关键操作:

1. 提取轨迹:
   - 使用 ann['prev'] 向前追溯历史
   - 使用 ann['next'] 向后追溯未来
   - 注意: 历史轨迹是倒序的，需要反转

2. 坐标转换:
   - 全局坐标 → Ego 坐标: global_to_ego()
   - Ego 坐标 → 像素坐标: ego_to_pixel()

3. 地图栅格化:
   - get_records_in_radius(): 获取附近地图元素
   - extract_polygon(): 获取几何形状
   - cv2.polylines() / cv2.fillPoly(): 绘制到图像

💡 实现建议:

1. 先实现 build_vocabulary() - 最简单
2. 再实现 extract_agent_trajectories() - 核心功能
3. 最后实现 rasterize_map() - 最复杂

📖 参考资源:

- nuScenes 官网: https://www.nuscenes.org/
- 官方教程: https://www.nuscenes.org/nuscenes#tutorials
- API 文档: https://github.com/nutonomy/nuscenes-devkit
""")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("  nuScenes 数据集交互式探索")
    print("=" * 70)
    
    # 加载数据
    print("\n正在加载 nuScenes 数据集...")
    try:
        nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=False)
        print(f"✅ 成功加载 {len(nusc.scene)} 个场景")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print("请确保数据在正确位置: data/nuscenes/")
        return
    
    # 1. 探索场景结构
    scene = explore_scene_structure(nusc)
    
    # 2. 探索样本结构
    sample = explore_sample_structure(nusc, scene)
    
    # 3. 探索标注结构
    ann = explore_annotation_structure(nusc, sample)
    
    if ann is None:
        print("没有可用的标注，退出")
        return
    
    # 4. 探索轨迹链接
    history, future = explore_trajectory_chain(nusc, ann)
    
    # 5. 探索 ego 位姿
    ego_pose = explore_ego_pose(nusc, sample)
    
    # 6. 探索地图结构
    nusc_map, ego_pose = explore_map_structure(nusc, sample)
    
    # 7. 可视化轨迹
    trajectory = visualize_trajectory(nusc, ann)
    
    # 8. 可视化 BEV 地图
    if nusc_map is not None:
        visualize_bev_map(nusc_map, ego_pose)
    
    # 打印总结
    print_summary()
    
    print("\n" + "=" * 70)
    print("  探索完成！")
    print("=" * 70)
    print("\n生成的文件:")
    print("  - trajectory_visualization.png - 轨迹可视化")
    if nusc_map is not None:
        print("  - bev_map_visualization.png - BEV 地图可视化")
    print("\n现在你可以开始实现 nuscenes_preprocessor.py 中的函数了！")


if __name__ == "__main__":
    main()
