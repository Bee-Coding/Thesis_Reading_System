"""
nuScenes 数据预处理模块

这个模块负责从 nuScenes 数据集中提取和处理轨迹数据，包括：
1. 轨迹提取：从场景中提取 agent 的历史和未来轨迹
2. HD 地图栅格化：将高精度地图转换为 BEV 图像
3. Goal 词汇表构建：使用 K-means 聚类构建目标点词汇表

TODO: 你需要实现以下 3 个核心函数：
1. extract_agent_trajectories() - 轨迹提取 (难度: ⭐⭐⭐, 预计 2-3 小时)
2. rasterize_map() - 地图栅格化 (难度: ⭐⭐⭐⭐, 预计 3-4 小时)
3. build_vocabulary() - 词汇表构建 (难度: ⭐, 预计 1 小时)
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import os
import pickle

# nuScenes imports
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.map_expansion.map_api import NuScenesMap
    from pyquaternion import Quaternion
except ImportError:
    print("警告: nuscenes-devkit 未安装，请运行: pip install nuscenes-devkit")
    NuScenes = None
    NuScenesMap = None
    Quaternion = None

# 本地工具函数
from . import nuscenes_utils


# ==================== 核心函数 1: 轨迹提取 ====================

def extract_agent_trajectories(
    nusc: NuScenes,
    scene_token: str,
    history_frames: int = 4,
    future_frames: int = 12,
    min_movement: float = 1.0,
    max_distance: float = 50.0,
    vehicle_categories: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    从 nuScenes 场景中提取 agent 轨迹
    
    ⚠️ TODO: 你需要实现这个函数！
    
    实现步骤：
    ========
    1. 获取场景信息
       - 使用 nusc.get('scene', scene_token) 获取场景
       - 获取第一帧: scene['first_sample_token']
    
    2. 遍历场景中的所有帧
       - 使用 while 循环遍历: sample_token = sample['next']
       - 对每一帧，获取 ego 车位姿
    
    3. 对每个 agent 提取轨迹
       a. 获取该帧的所有标注: sample['anns']
       b. 对每个标注:
          - 检查类别是否是车辆
          - 提取历史轨迹（向前查找 history_frames 帧）
          - 提取未来轨迹（向后查找 future_frames 帧）
          - 转换到 ego 坐标系
    
    4. 过滤和验证
       - 检查轨迹长度是否完整
       - 检查移动距离是否 >= min_movement
       - 检查是否在 ego 车 max_distance 范围内
    
    代码提示：
    ========
    # 获取场景
    scene = nusc.get('scene', scene_token)
    sample_token = scene['first_sample_token']
    
    # 遍历所有帧
    while sample_token:
        sample = nusc.get('sample', sample_token)
        
        # 获取 ego 位姿
        ego_pose_token = sample['data']['LIDAR_TOP']  # 使用 LIDAR 作为参考
        sample_data = nusc.get('sample_data', ego_pose_token)
        ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
        
        # 遍历该帧的所有标注
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            
            # 检查类别
            if ann['category_name'] not in vehicle_categories:
                continue
            
            # 提取轨迹...
            # TODO: 实现轨迹提取逻辑
            
        sample_token = sample['next']
    
    坐标转换提示：
    ============
    # 全局坐标转 ego 坐标
    from . import nuscenes_utils
    
    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = Quaternion(ego_pose['rotation'])
    
    points_ego = nuscenes_utils.global_to_ego(
        points_global, 
        ego_translation, 
        ego_rotation
    )
    
    轨迹提取提示：
    ============
    # 向前查找历史轨迹
    history_traj = []
    current_ann = ann
    for i in range(history_frames):
        if current_ann['prev'] == '':
            break
        current_ann = nusc.get('sample_annotation', current_ann['prev'])
        position = current_ann['translation'][:2]  # 只取 x, y
        history_traj.append(position)
    
    # 向后查找未来轨迹
    future_traj = []
    current_ann = ann
    for i in range(future_frames):
        if current_ann['next'] == '':
            break
        current_ann = nusc.get('sample_annotation', current_ann['next'])
        position = current_ann['translation'][:2]
        future_traj.append(position)
    
    Args:
        nusc: NuScenes 对象
        scene_token: 场景 token
        history_frames: 历史帧数 (默认 4 帧 = 2 秒 @ 2Hz)
        future_frames: 未来帧数 (默认 12 帧 = 6 秒 @ 2Hz)
        min_movement: 最小移动距离（米），用于过滤静止物体
        max_distance: 最大距离（米），只考虑 ego 车附近的 agent
        vehicle_categories: 车辆类别列表，如果为 None 则使用默认列表
    
    Returns:
        trajectories: 字典，包含：
            'history': (N, history_frames, 2) - 历史轨迹
            'future': (N, future_frames, 2) - 未来轨迹
            'goals': (N, 2) - 目标点（未来轨迹的终点）
            'scene_tokens': (N,) - 场景 token 列表
            'sample_tokens': (N,) - 样本 token 列表
    
    示例：
        >>> nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes')
        >>> scene_token = nusc.scene[0]['token']
        >>> trajs = extract_agent_trajectories(nusc, scene_token)
        >>> print(f"提取了 {len(trajs['goals'])} 条轨迹")
    """
    # TODO: 在这里实现你的代码
    # 提示: 返回格式应该是一个字典，包含上述所有键
    
    # 默认车辆类别
    if vehicle_categories is None:
        vehicle_categories = [
            'vehicle.car',
            'vehicle.truck',
            'vehicle.bus',
            'vehicle.trailer',
            'vehicle.construction',
            'vehicle.emergency.ambulance',
            'vehicle.emergency.police'
        ]
    
    # 初始化结果列表
    all_history = []
    all_future = []
    all_goals = []
    all_scene_tokens = []
    all_sample_tokens = []
    
    # ========================================
    # 获取场景
    scene = nusc.get('scene', scene_token)
    sample_token = scene['first_sample_token']

    # 遍历所有帧
    while sample_token:
        sample = nusc.get('sample', sample_token)
        
        # 获取 ego 位姿
        ego_pose_token = sample['data']['LIDAR_TOP']  # 使用 LIDAR 作为参考
        sample_data = nusc.get('sample_data', ego_pose_token)
        ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
        
        ego_translation = np.array(ego_pose['translation'])
        ego_rotation = Quaternion(ego_pose['rotation'])

        # 遍历该帧的所有标注
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            
            # 检查类别
            if ann['category_name'] not in vehicle_categories:
                continue
            
            # 提取轨迹...
            # 向前查找历史轨迹
            history_traj = []
            current_ann = ann
            for i in range(history_frames):
                if current_ann['prev'] == '':
                    break
                current_ann = nusc.get('sample_annotation', current_ann['prev'])
                position = current_ann['translation'][:2]  # 只取 x, y
                history_traj.append(position)
            # 反转历史轨迹（因为是倒序）
            history_traj = history_traj[::-1]
            
            # 向后查找未来轨迹
            future_traj = []
            current_ann = ann
            for i in range(future_frames):
                if current_ann['next'] == '':
                    break
                current_ann = nusc.get('sample_annotation', current_ann['next'])
                position = current_ann['translation'][:2]
                future_traj.append(position)
            if len(history_traj) != history_frames or len(future_frames) != future_frames:
                continue

            # 转换为numpy数组
            history_traj = np.array(history_traj)
            future_traj = np.array(future_traj)

            # 转到ego系
            history_ego = nuscenes_utils.global_to_ego(history_traj, 
                                                       ego_translation,
                                                       ego_rotation)
            future_ego = nuscenes_utils.global_to_ego(future_traj, 
                                                       ego_translation,
                                                       ego_rotation)
            
            # 检查移动距离
            total_distance = nuscenes_utils.compute_trajectory_length(future_ego)
            if total_distance < min_movement:
                continue

            # 检查是否在范围内
            goal_distance = np.linalg.norm(future_ego[-1])
            if goal_distance > max_distance:
                continue

            all_history.append(history_traj)
            all_future.append(future_traj)
            # 以真实轨迹最后一个点作为目标点
            all_goals.append(future_traj[-1])
            all_scene_tokens.append(scene_token)
            all_sample_tokens.append(sample_token)
            
        sample_token = sample['next']
    # ========================================

    return {
        'history': np.array(all_history),      # (N, history_frames, 2)
        'future': np.array(all_future),        # (N, future_frames, 2)
        'goals': np.array(all_goals),          # (N, 2)
        'scene_tokens': all_scene_tokens,      # List[str]
        'sample_tokens': all_sample_tokens     # List[str]
    }


# ==================== 核心函数 2: HD 地图栅格化 ====================

def rasterize_map(
    nusc_map: NuScenesMap,
    ego_pose: Dict,
    map_size: Tuple[int, int] = (200, 200),
    map_range: float = 50.0,
    map_layers: Optional[List[str]] = None
) -> np.ndarray:
    """
    将 HD 地图栅格化为 BEV 图像
    
    ⚠️ TODO: 你需要实现这个函数！
    
    实现步骤：
    ========
    1. 创建空白 BEV 图像
       - 尺寸: (map_size[0], map_size[1], 3)
       - 3 个通道分别对应: [车道线, 道路边界, 人行道]
    
    2. 获取 ego 车周围的地图元素
       - 使用 nusc_map.get_records_in_radius() 获取附近的地图记录
       - 对每个图层（lane, road_segment, walkway）分别处理
    
    3. 将地图元素转换到 ego 坐标系
       - 地图元素是全局坐标
       - 使用 nuscenes_utils.global_to_ego() 转换
    
    4. 栅格化到 BEV 图像
       - 将 ego 坐标转换为像素坐标
       - 使用 cv2.polylines() 绘制线段
       - 使用 cv2.fillPoly() 填充区域
    
    代码提示：
    ========
    # 1. 创建空白图像
    bev_image = np.zeros((map_size[0], map_size[1], 3), dtype=np.uint8)
    
    # 2. 获取 ego 位置
    ego_x = ego_pose['translation'][0]
    ego_y = ego_pose['translation'][1]
    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = Quaternion(ego_pose['rotation'])
    
    # 3. 获取附近的车道线
    lane_records = nusc_map.get_records_in_radius(
        ego_x, ego_y, map_range, ['lane', 'lane_connector']
    )
    
    # 4. 处理每条车道线
    for record_token in lane_records['lane']:
        # 获取车道线的多边形
        lane_record = nusc_map.get('lane', record_token)
        
        # 获取车道线的边界点
        # 提示: 使用 nusc_map.get_arcline_path() 获取车道中心线
        
        # 转换到 ego 坐标系
        # points_ego = nuscenes_utils.global_to_ego(...)
        
        # 转换到像素坐标
        # points_pixel = nuscenes_utils.ego_to_pixel(...)
        
        # 绘制到图像
        # cv2.polylines(bev_image[:, :, 0], ...)
    
    # 5. 类似地处理道路边界和人行道
    
    # 6. 归一化到 [0, 1]
    bev_image = bev_image.astype(np.float32) / 255.0
    
    # 7. 转换为 (C, H, W) 格式
    bev_image = bev_image.transpose(2, 0, 1)
    
    地图 API 提示：
    =============
    # 获取车道线的几何信息
    lane_record = nusc_map.get('lane', lane_token)
    
    # 获取车道线的多边形（外边界）
    polygon_token = lane_record['polygon_token']
    polygon = nusc_map.extract_polygon(polygon_token)
    exterior_coords = np.array(polygon.exterior.coords)  # (N, 2)
    
    # 或者获取车道中心线
    arcline_path = nusc_map.get_arcline_path(lane_token)
    centerline_coords = np.array(arcline_path)  # (N, 3), 只取前两列
    
    绘制提示：
    ========
    # 绘制线段（车道线）
    points_pixel = points_pixel.astype(np.int32)
    cv2.polylines(
        bev_image[:, :, channel_idx],
        [points_pixel],
        isClosed=False,
        color=255,
        thickness=2
    )
    
    # 填充多边形（道路区域）
    cv2.fillPoly(
        bev_image[:, :, channel_idx],
        [points_pixel],
        color=255
    )
    
    Args:
        nusc_map: NuScenesMap 对象
        ego_pose: ego 车位姿字典，包含 'translation' 和 'rotation'
        map_size: BEV 图像尺寸 (height, width)
        map_range: BEV 范围（米），表示 [-map_range, map_range]
        map_layers: 地图层列表，默认 ['lane', 'road_segment', 'walkway']
    
    Returns:
        bev_image: (3, H, W) BEV 特征图
            - Channel 0: 车道线 (lane)
            - Channel 1: 道路边界 (road_segment)
            - Channel 2: 人行道 (walkway)
    
    示例：
        >>> nusc_map = NuScenesMap(dataroot='data/nuscenes', map_name='boston-seaport')
        >>> ego_pose = {'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0]}
        >>> bev = rasterize_map(nusc_map, ego_pose)
        >>> print(bev.shape)  # (3, 200, 200)
    """
    # TODO: 在这里实现你的代码
    
    # 默认地图层
    if map_layers is None:
        map_layers = ['lane', 'road_segment', 'walkway']
    
    # ========================================
    # 1. 创建空白图像
    bev_image = np.zeros((map_size[0], map_size[1], 3), dtype=np.uint8)
    
    # 2. 获取 ego 位置
    ego_x = ego_pose['translation'][0]
    ego_y = ego_pose['translation'][1]
    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = Quaternion(ego_pose['rotation'])
    
    # 3. 获取附近的车道线
    lane_records = nusc_map.get_records_in_radius(
        ego_x, ego_y, map_range, ['lane', 'lane_connector']
    )
    
    # 4. 处理每条车道线
    for record_token in lane_records['lane']:
        
        lane_record = nusc_map.get('lane', record_token)
        
        # 获取车道线的多边形
        polygon_token = lane_record['polygon_token']
        polygon = nusc_map.extract_polygon(polygon_token)
        exterior_coords = np.array(polygon.exterior.coords)[:, 2]   # 只取x,y
        
        # 转换到 ego 坐标系
        coords_ego = nuscenes_utils.global_to_ego(exterior_coords, ego_translation, ego_rotation)
        
        # 转换到像素坐标
        coords_pixel = nuscenes_utils.ego_to_pixel(coords_ego, map_range, map_size)
        
        # 绘制到图像
        coords_pixel = coords_pixel.astype(np.int32)
        cv2.polylines(bev_image[:, :, 0], [coords_pixel],
                      isClosed=True, color=255, thickness=2)
    
    # 4.1 处理道路边界（Channel 1）
    road_records = nusc_map.get_records_in_radius(
        ego_x, ego_y, map_range, ['road_segment']
    )

    for record_token in road_records['road_segment']:

        road_record = nusc_map.get('road_segment', record_token)

        polygon_token = road_record['polygon_token']
        polygon = nusc_map.extract_polygon(polygon_token)
        exterior_coords = np.array(polygon.exterior.coords)[:, 2]   # 只取x,y
        # 转换到 ego 坐标系
        coords_ego = nuscenes_utils.global_to_ego(exterior_coords, ego_translation, ego_rotation)
        
        # 转换到像素坐标
        coords_pixel = nuscenes_utils.ego_to_pixel(coords_ego, map_range, map_size)

        # 填充道路区域
        cv2.fillPoly(bev_image[:, :, 1], [coords_pixel], color=255)
    
    # 4.2 处理人行道（Channel 2）
    walkway_records = nusc_map.get_records_in_radius(
        ego_x, ego_y, map_range, ['walkway']
    )

    for record_token in walkway_records['walkway']:

        walkway_record = nusc_map.get('walkway', record_token)

        polygon_token = walkway_record['polygon_token']
        polygon = nusc_map.extract_polygon(polygon_token)
        exterior_coords = np.array(polygon.exterior.coords)[:, 2]   # 只取x,y
        # 转换到 ego 坐标系
        coords_ego = nuscenes_utils.global_to_ego(exterior_coords, ego_translation, ego_rotation)
        
        # 转换到像素坐标
        coords_pixel = nuscenes_utils.ego_to_pixel(coords_ego, map_range, map_size)

        # 填充道路区域
        cv2.fillPoly(bev_image[:, :, 2], [coords_pixel], color=255)
    
    # 5. 归一化到 [0, 1]
    bev_image = bev_image.astype(np.float32) / 255.0
    
    # 6. 转换为 (C, H, W) 格式
    bev_image = bev_image.transpose(2, 0, 1)
    # ========================================
    
    # 返回格式示例
    return bev_image  # (3, H, W)


# ==================== 核心函数 3: Goal 词汇表构建 ====================

def build_vocabulary(
    goal_points: np.ndarray,
    n_clusters: int = 256,
    seed: int = 42
) -> np.ndarray:
    """
    使用 K-means 构建 goal 词汇表
    
    ⚠️ TODO: 你需要实现这个函数！
    
    这是最简单的函数，建议先实现这个来热身。
    
    实现步骤：
    ========
    1. 导入 sklearn 的 KMeans
    2. 创建 KMeans 对象，设置 n_clusters 和 random_state
    3. 对 goal_points 进行聚类
    4. 返回聚类中心作为词汇表
    
    代码提示：
    ========
    from sklearn.cluster import KMeans
    
    # 创建 KMeans 对象
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=10,  # 运行 10 次，选择最佳结果
        max_iter=300
    )
    
    # 拟合数据
    kmeans.fit(goal_points)
    
    # 获取聚类中心
    vocabulary = kmeans.cluster_centers_
    
    # 返回词汇表
    return vocabulary
    
    Args:
        goal_points: (N, 2) 所有训练轨迹的终点
        n_clusters: 聚类数量（词汇表大小）
        seed: 随机种子，保证可复现
    
    Returns:
        vocabulary: (n_clusters, 2) 聚类中心，作为 goal 词汇表
    
    示例：
        >>> goals = np.random.randn(1000, 2) * 10  # 1000 个随机目标点
        >>> vocab = build_vocabulary(goals, n_clusters=256)
        >>> print(vocab.shape)  # (256, 2)
    """
    # TODO: 在这里实现你的代码
    
    # ========================================
    # TODO: 在这里实现 K-means 聚类
    # ========================================
    
    raise NotImplementedError(
        "请实现 build_vocabulary() 函数！\n"
        "这是最简单的函数，只需要几行代码。\n"
        "参考上面的代码提示。"
    )
    
    # 返回格式示例
    # return vocabulary  # (n_clusters, 2)


# ==================== 辅助函数（已实现） ====================

def process_scene(
    nusc: NuScenes,
    nusc_maps: Dict[str, NuScenesMap],
    scene_token: str,
    config
) -> Dict[str, np.ndarray]:
    """
    处理单个场景，提取轨迹和 BEV 特征
    
    这个函数已经实现，会调用你实现的核心函数。
    
    Args:
        nusc: NuScenes 对象
        nusc_maps: 地图对象字典 {map_name: NuScenesMap}
        scene_token: 场景 token
        config: 配置对象
    
    Returns:
        scene_data: 字典，包含该场景的所有数据
    """
    print(f"处理场景: {scene_token}")
    
    # 1. 提取轨迹
    trajectories = extract_agent_trajectories(
        nusc=nusc,
        scene_token=scene_token,
        history_frames=config.history_frames,
        future_frames=config.future_frames,
        min_movement=config.min_movement,
        max_distance=config.max_distance,
        vehicle_categories=config.vehicle_categories
    )
    
    n_trajectories = len(trajectories['goals'])
    print(f"  提取了 {n_trajectories} 条轨迹")
    
    if n_trajectories == 0:
        return None
    
    # 2. 为每条轨迹生成 BEV 特征
    bev_features = []
    
    scene = nusc.get('scene', scene_token)
    map_name = nusc.get('log', scene['log_token'])['location']
    nusc_map = nusc_maps.get(map_name)
    
    if nusc_map is None:
        print(f"  警告: 地图 {map_name} 不可用，跳过 BEV 特征生成")
        # 使用零特征
        bev_features = np.zeros((n_trajectories, 3, config.bev_height, config.bev_width))
    else:
        for i in tqdm(range(n_trajectories), desc="  生成 BEV 特征"):
            sample_token = trajectories['sample_tokens'][i]
            sample = nusc.get('sample', sample_token)
            
            # 获取 ego 位姿
            ego_pose_token = sample['data']['LIDAR_TOP']
            sample_data = nusc.get('sample_data', ego_pose_token)
            ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
            
            # 栅格化地图
            try:
                bev = rasterize_map(
                    nusc_map=nusc_map,
                    ego_pose=ego_pose,
                    map_size=(config.bev_height, config.bev_width),
                    map_range=config.bev_range,
                    map_layers=config.map_layers
                )
                bev_features.append(bev)
            except Exception as e:
                print(f"    警告: BEV 生成失败 ({e})，使用零特征")
                bev_features.append(np.zeros((3, config.bev_height, config.bev_width)))
        
        bev_features = np.array(bev_features)
    
    # 3. 组合数据
    scene_data = {
        'history': trajectories['history'],
        'future': trajectories['future'],
        'goals': trajectories['goals'],
        'bev_features': bev_features,
        'scene_token': scene_token,
        'sample_tokens': trajectories['sample_tokens']
    }
    
    return scene_data


def preprocess_nuscenes(
    dataroot: str,
    version: str,
    output_dir: str,
    config,
    scenes: Optional[List[str]] = None
):
    """
    预处理 nuScenes 数据集
    
    这个函数已经实现，会调用你实现的核心函数。
    
    Args:
        dataroot: nuScenes 数据根目录
        version: 数据集版本 ('v1.0-mini' 或 'v1.0-trainval')
        output_dir: 输出目录
        config: 配置对象
        scenes: 要处理的场景列表，如果为 None 则处理所有场景
    """
    print("=" * 60)
    print("nuScenes 数据预处理")
    print("=" * 60)
    
    # 1. 加载 nuScenes
    print(f"\n加载 nuScenes {version}...")
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    
    # 2. 加载地图
    print("\n加载地图...")
    available_maps = ['boston-seaport', 'singapore-onenorth', 
                      'singapore-hollandvillage', 'singapore-queenstown']
    nusc_maps = {}
    for map_name in available_maps:
        try:
            nusc_maps[map_name] = NuScenesMap(dataroot=dataroot, map_name=map_name)
            print(f"  ✓ {map_name}")
        except Exception as e:
            print(f"  ✗ {map_name} (不可用)")
    
    # 3. 确定要处理的场景
    if scenes is None:
        scene_tokens = [s['token'] for s in nusc.scene]
    else:
        scene_tokens = [s['token'] for s in nusc.scene if s['name'] in scenes]
    
    print(f"\n将处理 {len(scene_tokens)} 个场景")
    
    # 4. 处理每个场景
    all_data = []
    for scene_token in scene_tokens:
        scene_data = process_scene(nusc, nusc_maps, scene_token, config)
        if scene_data is not None:
            all_data.append(scene_data)
    
    # 5. 合并所有数据
    print("\n合并数据...")
    merged_data = {
        'history': np.concatenate([d['history'] for d in all_data], axis=0),
        'future': np.concatenate([d['future'] for d in all_data], axis=0),
        'goals': np.concatenate([d['goals'] for d in all_data], axis=0),
        'bev_features': np.concatenate([d['bev_features'] for d in all_data], axis=0),
    }
    
    print(f"  总轨迹数: {len(merged_data['goals'])}")
    print(f"  History shape: {merged_data['history'].shape}")
    print(f"  Future shape: {merged_data['future'].shape}")
    print(f"  Goals shape: {merged_data['goals'].shape}")
    print(f"  BEV shape: {merged_data['bev_features'].shape}")
    
    # 6. 构建词汇表
    print("\n构建 goal 词汇表...")
    vocabulary = build_vocabulary(
        merged_data['goals'],
        n_clusters=config.vocab_size,
        seed=config.vocab_seed
    )
    print(f"  词汇表大小: {vocabulary.shape}")
    
    # 7. 保存数据
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n保存数据...")
    np.save(os.path.join(output_dir, 'history.npy'), merged_data['history'])
    np.save(os.path.join(output_dir, 'future.npy'), merged_data['future'])
    np.save(os.path.join(output_dir, 'goals.npy'), merged_data['goals'])
    np.save(os.path.join(output_dir, 'bev_features.npy'), merged_data['bev_features'])
    np.save(os.path.join(output_dir, 'vocabulary.npy'), vocabulary)
    
    # 保存元数据
    metadata = {
        'n_samples': len(merged_data['goals']),
        'history_frames': config.history_frames,
        'future_frames': config.future_frames,
        'bev_size': (config.bev_height, config.bev_width),
        'bev_range': config.bev_range,
        'vocab_size': config.vocab_size,
        'scenes': scenes
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n✅ 预处理完成！数据已保存到: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    # 测试代码
    print("这是 nuScenes 预处理模块")
    print("\n你需要实现以下 3 个核心函数：")
    print("1. extract_agent_trajectories() - 轨迹提取")
    print("2. rasterize_map() - 地图栅格化")
    print("3. build_vocabulary() - 词汇表构建")
    print("\n请参考函数文档中的详细说明和代码提示。")
