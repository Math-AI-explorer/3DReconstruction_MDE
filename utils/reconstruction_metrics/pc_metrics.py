import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors


def chamfer_distance_p(pcd1: np.array, pcd2: np.array, p: int=1, mode='kdtree') -> float:
    assert mode in ['kdtree', 'cdist']
    assert p in [1, 2]
    if pcd1.shape[1] != 3 or pcd2.shape[1] != 3:
        raise ValueError(
            f"""
            	Expected only coords in point cloud with dim size 3,
             	got {pcd1.shape[1]} and {pcd2.shape[2]}
             """
        )

    if mode == 'cdist':
        # pcd2_modified = pcd2[None, ...].repeat(pcd1.shape[0], axis=0) # 1xN2x3
        # pcd1_modified = pcd1[:, None, :].repeat(pcd2.shape[0], axis=1) # N1x1x3
        # dists = np.sum(np.abs(pcd1_modified - pcd2_modified), axis=-1)
        dists = cdist(pcd1, pcd2, 'minkowski', p=p)
        pcd1_chamfer = np.mean(np.min(dists, axis=1))
        pcd2_chamfer = np.mean(np.min(dists.T, axis=1))

    elif mode == 'kdtree':
        pcd1_nearest = pcd2[KDTree(pcd2).query(pcd1)[1], :]
        pcd2_nearest = pcd1[KDTree(pcd1).query(pcd2)[1], :]
        pcd1_dists = np.power(np.sum(np.power(np.abs(pcd1 - pcd1_nearest), p), axis=-1), 1/p)
        pcd2_dists = np.power(np.sum(np.power(np.abs(pcd2 - pcd2_nearest), p), axis=-1), 1/p)
        pcd1_chamfer = np.mean(pcd1_dists)
        pcd2_chamfer = np.mean(pcd2_dists)

    return pcd1_chamfer + pcd2_chamfer

def chamfer_distance_l2_o3d(pcd1: np.array, pcd2: np.array) -> float:
    if pcd1.shape[1] != 3 or pcd2.shape[1] != 3:
        raise ValueError(
            f"""
            	Expected only coords in point cloud with dim size 3,
             	got {pcd1.shape[1]} and {pcd2.shape[2]}
             """
        )

    pcd1_o3d = o3d.geometry.PointCloud()
    pcd1_o3d.points = o3d.utility.Vector3dVector(pcd1)

    pcd2_o3d = o3d.geometry.PointCloud()
    pcd2_o3d.points = o3d.utility.Vector3dVector(pcd2)

    pcd1_l2_dists = pcd1_o3d.compute_point_cloud_distance(pcd2_o3d)
    pcd2_l2_dists = pcd2_o3d.compute_point_cloud_distance(pcd1_o3d)

    return np.mean(pcd1_l2_dists) + np.mean(pcd2_l2_dists)

def f_score_pcd(source_points: np.ndarray, target_points: np.ndarray, threshold: float) -> float:
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(target_points)
    distances, _ = nbrs.kneighbors(source_points)
    
    TP = np.sum(distances < threshold)
    precision = TP / len(source_points)
    recall = TP / len(target_points)
    f_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return f_score

### MUST BE FIXED ###
# def f_score_pcd(pred_pcd: np.array, gt_pcd: np.array, p: int=1, threshold: float=0.05, mode='kdtree') -> float:
#     assert mode in ['kdtree', 'cdist']
#     assert p in [1, 2]
#     if pred_pcd.shape[1] != 3 or gt_pcd.shape[1] != 3:
#         raise ValueError(
#             f"""
#             	Expected only coords in point cloud with dim size 3,
#              	got {pred_pcd.shape[1]} and {gt_pcd.shape[2]}
#              """
#         )

#     if mode == 'cdist':
#         dists = cdist(pred_pcd, gt_pcd, 'minkowski', p=p)
#         pred_min_dists = np.min(dists, axis=1)
#         gt_min_dists = np.min(dists.T, axis=1)

#     elif mode == 'kdtree':
#         pred_pcd_nearest = gt_pcd[KDTree(gt_pcd).query(pred_pcd)[1], :]
#         pred_min_dists = np.power(np.sum(np.power(np.abs(pred_pcd - pred_pcd_nearest), p), axis=-1), 1/p)
#         gt_pcd_nearest = pred_pcd[KDTree(pred_pcd).query(gt_pcd)[1], :]
#         gt_min_dists = np.power(np.sum(np.power(np.abs(gt_pcd - gt_pcd_nearest), p), axis=-1), 1/p)

#     precision = np.sum(pred_min_dists < threshold) / pred_pcd.shape[0]
#     recall = np.sum(gt_min_dists < threshold) / gt_pcd.shape[0]
#     f_score_val = 2 * (precision * recall) / (precision + recall)

#     return f_score_val