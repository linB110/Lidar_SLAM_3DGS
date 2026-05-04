import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from utils.ICP import ICP
from utils.submap import SubmapManager
from utils.keyframe import KeyframeManager
from utils.posegraph import PoseGraph
from utils.loopclosure import LoopClosure
from utils.io import load_folder, load_pcd
from utils.viz import Visualizer

def distribute_pose_corrections(poses, kf_indices, kf_opt_poses):
    """
    Interpolate pose graph corrections to all non-keyframe poses using SLERP.
    """
    new_poses = [p.copy() for p in poses]
    
    for k in range(len(kf_indices) - 1):
        idx_A = kf_indices[k]
        idx_B = kf_indices[k+1]
        
        orig_A, orig_B = poses[idx_A], poses[idx_B]
        opt_A, opt_B = kf_opt_poses[k], kf_opt_poses[k+1]
        
        # Calculate correction transforms
        corr_A = opt_A @ np.linalg.inv(orig_A)
        corr_B = opt_B @ np.linalg.inv(orig_B)
        
        rot_A = R.from_matrix(corr_A[:3, :3])
        rot_B = R.from_matrix(corr_B[:3, :3])
        trans_A = corr_A[:3, 3]
        trans_B = corr_B[:3, 3]
        
        # Setup SLERP
        key_rots = R.from_quat(np.vstack([rot_A.as_quat(), rot_B.as_quat()]))
        try:
            slerp = Slerp([0, 1], key_rots)
        except Exception: # Fallback if SLERP fails (e.g., rotations too similar)
            slerp = None 
            
        # Apply interpolated correction to intermediate frames
        for j in range(idx_A, idx_B):
            t = (j - idx_A) / float(idx_B - idx_A)
            
            rot_t = slerp(t).as_matrix() if slerp else rot_A.as_matrix() # Use rot_A if slerp fails
            trans_t = trans_A + t * (trans_B - trans_A)
            
            corr_j = np.eye(4)
            corr_j[:3, :3] = rot_t
            corr_j[:3, 3] = trans_t
            
            new_poses[j] = corr_j @ poses[j]
            
    # Apply the last correction to any trailing frames
    if len(kf_indices) > 0:
        last_kf_idx = kf_indices[-1]
        last_corr = kf_opt_poses[-1] @ np.linalg.inv(poses[last_kf_idx])
        for j in range(last_kf_idx, len(poses)):
            new_poses[j] = last_corr @ poses[j]
            
    return new_poses


# ==========================================
# Main SLAM Pipeline
# ==========================================
'''
Path setting
'''

target = "/home/lab605/SLAM_3DGS/Track2"
folder = f"{target}/data/raw_pcd/"
files = load_folder(folder)

# Initialization
icp = ICP(max_correspondence_distance=1.0) 
submap_mgr = SubmapManager()
kf_manager = KeyframeManager()
pg = PoseGraph()
lc = LoopClosure(icp) 
viz = Visualizer()

pg.add_prior() 

prev_pose = np.eye(4)
prev_prev_pose = None
poses = []
timestamps = []

# Visualization & Interpolation state
full_map = np.zeros((0, 3))
kf_indices = []
kf_poses_list = []
last_kf_pose = np.eye(4)
current_kf_idx = 0 

# Filtering parameters
MIN_RANGE = 2.0 
MAX_RANGE = 50.0 
SOR_NB_NEIGHBORS = 20 
SOR_STD_RATIO = 2.0 

# main loop
for i, f in enumerate(files):
    print(f"\n--- Processing Frame {i} ---")
    pts_raw = load_pcd(f)
    if pts_raw.shape[1] > 3:
        pts_raw = pts_raw[:, :3]

    # --- preprocess ---
    # 1. remove NaN and Inf points
    pts_cleaned = pts_raw[np.isfinite(pts_raw).all(axis=1)]

    # 2. range filter
    distances = np.linalg.norm(pts_cleaned, axis=1)
    pts_filtered_range = pts_cleaned[(distances >= MIN_RANGE) & (distances <= MAX_RANGE)]

    # 3. SOR filter
    if pts_filtered_range.shape[0] < SOR_NB_NEIGHBORS * 2: 
        print(f"Warning: Frame {i} has too few points ({pts_filtered_range.shape[0]}) after range filter for SOR. Skipping SOR.")
        pts = pts_filtered_range
    else:
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pts_filtered_range)
        
        cl, ind = pcd_o3d.remove_statistical_outlier(nb_neighbors=SOR_NB_NEIGHBORS, std_ratio=SOR_STD_RATIO)
        pts = np.asarray(cl.points)

    if pts.shape[0] < 100: 
        print(f"[Frame {i}] Point cloud has too few valid points ({pts.shape[0]}) after filtering. Skipping ICP and using previous pose.")
        poses.append(prev_pose.copy()) 
        timestamps.append(f.split("/")[-1].replace(".pcd", ""))
        continue

    ts = f.split("/")[-1].replace(".pcd", "")

    # Initial frame
    if i == 0:
        curr_pose = np.eye(4) 
        poses.append(curr_pose.copy())
        timestamps.append(ts)
        
        submap_mgr.add_keyframe(pts, curr_pose)
        lc.add_keyframe(pts, curr_pose) 
        
        kf_indices.append(i)
        kf_poses_list.append(curr_pose.copy())
        prev_pose = curr_pose.copy() 
        continue

    # ===== Scan-to-Submap (Odometry) =====
    submap_pts = submap_mgr.get_latest_submap()
    if submap_pts is None or submap_pts.shape[0] < 100: 
        print(f"[Frame {i}] Submap is empty or has too few points ({submap_pts.shape[0] if submap_pts is not None else 0}). Skipping ICP and using previous pose.")
        poses.append(prev_pose.copy())
        timestamps.append(ts)
        continue
   
    # 1. Prediction (constant velocity model)
    if prev_prev_pose is None:
        # first prediction fallback
        pred_pose = prev_pose.copy()
    else:
        # relative motion between last two frames
        delta_T = np.linalg.inv(prev_prev_pose) @ prev_pose

        # constant velocity prediction
        pred_pose = prev_pose @ delta_T
   
    # 2. ICP refinement
    T_correction, score = icp.align(pts, submap_pts, init_T=pred_pose)
   
    print(f"[Frame {i}] ICP score={score}, "
              f"translation={T_correction[:3,3].round(3)}, "
              f"det_R={np.linalg.det(T_correction[:3,:3]):.4f}")

    # 3. update pose
    if score < 1000: # inliers < 1000 => fallback
        curr_pose = prev_pose.copy()
        print(f"[Frame {i}] ICP failed or low score ({score}) → using prev_pose as fallback.")
    else:
        curr_pose = T_correction
        print(f"[Frame {i}] Pose updated: current_pose_translation={curr_pose[:3,3].round(3)}")
    
    poses.append(curr_pose.copy())
    timestamps.append(ts)

    # ===== Keyframe Processing =====
    if kf_manager.is_keyframe(curr_pose):
        current_kf_idx += 1
        
        # 1. Add Node to Managers
        submap_mgr.add_keyframe(pts, curr_pose)
        lc.add_keyframe(pts, curr_pose)

        # 2. Add Odom Edge between Keyframes
        rel_pose = np.linalg.inv(last_kf_pose) @ curr_pose # 從 last_kf_pose 到 curr_pose 的相對變換
        pg.add_node(current_kf_idx, curr_pose)
        pg.add_odom(current_kf_idx - 1, current_kf_idx, rel_pose)

        last_kf_pose = curr_pose.copy() 
        kf_indices.append(i) 
        kf_poses_list.append(curr_pose.copy()) 
        
        # 3. Update Visual Map (Voxel downsampled)
        pts_w_for_map = (curr_pose[:3, :3] @ pts.T).T + curr_pose[:3, 3]
        if full_map.shape[0] == 0:
            full_map = pts_w_for_map
        else:
            full_map = np.concatenate([full_map, pts_w_for_map], axis=0)

        if full_map.shape[0] > 100000: 
            pcd_tmp = o3d.geometry.PointCloud()
            pcd_tmp.points = o3d.utility.Vector3dVector(full_map)
            pcd_tmp = pcd_tmp.voxel_down_sample(voxel_size=0.5) 
            full_map = np.asarray(pcd_tmp.points)

        # ===== Loop Closure =====
        loop_result = lc.detect(submap_mgr)
        if loop_result is not None:
            loop_idx, T_loop, score_lc = loop_result 
            print(f"Loop detected from kf {current_kf_idx} to kf {loop_idx} with score {score_lc}")
            pg.add_loop(loop_idx, current_kf_idx, T_loop)

            # ===== Optimize & Interpolate =====
            print("Optimizing pose graph...")
            kf_opt_dict = pg.optimize()
            
            kf_opt_poses = [kf_opt_dict[k] for k in sorted(kf_opt_dict.keys())]
            
            submap_mgr.update_poses(kf_opt_poses)
            lc.update_poses(kf_opt_poses)
            
            poses = distribute_pose_corrections(poses, kf_indices, kf_opt_poses)
            
            curr_pose = poses[-1].copy() # 
            last_kf_pose = curr_pose.copy() 
            kf_poses_list = kf_opt_poses 

    prev_prev_pose = prev_pose.copy()
    prev_pose = curr_pose.copy()

    # ===== Visualization =====
    pts_curr_w_viz = (curr_pose[:3, :3] @ pts.T).T + curr_pose[:3, 3]
    viz.update(pts_curr_w_viz, full_map, poses, kf_poses_list)

# ===== Save Output CSV =====
print("\n--- SLAM process finished ---")
header = "timestamp,m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33"

rows = [[str(t)] + T.reshape(-1).tolist() for t, T in zip(timestamps, poses)]

df = pd.DataFrame(rows, columns=header.split(','))
output_csv_path = f"{target}/lidar_poses.csv"
df.to_csv(output_csv_path, index=False)
print(f"Trajectory saved to {output_csv_path}")


