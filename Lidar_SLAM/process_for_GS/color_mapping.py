import open3d as o3d
import pandas as pd
import numpy as np
import os
import os
import glob
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# --- camera Intrinsics  ---
'''
Those are for Track1
'''
W, H = 640, 480
FX, FY = 653.143433778113, 657.670567367976
CX, CY = 299.1738577337179, 236.60674857178367

'''
Those are for Track2

W, H = 1440, 928
FX, FY = 1040.18078, 1038.55506
CX, CY = 720.04463, 464.33648
K (Intrinsics): [653.143433778113, 0.0, 299.1738577337179, 0.0, 657.670567367976, 236.60674857178367, 0.0, 0.0, 1.0]
D (Distortion): [0.020117038292372328, -0.05693984506726855, 0.0007786953444092887, 0.007650355486501124, -0.03524717637942092]
'''
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])

def load_pose_csv(file_path):
    """
    Robustly loads pose CSV in either Quaternion (9-col) or Matrix (17-col) format.
    Returns: (list of 4x4 matrices, list of int64 timestamps)
    """
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None, None

    try:
        df = pd.read_csv(file_path)
        
        # Ensure timestamp is treated as 64-bit integer for precision
        possible_keys = ['timestamp', 'timestamps', 'ts', 'time']
        ts_col = next((col for col in df.columns if col.lower() in possible_keys), None)
        if ts_col is None:
            print("Warning: No timestamp key found, falling back to column 0.")
            ts_col = df.columns[0]
        timestamps = df[ts_col].astype(np.int64).values
        
        poses = []

        # Check column count to determine format
        # Format A: id, ts, x, y, z, qx, qy, qz, qw (9 columns)
        if df.shape[1] == 9:
            data = df.iloc[:, 2:].values # Extract x, y, z, qx, qy, qz, qw
            for row in data:
                T = np.eye(4)
                T[:3, 3] = row[:3] # x, y, z
                # Scipy expects [qx, qy, qz, qw]
                T[:3, :3] = R.from_quat(row[3:]).as_matrix()
                poses.append(T)

        # Format B: ts, m00...m33 (17 columns)
        elif df.shape[1] == 17:
            data = df.iloc[:, 1:].values # Skip timestamp
            for row in data:
                T = row.reshape(4, 4)
                poses.append(T)

        else:
            raise ValueError(f"Unknown CSV format with {df.shape[1]} columns.")

        return poses, timestamps

    except Exception as e:
        print(f"Failed to load Pose CSV: {e}")
        return None, None


def build_map(lidar_poses, pose_ts, pcd_dir, voxel_size=0.2):
    """
    Builds global map by time-syncing PCD files with the closest available pose.
    Includes ego-vehicle spatial filtering and outlier removal.
    """
    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))
    combined_pcd = o3d.geometry.PointCloud()

    # Convert pose timestamps to numpy array for fast broadcasting/matching
    pose_ts = np.array(pose_ts, dtype=np.int64)

    print(f"Syncing and Merging {len(pcd_files)} PCD frames...")
    for pcd_path in tqdm(pcd_files):
        try:
            file_ts = np.int64(os.path.splitext(os.path.basename(pcd_path))[0])
        except ValueError:
            continue

        # Time Synchronization: Find the index of the closest pose timestamp
        closest_idx = np.argmin(np.abs(pose_ts - file_ts))

        pcd = o3d.io.read_point_cloud(pcd_path)
        pts = np.asarray(pcd.points)
        if pts.shape[1] > 3:
            pts = pts[:, :3]

        '''
        [Optional]:
         
        You can filter out some point for better look
        (maybe better gaussian init point guess)
        '''

        pcd.points = o3d.utility.Vector3dVector(pts)


        # Apply transformation and merge
        pcd.transform(lidar_poses[closest_idx])
        combined_pcd += pcd

    # --- Global Post-processing ---
    print("Performing global downsampling")
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=voxel_size)

    '''
    [Optional]:
        
    Or filter here, after all frames point stack into a map
    
    '''
        
    return combined_pcd


def colorize_map(pcd, img_dir, img_poses):
    """Projects image colors onto the point cloud using Z-buffer occlusion handling."""
    if img_poses is None or not os.path.exists(img_dir):
        print("Skipping colorization: Image data or poses not found.")
        return pcd

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    points = np.asarray(pcd.points)
    n_points = points.shape[0]

    # Initialize color accumulator and visibility counter
    color_sum = np.zeros((n_points, 3))
    color_counts = np.zeros((n_points, 1))

    print(f"Step 2: Colorizing point cloud ({len(img_files)} images)...")

    for i, img_path in tqdm(enumerate(img_files), total=len(img_files)):
        if i >= len(img_poses):
            break

        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

        # Transform points from World to Camera frame
        T_wc = img_poses[i]
        T_cw = np.linalg.inv(T_wc)
        pts_homo = np.hstack((points, np.ones((n_points, 1))))
        pts_cam = (T_cw @ pts_homo.T).T

        # Filter points behind the camera
        valid_z_mask = pts_cam[:, 2] > 0.1
        if not np.any(valid_z_mask): continue

        # Project 3D points to 2D image plane
        pts_2d_homo = (K @ pts_cam[valid_z_mask, :3].T).T
        u = (pts_2d_homo[:, 0] / pts_2d_homo[:, 2]).astype(int)
        v = (pts_2d_homo[:, 1] / pts_2d_homo[:, 2]).astype(int)
        z = pts_2d_homo[:, 2]

        # Filter points within image boundaries
        valid_uv_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)

        # Z-buffer occlusion handling
        depth_buffer = np.full((H, W), np.inf)
        valid_indices = np.where(valid_z_mask)[0][valid_uv_mask]
        u_valid, v_valid, z_valid = u[valid_uv_mask], v[valid_uv_mask], z[valid_uv_mask]

        for vv, uu, zz in zip(v_valid, u_valid, z_valid):
            if zz < depth_buffer[vv, uu]:
                depth_buffer[vv, uu] = zz

        # Accumulate colors for visible points
        for idx, vv, uu, zz in zip(valid_indices, v_valid, u_valid, z_valid):
            if zz <= depth_buffer[vv, uu] * 1.05:
                color_sum[idx] += img[vv, uu]
                color_counts[idx] += 1

    # Average colors based on visibility frequency
    final_colors = np.zeros((n_points, 3))
    colored_mask = (color_counts > 0).flatten()
    final_colors[colored_mask] = color_sum[colored_mask] / color_counts[colored_mask]

    # Assign grey to unprojected points
    final_colors[~colored_mask] = [0.2, 0.2, 0.2]

    pcd.colors = o3d.utility.Vector3dVector(final_colors)
    return pcd


def draw_camera_trajectory_and_axes(poses, step=20, scale=0.5):
    """
    poses: List of 4x4 transformation matrices.
    step: Interval at which to draw the coordinate frame (axis).
    scale: Size of the coordinate frame axes.
    """
    # Extract translation components for trajectory points
    points = [p[:3, 3] for p in poses]

    # 1. Create a red line set for the trajectory
    lines = [[i, i + 1] for i in range(len(points) - 1)]
    colors = [[1, 0, 0] for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # 2. Create coordinate frames at specified intervals
    geometries = [line_set]
    for i in range(0, len(poses), step):
        # Create a coordinate frame representing the camera pose
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
        # Apply the transformation matrix to align the frame with the pose
        axis.transform(poses[i])
        geometries.append(axis)

    return geometries


def main():
    target = "/home/lab605/SLAM_3DGS/Track1"

    # Lidar & Pose for mapping
    RAW_LIDAR_DIR = f"{target}/data/raw_pcd"
    LIDAR_POSE_PATH = f"{target}/Track1.csv"

    # Image & Pose for Colorization
    IMG_DIR = f"{target}/data/image"
    IMG_POSE_PATH = f"{target}/result/camera_pose.csv"

    # Output
    OUTPUT_PCD = f"{target}/result/Track2.pcd"

    # ---------------------------------------------------------
    # Map Building Section]
    # 1. Load Poses and building map
    poses, timestamps = load_pose_csv(LIDAR_POSE_PATH)

    if poses is None or timestamps is None:
        print("Lack of timestamp or poses")
        return
    full_map = build_map(poses, timestamps, RAW_LIDAR_DIR, voxel_size=0.1)
    o3d.io.write_point_cloud("output_map.pcd", full_map)
    print("Map processing complete.")

    # 2. Outlier Removal
    # nb_neighbors: Number of neighbors to analyze
    # std_ratio: Threshold based on standard deviation (lower = more aggressive)
    print("Performing outlier removal...")
    full_map, ind = full_map.remove_statistical_outlier(nb_neighbors=80, std_ratio=2.0)
    # Optional: Downsample if the SLAM map is too dense
    full_map = full_map.voxel_down_sample(voxel_size=0.2)

    # 3. Image Colorization
    print("Starting colorization...")
    img_poses, _ = load_pose_csv(IMG_POSE_PATH)
    if img_poses is not None:
        full_map = colorize_map(full_map, IMG_DIR, img_poses)
    else:
        print("Warning: Failed to load camera poses. Skipping colorization.")

    # 4. Save and Visualization
    o3d.io.write_point_cloud(OUTPUT_PCD, full_map)
    print(f"Colored map saved to: {OUTPUT_PCD}")

    print("\n[INFO] Visualization Legend:")
    print("Red lines: Camera Trajectory")
    print("Axes Colors: X-axis (Red), Y-axis (Green), Z-axis (Blue)")
    print("----------------------------------------------------------\n")

    # pose and orientation
    cam_geometries = draw_camera_trajectory_and_axes(img_poses, step=20, scale=0.8)

    o3d.visualization.draw_geometries(
        [full_map] + cam_geometries,
        window_name="Refined Color Map with Camera Path",
        width=1280,
        height=720,
        mesh_show_back_face=True
    )

if __name__ == "__main__":
    main()
