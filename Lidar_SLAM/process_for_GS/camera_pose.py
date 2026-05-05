import numpy as np
import os
import glob
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import pandas as pd


# --- camera Extrinsics  ---

'''
Those are for Track1
'''
def get_extrinsic_matrices():
    T_top_to_30f = make_T(0.079, -0.082, -0.112, -0.502, 0.506, -0.501, 0.490)
    T_30f_to_60f = make_T(-0.162, -0.003, 0.003, -0.001, -0.002, -0.000, 1.000)
    T_lidar_cam = T_top_to_30f @ T_30f_to_60f
    return T_lidar_cam
'''
Those are for Track2
'''
'''
def get_extrinsic_matrices():
     # Base transformation matrix
    T_lidar_cam = make_T(-0.190, -0.115, 0.070, -0.7071, 0.0, 0.0, 0.7071)

#     # 1. Pitch: 5.75 deg upward
    pitch_angle = np.radians(6.5)
    rot_pitch = R.from_euler('x', pitch_angle, degrees=False).as_matrix()

#     # 2. Yaw: -2.0 deg to the right
    yaw_angle = np.radians(-3.25)
    rot_yaw = R.from_euler('z', yaw_angle, degrees=False).as_matrix()

#     # Extract components
    rot_part = T_lidar_cam[:3, :3]
    trans_part = T_lidar_cam[:3, 3]

#     # 3. Apply
    new_rot = rot_yaw @ rot_part @ rot_pitch

    T_new = np.eye(4)
    T_new[:3, :3] = new_rot
    T_new[:3, 3] = trans_part

    return T_new
'''

def make_T(x, y, z, rx, ry, rz, rw):
    """Create a 4x4 homogeneous transformation matrix from translation and quaternion."""
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    T[:3, :3] = R.from_quat([rx, ry, rz, rw]).as_matrix()
    return T


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
    

def interpolate_pose(target_ts, lidar_ts, lidar_poses):
    """Interpolate the pose using linear interpolation for translation and Slerp for rotation."""
    idx = np.searchsorted(lidar_ts, target_ts)
    if idx == 0 or idx >= len(lidar_ts):
        return None

    # Calculate interpolation weights
    t0, t1 = lidar_ts[idx-1], lidar_ts[idx]
    alpha = (target_ts - t0) / (t1 - t0)

    #  Interpolation for translation
    p0, p1 = lidar_poses[idx-1][:3, 3], lidar_poses[idx][:3, 3]
    p_interp = (1 - alpha) * p0 + alpha * p1

    # Slerp for rotation
    key_rots = R.from_matrix([lidar_poses[idx-1][:3, :3], lidar_poses[idx][:3, :3]])
    key_times = [t0, t1]
    slerp = Slerp(key_times, key_rots)
    r_interp = slerp(target_ts)

    T = np.eye(4)
    T[:3, :3] = r_interp.as_matrix()
    T[:3, 3] = p_interp
    return T


def main():
    # path settings
    target = "/home/lab605/SLAM_3DGS/Track1"
    LIDAR_POSE_FILE = f"{target}/Track1.csv"
    IMG_DIR = f"{target}/data/image"
    OUTPUT_FILE = f"{target}/result/camera_pose.csv"

    # load lidar poses
    lidar_poses, lidar_ts = load_pose_csv(LIDAR_POSE_FILE)

    # get image timestamps
    img_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    img_ts = [int(os.path.basename(f).split('.')[0]) for f in img_files]

    T_extrinsic = get_extrinsic_matrices()

    # CSV Header definition
    header = "timestamp,m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33"

    print(f"Starting calculation for {len(img_ts)} image poses from {len(lidar_ts)} lidar poses...")
    results = []

    
    for ts in img_ts:
        # 1. use interpolationm to get camera pose from lidar pose
        T_base_at_ts = interpolate_pose(ts, lidar_ts, lidar_poses)

        if T_base_at_ts is not None:
            T_cam_world = T_base_at_ts @ T_extrinsic

            row_data = [ts] + T_cam_world.flatten().tolist()
            results.append(row_data)
    
    # save as csv
    print(f"saving to {OUTPUT_FILE}...")
    df_out = pd.DataFrame(results, columns=header.split(','))
    df_out['timestamp'] = df_out['timestamp'].astype(np.int64)
    df_out.to_csv(OUTPUT_FILE, index=False, float_format='%.10f')

    print(f"Successfully saved {len(results)} pose data.")

if __name__ == "__main__":
    main()
