"""
Estimate camera poses at test timestamps via interpolation of training poses.

Reads:
  - Training poses + timestamps from Dataset (camera_pose_<Track>.csv)
  - Test timestamps from test_frame_list_<Track>.txt  (one filename per line)
Writes:
  - test_pose_list_<Track>.txt  (one row = 16 floats, c2w 4x4 flattened)
"""

import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene.dataset_readers import Dataset


def interpolate_test_poses(track: str):
    repo_root = Path(__file__).resolve().parent
    data_dir = Path("/home/lab605/SLAM_3DGS") / track

    track_small = track.lower()
    test_list_file = repo_root / f"{track_small}/{track_small}_test_frame_list.txt"
    if not test_list_file.exists():
        raise FileNotFoundError(
            f"{track}-specific list not found at {test_list_file}. "
            f"Refusing to fall back — wrong list = wrong-track timestamps = extrapolated garbage poses."
        )

    output_file = repo_root / f"{track_small}/{track_small}_test_pose_list.txt"

    print(f"[Track] {track}")
    print(f"[Test list]  {test_list_file}")
    print(f"[Output]     {output_file}")

    # 1. Load training poses + timestamps via Dataset (already applies all path/format logic)
    dataset = Dataset(str(data_dir))
    train_ts = np.asarray(dataset.timestamps, dtype=np.float64)  # (N,)
    train_poses = np.stack(dataset.poses)                        # (N, 4, 4)

    # 2. Decompose poses into translation + quaternion for proper interpolation
    train_pos = train_poses[:, :3, 3]                            # (N, 3)
    train_quat = R.from_matrix(train_poses[:, :3, :3]).as_quat() # (N, 4) [qx, qy, qz, qw]

    # 3. Normalize timestamps to avoid float64 precision loss
    t0 = train_ts[0]
    norm_train_ts = train_ts - t0

    pos_interp = interp1d(norm_train_ts, train_pos, axis=0, kind='linear', fill_value="extrapolate")
    slerp = Slerp(norm_train_ts, R.from_quat(train_quat))

    # 4. Read test timestamps
    with open(test_list_file) as f:
        test_frames = [line.strip().split('.')[0] for line in f if line.strip()]
    print(f"[info] {len(test_frames)} test frames")

    # 5. Interpolate per test timestamp
    out_rows = []
    out_of_range = 0
    for fid in test_frames:
        t_norm = float(fid) - t0

        if t_norm < norm_train_ts[0] or t_norm > norm_train_ts[-1]:
            out_of_range += 1
            # Clamp slerp input (slerp doesn't extrapolate)
            t_norm_clamped = np.clip(t_norm, norm_train_ts[0], norm_train_ts[-1])
        else:
            t_norm_clamped = t_norm

        p = pos_interp(t_norm)                            # extrapolates
        Rm = slerp(t_norm_clamped).as_matrix()            # clamped (slerp can't extrapolate)

        T = np.eye(4)
        T[:3, :3] = Rm
        T[:3, 3] = p
        out_rows.append(T.flatten())

    if out_of_range:
        print(f"[warn] {out_of_range} test timestamps were outside training range "
              f"(rotation clamped to nearest endpoint, position extrapolated)")

    np.savetxt(output_file, out_rows, fmt='%.18e')
    print(f"[done] wrote {len(out_rows)} test poses to {output_file}")


if __name__ == "__main__":
    TRACK = sys.argv[1] if len(sys.argv) > 1 else "Track2"
    interpolate_test_poses(TRACK)
