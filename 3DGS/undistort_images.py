"""
Generate undistorted image copies for each track.

Output: <track>/data/image_undistorted/<basename>.{jpg,png}

- For tracks with non-zero D:  cv2.undistort with the ORIGINAL K (so geometry
  stays consistent — the resulting image still corresponds to the same K).
- For tracks with D = 0 (no calibration available):  just copy the originals,
  so downstream code can unconditionally read from `image_undistorted/`.

After running this, you should also re-run:
  - generate_sky_mask.py
  - generate_da3_depth.py
to regenerate per-pixel artefacts on the undistorted frames.

Edit DATA_ROOT and the TRACKS dict below if calibration values change.
"""

import shutil
import sys
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ---------------- Config ----------------
DATA_ROOT = Path("/home/lab605/SLAM_3DGS/")
OVERWRITE = False
# ----------------------------------------

TRACKS = {
    "Track1": {
        "K": np.array([[653.143433778113, 0.0,               299.1738577337179],
                       [0.0,              657.670567367976,  236.60674857178367],
                       [0.0,              0.0,               1.0]], dtype=np.float64),
        # k1, k2, p1, p2, k3  — from the rosbag CameraInfo (see lidar-slam/process/color_mapping.py)
        "D": np.array([0.020117038292372328, -0.05693984506726855,
                       0.0007786953444092887,  0.007650355486501124,
                       -0.03524717637942092], dtype=np.float64),
    },
    "Track2": {
        "K": np.array([[1040.18078, 0.0,         720.04463],
                       [0.0,        1038.55506,  464.33648],
                       [0.0,        0.0,         1.0]], dtype=np.float64),
        # TODO: Track2's actual D not preserved in code. Assume 0 (no distortion)
        # for now — replace if the rosbag CameraInfo becomes available.
        "D": np.zeros(5, dtype=np.float64),
    },
}


def process_track(track: str, K: np.ndarray, D: np.ndarray):
    src_dir = DATA_ROOT / track / "data" / "image"
    dst_dir = DATA_ROOT / track / "data" / "image_undistorted"
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_files = sorted(glob(str(src_dir / "*.jpg"))) + sorted(glob(str(src_dir / "*.png")))
    if not src_files:
        print(f"[{track}] no images in {src_dir}, skipping")
        return

    if not OVERWRITE:
        before = len(src_files)
        src_files = [p for p in src_files if not (dst_dir / Path(p).name).exists()]
        print(f"[{track}] {before - len(src_files)} already exist, processing {len(src_files)}")
        if not src_files:
            print(f"[{track}] nothing to do.")
            return

    sample = cv2.imread(src_files[0])
    H, W = sample.shape[:2]

    if np.allclose(D, 0):
        print(f"[{track}] D ≈ 0  → copying originals (no undistort)")
        for src in tqdm(src_files, desc=f"{track} copy"):
            shutil.copy2(src, dst_dir / Path(src).name)
        return

    print(f"[{track}] {W}x{H}, D=[{', '.join(f'{x:+.4f}' for x in D)}]")
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, R=None, newCameraMatrix=K, size=(W, H), m1type=cv2.CV_16SC2,
    )

    sampled_diffs = []
    for src in tqdm(src_files, desc=f"{track} undistort"):
        img = cv2.imread(src)
        und = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(dst_dir / Path(src).name), und)
        if len(sampled_diffs) < 5:
            sampled_diffs.append(
                float(np.abs(img.astype(np.int16) - und.astype(np.int16)).mean())
            )

    if sampled_diffs:
        print(f"[{track}] mean |orig - undist| over first 5 frames: "
              f"{np.mean(sampled_diffs):.3f} px (avg over all 3 channels)")


def main():
    for track, calib in TRACKS.items():
        print(f"\n{'='*60}\n{track}\n{'='*60}")
        process_track(track, calib["K"], calib["D"])
    print("\n[done]  Updated dataset_readers / generate_sky_mask / generate_da3_depth "
          "will pick up image_undistorted/ automatically.")


if __name__ == "__main__":
    main()
