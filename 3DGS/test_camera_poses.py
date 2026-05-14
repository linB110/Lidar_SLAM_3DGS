"""
Test script: Create Gaussians for point cloud + red Gaussians at each camera pose.
"""

import os
import sys
import struct
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene.dataset_readers import Dataset
from utils.sh_utils import RGB2SH


def save_ply(means, colors, scales, filename="visualization.ply"):
    """Save Gaussians as PLY file."""
    num_points = len(means)
    
    # Initialize with identity rotations
    quats = np.zeros((num_points, 4))
    quats[:, 3] = 1.0  # Identity: [0, 0, 0, 1]
    
    # Full opacity
    opacities = np.ones(num_points)
    
    with open(filename, 'wb') as f:
        # Header
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {num_points}\n".encode())
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        f.write(b"property float scale_0\n")
        f.write(b"property float scale_1\n")
        f.write(b"property float scale_2\n")
        f.write(b"property float rot_0\n")
        f.write(b"property float rot_1\n")
        f.write(b"property float rot_2\n")
        f.write(b"property float rot_3\n")
        f.write(b"property float opacity\n")
        f.write(b"property float f_dc_0\n")
        f.write(b"property float f_dc_1\n")
        f.write(b"property float f_dc_2\n")
        f.write(b"end_header\n")
        
        # Data
        for i in range(num_points):
            # Position
            f.write(struct.pack('f', means[i, 0]))
            f.write(struct.pack('f', means[i, 1]))
            f.write(struct.pack('f', means[i, 2]))
            # Scale (in log space)
            f.write(struct.pack('f', scales[i, 0]))
            f.write(struct.pack('f', scales[i, 1]))
            f.write(struct.pack('f', scales[i, 2]))
            # Rotation
            f.write(struct.pack('f', quats[i, 0]))
            f.write(struct.pack('f', quats[i, 1]))
            f.write(struct.pack('f', quats[i, 2]))
            f.write(struct.pack('f', quats[i, 3]))
            # Opacity
            f.write(struct.pack('f', opacities[i]))
            # Color (RGB in SH DC space)
            f.write(struct.pack('f', colors[i, 0]))
            f.write(struct.pack('f', colors[i, 1]))
            f.write(struct.pack('f', colors[i, 2]))
    
    print(f"Saved: {filename} with {num_points} Gaussians")


if __name__ == "__main__":
    data_dir = "/mnt/HDD4/ricky/sdc_hw/3dgs_slam/data/mid2dataset/Track2"
    # data_dir = "/mnt/HDD4/ricky/sdc_hw/3dgs_slam/data/mid2dataset/Track1"
    
    # Load dataset
    dataset = Dataset(data_dir)
    poses_c2w = dataset.get_poses_torch()  # [N, 4, 4]
    points, colors_rgb = dataset.get_pointcloud()
    
    print(f"Loaded {len(poses_c2w)} camera poses")
    print(f"Loaded {len(points)} point cloud points")
    
    # ========== Point Cloud Gaussians ==========
    pc_means = points.numpy()  # [N_pc, 3]
    pc_colors_rgb = colors_rgb.numpy()  # [N_pc, 3], already in [0, 1]
    
    # Convert RGB to SH DC space
    pc_colors_sh = RGB2SH(torch.from_numpy(pc_colors_rgb).float()).numpy()  # [N_pc, 3]
    
    # All scales = 0.02 in log space
    log_scale = np.log(0.02)
    pc_scales = np.full((len(points), 3), log_scale)  # [N_pc, 3]
    
    # ========== Camera Pose Gaussians ==========
    cam_means = poses_c2w[:, :3, 3].numpy()  # [N_cam, 3]
    
    # Red color converted to SH DC space
    red_rgb = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    red_sh = RGB2SH(red_rgb).numpy()  # [1, 3]
    cam_colors = np.tile(red_sh, (len(poses_c2w), 1))  # [N_cam, 3]
    
    cam_scales = np.full((len(poses_c2w), 3), log_scale)  # [N_cam, 3]
    
    # ========== Combine ==========
    all_means = np.vstack([pc_means, cam_means])
    all_colors = np.vstack([pc_colors_sh, cam_colors])
    all_scales = np.vstack([pc_scales, cam_scales])
    
    # Save
    output_dir = Path(".")
    output_dir.mkdir(exist_ok=True)
    ply_path = output_dir / "visualization.ply"
    
    save_ply(all_means, all_colors, all_scales, str(ply_path))
    
    print(f"\nCombined visualization:")
    print(f"  Point cloud: {len(points)} Gaussians (point colors)")
    print(f"  Camera poses: {len(poses_c2w)} Gaussians (red)")
    print(f"  Total: {len(all_means)} Gaussians")
    print(f"  All size: 0.02m")

