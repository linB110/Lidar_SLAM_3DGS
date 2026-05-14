"""
Dataset reader for loading camera poses, images, intrinsics, and point cloud.
"""

import os
import numpy as np
import pandas as pd
import cv2
import torch
import open3d as o3d
from pathlib import Path


class Dataset:
    """
    Loads and manages dataset (images, poses, intrinsics, point cloud).
    Expects directory layout produced by the lidar-slam pipeline:
        <data_dir>/
        ├── data/image/                       # camera frames
        └── result/
            ├── camera_pose_<Track>.csv       # header: timestamp,m00,...,m33
            ├── <Track>.pcd                   # colored map
            └── sky_masks/                    # produced by generate_sky_mask.py
    """

    # Hardcoded camera intrinsics (per SDC 2026 Spring Midterm II spec)
    INTRINSICS = {
        "Track1": dict(fx=653.143433778113, fy=657.670567367976,
                       cx=299.1738577337179, cy=236.60674857178367,
                       width=640, height=480),
        "Track2": dict(fx=1040.18078, fy=1038.55506,
                       cx=720.04463,  cy=464.33648,
                       width=1440, height=928),
    }

    def __init__(self, data_dir, track=None):
        """
        Args:
            data_dir: Track-level directory (e.g. data/mid2dataset/Track1)
            track:    "Track1" / "Track2"; if None, inferred from folder name
        """
        self.data_dir = Path(data_dir)
        self.track = track or self.data_dir.name
        if self.track not in self.INTRINSICS:
            raise ValueError(f"Unknown track '{self.track}'. Expected one of {list(self.INTRINSICS)}")

        # Prefer pre-undistorted images if available; fall back to raw.
        _undist = self.data_dir / "data" / "image_undistorted"
        _raw    = self.data_dir / "data" / "image"
        if _undist.exists():
            self.image_dir = _undist
            print(f"[Dataset] Using UNDISTORTED images: {self.image_dir}")
        else:
            self.image_dir = _raw
            print(f"[Dataset] Using RAW images (run undistort_images.py first): {self.image_dir}")
        self.poses_file = self.data_dir / "result" / f"camera_pose_{self.track}.csv"
        self.pointcloud_file = self.data_dir / "result" / f"{self.track}.pcd"
        self.mask_dir = self.data_dir / "result" / "sky_masks"
        self.da3_depth_dir = self.data_dir / "npz_out"

        self._load_poses()
        self._load_intrinsics()
        self._load_image_list()
        self._load_pointcloud()
        self._load_mask_list()
        self._load_da3_depth_list()

    def _load_poses(self):
        """Load camera poses from CSV (header + timestamp + 16 flattened c2w floats)."""
        df = pd.read_csv(self.poses_file)
        self.timestamps = df.iloc[:, 0].astype(np.int64).values
        pose_data = df.iloc[:, 1:].values.astype(np.float32)
        self.poses = [row.reshape(4, 4) for row in pose_data]
        print(f"[Dataset] Loaded {len(self.poses)} camera poses ({self.track})")

    def _load_intrinsics(self):
        """Set hardcoded intrinsics for the current track."""
        p = self.INTRINSICS[self.track]
        self.image_width = p['width']
        self.image_height = p['height']
        self.K = np.array([
            [p['fx'], 0.0,     p['cx']],
            [0.0,     p['fy'], p['cy']],
            [0.0,     0.0,     1.0]
        ], dtype=np.float32)
        print(f"[Dataset] Image size: {self.image_width}x{self.image_height}")
        print(f"[Dataset] Intrinsics: fx={p['fx']:.1f}, fy={p['fy']:.1f}, "
              f"cx={p['cx']:.1f}, cy={p['cy']:.1f}")
    
    def _load_image_list(self):
        """Load list of image filenames."""
        self.image_filenames = sorted(os.listdir(self.image_dir))
        print(f"[Dataset] Found {len(self.image_filenames)} images")
    
    def _load_pointcloud(self):
        """Load point cloud map."""
        pcd = o3d.io.read_point_cloud(str(self.pointcloud_file))
        self.points_map = np.asarray(pcd.points, dtype=np.float32)
        self.colors_map = np.asarray(pcd.colors, dtype=np.float32)
        print(f"[Dataset] Loaded point cloud with {len(self.points_map)} points")

    def _load_mask_list(self):
        """Pre-check mask availability (tolerant if not yet generated)."""
        if not self.mask_dir.exists():
            print(f"[Dataset] No sky_masks dir at {self.mask_dir} (run generate_sky_mask.py first)")
            self.mask_filenames = []
            return
        self.mask_filenames = sorted(os.listdir(self.mask_dir))
        print(f"[Dataset] Found {len(self.mask_filenames)} sky masks")

    def _load_da3_depth_list(self):
        """Pre-check DA3 depth availability (tolerant if not yet generated)."""
        if not self.da3_depth_dir.exists():
            print(f"[Dataset] No da3_depth dir at {self.da3_depth_dir} (run generate_da3_depth.py first)")
            self.da3_depth_filenames = []
            return
        self.da3_depth_filenames = sorted(os.listdir(self.da3_depth_dir))
        print(f"[Dataset] Found {len(self.da3_depth_filenames)} DA3 depth maps")
    
    def get_poses_torch(self):
        """Get poses as torch tensors (c2w matrices)."""
        poses_list = []
        for pose in self.poses:
            # Poses are already in c2w format (camera-to-world)
            poses_list.append(torch.from_numpy(pose).float())
        return torch.stack(poses_list)  # (N, 4, 4)
    
    def get_intrinsics_torch(self):
        """Get intrinsics as torch tensor."""
        return torch.from_numpy(self.K).float()  # (3, 3)
    
    def get_image(self, frame_idx):
        """
        Load image for a given frame.
        
        Args:
            frame_idx: Frame index
        
        Returns:
            Image as torch tensor (3, H, W) with values in [0, 1]
        """
        if frame_idx >= len(self.image_filenames):
            raise IndexError(f"Frame {frame_idx} out of range")
        
        image_path = self.image_dir / self.image_filenames[frame_idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        
        # Convert to torch tensor (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image
    
    def get_image_batch(self, frame_indices):
        """
        Load multiple images.
        
        Args:
            frame_indices: List of frame indices
        
        Returns:
            Stack of images (N, 3, H, W)
        """
        images = [self.get_image(i) for i in frame_indices]
        return torch.stack(images)
    
    def get_pointcloud(self):
        """
        Get point cloud map.
        
        Returns:
            points: (N, 3) torch tensor
            colors: (N, 3) torch tensor
        """
        points = torch.from_numpy(self.points_map).float()
        colors = torch.from_numpy(self.colors_map).float()
        return points, colors

    def get_mask(self, frame_idx):
        """Load a single binary sky mask."""
        mask_path = self.mask_dir / self.image_filenames[frame_idx].replace(".jpg", ".png").replace(".jpeg", ".png")
        if not mask_path.exists():
            return None
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return torch.from_numpy(mask) # (H, W)

    def get_da3_depth(self, frame_idx, device="cuda"):
        """Load DA3 metric depth + confidence for a frame.

        Returns:
            (depth, conf): float32 torch tensors of shape (H, W) at original image resolution,
            or (None, None) if no DA3 depth is available for this frame.
        """
        if not self.da3_depth_dir.exists():
            return None, None
        img_name = self.image_filenames[frame_idx]
        stem = Path(img_name).stem
        npz_path = self.da3_depth_dir / f"{stem}.npz"
        if not npz_path.exists():
            return None, None
        data = np.load(npz_path)
        depth = torch.from_numpy(data["depth"].astype(np.float32)).to(device)
        conf = torch.from_numpy(data["conf"].astype(np.float32)).to(device)
        return depth, conf

    def get_lidar_depth(self, frame_idx, device="cuda"):
        """
        Project the global point cloud into a camera frame to create a sparse depth map.
        
        Returns:
            depth_map: (H, W) torch tensor with depth values in meters
        """
        # 1. Get raw data from the class attributes
        points = torch.from_numpy(self.points_map).float().to(device) # (N, 3)
        pose_c2w = torch.from_numpy(self.poses[frame_idx]).float().to(device) # (4, 4)
        K = torch.from_numpy(self.K).float().to(device) # (3, 3)
        
        # 2. Transform points from World space to Camera space
        # P_cam = R_inv * (P_world - t) = w2c * P_world
        pose_w2c = torch.linalg.inv(pose_c2w)
        
        # Add homogeneous coordinate for matrix multiplication
        points_h = torch.cat([points, torch.ones((points.shape[0], 1), device=device)], dim=-1)
        p_cam = (pose_w2c @ points_h.T).T # (N, 4)
        
        # 3. Filter points
        # Only keep points in front of the camera (positive Z)
        z = p_cam[:, 2]
        mask = z > 0.1 # Near plane clipping
        
        p_cam = p_cam[mask]
        z = z[mask]
        
        # 4. Project to 2D pixel coordinates
        # [u, v, 1] = K * [x/z, y/z, 1]
        p_pix = (K @ (p_cam[:, :3] / z.unsqueeze(-1)).T).T # (N, 3)
        u = p_pix[:, 0].long()
        v = p_pix[:, 1].long()
        
        # 5. Filter points within image boundaries
        valid_mask = (u >= 0) & (u < self.image_width) & (v >= 0) & (v < self.image_height)
        u, v, z = u[valid_mask], v[valid_mask], z[valid_mask]
        
        # 6. Create sparse depth map with Z-buffer logic
        # Initialize with a large value so we can take the minimum depth for overlapping points
        depth_map = torch.zeros((self.image_height, self.image_width), device=device)
        
        # Sort by depth descending so that when we index_put, the closest points (last written) remain
        # This is a simple way to handle occlusion in sparse maps
        indices = torch.argsort(z, descending=True)
        u, v, z = u[indices], v[indices], z[indices]
        
        depth_map[v, u] = z
        
        return depth_map

    def __len__(self):
        """Get number of frames."""
        return len(self.poses)
    
    def __repr__(self):
        return f"Dataset(frames={len(self)}, points={len(self.points_map)}, size={self.image_width}x{self.image_height})"
