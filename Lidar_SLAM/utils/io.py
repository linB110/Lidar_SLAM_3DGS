import open3d as o3d
import numpy as np
import os

def load_pcd(file):
    pcd = o3d.io.read_point_cloud(file)
    pts = np.asarray(pcd.points)

    if pts.shape[1] > 3:
        pts = pts[:, :3]

    return pts

def load_folder(folder):
    files = sorted(os.listdir(folder))
    return [os.path.join(folder, f) for f in files if f.endswith(".pcd")]