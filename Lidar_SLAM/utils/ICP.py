import open3d as o3d
import numpy as np

class ICP:
    def __init__(self, voxel_size = 0.25, max_correspondence_distance=1.0):
        self.max_correspondence_distance = max_correspondence_distance
        self.voxel_size = voxel_size

    def align(self, source_pts, target_pts, init_T=np.eye(4)):

        # numpy -> Open3D
        source_pcd = o3d.geometry.PointCloud()
        target_pcd = o3d.geometry.PointCloud()

        source_pcd.points = o3d.utility.Vector3dVector(source_pts)
        target_pcd.points = o3d.utility.Vector3dVector(target_pts)

        # ===== estimate normals for target =====
        target_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=1.0,
                max_nn=30
            )
        )

        # voxel downsampling
        voxel_size = self.voxel_size
        source_pcd = source_pcd.voxel_down_sample(voxel_size)
        target_pcd = target_pcd.voxel_down_sample(voxel_size)

        source_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=1.0,
                max_nn=30
            )
        )

        # ===== point-to-plane ICP =====
        reg = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            self.max_correspondence_distance,
            init_T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )

        T = reg.transformation

        # score
        score = len(reg.correspondence_set)

        return T, score
