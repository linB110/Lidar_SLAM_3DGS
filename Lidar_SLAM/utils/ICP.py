# utils/ICP.py (Modified)
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

class ICP:
    def __init__(self, max_correspondence_distance, voxel_size=0.25):

        self.max_correspondence_distance = max_correspondence_distance
        self.voxel_size = voxel_size 
        
        # Point-to-Plane estimation with robust loss
        self.estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.TukeyLoss(k=0.8) # Tukey Loss for robustness
        )

    def align(self, source_pts, target_pts, init_T=np.eye(4), use_global_registration_fallback=False):
        
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_pts)
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_pts)

        # Preprocessing: Voxel downsampling (this will be the 'fine' level base)
        source_pcd_fine = source_pcd.voxel_down_sample(voxel_size=self.voxel_size)
        target_pcd_fine = target_pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # Estimate normals for point-to-plane ICP. Radius is tuned based on voxel size.
        # This parameter is crucial for normal quality.
        normal_radius_fine = self.voxel_size * 2.0 # Recommended 2-3x voxel_size
        target_pcd_fine.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius_fine, max_nn=30)
        )
        source_pcd_fine.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius_fine, max_nn=30)
        )

        current_init_T = init_T
        
        # Multi-scale ICP refinement (Coarsest -> Coarse -> Fine)

        # Stage 1: Coarsest Registration
        # Use a much larger voxel size and correspondence distance, fewer iterations.
        coarsest_voxel_size = self.voxel_size * 4.0 
        coarsest_mcd = self.max_correspondence_distance * 4.0 

        source_pcd_coarsest = source_pcd.voxel_down_sample(coarsest_voxel_size)
        target_pcd_coarsest = target_pcd.voxel_down_sample(coarsest_voxel_size)

        coarsest_normal_radius = coarsest_voxel_size * 2.0
        target_pcd_coarsest.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=coarsest_normal_radius, max_nn=30)
        )
        source_pcd_coarsest.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=coarsest_normal_radius, max_nn=30)
        )

        reg_coarsest = o3d.pipelines.registration.registration_icp(
            source_pcd_coarsest, target_pcd_coarsest,
            coarsest_mcd,
            current_init_T, # Use the initial guess (or global_reg result)
            self.estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50) # Fewer iterations for speed
        )
        # print(f"  [ICP Debug] Coarsest stage fitness: {reg_coarsest.fitness:.4f}")

        # Stage 2: Coarse Registration
        # Use a slightly larger voxel size/MCD than fine, more iterations than coarsest.
        coarse_voxel_size = self.voxel_size * 2.0
        coarse_mcd = self.max_correspondence_distance * 2.0 

        source_pcd_coarse = source_pcd.voxel_down_sample(coarse_voxel_size)
        target_pcd_coarse = target_pcd.voxel_down_sample(coarse_voxel_size)

        coarse_normal_radius = coarse_voxel_size * 2.0
        target_pcd_coarse.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=coarse_normal_radius, max_nn=30)
        )
        source_pcd_coarse.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=coarse_normal_radius, max_nn=30)
        )

        reg_coarse = o3d.pipelines.registration.registration_icp(
            source_pcd_coarse, target_pcd_coarse,
            coarse_mcd,
            reg_coarsest.transformation, # Use result from coarsest stage as initial for coarse
            self.estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80)
        )
        # print(f"  [ICP Debug] Coarse stage fitness: {reg_coarse.fitness:.4f}")

        # Stage 3: Fine Registration
        # Use the original voxel size and MCD, most iterations for precision.
        fine_reg = o3d.pipelines.registration.registration_icp(
            source_pcd_fine, target_pcd_fine, # Use the most dense downsampled PCs
            self.max_correspondence_distance,
            reg_coarse.transformation, # Use result from coarse stage as initial for fine
            self.estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=150) # Increased iterations for fine tuning
        )
        # print(f"  [ICP Debug] Fine stage fitness: {fine_reg.fitness:.4f}")

        T = fine_reg.transformation
        fitness = fine_reg.fitness
        inlier_rmse = fine_reg.inlier_rmse
        # Score calculation, ensure no division by zero
        score = fitness / (inlier_rmse + 1e-6) if inlier_rmse > 1e-6 else fitness * 1e6 # Large score if rmse is tiny

        return T, score, fitness, inlier_rmse


