import numpy as np

class LoopClosure:
    def __init__(
        self,
        icp,
        dist_thresh=0.5,
        fitness_thresh=0.5,
        max_candidates=25,
    ):
        """
        icp: ICP instance (small_gicp wrapper)
        """
        self.icp = icp
        self.dist_thresh = dist_thresh
        self.fitness_thresh = fitness_thresh
        self.max_candidates = max_candidates

        # State
        self.keyframes = []
        self.poses = []
        self.last_loop_accum = 0.0

    # Add Keyframe
    def add_keyframe(self, pts, pose):
        self.keyframes.append(pts)
        self.poses.append(pose)

    # Candidate selection
    def find_candidates(self, curr_idx):
        candidates = []
        curr_pose = self.poses[curr_idx]

        for idx in range(len(self.poses) - 1):
            if idx == curr_idx:
                continue
            
            # Calculate distance between current keyframe and candidate
            dist = np.linalg.norm(curr_pose[:3, 3] - self.poses[idx][:3, 3])
            if dist < self.dist_thresh:
                candidates.append(idx)

        return candidates

    # ICP matching
    def match(self, curr_idx, candidates, submap_mgr):
        best_idx = None
        best_T_ij = None
        best_fitness = 0.0
        best_rmse = np.inf

        curr_pts_local = self.keyframes[curr_idx]
        curr_pose = self.poses[curr_idx]  # world ← curr
    
        for idx in candidates:
            cand_pts_local = self.keyframes[idx]
            cand_pose = self.poses[idx]    # world ← cand

            # 1. KF pointcloud -> world
            curr_pts_w = (curr_pose[:3,:3] @ curr_pts_local.T).T + curr_pose[:3,3]
            cand_pts_w = (cand_pose[:3,:3] @ cand_pts_local.T).T + cand_pose[:3,3]

            # 2. ICP in world
            T_icp, score, fitness, inlier_rmse = self.icp.align(
                curr_pts_w, cand_pts_w, init_T=np.eye(4)
            )
    
            if fitness < self.fitness_thresh or inlier_rmse > 1.5:
                continue

            if (fitness > best_fitness) or \
               (np.isclose(fitness, best_fitness) and inlier_rmse < best_rmse):
                best_fitness = fitness
                best_rmse = inlier_rmse

                # 3. cand → cur
                # T_icp * curr_pose ≈ cand_pose
                # ⇒ cand_pose^-1 * (T_icp * curr_pose) ≈ I
                Z_ij = np.linalg.inv(cand_pose) @ (T_icp @ curr_pose)
                best_idx = idx
                best_T_ij = Z_ij

        if best_idx is not None:
            return best_idx, best_T_ij, best_fitness
            
        return None

    # Pose Update
    def update_poses(self, new_kf_poses):
        for i in range(len(new_kf_poses)):
            if i < len(self.poses):
                self.poses[i] = new_kf_poses[i].copy()

    # API
    def detect(self, submap_mgr):
        curr_idx = len(self.keyframes) - 1
        # Could block some too-early detections
        if curr_idx < 30:
            return None

        candidates = self.find_candidates(curr_idx)
        result = self.match(curr_idx, candidates, submap_mgr) 
        
        return result

