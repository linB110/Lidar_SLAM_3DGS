import gtsam
import numpy as np

class PoseGraph:
    def __init__(self):

        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.1)
        parameters.relinearizeSkip = 1 
        
        self.isam = gtsam.ISAM2(parameters)
        
        self.new_factors = gtsam.NonlinearFactorGraph()
        self.new_values = gtsam.Values()
        
        self.existing_keys = set()
    
    def add_node(self, idx, T):
        if idx not in self.existing_keys and not self.new_values.exists(idx):
            pose_f64 = T.astype(np.float64)
            self.new_values.insert(idx, gtsam.Pose3(pose_f64))
            self.existing_keys.add(idx)
        

    def add_prior(self, idx=0, T=np.eye(4)):
        pose_f64 = np.ascontiguousarray(T.astype(np.float64))
    

        sigmas = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3], 
                      dtype=np.float64, order='C')
        sigmas = np.ascontiguousarray(sigmas)
    
        noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
        pose3 = gtsam.Pose3(pose_f64)
    
        if not self.new_values.exists(idx) and idx not in self.existing_keys:
            self.new_values.insert(idx, pose3)
            self.existing_keys.add(idx)
        
        self.new_factors.add(gtsam.PriorFactorPose3(idx, pose3, noise))

        
    def add_odom(self, i, j, T):
        pose_f64 = T.astype(np.float64)

        noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1]*6, dtype=np.float64))
        
        self.new_factors.add(gtsam.BetweenFactorPose3(i, j, gtsam.Pose3(pose_f64), noise))
        
    def add_loop(self, i, j, T):
        pose_f64 = T.astype(np.float64)

        noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.08]*6, dtype=np.float64))
        
        self.new_factors.add(gtsam.BetweenFactorPose3(i, j, gtsam.Pose3(pose_f64), noise))

    def optimize(self):

        try:
            self.isam.update(self.new_factors, self.new_values)

            self.isam.update()
            self.isam.update()
        except Exception as e:
            print(f"[PoseGraph Error] ISAM update failed: {e}")
            return {}

        self.new_factors = gtsam.NonlinearFactorGraph()
        self.new_values = gtsam.Values()

        result = self.isam.calculateEstimate()
        
        opt_poses = {}
        for k in result.keys():
            opt_poses[k] = result.atPose3(k).matrix()

        return opt_poses
