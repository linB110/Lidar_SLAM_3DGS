import numpy as np

'''
Select KeyFrame base on distance and angle to last keyframe
'''
class KeyframeManager:
    def __init__(self, dist_thresh=1.0, ang_thresh=10):
        self.last_pose = None
        self.dist_thresh = dist_thresh
        self.ang_thresh = np.deg2rad(ang_thresh)

    def is_keyframe(self, pose):
        if self.last_pose is None:
            self.last_pose = pose
            return True

        dp = pose[:3,3] - self.last_pose[:3,3]
        dist = np.linalg.norm(dp)

        dR = pose[:3,:3] @ self.last_pose[:3,:3].T
        angle = np.arccos(np.clip((np.trace(dR)-1)/2, -1, 1))

        if dist > self.dist_thresh or angle > self.ang_thresh:
            self.last_pose = pose
            return True
            
        return False
