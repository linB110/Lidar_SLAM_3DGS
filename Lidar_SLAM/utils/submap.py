import numpy as np
from collections import deque

class SubmapManager:
    def __init__(self, window_size=3):
        self.window = window_size

        self.keyframes_pts = []
        self.keyframes_pose = []
        self.submaps = []

    def add_keyframe(self, pts, pose):
        if pts.shape[1] > 3:
            pts = pts[:, :3]
        self.keyframes_pts.append(pts)
        self.keyframes_pose.append(pose)
        submap = self.build_submap(len(self.keyframes_pts)-1)

        self.submaps.append(submap)

    def update_poses(self, new_kf_poses):
        """ Update KF poses to ensure submaps are built correctly """
        self.keyframes_pose = [p.copy() for p in new_kf_poses]
        # Clear old submaps to force rebuilding with correct poses
        self.submaps = []
        for i in range(len(self.keyframes_pts)):
            sub = self.build_submap(i)
            self.submaps.append(sub)
            
    def build_submap(self, idx):
        """ Stack previous keyframe as a submap """
        start = max(0, idx - self.window)
        end = min(len(self.keyframes_pts), idx + self.window + 1)

        pts_list = []

        for i in range(start, end):
            pts = self.keyframes_pts[i]
            pose = self.keyframes_pose[i]

            pts_w = (pose[:3,:3] @ pts.T).T + pose[:3,3]

            pts_list.append(pts_w)

        return np.concatenate(pts_list, axis=0)

    def get_latest_submap(self):
        if len(self.submaps) == 0:
            return None
        return self.submaps[-1]
