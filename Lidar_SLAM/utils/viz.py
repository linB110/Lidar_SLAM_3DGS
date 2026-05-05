import open3d as o3d
import numpy as np

class Visualizer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="SLAM Trajectory & Map", width=1280, height=720)
        
        # Geometries
        self.global_map = o3d.geometry.PointCloud()
        self.current_scan = o3d.geometry.PointCloud()
        self.kf_markers = o3d.geometry.PointCloud()
        self.traj = o3d.geometry.LineSet()

        # Render options
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.05, 0.05, 0.05])
        opt.point_size = 2.0  # Slightly larger points for better visibility
        
        self.first_view = True

    def update(self, curr_pts, global_pts, all_poses, kf_poses):
        # 1. Update Current Scan (Green)
        self.current_scan.points = o3d.utility.Vector3dVector(curr_pts)
        self.current_scan.paint_uniform_color([0.0, 1.0, 0.0])

        # 2. Update Global Map (Gray)
        if global_pts is not None:
            self.global_map.points = o3d.utility.Vector3dVector(global_pts)
            self.global_map.paint_uniform_color([0.65, 0.65, 0.65])

        # 3. Update Trajectory (Yellow)
        if len(all_poses) > 1:
            pts_traj = np.array([p[:3, 3] for p in all_poses])
            lines = [[i, i+1] for i in range(len(pts_traj)-1)]
            self.traj.points = o3d.utility.Vector3dVector(pts_traj)
            self.traj.lines = o3d.utility.Vector2iVector(lines)
            self.traj.paint_uniform_color([1.0, 1.0, 0.0])

        # 4. Update Keyframe Markers (Red, larger points)
        if len(kf_poses) > 0:
            kf_pts = np.array([p[:3, 3] for p in kf_poses])
            self.kf_markers.points = o3d.utility.Vector3dVector(kf_pts)
            self.kf_markers.paint_uniform_color([1.0, 0.0, 0.0])

        # Add geometries to scene
        if self.first_view:
            self.vis.add_geometry(self.global_map)
            self.vis.add_geometry(self.current_scan)
            self.vis.add_geometry(self.kf_markers)
            self.vis.add_geometry(self.traj)
            self.first_view = False
        
        # Update and Refresh
        self.vis.update_geometry(self.global_map)
        self.vis.update_geometry(self.current_scan)
        self.vis.update_geometry(self.kf_markers)
        self.vis.update_geometry(self.traj)

        # 5. Auto-adjust View
        if len(all_poses) > 0:
            view_ctl = self.vis.get_view_control()
            view_ctl.set_lookat(all_poses[-1][:3, 3])
            
            # Zoom out periodically to keep the full path in view
            if len(all_poses) % 20 == 0:
                view_ctl.set_zoom(0.3)
        
        self.vis.poll_events()
        self.vis.update_renderer()
