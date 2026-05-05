import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("Track1_test.pcd")

points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
full_data = np.hstack((points, colors))

# build Axises
# size: 長度 (公尺), origin: 原點位置
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])

print("顯示說明: 紅色=X, 綠色=Y, 藍色=Z")

'''
You can also load poses as trajectories and axes for visualization; refer to /utils/viz.py
'''

o3d.visualization.draw_geometries([pcd, axes], 
                                  window_name="Loaded Map with Axes",
                                  width=1280, 
                                  height=720)
