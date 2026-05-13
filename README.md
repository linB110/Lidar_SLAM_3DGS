## LiDAR SLAM 

input : lidar scan pcd file

output : lidar poses, camera poses, lidar map

steps : 
1. modify ICP.py if you wants different parameters or layers of ICP scan matching
2. modify main.py data IO path
3. run main.py
4. output lidar pose csv file 
5. cd 3DGS_process
6. python3 camera_pose.py to generate camera pose from lidar pose using SLERP
7. python3 color_mapping_norm.py to generate lidar map
8. python3 filter_pcd.py to remove outlier of map points

---

## 3DGS

input : RGB images, camera poses, lidar map, depth from depthAnythingV3

output : rendered ply file and visualization of training process

novel view sythesis : 

1. modify dataset_reader.py for data Io and parameters
2. modify train.py for training parameters
3. python3 train.py
4. check rendered result in training

---
