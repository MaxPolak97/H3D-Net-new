import open3d as o3d
import numpy as np

print("Load a ply point cloud, print it, and render it")
ply_point_cloud = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud('model_baseline_scans/baseline_scan10/plots/surface_2000.ply')
print(pcd)
print(np.asarray(pcd.points))
o3d.visualization.draw_geometries([pcd])

