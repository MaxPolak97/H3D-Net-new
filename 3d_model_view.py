import open3d as o3d
import numpy as np

print("Load a ply point cloud, print it, and render it")
pcd1 = o3d.io.read_point_cloud('model_baseline_scans/baseline_scan10/plots/surface_2000.ply')
pcd2 = o3d.io.read_point_cloud('model_baseline_scans/baseline_scan6/plots/surface_2000.ply')
pcd3 = o3d.io.read_point_cloud('model_baseline_scans/baseline_scan22/plots/surface_2000.ply')

pcd_array_1 = np.asarray(pcd1.points)
pcd_array_2 = np.asarray(pcd2.points)
pcd_array_3 = np.asarray(pcd3.points)

pcd_array_avg = np.concatenate((pcd_array_1, pcd_array_2, pcd_array_3), axis=0)

pcd_avg = o3d.geometry.PointCloud()

pcd_avg.points = o3d.utility.Vector3dVector(pcd_array_avg)

o3d.visualization.draw_geometries([pcd_avg])

