from open3d import *
import open3d.visualization

import open3d

import os

# Define the point cloud file path
pcd_file = "Data/export/pointcloud.ply"

# Check if the file exists
if not os.path.isfile(pcd_file):
    print(f"The file {pcd_file} does not exist.")
else:
    print("Loading point cloud from file.")
    # Load the point cloud data from the file
    pointcloud = open3d.io.read_point_cloud(pcd_file)

    print("Down sampling point cloud")
    pointcloud = pointcloud.voxel_down_sample(voxel_size=0.05)

    print("Removing outliers")
    # Apply radius outlier removal
    cl, ind = pointcloud.remove_statistical_outlier(20, 2.0)

    inlier_cloud = open3d.geometry.select_down_sample(cl, ind)

    print("Visualizing point cloud")
    # Visualize the point cloud data
    open3d.visualization.draw_geometries([inlier_cloud])
