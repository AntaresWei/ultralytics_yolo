import numpy as np
import open3d as o3d

# 读取深度
depth_array = np.load("outercover_s.npy").astype(np.float32)
depth_image = o3d.geometry.Image(depth_array)  # ← numpy → Open3D Image

# 读取彩色图
color_image = o3d.io.read_image(r"data\20260716112715_Color.png")

# 合并 RGB-D
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_image,
    depth_image,
    depth_scale=1.0,
    depth_trunc=20.0,
    convert_rgb_to_intensity=False
)

# 相机参数
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width=depth_array.shape[1],
    height=depth_array.shape[0],
    fx=620.7,
    fy=620.7,
    cx=638.5,
    cy=398.5,
)

# 生成点云（用 create_from_rgbd_image，不是 create_from_depth_image）
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

o3d.io.write_point_cloud("pointcloud1.ply", pcd)