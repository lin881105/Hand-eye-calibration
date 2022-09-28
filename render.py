import open3d as o3d
import numpy as np
import scipy.io as sio
from utils.tag_detector import TagDetector
from scipy.spatial.transform import Rotation as R

def create_rgbd(rgb, depth, intr, extr, dscale):
    assert rgb.shape[:2] == depth.shape
    (h, w) = depth.shape
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]

    ix, iy =  np.meshgrid(range(w), range(h))

    x_ratio = (ix.ravel() - cx) / fx
    y_ratio = (iy.ravel() - cy) / fy

    z = depth.ravel() / dscale
    x = z * x_ratio
    y = z * y_ratio

    points = np.vstack((x, y, z)).T
    colors = np.reshape(rgb,(depth.shape[0] * depth.shape[1], 3))
    colors = np.array([colors[:,2], colors[:,1], colors[:,0]]).T / 255.

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    pcd.transform(extr)

    return pcd

def main():
    mat_file = 'tf.mat'
    txt_file = 'tf.txt'
    mat_dict = sio.loadmat(mat_file)
    T_tag2camside = None
    with open(txt_file, 'r') as f:
        pose_arr = f.readlines()[0].split(',')
    T_tag2camside_pos = pose_arr[:3]
    T_tag2camside_rot = R.from_quat(pose_arr[3:]).as_matrix()
    T_tag2camside = np.identity(4)
    T_tag2camside[:3, 3] = T_tag2camside_pos
    T_tag2camside[:3, :3] = T_tag2camside_rot
    T_tag2camside = np.linalg.inv(T_tag2camside)
    
    print(mat_dict.keys())
    print(np.asarray(mat_dict['side_rgb']).shape)
    print(np.asarray(mat_dict['side_depth']).shape)

    rgb = np.asarray(mat_dict['side_rgb'])
    depth = np.asarray(mat_dict['side_depth'])
    intr = np.asarray(mat_dict['side_intr'])
    pcd = create_rgbd(rgb, depth, intr, T_tag2camside, dscale=1000)

    T_cam2ee = np.asarray(mat_dict['cam2ee'])
    T_base2gripper = np.asarray(mat_dict['base2gripper'])
    T_target2cam = np.asarray(mat_dict['target2cam'])
    T_tag2ee = T_cam2ee @ T_target2cam
    T_tag2base = np.linalg.inv(T_base2gripper) @ T_tag2ee
    # T_tag2ee = np.linalg.inv(T_cam2ee @ T_target2cam)
    # T_tag2base = T_base2gripper @ T_tag2ee

    coordinate_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    coordinate_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    coordinate_cam.transform(np.linalg.inv(T_target2cam))
    coordinate_ee = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    coordinate_ee.transform(np.linalg.inv(T_tag2ee))
    coordinate_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    coordinate_base.transform(np.linalg.inv(T_tag2base))
    o3d.visualization.draw_geometries([coordinate_origin, coordinate_cam, coordinate_ee ,coordinate_base,pcd])

    return 

if __name__=="__main__":
    main()