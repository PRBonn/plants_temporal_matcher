import open3d as o3d
import numpy as np
from PIL import Image


def threshold(
    pcl: o3d.geometry.PointCloud, axis: int, min_th, max_th
) -> o3d.geometry.PointCloud:
    """
    :param pcl: input PointCloud
    :return: desired point cloud with [min_th, max_th] inclusive in the specified axis
    """
    points = np.asarray(pcl.points)
    remove_mask = np.full(len(points), False)
    remove_mask = np.logical_or(points[:, axis] < min_th, remove_mask)
    remove_mask = np.logical_or(points[:, axis] > max_th, remove_mask)

    keep_mask = np.logical_not(remove_mask)
    keep_indices = np.flatnonzero(keep_mask)

    keep_cloud = pcl.select_by_index(keep_indices)
    return keep_cloud


def threshold_x(
    pcl: o3d.geometry.PointCloud, min_th, max_th
) -> o3d.geometry.PointCloud:
    """
    :param pcl: input PointCloud
    :return: desired point cloud with [min_z, max_z] inclusive
    """
    return threshold(pcl, 0, min_th, max_th)


def threshold_y(
    pcl: o3d.geometry.PointCloud, min_th, max_th
) -> o3d.geometry.PointCloud:
    """
    :param pcl: input PointCloud
    :return: desired point cloud with [min_y, max_y] inclusive
    """
    return threshold(pcl, 1, min_th, max_th)


def threshold_z(
    pcl: o3d.geometry.PointCloud, min_th, max_th
) -> o3d.geometry.PointCloud:
    """
    :param pcl: input PointCloud
    :return: desired point cloud with [min_z, max_z] inclusive
    """
    return threshold(pcl, 2, min_th, max_th)


def extract_stable_features(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    stable_pcd = threshold_z(pcd, 0.75, 0.9)  # 0, 1.2
    return stable_pcd


def extract_point_cloud(
    rgb_img: Image.Image,
    depth_img: Image.Image,
    camera_params: o3d.camera.PinholeCameraIntrinsic,
    min_depth_th: float = 0.6,
    max_depth_th: float = 1.2,
) -> o3d.geometry.PointCloud:
    # Convert the images into open3d representation
    o3d_img_rgb = o3d.geometry.Image(np.asarray(rgb_img))
    o3d_img_depth = o3d.geometry.Image(np.asarray(depth_img).astype(np.uint16))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_img_rgb, o3d_img_depth
    )

    # Extract the point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_params)

    # Filter out from the point cloud the elements over the thresholds
    pcd = threshold_z(pcd, min_depth_th, max_depth_th)

    return pcd


def clean_point_cloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    pcd = pcd.uniform_down_sample(20)
    # pcd = pcd.voxel_down_sample(0.01)
    # pcd, _ = pcd.remove_statistical_outlier(20, 0.5)
    # pcd, _ = pcd.remove_radius_outlier(16, 0.05)
    return pcd


def rigid_registration(
    query_pcd: o3d.geometry.PointCloud,
    ref_pcd: o3d.geometry.PointCloud,
    pose_guess: np.ndarray,
    cam_pose: np.ndarray,
    icp_threshold: float = 0.5,
) -> np.ndarray:
    """rigid_registration.

    :param query_pcd: Point cloud to register
    :type query_pcd: o3d.geometry.PointCloud
    :param ref_pcd: point cloud against which register (IMPORTANT: we
                    assume this point cloud has already normals estimated)
    :type ref_pcd: o3d.geometry.PointCloud
    :param pose_guess: Pose initial guess of the query point cloud
    :type pose_guess: np.ndarray
    :param cam_pose: Pose of the camera from which the observations are taken
    :type cam_pose: np.ndarray
    :param icp_threshold
    :type icp_threshold: float
    :rtype: np.ndarray
    """

    # Bring the query pcd in the robot local frame
    query_pcd.transform(np.linalg.inv(pose_guess))

    # Estimate the query pcd normals
    query_pcd.estimate_normals()

    # Orient the normals of the two pcds toward the camera
    query_pcd.orient_normals_towards_camera_location(cam_pose[0:3, 3])
    ref_pcd.orient_normals_towards_camera_location(cam_pose[0:3, 3])

    # Registration
    registration_result = o3d.pipelines.registration.registration_generalized_icp(
        query_pcd,
        ref_pcd,
        icp_threshold,
        pose_guess,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000),
    )

    return registration_result.transformation
