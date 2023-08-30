from typing import Tuple
from scipy.spatial import KDTree
import numpy as np
import open3d as o3d
from PIL import Image


def point_image_to_point3d(
    depth_img: Image.Image,
    pose: np.ndarray,
    camera_params: o3d.camera.PinholeCameraIntrinsic,
    h: int,
    w: int,
    pcd_kdtree: KDTree,
    min_depth_th,
    max_depth_th,  # 1.2
    scale_factor,
    nn_threshold,
) -> Tuple[bool, int]:
    """
    Function that, given the (u,v) coordinates of a depth image find the corresponding
    point in a point cloud pcd

    Returns
    ----------
    succ : bool
        If the nearest neigbor is found
    idx : int
        Index of the point in pcd that is found as nearest
        neighbor (-1 if succ is False)
    """

    # Get the depth value of the pixel
    depth_img_np = np.asarray(depth_img)
    z = depth_img_np[h][w]

    # Check that the depth is valid
    if (z / scale_factor) > min_depth_th and (z / scale_factor) < max_depth_th:
        # Compute the 3D position
        x = (
            (w - camera_params.get_principal_point()[0])
            * z
            / camera_params.get_focal_length()[0]
        )
        x /= scale_factor
        y = (
            (h - camera_params.get_principal_point()[1])
            * z
            / camera_params.get_focal_length()[1]
        )
        y /= scale_factor
        z /= scale_factor

        # Transform it according to pose (bring it in world frame)
        p = np.dot(pose, np.asarray([x, y, z, 1]))
        p = p[0:3] / p[3]

        # Search for the nearest neighbor of such point in pcd
        dists, idx = pcd_kdtree.query(
            p.reshape(-1, 3), k=1, distance_upper_bound=nn_threshold, workers=-1
        )

        # Check that it is found at least one neighbor
        if dists[0] == np.inf:
            return (False, -1)

        # Return the nearest neighbor of the unprojected point
        return (True, idx[0])

    else:
        return (False, -1)


def point3d_to_point_image(
    point3d: np.ndarray,
    pose: np.ndarray,
    camera_params: o3d.camera.PinholeCameraIntrinsic,
    scale_factor: float,
) -> Tuple[float, float]:
    p = np.dot(pose, np.append(point3d, 1))
    p = p[0:3] / p[3]
    p *= scale_factor

    u = (
        (camera_params.get_focal_length()[0] * p[0]) / p[2]
    ) + camera_params.get_principal_point()[0]
    v = (
        (camera_params.get_focal_length()[1] * p[1]) / p[2]
    ) + camera_params.get_principal_point()[1]

    return (u, v)
