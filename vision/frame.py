from dataclasses import dataclass
from PIL import Image
from typing import Tuple
import numpy as np
import cv2
import open3d as o3d


@dataclass
class Frame:
    rgb_img: Image.Image
    depth_img: Image.Image
    keypoints: Tuple[cv2.KeyPoint]
    descriptors: np.ndarray
    camera_intrinsics: o3d.camera.PinholeCameraIntrinsic
    camera_extrinsics: np.ndarray
