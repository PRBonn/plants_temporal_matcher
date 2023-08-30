import os
from typing import Tuple
from pathlib import Path
from utils.loading_tools import list_folders, load_camera_paramaters, load_descriptors
import numpy as np
import open3d as o3d
from PIL import Image
import cv2


class Camera:
    def __init__(self, folder: Path):
        self._folder = folder

        # Load the camera paramaters
        self._intrinsics, self._extrinsics = load_camera_paramaters(
            str(folder / "params.yaml")
        )

        # Check that rgb, depth and descriptors folder exist
        subfolders = list_folders(folder)
        assert "rgb" in subfolders, f"rgb does not exist in {folder}"
        assert "depth" in subfolders, f"depth does not exist in {folder}"
        assert "descriptors" in subfolders, f"descriptors does not exist in {folder}"

        # Load the filenames from each folder
        self._rgb_filenames = os.listdir(str(folder / "rgb"))
        self._rgb_filenames.sort()
        self._depth_filenames = os.listdir(str(folder / "depth"))
        self._depth_filenames.sort()
        self._descriptors_filenames = os.listdir(str(folder / "descriptors"))
        self._descriptors_filenames.sort()

        # Check that the number of elements in the three folders is the same
        assert len(self._rgb_filenames) == len(self._depth_filenames) and len(
            self._rgb_filenames
        ) == len(
            self._descriptors_filenames
        ), f"rgb, depth and descriptors do not contain the same number of elements in {folder}"

    def get_intrinsics(self) -> o3d.camera.PinholeCameraIntrinsic:
        return self._intrinsics

    def get_extrinsics(self) -> np.ndarray:
        return self._extrinsics

    def get_image(self, idx: int) -> Tuple[Image.Image, Image.Image]:
        rgb_img = Image.open(str(self._folder / "rgb" / self._rgb_filenames[idx]))
        depth_img = Image.open(str(self._folder / "depth" / self._depth_filenames[idx]))
        return (rgb_img, depth_img)

    def get_descriptor(self, idx: int) -> Tuple[Tuple[cv2.KeyPoint], np.ndarray]:
        return load_descriptors(
            str(self._folder / "descriptors" / self._descriptors_filenames[idx])
        )

    def __repr__(self) -> str:
        return str(self._folder)

    def __str__(self) -> str:
        return str(self._folder)

    def __len__(self):
        return len(self._rgb_filenames)
