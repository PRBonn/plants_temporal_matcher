import os
from typing import List, Tuple
from pathlib import Path
from utils.loading_tools import list_folders, load_kitty_poses
from vision.camera import Camera
import numpy as np
from vision.frame import Frame


class PATHoBotDataset(object):
    """PATHoBotDataset.
    This class allow to deal with a PATHoBon dataset with the
    following folder structure:
    - folder_name
        - cameras
            - cam_n
                - params.yaml
                - depth
                - rgb
                - descriptors
            - ...
    If you set reference_dataset to true in the constructor then also
    the files poses_kitty.txt and point_cloud.ply are expected inside
    the main folder.
    NOTE: all the names without extension are supposed to be folders.
    IMPORTANT: all depth, rgb and descriptors folder MUST have the same
    number of files inside, furthermore, at least one cam folder must be
    present. Then, the names of the images should be the same for each folder
    and each cameras.
    """

    def __init__(
        self, folder: Path, n_sensors_to_use: int = -1, reference_dataset: bool = False
    ) -> None:
        self._folder = folder
        self._index = -1

        # Check that we have the images folder
        assert "cameras" in list_folders(self._folder), f"cameras should be in {folder}"

        # Check that there is at least one camera folder
        self._cameras_names = list_folders(self._folder / "cameras")
        self._cameras_names.sort()
        if n_sensors_to_use < 1 or n_sensors_to_use >= len(self._cameras_names):
            self._n_cams = len(self._cameras_names)
        else:
            self._n_cams = n_sensors_to_use
            self._cameras_names = self._cameras_names[:n_sensors_to_use]
        assert (
            self._n_cams > 0
        ), f"should be at least one camera folder into {folder}/cameras"

        # Initialize the cameras
        self._cams = []
        for el in self._cameras_names:
            self._cams.append(Camera(folder / "cameras" / el))

        # Check that all the cameras contain the same number of elements
        self._dataset_lenght = len(self._cams[0])
        for el in self._cams:
            assert len(el) == self._dataset_lenght, f"Unconsistent lenght for cam {el}"

        # If we have a reference dataset load the corresponding poses
        # otherwise initialize empty poses
        if reference_dataset:
            assert "poses_kitty.txt" in os.listdir(
                self._folder
            ), "A reference dataset should contain the poses files: poses_kitty.txt"
            self._poses = load_kitty_poses(self._folder / "poses_kitty.txt")
        else:
            self._poses = [np.identity(4) for i in range(self._dataset_lenght)]

        # Check that the number of poses is consistent with the dataset length
        assert (
            len(self._poses) == self._dataset_lenght
        ), f"Poses length does not correspond to the number of images"

    def get_pose(self, idx: int) -> np.ndarray:
        return self._poses[idx]

    def set_pose(self, idx: int, new_pose: np.ndarray) -> None:
        self._poses[idx] = new_pose

    def get_element(self, idx: int) -> Tuple[List[Frame], np.ndarray]:
        frames = []
        for el in self._cams:
            rgb_img, depth_img = el.get_image(idx)
            kps, descs = el.get_descriptor(idx)
            frames.append(
                Frame(
                    rgb_img,
                    depth_img,
                    kps,
                    descs,
                    el.get_intrinsics(),
                    el.get_extrinsics(),
                )
            )
        return frames, self.get_pose(idx)

    def __len__(self) -> int:
        return self._dataset_lenght

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[List[Frame], np.ndarray]:
        self._index += 1
        if self._index >= self._dataset_lenght:
            self._index = -1
            raise StopIteration
        else:
            return self.get_element(self._index)
