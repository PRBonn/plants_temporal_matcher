# MIT License

# Copyright (c) 2023 Luca Lobefaro, Meher V. R. Malladi, Olga Vysotska, Tiziano Guadagnino, Cyrill Stachniss

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np
from pathlib import Path
from typing import Tuple, List
from vision.dataset import PATHoBotDataset
from vision.frame import Frame
from utils.loading_tools import load_vpr_results
from mapping.map import Map
import open3d as o3d
from mapping.mapping_tools import (
    extract_point_cloud,
    extract_stable_features,
    clean_point_cloud,
    rigid_registration,
)


class Localizer:
    def __init__(
        self,
        ref_dataset: PATHoBotDataset,
        vpr_filename: Path,
        depth_min_th: float,
        depth_max_th: float,
        icp_th: float,
    ) -> None:
        self._vpr_results = load_vpr_results(vpr_filename)
        self._ref_dataset = ref_dataset
        self._depth_min_th = depth_min_th
        self._depth_max_th = depth_max_th
        self._icp_th = icp_th

    def localize(
        self, query_idx: int, query_frames: List[Frame], world_map: Map
    ) -> Tuple[List[Frame], np.ndarray, np.ndarray]:
        # Solve the VPR to obtain an initial guess on the pose
        ref_frames, ref_pose = self.solve_vpr(query_idx)
        query_pose = ref_pose

        # Extract the pcd from the query frame
        new_query_pcd = o3d.geometry.PointCloud()
        for frame in query_frames:
            current_pcd = extract_point_cloud(
                frame.rgb_img,
                frame.depth_img,
                frame.camera_intrinsics,
                self._depth_min_th,
                self._depth_max_th,
            )

            # Bring it in world frame
            current_pcd.transform(query_pose @ frame.camera_extrinsics)

            # Add it to the query pcd
            new_query_pcd += current_pcd

        # Clean the pcd
        new_query_pcd = clean_point_cloud(new_query_pcd)

        # Extract stable features
        new_query_pcd_stable = extract_stable_features(new_query_pcd)

        # Register the stable features in the reference map (stable features)
        # IMPORTANT: use the first camera for normal orientation
        registered_pose = rigid_registration(
            new_query_pcd_stable,
            world_map.get_ref_pcd_stable_features(),
            query_pose,
            query_frames[0].camera_extrinsics,
            self._icp_th,
        )

        # Integrate the computed pcd in the map
        new_query_pcd.transform(np.linalg.inv(query_pose))
        new_query_pcd.transform(registered_pose)
        world_map.integrate_query_pcd(new_query_pcd)

        return ref_frames, registered_pose, ref_pose

    def solve_vpr(self, query_idx: int) -> Tuple[List[Frame], np.ndarray]:
        # Get the reference name associated to this query
        ref_idx = self._vpr_results[query_idx]

        # Get the corresponing pose and reference frame
        ref_frames, ref_pose = self._ref_dataset.get_element(ref_idx)

        return ref_frames, ref_pose
