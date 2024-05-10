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
import open3d as o3d
from typing import List, Tuple
from mapping.mapping_tools import (
    extract_stable_features,
)
from utils.loading_tools import load_point_cloud
import numpy as np
from scipy.spatial import KDTree
from pathlib import Path
import os
import copy


class Map:
    def __init__(self, ref_folder: Path) -> None:
        """__init__.
        Function that initialize the world map. It has to be
        initialized with the folder containing the reference
        dataset

        :param ref_folder: folder containing the reference dataset
        :type ref_folder: Path
        :rtype: None
        """
        # Load the reference point cloud
        assert "voxelized_map.ply" in os.listdir(
            str(ref_folder / "mapping_out")
        ), "Reference folder should contain the file 'voxelized_map.ply'"
        self._ref_pcd = load_point_cloud(
            ref_folder / "mapping_out" / "voxelized_map.ply"
        )

        # Pre-compute needed information on the reference point cloud
        self._ref_pcd.estimate_normals()
        self._ref_pcd_kdtree = KDTree(np.asarray(self._ref_pcd.points))
        self._ref_pcd_stable_features = extract_stable_features(self._ref_pcd)

        # Initialize the attributes that will contain the information about the new map
        self._query_pcd = []
        self._point_matches = []

    def integrate_query_pcd(self, query_pcd: o3d.geometry.PointCloud) -> None:
        self._query_pcd.append(query_pcd)
        self._point_matches.append([])

    def add_matches(self, idx: int, matches: List[Tuple[int, int]]) -> None:
        self._point_matches[idx].append(matches)

    def get_matches(self, idx: int, frame_idx: int) -> List[Tuple[int, int]]:
        return self._point_matches[idx][frame_idx]

    def get_n_query_pcds(self) -> int:
        return len(self._query_pcd)

    def get_query_pcd(self, idx: int) -> o3d.geometry.PointCloud:
        return self._query_pcd[idx]

    def get_query_pcd_kdtree(self, idx: int) -> KDTree:
        return KDTree(np.asarray(self.get_query_pcd(idx).points))

    def get_ref_pcd(self) -> o3d.geometry.PointCloud:
        return self._ref_pcd

    def get_ref_pcd_stable_features(self) -> o3d.geometry.PointCloud:
        return self._ref_pcd_stable_features

    def get_ref_pcd_kdtree(self) -> KDTree:
        return self._ref_pcd_kdtree

    def visualize_reference(self) -> None:
        """visualize_reference.
        Function that allows to visualize ony the reference
        map (acquired at time-step t).

        :rtype: None
        """
        o3d.visualization.draw_geometries([self._ref_pcd])

    def visualize_query(self) -> None:
        """visualize_query.
        Function that allows to visualize the map built
        so far (at time-step t+1).

        :rtype: None
        """
        o3d.visualization.draw_geometries([self._query_pcd])

    def visualize_map(self, with_associations: bool = False) -> None:
        map_to_visualize = [self._ref_pcd]
        for el, matches_frame in zip(self._query_pcd, self._point_matches):
            if len(el.points) > 0:
                # Create a colored version of the current query pcd
                colored_el = copy.deepcopy(el)
                colored_el.paint_uniform_color([1, 0.706, 0])
                map_to_visualize.append(colored_el)

                if with_associations:
                    for matches in matches_frame:
                        # Add the lines that connect the matches
                        map_to_visualize.append(self._get_matches_lines(el, matches))
        o3d.visualization.draw_geometries(map_to_visualize)

    def _get_matches_lines(
        self,
        query_pcd: o3d.geometry.PointCloud,
        pcd_matches: List[Tuple[int, int]],
    ) -> o3d.geometry.LineSet:
        # Get the pcd points
        query_pcd_points = np.asarray(query_pcd.points)
        reference_pcd_points = np.asarray(self._ref_pcd.points)

        # Define an open3D line set
        line_set = o3d.geometry.LineSet()

        # Take all the matched points and, for each of them, create a line
        points = []
        lines = []
        current_point_idx = 0
        for match in pcd_matches:
            # Add the query point of the current match
            points.append(query_pcd_points[match[0]])
            current_point_idx += 1

            # Add the reference point of the current match
            points.append(reference_pcd_points[match[1]])
            current_point_idx += 1

            # Add a line connecting the two points
            lines.append([current_point_idx - 2, current_point_idx - 1])

        # Color the lines of blues
        colors = [[1, 0, 0] for i in range(len(lines))]

        # Fill the open3d line set
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        # Return it
        return line_set
