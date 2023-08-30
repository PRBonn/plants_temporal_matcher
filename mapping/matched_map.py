from tqdm import tqdm
import open3d as o3d
from typing import List, Tuple
import numpy as np
from pathlib import Path
from utils.loading_tools import load_point_cloud
import os
from scipy.spatial import KDTree
import copy


class MatchedMap:
    def __init__(self, ref_folder: Path, query_folder: Path) -> None:
        # Load the reference point cloud
        assert "voxelized_map.ply" in os.listdir(
            str(ref_folder / "mapping_out")
        ), "Reference folder should contain the file 'mapping_out/voxelized_map.ply'"
        self._ref_pcd = load_point_cloud(ref_folder / "mapping_out/voxelized_map.ply")

        # Load the query point cloud
        pcd_filename = f"voxelized_map.ply"
        assert pcd_filename in os.listdir(
            str(query_folder / "mapping_out")
        ), "Query folder should contain the file 'mapping_out/voxelized_map.ply'"
        self._query_pcd = load_point_cloud(query_folder / "mapping_out" / pcd_filename)

        # Pre-compute needed information
        self._ref_pcd.estimate_normals()
        self._query_pcd.estimate_normals()
        self._ref_pcd_points = np.asarray(self._ref_pcd.points)
        self._query_pcd_points = np.asarray(self._query_pcd.points)
        self._ref_pcd_kdtree = KDTree(self._ref_pcd_points)
        self._query_pcd_kdtree = KDTree(self._query_pcd_points)
        self._ref_pcd_size = len(self._ref_pcd.points)
        self._query_pcd_size = len(self._query_pcd.points)

        # Initialize all the book-keeping for reference points
        self._ref_point_transformations = []
        self._ref_point_transformations_dists = []
        for _ in range(self._ref_pcd_size):
            self._ref_point_transformations.append(None)
            self._ref_point_transformations_dists.append(np.inf)

        # Initialize all the book-keeping for the query points
        self._query_point_transformations = []
        self._query_point_transformations_dists = []
        self._point_matches_query2ref = []
        for _ in range(self._query_pcd_size):
            self._query_point_transformations.append(None)
            self._query_point_transformations_dists.append(np.inf)
            self._point_matches_query2ref.append(None)

    def size_matches(self) -> int:
        return sum(el is not None for el in self._point_matches_query2ref)

    def get_ref_pcd(self) -> o3d.geometry.PointCloud:
        return self._ref_pcd

    def get_query_pcd(self) -> o3d.geometry.PointCloud:
        return self._query_pcd

    def get_ref_pcd_kdtree(self) -> KDTree:
        return self._ref_pcd_kdtree

    def get_query_pcd_kdtree(self) -> KDTree:
        return self._query_pcd_kdtree

    def add_matches(self, matches: List[Tuple[int, int]]) -> None:
        # For every new match
        for match in matches:
            # Add the new association if we do not have already a
            # point associated to this query
            if self._point_matches_query2ref[match[0]] is None:
                self._point_matches_query2ref[match[0]] = match[1]

    def get_matches(self) -> List[Tuple[int, int]]:
        return self._point_matches

    def visualize_ref_map(self) -> None:
        o3d.visualization.draw_geometries([self._ref_pcd])

    def visualize_query_map(self) -> None:
        o3d.visualization.draw_geometries([self._query_pcd])

    def visualize(self, with_associations: bool = False) -> None:
        colored_query_pcd = copy.deepcopy(self._query_pcd)
        colored_query_pcd.paint_uniform_color([1, 0.706, 0])
        if with_associations:
            o3d.visualization.draw_geometries(
                [self._ref_pcd, colored_query_pcd, self._get_matches_lines()]
            )
        else:
            o3d.visualization.draw_geometries([self._ref_pcd, colored_query_pcd])

    def save_matches(self, filename: Path) -> None:
        with open(filename, "w") as file:
            for query_id, ref_id in enumerate(self._point_matches_query2ref):
                if ref_id is not None:
                    file.write(f"{query_id} {ref_id}\n")

    def save_points_transformations(
        self, ref_filename: Path, query_filename: Path
    ) -> None:
        with open(ref_filename, "w") as file:
            for trans in self._ref_point_transformations:
                if trans is None:
                    file.write("None\n")
                else:
                    file.write(f"{trans[0]} {trans[1]} {trans[2]}\n")
        with open(query_filename, "w") as file:
            for trans in self._query_point_transformations:
                if trans is None:
                    file.write("None\n")
                else:
                    file.write(f"{trans[0]} {trans[1]} {trans[2]}\n")

    def load_matches(self, filename: Path) -> None:
        with open(filename, "r") as file:
            for line in file:
                query_pt_idx, ref_pt_idx = line.split()
                self._point_matches_query2ref[int(query_pt_idx)] = int(ref_pt_idx)

    def load_points_transformations(
        self, ref_filename: Path, query_filename: Path
    ) -> None:
        with open(ref_filename, "r") as file:
            for point_idx, line in enumerate(file):
                if line == "None\n":
                    self._ref_point_transformations[point_idx] = None
                else:
                    x, y, z = line.split()
                    self._ref_point_transformations[point_idx] = [
                        float(x),
                        float(y),
                        float(z),
                    ]
        assert len(self._ref_point_transformations) == self._ref_pcd_size
        with open(query_filename, "r") as file:
            for point_idx, line in enumerate(file):
                if line == "None\n":
                    self._query_point_transformations[point_idx] = None
                else:
                    x, y, z = line.split()
                    self._query_point_transformations[point_idx] = [
                        float(x),
                        float(y),
                        float(z),
                    ]
        assert len(self._query_point_transformations) == self._query_pcd_size

    # This function computes the transformation of each reference point for which
    # we have a sparse association (an association from lobefaro2023iros)
    # Each of this point is called "fixed point"
    def compute_points_transformations_from_matches(self) -> None:
        for query_pt_idx, ref_pt_idx in enumerate(self._point_matches_query2ref):
            if ref_pt_idx is not None:
                self._ref_point_transformations[ref_pt_idx] = (
                    self._query_pcd_points[query_pt_idx]
                    - self._ref_pcd_points[ref_pt_idx]
                )
                self._ref_point_transformations_dists[ref_pt_idx] = 0
                self._query_point_transformations[query_pt_idx] = (
                    self._ref_pcd_points[ref_pt_idx]
                    - self._query_pcd_points[query_pt_idx]
                )
                self._query_point_transformations_dists[query_pt_idx] = 0

    def _find_nearest_points(
        self, point: np.ndarray, map_kd_tree: KDTree, n_points: int, nn_threshold: float
    ) -> Tuple[List[int], List[float]]:
        dists, ids = map_kd_tree.query(
            point.reshape(-1, 3),
            k=n_points,
            distance_upper_bound=nn_threshold,
            workers=-1,
        )

        # If we are searching only for one point deal with th results in a different way
        if n_points == 1:
            if dists[0] != np.inf:
                return [ids[0]], [dists[0]]
            else:
                return [], []

        # If, instead, we are searching for more points, bulid the resulting array
        nearest_ids = []
        nearest_dists = []
        for dist, idx in zip(dists[0], ids[0]):
            if dist != np.inf:
                nearest_ids.append(idx)
                nearest_dists.append(dist)
        return nearest_ids, nearest_dists

    def propagate_T(
        self, nearest_point_same_map_th, nearest_point_other_map_th
    ) -> None:
        # Initialization
        ref_points = np.asarray(self._ref_pcd.points)
        query_points = np.asarray(self._query_pcd.points)

        # Propagate transformation moving ref points
        for query_pt_idx, ref_pt_idx in tqdm(
            enumerate(self._point_matches_query2ref),
            total=len(self._point_matches_query2ref),
            desc="Propagating from reference map to query map",
        ):
            # We do not have a transformation for this point
            if ref_pt_idx is None:
                continue

            # Take the transformation associated to this match
            current_t = self._ref_point_transformations[ref_pt_idx]

            # Find the nearest points in the reference for the current match
            nearest_pts_ids, _ = self._find_nearest_points(
                ref_points[ref_pt_idx],
                self._ref_pcd_kdtree,
                16,
                nearest_point_same_map_th,
            )

            for nn_pt_id in nearest_pts_ids:
                # If this point has already a "consolidated" match, skip it
                if self._ref_point_transformations_dists[nn_pt_id] == 0.0:
                    continue

                # Move the point according to the current t
                moved_pt = ref_points[nn_pt_id] + current_t

                # Search the correspondence in the query pcd after moving the point
                (
                    nearest_query_pts_ids,
                    nearest_query_pts_dists,
                ) = self._find_nearest_points(
                    moved_pt, self._query_pcd_kdtree, 1, nearest_point_other_map_th
                )

                # If we do not find one, skip this point
                if len(nearest_query_pts_ids) < 1:
                    continue

                # Take the absolute value of the distance
                nearest_query_pts_dists[0] = abs(nearest_query_pts_dists[0])

                # If we do not have a transformation for the current point,
                # or we have a better transformation than before, save it
                if (
                    self._ref_point_transformations[nn_pt_id] is None
                    or nearest_query_pts_dists[0]
                    < self._ref_point_transformations_dists[nn_pt_id]
                ):
                    self._ref_point_transformations[nn_pt_id] = (
                        query_points[nearest_query_pts_ids[0]] - ref_points[nn_pt_id]
                    )
                    self._ref_point_transformations_dists[
                        nn_pt_id
                    ] = nearest_query_pts_dists[0]
                    self._query_point_transformations[nearest_query_pts_ids[0]] = (
                        ref_points[nn_pt_id] - query_points[nearest_query_pts_ids[0]]
                    )
                    self._query_point_transformations_dists[
                        nearest_query_pts_ids[0]
                    ] = nearest_query_pts_dists[0]
                    self._point_matches_query2ref[nearest_query_pts_ids[0]] = nn_pt_id

        # Propagate transformations moving query points
        for query_pt_idx, ref_pt_idx in tqdm(
            enumerate(self._point_matches_query2ref),
            total=len(self._point_matches_query2ref),
            desc="Propagating from query map to reference map",
        ):
            # We do not have a transformation for this point
            if ref_pt_idx is None:
                continue

            # Take the transformation associated to this match
            current_t = self._query_point_transformations[query_pt_idx]

            # Find the nearest points in the query for the current match
            nearest_pts_ids, _ = self._find_nearest_points(
                query_points[query_pt_idx],
                self._query_pcd_kdtree,
                16,
                nearest_point_same_map_th,
            )

            for nn_pt_id in nearest_pts_ids:
                # If this point has already a "consolidated" match, skip it
                if self._query_point_transformations_dists[nn_pt_id] == 0.0:
                    continue

                # Move the opint according to the current t
                moved_pt = query_points[nn_pt_id] + current_t

                # Search the correspondence in the ref pcd after moving the point
                nearest_ref_pts_ids, nearest_ref_pts_dists = self._find_nearest_points(
                    moved_pt, self._ref_pcd_kdtree, 1, nearest_point_other_map_th
                )

                # If we do not find one, skip this point
                if len(nearest_ref_pts_ids) < 1:
                    continue

                # Take the absolute value of the distance
                nearest_ref_pts_dists[0] = abs(nearest_ref_pts_dists[0])

                # If we do not have a transformation for the current point,
                # or we have a better transformation than before, save it
                if (
                    self._query_point_transformations[nn_pt_id] is None
                    or nearest_ref_pts_dists[0]
                    < self._query_point_transformations_dists[nn_pt_id]
                ):
                    self._ref_point_transformations[nearest_ref_pts_ids[0]] = (
                        query_points[nn_pt_id] - ref_points[nearest_ref_pts_ids[0]]
                    )
                    self._ref_point_transformations_dists[
                        nearest_ref_pts_ids[0]
                    ] = nearest_ref_pts_dists[0]
                    self._query_point_transformations[nn_pt_id] = (
                        ref_points[nearest_ref_pts_ids[0]] - query_points[nn_pt_id]
                    )
                    self._query_point_transformations_dists[
                        nn_pt_id
                    ] = nearest_ref_pts_dists[0]
                    self._point_matches_query2ref[nn_pt_id] = nearest_ref_pts_ids[0]

        # Consolidate found transformations by setting distances to 0
        self._ref_point_transformations_dists = [
            0 if el is not None else None
            for el in self._ref_point_transformations_dists
        ]

    # Create a new point cloud containing all the points from both ref and query
    # and give the interpolated position of those points that have a transformation
    # associated
    def get_interpolated_pcd(self, t: float) -> o3d.geometry.PointCloud:
        # Initialization
        assert t >= 0 and t <= 1

        # Take the points from the reference pcd
        interpolated_points = []
        interpolated_colors = []
        # ref_pcd_points = np.asarray(self._ref_pcd.points)
        # ref_pcd_colors = np.asarray(self._ref_pcd.colors)
        # for idx, transf in enumerate(self._ref_point_transformations):
        #     if transf is not None:
        #         interpolated_colors.append(ref_pcd_colors[idx])
        #         interpolated_points.append(
        #             ref_pcd_points[idx] + [t * el for el in transf]
        #         )
        #     # else:
        #     #     interpolated_colors.append(ref_pcd_colors[idx])
        #     #     interpolated_points.append(ref_pcd_points[idx])

        # Take the points from the query pcd
        ref_pcd_colors = np.asarray(self._ref_pcd.colors)
        query_pcd_points = np.asarray(self._query_pcd.points)
        query_pcd_colors = np.asarray(self._query_pcd.colors)
        for idx, transf in enumerate(self._query_point_transformations):
            if transf is not None:
                interpolated_colors.append(
                    [
                        (t * query_color) + ((1 - t) * ref_color)
                        for query_color, ref_color in zip(
                            query_pcd_colors[idx],
                            ref_pcd_colors[self._point_matches_query2ref[idx]],
                        )
                    ]
                )
                interpolated_points.append(
                    query_pcd_points[idx] + [(1 - t) * el for el in transf]
                )
            # else:
            #     interpolated_colors.append(query_pcd_colors[idx])
            #     interpolated_points.append(query_pcd_points[idx])
        interpolated_pcd = o3d.geometry.PointCloud()
        interpolated_pcd.points.extend(np.asarray(interpolated_points))
        interpolated_pcd.colors.extend(np.asarray(interpolated_colors))

        return interpolated_pcd

    def get_interpolated_ref(self) -> o3d.geometry.PointCloud:
        interpolated_points = []
        interpolated_colors = []
        ref_pcd_points = np.asarray(self._ref_pcd.points)
        ref_pcd_colors = np.asarray(self._ref_pcd.colors)
        query_pcd_colors = np.asarray(self._query_pcd.colors)
        for idx, transf in enumerate(self._ref_point_transformations):
            if transf is not None:
                interpolated_colors.append(
                    # TODO: color is no interpolated here
                    ref_pcd_colors[idx]
                )
                interpolated_points.append(ref_pcd_points[idx] + [el for el in transf])
            else:
                interpolated_colors.append(ref_pcd_colors[idx])
                interpolated_points.append(ref_pcd_points[idx])

        interpolated_ref = o3d.geometry.PointCloud()
        interpolated_ref.points.extend(np.asarray(interpolated_points))
        interpolated_ref.colors.extend(np.asarray(interpolated_colors))

        return interpolated_ref

    def _get_matches_lines(
        self,
    ) -> o3d.geometry.LineSet:
        # Get the pcd points
        query_pcd_points = np.asarray(self._query_pcd.points)
        reference_pcd_points = np.asarray(self._ref_pcd.points)

        # Define an open3D line set
        line_set = o3d.geometry.LineSet()

        # Take all the matched points and, for each of them, create a line
        points = []
        lines = []
        current_point_idx = 0
        for query_id, ref_id in enumerate(self._point_matches_query2ref):
            # Ignore if this is not an actual association
            if ref_id is None:
                continue

            # Add the query point of the current match
            points.append(query_pcd_points[query_id])
            current_point_idx += 1

            # Add the reference point of the current match
            points.append(reference_pcd_points[ref_id])
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
