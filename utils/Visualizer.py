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
from functools import partial
import copy
from typing import List, Callable
import open3d as o3d
import numpy as np

from mapping.matched_map import MatchedMap

WHITE = np.array([1.0, 1.0, 1.0])
BLACK = np.array([0, 0, 0]) / 255.0
BLUE = np.array([0.4, 0.5, 0.9])
YELLOW = np.array([1, 0.706, 0])
RED = np.array([128, 0, 0]) / 255.0


class InterpolationVisualizer:
    def __init__(self, matched_map: MatchedMap, visualize_complete_maps: bool = False):
        # Initialize the data
        self._matched_map = matched_map
        self._visualize_complete_maps = visualize_complete_maps
        self._ref_visible = False
        self._query_visible = False
        self._keep_running = True
        self._current_t = 0

        # Initialize the colors
        self._colors = [BLACK, WHITE, BLUE]
        self._pcds_colors = [None, BLUE, YELLOW, RED]
        self._current_color = 0
        self._current_ref_color = 0
        self._current_query_color = 0

        # Intiialize the point cloud to visualize
        self._initialize_visualized_point_clouds()

        # Initialize the _visualizers
        self._vis = o3d.visualization.VisualizerWithKeyCallback()
        if visualize_complete_maps:
            self._ref_vis = o3d.visualization.Visualizer()
            self._query_vis = o3d.visualization.Visualizer()
        self._register_key_callbacks()
        self._initialize_visualizers()

    def update(self):
        self._vis.update_geometry(self._interpolated_pcd)
        self._vis.update_geometry(self._ref_pcd)
        self._vis.update_geometry(self._query_pcd)
        self._vis.poll_events()
        if self._visualize_complete_maps:
            self._ref_vis.poll_events()
            self._query_vis.poll_events()
        self._vis.update_renderer()
        if self._visualize_complete_maps:
            self._ref_vis.update_renderer()
            self._query_vis.update_renderer()
        return self._keep_running

    def _initialize_visualized_point_clouds(self):
        self._interpolated_pcd = self._matched_map.get_interpolated_pcd(
            self._current_t / 10.0
        )
        self._ref_pcd = o3d.geometry.PointCloud()
        self._query_pcd = o3d.geometry.PointCloud()
        self._update_ref_pcd()
        self._update_query_pcd()

    def _initialize_visualizers(self):
        # Create windows
        self._vis.create_window(
            window_name="INTERPOLATION VISUALIZER", width=1920, height=1080
        )
        if self._visualize_complete_maps:
            self._ref_vis.create_window(
                window_name="TIMESTEP 0", width=1920, height=1080
            )
            self._query_vis.create_window(
                window_name="TIMESTEP 1", width=1920, height=1080
            )

        # Add geometries to the visualizer
        self._vis.add_geometry(self._interpolated_pcd)
        self._vis.add_geometry(self._ref_pcd)
        self._vis.add_geometry(self._query_pcd)
        if self._visualize_complete_maps:
            self._ref_vis.add_geometry(self._matched_map.get_ref_pcd())
            self._query_vis.add_geometry(self._matched_map.get_query_pcd())

        # Set visualizers properties
        self._set_background(self._vis, self._current_color)
        self._vis.get_render_option().point_size = 3
        if self._visualize_complete_maps:
            self._ref_vis.get_render_option().point_size = 3
            self._query_vis.get_render_option().point_size = 3
        print(
            f"Press:\n"
            "\t[N] to move forward in time\n"
            "\t[B] to move backward in time\n"
            "\t[C] to change the background color\n"
            "\t[0] to visualize/change color of the map at timestep 0\n"
            "\t[1] to visualize/change color of the map at timestep 1\n"
            "\t[A] to visualize the points associations\n"
            "\t[Q] to close the visualizer\n"
        )

    def _register_key_callback(self, keys: List, callback: Callable):
        for key in keys:
            self._vis.register_key_callback(ord(str(key)), partial(callback))

    def _register_key_callbacks(self):
        self._register_key_callback(["N"], self._next_timestep)
        self._register_key_callback(["B"], self._previous_timestep)
        self._register_key_callback(["C"], self._change_background_color)
        self._register_key_callback(["0"], self._change_ref_color)
        self._register_key_callback(["Q"], self._close_visualizer)
        self._register_key_callback(["A"], self._visualize_associations)
        self._register_key_callback(["Q"], self._close_visualizer)

    def _update_ref_pcd(self):
        if self._pcds_colors[self._current_ref_color] is None:
            self._ref_pcd.points = o3d.utility.Vector3dVector([])
            self._ref_pcd.colors = o3d.utility.Vector3dVector([])
            return
        elif len(self._ref_pcd.points) == 0:
            ref_pcd = copy.deepcopy(self._matched_map.get_ref_pcd())
            self._ref_pcd.points = o3d.utility.Vector3dVector(ref_pcd.points)
            self._ref_pcd.colors = o3d.utility.Vector3dVector(ref_pcd.colors)
        self._ref_pcd.paint_uniform_color(self._pcds_colors[self._current_ref_color])

    def _update_query_pcd(self):
        if self._pcds_colors[self._current_query_color] is None:
            self._query_pcd.points = o3d.utility.Vector3dVector([])
            self._query_pcd.colors = o3d.utility.Vector3dVector([])
            return
        elif len(self._query_pcd.points) == 0:
            query_pcd = copy.deepcopy(self._matched_map.get_query_pcd())
            self._query_pcd.points = o3d.utility.Vector3dVector(query_pcd.points)
            self._query_pcd.colors = o3d.utility.Vector3dVector(query_pcd.colors)
        self._query_pcd.paint_uniform_color(
            self._pcds_colors[self._current_query_color]
        )

    def _update_interpolated_pcd(self, t: int):
        interpolated_pcd = self._matched_map.get_interpolated_pcd(t / 10.0)
        self._interpolated_pcd.points = o3d.utility.Vector3dVector(
            interpolated_pcd.points
        )
        self._interpolated_pcd.colors = o3d.utility.Vector3dVector(
            interpolated_pcd.colors
        )
        print(f"Time: {t}")

    def _next_timestep(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        if self._current_t < 10:
            self._current_t += 1
            assert self._current_t >= 0 and self._current_t <= 10
            self._update_interpolated_pcd(self._current_t)

    def _previous_timestep(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        if self._current_t > 0:
            self._current_t -= 1
            assert self._current_t >= 0 and self._current_t <= 10
            self._update_interpolated_pcd(self._current_t)

    def _set_background(
        self, vis: o3d.visualization.VisualizerWithKeyCallback, color_idx: int
    ):
        assert color_idx >= 0 and color_idx < len(self._colors)
        vis.get_render_option().background_color = self._colors[color_idx]
        if self._visualize_complete_maps:
            self._ref_vis.get_render_option().background_color = self._colors[color_idx]
            self._query_vis.get_render_option().background_color = self._colors[
                color_idx
            ]

    def _change_ref_color(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        self._current_ref_color += 1
        self._current_ref_color = self._current_ref_color % len(self._pcds_colors)
        self._update_ref_pcd()

    def _change_query_color(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        self._current_query_color += 1
        self._current_query_color = self._current_query_color % len(self._pcds_colors)
        self._update_query_pcd()

    def _change_background_color(
        self, vis: o3d.visualization.VisualizerWithKeyCallback
    ):
        self._current_color += 1
        self._current_color = self._current_color % len(self._colors)
        self._set_background(vis, self._current_color)

    def _visualize_associations(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        self._matched_map.visualize(True)

    def _close_visualizer(self, vis):
        vis.destroy_window()
        self._keep_running = False
