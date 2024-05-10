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
"""
This script is the code for the paper lobefaro2023iros. 
It takes a reference map and, for each incoming query image 
(image taken some time after the reference sequence)
it extract the point cloud, register it on the reference 
map and computes the 3D point associations.
"""

from typing_extensions import Annotated
import typer
from pathlib import Path
from mapping.map import Map
from mapping.localizer import Localizer
from utils.loading_tools import load_config, get_folders_name_from_number
from vision.dataset import PATHoBotDataset
from vision.matching import visual_match, unproject_matches, render_map_matches
import open3d as o3d


def main(
    dataset_folder_str: Annotated[
        str,
        typer.Argument(
            help="The path to the folder where it is contained the dataset to deal with."
        ),
    ] = "dataset/",
    ref_number: Annotated[
        int, typer.Option(help="Number of the dataset to use as reference")
    ] = 1,
    query_number: Annotated[
        int, typer.Option(help="Number of the dataset to use as query")
    ] = 2,
    row_number: Annotated[int, typer.Option(help="Number of the row to use")] = 3,
    render_matches: bool = typer.Option(
        False, help="Flag to visualize the 3D matches projected on the images."
    ),
    visualize_map: bool = typer.Option(False, help="Flag to visualize the map."),
    config_filename: Annotated[
        str,
        typer.Option(
            help="Path to the file containing the configration of the system."
        ),
    ] = "config/config.yaml",
):
    ########## INITIALIZATION ##########
    print("INITIALIZATION")

    # Load the configration file
    cfg = load_config(config_filename)

    # Initizialization
    dataset_folder = Path(dataset_folder_str)
    n_sensors = cfg["general"]["n_sensors"]

    # Take the name of the datasets folder to use
    ref_folder_name = get_folders_name_from_number(dataset_folder, ref_number)
    query_folder_name = get_folders_name_from_number(dataset_folder, query_number)

    # Initialize folders to use
    ref_folder = dataset_folder / f"{ref_folder_name}/row{row_number}"
    query_folder = dataset_folder / f"{query_folder_name}/row{row_number}"

    # Load the datasets
    ref_dataset = PATHoBotDataset(ref_folder, n_sensors, reference_dataset=True)
    query_dataset = PATHoBotDataset(query_folder, n_sensors)

    # Load the reference map
    world_map = Map(ref_folder)

    # Create the localizer
    localizer = Localizer(
        ref_dataset,
        ref_folder / f"vpr_matches_ids_{ref_number}_{query_number}.csv",
        cfg["general"]["depth_min_th"],
        cfg["general"]["depth_max_th"],
        cfg["general"]["icp_th"],
    )

    ########## START ##########
    print("STARTED")

    # For each incoming image
    for (
        idx,
        (query_frames, query_pose),
    ) in enumerate(query_dataset):
        # LOGGING
        print()
        print(f"Frame {idx}: ", end="")

        if idx < cfg["general"]["first_frame_idx"]:
            print(f"Frame {idx} skipped")
            world_map.integrate_query_pcd(o3d.geometry.PointCloud())
            continue

        # Localize the current query
        ref_frames, query_pose, ref_pose = localizer.localize(
            idx, query_frames, world_map
        )

        # Compute visual matches between the query and found reference from the localizer
        frames_matches, frames_matches_mask = visual_match(query_frames, ref_frames)

        # Translate visual matches into map matches
        unproject_matches(
            frames_matches,
            frames_matches_mask,
            query_frames,
            ref_frames,
            query_pose,
            ref_pose,
            idx,
            world_map,
            cfg["point_unprojection"]["min_depth_th"],
            cfg["point_unprojection"]["max_depth_th"],
            cfg["point_unprojection"]["scale_factor"],
            cfg["sparse_matcher"]["unprojected_keypts_nn_th"],
        )

        # Update the pose of the query
        query_dataset.set_pose(idx, query_pose)

        # Visualize the map matches rendered on the images if required
        if render_matches:
            render_map_matches(
                query_frames,
                ref_frames,
                query_pose,
                ref_pose,
                idx,
                world_map,
                cfg["point_unprojection"]["scale_factor"],
            )

        # Visualize the map if required
        if visualize_map:
            print("Press q to continue (CTRL-c to quit).")
            world_map.visualize_map(True)

    print("FINISHED")


if __name__ == "__main__":
    typer.run(main)
