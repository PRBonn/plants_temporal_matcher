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
This script is an extension of the approach used in lobefaro2023iros.
In this case we use two maps (already aligned) and we compute a set of
3D correspondences between the two maps.
"""

from typing_extensions import Annotated
import typer
from pathlib import Path
from mapping.matched_map import MatchedMap
from vision.dataset import PATHoBotDataset
from vision.matching import (
    image_to_3d_matches,
    visual_match,
    resolve_image_association,
)
from utils.loading_tools import (
    get_folders_name_from_number,
    load_config,
    load_vpr_results,
    load_kitty_poses,
)


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
    row_number: Annotated[int, typer.Option(help="Number of the row to use")] = 2,
    config_filename: Annotated[
        str,
        typer.Option(
            help="Path to the file containing the configration of the system."
        ),
    ] = "config/config.yaml",
):
    ########## INITIALIZATION ##########
    print("INITIALIZATION")

    # Load the configuration file
    cfg = load_config(config_filename)

    # Initizialization
    dataset_folder = Path(dataset_folder_str)
    n_frames = cfg["general"]["n_frames"]
    n_ref_frames = cfg["general"]["n_ref_frames"]
    n_sensors = cfg["general"]["n_sensors"]

    # Take the name of the datasets folder to use
    ref_folder_name = get_folders_name_from_number(dataset_folder, ref_number)
    query_folder_name = get_folders_name_from_number(dataset_folder, query_number)

    # Initialize folders to use
    ref_folder = dataset_folder / f"{ref_folder_name}/row{row_number}"
    query_folder = dataset_folder / f"{query_folder_name}/row{row_number}"
    ref_poses_filename = ref_folder / "mapping_out" / "mapping_poses.txt"
    query_poses_filename = query_folder / "mapping_out" / f"mapping_poses.txt"

    # Initialize output folder
    output_folder = dataset_folder / "temporal_matcher_out"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_filename = (
        output_folder
        / f"sparse_matches_row{row_number}_{ref_number}_{query_number}.txt"
    )

    # Load the datasets
    ref_dataset = PATHoBotDataset(ref_folder, n_sensors, reference_dataset=True)
    query_dataset = PATHoBotDataset(query_folder, n_sensors)
    if n_ref_frames <= 0:
        n_ref_frames = len(ref_dataset)

    # Load the maps
    matched_map = MatchedMap(ref_folder, query_folder)

    # Load the poses
    ref_poses = load_kitty_poses(ref_poses_filename)
    query_poses = load_kitty_poses(query_poses_filename)

    # Load the vpr results (if required)
    query_idx2ref_idx = load_vpr_results(
        ref_folder / f"vpr_matches_ids_{ref_number}_{query_number}.csv"
    )

    ########## START ##########
    print("STARTED")

    # For each incoming image
    for (
        current_frame_idx,
        (query_frames, query_pose),
    ) in enumerate(query_dataset):
        # LOGGING
        print(f"Frame {current_frame_idx}: ", end="")

        # Stop when we reach the number of frames to deal with
        if n_frames > 0 and current_frame_idx == n_frames:
            break

        # Get the reference frame idx associated to the current frame idx
        ref_frame_idx = resolve_image_association(
            query_idx2ref_idx,
            current_frame_idx,
            query_poses[current_frame_idx],
            ref_poses,
        )
        if ref_frame_idx < 0 or ref_frame_idx >= n_ref_frames:
            print(f"Frame skipped, the corresponding frame idx is not mapped")
            continue

        # Get the poses of query and reference
        query_pose = query_poses[current_frame_idx]
        ref_pose = ref_poses[ref_frame_idx]

        # Get the reference frames
        ref_frames, _ = ref_dataset.get_element(ref_frame_idx)

        # Compute visual matches between the query and found reference from the localizer
        frames_matches, frames_matches_mask = visual_match(query_frames, ref_frames)

        # Translate visual matches into map matches
        image_to_3d_matches(
            frames_matches,
            frames_matches_mask,
            query_frames,
            ref_frames,
            query_pose,
            ref_pose,
            matched_map,
            cfg["sparse_matcher"]["max_dist_associated_points"],
            cfg["point_unprojection"]["min_depth_th"],
            cfg["point_unprojection"]["max_depth_th"],
            cfg["point_unprojection"]["scale_factor"],
            cfg["sparse_matcher"]["unprojected_keypts_nn_th"],
        )

    print(f"# TOTAL MATCHES: {matched_map.size_matches()}")

    # Save the matches
    print("SAVING THE MATCHES")
    matched_map.save_matches(output_filename)

    # Visualization
    print("VISUALIZING THE RESULTS (Press q to exit)")
    matched_map.visualize(True)

    print("FINISHED")


if __name__ == "__main__":
    typer.run(main)
