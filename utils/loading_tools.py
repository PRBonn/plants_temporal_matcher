import os
from pathlib import Path
from typing import Dict, Tuple, List, Any
import csv
import numpy as np
import open3d as o3d
from pathlib import Path
import open3d as o3d
from utils.geometry_tools import transformation_matrix_from_t_and_q
import yaml
import cv2


def load_config(cfg_filename) -> Dict[str, Any]:
    with open(cfg_filename) as cfg_file:
        data = yaml.safe_load(cfg_file)
    return data


def load_camera_paramaters(
    filename: str, img_width: int = 1280, img_height: int = 720
) -> Tuple[o3d.camera.PinholeCameraIntrinsic, np.ndarray]:
    with open(filename, "r") as file:
        yaml_file = yaml.load(file, Loader=yaml.FullLoader)
    intrinsics = yaml_file["intrinsics"]
    extrinsics = yaml_file["extrinsics"]

    return o3d.camera.PinholeCameraIntrinsic(
        img_width,
        img_height,
        intrinsics[0][0],
        intrinsics[1][1],
        intrinsics[0][2],
        intrinsics[1][2],
    ), transformation_matrix_from_t_and_q(
        [extrinsics[0][0], extrinsics[0][1], extrinsics[0][2]],
        [extrinsics[1][3], extrinsics[1][0], extrinsics[1][1], extrinsics[1][2]],
    )


def load_vpr_results(filename: Path) -> Dict[int, int]:
    query2ref = {}

    with open(str(filename), "r") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            query2ref[int(row[0])] = int(row[1])

    return query2ref


def load_kitty_poses(filename: Path) -> np.ndarray:
    poses = np.loadtxt(str(filename), delimiter=" ")
    n = poses.shape[0]
    poses = np.concatenate(
        (poses, np.zeros((n, 3), dtype=np.float32), np.ones((n, 1), dtype=np.float32)),
        axis=1,
    )
    poses = poses.reshape((n, 4, 4))  # [N, 4, 4]
    return poses


def load_point_cloud(filename: Path) -> o3d.geometry.PointCloud:
    return o3d.io.read_point_cloud(str(filename))


def list_folders(folder: Path) -> List[str]:
    all_files = os.listdir(str(folder))
    return [el for el in all_files if os.path.isdir(str(folder / el))]


def load_descriptors(filename: str) -> Tuple[Tuple[cv2.KeyPoint], np.ndarray]:
    raw_kp_desc = np.load(filename)
    kps = ()
    descs = []
    for el in raw_kp_desc:
        kp = cv2.KeyPoint()
        kp.pt = (el[0], el[1])
        kps += (kp,)
        descs.append(el[2:])
    return kps, np.array(descs, dtype=np.float32)


def get_folders_name_from_number(dataset_folder: Path, folder_number: int) -> str:
    # Search for a folder in dataset_folder with the letter that starts with folder_number
    folder_name = ""
    for path in dataset_folder.iterdir():
        if path.is_dir() and path.name[0] == str(folder_number):
            folder_name = path.name
            break

    # If we didn't find one exit with error
    if folder_name == "":
        print(f"ERROR: No dataset found with number {folder_number}")
        exit(1)

    return folder_name
