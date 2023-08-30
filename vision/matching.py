from typing import List, Tuple, Dict
from mapping.matched_map import MatchedMap
from vision.frame import Frame
from vision.vision_tools import point_image_to_point3d, point3d_to_point_image
from utils.visualization_tools import visualize_visual_matches
import cv2
import numpy as np
from mapping.map import Map
import math


def resolve_image_association(
    query_idx2ref_idx, query_idx: int, query_pose, ref_poses, th: float = 0.01
) -> int:
    # Solve with VPR results
    if query_idx2ref_idx is not None:
        return query_idx2ref_idx[query_idx]

    best_dist = np.inf
    best_ref_idx = -1
    for ref_idx, ref_pose in enumerate(ref_poses):
        current_dist = abs(query_pose[0][3] - ref_pose[0][3])
        if current_dist < th and current_dist < best_dist:
            print(f"Query: {query_pose[0][3]}, Ref: {ref_pose[0][3]}")
            print(f"Ref+1: {ref_poses[ref_idx+1][0][3]}")
            print(f"Ref+2: {ref_poses[ref_idx+2][0][3]}")
            best_dist = current_dist
            best_ref_idx = ref_idx
    return best_ref_idx


def visual_match(
    query_frames: List[Frame], ref_frames: List[Frame], visualize: bool = False
) -> Tuple[List[Tuple[cv2.DMatch]], List[List[int]]]:
    # Create the matcher
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Compute matches for each cam
    frames_matches = []
    frames_matches_mask = []
    for query_frame, ref_frame in zip(query_frames, ref_frames):
        matches = matcher.match(query_frame.descriptors, ref_frame.descriptors)
        query_matched_keypoints = np.zeros((len(matches), 2), dtype=np.float32)
        reference_matched_keypoints = np.zeros((len(matches), 2), dtype=np.float32)
        for i, match in enumerate(matches):
            query_matched_keypoints[i, :] = query_frame.keypoints[match.queryIdx].pt
            reference_matched_keypoints[i, :] = ref_frame.keypoints[match.trainIdx].pt

        # Find the homography (ransac is used inside) to get inlier matches
        _, inliers_mask = cv2.findHomography(
            query_matched_keypoints, reference_matched_keypoints, cv2.USAC_ACCURATE, 30
        )
        inliers_mask = inliers_mask.ravel().tolist()

        frames_matches.append(matches)
        frames_matches_mask.append(inliers_mask)

        if visualize:
            visualize_visual_matches(
                query_frame.rgb_img,
                query_frame.keypoints,
                ref_frame.rgb_img,
                ref_frame.keypoints,
                matches,
                inliers_mask,
            )

    return frames_matches, frames_matches_mask


def image_to_3d_matches(
    frames_matches: List[Tuple[cv2.DMatch]],
    frames_matches_mask: List[List[int]],
    query_frames: List[Frame],
    ref_frames: List[Frame],
    query_pose: np.ndarray,
    ref_pose: np.ndarray,
    matched_map: MatchedMap,
    distance_th: float,
    min_depth_th: float,
    max_depth_th: float,
    scale_factor: float,
    unprojected_keypts_nn_th: float,
) -> None:
    # Initialization
    n_matches = 0

    for matches, matches_mask, query_frame, ref_frame in zip(
        frames_matches, frames_matches_mask, query_frames, ref_frames
    ):
        # Get the points and KDTrees from the map
        query_points = matched_map.get_query_pcd().points
        ref_points = matched_map.get_ref_pcd().points
        query_pcd_kdtree = matched_map.get_query_pcd_kdtree()
        ref_pcd_kdtree = matched_map.get_ref_pcd_kdtree()

        # Compute the pose from which the measurements are taken
        query_cam_pose = query_pose @ query_frame.camera_extrinsics
        ref_cam_pose = ref_pose @ ref_frame.camera_extrinsics

        # For each match
        map_matches = []
        dists_matches = []
        for match, inlier in zip(matches, matches_mask):
            # Discard it if it's not an inlier
            if not inlier:
                continue

            # Get the image points of this match
            query_pt = query_frame.keypoints[match.queryIdx].pt
            ref_pt = ref_frame.keypoints[match.trainIdx].pt

            # Get the ids of the 3D points corresponding to this image points
            succ, query_pcd_idx = point_image_to_point3d(
                query_frame.depth_img,
                query_cam_pose,
                query_frame.camera_intrinsics,
                int(query_pt[1]),
                int(query_pt[0]),
                query_pcd_kdtree,
                min_depth_th,
                max_depth_th,
                scale_factor,
                unprojected_keypts_nn_th,
            )
            if not succ:  # Check that we have a valid 3D point
                continue
            succ, ref_pcd_idx = point_image_to_point3d(
                ref_frame.depth_img,
                ref_cam_pose,
                ref_frame.camera_intrinsics,
                int(ref_pt[1]),
                int(ref_pt[0]),
                ref_pcd_kdtree,
                min_depth_th,
                max_depth_th,
                scale_factor,
                unprojected_keypts_nn_th,
            )
            if not succ:  # Check that we have a valid 3D point
                continue

            # Check that the distance of the two points is under a threshold
            current_dist = math.dist(
                query_points[query_pcd_idx], ref_points[ref_pcd_idx]
            )
            if current_dist > distance_th:
                continue

            # Save the current match and corresopnding points distance
            map_matches.append((query_pcd_idx, ref_pcd_idx))
            dists_matches.append(current_dist)

        # Insert the matches into the map filtered by distance
        threshold = np.mean(dists_matches) + 1 * np.std(dists_matches)
        filtered_matches = [
            match for match, dist in zip(map_matches, dists_matches) if dist < threshold
        ]
        matched_map.add_matches(filtered_matches)
        n_matches += len(filtered_matches)

    print(f"# MATCHES {n_matches}")

    return


def unproject_matches(
    frames_matches: List[Tuple[cv2.DMatch]],
    frames_matches_mask: List[List[int]],
    query_frames: List[Frame],
    ref_frames: List[Frame],
    query_pose: np.ndarray,
    ref_pose: np.ndarray,
    query_idx: int,
    world_map: Map,
    min_depth_th: float,
    max_depth_th: float,
    scale_factor: float,
    nn_threshold: float,
) -> None:
    for matches, matches_mask, query_frame, ref_frame in zip(
        frames_matches, frames_matches_mask, query_frames, ref_frames
    ):
        n_matches = unproject(
            matches,
            matches_mask,
            query_frame,
            ref_frame,
            query_pose,
            ref_pose,
            query_idx,
            world_map,
            0.05,
            min_depth_th,
            max_depth_th,
            scale_factor,
            nn_threshold,
        )

        print(f"# MATCHES {n_matches}")

    return


def unproject(
    matches: Tuple[cv2.DMatch],
    matches_mask: List[int],
    query_frame: Frame,
    ref_frame: Frame,
    query_pose: np.ndarray,
    ref_pose: np.ndarray,
    query_idx: int,
    world_map: Map,
    distance_th: float,
    min_depth_th: float,
    max_depth_th: float,
    scale_factor: float,
    unprojected_keypts_nn_th: float,
) -> int:
    # Get the points and KDTrees from the map
    query_points = world_map.get_query_pcd(query_idx).points
    ref_points = world_map.get_ref_pcd().points
    query_pcd_kdtree = world_map.get_query_pcd_kdtree(query_idx)
    ref_pcd_kdtree = world_map.get_ref_pcd_kdtree()

    # Compute the pose from which the measurements are taken
    query_cam_pose = query_pose @ query_frame.camera_extrinsics
    ref_cam_pose = ref_pose @ ref_frame.camera_extrinsics

    # For each match
    map_matches = []
    for match, inlier in zip(matches, matches_mask):
        # Discard it if it's not an inlier
        if not inlier:
            continue

        # Get the image points of this match
        query_pt = query_frame.keypoints[match.queryIdx].pt
        ref_pt = ref_frame.keypoints[match.trainIdx].pt

        # Get the ids of the 3D points corresponding to this image points
        succ, query_pcd_idx = point_image_to_point3d(
            query_frame.depth_img,
            query_cam_pose,
            query_frame.camera_intrinsics,
            int(query_pt[1]),
            int(query_pt[0]),
            query_pcd_kdtree,
            min_depth_th,
            max_depth_th,
            scale_factor,
            unprojected_keypts_nn_th,
        )
        if not succ:  # Check that we have a valid 3D point
            continue
        succ, ref_pcd_idx = point_image_to_point3d(
            ref_frame.depth_img,
            ref_cam_pose,
            ref_frame.camera_intrinsics,
            int(ref_pt[1]),
            int(ref_pt[0]),
            ref_pcd_kdtree,
            min_depth_th,
            max_depth_th,
            scale_factor,
            unprojected_keypts_nn_th,
        )
        if not succ:  # Check that we have a valid 3D point
            continue

        # Check that the distance of the two points is under a threshold
        if (
            math.dist(query_points[query_pcd_idx], ref_points[ref_pcd_idx])
            > distance_th
        ):
            continue

        # Save the current match
        map_matches.append((query_pcd_idx, ref_pcd_idx))

    # Insert the matches into the map
    world_map.add_matches(query_idx, map_matches)

    return len(map_matches)


def render_map_matches(
    query_frames: List[Frame],
    ref_frames: List[Frame],
    query_pose: np.ndarray,
    ref_pose: np.ndarray,
    query_idx: int,
    world_map: Map,
    scale_factor: float,
) -> None:
    for frame_idx, (query_frame, ref_frame) in enumerate(zip(query_frames, ref_frames)):
        query_kps, ref_kps, matches = project(
            query_frame,
            ref_frame,
            query_pose,
            ref_pose,
            query_idx,
            frame_idx,
            world_map,
            scale_factor,
        )
        inliers_mask = [1 for el in matches]
        visualize_visual_matches(
            query_frame.rgb_img,
            query_kps,
            ref_frame.rgb_img,
            ref_kps,
            matches,
            inliers_mask,
        )
    input("Close the image and press Enter to continue (CTRL-c to quit).")


def project(
    query_frame: Frame,
    ref_frame: Frame,
    query_pose: np.ndarray,
    ref_pose: np.ndarray,
    query_idx: int,
    frame_idx: int,
    world_map: Map,
    scale_factor: float,
) -> Tuple[Tuple[cv2.KeyPoint], Tuple[cv2.KeyPoint], Tuple[cv2.DMatch]]:
    # Initialize some variable
    query_kps = ()
    ref_kps = ()
    matches = ()

    # Precompute poses
    query_cam_pose_inv = np.linalg.inv(query_pose @ query_frame.camera_extrinsics)
    ref_cam_pose_inv = np.linalg.inv(ref_pose @ ref_frame.camera_extrinsics)

    # Get the map matches corresponding to query_idx and the points
    map_matches = world_map.get_matches(query_idx, frame_idx)
    query_points = world_map.get_query_pcd(query_idx).points
    ref_points = world_map.get_ref_pcd().points

    # For each match in map
    current_match_idx = 0
    for query_point_id, ref_point_id in map_matches:
        # Project the query point
        query_kp = cv2.KeyPoint()
        query_kp.pt = point3d_to_point_image(
            query_points[query_point_id],
            query_cam_pose_inv,
            query_frame.camera_intrinsics,
            scale_factor,
        )
        query_kps += (query_kp,)

        # Project the refernce point
        ref_kp = cv2.KeyPoint()
        ref_kp.pt = point3d_to_point_image(
            ref_points[ref_point_id],
            ref_cam_pose_inv,
            ref_frame.camera_intrinsics,
            scale_factor,
        )
        ref_kps += (ref_kp,)

        # Create the match
        new_match = cv2.DMatch()
        new_match.queryIdx = current_match_idx
        new_match.trainIdx = current_match_idx
        matches += (new_match,)
        current_match_idx += 1

    return query_kps, ref_kps, matches
