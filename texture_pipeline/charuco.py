"""Stage 1 — Camera pose estimation.

Two modes:
  1. ChArUco detection: detect the board in each RGB frame, solve PnP.
  2. Pre-computed: load camera poses from camera_pose.json files (robot
     kinematics + hand-eye calibration).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import CameraConfig, CharucoConfig
from .utils import FrameData, invert_pose


# ---------------------------------------------------------------------------
# ChArUco detection
# ---------------------------------------------------------------------------

_ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


def _make_charuco_board(cfg: CharucoConfig) -> tuple:
    """Create a CharucoBoard and ArucoDetector from config."""
    dict_id = _ARUCO_DICT_MAP.get(cfg.dictionary, cv2.aruco.DICT_4X4_100)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard(
        (cfg.squares_x, cfg.squares_y),
        cfg.square_length,
        cfg.marker_length,
        aruco_dict,
    )
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    return board, detector


def detect_charuco_pose(
    rgb_bgr: np.ndarray,
    camera_cfg: CameraConfig,
    charuco_cfg: CharucoConfig,
) -> Optional[np.ndarray]:
    """Detect ChArUco board in one RGB frame and return world-to-camera 4×4.

    The board IS the world frame (board origin = world origin).

    Returns:
        world2cam (4×4) if detection succeeds, else None.
    """
    board, detector = _make_charuco_board(charuco_cfg)
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None

    K = camera_cfg.K
    dist = np.zeros(5, dtype=np.float64)

    n_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board, cameraMatrix=K, distCoeffs=dist
    )

    if n_corners < charuco_cfg.min_corners:
        warnings.warn(f"Only {n_corners} ChArUco corners detected (need ≥{charuco_cfg.min_corners})")
        return None

    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, board, K, dist,
        np.zeros(3), np.zeros(3)
    )
    if not ok:
        return None

    R, _ = cv2.Rodrigues(rvec)
    world2cam = np.eye(4, dtype=np.float64)
    world2cam[:3, :3] = R
    world2cam[:3, 3] = tvec.ravel()
    return world2cam


def detect_all_charuco_poses(
    rgb_images: list[np.ndarray],
    camera_cfg: CameraConfig,
    charuco_cfg: CharucoConfig,
    names: list[str] | None = None,
) -> list[Optional[np.ndarray]]:
    """Run ChArUco detection on all frames.

    Returns a list of world2cam matrices (None where detection fails).
    Aborts if fewer than 2 frames succeed.
    """
    results: list[Optional[np.ndarray]] = []
    for i, rgb in enumerate(rgb_images):
        name = names[i] if names else str(i)
        pose = detect_charuco_pose(rgb, camera_cfg, charuco_cfg)
        if pose is None:
            print(f"  [charuco] Frame {name}: detection FAILED — skipping")
        else:
            print(f"  [charuco] Frame {name}: OK")
        results.append(pose)

    n_ok = sum(1 for p in results if p is not None)
    if n_ok < 2:
        raise RuntimeError(
            f"ChArUco detection succeeded on only {n_ok} frames. "
            "Check image quality and board visibility."
        )
    return results


# ---------------------------------------------------------------------------
# Pre-computed pose loading
# ---------------------------------------------------------------------------

def load_precomputed_pose(json_path: str | Path) -> np.ndarray:
    """Load a pre-computed camera pose from camera_pose.json.

    The JSON stores `transform_matrix` as the **camera-to-world** (cam2world)
    4×4 transform (source_frame = camera, target_frame = world).

    Returns:
        world2cam (4×4) — the standard extrinsic matrix.
    """
    with open(json_path) as f:
        d = json.load(f)
    cam2world = np.array(d["transform_matrix"], dtype=np.float64)
    return invert_pose(cam2world)


def load_precomputed_poses(
    json_paths: list[str | Path],
) -> list[np.ndarray]:
    """Load world2cam poses from a list of camera_pose.json files."""
    poses = []
    for p in json_paths:
        poses.append(load_precomputed_pose(p))
    return poses


# ---------------------------------------------------------------------------
# Frame loading (capture directory format)
# ---------------------------------------------------------------------------

def load_frame_from_capture_dir(
    capture_dir: str | Path,
    camera_name: str = "eye1",
    depth_scale: float = 1.0,
) -> FrameData:
    """Load a FrameData from one capture_* directory.

    Expected layout:
        capture_dir/
            {camera_name}/
                image.jpg
                aligned_depth_to_color.npz   # key 'depth', float32, metres
                camera_pose.json             # cam2world transform
                color_camera_info.json       # intrinsics
    """
    from .config import camera_config_from_info_json

    cam_dir = Path(capture_dir) / camera_name

    rgb_bgr = cv2.imread(str(cam_dir / "image.jpg"))
    if rgb_bgr is None:
        raise FileNotFoundError(f"RGB image not found: {cam_dir / 'image.jpg'}")

    depth_npz = np.load(str(cam_dir / "aligned_depth_to_color.npz"))
    depth_m = depth_npz["depth"].astype(np.float32) * depth_scale

    cam_cfg = camera_config_from_info_json(cam_dir / "color_camera_info.json")

    world2cam = load_precomputed_pose(cam_dir / "camera_pose.json")
    cam2world = invert_pose(world2cam)

    name = Path(capture_dir).name
    return FrameData(
        rgb=rgb_bgr,
        depth_m=depth_m,
        cam2world=cam2world,
        world2cam=world2cam,
        K=cam_cfg.K,
        name=name,
    )


def load_frames_from_dataset(
    dataset_dir: str | Path,
    camera_name: str = "eye1",
    pattern: str = "capture_*",
    max_frames: int | None = None,
) -> list[FrameData]:
    """Load all frames from a dataset directory containing capture_* subdirs."""
    dataset_dir = Path(dataset_dir)
    capture_dirs = sorted(dataset_dir.glob(pattern))
    if not capture_dirs:
        raise FileNotFoundError(f"No captures matching '{pattern}' in {dataset_dir}")

    if max_frames is not None:
        capture_dirs = capture_dirs[:max_frames]

    frames = []
    for d in capture_dirs:
        try:
            frame = load_frame_from_capture_dir(d, camera_name)
            frames.append(frame)
            print(f"  [loader] Loaded {frame.name}")
        except Exception as e:
            print(f"  [loader] Skipping {d.name}: {e}")

    if not frames:
        raise RuntimeError("No frames could be loaded from the dataset.")
    return frames


# ---------------------------------------------------------------------------
# Flat rgb/ depth/ directory format (spec layout)
# ---------------------------------------------------------------------------

def load_frames_from_flat_dirs(
    rgb_dir: str | Path,
    depth_dir: str | Path,
    pose_dir: str | Path | None,
    camera_cfg: CameraConfig,
    charuco_cfg: CharucoConfig | None = None,
) -> list[FrameData]:
    """Load frames from the flat rgb/ depth/ directory layout described in the spec.

    Poses come from either:
      - JSON files in pose_dir (one per frame, same stem as RGB).
      - ChArUco detection if pose_dir is None and charuco_cfg is provided.
    """
    rgb_dir = Path(rgb_dir)
    depth_dir = Path(depth_dir)

    rgb_paths = sorted(rgb_dir.glob("*.png")) + sorted(rgb_dir.glob("*.jpg"))
    rgb_paths = sorted(set(rgb_paths))

    depth_paths = sorted(depth_dir.glob("*.png")) + sorted(depth_dir.glob("*.npz"))
    depth_paths = sorted(set(depth_paths))

    if len(rgb_paths) != len(depth_paths):
        raise ValueError(
            f"RGB count ({len(rgb_paths)}) ≠ depth count ({len(depth_paths)})"
        )

    frames: list[FrameData] = []
    for rgb_p, dep_p in zip(rgb_paths, depth_paths):
        rgb = cv2.imread(str(rgb_p))
        if rgb is None:
            print(f"  [loader] Skipping {rgb_p.name}: cannot read image")
            continue

        # Load depth
        if dep_p.suffix == ".npz":
            depth_raw = np.load(str(dep_p))["depth"].astype(np.float32)
            depth_m = depth_raw * camera_cfg.depth_scale
        else:
            # 16-bit PNG in mm (spec default)
            depth_raw = cv2.imread(str(dep_p), cv2.IMREAD_ANYDEPTH).astype(np.float32)
            depth_m = depth_raw * camera_cfg.depth_scale

        frames.append(FrameData(
            rgb=rgb,
            depth_m=depth_m,
            cam2world=np.eye(4),  # placeholder
            world2cam=np.eye(4),
            K=camera_cfg.K,
            name=rgb_p.stem,
        ))

    # Determine poses
    if pose_dir is not None:
        pose_dir = Path(pose_dir)
        for frame in frames:
            json_p = pose_dir / (frame.name + ".json")
            if json_p.exists():
                world2cam = load_precomputed_pose(json_p)
                frame.world2cam = world2cam
                frame.cam2world = invert_pose(world2cam)
            else:
                print(f"  [loader] No pose file for {frame.name}")

    elif charuco_cfg is not None:
        rgbs = [f.rgb for f in frames]
        names = [f.name for f in frames]
        poses = detect_all_charuco_poses(rgbs, camera_cfg, charuco_cfg, names)
        for frame, pose in zip(frames, poses):
            if pose is not None:
                frame.world2cam = pose
                frame.cam2world = invert_pose(pose)

    return frames
