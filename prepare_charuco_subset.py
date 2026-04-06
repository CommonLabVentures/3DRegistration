#!/usr/bin/env python3
"""Estimate ChArUco poses, select oblique views, and build a subset dataset."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass
class CameraIntrinsics:
    fx: float = 605.1414184570312
    fy: float = 604.7616577148438
    cx: float = 417.1040954589844
    cy: float = 250.10911560058594
    width: int = 848
    height: int = 480

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


@dataclass
class CharucoSpec:
    squares_x: int = 12
    squares_y: int = 8
    marker_square_ratio: float = 0.75
    dictionary_name: str = "DICT_4X4_100"
    min_corners: int = 6


@dataclass
class DetectionResult:
    frame_id: str
    num_charuco_corners: int
    charuco_corners: np.ndarray
    charuco_ids: np.ndarray


@dataclass
class PoseEstimate:
    frame_id: str
    num_charuco_corners: int
    square_length_m: float
    world2cam: list[list[float]]
    cam2world: list[list[float]]
    camera_position_world_m: list[float]
    camera_forward_world: list[float]
    optical_axis_angle_deg: float


_ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate ChArUco poses and create an optical-axis-filtered subset dataset."
    )
    parser.add_argument("--data-dir", default="clean_dataset", help="Input clean dataset directory.")
    parser.add_argument(
        "--output-dir",
        default="subset_dataset",
        help="Output directory for subset rgb/depth/poses and contact sheet.",
    )
    parser.add_argument(
        "--min-optical-axis-angle-deg",
        "--min-oblique-angle-deg",
        dest="min_optical_axis_angle_deg",
        type=float,
        default=20.0,
        help="Minimum off-normal angle between the camera optical axis and board normal.",
    )
    parser.add_argument(
        "--max-views",
        type=int,
        default=None,
        help="Optional cap on number of selected views, ranked by optical-axis angle.",
    )
    parser.add_argument(
        "--dedupe-position-tol-m",
        type=float,
        default=0.01,
        help="Merge selected views whose solved camera positions are within this tolerance.",
    )
    parser.add_argument(
        "--sheet-columns",
        type=int,
        default=2,
        help="Number of columns in the subset contact sheet.",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=380,
        help="Width of each RGB/depth tile in the preview panels.",
    )
    return parser


def make_board(spec: CharucoSpec, square_length_m: float = 1.0):
    dict_id = _ARUCO_DICT_MAP[spec.dictionary_name]
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        square_length_m,
        square_length_m * spec.marker_square_ratio,
        aruco_dict,
    )
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    return board, detector


def invert_pose(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def detect_charuco(
    image_bgr: np.ndarray,
    intr: CameraIntrinsics,
    spec: CharucoSpec,
) -> tuple[DetectionResult | None, np.ndarray]:
    board, detector = make_board(spec, square_length_m=1.0)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    vis = image_bgr.copy()

    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None, vis

    cv2.aruco.drawDetectedMarkers(vis, corners, ids)

    dist = np.zeros(5, dtype=np.float64)
    n_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners,
        ids,
        gray,
        board,
        cameraMatrix=intr.K,
        distCoeffs=dist,
    )
    if n_corners < spec.min_corners:
        return None, vis

    cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
    return (
        DetectionResult(
            frame_id="",
            num_charuco_corners=int(n_corners),
            charuco_corners=charuco_corners.copy(),
            charuco_ids=charuco_ids.copy(),
        ),
        vis,
    )


def sample_depth(depth_m: np.ndarray, u: float, v: float, radius: int = 1) -> float:
    x = int(round(u))
    y = int(round(v))
    x0 = max(0, x - radius)
    x1 = min(depth_m.shape[1], x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(depth_m.shape[0], y + radius + 1)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))


def backproject_pixel(intr: CameraIntrinsics, u: float, v: float, z: float) -> np.ndarray:
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    return np.array([x, y, z], dtype=np.float64)


def estimate_square_length_m(
    detections: list[DetectionResult],
    depth_dir: Path,
    intr: CameraIntrinsics,
    spec: CharucoSpec,
) -> float:
    stride = spec.squares_x - 1
    distances: list[float] = []

    for det in detections:
        depth = np.load(str(depth_dir / f"{det.frame_id}.npz"))["depth"].astype(np.float32)
        points_by_id: dict[int, np.ndarray] = {}
        for corner, corner_id in zip(det.charuco_corners.reshape(-1, 2), det.charuco_ids.ravel()):
            z = sample_depth(depth, float(corner[0]), float(corner[1]))
            if z <= 0:
                continue
            points_by_id[int(corner_id)] = backproject_pixel(intr, float(corner[0]), float(corner[1]), z)

        for corner_id, pt in points_by_id.items():
            col = corner_id % stride
            row = corner_id // stride
            right_id = corner_id + 1
            down_id = corner_id + stride

            if col + 1 < stride and right_id in points_by_id:
                d = float(np.linalg.norm(pt - points_by_id[right_id]))
                if 0.01 <= d <= 0.08:
                    distances.append(d)
            if row + 1 < (spec.squares_y - 1) and down_id in points_by_id:
                d = float(np.linalg.norm(pt - points_by_id[down_id]))
                if 0.01 <= d <= 0.08:
                    distances.append(d)

    if not distances:
        raise RuntimeError("Could not estimate board square length from depth.")
    return float(np.median(np.array(distances, dtype=np.float64)))


def solve_pose_from_detection(
    det: DetectionResult,
    intr: CameraIntrinsics,
    spec: CharucoSpec,
    square_length_m: float,
) -> tuple[PoseEstimate, np.ndarray, np.ndarray]:
    board, _ = make_board(spec, square_length_m=square_length_m)
    obj_pts_all = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    obj_pts = obj_pts_all[det.charuco_ids.ravel()]
    img_pts = det.charuco_corners.reshape(-1, 2).astype(np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        intr.K,
        np.zeros(5, dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError(f"solvePnP failed for frame {det.frame_id}")

    R, _ = cv2.Rodrigues(rvec)
    world2cam = np.eye(4, dtype=np.float64)
    world2cam[:3, :3] = R
    world2cam[:3, 3] = tvec.ravel()
    cam2world = invert_pose(world2cam)
    cam_pos = cam2world[:3, 3]

    # OpenCV camera coordinates look along +Z. Transform that forward axis into
    # the ChArUco/world frame and compare it against the board normal.
    camera_forward_world = cam2world[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    camera_forward_world /= max(np.linalg.norm(camera_forward_world), 1e-9)
    board_normal_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    optical_axis_angle_deg = float(
        np.degrees(
            np.arccos(
                np.clip(abs(np.dot(camera_forward_world, board_normal_world)), -1.0, 1.0)
            )
        )
    )

    pose = PoseEstimate(
        frame_id=det.frame_id,
        num_charuco_corners=det.num_charuco_corners,
        square_length_m=square_length_m,
        world2cam=world2cam.tolist(),
        cam2world=cam2world.tolist(),
        camera_position_world_m=cam_pos.tolist(),
        camera_forward_world=camera_forward_world.tolist(),
        optical_axis_angle_deg=optical_axis_angle_deg,
    )
    return pose, rvec, tvec


def colorize_depth(depth_m: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    valid_fraction = float(np.count_nonzero(valid)) / float(depth_m.size)

    out = np.full((*depth_m.shape, 3), 30, dtype=np.uint8)
    if not np.any(valid):
        return out, 0.0, 0.0, valid_fraction

    values = depth_m[valid]
    d_min = float(np.percentile(values, 2.0))
    d_max = float(np.percentile(values, 98.0))
    if d_max <= d_min:
        d_min = float(values.min())
        d_max = float(values.max())
    if d_max <= d_min:
        d_max = d_min + 1e-3

    norm = np.clip((depth_m - d_min) / (d_max - d_min), 0.0, 1.0)
    out = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[~valid] = (30, 30, 30)
    return out, d_min, d_max, valid_fraction


def resize_width(image_bgr: np.ndarray, target_width: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    scale = target_width / float(w)
    target_height = max(1, int(round(h * scale)))
    return cv2.resize(image_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)


def make_preview_panel(
    frame_id: str,
    rgb_bgr: np.ndarray,
    depth_bgr: np.ndarray,
    pose: PoseEstimate,
    depth_min_m: float,
    depth_max_m: float,
    valid_fraction: float,
    tile_width: int,
) -> np.ndarray:
    rgb_tile = resize_width(rgb_bgr, tile_width)
    depth_tile = resize_width(depth_bgr, tile_width)
    target_h = min(rgb_tile.shape[0], depth_tile.shape[0])
    rgb_tile = cv2.resize(rgb_tile, (tile_width, target_h), interpolation=cv2.INTER_AREA)
    depth_tile = cv2.resize(depth_tile, (tile_width, target_h), interpolation=cv2.INTER_AREA)
    spacer = np.full((target_h, 12, 3), 18, dtype=np.uint8)
    header = np.full((32, tile_width * 2 + 12, 3), 245, dtype=np.uint8)
    footer = np.full((92, tile_width * 2 + 12, 3), 245, dtype=np.uint8)

    cv2.putText(header, "RGB", (16, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(
        header,
        "False-Colored Depth",
        (tile_width + 28, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(footer, frame_id, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(
        footer,
        f"Optical-axis angle: {pose.optical_axis_angle_deg:.1f} deg   ChArUco corners: {pose.num_charuco_corners}",
        (16, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        footer,
        f"Depth range: {depth_min_m:.3f}m to {depth_max_m:.3f}m   Valid depth: {valid_fraction * 100.0:.1f}%",
        (16, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )

    body = np.hstack([rgb_tile, spacer, depth_tile])
    return np.vstack([header, body, footer])


def write_contact_sheet(panels: list[np.ndarray], output_path: Path, columns: int) -> None:
    columns = max(1, columns)
    panel_h, panel_w = panels[0].shape[:2]
    rows = math.ceil(len(panels) / columns)
    canvas = np.full((rows * panel_h, columns * panel_w, 3), 255, dtype=np.uint8)
    for idx, panel in enumerate(panels):
        row = idx // columns
        col = idx % columns
        y0 = row * panel_h
        x0 = col * panel_w
        canvas[y0:y0 + panel_h, x0:x0 + panel_w] = panel
    Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(output_path)


def dedupe_by_camera_position(
    entries: list[tuple[PoseEstimate, np.ndarray]],
    tol_m: float,
) -> list[tuple[PoseEstimate, np.ndarray]]:
    kept: list[tuple[PoseEstimate, np.ndarray]] = []
    for pose, panel in entries:
        pos = np.array(pose.camera_position_world_m, dtype=np.float64)
        duplicate = False
        for kept_pose, _ in kept:
            kept_pos = np.array(kept_pose.camera_position_world_m, dtype=np.float64)
            if np.linalg.norm(pos - kept_pos) <= tol_m:
                duplicate = True
                break
        if not duplicate:
            kept.append((pose, panel))
    return kept


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    args = build_parser().parse_args()

    data_dir = Path(args.data_dir)
    rgb_dir = data_dir / "rgb"
    depth_dir = data_dir / "depth"
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    poses_dir = output_dir / "poses"
    subset_rgb_dir = output_dir / "rgb"
    subset_depth_dir = output_dir / "depth"
    overlays_dir = output_dir / "overlays"

    intr = CameraIntrinsics()
    spec = CharucoSpec()

    rgb_paths = sorted(rgb_dir.glob("*.jpg"))
    if not rgb_paths:
        raise SystemExit(f"No RGB files found in {rgb_dir}")

    all_results: list[dict] = []
    detections: list[DetectionResult] = []

    for rgb_path in rgb_paths:
        frame_id = rgb_path.stem
        depth_path = depth_dir / f"{frame_id}.npz"
        if not depth_path.exists():
            print(f"[pose] Skip {frame_id}: missing depth file")
            continue

        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            print(f"[pose] Skip {frame_id}: cannot read RGB")
            continue

        det, overlay = detect_charuco(rgb_bgr, intr, spec)
        overlay_path = overlays_dir / f"{frame_id}.png"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(overlay_path), overlay)

        record: dict = {
            "frame_id": frame_id,
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "overlay_path": str(overlay_path),
            "pose_detected": det is not None,
        }

        if det is None:
            all_results.append(record)
            print(f"[pose] {frame_id}: detection failed")
            continue

        det.frame_id = frame_id
        detections.append(det)
        print(f"[pose] {frame_id}: detected {det.num_charuco_corners} ChArUco corners")

    if not detections:
        raise SystemExit("No valid ChArUco detections found in the dataset.")

    square_length_m = estimate_square_length_m(detections, depth_dir, intr, spec)
    marker_length_m = square_length_m * spec.marker_square_ratio
    print(
        f"[pose] Estimated square length = {square_length_m:.5f} m "
        f"(marker length assumed {marker_length_m:.5f} m)"
    )

    selected_entries: list[tuple[PoseEstimate, np.ndarray]] = []
    for det in detections:
        frame_id = det.frame_id
        rgb_path = rgb_dir / f"{frame_id}.jpg"
        depth_path = depth_dir / f"{frame_id}.npz"
        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        pose, rvec, tvec = solve_pose_from_detection(det, intr, spec, square_length_m)

        all_results.append(
            {
                "frame_id": frame_id,
                "rgb_path": str(rgb_path),
                "depth_path": str(depth_path),
                "overlay_path": str(overlays_dir / f"{frame_id}.png"),
                "pose_detected": True,
                **asdict(pose),
            }
        )

        overlay = cv2.imread(str(overlays_dir / f"{frame_id}.png"), cv2.IMREAD_COLOR)
        if overlay is not None:
            cv2.drawFrameAxes(
                overlay,
                intr.K,
                np.zeros(5, dtype=np.float64),
                rvec,
                tvec,
                square_length_m * 2.0,
            )
            cv2.imwrite(str(overlays_dir / f"{frame_id}.png"), overlay)

        print(
            f"[pose] {frame_id}: optical_axis_angle={pose.optical_axis_angle_deg:.1f} deg, "
            f"camera_z={pose.camera_position_world_m[2]:.3f} m"
        )

        if pose.optical_axis_angle_deg < args.min_optical_axis_angle_deg:
            continue

        depth = np.load(str(depth_path))["depth"].astype(np.float32)
        depth_bgr, d_min, d_max, valid_fraction = colorize_depth(depth)
        panel = make_preview_panel(
            frame_id,
            rgb_bgr,
            depth_bgr,
            pose,
            d_min,
            d_max,
            valid_fraction,
            args.tile_width,
        )
        selected_entries.append((pose, panel))

    if not selected_entries:
        raise SystemExit(
            f"No views survived min optical-axis angle {args.min_optical_axis_angle_deg:.1f} deg. "
            "Lower the threshold and try again."
        )

    if args.max_views is not None and len(selected_entries) > args.max_views:
        selected_entries = sorted(
            selected_entries,
            key=lambda item: item[0].optical_axis_angle_deg,
            reverse=True,
        )[: args.max_views]

    selected_entries.sort(key=lambda item: item[0].frame_id)
    selected_entries = dedupe_by_camera_position(selected_entries, args.dedupe_position_tol_m)
    selection_records = [item[0] for item in selected_entries]
    selected_panels = [item[1] for item in selected_entries]
    selected_ids = [p.frame_id for p in selection_records]

    output_dir.mkdir(parents=True, exist_ok=True)
    poses_dir.mkdir(parents=True, exist_ok=True)
    subset_rgb_dir.mkdir(parents=True, exist_ok=True)
    subset_depth_dir.mkdir(parents=True, exist_ok=True)

    for pose in selection_records:
        frame_id = pose.frame_id
        copy_file(rgb_dir / f"{frame_id}.jpg", subset_rgb_dir / f"{frame_id}.jpg")
        copy_file(depth_dir / f"{frame_id}.npz", subset_depth_dir / f"{frame_id}.npz")
        with open(poses_dir / f"{frame_id}.json", "w", encoding="utf-8") as f:
            json.dump(asdict(pose), f, indent=2)

    with open(output_dir / "all_pose_estimates.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    with open(output_dir / "selected_views.json", "w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in selection_records], f, indent=2)

    metadata = {
        "camera_intrinsics": asdict(intr),
        "charuco_spec": {
            **asdict(spec),
            "estimated_square_length_m": square_length_m,
            "estimated_marker_length_m": marker_length_m,
        },
        "selection": {
            "min_optical_axis_angle_deg": args.min_optical_axis_angle_deg,
            "max_views": args.max_views,
            "dedupe_position_tol_m": args.dedupe_position_tol_m,
            "selected_frame_ids": selected_ids,
        },
    }
    with open(output_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    write_contact_sheet(selected_panels, output_dir / "contact_sheet.png", args.sheet_columns)

    print(f"[subset] Selected {len(selected_ids)} view(s): {', '.join(selected_ids)}")
    print(f"[subset] Output written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
