#!/usr/bin/env python3
"""Extract battery-only RGB cutouts from the selected ChArUco subset."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

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
    squares_x: int
    squares_y: int
    square_length_m: float


@dataclass
class SegmentationRecord:
    frame_id: str
    bbox_xywh: list[int]
    mask_area_px: int
    seed_area_px: int
    fallback_used: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract battery-only cutouts from subset_dataset.")
    parser.add_argument("--data-dir", default="subset_dataset", help="Subset dataset directory.")
    parser.add_argument(
        "--output-dir",
        default="subset_dataset/battery_only",
        help="Output directory for masks, cutouts, and diagnostics.",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=120,
        help="Minimum connected-component area for the final battery mask.",
    )
    parser.add_argument(
        "--sheet-columns",
        type=int,
        default=2,
        help="Number of columns in the output contact sheet.",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=320,
        help="Tile width used in the diagnostic sheet.",
    )
    return parser


def load_metadata(data_dir: Path) -> tuple[CameraIntrinsics, CharucoSpec]:
    metadata = json.loads((data_dir / "dataset_metadata.json").read_text(encoding="utf-8"))
    intr = CameraIntrinsics(**metadata["camera_intrinsics"])
    spec = metadata["charuco_spec"]
    charuco = CharucoSpec(
        squares_x=int(spec["squares_x"]),
        squares_y=int(spec["squares_y"]),
        square_length_m=float(spec["estimated_square_length_m"]),
    )
    return intr, charuco


def load_pose(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return np.array(payload["world2cam"], dtype=np.float64), np.array(payload["cam2world"], dtype=np.float64)


def project_points(points_world: np.ndarray, world2cam: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    points_cam = (world2cam[:3, :3] @ points_world.T).T + world2cam[:3, 3]
    uvw = (intr.K @ points_cam.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def board_polygon_mask(shape: tuple[int, int], intr: CameraIntrinsics, spec: CharucoSpec, world2cam: np.ndarray) -> np.ndarray:
    board_corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [spec.squares_x * spec.square_length_m, 0.0, 0.0],
            [spec.squares_x * spec.square_length_m, spec.squares_y * spec.square_length_m, 0.0],
            [0.0, spec.squares_y * spec.square_length_m, 0.0],
        ],
        dtype=np.float64,
    )
    uv = project_points(board_corners, world2cam, intr)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(uv).astype(np.int32), 255)
    # Pull the mask inwards to avoid tape and edge-depth artifacts.
    mask = cv2.erode(mask, np.ones((9, 9), dtype=np.uint8))
    return mask


def estimate_board_depth(depth_m: np.ndarray, intr: CameraIntrinsics, world2cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth_m.shape
    ys, xs = np.indices((h, w))
    rays = np.stack(
        [
            (xs - intr.cx) / intr.fx,
            (ys - intr.cy) / intr.fy,
            np.ones((h, w), dtype=np.float32),
        ],
        axis=-1,
    ).astype(np.float64)
    board_normal_cam = world2cam[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    plane_offset = float(board_normal_cam @ world2cam[:3, 3])
    denom = rays @ board_normal_cam
    valid = denom > 1e-6
    expected = np.full((h, w), np.nan, dtype=np.float32)
    expected[valid] = (plane_offset / denom[valid]).astype(np.float32)
    return expected, valid


def connected_components(mask: np.ndarray, min_area: int) -> list[dict]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    components: list[dict] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y, w, h = stats[label, :4].tolist()
        centroid = centroids[label].tolist()
        components.append(
            {
                "label": label,
                "area": area,
                "bbox": [int(x), int(y), int(w), int(h)],
                "centroid": [float(centroid[0]), float(centroid[1])],
                "mask": (labels == label).astype(np.uint8) * 255,
            }
        )
    return components


def largest_inside(mask: np.ndarray, min_area: int) -> np.ndarray:
    comps = connected_components(mask, min_area)
    if not comps:
        return np.zeros_like(mask)
    comps.sort(key=lambda item: item["area"], reverse=True)
    return comps[0]["mask"]


def pick_seed_mask(
    rgb_bgr: np.ndarray,
    depth_m: np.ndarray,
    board_mask: np.ndarray,
    board_depth_m: np.ndarray,
) -> tuple[np.ndarray, bool]:
    valid_depth = np.isfinite(depth_m) & (depth_m > 0)
    inside_board = board_mask > 0
    invalid_mask = ((~valid_depth) & inside_board).astype(np.uint8) * 255
    invalid_mask = cv2.morphologyEx(invalid_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    invalid_mask = cv2.morphologyEx(invalid_mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    invalid_seed = largest_inside(invalid_mask, min_area=30)
    if np.count_nonzero(invalid_seed) > 0:
        return invalid_seed, False

    residual = np.zeros_like(depth_m, dtype=np.float32)
    good = valid_depth & inside_board & np.isfinite(board_depth_m)
    residual[good] = board_depth_m[good] - depth_m[good]
    residual_mask = np.zeros_like(board_mask)
    residual_mask[good] = ((residual[good] > 0.0025) & (residual[good] < 0.03)).astype(np.uint8) * 255
    residual_mask = cv2.morphologyEx(residual_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    residual_mask = cv2.morphologyEx(residual_mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    residual_seed = largest_inside(residual_mask, min_area=40)
    if np.count_nonzero(residual_seed) > 0:
        return residual_seed, False

    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    sat_mask = ((hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 45) & inside_board).astype(np.uint8) * 255
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    sat_seed = largest_inside(sat_mask, min_area=30)
    return sat_seed, True


def padded_bbox(mask: np.ndarray, pad_x: int = 60, pad_y: int = 40, min_w: int = 140, min_h: int = 90) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Cannot build a bounding box from an empty mask.")
    x0 = max(0, int(xs.min()) - pad_x)
    y0 = max(0, int(ys.min()) - pad_y)
    x1 = int(xs.max()) + pad_x + 1
    y1 = int(ys.max()) + pad_y + 1

    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    width = max(min_w, x1 - x0)
    height = max(min_h, y1 - y0)
    x0 = int(round(cx - width / 2.0))
    y0 = int(round(cy - height / 2.0))
    x1 = x0 + width
    y1 = y0 + height
    return x0, y0, x1, y1


def clamp_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, x1)
    y1 = min(h, y1)
    return x0, y0, x1, y1


def run_grabcut(
    rgb_bgr: np.ndarray,
    board_mask: np.ndarray,
    seed_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    residual_fg_mask: np.ndarray,
) -> np.ndarray:
    h, w = board_mask.shape
    gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    inside_board = board_mask > 0
    gc_mask[inside_board] = cv2.GC_PR_BGD

    x0, y0, x1, y1 = bbox
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[y0:y1, x0:x1] = 255
    gc_mask[(roi_mask > 0) & inside_board] = cv2.GC_PR_FGD

    ring = np.zeros_like(board_mask)
    ring[max(0, y0 - 10):min(h, y1 + 10), max(0, x0 - 10):min(w, x1 + 10)] = 255
    ring[y0:y1, x0:x1] = 0
    gc_mask[(ring > 0) & inside_board] = cv2.GC_PR_BGD

    sure_fg = cv2.dilate(seed_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
    sure_fg = cv2.bitwise_and(sure_fg, board_mask)
    gc_mask[sure_fg > 0] = cv2.GC_FGD

    sure_bg = cv2.bitwise_and(board_mask, cv2.bitwise_not(roi_mask))
    gc_mask[sure_bg > 0] = cv2.GC_BGD
    gc_mask[(residual_fg_mask > 0) & (roi_mask > 0)] = cv2.GC_PR_FGD

    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(rgb_bgr, gc_mask, None, bg_model, fg_model, 5, cv2.GC_INIT_WITH_MASK)

    result = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    result = cv2.bitwise_and(result, board_mask)
    return result


def select_component(mask: np.ndarray, seed_mask: np.ndarray, min_area: int) -> np.ndarray:
    comps = connected_components(mask, min_area=min_area)
    if not comps:
        return np.zeros_like(mask)
    seed_dilated = cv2.dilate(seed_mask, np.ones((21, 21), dtype=np.uint8), iterations=1)
    overlapping = [comp for comp in comps if np.count_nonzero(cv2.bitwise_and(comp["mask"], seed_dilated)) > 0]
    if overlapping:
        overlapping.sort(key=lambda item: item["area"], reverse=True)
        return overlapping[0]["mask"]
    comps.sort(key=lambda item: item["area"], reverse=True)
    return comps[0]["mask"]


def seed_principal_axis(seed_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    ys, xs = np.where(seed_mask > 0)
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 1e-6)
    eigenvectors = eigenvectors[:, order]
    axis = eigenvectors[:, 0]
    major_std = float(np.sqrt(eigenvalues[0]))
    minor_std = float(np.sqrt(eigenvalues[1]))
    return centroid, axis, major_std, minor_std


def find_cap_circle(rgb_bgr: np.ndarray, seed_mask: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, float] | None:
    ys, xs = np.where(seed_mask > 0)
    x0 = max(0, int(xs.min()) - 80)
    x1 = min(rgb_bgr.shape[1], int(xs.max()) + 81)
    y0 = max(0, int(ys.min()) - 60)
    y1 = min(rgb_bgr.shape[0], int(ys.max()) + 61)
    crop = rgb_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=18,
        param1=100,
        param2=12,
        minRadius=6,
        maxRadius=12,
    )
    if circles is None:
        return None

    centroid, _, _, _ = seed_principal_axis(seed_mask)
    axis = axis / max(np.linalg.norm(axis), 1e-6)
    axis_perp = np.array([-axis[1], axis[0]], dtype=np.float64)

    best_score = None
    best_circle = None
    for x, y, radius in circles[0]:
        center = np.array([x + x0, y + y0], dtype=np.float64)
        delta = center - centroid
        along = abs(float(delta @ axis))
        perp = abs(float(delta @ axis_perp))
        if along < 4.0 or along > 45.0 or perp > 20.0:
            continue
        score = along - 1.5 * perp
        if best_score is None or score > best_score:
            best_score = score
            best_circle = (center, float(radius))
    return best_circle


def capsule_mask(shape: tuple[int, int], start_xy: np.ndarray, end_xy: np.ndarray, radius_px: float) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    start = tuple(np.round(start_xy).astype(np.int32))
    end = tuple(np.round(end_xy).astype(np.int32))
    thickness = max(1, int(round(2.0 * radius_px)))
    cv2.line(mask, start, end, 255, thickness, cv2.LINE_AA)
    cv2.circle(mask, start, max(1, int(round(radius_px))), 255, -1, cv2.LINE_AA)
    cv2.circle(mask, end, max(1, int(round(radius_px))), 255, -1, cv2.LINE_AA)
    return mask


def build_battery_mask(rgb_bgr: np.ndarray, board_mask: np.ndarray, seed_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    centroid, axis, major_std, minor_std = seed_principal_axis(seed_mask)
    circle = find_cap_circle(rgb_bgr, seed_mask, axis)
    if circle is None:
        cap_center = centroid - axis * max(10.0, 1.2 * major_std)
        cap_radius = max(7.0, 1.15 * minor_std)
    else:
        cap_center, cap_radius = circle

    body_dir = centroid - cap_center
    body_norm = np.linalg.norm(body_dir)
    if body_norm < 1e-6:
        body_dir = axis
    else:
        body_dir = body_dir / body_norm

    # The battery is too similar to the black board markers for a reliable
    # unconstrained segmentation, so we use a capsule with AA-like proportions.
    radius_px = float(np.clip(0.6 * (cap_radius + minor_std), 7.0, 10.5))
    centerline_length_px = float(max(4.8 * radius_px, 4.8 * major_std))
    other_end_center = cap_center + body_dir * centerline_length_px
    mask = capsule_mask(board_mask.shape, cap_center, other_end_center, radius_px)
    mask = cv2.bitwise_and(mask, board_mask)
    mask = cv2.bitwise_or(mask, cv2.dilate(seed_mask, np.ones((5, 5), dtype=np.uint8), iterations=1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    mask = cv2.bitwise_and(mask, board_mask)
    return mask, cap_center, radius_px


def create_checkerboard(shape: tuple[int, int], tile: int = 12) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices((h, w))
    board = ((xx // tile + yy // tile) % 2).astype(np.uint8)
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[board == 0] = (225, 225, 225)
    img[board == 1] = (195, 195, 195)
    return img


def compose_cutout(rgb_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bg = create_checkerboard(mask.shape)
    out = bg.copy()
    keep = mask > 0
    out[keep] = rgb_bgr[keep]
    return out


def colorize_depth(depth_m: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    out = np.full((*depth_m.shape, 3), 30, dtype=np.uint8)
    if not np.any(valid):
        return out
    d_min = float(np.percentile(depth_m[valid], 2.0))
    d_max = float(np.percentile(depth_m[valid], 98.0))
    if d_max <= d_min:
        d_max = d_min + 1e-3
    norm = np.clip((depth_m - d_min) / (d_max - d_min), 0.0, 1.0)
    out = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[~valid] = (30, 30, 30)
    return out


def resize_width(image_bgr: np.ndarray, target_width: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    scale = target_width / float(w)
    target_height = max(1, int(round(h * scale)))
    return cv2.resize(image_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)


def make_panel(
    frame_id: str,
    rgb_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    cutout_bgr: np.ndarray,
    tile_width: int,
) -> np.ndarray:
    rgb_tile = resize_width(rgb_bgr, tile_width)
    overlay_tile = resize_width(overlay_bgr, tile_width)
    cutout_tile = resize_width(cutout_bgr, tile_width)
    target_h = min(rgb_tile.shape[0], overlay_tile.shape[0], cutout_tile.shape[0])
    rgb_tile = cv2.resize(rgb_tile, (tile_width, target_h), interpolation=cv2.INTER_AREA)
    overlay_tile = cv2.resize(overlay_tile, (tile_width, target_h), interpolation=cv2.INTER_AREA)
    cutout_tile = cv2.resize(cutout_tile, (tile_width, target_h), interpolation=cv2.INTER_AREA)
    spacer = np.full((target_h, 10, 3), 18, dtype=np.uint8)
    header = np.full((36, tile_width * 3 + 20, 3), 245, dtype=np.uint8)
    footer = np.full((44, tile_width * 3 + 20, 3), 245, dtype=np.uint8)
    cv2.putText(header, "RGB", (16, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(header, "Mask Overlay", (tile_width + 26, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(header, "Battery Cutout", (tile_width * 2 + 36, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(footer, frame_id, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2, cv2.LINE_AA)
    body = np.hstack([rgb_tile, spacer, overlay_tile, spacer, cutout_tile])
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


def mask_overlay(rgb_bgr: np.ndarray, board_mask: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    overlay = rgb_bgr.copy()
    board_outline = cv2.Canny(board_mask, 30, 90)
    overlay[board_outline > 0] = (255, 180, 0)
    overlay[mask > 0] = cv2.addWeighted(overlay, 0.35, np.full_like(overlay, (40, 210, 40)), 0.65, 0)[mask > 0]
    x0, y0, x1, y1 = bbox
    cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (0, 160, 255), 2, cv2.LINE_AA)
    return overlay


def mask_overlay_with_cap(
    rgb_bgr: np.ndarray,
    board_mask: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    cap_center: np.ndarray,
    cap_radius_px: float,
) -> np.ndarray:
    overlay = mask_overlay(rgb_bgr, board_mask, mask, bbox)
    center = tuple(np.round(cap_center).astype(np.int32))
    cv2.circle(overlay, center, max(1, int(round(cap_radius_px))), (0, 0, 255), 2, cv2.LINE_AA)
    return overlay


def tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def main() -> int:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    rgba_dir = output_dir / "rgba"
    mask_dir = output_dir / "masks"
    crop_dir = output_dir / "crops"
    depth_dir = output_dir / "masked_depth"
    diag_dir = output_dir / "diagnostics"
    for path in [rgba_dir, mask_dir, crop_dir, depth_dir, diag_dir]:
        path.mkdir(parents=True, exist_ok=True)

    intr, spec = load_metadata(data_dir)
    pose_paths = sorted((data_dir / "poses").glob("*.json"))
    if not pose_paths:
        raise SystemExit(f"No pose files found in {(data_dir / 'poses')}")

    panels: list[np.ndarray] = []
    records: list[SegmentationRecord] = []

    for pose_path in pose_paths:
        frame_id = pose_path.stem
        rgb_path = data_dir / "rgb" / f"{frame_id}.jpg"
        depth_path = data_dir / "depth" / f"{frame_id}.npz"
        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise SystemExit(f"Failed to read RGB image {rgb_path}")
        depth_m = np.load(str(depth_path))["depth"].astype(np.float32)
        world2cam, _ = load_pose(pose_path)

        board_mask = board_polygon_mask(depth_m.shape, intr, spec, world2cam)
        board_depth_m, _ = estimate_board_depth(depth_m, intr, world2cam)
        seed_mask, fallback_used = pick_seed_mask(rgb_bgr, depth_m, board_mask, board_depth_m)

        if np.count_nonzero(seed_mask) == 0:
            raise SystemExit(f"No seed pixels found for frame {frame_id}; segmentation needs manual help.")

        bbox = clamp_bbox(padded_bbox(seed_mask), depth_m.shape)
        final_mask, cap_center, cap_radius_px = build_battery_mask(rgb_bgr, board_mask, seed_mask)
        if np.count_nonzero(final_mask) < args.min_component_area:
            final_mask = cv2.dilate(seed_mask, np.ones((9, 9), dtype=np.uint8), iterations=1)
            cap_center = np.array([0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])], dtype=np.float64)
            cap_radius_px = 8.0

        x0, y0, x1, y1 = tight_bbox(final_mask)
        rgba = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = final_mask
        masked_depth = np.where(final_mask > 0, depth_m, 0.0).astype(np.float32)

        Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)).save(rgba_dir / f"{frame_id}.png")
        Image.fromarray(final_mask).save(mask_dir / f"{frame_id}.png")
        Image.fromarray(cv2.cvtColor(rgba[y0:y1, x0:x1], cv2.COLOR_BGRA2RGBA)).save(crop_dir / f"{frame_id}.png")
        np.savez_compressed(depth_dir / f"{frame_id}.npz", depth=masked_depth[y0:y1, x0:x1])

        overlay = mask_overlay_with_cap(rgb_bgr, board_mask, final_mask, bbox, cap_center, cap_radius_px)
        cv2.imwrite(str(diag_dir / f"{frame_id}_overlay.png"), overlay)
        cv2.imwrite(str(diag_dir / f"{frame_id}_depth.png"), colorize_depth(masked_depth))
        panels.append(make_panel(frame_id, rgb_bgr, overlay, compose_cutout(rgb_bgr, final_mask), args.tile_width))

        records.append(
            SegmentationRecord(
                frame_id=frame_id,
                bbox_xywh=[int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                mask_area_px=int(np.count_nonzero(final_mask)),
                seed_area_px=int(np.count_nonzero(seed_mask)),
                fallback_used=fallback_used,
            )
        )

    write_contact_sheet(panels, output_dir / "contact_sheet.png", args.sheet_columns)
    (output_dir / "segmentation_summary.json").write_text(
        json.dumps([record.__dict__ for record in records], indent=2),
        encoding="utf-8",
    )
    print(f"[cutout] Wrote battery-only assets to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
