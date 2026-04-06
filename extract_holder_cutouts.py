#!/usr/bin/env python3
"""Extract holder-only cutouts from the selected ChArUco subset."""

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract holder-only cutouts from a holder subset dataset.")
    parser.add_argument("--data-dir", default="holder_subset_dataset", help="Subset dataset directory.")
    parser.add_argument(
        "--output-dir",
        default="holder_subset_dataset/holder_only",
        help="Output directory for masks, cutouts, and diagnostics.",
    )
    parser.add_argument("--sheet-columns", type=int, default=2, help="Columns in the contact sheet.")
    parser.add_argument("--tile-width", type=int, default=320, help="Tile width for contact sheet panels.")
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
    points_cam = (world2cam[:3, :3] @ board_corners.T).T + world2cam[:3, 3]
    uvw = (intr.K @ points_cam.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(uv).astype(np.int32), 255)
    return cv2.erode(mask, np.ones((9, 9), dtype=np.uint8))


def extract_mask(rgb_bgr: np.ndarray, board_mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (
        (hue > 80)
        & (hue < 130)
        & (sat > 60)
        & (val > 40)
        & (board_mask > 0)
    ).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    label = int(np.argmax(areas)) + 1
    return (labels == label).astype(np.uint8) * 255


def create_checkerboard(shape: tuple[int, int], tile: int = 12) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices((h, w))
    board = ((xx // tile + yy // tile) % 2).astype(np.uint8)
    out = np.empty((h, w, 3), dtype=np.uint8)
    out[board == 0] = (225, 225, 225)
    out[board == 1] = (195, 195, 195)
    return out


def compose_cutout(rgb_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bg = create_checkerboard(mask.shape)
    out = bg.copy()
    out[mask > 0] = rgb_bgr[mask > 0]
    return out


def tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


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
    return cv2.resize(image_bgr, (target_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def make_panel(frame_id: str, rgb_bgr: np.ndarray, overlay_bgr: np.ndarray, cutout_bgr: np.ndarray, tile_width: int) -> np.ndarray:
    rgb_tile = resize_width(rgb_bgr, tile_width)
    overlay_tile = resize_width(overlay_bgr, tile_width)
    cutout_tile = resize_width(cutout_bgr, tile_width)
    target_h = min(rgb_tile.shape[0], overlay_tile.shape[0], cutout_tile.shape[0])
    rgb_tile = cv2.resize(rgb_tile, (tile_width, target_h), interpolation=cv2.INTER_AREA)
    overlay_tile = cv2.resize(overlay_tile, (tile_width, target_h), interpolation=cv2.INTER_AREA)
    cutout_tile = cv2.resize(cutout_tile, (tile_width, target_h), interpolation=cv2.INTER_AREA)
    spacer = np.full((target_h, 10, 3), 18, dtype=np.uint8)
    header = np.full((36, tile_width * 3 + 20, 3), 245, dtype=np.uint8)
    footer = np.full((42, tile_width * 3 + 20, 3), 245, dtype=np.uint8)
    cv2.putText(header, "RGB", (16, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(header, "Mask Overlay", (tile_width + 26, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(header, "Holder Cutout", (tile_width * 2 + 36, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(footer, frame_id, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2, cv2.LINE_AA)
    body = np.hstack([rgb_tile, spacer, overlay_tile, spacer, cutout_tile])
    return np.vstack([header, body, footer])


def write_contact_sheet(panels: list[np.ndarray], output_path: Path, columns: int) -> None:
    panel_h, panel_w = panels[0].shape[:2]
    rows = math.ceil(len(panels) / max(1, columns))
    canvas = np.full((rows * panel_h, max(1, columns) * panel_w, 3), 255, dtype=np.uint8)
    for idx, panel in enumerate(panels):
        row = idx // max(1, columns)
        col = idx % max(1, columns)
        canvas[row * panel_h:(row + 1) * panel_h, col * panel_w:(col + 1) * panel_w] = panel
    Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(output_path)


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
    panels: list[np.ndarray] = []
    summary: list[dict] = []

    for pose_path in sorted((data_dir / "poses").glob("*.json")):
        frame_id = pose_path.stem
        world2cam = np.array(json.loads(pose_path.read_text(encoding="utf-8"))["world2cam"], dtype=np.float64)
        rgb_bgr = cv2.imread(str(data_dir / "rgb" / f"{frame_id}.jpg"), cv2.IMREAD_COLOR)
        depth_m = np.load(str(data_dir / "depth" / f"{frame_id}.npz"))["depth"].astype(np.float32)
        board_mask = board_polygon_mask(depth_m.shape, intr, spec, world2cam)
        mask = extract_mask(rgb_bgr, board_mask)
        bbox = tight_bbox(mask)

        rgba = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = mask
        masked_depth = np.where(mask > 0, depth_m, 0.0).astype(np.float32)
        x0, y0, x1, y1 = bbox

        Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)).save(rgba_dir / f"{frame_id}.png")
        Image.fromarray(mask).save(mask_dir / f"{frame_id}.png")
        Image.fromarray(cv2.cvtColor(rgba[y0:y1, x0:x1], cv2.COLOR_BGRA2RGBA)).save(crop_dir / f"{frame_id}.png")
        np.savez_compressed(depth_dir / f"{frame_id}.npz", depth=masked_depth[y0:y1, x0:x1])

        overlay = rgb_bgr.copy()
        overlay[mask > 0] = cv2.addWeighted(overlay, 0.35, np.full_like(overlay, (40, 210, 40)), 0.65, 0)[mask > 0]
        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (0, 160, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(diag_dir / f"{frame_id}_overlay.png"), overlay)
        cv2.imwrite(str(diag_dir / f"{frame_id}_depth.png"), colorize_depth(masked_depth))

        panels.append(make_panel(frame_id, rgb_bgr, overlay, compose_cutout(rgb_bgr, mask), args.tile_width))
        summary.append(
            {
                "frame_id": frame_id,
                "bbox_xywh": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                "mask_area_px": int(np.count_nonzero(mask)),
            }
        )

    write_contact_sheet(panels, output_dir / "contact_sheet.png", args.sheet_columns)
    (output_dir / "segmentation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[holder-cutout] Wrote holder-only assets to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
