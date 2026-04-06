#!/usr/bin/env python3
"""Visualize RGB + aligned depth for a capture_* dataset.

Generates:
  - one side-by-side PNG per capture
  - a contact sheet PNG across the whole dataset

Example:
  .venv/bin/python visualize_captures.py \
      --data-dir ./dataset \
      --output-dir ./output/capture_viz
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass
class CapturePreview:
    capture_name: str
    panel_bgr: np.ndarray
    depth_min_m: float
    depth_max_m: float
    valid_fraction: float
    has_pose: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render side-by-side RGB and aligned depth previews for capture_* folders."
    )
    parser.add_argument(
        "--data-dir",
        default="dataset",
        help="Directory containing capture_* folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/capture_viz",
        help="Directory where preview PNGs will be written.",
    )
    parser.add_argument(
        "--camera-name",
        default="eye1",
        help="Camera subdirectory inside each capture folder.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of captures to render.",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=480,
        help="Width in pixels for each RGB/depth tile inside a panel.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=2,
        help="Number of columns in the contact sheet.",
    )
    return parser


def _load_capture(capture_dir: Path, camera_name: str) -> tuple[np.ndarray, np.ndarray, bool]:
    cam_dir = capture_dir / camera_name
    rgb = cv2.imread(str(cam_dir / "image.jpg"), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"Missing RGB image: {cam_dir / 'image.jpg'}")

    depth_path = cam_dir / "aligned_depth_to_color.npz"
    if not depth_path.exists():
        raise FileNotFoundError(f"Missing depth file: {depth_path}")
    depth_npz = np.load(str(depth_path))
    if "depth" not in depth_npz:
        raise KeyError(f"No 'depth' array found in {depth_path}")
    depth = depth_npz["depth"].astype(np.float32)

    has_pose = (cam_dir / "camera_pose.json").exists()
    return rgb, depth, has_pose


def _colorize_depth(depth_m: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    valid_fraction = float(np.count_nonzero(valid)) / float(depth_m.size)

    color = np.full((*depth_m.shape, 3), 30, dtype=np.uint8)
    if not np.any(valid):
        return color, 0.0, 0.0, valid_fraction

    valid_depth = depth_m[valid]
    d_min = float(np.percentile(valid_depth, 2.0))
    d_max = float(np.percentile(valid_depth, 98.0))
    if d_max <= d_min:
        d_min = float(valid_depth.min())
        d_max = float(valid_depth.max())
    if d_max <= d_min:
        d_max = d_min + 1e-3

    norm = np.clip((depth_m - d_min) / (d_max - d_min), 0.0, 1.0)
    norm_u8 = (norm * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(norm_u8, cv2.COLORMAP_TURBO)
    color[~valid] = (30, 30, 30)

    return color, d_min, d_max, valid_fraction


def _resize_for_panel(img_bgr: np.ndarray, tile_width: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = tile_width / float(w)
    tile_height = max(1, int(round(h * scale)))
    return cv2.resize(img_bgr, (tile_width, tile_height), interpolation=cv2.INTER_AREA)


def _make_panel(
    capture_name: str,
    rgb_bgr: np.ndarray,
    depth_bgr: np.ndarray,
    depth_min_m: float,
    depth_max_m: float,
    valid_fraction: float,
    has_pose: bool,
    tile_width: int,
) -> np.ndarray:
    rgb_tile = _resize_for_panel(rgb_bgr, tile_width)
    depth_tile = _resize_for_panel(depth_bgr, tile_width)

    if rgb_tile.shape[0] != depth_tile.shape[0]:
        target_h = min(rgb_tile.shape[0], depth_tile.shape[0])
        rgb_tile = cv2.resize(rgb_tile, (tile_width, target_h), interpolation=cv2.INTER_AREA)
        depth_tile = cv2.resize(depth_tile, (tile_width, target_h), interpolation=cv2.INTER_AREA)

    separator = np.full((rgb_tile.shape[0], 12, 3), 18, dtype=np.uint8)
    body = np.hstack([rgb_tile, separator, depth_tile])

    footer_h = 96
    footer = np.full((footer_h, body.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(
        footer,
        capture_name,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    info_1 = f"Depth range: {depth_min_m:.3f}m to {depth_max_m:.3f}m"
    info_2 = f"Valid depth: {valid_fraction * 100.0:.1f}%   Pose file: {'yes' if has_pose else 'no'}"
    cv2.putText(
        footer,
        info_1,
        (16, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        footer,
        info_2,
        (16, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )

    header = np.full((32, body.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(header, "RGB", (16, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(
        header,
        "Aligned Depth",
        (tile_width + 28, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    return np.vstack([header, body, footer])


def render_capture_preview(capture_dir: Path, camera_name: str, tile_width: int) -> CapturePreview:
    rgb_bgr, depth_m, has_pose = _load_capture(capture_dir, camera_name)
    depth_bgr, depth_min_m, depth_max_m, valid_fraction = _colorize_depth(depth_m)
    panel_bgr = _make_panel(
        capture_dir.name,
        rgb_bgr,
        depth_bgr,
        depth_min_m,
        depth_max_m,
        valid_fraction,
        has_pose,
        tile_width,
    )
    return CapturePreview(
        capture_name=capture_dir.name,
        panel_bgr=panel_bgr,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        valid_fraction=valid_fraction,
        has_pose=has_pose,
    )


def _write_contact_sheet(previews: list[CapturePreview], output_path: Path, columns: int) -> None:
    if not previews:
        raise ValueError("No previews to write.")

    columns = max(1, columns)
    panel_h, panel_w = previews[0].panel_bgr.shape[:2]
    rows = math.ceil(len(previews) / columns)
    canvas = np.full((rows * panel_h, columns * panel_w, 3), 255, dtype=np.uint8)

    for idx, preview in enumerate(previews):
        row = idx // columns
        col = idx % columns
        y0 = row * panel_h
        x0 = col * panel_w
        canvas[y0:y0 + panel_h, x0:x0 + panel_w] = preview.panel_bgr

    Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(output_path)


def main() -> int:
    args = build_parser().parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture_dirs = sorted(data_dir.glob("capture_*"))
    if args.limit is not None:
        capture_dirs = capture_dirs[:args.limit]
    if not capture_dirs:
        raise SystemExit(f"No capture_* folders found in {data_dir}")

    previews: list[CapturePreview] = []
    for capture_dir in capture_dirs:
        try:
            preview = render_capture_preview(capture_dir, args.camera_name, args.tile_width)
            previews.append(preview)
            panel_path = output_dir / f"{capture_dir.name}.png"
            Image.fromarray(cv2.cvtColor(preview.panel_bgr, cv2.COLOR_BGR2RGB)).save(panel_path)
            print(f"[viz] Wrote {panel_path}")
        except Exception as exc:
            print(f"[viz] Skipped {capture_dir.name}: {exc}")

    if not previews:
        raise SystemExit("No capture previews were generated.")

    contact_path = output_dir / "contact_sheet.png"
    _write_contact_sheet(previews, contact_path, args.columns)
    print(f"[viz] Wrote {contact_path}")
    print(f"[viz] Rendered {len(previews)} capture previews.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
