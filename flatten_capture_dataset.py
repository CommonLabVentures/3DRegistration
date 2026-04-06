#!/usr/bin/env python3
"""Flatten raw capture folders into paired rgb/depth datasets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flatten raw capture folders into rgb/depth pairs.")
    parser.add_argument("--capture-root", required=True, help="Directory containing capture_* folders.")
    parser.add_argument("--output-dir", required=True, help="Directory to write rgb/ and depth/ into.")
    parser.add_argument("--camera-name", default="eye1", help="Camera subdirectory to use.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    capture_root = Path(args.capture_root)
    output_dir = Path(args.output_dir)
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    captures = sorted(capture_root.glob("capture_*"))
    if not captures:
        raise SystemExit(f"No capture_* folders found under {capture_root}")

    manifest: list[dict] = []
    for idx, capture_dir in enumerate(captures):
        frame_id = f"{idx:03d}"
        camera_dir = capture_dir / args.camera_name
        rgb_src = camera_dir / "image.jpg"
        depth_src = camera_dir / "aligned_depth_to_color.npz"
        if not rgb_src.exists() or not depth_src.exists():
            print(f"[flatten] skip {capture_dir.name}: missing RGB or depth")
            continue

        rgb_dst = rgb_dir / f"{frame_id}.jpg"
        depth_dst = depth_dir / f"{frame_id}.npz"
        shutil.copy2(rgb_src, rgb_dst)

        depth = np.load(str(depth_src))["depth"].astype(np.float32)
        np.savez_compressed(depth_dst, depth=depth)

        manifest.append(
            {
                "frame_id": frame_id,
                "capture_name": capture_dir.name,
                "rgb_path": str(rgb_dst),
                "depth_path": str(depth_dst),
                "source_rgb_path": str(rgb_src),
                "source_depth_path": str(depth_src),
            }
        )

    metadata = {
        "capture_root": str(capture_root),
        "camera_name": args.camera_name,
        "num_frames": len(manifest),
        "frames": manifest,
    }
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Flattened Capture Dataset",
                "",
                f"Source root: `{capture_root}`",
                f"Camera: `{args.camera_name}`",
                "",
                "Frames are sorted by capture folder name and renumbered into `rgb/` and `depth/`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[flatten] Wrote {len(manifest)} frame(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
