"""CLI entry point for the textured mesh pipeline.

Examples
--------
# Full pipeline from capture_* dataset (robot data format)
python -m texture_pipeline.run \\
    --data-dir ./dataset \\
    --meshes ./meshes/battery.stl \\
    --output ./output

# Full pipeline from flat rgb/ depth/ directories
python -m texture_pipeline.run \\
    --rgb-dir ./rgb --depth-dir ./depth \\
    --meshes ./meshes/widget.stl \\
    --config config.yaml \\
    --output ./output

# Run with manual point-picking initialization
python -m texture_pipeline.run \\
    --data-dir ./dataset \\
    --meshes ./meshes/battery.stl \\
    --output ./output \\
    --init-method points

# Limit to N frames (for quick tests)
python -m texture_pipeline.run \\
    --data-dir ./dataset \\
    --meshes ./meshes/battery.stl \\
    --output ./output \\
    --max-frames 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="texture_pipeline",
        description="Texture an untextured 3D mesh using RGBD images.",
    )

    # --- Input source (mutually exclusive sets) ---
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--data-dir", metavar="DIR",
        help="Dataset directory containing capture_* subdirectories (robot data format).",
    )
    input_group.add_argument(
        "--rgb-dir", metavar="DIR",
        help="Directory of RGB images (flat format).",
    )

    p.add_argument(
        "--depth-dir", metavar="DIR",
        help="Directory of depth images (required with --rgb-dir).",
    )
    p.add_argument(
        "--pose-dir", metavar="DIR", default=None,
        help="Directory of camera_pose.json files (flat format). "
             "If absent, poses are estimated from ChArUco board detection.",
    )

    # --- Meshes ---
    p.add_argument(
        "--meshes", nargs="+", required=True, metavar="MESH",
        help="Path(s) to untextured mesh file(s) (STL or OBJ).",
    )
    p.add_argument(
        "--mesh-scale", type=float, default=None, metavar="S",
        help="Scale factor applied to mesh vertices before alignment "
             "(e.g. 0.001 to convert mm → m). Auto-detected if omitted.",
    )

    # --- Config ---
    p.add_argument(
        "--config", metavar="YAML", default=None,
        help="Path to config.yaml. Required for flat format; optional for capture format.",
    )
    p.add_argument(
        "--camera-name", default="eye1", metavar="NAME",
        help="Camera sub-directory name within each capture_* folder (default: eye1).",
    )

    # --- Output ---
    p.add_argument(
        "--output", "-o", required=True, metavar="DIR",
        help="Output directory for OBJ / MTL / PNG files.",
    )

    # --- Pipeline control ---
    p.add_argument(
        "--init-method",
        choices=["centroid", "pca", "exhaustive", "points"],
        default=None,
        help="Coarse alignment initialization method (default: centroid).",
    )
    p.add_argument(
        "--max-frames", type=int, default=None, metavar="N",
        help="Maximum number of frames to load (useful for quick tests).",
    )
    p.add_argument(
        "--atlas-size", type=int, default=None, metavar="PX",
        help="Override texture atlas size in pixels (e.g. 2048).",
    )
    p.add_argument(
        "--seam-blend",
        choices=["none", "laplacian", "poisson"],
        default=None,
        help="Seam blending method override.",
    )
    p.add_argument(
        "--no-diagnostics", action="store_true",
        help="Skip saving diagnostic images.",
    )
    p.add_argument(
        "--no-glb", action="store_true",
        help="Skip GLB export.",
    )
    p.add_argument(
        "--interactive", action="store_true",
        help="Enable interactive steps (manual point picking for alignment).",
    )

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Lazy imports so --help is instant
    from .config import default_config, load_config
    from .pipeline import TexturePipeline

    config = None
    if args.config:
        config = load_config(args.config)
    elif args.data_dir:
        cam_info_paths = sorted(
            Path(args.data_dir).glob(f"capture_*/{args.camera_name}/color_camera_info.json")
        )
        if not cam_info_paths:
            parser.error(
                f"No camera info files found under {args.data_dir}/capture_*/{args.camera_name}/"
            )
        config = default_config(cam_info_paths[0])

    # Apply CLI overrides to config fields.
    if args.atlas_size is not None:
        config.texturing.atlas_size = args.atlas_size
    if args.seam_blend is not None:
        config.texturing.seam_blend = args.seam_blend
    if args.init_method is not None:
        config.alignment.init_method = args.init_method

    # Build the pipeline
    extra = dict(
        init_method=args.init_method,
        interactive=args.interactive,
        save_diagnostics=not args.no_diagnostics,
        mesh_scale=args.mesh_scale,
        export_glb=not args.no_glb,
    )

    if args.data_dir:
        pipe = TexturePipeline.from_dataset_dir(
            dataset_dir=args.data_dir,
            mesh_paths=args.meshes,
            output_dir=args.output,
            config=config,
            camera_name=args.camera_name,
            max_frames=args.max_frames,
            **extra,
        )
    else:
        if not args.depth_dir:
            parser.error("--depth-dir is required when using --rgb-dir")
        if config is None:
            parser.error("--config is required when using --rgb-dir / --depth-dir")

        pipe = TexturePipeline.from_flat_dirs(
            rgb_dir=args.rgb_dir,
            depth_dir=args.depth_dir,
            mesh_paths=args.meshes,
            output_dir=args.output,
            config=config,
            pose_dir=args.pose_dir,
            **extra,
        )

    pipe.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
