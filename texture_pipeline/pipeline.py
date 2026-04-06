"""Main orchestrator — TexturePipeline class.

Wires together Stages 1–5 and handles the two input data formats:
  A. Capture-directory format: dataset_dir/capture_*/eye1/ (actual robot data)
  B. Flat format: rgb/, depth/, poses/ (spec layout)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import trimesh

from .alignment import align_mesh
from .charuco import (
    detect_all_charuco_poses,
    load_frames_from_dataset,
    load_frames_from_flat_dirs,
)
from .config import PipelineConfig, default_config, load_config
from .export import export_obj
from .segmentation import aggregate_plane, segment_all_frames
from .texturing import texture_mesh
from .utils import FrameData
from .viz import save_all_diagnostics


def _prepare_mesh(mesh: trimesh.Trimesh, mesh_scale: Optional[float]) -> trimesh.Trimesh:
    """Scale and centre a mesh for world-frame alignment.

    If mesh_scale is None, auto-detects the unit from the largest extent:
      - extents > 0.5  → assume millimetres → scale by 0.001
      - extents > 0.05 → assume centimetres  → scale by 0.01
      - otherwise      → assume metres, no scaling
    Then centres the mesh at the origin so the alignment's centroid
    estimation works without numeric offsets.
    """
    import numpy as np

    # Determine scale
    max_extent = float(mesh.extents.max())
    if mesh_scale is None:
        if max_extent > 0.5:          # almost certainly mm
            mesh_scale = 0.001
            print(f"  [pipeline] Auto-detected mesh units: mm  (max extent {max_extent:.1f}) → scale 0.001")
        elif max_extent > 0.05:       # cm
            mesh_scale = 0.01
            print(f"  [pipeline] Auto-detected mesh units: cm  (max extent {max_extent:.1f}) → scale 0.01")
        else:
            mesh_scale = 1.0
            print(f"  [pipeline] Mesh units appear to be metres (max extent {max_extent:.4f})")
    else:
        print(f"  [pipeline] Applying user mesh scale: {mesh_scale}")

    # Apply scale and centre at origin
    import copy
    mesh = copy.deepcopy(mesh)
    mesh.vertices = mesh.vertices * mesh_scale
    mesh.vertices -= mesh.centroid       # centre so alignment starts at (0,0,0)
    mesh._cache.clear()
    return mesh


class TexturePipeline:
    """End-to-end textured mesh pipeline.

    Usage::

        pipe = TexturePipeline.from_dataset_dir(
            dataset_dir="./dataset",
            mesh_paths=["./meshes/battery.stl"],
            output_dir="./output",
        )
        pipe.run()
    """

    def __init__(
        self,
        frames: list[FrameData],
        mesh_paths: list[str | Path],
        output_dir: str | Path,
        config: PipelineConfig,
        init_method: Optional[str] = None,
        interactive: bool = False,
        save_diagnostics: bool = True,
        mesh_scale: Optional[float] = None,
        export_glb: bool = True,
    ):
        self.frames = frames
        self.mesh_paths = [Path(p) for p in mesh_paths]
        self.output_dir = Path(output_dir)
        self.config = config
        self.init_method = init_method
        self.interactive = interactive
        self.save_diagnostics = save_diagnostics
        self.mesh_scale = mesh_scale  # explicit scale; None = auto-detect
        self.export_glb = export_glb

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dataset_dir(
        cls,
        dataset_dir: str | Path,
        mesh_paths: list[str | Path],
        output_dir: str | Path,
        config: Optional[PipelineConfig] = None,
        camera_name: str = "eye1",
        max_frames: Optional[int] = None,
        **kwargs,
    ) -> "TexturePipeline":
        """Build pipeline from capture_* directory dataset."""
        dataset_dir = Path(dataset_dir)

        # Load frames
        print(f"[pipeline] Loading frames from {dataset_dir} …")
        frames = load_frames_from_dataset(
            dataset_dir, camera_name=camera_name, max_frames=max_frames
        )
        print(f"[pipeline] Loaded {len(frames)} frames")

        # Auto-detect config from first frame's camera info if not provided
        if config is None:
            first_cam_info = (
                sorted(dataset_dir.glob(f"capture_*/{camera_name}/color_camera_info.json"))[0]
            )
            config = default_config(first_cam_info)

        return cls(frames, mesh_paths, output_dir, config, **kwargs)

    @classmethod
    def from_flat_dirs(
        cls,
        rgb_dir: str | Path,
        depth_dir: str | Path,
        mesh_paths: list[str | Path],
        output_dir: str | Path,
        config: PipelineConfig,
        pose_dir: Optional[str | Path] = None,
        **kwargs,
    ) -> "TexturePipeline":
        """Build pipeline from flat rgb/ depth/ directory layout."""
        print(f"[pipeline] Loading frames from {rgb_dir} / {depth_dir} …")
        frames = load_frames_from_flat_dirs(
            rgb_dir,
            depth_dir,
            pose_dir,
            config.camera,
            charuco_cfg=config.charuco if pose_dir is None else None,
        )
        # Filter frames with valid poses
        valid_frames = [f for f in frames if not (f.world2cam == 0).all()]
        print(f"[pipeline] Loaded {len(valid_frames)}/{len(frames)} frames with valid poses")
        return cls(valid_frames, mesh_paths, output_dir, config, **kwargs)

    # ------------------------------------------------------------------
    # Stage runners
    # ------------------------------------------------------------------

    def run_segmentation(self, plane=None):
        """Stage 2: produce per-frame object masks."""
        print("\n[Stage 2] Segmentation")
        masks = segment_all_frames(
            self.frames,
            self.config.segmentation,
            plane=plane,
            use_grabcut=True,
        )
        return masks

    def run_alignment(
        self,
        mesh: trimesh.Trimesh,
        masks: list,
        init_method: Optional[str] = None,
    ) -> "np.ndarray":
        """Stage 3: align mesh to scene."""
        print("\n[Stage 3] Alignment")
        T_obj = align_mesh(
            mesh,
            self.frames,
            masks,
            self.config.alignment,
            init_method=init_method or self.init_method,
        )
        return T_obj

    def run_texturing(self, mesh: trimesh.Trimesh, T_obj, masks=None) -> tuple:
        """Stage 4: texture the mesh."""
        print("\n[Stage 4] Texturing")
        mesh_uv, uvs, atlas = texture_mesh(
            mesh, self.frames, T_obj, self.config.texturing, masks=masks
        )
        return mesh_uv, uvs, atlas

    def run_export(
        self,
        mesh_uv: trimesh.Trimesh,
        uvs,
        T_obj,
        atlas,
        name: str,
    ) -> dict:
        """Stage 5: export OBJ + MTL + PNG."""
        print("\n[Stage 5] Export")
        paths = export_obj(
            mesh_uv,
            uvs,
            T_obj,
            atlas,
            name,
            self.output_dir,
            also_glb=self.export_glb,
        )
        return paths

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self, n_objects: int = 1) -> list[dict]:
        """Run the full pipeline for all meshes.

        Args:
            n_objects: number of distinct objects in the scene (used for
                multi-object connected-component separation).

        Returns:
            List of output path dicts, one per mesh.
        """
        print(f"\n{'='*60}")
        print(f"  Textured Mesh Pipeline — {len(self.frames)} frames, "
              f"{len(self.mesh_paths)} mesh(es)")
        print(f"{'='*60}")

        # Stage 2: shared segmentation across all meshes
        masks = self.run_segmentation()

        results = []
        for mesh_path in self.mesh_paths:
            name = mesh_path.stem
            print(f"\n--- Processing mesh: {name} ---")

            # Load mesh
            mesh = trimesh.load(str(mesh_path), force="mesh")
            if not isinstance(mesh, trimesh.Trimesh):
                mesh = trimesh.util.concatenate(mesh.dump())

            mesh = _prepare_mesh(mesh, self.mesh_scale)
            print(f"  [pipeline] Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
                  f"extents={mesh.extents}")

            # Stage 3: alignment
            T_obj = self.run_alignment(mesh, masks, init_method=self.init_method)

            # Stage 4: texturing
            mesh_uv, uvs, atlas = self.run_texturing(mesh, T_obj, masks=masks)

            # Stage 5: export
            paths = self.run_export(mesh_uv, uvs, T_obj, atlas, name)

            # Diagnostics saved after texturing so atlas is available
            if self.save_diagnostics:
                save_all_diagnostics(
                    self.frames, masks, mesh, T_obj,
                    atlas_bgr=atlas,
                    output_dir=self.output_dir / name,
                )
            results.append(paths)

            print(f"\n  [pipeline] Done: {name}")
            for k, p in paths.items():
                print(f"    {k:8s}: {p}")

        print(f"\n{'='*60}")
        print("  Pipeline complete.")
        print(f"{'='*60}\n")
        return results
