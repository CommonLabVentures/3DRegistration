"""Configuration dataclasses and YAML/JSON loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


@dataclass
class CameraConfig:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    depth_scale: float = 1.0  # multiply raw depth by this to get meters; 1.0 if already meters

    @property
    def K(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)


@dataclass
class CharucoConfig:
    squares_x: int = 5
    squares_y: int = 7
    square_length: float = 0.04   # meters
    marker_length: float = 0.03   # meters
    dictionary: str = "DICT_4X4_100"
    min_corners: int = 6


@dataclass
class SegmentationConfig:
    board_plane_inlier_threshold: float = 0.008   # meters
    object_min_height: float = 0.005              # above board plane
    object_max_height: float = 0.30
    grabcut_iters: int = 5


@dataclass
class AlignmentConfig:
    method: str = "silhouette"           # "silhouette" or "icp_then_silhouette"
    init_method: str = "centroid"        # "centroid", "pca", "exhaustive", "points"
    optimizer: str = "Powell"
    max_iter: int = 500
    iou_warn_threshold: float = 0.3


@dataclass
class TexturingConfig:
    atlas_size: int = 2048
    view_angle_threshold_deg: float = 75.0
    seam_blend: str = "laplacian"        # "laplacian", "poisson", "none"
    seam_blend_iters: int = 10
    unseen_fill: str = "neighbor"        # "neighbor", "magenta", "inpaint"


@dataclass
class PipelineConfig:
    camera: CameraConfig
    charuco: CharucoConfig = field(default_factory=CharucoConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    texturing: TexturingConfig = field(default_factory=TexturingConfig)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> PipelineConfig:
    """Load a PipelineConfig from a YAML file."""
    with open(path) as f:
        d = yaml.safe_load(f)

    cam_d = d["camera"]
    camera = CameraConfig(
        fx=float(cam_d["fx"]),
        fy=float(cam_d["fy"]),
        cx=float(cam_d["cx"]),
        cy=float(cam_d["cy"]),
        width=int(cam_d["width"]),
        height=int(cam_d["height"]),
        depth_scale=float(cam_d.get("depth_scale", 1.0)),
    )

    charuco = CharucoConfig()
    if "charuco" in d:
        c = d["charuco"]
        charuco = CharucoConfig(
            squares_x=int(c.get("squares_x", charuco.squares_x)),
            squares_y=int(c.get("squares_y", charuco.squares_y)),
            square_length=float(c.get("square_length", charuco.square_length)),
            marker_length=float(c.get("marker_length", charuco.marker_length)),
            dictionary=c.get("dictionary", charuco.dictionary),
            min_corners=int(c.get("min_corners", charuco.min_corners)),
        )

    seg = SegmentationConfig()
    if "segmentation" in d:
        s = d["segmentation"]
        seg = SegmentationConfig(
            board_plane_inlier_threshold=float(s.get("board_plane_inlier_threshold", seg.board_plane_inlier_threshold)),
            object_min_height=float(s.get("object_min_height", seg.object_min_height)),
            object_max_height=float(s.get("object_max_height", seg.object_max_height)),
            grabcut_iters=int(s.get("grabcut_iters", seg.grabcut_iters)),
        )

    aln = AlignmentConfig()
    if "alignment" in d:
        a = d["alignment"]
        aln = AlignmentConfig(
            method=a.get("method", aln.method),
            init_method=a.get("init_method", aln.init_method),
            optimizer=a.get("optimizer", aln.optimizer),
            max_iter=int(a.get("max_iter", aln.max_iter)),
            iou_warn_threshold=float(a.get("iou_warn_threshold", aln.iou_warn_threshold)),
        )

    tex = TexturingConfig()
    if "texturing" in d:
        t = d["texturing"]
        tex = TexturingConfig(
            atlas_size=int(t.get("atlas_size", tex.atlas_size)),
            view_angle_threshold_deg=float(t.get("view_angle_threshold_deg", tex.view_angle_threshold_deg)),
            seam_blend=t.get("seam_blend", tex.seam_blend),
            seam_blend_iters=int(t.get("seam_blend_iters", tex.seam_blend_iters)),
            unseen_fill=t.get("unseen_fill", tex.unseen_fill),
        )

    return PipelineConfig(camera=camera, charuco=charuco, segmentation=seg, alignment=aln, texturing=tex)


def camera_config_from_info_json(path: str | Path) -> CameraConfig:
    """Build a CameraConfig from a ROS-style color_camera_info.json."""
    with open(path) as f:
        d = json.load(f)
    return CameraConfig(
        fx=float(d["fx"]),
        fy=float(d["fy"]),
        cx=float(d["cx"]),
        cy=float(d["cy"]),
        width=int(d["width"]),
        height=int(d["height"]),
        depth_scale=1.0,  # depth already in metres
    )


def default_config(camera_info_path: str | Path) -> PipelineConfig:
    """Build a PipelineConfig using defaults + camera intrinsics from JSON."""
    camera = camera_config_from_info_json(camera_info_path)
    return PipelineConfig(camera=camera)
