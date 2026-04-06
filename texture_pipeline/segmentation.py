"""Stage 2 — Object segmentation.

Layered strategy:
  1. Back-project valid depth pixels to world frame.
  2. RANSAC plane fit to the dominant flat surface (the board / table).
  3. Height-threshold everything above the plane → coarse object mask.
  4. GrabCut refinement using the coarse mask as seed.
"""

from __future__ import annotations

import warnings
from typing import Optional

import cv2
import numpy as np

from .config import SegmentationConfig
from .utils import FrameData, backproject_depth, invert_pose


# ---------------------------------------------------------------------------
# Board plane fitting
# ---------------------------------------------------------------------------

def _to_world(pts_cam: np.ndarray, cam2world: np.ndarray) -> np.ndarray:
    """Transform (N, 3) points from camera frame to world frame."""
    N = len(pts_cam)
    h = np.ones((N, 4), dtype=np.float64)
    h[:, :3] = pts_cam
    return (cam2world @ h.T).T[:, :3]


def fit_plane_ransac(
    pts: np.ndarray,
    inlier_thresh: float = 0.008,
    n_iterations: int = 500,
    min_inliers: int = 50,
) -> Optional[np.ndarray]:
    """RANSAC plane fit for a set of 3-D points.

    Returns:
        plane: (4,) coefficients [a, b, c, d] s.t. ax+by+cz+d=0,
               normalised so that ||(a,b,c)||=1, or None if failed.
    """
    if len(pts) < 3:
        return None

    best_plane = None
    best_count = 0
    rng = np.random.default_rng(0)

    for _ in range(n_iterations):
        idx = rng.choice(len(pts), 3, replace=False)
        p0, p1, p2 = pts[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-8:
            continue
        normal /= norm_len
        d = -normal @ p0
        dists = np.abs(pts @ normal + d)
        count = np.sum(dists < inlier_thresh)
        if count > best_count:
            best_count = count
            best_plane = np.append(normal, d)

    if best_plane is None or best_count < min_inliers:
        return None

    # Refine with all inliers
    dists = np.abs(pts @ best_plane[:3] + best_plane[3])
    inliers = pts[dists < inlier_thresh]
    if len(inliers) < 3:
        return best_plane

    # Least-squares fit
    centroid = inliers.mean(axis=0)
    _, _, Vt = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = Vt[-1]
    d = -normal @ centroid
    # Ensure normal points upward (positive z in world = up)
    if normal[2] < 0:
        normal = -normal
        d = -d
    return np.append(normal, d)


def fit_board_plane(
    frame: FrameData,
    seg_cfg: SegmentationConfig,
) -> Optional[np.ndarray]:
    """Fit a plane to the dominant flat surface visible in one frame.

    Returns plane (4,) in world frame, or None.
    """
    pts_cam, _ = backproject_depth(frame.depth_m, frame.K)
    if len(pts_cam) < 50:
        return None

    pts_world = _to_world(pts_cam, frame.cam2world)

    # Only keep points near z≈0 in world (the board lives near the floor/table)
    # Use a generous window to capture the board regardless of tilt.
    z_median = np.median(pts_world[:, 2])
    near_ground = np.abs(pts_world[:, 2] - z_median) < 0.15  # 15 cm window
    pts_board = pts_world[near_ground]

    return fit_plane_ransac(
        pts_board,
        inlier_thresh=seg_cfg.board_plane_inlier_threshold,
    )


def aggregate_plane(frames: list[FrameData], seg_cfg: SegmentationConfig) -> Optional[np.ndarray]:
    """Fit a robust board plane by aggregating depth across all frames."""
    all_pts: list[np.ndarray] = []
    for frame in frames:
        pts_cam, _ = backproject_depth(frame.depth_m, frame.K)
        if len(pts_cam) < 10:
            continue
        pts_world = _to_world(pts_cam, frame.cam2world)
        all_pts.append(pts_world)

    if not all_pts:
        return None

    all_pts_np = np.concatenate(all_pts, axis=0)

    z_median = np.median(all_pts_np[:, 2])
    near_ground = np.abs(all_pts_np[:, 2] - z_median) < 0.15
    pts_board = all_pts_np[near_ground]

    return fit_plane_ransac(
        pts_board,
        inlier_thresh=seg_cfg.board_plane_inlier_threshold,
    )


def height_above_plane(pts_world: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """Compute signed height of each world point above the plane."""
    return pts_world @ plane[:3] + plane[3]


# ---------------------------------------------------------------------------
# Per-frame mask generation
# ---------------------------------------------------------------------------

def depth_height_mask(
    frame: FrameData,
    plane: np.ndarray,
    seg_cfg: SegmentationConfig,
) -> np.ndarray:
    """Return a binary mask of pixels whose depth back-projects to a point
    that is between object_min_height and object_max_height above the plane.

    Returns (H, W) uint8 mask.
    """
    H, W = frame.depth_m.shape
    mask = np.zeros((H, W), dtype=np.uint8)

    pts_cam, pixel_coords = backproject_depth(frame.depth_m, frame.K)
    if len(pts_cam) == 0:
        return mask

    pts_world = _to_world(pts_cam, frame.cam2world)
    heights = height_above_plane(pts_world, plane)

    valid = (
        (heights >= seg_cfg.object_min_height) &
        (heights <= seg_cfg.object_max_height)
    )

    ys = pixel_coords[valid, 0]
    xs = pixel_coords[valid, 1]
    mask[ys, xs] = 255
    return mask


def grabcut_refine(
    rgb_bgr: np.ndarray,
    coarse_mask: np.ndarray,
    n_iter: int = 5,
    erode_k: int = 5,
    dilate_k: int = 15,
) -> np.ndarray:
    """Refine a coarse binary mask using GrabCut.

    Args:
        rgb_bgr: (H, W, 3) BGR image.
        coarse_mask: (H, W) uint8, non-zero = probable foreground.
        n_iter: GrabCut iterations.
        erode_k: erosion kernel size for definite-foreground seed.
        dilate_k: dilation kernel size for definite-background border.

    Returns:
        refined_mask: (H, W) uint8, 255 = foreground.
    """
    H, W = rgb_bgr.shape[:2]

    if np.count_nonzero(coarse_mask) == 0:
        return coarse_mask.copy()

    # GrabCut model buffers
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Build mask for GrabCut
    gc_mask = np.full((H, W), cv2.GC_PR_BGD, dtype=np.uint8)

    # Definite foreground: heavily eroded coarse mask
    kern_fg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
    def_fg = cv2.erode(coarse_mask, kern_fg, iterations=2)

    # Definite background: pixels far outside dilated mask
    kern_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
    def_bg_region = cv2.dilate(coarse_mask, kern_bg, iterations=2)
    gc_mask[def_bg_region == 0] = cv2.GC_BGD

    # Probable foreground: dilated coarse mask
    gc_mask[coarse_mask > 0] = cv2.GC_PR_FGD
    gc_mask[def_fg > 0] = cv2.GC_FGD

    try:
        cv2.grabCut(rgb_bgr, gc_mask, None, bgd_model, fgd_model, n_iter, cv2.GC_INIT_WITH_MASK)
    except cv2.error as e:
        warnings.warn(f"GrabCut failed: {e}")
        return coarse_mask.copy()

    refined = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return refined


def segment_frame(
    frame: FrameData,
    plane: np.ndarray,
    seg_cfg: SegmentationConfig,
    use_grabcut: bool = True,
) -> np.ndarray:
    """Segment objects from board in one frame.

    Returns (H, W) uint8 mask (255 = object).
    """
    coarse = depth_height_mask(frame, plane, seg_cfg)

    # Morphological cleanup of the coarse mask
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    coarse = cv2.morphologyEx(coarse, cv2.MORPH_CLOSE, kern, iterations=2)
    coarse = cv2.morphologyEx(coarse, cv2.MORPH_OPEN, kern, iterations=1)

    if use_grabcut and np.count_nonzero(coarse) > 0:
        return grabcut_refine(frame.rgb, coarse, n_iter=seg_cfg.grabcut_iters)
    return coarse


def segment_all_frames(
    frames: list[FrameData],
    seg_cfg: SegmentationConfig,
    plane: Optional[np.ndarray] = None,
    use_grabcut: bool = True,
) -> list[np.ndarray]:
    """Segment objects in all frames. Returns per-frame masks (H, W) uint8.

    Args:
        frames: list of FrameData.
        seg_cfg: segmentation config.
        plane: pre-fitted world-frame plane; if None, will be fitted here.
        use_grabcut: whether to run GrabCut refinement.

    Returns:
        masks: list of (H, W) uint8 masks (255 = foreground object).
    """
    if plane is None:
        print("  [seg] Fitting board plane from all frames …")
        plane = aggregate_plane(frames, seg_cfg)
        if plane is None:
            raise RuntimeError(
                "Board plane fitting failed. "
                "Check that depth covers the board surface."
            )
        print(f"  [seg] Board plane normal: {plane[:3]}, d={plane[3]:.4f}")

    masks = []
    for i, frame in enumerate(frames):
        mask = segment_frame(frame, plane, seg_cfg, use_grabcut=use_grabcut)
        n_px = np.count_nonzero(mask)
        if n_px == 0:
            print(f"  [seg] Frame {frame.name}: empty mask — skipping")
        else:
            print(f"  [seg] Frame {frame.name}: {n_px} foreground pixels")
        masks.append(mask)

    return masks


# ---------------------------------------------------------------------------
# Multi-object separation
# ---------------------------------------------------------------------------

def separate_objects(
    mask: np.ndarray,
    n_objects: int | None = None,
    min_area_px: int = 200,
) -> list[np.ndarray]:
    """Split a combined foreground mask into per-object masks via connected components.

    Args:
        mask: (H, W) uint8 foreground mask.
        n_objects: expected number of objects; if None, returns all CCs above min_area_px.
        min_area_px: minimum CC area to be considered an object.

    Returns:
        list of (H, W) uint8 masks, one per object, sorted by area (largest first).
    """
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Skip background (label 0)
    components = []
    for label in range(1, n_cc):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area_px:
            comp_mask = (labels == label).astype(np.uint8) * 255
            components.append((area, comp_mask))

    components.sort(key=lambda x: x[0], reverse=True)

    if n_objects is not None:
        components = components[:n_objects]

    return [m for _, m in components]
