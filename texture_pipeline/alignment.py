"""Stage 3 — Mesh-to-scene alignment.

Phase A: coarse initialization (centroid + depth, PCA, exhaustive rotation search,
         or manual point picking).
Phase B: silhouette-based 6-DoF refinement with scipy.optimize.minimize.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Optional

import cv2
import numpy as np
import scipy.optimize
import trimesh

from .config import AlignmentConfig
from .utils import (
    FrameData,
    backproject_depth,
    compute_iou,
    matrix_to_pose_vec,
    pose_vec_to_matrix,
    render_mesh_silhouette,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _euler_xyz_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Return a rotation matrix for intrinsic XYZ Euler angles."""
    sx, cx = np.sin(rx), np.cos(rx)
    sy, cy = np.sin(ry), np.cos(ry)
    sz, cz = np.sin(rz), np.cos(rz)

    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _world_centroid_from_depth(
    frames: list[FrameData],
    masks: list[np.ndarray],
) -> Optional[np.ndarray]:
    """Estimate the 3-D centroid of the masked object in world frame."""
    all_pts: list[np.ndarray] = []
    for frame, mask in zip(frames, masks):
        if np.count_nonzero(mask) == 0:
            continue
        pts_cam, _ = backproject_depth(frame.depth_m, frame.K, mask=(mask > 0))
        if len(pts_cam) == 0:
            continue
        N = len(pts_cam)
        h = np.ones((N, 4), dtype=np.float64)
        h[:, :3] = pts_cam
        pts_world = (frame.cam2world @ h.T).T[:, :3]
        all_pts.append(pts_world)

    if not all_pts:
        return None

    pts = np.concatenate(all_pts, axis=0)
    # Median is more robust than mean for noisy depth
    return np.median(pts, axis=0)


def _candidate_rotations() -> list[np.ndarray]:
    """Generate rotation candidates for exhaustive search.

    Samples at 45° increments — 8³ = 512 rotations, pruned to keep det=+1.
    Also includes the 24 axis-aligned cube rotations for well-aligned objects.
    """
    rotations = []
    angles = np.deg2rad(np.arange(0, 360, 45))
    for rx in angles:
        for ry in angles:
            for rz in angles[::2]:   # 90° steps in z to limit count
                rotations.append(_euler_xyz_matrix(rx, ry, rz))
    return rotations


def _filter_valid_frames(
    T_obj: np.ndarray,
    frames: list[FrameData],
    masks: list[np.ndarray],
) -> tuple[list[FrameData], list[np.ndarray]]:
    """Return only frames where the object centroid projects inside the image."""
    from .utils import project_points
    centroid = T_obj[:3, 3]
    valid_frames, valid_masks = [], []
    for frame, mask in zip(frames, masks):
        if np.count_nonzero(mask) == 0:
            continue
        pt, depth = project_points(centroid[np.newaxis], frame.K, frame.world2cam)
        H, W = frame.rgb.shape[:2]
        if depth[0] > 0 and 0 <= pt[0, 0] < W and 0 <= pt[0, 1] < H:
            valid_frames.append(frame)
            valid_masks.append(mask)
    return valid_frames, valid_masks


def _evaluate_pose_score(
    T_obj: np.ndarray,
    mesh: trimesh.Trimesh,
    frames: list[FrameData],
    masks: list[np.ndarray],
    scale: float = 0.25,
) -> float:
    """Silhouette precision: fraction of the projected mesh inside the foreground mask.

    Unlike IoU, this is robust when the mask is much larger than the object
    (e.g. mask includes a holder/tray around the target object).
    Falls back to IoU when silhouette and mask are similar in area.
    """
    scores = []
    for frame, mask in zip(frames, masks):
        if np.count_nonzero(mask) == 0:
            continue
        sil = render_mesh_silhouette(
            mesh, frame.K, frame.world2cam, T_obj,
            frame.rgb.shape[:2], scale=scale,
        )
        sil_bool = sil > 0
        H, W = sil.shape
        h, w = mask.shape[:2]
        if (h, w) != (H, W):
            mask_s = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            mask_s = mask
        mask_bool = mask_s > 0

        n_sil = sil_bool.sum()
        if n_sil == 0:
            continue

        n_mask = mask_bool.sum()
        n_intersect = (sil_bool & mask_bool).sum()

        # Precision: what fraction of the silhouette is inside the mask
        precision = n_intersect / n_sil
        # Recall: what fraction of the mask is covered by the silhouette (penalises small sil)
        recall = n_intersect / max(n_mask, 1)

        # When mask ≈ object size: use IoU; when mask >> object: use precision
        size_ratio = n_sil / max(n_mask, 1)
        if size_ratio > 0.3:
            # Mask and silhouette are comparable size: use harmonic mean (F1/IoU-like)
            score = compute_iou(sil_bool, mask_bool)
        else:
            # Mask much larger than silhouette (includes holder etc.): precision-dominated
            score = 0.7 * precision + 0.3 * recall

        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Phase A: Coarse initialization
# ---------------------------------------------------------------------------

def init_centroid(
    mesh: trimesh.Trimesh,
    frames: list[FrameData],
    masks: list[np.ndarray],
) -> np.ndarray:
    """Place the mesh centroid at the depth-estimated object centroid, then find
    the best axis-aligned rotation using the precision metric.

    Returns T_obj (4×4).
    """
    centroid_world = _world_centroid_from_depth(frames, masks)
    if centroid_world is None:
        warnings.warn("Could not estimate object centroid from depth; using origin.")
        centroid_world = np.zeros(3)
    print(f"  [align] Depth centroid (world): {centroid_world}")

    mesh_centroid = np.array(mesh.centroid)

    # Quick rotation sweep: 90° steps only (24 orientations)
    best_T = None
    best_score = -1.0
    for rx in [0, np.pi/2, np.pi, 3*np.pi/2]:
        for ry in [0, np.pi/2, np.pi, 3*np.pi/2]:
            for rz in [0, np.pi/2, np.pi, 3*np.pi/2]:
                R = _euler_xyz_matrix(rx, ry, rz)
                if abs(np.linalg.det(R) - 1) > 0.01:
                    continue
                T_obj = np.eye(4, dtype=np.float64)
                T_obj[:3, :3] = R
                T_obj[:3, 3] = centroid_world - R @ mesh_centroid

                vf, vm = _filter_valid_frames(T_obj, frames, masks)
                if not vf:
                    continue
                score = _evaluate_pose_score(T_obj, mesh, vf, vm, scale=0.5)
                if score > best_score:
                    best_score = score
                    best_T = T_obj.copy()

    if best_T is None:
        best_T = np.eye(4, dtype=np.float64)
        best_T[:3, 3] = centroid_world - mesh_centroid

    print(f"  [align] Centroid+rotation init: best score = {best_score:.4f}")
    return best_T


def init_pca(
    mesh: trimesh.Trimesh,
    frames: list[FrameData],
    masks: list[np.ndarray],
) -> np.ndarray:
    """Align mesh principal axes to the point-cloud principal axes.

    Resolves 4-fold sign ambiguity by evaluating silhouette IoU.

    Returns T_obj (4×4).
    """
    # Gather depth point cloud
    all_pts: list[np.ndarray] = []
    for frame, mask in zip(frames, masks):
        if np.count_nonzero(mask) == 0:
            continue
        pts_cam, _ = backproject_depth(frame.depth_m, frame.K, mask=(mask > 0))
        if len(pts_cam) == 0:
            continue
        N = len(pts_cam)
        h = np.ones((N, 4), dtype=np.float64)
        h[:, :3] = pts_cam
        all_pts.append((frame.cam2world @ h.T).T[:, :3])

    if not all_pts:
        return init_centroid(mesh, frames, masks)

    pts = np.concatenate(all_pts, axis=0)
    cloud_centroid = pts.mean(axis=0)
    _, _, Vt_cloud = np.linalg.svd(pts - cloud_centroid, full_matrices=False)
    axes_cloud = Vt_cloud  # rows are principal axes

    # Mesh principal axes
    verts = np.array(mesh.vertices)
    mesh_centroid = verts.mean(axis=0)
    _, _, Vt_mesh = np.linalg.svd(verts - mesh_centroid, full_matrices=False)
    axes_mesh = Vt_mesh

    # Resolve sign ambiguity (2^3 = 8 combinations, but keep det=+1)
    best_T = None
    best_iou = -1.0
    for signs in itertools.product([1, -1], repeat=3):
        R_cloud = np.diag(signs) @ axes_cloud        # signed cloud axes as rows
        R = R_cloud.T @ axes_mesh                    # mesh → cloud rotation
        if np.linalg.det(R) < 0:
            continue
        T_obj = np.eye(4, dtype=np.float64)
        T_obj[:3, :3] = R
        T_obj[:3, 3] = cloud_centroid - R @ mesh_centroid
        iou = _evaluate_pose_score(T_obj, mesh, frames, masks)
        if iou > best_iou:
            best_iou = iou
            best_T = T_obj.copy()

    print(f"  [align] PCA init: best IoU = {best_iou:.3f}")
    return best_T if best_T is not None else init_centroid(mesh, frames, masks)


def init_exhaustive(
    mesh: trimesh.Trimesh,
    frames: list[FrameData],
    masks: list[np.ndarray],
) -> np.ndarray:
    """Fix translation at depth centroid, sweep rotations exhaustively.

    Only evaluates on frames where the object centroid projects inside the image.

    Returns T_obj (4×4).
    """
    centroid_world = _world_centroid_from_depth(frames, masks)
    if centroid_world is None:
        centroid_world = np.zeros(3)
    mesh_centroid = np.array(mesh.centroid)

    rotations = _candidate_rotations()
    best_T = None
    best_score = -1.0

    for R in rotations:
        T_obj = np.eye(4, dtype=np.float64)
        T_obj[:3, :3] = R
        T_obj[:3, 3] = centroid_world - R @ mesh_centroid
        vf, vm = _filter_valid_frames(T_obj, frames, masks)
        if not vf:
            continue
        score = _evaluate_pose_score(T_obj, mesh, vf, vm, scale=0.25)
        if score > best_score:
            best_score = score
            best_T = T_obj.copy()

    print(f"  [align] Exhaustive init: best score = {best_score:.3f}")
    return best_T if best_T is not None else np.eye(4)


def init_manual_points(
    mesh: trimesh.Trimesh,
    frame: FrameData,
) -> np.ndarray:
    """Interactive: pick ≥3 correspondences between image and mesh, solve PnP.

    The user clicks 2-D points in the RGB image (matplotlib) and provides
    matching 3-D points on the mesh (mesh vertex indices or known coordinates).

    Returns T_obj (4×4) in world frame.
    """
    import matplotlib.pyplot as plt

    print("\n  [align] Manual point-picking mode.")
    print("  Click 4 distinctive points on the object in the image.")
    print("  Then provide the corresponding 3-D mesh coordinates when prompted.\n")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(cv2.cvtColor(frame.rgb, cv2.COLOR_BGR2RGB))
    ax.set_title("Click 4+ corresponding points on the object, then close the window")
    clicked = plt.ginput(n=4, timeout=0)
    plt.close(fig)

    if len(clicked) < 3:
        warnings.warn("Need ≥3 points; falling back to centroid init.")
        return init_centroid(mesh, [frame], [np.ones(frame.rgb.shape[:2], dtype=np.uint8) * 255])

    pts_2d = np.array(clicked, dtype=np.float64)
    print(f"\n  You clicked {len(pts_2d)} image points: {pts_2d}")
    print("  Now enter the matching 3-D mesh coordinates (in metres).")
    pts_3d = []
    for i, pt in enumerate(pts_2d):
        print(f"  Point {i+1} (image xy = {pt}): enter x y z (space-separated): ", end="")
        while True:
            try:
                vals = list(map(float, input().split()))
                assert len(vals) == 3
                pts_3d.append(vals)
                break
            except Exception:
                print("  Invalid input, try again: ", end="")

    pts_3d_np = np.array(pts_3d, dtype=np.float64)

    K = frame.K
    dist = np.zeros(5)
    ok, rvec, tvec, _ = cv2.solvePnPRansac(
        pts_3d_np, pts_2d, K, dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        warnings.warn("solvePnP failed; falling back to centroid.")
        return init_centroid(mesh, [frame], [np.ones(frame.rgb.shape[:2], dtype=np.uint8) * 255])

    R_obj_to_cam, _ = cv2.Rodrigues(rvec)
    # T_obj_in_cam: mesh → camera
    T_obj_cam = np.eye(4, dtype=np.float64)
    T_obj_cam[:3, :3] = R_obj_to_cam
    T_obj_cam[:3, 3] = tvec.ravel()

    # Convert to world frame: T_obj_world = cam2world @ T_obj_cam
    T_obj_world = frame.cam2world @ T_obj_cam
    return T_obj_world


# ---------------------------------------------------------------------------
# Phase B: Silhouette-based 6-DoF refinement
# ---------------------------------------------------------------------------

def silhouette_cost(
    x: np.ndarray,
    mesh: trimesh.Trimesh,
    frames: list[FrameData],
    masks: list[np.ndarray],
    depth_weight: float = 0.0,
    render_scale: float = 0.5,
) -> float:
    """Negative precision score across frames where the object is visible."""
    T_obj = pose_vec_to_matrix(x)
    vf, vm = _filter_valid_frames(T_obj, frames, masks)
    if not vf:
        return 1.0  # worst possible score

    score = _evaluate_pose_score(T_obj, mesh, vf, vm, scale=render_scale)

    if depth_weight > 0:
        depth_score = _depth_consistency(T_obj, mesh, vf, vm)
        return -(score + depth_weight * depth_score)

    return -score


def _depth_consistency(
    T_obj: np.ndarray,
    mesh: trimesh.Trimesh,
    frames: list[FrameData],
    masks: list[np.ndarray],
) -> float:
    """Average correlation between rendered depth and observed depth."""
    scores = []
    for frame, mask in zip(frames, masks):
        if np.count_nonzero(mask) == 0:
            continue

        from .utils import project_points
        verts = np.array(mesh.vertices, dtype=np.float64)
        N = len(verts)
        h = np.ones((N, 4))
        h[:, :3] = verts
        obj2world = T_obj
        world_verts = (obj2world @ h.T).T[:, :3]

        pts_2d, depths_cam = project_points(world_verts, frame.K, frame.world2cam)
        valid = depths_cam > 0
        if not np.any(valid):
            continue

        # For each valid projected vertex, compare rendered depth to observed
        u = pts_2d[valid, 0].astype(int)
        v_px = pts_2d[valid, 1].astype(int)
        H, W = frame.depth_m.shape
        in_img = (u >= 0) & (u < W) & (v_px >= 0) & (v_px < H)
        u = u[in_img]
        v_px = v_px[in_img]
        r_depths = depths_cam[valid][in_img]
        o_depths = frame.depth_m[v_px, u]
        valid_obs = o_depths > 0
        if np.sum(valid_obs) < 5:
            continue
        diff = np.abs(r_depths[valid_obs] - o_depths[valid_obs])
        scores.append(1.0 / (1.0 + diff.mean()))

    return float(np.mean(scores)) if scores else 0.0


def silhouette_refine(
    mesh: trimesh.Trimesh,
    frames: list[FrameData],
    masks: list[np.ndarray],
    T_init: np.ndarray,
    aln_cfg: AlignmentConfig,
) -> np.ndarray:
    """Optimize 6-DoF object pose to maximize silhouette IoU.

    Returns T_obj (4×4) in world frame.
    """
    x0 = matrix_to_pose_vec(T_init)

    # Two-phase optimization: coarse (low-res) then fine (full-res)
    result = None
    for render_scale, maxiter in [(0.25, aln_cfg.max_iter // 2), (0.5, aln_cfg.max_iter)]:
        result = scipy.optimize.minimize(
            silhouette_cost,
            x0,
            args=(mesh, frames, masks, 0.0, render_scale),
            method=aln_cfg.optimizer,
            options={"maxiter": maxiter, "disp": False},
        )
        x0 = result.x

    T_opt = pose_vec_to_matrix(result.x)
    vf, vm = _filter_valid_frames(T_opt, frames, masks)
    final_score = _evaluate_pose_score(T_opt, mesh, vf if vf else frames, vm if vf else masks, scale=1.0)
    print(f"  [align] Refinement done. Score = {final_score:.4f} on {len(vf)} frames")

    if final_score < aln_cfg.iou_warn_threshold:
        warnings.warn(
            f"Alignment score ({final_score:.3f}) is below threshold "
            f"({aln_cfg.iou_warn_threshold}). Check the debug overlay images in output/diagnostics/."
        )

    return T_opt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def align_mesh(
    mesh: trimesh.Trimesh,
    frames: list[FrameData],
    masks: list[np.ndarray],
    aln_cfg: AlignmentConfig,
    init_method: str | None = None,
    interactive_frame_idx: int = 0,
) -> np.ndarray:
    """Full alignment pipeline: coarse init + silhouette refinement.

    Args:
        mesh: untextured mesh in local object frame.
        frames: list of FrameData with valid poses.
        masks: per-frame foreground masks (255 = object).
        aln_cfg: alignment config.
        init_method: override for initialization method.
        interactive_frame_idx: which frame to use for manual point picking.

    Returns:
        T_obj (4×4): object-to-world transform.
    """
    method = init_method or aln_cfg.init_method
    print(f"  [align] Coarse initialization: method='{method}'")

    if method == "points":
        T_init = init_manual_points(mesh, frames[interactive_frame_idx])

    elif method == "pca":
        T_init = init_pca(mesh, frames, masks)

    elif method == "exhaustive":
        T_init = init_exhaustive(mesh, frames, masks)

    else:  # "centroid" (default)
        T_init = init_centroid(mesh, frames, masks)

    print(f"  [align] Starting silhouette refinement …")
    T_opt = silhouette_refine(mesh, frames, masks, T_init, aln_cfg)
    return T_opt
