"""Shared geometric helpers: projection, silhouette rendering, pose conversions."""

from __future__ import annotations

import cv2
import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# Pose / rotation helpers
# ---------------------------------------------------------------------------

def pose_vec_to_matrix(x: np.ndarray) -> np.ndarray:
    """Convert 6-DoF vector [tx, ty, tz, rx, ry, rz] to a 4×4 SE(3) matrix.
    Rotation encoded as Rodrigues (axis-angle) vector.
    """
    tx, ty, tz, rx, ry, rz = x
    R, _ = cv2.Rodrigues(np.array([rx, ry, rz], dtype=np.float64))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def matrix_to_pose_vec(T: np.ndarray) -> np.ndarray:
    """Convert 4×4 SE(3) matrix to 6-DoF vector [tx, ty, tz, rx, ry, rz]."""
    R = T[:3, :3].copy()
    rvec, _ = cv2.Rodrigues(R)
    return np.array([T[0, 3], T[1, 3], T[2, 3], *rvec.ravel()], dtype=np.float64)


def invert_pose(T: np.ndarray) -> np.ndarray:
    """Invert a 4×4 rigid-body transform efficiently."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=T.dtype)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_points(
    pts_world: np.ndarray,
    K: np.ndarray,
    world2cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3-D world points into an image.

    Args:
        pts_world: (N, 3) float64 points in world frame.
        K: (3, 3) camera intrinsic matrix.
        world2cam: (4, 4) extrinsic matrix (world → camera).

    Returns:
        pts_2d: (N, 2) pixel coordinates (may be outside image bounds).
        depths: (N,) z-values in camera frame (positive = in front).
    """
    N = len(pts_world)
    pts_h = np.ones((N, 4), dtype=np.float64)
    pts_h[:, :3] = pts_world
    pts_cam = (world2cam @ pts_h.T).T[:, :3]          # (N, 3)
    depths = pts_cam[:, 2]
    # Avoid divide-by-zero
    safe_z = np.where(depths > 0, depths, 1e-6)
    u = (K[0, 0] * pts_cam[:, 0] / safe_z + K[0, 2])
    v = (K[1, 1] * pts_cam[:, 1] / safe_z + K[1, 2])
    pts_2d = np.stack([u, v], axis=1)
    return pts_2d, depths


def backproject_depth(
    depth_m: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a depth image to 3-D points in camera frame.

    Args:
        depth_m: (H, W) float32/float64 depth in metres.
        K: (3, 3) intrinsic matrix.
        mask: optional (H, W) bool; only back-project where True & depth > 0.

    Returns:
        pts_cam: (N, 3) points in camera frame.
        pixel_coords: (N, 2) int array of (v, u) pixel positions.
    """
    H, W = depth_m.shape
    valid = depth_m > 0
    if mask is not None:
        valid = valid & mask.astype(bool)

    ys, xs = np.where(valid)
    z = depth_m[ys, xs].astype(np.float64)
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[1, 1]
    pts_cam = np.stack([x, y, z], axis=1)
    return pts_cam, np.stack([ys, xs], axis=1)


# ---------------------------------------------------------------------------
# Silhouette rendering
# ---------------------------------------------------------------------------

def render_mesh_silhouette(
    mesh: trimesh.Trimesh,
    K: np.ndarray,
    world2cam: np.ndarray,
    obj_pose: np.ndarray,
    img_shape: tuple[int, int],
    scale: float = 1.0,
) -> np.ndarray:
    """Rasterize the silhouette of a mesh into a binary image.

    Args:
        mesh: Trimesh object in its local (object) frame.
        K: (3, 3) camera intrinsic matrix.
        world2cam: (4, 4) world-to-camera extrinsic.
        obj_pose: (4, 4) object-to-world transform (places mesh in world frame).
        img_shape: (H, W) of the output mask.
        scale: optional downscale factor for speed during optimization.

    Returns:
        mask: (H, W) uint8 silhouette image (0 or 255).
    """
    H, W = img_shape[:2]
    out_H, out_W = int(H * scale), int(W * scale)

    # Combined transform: object → world → camera
    obj2cam = world2cam @ obj_pose

    vertices = np.asarray(mesh.vertices, dtype=np.float64)  # (V, 3)
    faces = np.asarray(mesh.faces, dtype=np.int32)           # (F, 3)

    # Transform vertices to camera frame
    V_h = np.ones((len(vertices), 4), dtype=np.float64)
    V_h[:, :3] = vertices
    V_cam = (obj2cam @ V_h.T).T[:, :3]  # (V, 3)

    depths_v = V_cam[:, 2]

    # Scale K for downsampled render
    Ks = K.copy()
    if scale != 1.0:
        Ks = K.copy()
        Ks[0, 0] *= scale
        Ks[1, 1] *= scale
        Ks[0, 2] *= scale
        Ks[1, 2] *= scale

    safe_z = np.where(depths_v > 0, depths_v, 1e-6)
    u = (Ks[0, 0] * V_cam[:, 0] / safe_z + Ks[0, 2])
    v = (Ks[1, 1] * V_cam[:, 1] / safe_z + Ks[1, 2])
    pts_2d = np.stack([u, v], axis=1)  # (V, 2)

    # Keep only faces where all 3 vertices are in front of the camera
    face_depths = depths_v[faces]          # (F, 3)
    front_facing = np.all(face_depths > 0, axis=1)

    # Keep faces with vertices roughly in image bounds (generous margin)
    face_pts = pts_2d[faces]              # (F, 3, 2)
    margin = 50
    in_bounds = (
        np.all(face_pts[:, :, 0] >= -margin, axis=1) &
        np.all(face_pts[:, :, 0] < out_W + margin, axis=1) &
        np.all(face_pts[:, :, 1] >= -margin, axis=1) &
        np.all(face_pts[:, :, 1] < out_H + margin, axis=1)
    )

    valid = front_facing & in_bounds
    tris = face_pts[valid].astype(np.int32)  # (N, 3, 2)

    silhouette = np.zeros((out_H, out_W), dtype=np.uint8)
    if len(tris) > 0:
        cv2.fillPoly(silhouette, tris, 255)

    return silhouette


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute intersection-over-union between two binary masks."""
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return float(intersection) / float(union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Barycentric
# ---------------------------------------------------------------------------

def barycentric_coords(
    p: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> tuple[float, float, float]:
    """Compute barycentric coordinates of point p w.r.t. triangle (v0, v1, v2).
    Works in 2-D. Returns (w0, w1, w2) such that p = w0*v0 + w1*v1 + w2*v2.
    """
    denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
    if abs(denom) < 1e-10:
        return (0.0, 0.0, 0.0)
    w0 = ((v1[1] - v2[1]) * (p[0] - v2[0]) + (v2[0] - v1[0]) * (p[1] - v2[1])) / denom
    w1 = ((v2[1] - v0[1]) * (p[0] - v2[0]) + (v0[0] - v2[0]) * (p[1] - v2[1])) / denom
    w2 = 1.0 - w0 - w1
    return (w0, w1, w2)


def barycentric_coords_batch(
    pts: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> np.ndarray:
    """Vectorized barycentric coords for many points and ONE triangle.

    Args:
        pts: (N, 2) query points.
        v0, v1, v2: (2,) triangle vertices in 2-D.

    Returns:
        bary: (N, 3) barycentric weights.
    """
    denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
    if abs(denom) < 1e-10:
        return np.zeros((len(pts), 3), dtype=np.float64)
    w0 = ((v1[1] - v2[1]) * (pts[:, 0] - v2[0]) + (v2[0] - v1[0]) * (pts[:, 1] - v2[1])) / denom
    w1 = ((v2[1] - v0[1]) * (pts[:, 0] - v2[0]) + (v0[0] - v2[0]) * (pts[:, 1] - v2[1])) / denom
    w2 = 1.0 - w0 - w1
    return np.stack([w0, w1, w2], axis=1)


# ---------------------------------------------------------------------------
# Frame data container
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class FrameData:
    """All data for one captured frame."""
    rgb: np.ndarray           # (H, W, 3) BGR uint8
    depth_m: np.ndarray       # (H, W) float32, metres; 0 where invalid
    cam2world: np.ndarray     # (4, 4) camera-to-world transform
    world2cam: np.ndarray     # (4, 4) world-to-camera extrinsic
    K: np.ndarray             # (3, 3) intrinsic matrix
    name: str = ""            # frame identifier (e.g. capture timestamp)
