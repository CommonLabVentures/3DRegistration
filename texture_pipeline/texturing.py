"""Stage 4 — Texture projection.

Sub-steps:
  4.1  UV unwrapping via xatlas.
  4.2  Best-view selection per face (frontality × resolution × visibility).
  4.3  Atlas rasterization (vectorised per-face UV sampling).
  4.4  Seam blending (Laplacian or Poisson).
  4.5  Unseen-face fill.
"""

from __future__ import annotations

import warnings
from typing import Optional

import cv2
import numpy as np
import trimesh
from PIL import Image

from .config import TexturingConfig
from .utils import FrameData, barycentric_coords_batch, project_points


def _prepare_projection_masks(
    masks: Optional[list[np.ndarray]],
    dilate_k: int = 9,
) -> Optional[list[Optional[np.ndarray]]]:
    """Dilate non-empty segmentation masks to tolerate small alignment error."""
    if masks is None:
        return None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
    prepared: list[Optional[np.ndarray]] = []
    for mask in masks:
        if mask is None or np.count_nonzero(mask) == 0:
            prepared.append(None)
        else:
            prepared.append(cv2.dilate(mask, kernel, iterations=1))
    return prepared


def _projection_support(
    frame: FrameData,
    mask_img: Optional[np.ndarray],
    img_u: np.ndarray,
    img_v: np.ndarray,
    pred_depth: np.ndarray,
    depth_tol_m: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mask support and soft one-sided occlusion weights for image samples."""
    H, W = frame.depth_m.shape
    u_nn = np.clip(np.rint(img_u).astype(int), 0, W - 1)
    v_nn = np.clip(np.rint(img_v).astype(int), 0, H - 1)

    support = np.ones(len(u_nn), dtype=bool)
    if mask_img is not None:
        support = mask_img[v_nn, u_nn] > 0

    depth_weight = np.ones(len(u_nn), dtype=np.float64)
    obs_depth = frame.depth_m[v_nn, u_nn]
    has_depth = obs_depth > 0
    if np.any(has_depth):
        # Only penalize when the rendered point is behind an observed surface.
        # If observed depth is farther away, it may just be the board/background
        # showing through missing object depth, which should not suppress texture.
        delta = pred_depth[has_depth] - obs_depth[has_depth]
        occluded = delta > depth_tol_m
        depth_weight_slice = np.ones(np.count_nonzero(has_depth), dtype=np.float64)
        depth_weight_slice[occluded] = np.exp(
            -0.5 * ((delta[occluded] - depth_tol_m) / depth_tol_m) ** 2
        )
        depth_weight[has_depth] = depth_weight_slice

    return support, depth_weight


# ---------------------------------------------------------------------------
# 4.1  UV Unwrapping
# ---------------------------------------------------------------------------

def unwrap_uvs(
    mesh: trimesh.Trimesh,
    atlas_size: int = 2048,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """UV-unwrap a mesh using xatlas.

    Returns:
        mesh_uv: new Trimesh with UV coords embedded in visual.
        uvs: (V_new, 2) UV coordinates in [0, 1].
    """
    try:
        import xatlas
    except ImportError:
        warnings.warn(
            "xatlas not installed. Falling back to per-face planar UV projection."
        )
        return _planar_uv_fallback(mesh)

    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.uint32)
    normals = np.array(mesh.vertex_normals, dtype=np.float32)

    vmapping, indices, uvs = xatlas.parametrize(vertices, faces, normals)

    # Build new mesh with remapped vertices
    new_verts = vertices[vmapping]
    mesh_uv = trimesh.Trimesh(
        vertices=new_verts,
        faces=indices,
        process=False,
    )
    # Store original vertex indices for color lookup
    mesh_uv.metadata["vmapping"] = vmapping
    mesh_uv.metadata["uvs"] = uvs            # (V_new, 2) in [0,1]
    mesh_uv.metadata["atlas_size"] = atlas_size

    print(f"  [tex] xatlas: {len(vertices)} → {len(new_verts)} verts, "
          f"{len(faces)} → {len(indices)} faces")
    return mesh_uv, uvs


def _planar_uv_fallback(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Simple per-face planar UV projection as fallback when xatlas is absent."""
    verts = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)

    n_faces = len(faces)
    # Lay out each triangle in a grid in UV space
    cols = int(np.ceil(np.sqrt(n_faces)))
    cell = 1.0 / cols

    new_verts = []
    new_faces = []
    uvs_list = []

    for fi, face in enumerate(faces):
        row = fi // cols
        col = fi % cols
        offset_u = col * cell
        offset_v = row * cell
        tri_verts = verts[face]  # (3, 3)
        base_idx = fi * 3
        new_verts.extend(tri_verts)
        new_faces.append([base_idx, base_idx + 1, base_idx + 2])
        # Small triangle in UV cell
        s = cell * 0.9
        uvs_list.extend([
            [offset_u, offset_v],
            [offset_u + s, offset_v],
            [offset_u, offset_v + s],
        ])

    mesh_uv = trimesh.Trimesh(vertices=np.array(new_verts), faces=np.array(new_faces), process=False)
    uvs = np.array(uvs_list, dtype=np.float32)
    mesh_uv.metadata["uvs"] = uvs
    mesh_uv.metadata["atlas_size"] = 1024
    mesh_uv.metadata["vmapping"] = np.arange(len(np.array(new_verts)))
    return mesh_uv, uvs


# ---------------------------------------------------------------------------
# 4.2  Best-view selection per face
# ---------------------------------------------------------------------------

def _z_buffer_visibility(
    mesh: trimesh.Trimesh,
    K: np.ndarray,
    world2cam: np.ndarray,
    obj_pose: np.ndarray,
    img_shape: tuple[int, int],
) -> np.ndarray:
    """Compute per-face visibility via a software z-buffer.

    Returns:
        visible: (F,) bool array.
    """
    H, W = img_shape[:2]
    obj2cam = world2cam @ obj_pose

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    V_h = np.ones((len(verts), 4))
    V_h[:, :3] = verts
    V_cam = (obj2cam @ V_h.T).T[:, :3]
    depths_v = V_cam[:, 2]

    safe_z = np.where(depths_v > 0, depths_v, 1e-6)
    u_px = (K[0, 0] * V_cam[:, 0] / safe_z + K[0, 2])
    v_px = (K[1, 1] * V_cam[:, 1] / safe_z + K[1, 2])
    pts_2d = np.stack([u_px, v_px], axis=1)

    # Rasterize z-buffer
    zbuf = np.full((H, W), np.inf)
    face_id_buf = np.full((H, W), -1, dtype=np.int32)

    for fi, face in enumerate(faces):
        d = depths_v[face]
        if np.any(d <= 0):
            continue
        tri = pts_2d[face].astype(np.int32)
        x_min = max(tri[:, 0].min(), 0)
        x_max = min(tri[:, 0].max(), W - 1)
        y_min = max(tri[:, 1].min(), 0)
        y_max = min(tri[:, 1].max(), H - 1)
        if x_min > x_max or y_min > y_max:
            continue

        for py in range(y_min, y_max + 1):
            for px in range(x_min, x_max + 1):
                p = np.array([px, py], dtype=np.float64)
                w0, w1, w2 = _bary2d(p, pts_2d[face[0]], pts_2d[face[1]], pts_2d[face[2]])
                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue
                z = w0 * d[0] + w1 * d[1] + w2 * d[2]
                if z < zbuf[py, px]:
                    zbuf[py, px] = z
                    face_id_buf[py, px] = fi

    visible_faces = set(np.unique(face_id_buf[face_id_buf >= 0]))
    return np.array([fi in visible_faces for fi in range(len(faces))])


def _bary2d(p, v0, v1, v2):
    denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
    if abs(denom) < 1e-10:
        return -1, -1, -1
    w0 = ((v1[1] - v2[1]) * (p[0] - v2[0]) + (v2[0] - v1[0]) * (p[1] - v2[1])) / denom
    w1 = ((v2[1] - v0[1]) * (p[0] - v2[0]) + (v0[0] - v2[0]) * (p[1] - v2[1])) / denom
    w2 = 1.0 - w0 - w1
    return w0, w1, w2


def select_best_views(
    mesh_uv: trimesh.Trimesh,
    frames: list[FrameData],
    obj_pose: np.ndarray,
    tex_cfg: TexturingConfig,
    masks: Optional[list[np.ndarray]] = None,
) -> np.ndarray:
    """For each face, select the best camera view.

    Score = frontality × (1/dist²) × in_frame × visibility.

    Returns:
        best_view: (F,) int array, index into frames (-1 = no valid view).
    """
    faces = np.asarray(mesh_uv.faces, dtype=np.int32)
    verts_world = (obj_pose @ np.column_stack([
        mesh_uv.vertices, np.ones(len(mesh_uv.vertices))
    ]).T).T[:, :3]

    face_centers = verts_world[faces].mean(axis=1)       # (F, 3)
    face_normals_local = np.array(mesh_uv.face_normals)  # (F, 3)
    # Transform normals to world frame (rotation only)
    R = obj_pose[:3, :3]
    face_normals_world = (R @ face_normals_local.T).T     # (F, 3)

    # Normalise
    nn = np.linalg.norm(face_normals_world, axis=1, keepdims=True)
    face_normals_world = np.where(nn > 1e-8, face_normals_world / nn, face_normals_world)

    angle_thresh = np.cos(np.deg2rad(tex_cfg.view_angle_threshold_deg))
    n_faces = len(faces)
    best_view = np.full(n_faces, -1, dtype=np.int32)
    best_score = np.full(n_faces, -np.inf)
    projection_masks = _prepare_projection_masks(masks)

    for cam_idx, frame in enumerate(frames):
        cam_pos_world = frame.cam2world[:3, 3]
        K = frame.K
        H, W = frame.rgb.shape[:2]
        mask_img = None if projection_masks is None else projection_masks[cam_idx]

        # View direction for each face (from face center to camera)
        view_dirs = cam_pos_world[np.newaxis, :] - face_centers   # (F, 3)
        dists = np.linalg.norm(view_dirs, axis=1)                  # (F,)
        view_dirs_n = view_dirs / np.maximum(dists[:, np.newaxis], 1e-8)

        # Frontality: dot product of face normal and view direction
        frontality = np.einsum("ij,ij->i", face_normals_world, view_dirs_n)  # (F,)
        back_facing = frontality < angle_thresh  # faces pointing away

        # Resolution proxy: 1/dist²
        resolution = 1.0 / np.maximum(dists ** 2, 1e-6)

        # Project face centers and check in-bounds
        pts_2d, depths = project_points(face_centers, K, frame.world2cam)
        in_frame = (
            (depths > 0) &
            (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < W) &
            (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < H)
        )

        score = frontality * resolution
        score[back_facing] = -np.inf
        score[~in_frame] = -np.inf
        valid_idx = np.where(np.isfinite(score))[0]
        if len(valid_idx) > 0:
            support, depth_weight = _projection_support(
                frame,
                mask_img,
                pts_2d[valid_idx, 0],
                pts_2d[valid_idx, 1],
                depths[valid_idx],
            )
            score[valid_idx] *= depth_weight
            score[valid_idx[~support]] = -np.inf

        improved = score > best_score
        best_score = np.where(improved, score, best_score)
        best_view = np.where(improved, cam_idx, best_view)

    n_assigned = np.sum(best_view >= 0)
    print(f"  [tex] Best-view: {n_assigned}/{n_faces} faces assigned")
    return best_view


# ---------------------------------------------------------------------------
# 4.3  Atlas rasterisation
# ---------------------------------------------------------------------------

def rasterize_atlas(
    mesh_uv: trimesh.Trimesh,
    uvs: np.ndarray,
    frames: list[FrameData],
    obj_pose: np.ndarray,
    best_view: np.ndarray,
    atlas_size: int,
    masks: Optional[list[np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Paint the texture atlas by projecting each UV-triangle back to its best camera.

    Args:
        mesh_uv: Trimesh with remapped vertices.
        uvs: (V_uv, 2) UV coordinates in [0, 1].
        frames: list of FrameData.
        obj_pose: (4, 4) object-to-world.
        best_view: (F,) per-face camera index.
        atlas_size: texture atlas resolution.

    Returns:
        atlas: (H_atlas, W_atlas, 3) uint8 BGR atlas.
        covered: (H_atlas, W_atlas) bool, True where a pixel was written.
    """
    A = atlas_size
    atlas = np.zeros((A, A, 3), dtype=np.uint8)
    covered = np.zeros((A, A), dtype=bool)

    faces = np.asarray(mesh_uv.faces, dtype=np.int32)
    verts_local = np.asarray(mesh_uv.vertices, dtype=np.float64)
    projection_masks = _prepare_projection_masks(masks)

    # Transform mesh vertices to world frame once
    N = len(verts_local)
    V_h = np.ones((N, 4))
    V_h[:, :3] = verts_local
    verts_world = (obj_pose @ V_h.T).T[:, :3]

    for fi, face in enumerate(faces):
        cam_idx = int(best_view[fi])
        if cam_idx < 0:
            continue

        frame = frames[cam_idx]
        K = frame.K
        H_img, W_img = frame.rgb.shape[:2]
        mask_img = None if projection_masks is None else projection_masks[cam_idx]

        # UV triangle (in pixel space on atlas)
        uv_tri = uvs[face]  # (3, 2) in [0, 1]
        uv_px = uv_tri * (A - 1)  # (3, 2)

        # Bounding box of UV triangle
        u_min = max(int(np.floor(uv_px[:, 0].min())), 0)
        u_max = min(int(np.ceil(uv_px[:, 0].max())), A - 1)
        v_min = max(int(np.floor(uv_px[:, 1].min())), 0)
        v_max = min(int(np.ceil(uv_px[:, 1].max())), A - 1)

        if u_min > u_max or v_min > v_max:
            continue

        # Grid of candidate atlas pixels
        uu, vv = np.meshgrid(
            np.arange(u_min, u_max + 1, dtype=np.float64),
            np.arange(v_min, v_max + 1, dtype=np.float64),
        )
        pts_uv = np.stack([uu.ravel(), vv.ravel()], axis=1)  # (M, 2)

        # Barycentric coords in UV space
        bary = barycentric_coords_batch(pts_uv, uv_px[0], uv_px[1], uv_px[2])
        inside = np.all(bary >= -1e-6, axis=1)

        if not np.any(inside):
            continue

        bary_in = bary[inside]                           # (K, 3)
        pts_uv_in = pts_uv[inside].astype(int)           # (K, 2)

        # Interpolate 3-D world position from barycentric weights
        p3d_world = (
            bary_in[:, 0:1] * verts_world[face[0]] +
            bary_in[:, 1:2] * verts_world[face[1]] +
            bary_in[:, 2:3] * verts_world[face[2]]
        )  # (K, 3)

        # Project to camera image
        p3d_cam_h = np.ones((len(p3d_world), 4))
        p3d_cam_h[:, :3] = p3d_world
        p3d_cam = (frame.world2cam @ p3d_cam_h.T).T[:, :3]

        depths = p3d_cam[:, 2]
        valid = depths > 0
        if not np.any(valid):
            continue

        safe_z = np.where(depths > 0, depths, 1e-6)
        img_u = K[0, 0] * p3d_cam[:, 0] / safe_z + K[0, 2]
        img_v = K[1, 1] * p3d_cam[:, 1] / safe_z + K[1, 2]

        in_img = (
            valid &
            (img_u >= 0) & (img_u < W_img - 1) &
            (img_v >= 0) & (img_v < H_img - 1)
        )

        if not np.any(in_img):
            continue

        # Bilinear sample from RGB image
        img_u_v = img_u[in_img]
        img_v_v = img_v[in_img]
        atlas_u = pts_uv_in[in_img, 0]
        atlas_v = pts_uv_in[in_img, 1]
        proj_depth = depths[in_img]

        support, depth_weight = _projection_support(
            frame,
            mask_img,
            img_u_v,
            img_v_v,
            proj_depth,
        )
        keep = support & (depth_weight > 0.2)
        if not np.any(keep):
            continue

        img_u_v = img_u_v[keep]
        img_v_v = img_v_v[keep]
        atlas_u = atlas_u[keep]
        atlas_v = atlas_v[keep]

        # Bilinear interpolation
        x0 = img_u_v.astype(int)
        y0 = img_v_v.astype(int)
        x1 = np.clip(x0 + 1, 0, W_img - 1)
        y1 = np.clip(y0 + 1, 0, H_img - 1)
        fx = img_u_v - x0
        fy = img_v_v - y0

        rgb = frame.rgb.astype(np.float32)
        c00 = rgb[y0, x0]     # (K, 3)
        c10 = rgb[y0, x1]
        c01 = rgb[y1, x0]
        c11 = rgb[y1, x1]

        colors = (
            c00 * (1 - fx[:, None]) * (1 - fy[:, None]) +
            c10 * fx[:, None] * (1 - fy[:, None]) +
            c01 * (1 - fx[:, None]) * fy[:, None] +
            c11 * fx[:, None] * fy[:, None]
        ).astype(np.uint8)

        atlas[atlas_v, atlas_u] = colors
        covered[atlas_v, atlas_u] = True

    n_covered = np.count_nonzero(covered)
    pct = 100.0 * n_covered / (A * A)
    print(f"  [tex] Atlas: {n_covered} / {A*A} pixels covered ({pct:.1f}%)")
    return atlas, covered


# ---------------------------------------------------------------------------
# 4.4  Seam blending
# ---------------------------------------------------------------------------

def blend_seams_laplacian(
    atlas: np.ndarray,
    covered: np.ndarray,
    n_iters: int = 10,
) -> np.ndarray:
    """Iterative Laplacian smoothing along seam edges in the atlas.

    Smooths the covered region by averaging with neighbours, leaving
    fully uncovered pixels untouched.
    """
    result = atlas.astype(np.float32)
    mask_f = covered.astype(np.float32)

    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32) / 4.0

    for _ in range(n_iters):
        neighbor_sum = cv2.filter2D(result, -1, kernel)
        # Only update covered pixels; blend 10% toward neighbourhood
        result = np.where(covered[:, :, None], result * 0.9 + neighbor_sum * 0.1, result)

    return np.clip(result, 0, 255).astype(np.uint8)


def blend_seams_poisson(
    atlas: np.ndarray,
    covered: np.ndarray,
) -> np.ndarray:
    """Poisson blending over the covered atlas region.

    Uses cv2.inpaint with a thin seam mask as the blending domain.
    """
    # Find seam pixels: covered pixels adjacent to other covered pixels with
    # different source (proxy: gradient magnitude above threshold).
    gray = cv2.cvtColor(atlas, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad = cv2.Laplacian(gray, cv2.CV_32F)
    seam = (np.abs(grad) > 15) & covered
    seam_u8 = seam.astype(np.uint8) * 255

    if np.count_nonzero(seam_u8) == 0:
        return atlas

    blended = cv2.inpaint(atlas, seam_u8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return blended


# ---------------------------------------------------------------------------
# 4.5  Unseen face fill
# ---------------------------------------------------------------------------

def fill_unseen(
    atlas: np.ndarray,
    covered: np.ndarray,
    method: str = "neighbor",
) -> np.ndarray:
    """Fill uncovered atlas pixels.

    Args:
        method: "neighbor" (inpaint), "magenta" (mark), "inpaint" (cv2.INPAINT_NS).
    """
    if method == "magenta":
        result = atlas.copy()
        result[~covered] = [255, 0, 255]  # magenta in BGR
        return result

    if np.all(covered):
        return atlas

    uncovered_u8 = (~covered).astype(np.uint8) * 255

    if method == "inpaint":
        return cv2.inpaint(atlas, uncovered_u8, inpaintRadius=5, flags=cv2.INPAINT_NS)

    # "neighbor": fast inpaint (TELEA)
    return cv2.inpaint(atlas, uncovered_u8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def texture_mesh(
    mesh: trimesh.Trimesh,
    frames: list[FrameData],
    obj_pose: np.ndarray,
    tex_cfg: TexturingConfig,
    masks: Optional[list[np.ndarray]] = None,
) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    """Full texturing pipeline.

    Returns:
        mesh_uv: Trimesh with UV coordinates in metadata.
        uvs: (V_uv, 2) UV coordinates.
        atlas: (A, A, 3) uint8 BGR texture atlas.
    """
    # 4.1 UV unwrap
    print("  [tex] UV unwrapping …")
    mesh_uv, uvs = unwrap_uvs(mesh, atlas_size=tex_cfg.atlas_size)

    # 4.2 Best-view selection
    print("  [tex] Selecting best view per face …")
    best_view = select_best_views(mesh_uv, frames, obj_pose, tex_cfg, masks=masks)

    # 4.3 Atlas rasterisation
    print("  [tex] Rasterising atlas …")
    atlas, covered = rasterize_atlas(
        mesh_uv, uvs, frames, obj_pose, best_view, tex_cfg.atlas_size, masks=masks
    )

    # 4.4 Seam blending
    print(f"  [tex] Seam blending ({tex_cfg.seam_blend}) …")
    if tex_cfg.seam_blend == "poisson":
        atlas = blend_seams_poisson(atlas, covered)
    elif tex_cfg.seam_blend == "laplacian":
        atlas = blend_seams_laplacian(atlas, covered, n_iters=tex_cfg.seam_blend_iters)
    # else "none" — skip

    # 4.5 Fill unseen faces
    print(f"  [tex] Filling unseen faces ({tex_cfg.unseen_fill}) …")
    atlas = fill_unseen(atlas, covered, method=tex_cfg.unseen_fill)

    return mesh_uv, uvs, atlas
