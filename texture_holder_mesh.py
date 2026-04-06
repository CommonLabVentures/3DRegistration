#!/usr/bin/env python3
"""Align and texture the holder mesh from the holder capture set."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import trimesh
import xatlas
from PIL import Image
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from extract_holder_cutouts import board_polygon_mask, extract_mask, load_metadata


@dataclass
class FrameData:
    frame_id: str
    rgb_bgr: np.ndarray
    depth_m: np.ndarray
    mask: np.ndarray
    world2cam: np.ndarray
    cam2world: np.ndarray
    optical_axis_angle_deg: float
    camera_position_world_m: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Align and texture the battery holder mesh.")
    parser.add_argument("--clean-data-dir", default="holder_clean_dataset", help="Flattened holder rgb/depth dataset.")
    parser.add_argument(
        "--pose-dir",
        default="holder_subset_dataset",
        help="Directory containing all_pose_estimates.json and dataset_metadata.json.",
    )
    parser.add_argument("--mesh-path", default="holder_dataset/battery-holder.stl", help="Holder STL path.")
    parser.add_argument("--output-dir", default="holder_output", help="Output directory.")
    parser.add_argument("--atlas-size", type=int, default=2048, help="Texture atlas size.")
    return parser


def load_frames(clean_data_dir: Path, pose_dir: Path) -> list[FrameData]:
    intr, spec = load_metadata(pose_dir)
    records = json.loads((pose_dir / "all_pose_estimates.json").read_text(encoding="utf-8"))
    frames: list[FrameData] = []
    for record in records:
        if not record.get("pose_detected"):
            continue
        frame_id = record["frame_id"]
        rgb_bgr = cv2.imread(str(clean_data_dir / "rgb" / f"{frame_id}.jpg"), cv2.IMREAD_COLOR)
        depth_m = np.load(str(clean_data_dir / "depth" / f"{frame_id}.npz"))["depth"].astype(np.float32)
        world2cam = np.array(record["world2cam"], dtype=np.float64)
        cam2world = np.array(record["cam2world"], dtype=np.float64)
        board_mask = board_polygon_mask(depth_m.shape, intr, spec, world2cam)
        mask = extract_mask(rgb_bgr, board_mask)
        frames.append(
            FrameData(
                frame_id=frame_id,
                rgb_bgr=rgb_bgr,
                depth_m=depth_m,
                mask=mask,
                world2cam=world2cam,
                cam2world=cam2world,
                optical_axis_angle_deg=float(record["optical_axis_angle_deg"]),
                camera_position_world_m=np.array(record["camera_position_world_m"], dtype=np.float64),
            )
        )
    return frames


def normalize_mesh(mesh_path: Path) -> tuple[trimesh.Trimesh, float]:
    mesh = trimesh.load(mesh_path, force="mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(mesh.faces, dtype=np.int32).copy()
    scale = 0.001 if float(mesh.extents.max()) > 1.0 else 1.0
    vertices *= scale
    vertices[:, 0] -= 0.5 * (vertices[:, 0].min() + vertices[:, 0].max())
    vertices[:, 1] -= 0.5 * (vertices[:, 1].min() + vertices[:, 1].max())
    vertices[:, 2] -= vertices[:, 2].min()
    vertices[:, 2] *= -1.0
    faces = faces[:, [0, 2, 1]]
    normalized = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    normalized.fix_normals()
    return normalized, scale


def masked_world_points(frame: FrameData, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    ys, xs = np.where((frame.mask > 0) & np.isfinite(frame.depth_m) & (frame.depth_m > 0))
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float64)
    z = frame.depth_m[ys, xs].astype(np.float64)
    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z
    points_cam = np.stack([x, y, z], axis=0)
    return (frame.cam2world[:3, :3] @ points_cam + frame.cam2world[:3, 3:4]).T


def build_alignment_cloud(frames: list[FrameData], intr) -> np.ndarray:
    normal_frames = [frame for frame in frames if frame.optical_axis_angle_deg <= 5.0]
    if not normal_frames:
        normal_frames = frames
    all_points = [masked_world_points(frame, intr.fx, intr.fy, intr.cx, intr.cy) for frame in normal_frames]
    merged = np.concatenate([pts for pts in all_points if len(pts) > 0], axis=0)
    merged = merged[(merged[:, 2] > -0.04) & (merged[:, 2] < 0.005)]
    return merged


def rigid_transform(tx: float, ty: float, yaw: float, tz: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transform[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return transform


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


def alignment_cost(params: np.ndarray, observed_points: np.ndarray, sampled_points_local: np.ndarray) -> float:
    tx, ty, yaw, tz = params
    transform = rigid_transform(tx, ty, yaw, tz)
    transformed = apply_transform(sampled_points_local, transform)
    tree = cKDTree(transformed)
    distances, _ = tree.query(observed_points, k=1)
    return float(np.mean(np.clip(distances, 0.0, 0.02)))


def align_mesh(mesh: trimesh.Trimesh, observed_points: np.ndarray) -> tuple[np.ndarray, dict]:
    sampled_points_local = mesh.sample(8000)
    observed_xy_centroid = observed_points[:, :2].mean(axis=0)

    coarse_best = None
    for deg in range(0, 360, 15):
        yaw = math.radians(deg)
        params = np.array([observed_xy_centroid[0], observed_xy_centroid[1], yaw, 0.0], dtype=np.float64)
        score = alignment_cost(params, observed_points, sampled_points_local)
        if coarse_best is None or score < coarse_best[0]:
            coarse_best = (score, params)

    assert coarse_best is not None
    result = minimize(
        alignment_cost,
        coarse_best[1],
        args=(observed_points, sampled_points_local),
        method="Powell",
        options={"maxiter": 200, "xtol": 1e-4, "ftol": 1e-5},
    )
    transform = rigid_transform(*result.x)
    report = {
        "coarse_score_m": float(coarse_best[0]),
        "final_score_m": float(result.fun),
        "optimized_params": {
            "tx_m": float(result.x[0]),
            "ty_m": float(result.x[1]),
            "yaw_deg": float(np.degrees(result.x[2])),
            "tz_m": float(result.x[3]),
        },
        "success": bool(result.success),
        "message": str(result.message),
    }
    return transform, report


def transform_mesh_vertices(mesh: trimesh.Trimesh, transform: np.ndarray) -> np.ndarray:
    return apply_transform(np.asarray(mesh.vertices, dtype=np.float64), transform)


def project_points(points_world: np.ndarray, world2cam: np.ndarray, intr) -> tuple[np.ndarray, np.ndarray]:
    points_cam = (world2cam[:3, :3] @ points_world.T).T + world2cam[:3, 3]
    uvw = (intr.K @ points_cam.T).T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-9)
    return uv, points_cam[:, 2]


def render_overlay(rgb_bgr: np.ndarray, vertices_world: np.ndarray, faces: np.ndarray, frame: FrameData, intr) -> np.ndarray:
    overlay = rgb_bgr.copy()
    uv, z = project_points(vertices_world, frame.world2cam, intr)
    for tri in faces:
        tri = np.asarray(tri, dtype=np.int32)
        if np.any(z[tri] <= 1e-4):
            continue
        pts = np.round(uv[tri]).astype(np.int32)
        if np.any(pts[:, 0] < -2) or np.any(pts[:, 0] > intr.width + 2) or np.any(pts[:, 1] < -2) or np.any(pts[:, 1] > intr.height + 2):
            continue
        cv2.polylines(overlay, [pts.reshape(-1, 1, 2)], True, (0, 255, 255), 1, cv2.LINE_AA)
    return overlay


def bilinear_sample_color(image_bgr: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    xs = np.clip(xs, 0.0, w - 1.001)
    ys = np.clip(ys, 0.0, h - 1.001)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = xs - x0
    wy = ys - y0
    c00 = image_bgr[y0, x0].astype(np.float32)
    c10 = image_bgr[y0, x1].astype(np.float32)
    c01 = image_bgr[y1, x0].astype(np.float32)
    c11 = image_bgr[y1, x1].astype(np.float32)
    top = c00 * (1.0 - wx[:, None]) + c10 * wx[:, None]
    bottom = c01 * (1.0 - wx[:, None]) + c11 * wx[:, None]
    return top * (1.0 - wy[:, None]) + bottom * wy[:, None]


def bilinear_sample_scalar(image: np.ndarray, x: float, y: float) -> float:
    h, w = image.shape[:2]
    x = float(np.clip(x, 0.0, w - 1.001))
    y = float(np.clip(y, 0.0, h - 1.001))
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    wx = x - x0
    wy = y - y0
    c00 = float(image[y0, x0])
    c10 = float(image[y0, x1])
    c01 = float(image[y1, x0])
    c11 = float(image[y1, x1])
    return (1.0 - wy) * ((1.0 - wx) * c00 + wx * c10) + wy * ((1.0 - wx) * c01 + wx * c11)


def face_world_normals(vertices_world: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices_world[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(norms, 1e-9)
    return normals


def choose_face_views(vertices_world: np.ndarray, faces: np.ndarray, frames: list[FrameData], intr) -> list[int]:
    normals = face_world_normals(vertices_world, faces)
    centers = vertices_world[faces].mean(axis=1)
    face_views: list[int] = []
    dilated_masks = [cv2.dilate(frame.mask, np.ones((9, 9), dtype=np.uint8), iterations=1) for frame in frames]

    for face_idx, face in enumerate(faces):
        best_view = -1
        best_score = -1.0
        tri_world = vertices_world[face]
        normal = normals[face_idx]
        center = centers[face_idx]
        for frame_idx, frame in enumerate(frames):
            ray = frame.camera_position_world_m - center
            distance = float(np.linalg.norm(ray))
            if distance <= 1e-6:
                continue
            ray /= distance
            frontality = float(np.dot(normal, ray))
            if frontality <= 0.02:
                continue

            uv, z = project_points(tri_world, frame.world2cam, intr)
            if np.any(z <= 1e-4):
                continue
            if np.any(uv[:, 0] < 1.0) or np.any(uv[:, 0] > intr.width - 2.0) or np.any(uv[:, 1] < 1.0) or np.any(uv[:, 1] > intr.height - 2.0):
                continue

            center_uv, center_z = project_points(center[None, :], frame.world2cam, intr)
            center_x = float(center_uv[0, 0])
            center_y = float(center_uv[0, 1])
            if bilinear_sample_scalar(dilated_masks[frame_idx], center_x, center_y) < 64.0:
                continue

            depth_score = 1.0
            depth_value = bilinear_sample_scalar(frame.depth_m, center_x, center_y)
            if depth_value > 0.0:
                depth_error = abs(depth_value - float(center_z[0]))
                depth_score = max(0.0, 1.0 - depth_error / 0.01)
                if depth_score <= 0.0:
                    continue

            score = frontality * depth_score / (distance * distance)
            if score > best_score:
                best_score = score
                best_view = frame_idx
        face_views.append(best_view)
    return face_views


def rasterize_texture(
    vertices_world: np.ndarray,
    faces: np.ndarray,
    frames: list[FrameData],
    intr,
    atlas_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vmapping, atlas_faces, atlas_uvs = xatlas.parametrize(
        np.asarray(vertices_world, dtype=np.float32),
        np.asarray(faces, dtype=np.int32),
    )
    atlas_faces = np.asarray(atlas_faces, dtype=np.int32)
    atlas_uvs = np.asarray(atlas_uvs, dtype=np.float32)
    vmapping = np.asarray(vmapping, dtype=np.int32)
    remapped_vertices = vertices_world[vmapping]
    face_views = choose_face_views(vertices_world, faces, frames, intr)

    atlas = np.zeros((atlas_size, atlas_size, 3), dtype=np.uint8)
    filled = np.zeros((atlas_size, atlas_size), dtype=np.uint8)
    dilated_masks = [cv2.dilate(frame.mask, np.ones((9, 9), dtype=np.uint8), iterations=1) for frame in frames]

    for face_idx, atlas_face in enumerate(atlas_faces):
        frame_idx = face_views[face_idx]
        if frame_idx < 0:
            continue
        frame = frames[frame_idx]

        uv = atlas_uvs[atlas_face]
        uv_pixels = np.column_stack(
            [
                uv[:, 0] * (atlas_size - 1),
                (1.0 - uv[:, 1]) * (atlas_size - 1),
            ]
        )
        tri_world = remapped_vertices[atlas_face]

        min_x = max(0, int(np.floor(np.min(uv_pixels[:, 0]))))
        max_x = min(atlas_size - 1, int(np.ceil(np.max(uv_pixels[:, 0]))))
        min_y = max(0, int(np.floor(np.min(uv_pixels[:, 1]))))
        max_y = min(atlas_size - 1, int(np.ceil(np.max(uv_pixels[:, 1]))))
        if min_x > max_x or min_y > max_y:
            continue

        xs, ys = np.meshgrid(np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1))
        sample_points = np.stack([xs + 0.5, ys + 0.5], axis=-1).reshape(-1, 2)

        a = uv_pixels[0]
        b = uv_pixels[1]
        c = uv_pixels[2]
        v0 = b - a
        v1 = c - a
        denom = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(denom) < 1e-9:
            continue
        rel = sample_points - a
        w1 = (rel[:, 0] * v1[1] - v1[0] * rel[:, 1]) / denom
        w2 = (v0[0] * rel[:, 1] - rel[:, 0] * v0[1]) / denom
        w0 = 1.0 - w1 - w2
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not np.any(inside):
            continue

        weights = np.stack([w0[inside], w1[inside], w2[inside]], axis=1)
        points_world = weights @ tri_world
        uv_image, _ = project_points(points_world, frame.world2cam, intr)
        inside_image = (
            (uv_image[:, 0] >= 0.0)
            & (uv_image[:, 0] < intr.width - 1.0)
            & (uv_image[:, 1] >= 0.0)
            & (uv_image[:, 1] < intr.height - 1.0)
        )
        if not np.any(inside_image):
            continue

        atlas_xy = sample_points[inside][inside_image]
        uv_valid = uv_image[inside_image]
        mask_values = np.array(
            [bilinear_sample_scalar(dilated_masks[frame_idx], float(x), float(y)) for x, y in uv_valid],
            dtype=np.float32,
        )
        keep = mask_values >= 64.0
        if not np.any(keep):
            continue

        atlas_xy = atlas_xy[keep]
        uv_valid = uv_valid[keep]
        colors = bilinear_sample_color(frame.rgb_bgr, uv_valid[:, 0], uv_valid[:, 1]).astype(np.uint8)
        atlas[atlas_xy[:, 1].astype(np.int32), atlas_xy[:, 0].astype(np.int32)] = colors
        filled[atlas_xy[:, 1].astype(np.int32), atlas_xy[:, 0].astype(np.int32)] = 255

    if np.count_nonzero(filled) == 0:
        raise RuntimeError("Texture bake produced an empty atlas.")

    atlas_filled = atlas.copy()
    fill_mask = cv2.bitwise_not(filled)
    atlas_filled = cv2.inpaint(atlas_filled, fill_mask, 3, cv2.INPAINT_TELEA)
    return remapped_vertices, atlas_faces, atlas_uvs, atlas_filled


def export_obj(
    vertices_world: np.ndarray,
    faces_uv: np.ndarray,
    uvs: np.ndarray,
    output_dir: Path,
    texture_name: str,
    object_name: str,
) -> None:
    obj_path = output_dir / f"{object_name}.obj"
    mtl_path = output_dir / f"{object_name}.mtl"
    obj_lines = [f"mtllib {mtl_path.name}", f"o {object_name}"]
    for vertex in vertices_world:
        obj_lines.append(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}")
    for uv in uvs:
        obj_lines.append(f"vt {uv[0]:.8f} {uv[1]:.8f}")
    obj_lines.append("usemtl material_0")
    for face in faces_uv:
        a, b, c = face + 1
        obj_lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
    obj_path.write_text("\n".join(obj_lines) + "\n", encoding="utf-8")

    mtl_lines = [
        "newmtl material_0",
        "Ka 1.000 1.000 1.000",
        "Kd 1.000 1.000 1.000",
        "Ks 0.000 0.000 0.000",
        "d 1.0",
        f"map_Kd {texture_name}",
    ]
    mtl_path.write_text("\n".join(mtl_lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    clean_data_dir = Path(args.clean_data_dir)
    pose_dir = Path(args.pose_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    intr, _ = load_metadata(pose_dir)
    frames = load_frames(clean_data_dir, pose_dir)
    mesh, scale = normalize_mesh(Path(args.mesh_path))

    observed_points = build_alignment_cloud(frames, intr)
    transform, report = align_mesh(mesh, observed_points)
    vertices_world = transform_mesh_vertices(mesh, transform)

    for frame in frames:
        overlay = render_overlay(frame.rgb_bgr, vertices_world, np.asarray(mesh.faces), frame, intr)
        cv2.imwrite(str(overlay_dir / f"{frame.frame_id}.png"), overlay)

    remapped_vertices_world, atlas_faces, atlas_uvs, atlas_image = rasterize_texture(
        vertices_world,
        np.asarray(mesh.faces, dtype=np.int32),
        frames,
        intr,
        args.atlas_size,
    )

    texture_name = "battery-holder_texture.png"
    Image.fromarray(cv2.cvtColor(atlas_image, cv2.COLOR_BGR2RGB)).save(output_dir / texture_name)
    export_obj(
        remapped_vertices_world,
        atlas_faces,
        atlas_uvs,
        output_dir,
        texture_name=texture_name,
        object_name="battery-holder",
    )

    report.update(
        {
            "mesh_scale_applied": scale,
            "num_frames_used": len(frames),
            "num_alignment_points": int(len(observed_points)),
            "texture_atlas_size": args.atlas_size,
        }
    )
    (output_dir / "alignment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[holder-texture] Wrote textured mesh to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
