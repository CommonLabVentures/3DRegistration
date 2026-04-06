"""Visualization and validation helpers.

All functions display via matplotlib (non-blocking unless show=True is passed).
For headless environments, images can be saved to disk instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import trimesh

from .utils import FrameData, render_mesh_silhouette


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _show_or_save(img_rgb: np.ndarray, title: str, save_path: Optional[Path] = None) -> None:
    """Display an RGB image via matplotlib, or save it to disk."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use("TkAgg") if save_path is None else None
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(img_rgb)
        ax.set_title(title)
        ax.axis("off")
        if save_path:
            plt.savefig(str(save_path), bbox_inches="tight", dpi=150)
            plt.close(fig)
        else:
            plt.tight_layout()
            plt.show()
    except Exception:
        # Fallback: write PNG
        if save_path:
            cv2.imwrite(str(save_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def _overlay_mask(rgb_bgr: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.4) -> np.ndarray:
    """Blend a binary mask as a coloured overlay on an BGR image."""
    overlay = rgb_bgr.copy()
    overlay[mask > 0] = (
        (1 - alpha) * rgb_bgr[mask > 0] +
        alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    return overlay


# ---------------------------------------------------------------------------
# Stage-level visualization
# ---------------------------------------------------------------------------

def visualize_charuco_detection(
    frame: FrameData,
    world2cam: np.ndarray,
    charuco_cfg,
    save_path: Optional[Path] = None,
) -> None:
    """Draw detected ChArUco corners on the RGB image."""
    from .charuco import _make_charuco_board
    board, detector = _make_charuco_board(charuco_cfg)

    gray = cv2.cvtColor(frame.rgb, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    vis = frame.rgb.copy()
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        K = frame.K
        dist = np.zeros(5)
        n, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board, cameraMatrix=K, distCoeffs=dist
        )
        if n > 0:
            cv2.aruco.drawDetectedCornersCharuco(vis, ch_corners, ch_ids)

    _show_or_save(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), f"ChArUco — {frame.name}", save_path)


def visualize_masks(
    frames: list[FrameData],
    masks: list[np.ndarray],
    save_dir: Optional[Path] = None,
) -> None:
    """Show segmentation masks overlaid on RGB images."""
    for frame, mask in zip(frames, masks):
        vis = _overlay_mask(frame.rgb, mask)
        sp = None if save_dir is None else save_dir / f"mask_{frame.name}.png"
        _show_or_save(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), f"Mask — {frame.name}", sp)


def visualize_alignment(
    mesh: trimesh.Trimesh,
    frames: list[FrameData],
    masks: list[np.ndarray],
    obj_pose: np.ndarray,
    save_dir: Optional[Path] = None,
    wireframe_color: tuple = (0, 255, 0),
) -> None:
    """Project mesh wireframe onto each RGB frame and display.

    This is the human-in-the-loop validation step for Stage 3.
    """
    for frame, mask in zip(frames, masks):
        vis = frame.rgb.copy()

        # Render silhouette at full resolution
        sil = render_mesh_silhouette(
            mesh, frame.K, frame.world2cam, obj_pose, frame.rgb.shape[:2], scale=1.0
        )

        # Draw silhouette contours as a wireframe proxy
        contours, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, wireframe_color, 2)

        # Also show mask as semi-transparent red
        vis = _overlay_mask(vis, mask, color=(0, 0, 255), alpha=0.2)

        # Overlay IoU score
        from .utils import compute_iou
        iou = compute_iou(sil > 0, mask > 0)
        cv2.putText(vis, f"IoU={iou:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        sp = None if save_dir is None else save_dir / f"align_{frame.name}.png"
        _show_or_save(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), f"Alignment — {frame.name}", sp)


def visualize_atlas(
    atlas_bgr: np.ndarray,
    covered: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
) -> None:
    """Display the texture atlas, optionally highlighting uncovered regions."""
    vis = atlas_bgr.copy()
    if covered is not None:
        vis[~covered] = [30, 30, 30]  # dark grey for uncovered pixels
    _show_or_save(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), "Texture Atlas", save_path)


def save_all_diagnostics(
    frames: list[FrameData],
    masks: list[np.ndarray],
    mesh: trimesh.Trimesh,
    obj_pose: np.ndarray,
    atlas_bgr: np.ndarray,
    output_dir: Path,
) -> None:
    """Save all diagnostic images to output_dir/diagnostics/."""
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    visualize_masks(frames, masks, save_dir=diag_dir)
    visualize_alignment(mesh, frames, masks, obj_pose, save_dir=diag_dir)
    if atlas_bgr is not None:
        visualize_atlas(atlas_bgr, save_path=diag_dir / "atlas.png")
    print(f"  [viz] Diagnostics saved to {diag_dir}")
