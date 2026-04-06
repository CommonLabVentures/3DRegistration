# 3DRegistration

This branch, `test_attempt_1`, is a preserved snapshot of the first end-to-end
textured-mesh attempt for the AA battery capture set.

## What is in this branch

- `dataset/`
  - Original capture folders with RGB, aligned depth, robot metadata, TF dumps,
    and any available precomputed camera poses.
- `dataset/CAD_assets/AAbattery.stl`
  - The AA battery mesh used for alignment and texturing.
- `texture_pipeline/`
  - Experimental multi-stage pipeline for segmentation, alignment, texturing,
    and export.
- `visualize_captures.py`
  - Utility to render RGB plus depth previews for the capture set.
- `output/`
  - Generated artifacts from the first attempt, including textured mesh exports
    and diagnostics.

## Snapshot purpose

This branch exists so the first pipeline attempt is fully preserved before a
clean restart. The next iteration is intended to ignore the old precomputed
poses and rebuild the workflow from a simplified dataset containing only RGB and
depth frames, with camera poses recomputed from the visible ChArUco board.

## Known issues in attempt 1

- Segmentation is unstable across views because the board dominates the depth.
- Only a subset of frames produced useful object masks.
- Alignment can succeed, but texturing quality is still sensitive to bad masks
  and weak visibility coverage.
- The branch contains exploratory outputs and should be treated as a record of
  the first pass, not the final workflow.

## Useful files

- Final textured export from this attempt:
  - `output/final_aabattery_pca/AAbattery.obj`
  - `output/final_aabattery_pca/AAbattery.mtl`
  - `output/final_aabattery_pca/AAbattery_texture.png`
  - `output/final_aabattery_pca/AAbattery.glb`
- Capture previews:
  - `output/capture_viz/contact_sheet.png`

## Notes

- The virtual environment `.venv/` is intentionally not committed.
- Python cache directories are intentionally not committed.
