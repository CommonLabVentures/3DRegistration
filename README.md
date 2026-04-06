# 3DRegistration

This branch, `clean_dataset_restart`, is the reset-from-data version of the reconstruction work.
The goal is to use only RGB images, aligned depth, and a visible ChArUco board to recover camera
poses, isolate the object on the board, align a known mesh, and bake a texture atlas.

The earlier attempt based on robot/exported poses was preserved on branch `test_attempt_1`
at commit `559253a`.

## Current state

Two object runs exist in this branch:

- `clean_dataset`
  - AA battery capture set flattened into `rgb/` and `depth/`
  - ChArUco pose solve and subset selection completed
  - battery cutout extraction completed
  - mesh texturing was not pursued further because the battery is too small in the frames
  - conclusion: recapture at higher effective resolution is needed

- `holder_dataset`
  - larger blue battery holder captured on the same board
  - full end-to-end first pass completed:
    - raw capture flattening
    - ChArUco pose solve
    - oblique subset selection
    - holder segmentation
    - mesh alignment
    - texture baking
  - result is usable as a first pass, but the texture atlas still has projection artifacts

## Repository layout

- `clean_dataset/`
  - flattened AA battery RGBD dataset
- `subset_dataset/`
  - ChArUco-solved AA battery subset
- `subset_dataset/battery_only/`
  - battery-only masks and cutouts
- `holder_dataset/`
  - raw holder capture folders plus `battery-holder.stl`
- `holder_clean_dataset/`
  - flattened holder RGBD dataset derived from `holder_dataset`
- `holder_subset_dataset/`
  - ChArUco-solved holder subset
- `holder_subset_dataset/holder_only/`
  - holder-only masks and cutouts
- `holder_output/`
  - first-pass textured holder mesh and diagnostics
- `meshes/AAbattery.stl`
  - AA battery CAD mesh

## Camera and board assumptions

The processing scripts currently assume the RealSense color intrinsics below:

- width: `848`
- height: `480`
- fx: `605.1414184570312`
- fy: `604.7616577148438`
- cx: `417.1040954589844`
- cy: `250.10911560058594`
- distortion: zero

The ChArUco solve uses:

- dictionary: `DICT_4X4_100`
- inferred board layout: `12 x 8`
- inferred square length: about `0.0228 m`
- inferred marker length: `0.75 * square_length`

Important note:
The square size is inferred from depth, not from a trusted board spec file. If the true printed
square length is known, replacing the inferred value is the right next step.

## Scripts

### Dataset preparation

- [flatten_capture_dataset.py](/home/sambit/Code/3DRegistration/flatten_capture_dataset.py)
  - converts raw `capture_*/eye1/` folders into a clean paired `rgb/` and `depth/` dataset

- [prepare_charuco_subset.py](/home/sambit/Code/3DRegistration/prepare_charuco_subset.py)
  - detects the ChArUco board
  - estimates board-frame camera poses
  - computes view angle using the camera optical axis vs board normal
  - filters out near-normal views
  - writes a reduced subset and a contact sheet

### Object extraction

- [extract_battery_cutouts.py](/home/sambit/Code/3DRegistration/extract_battery_cutouts.py)
  - battery-specific extractor
  - uses board masking plus depth-hole seeding
  - falls back to a shape-constrained AA battery mask because the battery is too small for stable
    unconstrained segmentation

- [extract_holder_cutouts.py](/home/sambit/Code/3DRegistration/extract_holder_cutouts.py)
  - holder-specific extractor
  - uses board masking plus a simple blue-color threshold
  - works reliably because the holder is large and chromatically distinct from the board

### Mesh texturing

- [texture_holder_mesh.py](/home/sambit/Code/3DRegistration/texture_holder_mesh.py)
  - normalizes the holder STL to meters
  - uses near-normal holder depth views to build a board-frame point cloud
  - aligns the mesh with a coarse yaw sweep plus Powell refinement
  - unwraps UVs with `xatlas`
  - projects texture from the RGB views into a baked atlas
  - exports OBJ, MTL, and texture PNG

## AA battery run

### Inputs

- dataset: [clean_dataset](/home/sambit/Code/3DRegistration/clean_dataset)
- mesh: [AAbattery.stl](/home/sambit/Code/3DRegistration/meshes/AAbattery.stl)

### Generated outputs

- subset contact sheet: [subset_dataset/contact_sheet.png](/home/sambit/Code/3DRegistration/subset_dataset/contact_sheet.png)
- selected views: [subset_dataset/selected_views.json](/home/sambit/Code/3DRegistration/subset_dataset/selected_views.json)
- battery cutout sheet: [subset_dataset/battery_only/contact_sheet.png](/home/sambit/Code/3DRegistration/subset_dataset/battery_only/contact_sheet.png)

### Selected subset

The battery subset currently keeps these more oblique views:

- `001` at about `30.1 deg`
- `002` at about `27.1 deg`
- `004` at about `36.3 deg`
- `005` at about `45.5 deg`

### Outcome

The battery occupies too few pixels for reliable texture extraction. Segmentation had to be forced
with a shape prior, and that is not a strong enough base for a good final textured mesh.

Practical conclusion:

- keep the pose/subset code
- recapture the battery with higher object scale in the image
- do not spend more time polishing the current battery run

## Holder run

### Inputs

- raw captures: [holder_dataset](/home/sambit/Code/3DRegistration/holder_dataset)
- mesh: [battery-holder.stl](/home/sambit/Code/3DRegistration/holder_dataset/battery-holder.stl)

### Generated outputs

- flattened holder dataset: [holder_clean_dataset](/home/sambit/Code/3DRegistration/holder_clean_dataset)
- subset sheet: [holder_subset_dataset/contact_sheet.png](/home/sambit/Code/3DRegistration/holder_subset_dataset/contact_sheet.png)
- holder cutout sheet: [holder_subset_dataset/holder_only/contact_sheet.png](/home/sambit/Code/3DRegistration/holder_subset_dataset/holder_only/contact_sheet.png)
- alignment report: [alignment_report.json](/home/sambit/Code/3DRegistration/holder_output/alignment_report.json)
- textured mesh:
  - [battery-holder.obj](/home/sambit/Code/3DRegistration/holder_output/battery-holder.obj)
  - [battery-holder.mtl](/home/sambit/Code/3DRegistration/holder_output/battery-holder.mtl)
  - [battery-holder_texture.png](/home/sambit/Code/3DRegistration/holder_output/battery-holder_texture.png)

### Holder subset

The holder run selected the same four oblique camera positions as the battery run:

- `001` at about `30.1 deg`
- `002` at about `27.1 deg`
- `004` at about `36.3 deg`
- `005` at about `45.5 deg`

For geometry alignment, the near-normal views were also useful. The final mesh fit used all solved
views for texturing and the near-normal views for the most stable depth alignment.

### Alignment result

The current holder fit report is:

- coarse mean nearest-surface score: about `1.96 mm`
- refined mean nearest-surface score: about `1.44 mm`
- optimized yaw: about `88.55 deg`
- optimized translation:
  - `tx = 0.13685 m`
  - `ty = 0.09263 m`
  - `tz = 0.00437 m`

This is a reasonable first-pass fit. The wireframe overlays in
[holder_output/overlays](/home/sambit/Code/3DRegistration/holder_output/overlays)
should be used as the main sanity check.

### Texture result

The holder texture bake works, but it is still a prototype:

- UV unwrap is automatic via `xatlas`
- face-to-view assignment is heuristic
- there is no robust mesh occlusion z-buffer
- there is no seam-aware blending pass
- the atlas is inpainted where direct writes are missing

So the exported textured mesh is valid and inspectable, but not yet production quality.

## Reproduce

### 1. Flatten a raw capture dataset

```bash
./.venv/bin/python flatten_capture_dataset.py \
  --capture-root holder_dataset \
  --output-dir holder_clean_dataset \
  --camera-name eye1
```

### 2. Solve ChArUco poses and create a reduced subset

```bash
./.venv/bin/python prepare_charuco_subset.py \
  --data-dir holder_clean_dataset \
  --output-dir holder_subset_dataset \
  --min-optical-axis-angle-deg 15
```

### 3. Extract object-only cutouts

For the holder:

```bash
./.venv/bin/python extract_holder_cutouts.py \
  --data-dir holder_subset_dataset \
  --output-dir holder_subset_dataset/holder_only
```

For the battery:

```bash
./.venv/bin/python extract_battery_cutouts.py \
  --data-dir subset_dataset \
  --output-dir subset_dataset/battery_only
```

### 4. Align and texture the holder mesh

```bash
./.venv/bin/python texture_holder_mesh.py \
  --clean-data-dir holder_clean_dataset \
  --pose-dir holder_subset_dataset \
  --mesh-path holder_dataset/battery-holder.stl \
  --output-dir holder_output
```

## Dependencies

The current scripts expect these Python packages in `.venv`:

- `opencv-contrib-python`
- `numpy`
- `scipy`
- `trimesh`
- `xatlas`
- `Pillow`

`open3d` is not currently installed and is not required by the current holder path.

## Known problems

- The board metric scale is inferred from depth instead of being read from a ground-truth board
  spec.
- The battery is too small in the images to support a good texture bake.
- The holder texture bake does not yet do proper occlusion handling.
- The holder texture bake does not yet blend seams between views.
- The holder alignment is depth-assisted but not yet validated against a stronger silhouette
  renderer.

## Recommended next steps

1. Recapture the AA battery so it occupies significantly more pixels.
2. Replace the inferred ChArUco square length with the real printed square length.
3. Improve holder texture projection with:
   - face visibility checks
   - depth-aware rejection
   - seam blending
4. If the holder fit needs to be tightened further, add silhouette scoring on top of the current
   depth fit.
