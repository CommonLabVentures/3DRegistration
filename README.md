# 3DRegistration

This branch, `clean_dataset_restart`, is the fresh restart point.

The previous pipeline attempt was preserved on branch `test_attempt_1` in commit
`559253a`. This branch intentionally drops the old pipeline code, robot metadata,
TF dumps, joint logs, and precomputed camera poses.

## What remains

- `clean_dataset/rgb/`
  - 13 RGB frames, named `000.jpg` through `012.jpg`
- `clean_dataset/depth/`
  - 13 aligned depth arrays, named `000.npz` through `012.npz`
  - each `.npz` contains a single array named `depth`
  - depth is `float32` in metres
- `meshes/AAbattery.stl`
  - AA battery mesh for the new alignment/texturing workflow

## What was intentionally removed

- `camera_pose.json`
- all `TF/` folders
- all joint-state JSON files
- all previous pipeline code and previous output artifacts

## Camera intrinsics

Use these intrinsics when recomputing ChArUco poses:

- width: `848`
- height: `480`
- fx: `605.1414184570312`
- fy: `604.7616577148438`
- cx: `417.1040954589844`
- cy: `250.10911560058594`
- distortion: zero

## ChArUco board metadata

- squares_x: `5`
- squares_y: `7`
- square_length: `0.04` m
- marker_length: `0.03` m
- dictionary: `DICT_4X4_100`

## Frame order

The numbered files preserve the original sorted capture order:

- `000` -> `capture_20260406_154157_814251`
- `001` -> `capture_20260406_154216_522076`
- `002` -> `capture_20260406_154222_535314`
- `003` -> `capture_20260406_154232_056267`
- `004` -> `capture_20260406_154248_258571`
- `005` -> `capture_20260406_154254_271792`
- `006` -> `capture_20260406_154312_144470`
- `007` -> `capture_20260406_154329_683067`
- `008` -> `capture_20260406_154335_863338`
- `009` -> `capture_20260406_154345_217255`
- `010` -> `capture_20260406_154401_419569`
- `011` -> `capture_20260406_154407_432825`
- `012` -> `capture_20260406_154425_472526`

## Restart intent

The next implementation should:

1. detect the ChArUco board from the RGB frames
2. solve camera poses from the board instead of using robot-derived poses
3. rebuild segmentation, alignment, and texturing against this simplified dataset
