# Clean Dataset

This dataset contains only the inputs required for a clean restart:

- RGB frames in `rgb/`
- aligned depth arrays in `depth/`

Depth files are `.npz` archives with one array named `depth`, stored in metres.

The files are paired by zero-padded frame index:

- `rgb/000.jpg` pairs with `depth/000.npz`
- ...
- `rgb/012.jpg` pairs with `depth/012.npz`
