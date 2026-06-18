# Density Field Rasterizer

This directory contains installable copies of the CUDA rasterizer extensions.

## Canonical Source

The canonical source for these rasterizers lives in `camera-aero/density_field_rasterizer/`
(not tracked in this repository — see `.gitignore`).

## Variants

- `gaussian_rasterizer_simple_small` — default forward/backward rasterizer for single-scale rendering
- `gaussian_rasterizer_simple_large` — higher capacity variant (more Gaussians per tile)
- `gaussian_rasterizer_simple_small_decoupled` — decoupled normalization variant

## Build

```bash
cd density_field_rasterizer/gaussian_rasterizer_simple_large
python setup.py install
```

Replace `gaussian_rasterizer_simple_large` with the desired variant.

Requires: CUDA toolkit, PyTorch with CUDA support, GLM headers (included in `third_party/glm`).
