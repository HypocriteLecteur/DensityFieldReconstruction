# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-View Density Field Reconstruction (MV-DFR): reconstructs 3D density fields of dynamic point clouds (e.g., bird flocks) from 2+ synchronized camera views. The method represents the 3D density field as an isotropic Gaussian Mixture Model (GMM) optimized via differentiable rendering — projecting the GMM through known camera extrinsics/intrinsics and minimizing loss against observed 2D density images.

## Environment & Build

```bash
conda create -n mv-dfr Python=3.12
conda activate mv-dfr
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r environment.txt
```

The differentiable rasterizer must be built from source (CUDA C++ extensions):

```bash
cd density_field_rasterizer/gaussian_rasterizer_simple_large
python setup.py install   # try `build` if install fails
```

There are three rasterizer variants, all under `density_field_rasterizer/` (the canonical copies live in `camera-aero/density_field_rasterizer/`):
- `gaussian_rasterizer_simple_small` — default forward/backward rasterizer for single-scale rendering
- `gaussian_rasterizer_simple_large` — higher capacity variant (more Gaussians per tile)
- `gaussian_rasterizer_simple_small_decoupled` — decoupled normalization variant

After building rasterizers, install the `dfr` package in development mode:

```bash
pip install -e .
```

This makes `dfr` and `experiments` importable from any directory without `sys.path` hacks.

## Running Tests

```bash
pytest test/test_density_field_rasterizer.py   # single test file (GPU required)
```

## Architecture

### Core Pipeline (`dfr/`)

```
Camera images/points → DensityReconstructor.process_frame()
  1. estimate_swarm_center()           — triangulate 3D centroid from 2D image centroids
  2. reconstruction_scale_determination() — visual hull carving → ellipsoid fit → adaptive scale selection
  3. generate_scale_space_img()        — FFT-based Gaussian blur pyramid at the computed scale
  4. setup_gaussian_scale_space()      — initialize GMM (means from farthest-point sampled voxels, weights from density mass, radii from volume)
  5. train_gaussian_scale_space()      — Adam optimization loop with the differentiable rasterizer
```

**Key modules:**

| Module | Role |
|---|---|
| `dfr/density_field_reconstructor.py` | Orchestrator: ties together all pipeline steps in `process_frame()` |
| `dfr/density_field_model.py` | `GaussianModel`: GMM parameters (`_xyz`, `_radius`, `_weights`), optimizer setup with per-parameter LR scheduling, rasterization forward/backward, pruning, splitting, Adam state management |
| `dfr/camera_system.py` | `Camera`, `MultiCameraSystem` — camera geometry and multi-view coordination. Rendering code lives in `dfr/rendering.py` |
| `dfr/rendering.py` | `RenderStrategy` (projection / CuPy circles / Gaussian rasterizer), `select_rasterizer()` canonical variant selector, CuPy CUDA kernels |
| `dfr/center_estimator.py` | `estimate_center_from_point_sets()`, `estimate_center_from_images()` — triangulate 3D swarm center |
| `dfr/config.py` | `TrainingParams`, `ReconstructionParams` — typed @dataclass configs with dict backward compatibility |
| `dfr/model_checkpoint.py` | `build_checkpoint()`, `restore_model_from_checkpoint()` — trainable model serialization |
| `experiments/common.py` | Shared utilities for experiment scripts: `setup_logger()`, `setup_camera_system()`, `print_global_metrics()` |
| `dfr/camera_state.py` | `CameraState` (pose `[x,y,z,qx,qy,qz,qw]`) and `CameraStateUE4` (raw `R,T` matrices) — both expose `K`, `R`, `T`, `P` tensors on GPU |
| `dfr/reconstruction_scale_determination.py` | Visual hull reconstruction via frustum intersection, voxel carving with dilated masks, farthest-point sampling, statistical AABB estimation |
| `dfr/mode_finding.py` | Mean-shift clustering + DBSCAN for mode counting; analytic 4PL model to predict scale from desired mode count; PBC-aware variants |
| `dfr/utils.py` | GMM dissimilarity (ISE/RISE), GMM density evaluation, 3D metric computation (TP/FP/FN mass), encircling camera generation on circles or Fibonacci spheres |
| `dfr/gaussian_mixture_reduction.py` | `GMR`: greedy GMM reduction (Runnalls' algorithm) to merge components |
| `dfr/visualizer.py` | `MultiGMMPlotter`: matplotlib 3D visualization of GMMs with interactive toggles |
| `dfr/dataset_io.py` | `DatasetInterface` ABC + `DatasetFactory` for loading biological swarm data (`.mat`, `.npy`, `.pkl`, `.csv` formats) |
| `dfr/simulation_config.py` | YAML-based configuration for simulation runs |

### Camera Coordinate Conventions

The `wrd_to_cam()` static method applies a `base2cam` transform `[[0,-1,0],[0,0,-1],[1,0,0]]` to convert world-frame rotation/translation into the camera coordinate frame where X=right, Y=down, Z=forward. For standard poses, rotation is inverted via `Rotation.from_quat(pose[3:]).as_matrix().T`.

### GMM Optimization Details

- **Rasterizer forward**: projects 3D isotropic Gaussians to 2D via `R, T, K` and accumulates density onto an image grid (CUDA)
- **Rasterizer backward**: computes gradients w.r.t. means, radii, and weights
- **Regularization**: size-aware repulsion loss (prevents Gaussians from overlapping excessively) + homogeneity loss on log-radii
- **Pruning**: removes Gaussians with negative weights/radii (every 9 iters) and those outside all camera frustums (every 40 iters)
- **Learning rate**: linear decay schedule, per-parameter (`xyz`, `radius`, `weights`), initialized proportionally to parameter magnitudes

### Experiment Scripts (`experiments/`)

| Script | Purpose |
|---|---|
| `run_scenarios.py` | Main batch runner: loads datasets, generates camera setups, runs reconstruction, computes ISE metrics |
| `run_scenarios_table_2.py`, `table_3.py`, `table_4.py` | Per-table experiment configurations from the paper |
| `run_scenarios_flock.py` | Flock-specific experiments |
| `run_scenarios_ue4.py` | UE4-rendered synthetic data experiments |
| `run_post_processing.py` | GMM reduction and metrics computation after reconstruction |
| `search_regularization_parameters.py` | Hyperparameter grid search |
| `search_learning_parameters.py` | Learning rate grid search |
| `visualize.py` | Result visualization |
| `power_law.py` | Analytic scale-vs-mode-count relationship experiments |
| `dfr_plot.py` | Publication-quality plotting utilities |

### Important Notes

- **`pip install -e .` makes `dfr` and `experiments` importable from any directory** — `sys.path` hacks are no longer needed
- The `density_field_rasterizer/` directory at root has deleted CUDA files (tracked as `D` in git) — the canonical rasterizer source is in `camera-aero/density_field_rasterizer/`
- The `density_field_reconstruction_copy/` directory is a legacy snapshot, not the current code
- GPU (CUDA) is required for training; CuPy is optional but enables the `cuda_circles` renderer
- Trained checkpoints and history are saved as `.pth` files via `GaussianModel.write_checkpoints()` / `save_history()`
