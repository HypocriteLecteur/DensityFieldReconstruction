# Density Field Reconstruction (DFR)

DFR reconstructs a three-dimensional density field from synchronized
multi-camera observations of moving groups such as bird flocks and simulated
swarms. The repository also contains research code for mode counting, scale
analysis, Gaussian-mixture reduction, reconstruction evaluation, and paper
figures.

The project is in an incremental usability refactor. The commands below
describe the code that works today. The planned high-level
`load -> analyze -> reconstruct -> evaluate` API and migration status are in
[`TODO.md`](TODO.md).

## Repository map

```text
dfr/                       Reusable reconstruction and analysis package
experiments/               Research entry points and publication scripts
scenarios/<name>/config.yaml
                           Dataset paths, camera intrinsics, and run settings
dataset/                   Local input data (ignored by Git)
density_field_rasterizer/  Custom CUDA extension sources
tests/                     CPU characterization and CUDA smoke tests
outputs/                   Canonical destination for new generated artifacts
results/, figs/            Legacy generated artifacts being migrated
```

Run current experiment commands from the repository root. Several legacy
scripts still resolve paths relative to the current working directory.

## Supported environment

The supported Python range is 3.12-3.13. The current Windows development
environment was verified with Python 3.13.13, PyTorch 2.12.0+cu132, CUDA Toolkit
13.3, and an NVIDIA RTX 4060 Laptop GPU. CPU-only dataset, geometry, mode,
metric, and checkpoint tests are supported; reconstruction requires CUDA and
the custom rasterizer extensions.

### Install

1. Create and activate an environment:

   ```powershell
   conda create -n dfr python=3.13
   conda activate dfr
   ```

2. Install a PyTorch build compatible with the machine's CUDA driver/toolkit.
   Use the command from the PyTorch installation selector rather than assuming
   the CUDA version used by the current workstation.

3. Install DFR and its test dependencies from the repository root:

   ```powershell
   python -m pip install -e ".[test]"
   ```

   `environment.txt` remains available as a simple dependency list:

   ```powershell
   python -m pip install -r environment.txt
   ```

4. For CUDA reconstruction, install the rasterizer variants used by the target
   workflow. Import PyTorch before testing the installed modules on Windows.

   ```powershell
   python -m pip install --no-build-isolation ./density_field_rasterizer/gaussian_rasterizer_simple_small
   python -m pip install --no-build-isolation ./density_field_rasterizer/gaussian_rasterizer_simple_large
   python -m pip install --no-build-isolation ./density_field_rasterizer/gaussian_rasterizer_simple_small_decoupled
   ```

   The large rasterizer is required by the main reconstruction/model modules.
   CuPy is optional and is used by the `cuda_circles` rendering path. For a
   CUDA 13 environment:

   ```powershell
   python -m pip install -e ".[cuda13]"
   ```

### Rasterizer troubleshooting

- **`UnicodeDecodeError` on non-English Windows:** PyTorch's extension helper
  may decode MSVC output with the OEM code page. In
  `torch/utils/cpp_extension.py`, using
  `SUBPROCESS_DECODE_ARGS = ('oem', 'replace')` avoids failures on Unicode
  compiler output.
- **CCCL/CUB errors with CUDA 13.3+:** the rasterizer build scripts enable the
  MSVC standard-conforming preprocessor. If maintaining a different build
  script, pass `/Zc:preprocessor` through the host compiler.
- **DLL load failure:** import `torch` before a rasterizer module so PyTorch's
  DLLs are loaded first.
- **Architecture mismatch:** `TORCH_CUDA_ARCH_LIST` is currently set in the
  rasterizer build scripts. Adjust it when building for a GPU architecture
  other than the configured target.

## Load a dataset

Scenario configuration connects a short scenario name to a local data file.
The current API is explicit but requires only the configuration and factory:

```python
from pathlib import Path

from dfr.dataset_io import DatasetFactory
from dfr.simulation_config import SimulationConfig

name = "jackdaw2"
config = SimulationConfig(Path("scenarios") / name / "config.yaml")
dataset = DatasetFactory().get_dataset(config.data_file)

print(dataset.trajectories.shape)       # (frames, maximum agents, 3)
positions = dataset.positions_at_time_step(2800)  # NaN-padded agents removed
print(positions.shape)
```

Supported loader formats currently include `.npy`, several `.npz` layouts,
MATLAB `.mat`, `.rtf`, `.hdf5`, and project-specific CSV data. Positions use
the shape `(frames, agents, 3)` after loading. Some source formats also provide
velocities.

Configured scenarios include `boids`, `boids_multi`, `cluster`, `clutter`,
`jackdaw`, `jackdaw2`, `starling`, `swift`, `ue4`, and project-specific drone
datasets. Their large source datasets are local and ignored by Git; a config is
not usable until its `data_file` exists.

## Analyze a dataset

### Count density modes at one scale

```python
import torch

from dfr.mode_finding import mode_counting

points = torch.as_tensor(positions, dtype=torch.float32)
count = mode_counting(
    positions_torch=points,
    modes=points.clone(),
    scale=1.0,
    max_iter=1000,
    tol=1e-2,
)
print(count)
```

Mode count depends on coordinate units and scale. Start with a small frame
sample and inspect the scenario's physical scale before launching a sweep.

### Run scale/model-order analyses

The most structured current analysis entry points expose command-line help:

```powershell
python -m experiments.plot_dra_scale_model_order --help
python -m experiments.plot_dra_scale_model_order --datasets jackdaw2 --output-dir outputs/analysis/jackdaw2-dra

python -m experiments.fit_dra_multiframe --help
python -m experiments.fit_dra_multiframe --datasets jackdaw2 --frames-per-dataset 3 --output-dir outputs/analysis/jackdaw2-multiframe
```

These analyses are CUDA-intensive and cache intermediate `.npz` data so a run
can resume. `parameter_manifold.py`, `parameter_manifold_2pl.py`, and
`power_law.py` contain additional research analyses, but still use hard-coded
settings and should be inspected before running.

## Reconstruct and evaluate

The high-level reconstruction API described in `TODO.md` is not implemented
yet. The current end-to-end path is the scenario runner:

1. Edit `CAM_NUM`, `LOG_NAME`, `DATASET_RUNS`, and relevant flags near the top
   of `experiments/run_scenarios.py`.
2. Confirm every selected scenario's `config.yaml` and `data_file`.
3. Run from the repository root:

   ```powershell
   python -m experiments.run_scenarios
   ```

The runner loads frames, generates/aims cameras, simulates observations,
selects a reconstruction scale, initializes and trains the Gaussian model, and
writes checkpoints/statistics to scenario log directories. This path requires
CUDA and `gaussian_rasterizer_simple_large`.

To evaluate already generated checkpoints with the current fixed dataset and
camera combinations:

```powershell
python -m experiments.compute_metrics_from_pretrained
```

This evaluator reads the hard-coded `DATASET_RUNS` and camera counts in its
source. Review those settings before running it.

## Generated outputs

`outputs/` is the canonical root for all new work and is ignored by Git. Until
the migration is complete, active legacy scripts may still write elsewhere:

| Location | Current use | Policy |
|---|---|---|
| `outputs/` | New run artifacts | Preferred; organize by workflow/run |
| `results/` | Scale/model-order caches and fits | Legacy; do not add new producers |
| `figs/` | Publication and diagnostic figures | Legacy; do not add new producers |
| `scenarios/*/logs/` | Reconstruction checkpoints/statistics | Legacy runner output |
| Repository root | A few grid-search CSV/log files | Legacy; migration required |

New code should use this provisional layout:

```text
outputs/<workflow>/<run-id>/
  manifest.json
  config.yaml
  data/
  checkpoints/
  metrics/
  figures/
  logs/
  cache/
```

Library functions should return results and save only when given an explicit
path. The artifact manager that enforces this contract is planned in Phase 3.

## Tests

Run the CPU safety net (the normal refactor loop):

```powershell
python -m pytest -m "not cuda"
```

Run CUDA extension smoke tests separately:

```powershell
python -m pytest -m cuda
```

Pytest only discovers files under `tests/`; it intentionally does not collect
interactive files such as `experiments/dataset_viewer_test.py`. CUDA tests skip
cleanly when the compiled extensions or a CUDA device are unavailable.

## Experiment script catalog

Most scripts predate the target workflow API. "Configured in source" means the
module has constants or an active function call that must be reviewed before
execution.

| Script | Purpose / behavior |
|---|---|
| `common.py` | Shared logging, scenario loading, camera setup, and metric formatting for runners; not an entry point. |
| `compute_metrics_from_pretrained.py` | Recompute density metrics from saved iteration checkpoints; configured in source. |
| `dataset_viewer_test.py` | Interactive dataset/camera/scale viewer; performs work at import and is excluded from pytest. |
| `dfr_plot.py` | Legacy 3,900-line collection of reconstruction, DRA, camera, table, and publication plots; active call configured at the bottom. |
| `fit_dra_multiframe.py` | CLI for multi-frame DRA/model-order fitting, caches, reports, and fit figures. |
| `generate_scene_animations.py` | Generate trajectory and ground-truth density MP4 animations; datasets configured in source. |
| `inspect_3d_error.py` | Inspect and plot 3D reconstruction error from existing run data. |
| `inspect_scenarios.py` | Interactive scenario/camera inspection. |
| `investigate_initialization.py` | Interactive investigation of GMM initialization behavior. |
| `mechanistic_derivation.py` | Generate figures/tests for the analytical mode-count scaling derivation. |
| `parameter_manifold.py` | Fit, cluster, validate, and plot the 3-parameter mode-curve manifold. |
| `parameter_manifold_2pl.py` | Two-parameter manifold and temporal/N-dependence analyses. |
| `plot_dra_scale_model_order.py` | CUDA CLI for DRA over scale and model order with resumable caches and surface fits. |
| `plotting_utils.py` | Shared plotting/math helpers used by DRA figures; not an entry point. |
| `power_law.py` | Large exploratory collection for synthetic/empirical mode-count scaling laws; active analysis configured at the bottom. |
| `rasterizer_optimize.py` | Benchmark/inspect custom rasterizer performance. |
| `reconstruction_scale_determination.py` | Legacy reconstruction-scale experiments and visualizations; configured in source. |
| `run_post_processing.py` | Post-process saved reconstruction runs and metrics; configured in source. |
| `run_scenarios.py` | Main multi-scenario reconstruction runner; datasets/cameras configured in source. |
| `run_scenarios_angle_sweep.py` | Camera-angle, convergence, voxel, and initialization sensitivity experiments. |
| `run_scenarios_flock.py` | Reconstruction workflow for flock/image-oriented datasets. |
| `run_scenarios_table_2.py` | Legacy runner variant for Table 2 experiments. |
| `run_scenarios_table_3.py` | Legacy runner variant for Table 3 experiments. |
| `run_scenarios_table_4.py` | Legacy runner variant for Table 4 experiments. |
| `run_scenarios_ue4.py` | UE4 image detection and reconstruction workflow with external image paths. |
| `search_learning_parameters.py` | Resumable grid search for learning-rate settings; writes a root CSV. |
| `search_regularization_parameters.py` | Resumable grid search for regularization settings; writes a root CSV. |
| `synthetic_benchmark.py` | Generate/cached synthetic point-process benchmarks for manifold validation. |
| `validate_mode_counting.py` | Validate mode counting on separated synthetic Gaussian clusters and plot errors. |
| `visualize.py` | Open the interactive GMM viewer for saved checkpoints; configured in source. |
| `__init__.py` | Marks `experiments` as a package; not an entry point. |

## Contributing and refactoring

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for test tiers and artifact rules.
[`TODO.md`](TODO.md) is the handoff document and must be updated in every
refactor commit. The local annotated tag `v0.1.0` is the pre-refactor stable
baseline.
