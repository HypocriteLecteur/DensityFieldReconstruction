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

Scenario configuration connects a short name to a local data file. The public
loader accepts a registered scenario name, a scenario YAML file, an explicit
data path, or a resolved `DatasetSpec`:

```python
from pathlib import Path

import dfr

dataset = dfr.load_dataset("jackdaw2")
# Equivalent explicit forms:
dataset = dfr.load_dataset(Path("scenarios/jackdaw2/config.yaml"))
dataset = dfr.load_dataset(Path("dataset/mobbing_flock_06.npz"))

print(dataset.trajectories.shape)       # (frames, maximum agents, 3)
positions = dataset.positions_at_time_step(2800)  # NaN-padded agents removed
print(positions.shape)
```

Pass `project_root=...` when scenarios/data live outside this checkout. All
resolved paths become absolute, so later frame access does not depend on the
working directory:

```python
dataset = dfr.load_dataset("my-scenario", project_root="D:/research/dfr-project")
spec = dfr.resolve_dataset("my-scenario", project_root="D:/research/dfr-project")
print(spec.config_path, spec.data_path)
```

The returned object follows `dfr.Dataset`. Positions and optional velocities
use `(frames, agents, 3)`. `len(dataset)`/`dataset.frame_count` report frame
count; `timestamps`, `coordinate_system`, and `metadata` expose optional source
information; `ground_truth_positions` is the trajectory used by current DFR
evaluation. Missing optional velocities raise an actionable error rather than
silently returning fabricated data.

Use the shared selector before an analysis or reconstruction:

```python
frames = dfr.select_frame_indices(dataset, [0, 10, -1])
sampled = dfr.select_frame_indices(dataset, slice(0, None, 20))
```

Invalid indices, empty selections, missing files, unsupported extensions, and
malformed supported files are reported separately.

### Supported loader schemas

Every format uses the same loading call: `dfr.load_dataset(path)`. Minimal
source schemas are:

| Format | Required source schema | Minimal creation/use example |
|---|---|---|
| `.npy` | One numeric `(frames, agents, 3)` array | `np.save("points.npy", positions)` |
| `.npz` standard | `trajectories`; optional same-shaped `velocities` | `np.savez("points.npz", trajectories=positions, velocities=velocities)` |
| `.npz` positions | A `(frames, agents, 3)` `positions` key | `np.savez("points.npz", positions=positions)` |
| `.mat` | MATLAB struct `swarm_data.positions` shaped `(3, agents, frames)` | `dfr.load_dataset("swarm.mat")` |
| `.rtf` | Header `#  x(t1) ... z(t2)` followed by six numeric columns per agent | `dfr.load_dataset("two-frame-flock.rtf")` |
| `.hdf5` | Integer timestamp groups containing `tid`, `x/y/z`, and `vx/vy/vz` datasets | `dfr.load_dataset("tracked-agents.hdf5")` |
| `.csv` | `Time` plus complete `Drone##_X/Y/Z` column groups | `dfr.load_dataset("drones.csv")` |

The HDF5 loader creates memory-mapped `.traj.cache.npy` and `.vel.cache.npy`
files beside the source. The drone CSV loader swaps source X/Y axes and records
that conversion in `dataset.coordinate_system`.

`dfr.dataset_io.DatasetFactory` remains available as a compatibility API for
older scripts, but new code should use `dfr.load_dataset`.

## Typed workflow configuration

Common workflow settings use composable, YAML/JSON-safe dataclasses:

```python
from dfr import (
    AnalysisConfig,
    CameraConfig,
    EvaluationConfig,
    OutputConfig,
    RunConfig,
    resolve_dataset,
)
from dfr.config import ReconstructionParams, TrainingParams

camera = CameraConfig.encircling(count=4, padding=1.25, is_3d=False)
# Or provide [x, y, z, qx, qy, qz, qw] rows:
camera = CameraConfig.explicit([
    [-10, 0, 0, 0, 0, 0, 1],
    [0, -10, 0, 0, 0, 0, 1],
])

analysis = AnalysisConfig(frames=(0, 10, 20), scales=(0.5, 1.0, 2.0))
evaluation = EvaluationConfig(voxel_resolution=0.25, batch_size=200_000)

run = RunConfig(
    dataset=resolve_dataset("jackdaw2"),
    output=OutputConfig(
        workflow="analysis",
        name="jackdaw2 scale analysis",
        run_id="jackdaw2-scales",
    ),
    camera=camera,
    analysis=analysis,
    evaluation=evaluation,
    seed=12345,
)
print(run.serializable())
```

`RunConfig` schema version 1 also composes the existing `TrainingParams` and
`ReconstructionParams`. `RunConfig.from_dict(...)` restores a configuration
loaded from YAML/JSON; unknown schema versions fail explicitly.

Configured scenarios include `boids`, `boids_multi`, `cluster`, `clutter`,
`jackdaw`, `jackdaw2`, `starling`, `swift`, `ue4`, and project-specific drone
datasets. Their large source datasets are local and ignored by Git; a config is
not usable until its `data_file` exists.

## Analyze a dataset

The shortest supported workflow uses the public facade and an explicit
analysis kind. It never saves implicitly:

```python
import dfr

dataset = dfr.load_dataset("jackdaw2")
curve = dfr.analyze(
    dataset,
    kind="modes",
    config=dfr.AnalysisConfig(
        frames=(2800,),
        scales=(0.5, 0.75, 1.0, 1.5, 2.0),
        device="cuda",
    ),
)
print(curve.mode_counts)
```

`kind="modes"` interprets scales in dataset coordinate units.
`kind="dra"` interprets them as multiples of mean nearest-neighbour distance
and performs the CUDA-intensive scale/model-order reconstruction analysis.
The facade currently accepts exactly one frame; use the lower-level APIs for
custom or multiframe pipelines.

### Count density modes at one scale

```python
from dfr import load_dataset
from dfr.analysis import analyze_dataset_modes, count_modes

dataset = load_dataset("jackdaw2")
positions = dataset.positions_at_time_step(2800)
count = count_modes(positions, scale=1.0)

curve = analyze_dataset_modes(
    dataset,
    frame=2800,
    scales=(0.5, 0.75, 1.0, 1.5, 2.0),
)
curve.save_npz("outputs/mode_curve.npz")  # explicit path; no implicit saving
print(count, curve.mode_counts)
```

Mode count depends on coordinate units and scale. Start with a small frame
sample and inspect the scenario's physical scale before launching a sweep.

### Run scale/model-order analyses

The most structured current analysis entry points expose command-line help:

```powershell
python -m experiments.plot_dra_scale_model_order --help
python -m experiments.plot_dra_scale_model_order --datasets jackdaw2 --output-root outputs --run-id jackdaw2-dra

python -m experiments.fit_dra_multiframe --help
python -m experiments.fit_dra_multiframe --datasets jackdaw2 --frames-per-dataset 3 --output-root outputs --run-id jackdaw2-multiframe
```

These analyses are CUDA-intensive and cache intermediate `.npz` data so a run
can resume. The parameter-manifold, mechanistic, synthetic, and validation
analyses now expose explicit managed CLIs. `power_law.py` remains a legacy
study collection and requires a named experiment subcommand; see
`experiments/README.md` before running it.

Reusable DRA computation and fitting now live under `dfr.analysis`, including
`ScaleAnalysisResult`, `create_scale_analysis`,
`compute_scale_model_order_surface`, `fit_dra_surface`, and multiframe fitting.
`ModeCurveResult`, `ScaleAnalysisResult`, and `ManifoldAnalysisResult` contain
data only and support explicit NPZ save/load. Plotting and managed-run decisions
remain in the experiment entry points.

Reusable analysis plots live under `dfr.plotting`. The mode-count and DRA
surface helpers accept typed result objects directly, still return Matplotlib
`Figure`/`Axes` handles, and do not save unless the caller explicitly asks:

```python
from dfr.plotting import (
    plot_dra_scale_model_order_surface,
    plot_mode_count_curve,
    save_figure,
)

fig, ax = plot_mode_count_curve(curve)
save_figure(fig, "outputs/mode_curve.png")

# dra_result is a ScaleAnalysisResult, for example from dfr.analyze(..., kind="dra")
fig, ax, surface = plot_dra_scale_model_order_surface(dra_result)
```

### Fit the parameter manifold

Reusable centered-3PL fitting and cache compatibility are also available
without importing an experiment:

```python
import numpy as np

import dfr
from dfr.analysis import fit_centered_3pl_curves, load_legacy_manifold_cache

dataset = dfr.load_dataset("jackdaw")
frame_ids = np.arange(350, 550)
animal_counts = [len(dataset.positions_at_time_step(frame)) for frame in frame_ids]
cache = load_legacy_manifold_cache("scenarios/jackdaw")
fit = fit_centered_3pl_curves(
    frame_ids=frame_ids,
    animal_counts=animal_counts,
    scale_ranges=cache.scale_ranges,
    mode_counts=cache.mode_counts,
    dataset_name="jackdaw",
)
fit.result.save_npz("outputs/jackdaw-manifold.npz")
print(fit.result.parameters)  # k, sigma_half, log10_gamma
```

`scale_for_mode_count` inverts a fitted curve to select a scale for a desired
mode count. `fit_shape_curve` and `project_to_shape_curve` provide the
intrinsic shape analysis. The legacy loader reads `modes.npy`,
`scale_range.npy`, and optional `nn_dists.npy`; new fitted tables should use
`ManifoldAnalysisResult.save_npz` at an explicit managed output path.

`python -m experiments.parameter_manifold --no-display` reproduces the full
historic study. Inputs are the four configured biological datasets and their
legacy scenario caches. It performs CUDA mode counting if a cache is missing,
then fits the shared package model and runs PCA, t-SNE/UMAP, clustering, and
publication plotting. Runtime can be minutes to hours depending on cache state;
the script still owns its historic figures and cache-generation policy while
the reusable numerical operations live in `dfr.analysis`.

## Reconstruct and evaluate

The public reconstruction workflow accepts a loaded dataset, explicit frames,
camera configuration, optional fixed scale, and optional managed output:

```python
import dfr

dataset = dfr.load_dataset("jackdaw2")
run = dfr.reconstruct(
    dataset,
    frames=(2800,),
    cameras=dfr.CameraConfig.encircling(count=4, padding=1.0),
    scale=1.0,  # omit for adaptive scale selection
    output=dfr.OutputConfig(
        workflow="reconstruction",
        name="jackdaw2 demo",
        run_id="jackdaw2-frame-2800",
    ),
)
frame = run.frames[0]
print(frame.means, frame.radii, frame.weights, frame.scale)
print(run.run_dir)
```

Reconstruction plotting helpers can consume the typed frame result directly and
still leave saving as an explicit caller choice:

```python
from dfr.plotting import plot_frame_reconstruction_gmm_3d, save_figure

fig, ax, artists = plot_frame_reconstruction_gmm_3d(frame)
save_figure(fig, "outputs/reconstruction-gmm.png")
```

Pass `CameraConfig.explicit(...)` to use exact camera poses. Multiple frame
indices produce one `FrameReconstruction` per frame in the requested order.
Omit `output` to return arrays without writing files. The current backend
requires CUDA and a scenario YAML providing image dimensions, intrinsics, and
clip planes; a dataset loaded by scenario name retains that path automatically.
Advanced callers can pass `TrainingParams` and `ReconstructionParams` through
the `training=` and `reconstruction=` arguments. For experiment sweeps,
`frame_scales=` accepts one fixed scale per selected frame,
`projection_noise_std=` applies reproducible bounded pixel noise, and
`use_decoupled=` selects the existing decoupled model backend.

For a repeatable named-dataset sweep, put the frame range and scientific
controls in a `ScenarioRunSpec`:

```python
spec = dfr.ScenarioRunSpec(
    dataset="jackdaw2",
    start=2800,
    stop=2841,
    step=20,
    cameras=dfr.CameraConfig.encircling(count=4),
    use_ground_truth_scales=True,
    output=dfr.OutputConfig(
        workflow="reconstruction",
        name="jackdaw2 camera study",
        run_id="jackdaw2-cam-4",
    ),
)
run = dfr.run_scenario(spec, project_root=".")
```

`run_scenarios(...)` accepts several specifications. The runner loads each
dataset, resolves frames, reads its ordered `reconstruction_scale.npz` cache
when requested, calls `dfr.reconstruct`, and saves aggregate timing/loss/scale
statistics into the same managed run.

Measured or externally detected projections use a separate input contract:
`ExternalObservationFrame` supplies ground-truth positions, one 2D point array
per camera, the camera system that interprets those points, optional per-frame
camera poses, and visibility. `dfr.reconstruct_observations(...)` feeds those
observations into the same `ReconstructionRun`/managed-artifact result format.
Use this path for calibrated flock detections or UE4/image-derived detections
instead of replacing measured points with simulated projections.

Evaluate an in-memory reconstruction against its source positions (or pass an
explicit ground-truth dataset):

```python
evaluation = dfr.evaluate(
    run,
    ground_truth=dataset,
    config=dfr.EvaluationConfig(
        voxel_resolution=0.25,
        batch_size=200_000,
        device="cuda",
    ),
)
print(evaluation.summary.recall)
print(evaluation.summary.hallucination)
print(evaluation.summary.dmota)
```

Evaluation plots accept the typed run or frame results directly and still leave
file output explicit:

```python
from dfr.plotting import (
    plot_evaluation_metric_series,
    plot_evaluation_summary,
    save_figure,
)

fig, ax = plot_evaluation_summary(evaluation)
save_figure(fig, "outputs/evaluation-summary.png")

fig, ax = plot_evaluation_metric_series(evaluation)
save_figure(fig, "outputs/evaluation-metrics-by-frame.png")
```

`dfr.evaluate("outputs/reconstruction/<run-id>", config=...)` loads managed
reconstruction arrays directly. Bounds default to the ground-truth extent plus
three reconstruction scales; set `EvaluationConfig.bounds` for an exact common
grid. Evaluation returns data without writing unless an
`OutputConfig(workflow="evaluation", ...)` is supplied. Per-frame and aggregate
JSON metrics then go under `outputs/evaluation/<run-id>/metrics/`.

For a quick managed one-frame reconstruction, use the transitional CLI:

```powershell
python -m experiments.reconstruct_one_frame --dataset jackdaw2 --frame 2800 --camera-count 2 --scale 1.0 --iterations 100 --run-id jackdaw2-frame-2800
```

Omit `--scale` to use the current adaptive scale selector. The command writes
the resolved config and manifest at the run root, reconstructed arrays under
`data/`, the final Gaussian checkpoint under `checkpoints/`, and the summary
and timing metrics under `metrics/`. Use `--help` for voxel, seed, output-root,
resume, and overwrite controls.

The three publication table variants are now explicit, side-effect-free CLIs:

```powershell
python -m experiments.run_scenarios_table_2 reconstruct --help
python -m experiments.run_scenarios_table_3 reconstruct --datasets starling --camera-counts 2 --run-id-prefix table3-smoke
python -m experiments.run_scenarios_table_4 run --noise-levels 0 1 2 --camera-counts 2
```

Table 2 is the 100-iteration camera-count preset, Table 3 is its 500-iteration
counterpart, and Table 4 is the projection-noise preset (historically active
for `starling`). `reconstruct` creates managed reconstruction runs; `run` also
evaluates each result into a matching managed evaluation run. No study starts
merely by importing one of these modules.

The transitional general multi-scenario path remains available:

1. Edit `CAM_NUM`, `LOG_NAME`, `DATASET_RUNS`, and relevant flags near the top
   of `experiments/run_scenarios.py`.
2. Confirm every selected scenario's `config.yaml` and `data_file`.
3. Run from the repository root:

   ```powershell
   python -m experiments.run_scenarios
   ```

Its active reconstruction path now resolves datasets/cameras through package
services and calls `dfr.run_scenario`. When `IS_LOGGING` is enabled, final frame
artifacts and aggregate statistics use managed reconstruction output rather
than scenario log directories. Baseline and metric helpers in that module
remain transitional. This path requires CUDA and
`gaussian_rasterizer_simple_large`.

To evaluate already generated checkpoints with the current fixed dataset and
camera combinations:

```powershell
python -m experiments.compute_metrics_from_pretrained
```

This evaluator reads the hard-coded `DATASET_RUNS` and camera counts in its
source, but now delegates density overlap and aggregate metric equations to
`dfr.evaluation`. Review its legacy dataset/checkpoint settings before running.

## Generated outputs

`outputs/` is the canonical root for all new work and is ignored by Git.
`OutputConfig` and `RunArtifacts` now create managed runs with provenance and
safe persistence:

```python
from dfr import OutputConfig, RunArtifacts

artifacts = RunArtifacts.create(
    OutputConfig(
        workflow="analysis",
        name="jackdaw2 scale sweep",
        run_id="jackdaw2-scale-sweep",
        resume=True,
    ),
    resolved_config={"dataset": "jackdaw2", "frames": [2800]},
    device="cuda",
)

artifacts.save_json(
    "summary.json",
    {"recommended_scale": 1.25},
    category="metrics",
    overwrite=True,
)
print(artifacts.run_dir)
```

Relative output roots resolve from the project root, not `os.getcwd()`.
Existing run IDs require an explicit policy: `resume=True` preserves artifacts
and verifies that the resolved scientific config matches; `overwrite=True`
replaces the managed run. The two policies are mutually exclusive.

Until migration is complete, active legacy scripts may still write elsewhere:

| Location | Current use | Policy |
|---|---|---|
| `outputs/` | New run artifacts | Preferred; organize by workflow/run |
| `results/` | Scale/model-order caches and fits | Legacy; do not add new producers |
| `figs/` | Publication and diagnostic figures | Legacy; do not add new producers |
| `scenarios/*/logs/` | Reconstruction checkpoints/statistics | Legacy runner output |
| Repository root | A few grid-search CSV/log files | Legacy; migration required |

Managed runs use this layout:

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

Every managed run writes a schema-versioned `manifest.json` (timestamp, Git
commit, package version, device, and metadata) plus a fully resolved
`config.yaml`. JSON, NPZ, checkpoint, and figure writers reject path traversal
and require explicit overwrite for existing files. Library functions should
still return results and save only when given an explicit output configuration.

`experiments.plot_dra_scale_model_order` is the first migrated analysis. It
writes resumable sweep caches to `cache/`, fit arrays to `data/`, summaries to
`metrics/`, and its surface plot to `figures/`. Use `--overwrite-run` to replace
its entire run or `--force` to recompute cached sweep values inside a resumed
run.

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

Detailed support status, inputs, outputs, runtime expectations, and commands
for every analysis-oriented entry point are maintained in
[`experiments/README.md`](experiments/README.md). In particular, supported
Phase 4 analyses use managed output, legacy power/scale studies require an
explicit subcommand, and direct `dfr_plot.py` execution is disabled until its
Phase 6 decomposition.

Most scripts predate the target workflow API. "Configured in source" means the
module has constants or an active function call that must be reviewed before
execution.

| Script | Purpose / behavior |
|---|---|
| `common.py` | Compatibility adapters over package scenario/camera/evaluation services plus runner logging/noise helpers. |
| `compute_metrics_from_pretrained.py` | Recompute density metrics from saved iteration checkpoints; configured in source. |
| `dataset_viewer_test.py` | Interactive dataset/camera/scale viewer; performs work at import and is excluded from pytest. |
| `dfr_plot.py` | Legacy 3,900-line publication/reconstruction plot collection; Phase 6 inventory is in `experiments/DFR_PLOT_CATALOG.md`, and camera/trajectory/2D projection/mode-count/DRA-surface/multiscale-density/density-GMM rendering now delegates to `dfr.plotting`. |
| `fit_dra_multiframe.py` | CLI for multi-frame DRA/model-order fitting, caches, reports, and fit figures using shared save defaults. |
| `generate_scene_animations.py` | Generate trajectory and ground-truth density MP4 animations; datasets configured in source. |
| `inspect_3d_error.py` | Inspect and plot 3D reconstruction error from existing run data. |
| `inspect_scenarios.py` | Interactive scenario/camera inspection. |
| `investigate_initialization.py` | Interactive investigation of GMM initialization behavior. |
| `mechanistic_derivation.py` | Managed CUDA CLI for analytical mode-count scaling comparisons using shared save defaults. |
| `parameter_manifold.py` | Managed 3PL fit/clustering/publication CLI using package manifold computation, style, and save defaults. |
| `parameter_manifold_2pl.py` | Managed symmetric-2PL manifold and temporal/N-dependence CLI using package style and save defaults. |
| `plot_dra_scale_model_order.py` | CUDA CLI for DRA over scale and model order with resumable caches, surface fits, package plotting primitives, and shared figure-save defaults. |
| `plot_publication_noise_robustness.py` | Explicit publication figure CLI for the hard-coded noise-robustness table formerly embedded in `dfr_plot.py`. |
| `plot_publication_table2.py` | Explicit publication figure CLI for the hard-coded Table 2 capacity/tradeoff figures formerly embedded in `dfr_plot.py`. |
| `plot_publication_time_efficiency.py` | Explicit publication figure CLI for the hard-coded training-time scaling table formerly embedded in `dfr_plot.py`. |
| `plotting_utils.py` | Legacy plotting/math compatibility helpers used by DRA figures; style/layout and density/GMM rendering helpers delegate to `dfr.plotting`; not an entry point. |
| `power_law.py` | Legacy exploratory mode-count scaling studies; requires an explicit experiment subcommand. |
| `rasterizer_optimize.py` | Benchmark/inspect custom rasterizer performance. |
| `reconstruction_scale_determination.py` | Legacy reconstruction-scale studies; requires an explicit experiment subcommand. |
| `reconstruct_one_frame.py` | Thin one-frame wrapper over `dfr.reconstruct`, with managed config, checkpoint, arrays, and metrics. |
| `run_post_processing.py` | Post-process saved reconstruction runs and metrics; configured in source. |
| `run_scenarios.py` | Main multi-scenario runner; active reconstruction dispatches through `dfr.reconstruct`, while legacy baseline/metric helpers remain. |
| `run_scenarios_angle_sweep.py` | Explicit-dispatch camera-angle/convergence studies; ordinary reconstruction is consolidated through `ScenarioRunSpec`. |
| `run_scenarios_flock.py` | Explicit external-detection flock workflow with required data/calibration paths; primary run uses `dfr.reconstruct_observations`. |
| `publication_scenarios.py` | Shared typed presets/CLI implementation for publication Tables 2–4. |
| `run_scenarios_table_2.py` | Thin managed Table 2 (100-iteration camera-count) entry point. |
| `run_scenarios_table_3.py` | Thin managed Table 3 (500-iteration camera-count) entry point. |
| `run_scenarios_table_4.py` | Thin managed Table 4 projection-noise entry point. |
| `run_scenarios_ue4.py` | Explicit UE4 workflow requiring three image roots; primary run uses `dfr.reconstruct_observations` and managed artifacts. |
| `search_learning_parameters.py` | Resumable grid search for learning-rate settings; writes a root CSV. |
| `search_regularization_parameters.py` | Resumable grid search for regularization settings; writes a root CSV. |
| `synthetic_benchmark.py` | Managed CUDA CLI for synthetic point-process manifold benchmarks. |
| `validate_mode_counting.py` | Managed CUDA validation CLI for separated synthetic Gaussian clusters using shared figure-save defaults. |
| `visualize.py` | Open the interactive GMM viewer for saved checkpoints; configured in source. |
| `__init__.py` | Marks `experiments` as a package; not an entry point. |

## Contributing and refactoring

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for test tiers and artifact rules.
[`TODO.md`](TODO.md) is the handoff document and must be updated in every
refactor commit. The local annotated tag `v0.1.0` is the pre-refactor stable
baseline.
