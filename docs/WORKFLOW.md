# DFR Workflow Examples

The refactor target is a small, predictable workflow:

```text
load_dataset -> analyze -> reconstruct -> evaluate -> plot
```

Library calls return typed data objects by default. They write files only when
you pass an explicit output path, `OutputConfig`, or `RunArtifacts` object.
World-coordinate arrays use shape `(frames, agents, 3)` for trajectories and
`(agents, 3)` for one clean frame.

## 1. CPU-safe toy dataset and analysis

This example creates a tiny local dataset, loads it through the public loader,
computes a mode-count curve, and saves one plot explicitly. It does not need
the repository's large ignored datasets.

The same example is available as a checked script:

```powershell
python examples/toy_workflow.py
```

```python
from pathlib import Path

import numpy as np

import dfr
from dfr.analysis import analyze_dataset_modes
from dfr.plotting import plot_mode_count_curve, save_figure

root = Path("outputs/docs/toy_project")
data_dir = root / "dataset"
scenario_dir = root / "scenarios" / "toy"
data_dir.mkdir(parents=True, exist_ok=True)
scenario_dir.mkdir(parents=True, exist_ok=True)

trajectories = np.array(
    [
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [2.0, 0.0, 0.0]],
        [[0.1, 0.0, 0.0], [1.1, 0.2, 0.0], [2.1, 0.0, 0.0]],
    ],
    dtype=np.float32,
)
np.save(data_dir / "toy.npy", trajectories)
(scenario_dir / "config.yaml").write_text(
    "data_file: dataset/toy.npy\n",
    encoding="utf-8",
)

dataset = dfr.load_dataset("toy", project_root=root)
positions = dataset.positions_at_time_step(0)
print(dataset.trajectories.shape)  # (2, 3, 3)
print(positions.shape)             # (3, 3)

curve = analyze_dataset_modes(
    dataset,
    frame=0,
    scales=(0.1, 0.2, 0.4, 0.8, 1.6, 3.2),
    device="cpu",
)
print(curve.mode_counts)

figure, axes = plot_mode_count_curve(curve)
save_figure(figure, root / "mode_curve.png")
```

Notes:

- `load_dataset` accepts a scenario name, YAML config path, explicit data path,
  or `DatasetSpec`.
- `analyze_dataset_modes` uses world-coordinate scales.
- `plot_mode_count_curve` returns `(Figure, Axes)` and does not save by itself.

## 2. High-level analysis facade

Use `dfr.analyze` when you want the common package-level entry point:

```python
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
```

The facade currently accepts one frame at a time. `kind="modes"` uses
world-coordinate scales. `kind="dra"` uses scales normalized by the frame's mean
nearest-neighbour distance and requires CUDA.

## 3. Reconstruction with managed output

Reconstruction requires CUDA, a compiled rasterizer, and a scenario YAML with
camera intrinsics/image settings. Load by scenario name so the dataset retains
that config path.

```python
dataset = dfr.load_dataset("jackdaw2")

run = dfr.reconstruct(
    dataset,
    frames=(2800,),
    cameras=dfr.CameraConfig.encircling(count=4, padding=1.0),
    scale=1.0,  # omit for adaptive scale selection
    output=dfr.OutputConfig(
        workflow="reconstruction",
        name="jackdaw2 docs demo",
        run_id="docs-jackdaw2-frame-2800",
    ),
)

frame = run.frames[0]
print(frame.means.shape)       # (gaussians, 3)
print(frame.camera_poses.shape)  # (cameras, 7)
print(run.run_dir)
```

Without `output=...`, `dfr.reconstruct` returns the same typed arrays but
writes nothing. With `OutputConfig`, files go under:

```text
outputs/reconstruction/<run-id>/
  config.yaml
  manifest.json
  data/
  checkpoints/
  metrics/
```

Use `CameraConfig.explicit([...])` when you already have poses in
`(x, y, z, qx, qy, qz, qw)` order.

## 4. Evaluation and plots

`dfr.evaluate` accepts an in-memory `ReconstructionRun` or a managed
reconstruction run directory.

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

Evaluation writes only when given `OutputConfig(workflow="evaluation", ...)`.
Plot helpers consume typed result objects and still leave saving explicit:

```python
from dfr.plotting import (
    plot_evaluation_metric_series,
    plot_evaluation_summary,
    plot_frame_reconstruction_gmm_3d,
    save_figure,
)

figure, axes, artists = plot_frame_reconstruction_gmm_3d(frame)
save_figure(figure, "outputs/docs/reconstruction-gmm.png")

figure, axes = plot_evaluation_summary(evaluation)
save_figure(figure, "outputs/docs/evaluation-summary.png")

figure, axes = plot_evaluation_metric_series(evaluation)
save_figure(figure, "outputs/docs/evaluation-series.png")
```

## 5. Scenario sweeps and external observations

For repeatable named-scenario sweeps, use `ScenarioRunSpec`:

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
        name="jackdaw2 docs sweep",
        run_id="docs-jackdaw2-sweep",
    ),
)

run = dfr.run_scenario(spec, project_root=".")
```

Use `ExternalObservationFrame` and `dfr.reconstruct_observations` when 2D
projections come from measured detections or rendered images instead of the DFR
simulator. Each observation supplies:

- world-coordinate `positions` shaped `(agents, 3)`;
- one pixel-coordinate projection array per camera, shaped `(visible_agents, 2)`;
- the camera system that interprets those projections;
- optional camera poses shaped `(cameras, 7)`.

## 6. Output rules to remember

- New generated work belongs under ignored `outputs/`.
- Package-level functions return typed data and do not write artifacts unless
  given an explicit output object/path.
- `RunArtifacts` protects managed categories from absolute paths and path
  traversal.
- Legacy `figs/`, `results/`, and `scenarios/*/logs/` producers should not be
  added to new code.
