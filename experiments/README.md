# Experiment Entry Points

This directory contains supported reproducibility commands, legacy studies,
diagnostic viewers, and shared helpers. Only commands in the **Supported
analysis CLIs** table are part of the Phase 4 analysis surface. They parse
their configuration explicitly, call reusable computation from `dfr.analysis`
where available, and put new artifacts under a managed
`outputs/analysis/<run-id>/` directory.

All CUDA analyses should first be tried with `--help`. Use a unique `--run-id`,
or explicitly select `--resume`/`--overwrite-run`; an existing run is never
silently replaced.

## Supported analysis CLIs

| Command | Inputs | Managed outputs | Runtime / example |
|---|---|---|---|
| `python -m experiments.plot_dra_scale_model_order` | Named datasets, one frame per dataset, normalized mean-NND scale grid, model-order grid, voxel controls | DRA cache, fitted surfaces, JSON metrics, surface figure | CUDA intensive; minutes per uncached frame. Example: `--datasets jackdaw2 --run-id jackdaw2-dra` |
| `python -m experiments.fit_dra_multiframe` | Named datasets, sampled frames, normalized scale/model-order grid | Per-frame caches, pooled fit arrays/metrics, figures | CUDA intensive; potentially hours uncached. Example: `--datasets jackdaw2 --frames-per-dataset 3 --run-id jackdaw2-multiframe` |
| `python -m experiments.parameter_manifold` | `swift`, `starling`, `jackdaw`, and `jackdaw2`; legacy mode-curve caches or source data; saturation/seed | 3PL parameters, cluster labels, shape projection, summary, nine figures | Cached fitting takes minutes; missing CUDA mode caches can take hours. Example: `--no-display --run-id manifold-3pl` |
| `python -m experiments.parameter_manifold_2pl` | Same biological mode caches as the 3PL workflow | 2PL parameter table, summary, four figures | Usually minutes with caches. Example: `--no-display --run-id manifold-2pl` |
| `python -m experiments.mechanistic_derivation` | Synthetic agent/trial counts plus cached empirical datasets | Synthetic curve arrays and comparison figure | CUDA; runtime grows with `agents * trials * scales`. Example: `--agents 200 --trials 5 --no-display --run-id mechanism` |
| `python -m experiments.synthetic_benchmark` | Trial count, agent count, seed | Synthetic process 3PL parameter/label caches | CUDA; minutes to hours, especially LGCP generation. Example: `--trials 5 --agents 200 --run-id synthetic-smoke` |
| `python -m experiments.validate_mode_counting` | Fixed validation grid and explicit seed | JSON validation summary and accuracy figure | CUDA; dozens of mode sweeps. Example: `--no-display --seed 42 --run-id validate-modes` |

Common options on the newly migrated manifold/synthetic/validation commands:

- `--project-root`: repository/data root, independent of the process working directory.
- `--output-root`: output root relative to the project root; defaults to `outputs`.
- `--run-id`: stable run identifier.
- `--resume` or `--overwrite-run`: explicit collision policy.
- `--no-display`: suppress interactive Matplotlib windows where figures exist.

The manifold workflows cache `modes.npy`, `scale_range.npy`, and `nn_dists.npy`
inside each managed analysis run. Resume that run to continue an expensive
calculation; they do not create scenario-local caches. Historical power-law and
reconstruction-scale studies were retired in Phase 8 and remain recoverable
from local Git history only.

## Deferred plotting decomposition

The former mixed publication/reconstruction plotting archive was removed in
Phase 8. `python -m experiments.plot_catalog --list-functions` lists its 30
frozen public names without importing historical code; the full migration
record remains in [`DFR_PLOT_CATALOG.md`](DFR_PLOT_CATALOG.md). Use
`dfr.plotting` for reusable camera, trajectory, projection, mode-count, DRA,
multiscale-density, and density/GMM plots. Use
`plot_publication_table2.py`, `plot_publication_time_efficiency.py`, and
`plot_publication_noise_robustness.py` for their named hard-coded publication
figures. Archive-only historical functions are available through local Git
history and `v0.1.0`, not through an active experiment import.

Supported analysis CLIs now route figure exports through
`dfr.plotting.save_figure` or managed artifact helpers instead of direct
Matplotlib `savefig` calls.

## Diagnostics and non-analysis workflows

### Publication reconstruction tables

Tables 2–4 share `publication_scenarios.py` and the package-level
`ScenarioRunSpec` runner. Their historical filenames remain as thin commands:

- `python -m experiments.run_scenarios_table_2 reconstruct --help` — camera
  counts 2/3/5, 100 optimization iterations, four biological datasets.
- `python -m experiments.run_scenarios_table_3 reconstruct --help` — the same
  study with 500 iterations.
- `python -m experiments.run_scenarios_table_4 reconstruct --help` —
  projection-noise study; the historical active preset selects `starling`.

Use action `run` instead of `reconstruct` to evaluate each in-memory result.
Both stages use `outputs/<workflow>/<run-id>/`; dataset, camera, noise, and
iteration settings are encoded in each run ID. Commands require an action so
importing or accidentally invoking a module never launches a long CUDA study.

Publication figure split-outs are explicit commands too:

- `python -m experiments.plot_publication_table2 --help` — render the
  hard-coded Table 2 capacity-scaling and recall/hallucination tradeoff
  figures formerly owned by `dfr_plot.py`.
- `python -m experiments.plot_publication_time_efficiency --help` — render the
  hard-coded training-time scaling figure formerly owned by `dfr_plot.py`.
- `python -m experiments.plot_publication_noise_robustness --help` — render
  the hard-coded noise-robustness figure formerly owned by `dfr_plot.py`.

The remaining angle, flock, and UE4 differences are cataloged in
`RUNNER_SPECIALIZATIONS.md`. All three now require explicit command dispatch.
The ordinary angle reconstruction uses `ScenarioRunSpec`; measured flock
detections and UE4 image detections remain legacy specializations until the
package exposes a typed external-observation workflow.

- Rendering/animation tools: `generate_scene_animations.py`, a managed CLI
  that writes MP4 files under `outputs/animations/<run-id>/figures/`; use
  `python -m experiments.generate_scene_animations --help` before launching
  the configured CUDA/FFmpeg study.
- Reconstruction workflows: `reconstruct_one_frame.py` and all
  `run_scenarios*.py` files. Historical checkpoint evaluation and unmanaged
  hyperparameter-search scripts were retired in Phase 8; use package
  evaluation, managed scenario runs, and versioned configs instead.

Those scripts belong to Phases 5 and 6 rather than the Phase 4 analysis API.
Some predate managed artifacts; consult the root `README.md` catalog before
running them.
