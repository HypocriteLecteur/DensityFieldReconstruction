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

The manifold workflows still read the historic `modes.npy`,
`scale_range.npy`, and `nn_dists.npy` files under each scenario. If these
expensive caches are absent, the 3PL workflow may generate them in place for
compatibility. Fitted results and all newly selected figures use managed output.

## Legacy studies with explicit dispatch

These modules preserve research-history functions that have not yet been
promoted to stable package APIs. They no longer run an arbitrary hard-coded
study merely because the file was executed. A subcommand is required:

- `python -m experiments.power_law --help`
- `python -m experiments.reconstruction_scale_determination --help`

They may still write legacy `figs/` or scenario cache locations and are not a
supported output contract. Use them only to reproduce an identified historical
study, then migrate that study separately if it remains scientifically active.

## Deferred plotting decomposition

`dfr_plot.py` is a mixed 3,900-line publication/reconstruction plotting archive,
not a supported analysis CLI. Direct execution no longer launches its former
hard-coded animation. `python -m experiments.dfr_plot --list-functions` lists
the retained functions. Their inventory and decomposition are Phase 6 work.

`plotting_utils.py` is a helper module rather than an entry point.

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

- Interactive diagnostics: `dataset_viewer_test.py`, `inspect_scenarios.py`,
  `inspect_3d_error.py`, and `investigate_initialization.py`.
- Rendering/animation tools: `generate_scene_animations.py`,
  `rasterizer_optimize.py`, `run_post_processing.py`, and `visualize.py`.
- Reconstruction/tuning workflows: `reconstruct_one_frame.py`,
  `compute_metrics_from_pretrained.py`, `search_learning_parameters.py`,
  `search_regularization_parameters.py`, and all `run_scenarios*.py` files.

Those scripts belong to Phases 5 and 6 rather than the Phase 4 analysis API.
Some predate managed artifacts; consult the root `README.md` catalog before
running them.
