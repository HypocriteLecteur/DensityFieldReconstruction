# DFR Refactor Roadmap and Handoff

This file is the single source of truth for the refactor. Keep it current in
every refactor commit so another developer or agent can resume without relying
on chat history.

## Status

- **Current phase:** Phase 0 complete; Phase 1 is next.
- **Stable baseline:** annotated tag `v0.1.0`, commit `7cde21e`.
- **Baseline verification:** `git diff --check` and Python syntax compilation
  passed on 2026-07-06. Full reconstruction tests were not run because the
  available runtime does not include the project's CUDA rasterizer stack.
- **Canonical generated-output root:** `outputs/` (decision made; migration not
  started).
- **Compatibility policy:** preserve scientific behavior first; remove legacy
  entry points only after their replacements have characterization tests and a
  documented migration path.

## Why This Refactor Is Needed

The 2026-07-06 inventory found:

- `README.md` ends in an unfinished "Run Code" section and does not document
  datasets, analyses, reconstruction, evaluation, scripts, or outputs.
- `dfr/` contains 16 Python files and about 6,019 lines, but `dfr/__init__.py`
  is empty and several main classes have no class-level documentation.
- `experiments/` contains 31 Python files and about 21,124 lines.
- `experiments/dfr_plot.py` alone is 3,936 lines with 36 top-level functions.
- Scenario runners repeat large workflows. For example,
  `run_single_scenario`, `run_multi_scenarios`, metrics functions, baseline
  functions, and plotting functions appear in five to eight scripts each.
- Reusable loading, camera setup, scale analysis, metrics, and plotting logic
  lives in experiment scripts instead of the package.
- Generated files are written inconsistently to `figs/`, `results/`,
  `outputs/`, the repository root, and scenario log directories.

## Target User Experience

The final API should make the common path obvious and composable. Exact names
may change during implementation, but the workflow should remain this small:

```python
import dfr

dataset = dfr.load_dataset("jackdaw2")

analysis = dfr.analyze(
    dataset,
    frames=[2800],
    scales=dfr.ScaleRange(start=0.5, stop=3.0, count=30),
)

run = dfr.reconstruct(
    dataset,
    frames=[2800],
    cameras=dfr.CameraConfig.encircling(count=4),
    scale=analysis.recommended_scale,
    output=dfr.OutputConfig(name="jackdaw2-demo"),
)

evaluation = dfr.evaluate(run, ground_truth=dataset)
dfr.plot.reconstruction(run, evaluation=evaluation)
```

Equivalent command-line workflows should be available for reproducible runs:

```text
dfr dataset inspect jackdaw2
dfr analyze --dataset jackdaw2 --frames 2800 --config configs/analysis.yaml
dfr reconstruct --dataset jackdaw2 --config configs/reconstruction.yaml
dfr evaluate outputs/reconstruct/<run-id>
```

## Target Architecture

Keep experiment scripts thin. Reusable scientific behavior belongs in `dfr/`;
scripts should only parse arguments, construct configuration, call a package
workflow, and report the returned result.

```text
dfr/
  __init__.py                 # Small, documented public API
  config.py                   # Shared typed configuration
  data/
    base.py                   # Dataset protocol and frame types
    loaders.py                # Format-specific loaders
    registry.py               # Resolve scenario name or explicit path
  analysis/
    modes.py                  # Mode counting and mode curves
    scales.py                 # Scale sweeps and recommended-scale analysis
    manifold.py               # Reusable parameter-manifold computations
    metrics.py                # Evaluation metrics, no plotting or file I/O
    results.py                # Typed analysis result objects
  cameras/
    config.py                 # User-facing camera configuration
    geometry.py               # Camera generation and projection helpers
    system.py                 # Existing camera system implementation
  reconstruction/
    pipeline.py               # High-level reconstruction workflow
    reconstructor.py          # Core reconstruction implementation
    model.py                  # Density/Gaussian model
    results.py                # Typed reconstruction result objects
  plotting/
    analysis.py               # Mode/scale/manifold figures
    reconstruction.py         # Density, camera, GMM, and evaluation figures
    style.py                  # Shared style and save helpers
  evaluation/
    evaluator.py              # High-level evaluation workflow
    metrics.py                # TP/FP/FN/dMOTA and related metrics
  artifacts.py                # Run directories, manifests, paths, save/load
  workflows.py                # load/analyze/reconstruct/evaluate facade
  cli.py                      # Optional CLI after Python API stabilizes
experiments/
  ...                         # Thin, documented reproducibility entry points
configs/
  ...                         # Versioned example analysis/run configurations
outputs/                      # Ignored generated artifacts only
tests/
  unit/                       # CPU tests for package components
  integration/                # Small workflow tests
  cuda/                       # Explicitly marked GPU/rasterizer tests
```

This is a responsibility map, not a requirement to move every existing module
immediately. Prefer incremental adapters over a repository-wide rename.

## Output Contract

All newly generated artifacts must go through one output manager and default to:

```text
outputs/<workflow>/<run-id>/
  manifest.json               # Schema version, timestamp, commit, device
  config.yaml                 # Fully resolved run configuration
  data/                       # Computed arrays and reusable intermediate data
  checkpoints/                # Model checkpoints
  metrics/                    # Machine-readable metrics
  figures/                    # Figures produced from this run
  logs/                       # Text logs
  cache/                      # Safe-to-delete resumable computations
```

Rules:

- Library functions return data/results and do not save unless an explicit
  `OutputConfig` or path is supplied.
- Paths are `pathlib.Path` values and are resolved relative to an explicit
  project/output root, never implicitly from `os.getcwd()`.
- Do not write generated analysis data into `scenarios/` or `dataset/`.
- Do not introduce new writes to `figs/` or `results/`.
- Existing artifacts remain untouched until their producing scripts migrate.
- Generated outputs stay ignored by Git. Small curated fixtures belong under
  `tests/fixtures/`; publication assets need an explicit export process.

## Execution Plan

### Phase 0 - Preserve and Inventory

- [x] Snapshot all pre-refactor tracked and untracked work in commit `7cde21e`.
- [x] Create annotated stable tag `v0.1.0` at the snapshot.
- [x] Inventory Python file sizes, public documentation, duplicate top-level
  functions, and output path usage.
- [x] Define target workflows, architecture, output contract, and migration
  order in this file.
- [ ] Push `main` and tag `v0.1.0` to `origin`.

### Phase 1 - Safety Net and Project Contract

- [ ] Replace the unfinished README with a task-oriented guide covering:
  project purpose, supported datasets, installation, optional CUDA rasterizers,
  first dataset load, first analysis, first reconstruction/evaluation, output
  layout, script catalog, tests, and troubleshooting.
- [ ] Add a concise `CONTRIBUTING.md` describing environment setup, formatting,
  testing tiers, artifact policy, and how to update this TODO.
- [ ] Decide and document supported Python/CUDA/PyTorch versions in
  `pyproject.toml`; declare runtime and optional dependencies.
- [ ] Move/rename `test/` to `tests/` only after confirming test discovery is
  unchanged.
- [ ] Add CPU characterization tests for dataset loading, frame selection,
  camera geometry, mode counting, scale selection, metrics, and checkpoint I/O.
- [ ] Add marked CUDA smoke tests for one tiny reconstruction.
- [ ] Capture one small golden workflow fixture with tolerances; do not commit
  a large generated dataset.
- [ ] Add a test command that skips CUDA cleanly when unavailable.

**Exit criteria:** a new contributor can install the package and run CPU tests;
the current scientific behavior is protected before modules move.

### Phase 2 - Canonical Data API

- [ ] Define a documented dataset protocol with length/frame access, positions,
  optional velocities/timestamps, coordinate metadata, and ground truth.
- [ ] Introduce `DatasetSpec` and a registry that resolves either a known
  scenario name or an explicit config/data path.
- [ ] Refactor `DatasetFactory` behind `dfr.load_dataset(...)`; keep a temporary
  compatibility wrapper for current callers.
- [ ] Replace implicit working-directory assumptions with explicit project and
  data roots.
- [ ] Define frame-selection helpers shared by analysis and reconstruction.
- [ ] Validate errors for missing files, unsupported formats, invalid frames,
  and absent optional fields.
- [ ] Document every supported loader and include one minimal example each.

**Exit criteria:** loading a named scenario or explicit dataset takes one call,
has a stable return contract, and requires no imports from `experiments`.

### Phase 3 - Artifact and Configuration Foundation

- [ ] Add `OutputConfig`, run ID generation, and a `RunArtifacts` path manager.
- [ ] Write resolved config and a versioned manifest for every saved run.
- [ ] Centralize JSON/NPZ/checkpoint/figure saving with explicit overwrite and
  resume behavior.
- [ ] Migrate one representative analysis script and one reconstruction runner
  to the output contract before migrating the rest.
- [ ] Add a temporary warning/helper for legacy `figs/` and `results/` writes.
- [ ] Add `DatasetSpec`, `CameraConfig`, `AnalysisConfig`, `EvaluationConfig`,
  and top-level `RunConfig`, building on existing typed reconstruction configs.
- [ ] Ensure configs round-trip through YAML/JSON without Python-only values.

**Exit criteria:** migrated workflows place everything under one predictable run
directory and can be reproduced from the saved config and manifest.

### Phase 4 - Analysis API

- [ ] Move reusable mode-counting behavior from experiment scripts into
  `dfr.analysis`; keep visualization separate from computation.
- [ ] Extract scale sweep, model-order surface, fitting, caching, and recommended
  scale selection from `parameter_manifold*.py`,
  `plot_dra_scale_model_order.py`, `fit_dra_multiframe.py`, and related code.
- [ ] Define typed `ModeCurveResult`, `ScaleAnalysisResult`, and
  `ManifoldAnalysisResult` objects with save/load support.
- [ ] Provide `dfr.analyze(...)` plus lower-level functions for researchers who
  need custom pipelines.
- [ ] Make sampling, frame selection, random seeds, bounds, and scale grids
  explicit in configuration.
- [ ] Add unit tests against current cached/golden results with numeric
  tolerances.
- [ ] Reduce analysis scripts to CLI/config wrappers and document each script's
  inputs, outputs, runtime expectations, and example command.

**Exit criteria:** mode count and scale analysis can be run from an imported API
without copying code or invoking an experiment module.

### Phase 5 - Reconstruction and Evaluation Workflows

- [ ] Define a user-facing `CameraConfig` supporting explicit poses and common
  generated layouts (including encircling cameras).
- [ ] Extract scenario loading/camera setup from `experiments/common.py` into
  package services; experiment-specific presentation stays in `experiments/`.
- [ ] Add typed `ReconstructionRequest`, `FrameReconstruction`, and
  `ReconstructionRun` results.
- [ ] Provide a high-level `dfr.reconstruct(...)` that accepts dataset, frames,
  cameras, scale, reconstruction config, device, seed, and output config.
- [ ] Keep `DensityReconstructor.process_frame()` as a compatibility layer until
  all active scripts migrate.
- [ ] Separate metric computation from plotting and provide
  `dfr.evaluate(...)` with typed results.
- [ ] Consolidate the repeated scenario/table/flock/angle-sweep runner loops
  into one configurable runner.
- [ ] Verify representative old/new runs agree within declared tolerances.

**Exit criteria:** one concise API/config drives load -> camera setup ->
reconstruct -> evaluate, and scenario runners no longer duplicate the pipeline.

### Phase 6 - Plotting Decomposition

- [ ] Freeze a catalog of all 36 functions in `experiments/dfr_plot.py`, their
  callers, input data, and output files before moving them.
- [ ] Classify each function as reusable package plotting, experiment-only
  figure, computation that belongs in analysis/evaluation, or obsolete.
- [ ] Move shared style/save/layout logic from `experiments/plotting_utils.py`
  and duplicated scripts into `dfr.plotting`.
- [ ] Make plotting functions accept result objects/axes and return Figure/Axes;
  saving is optional and uses the output manager.
- [ ] Split publication/table-specific figures into small, named experiment
  scripts rather than one replacement monolith.
- [ ] Add headless smoke tests for representative 2D, 3D, camera, scale, and
  evaluation plots.
- [ ] Remove `experiments/dfr_plot.py` only after callers and documented commands
  migrate.

**Exit criteria:** no active plotting module is a grab bag, reusable plots are
importable, and every remaining experiment figure has a documented command.

### Phase 7 - Core Documentation and Public API

- [ ] Add module docstrings and complete public docstrings for data, camera,
  model, reconstruction, scale, mode, evaluation, plotting, and artifact APIs.
- [ ] Document units, coordinate systems, shapes/dtypes, device behavior,
  randomness, side effects, exceptions, and return contracts where relevant.
- [ ] Export only the intended common API from `dfr/__init__.py`; keep advanced
  APIs accessible from their submodules.
- [ ] Add runnable examples and API reference generation or a lightweight
  `docs/` tree.
- [ ] Add a script catalog table to the README and a module ownership map to
  developer docs.
- [ ] Test README/example commands in a clean environment where practical.

**Exit criteria:** common tasks are discoverable from README/API docs without
reading implementation or experiment source.

### Phase 8 - Cleanup and Release

- [ ] Remove compatibility wrappers only after all active callers migrate.
- [ ] Remove confirmed-dead copies and duplicated functions; use Git history
  rather than keeping backup files.
- [ ] Decide whether historical scripts belong in an archive branch, paper
  reproduction directory, or should remain at the stable tag only.
- [ ] Confirm `figs/` and `results/` are no longer written by active code, then
  archive or remove them in a separately reviewed change.
- [ ] Run CPU tests, CUDA tests, representative analyses, and end-to-end
  reconstruction/evaluation.
- [ ] Review docs from a clean clone and verify output paths are reproducible.
- [ ] Tag the completed refactor as the next semantic version and write release
  notes with migration examples from `v0.1.0`.

## Immediate Next Actions

The next agent should work on Phase 1 only:

1. Read this file and `README.md`, then inspect `pyproject.toml`,
   `environment.txt`, and existing tests.
2. Confirm a non-CUDA test command and document the actual environment; do not
   guess dependency versions.
3. Add characterization tests before moving package or experiment code.
4. Rewrite the README around currently working commands, clearly labeling the
   proposed high-level API as future work until implemented.
5. Update the Status, checklist, Decisions, and Handoff Log below in the same
   commit.

## Decisions

- **2026-07-06 - Stable marker:** use annotated tag `v0.1.0` at `7cde21e`.
  Reason: package metadata already declares version 0.1.0 and no prior tags
  exist.
- **2026-07-06 - Output root:** use ignored `outputs/` for all generated
  artifacts. Reason: "results" is ambiguous and `outputs/` already has an ignore
  policy; per-run subdirectories retain the distinctions between data, metrics,
  figures, checkpoints, logs, and cache.
- **2026-07-06 - Migration style:** incrementally add package APIs and adapters
  before deleting experiment code. Reason: CUDA-dependent scientific behavior
  needs comparison points and cannot safely survive a big-bang rewrite.
- **2026-07-06 - API shape:** center the package on load, analyze, reconstruct,
  evaluate, and plot workflows with typed configs/results. Reason: these map to
  the actual research tasks and allow both concise defaults and lower-level use.

## Handoff Log

Add one newest-first entry per working session. Include commit(s), verification,
known failures, and the exact next step.

### 2026-07-06 - Baseline and architecture plan

- Preserved the current working tree in `7cde21e` and tagged it `v0.1.0`.
- Audited documentation, package/experiment size, duplicate function names, and
  generated-output paths.
- Created this roadmap; no refactor implementation has started.
- Verification: `git diff --check`; bundled Python `compileall` for `dfr/` and
  `experiments/` (passed, with two pre-existing invalid-escape warnings in
  `experiments/power_law.py`).
- Known limitation: no full CUDA/rasterizer test run.
- Next step: Phase 1 safety-net and documentation work.

## Handoff Rules

Before ending any refactor session:

1. Check off only work that is actually complete.
2. Update **Status** and **Immediate Next Actions**.
3. Record architectural choices in **Decisions** rather than leaving them only
   in code or chat.
4. Add a **Handoff Log** entry with changed files/commits and test outcomes.
5. Leave the worktree understandable: report intentional uncommitted files and
   never hide failed tests.
6. Keep changes scoped to one phase or independently reviewable slice.
