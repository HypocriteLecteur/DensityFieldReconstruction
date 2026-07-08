# DFR Refactor Roadmap and Handoff

This file is the single source of truth for the refactor. Keep it current in
every refactor commit so another developer or agent can resume without relying
on chat history.

## Status

- **Current phase:** Phase 6 plotting decomposition is in progress. The
  `experiments/dfr_plot.py` function catalog is frozen in
  `experiments/DFR_PLOT_CATALOG.md`; reusable camera-configuration,
  trajectory-snapshot, 2D projection/GMM, mode-count curve, DRA
  scale/model-order surface, multiscale density, 3D density/GMM rendering,
  shared style/save/layout primitives, typed analysis-result plotting paths,
  and typed frame-reconstruction GMM plotting have moved to `dfr.plotting`.
- **Stable baseline:** annotated tag `v0.1.0`, commit `7cde21e`.
- **Version storage:** local Git repository only; do not push unless the owner
  explicitly changes this policy.
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
- [x] Keep `main` and tag `v0.1.0` locally; no remote push is required.

### Phase 1 - Safety Net and Project Contract

- [x] Replace the unfinished README with a task-oriented guide covering:
  project purpose, supported datasets, installation, optional CUDA rasterizers,
  first dataset load, first analysis, first reconstruction/evaluation, output
  layout, script catalog, tests, and troubleshooting.
- [x] Add a concise `CONTRIBUTING.md` describing environment setup, formatting,
  testing tiers, artifact policy, and how to update this TODO.
- [x] Decide and document supported Python/CUDA/PyTorch versions in
  `pyproject.toml`; declare runtime and optional dependencies.
- [x] Move/rename `test/` to `tests/` only after confirming test discovery is
  unchanged.
- [x] Add CPU characterization tests for dataset loading, frame selection,
  camera geometry, mode counting, scale selection, metrics, and checkpoint I/O.
- [x] Add marked CUDA rasterizer smoke tests with tiny inputs.
- [x] Add a marked CUDA smoke test for one tiny end-to-end reconstruction.
- [x] Capture one small golden workflow fixture with tolerances; do not commit
  a large generated dataset.
- [x] Add a test command that skips CUDA cleanly when unavailable.

**Exit criteria:** a new contributor can install the package and run CPU tests;
the current scientific behavior is protected before modules move.

### Phase 2 - Canonical Data API

- [x] Define a documented dataset protocol with length/frame access, positions,
  optional velocities/timestamps, coordinate metadata, and ground truth.
- [x] Introduce `DatasetSpec` and a registry that resolves either a known
  scenario name or an explicit config/data path.
- [x] Refactor `DatasetFactory` behind `dfr.load_dataset(...)`; keep a temporary
  compatibility wrapper for current callers.
- [x] Replace implicit working-directory assumptions in the canonical loading
  API and migrated DRA callers with explicit project and data roots.
- [x] Define frame-selection helpers shared by analysis and reconstruction.
- [x] Validate errors for missing files, unsupported formats, invalid frames,
  and absent optional fields.
- [x] Document every supported loader and include one minimal example each.

**Exit criteria:** loading a named scenario or explicit dataset takes one call,
has a stable return contract, and requires no imports from `experiments`.

### Phase 3 - Artifact and Configuration Foundation

- [x] Add `OutputConfig`, run ID generation, and a `RunArtifacts` path manager.
- [x] Write resolved config and a versioned manifest for every saved run.
- [x] Centralize JSON/NPZ/checkpoint/figure saving with explicit overwrite and
  resume behavior.
- [x] Migrate one representative analysis script and one reconstruction runner
  to the output contract before migrating the rest.
- [x] Add a temporary warning/helper for legacy `figs/` and `results/` writes.
- [x] Add `DatasetSpec` (completed in Phase 2).
- [x] Add `CameraConfig`, `AnalysisConfig`, `EvaluationConfig`, and top-level
  `RunConfig`, building on existing typed reconstruction configs.
- [x] Ensure current dataclass, path, NumPy, tensor, device, and resolved output
  configs serialize through YAML/JSON without Python-only values.

**Exit criteria:** migrated workflows place everything under one predictable run
directory and can be reproduced from the saved config and manifest.

### Phase 4 - Analysis API

- [x] Move reusable mode-counting behavior from experiment scripts into
  `dfr.analysis`; keep visualization separate from computation.
- [x] Extract DRA scale/model-order surface computation and single/multiframe
  fitting from `plot_dra_scale_model_order.py` and `fit_dra_multiframe.py`.
- [x] Extract parameter-manifold fitting, caching, and recommended-scale
  selection from `parameter_manifold*.py` and related code.
- [x] Define typed `ModeCurveResult`, `ScaleAnalysisResult`, and
  `ManifoldAnalysisResult` objects with save/load support.
- [x] Provide `dfr.analyze(...)` plus lower-level functions for researchers who
  need custom pipelines.
- [x] Make sampling, frame selection, random seeds, bounds, and scale grids
  explicit in configuration.
- [x] Add deterministic CPU golden-fit/cache/result tests and a CUDA identity
  test for extracted DRA computation.
- [x] Add compatibility tests against representative existing research caches.
- [x] Reduce all supported analysis scripts to explicit CLI/config entry points
  and document each script's
  inputs, outputs, runtime expectations, and example command.

**Exit criteria:** mode count and scale analysis can be run from an imported API
without copying code or invoking an experiment module.

### Phase 5 - Reconstruction and Evaluation Workflows

- [x] Define a user-facing `CameraConfig` supporting explicit poses and common
  generated layouts (including encircling cameras).
- [x] Extract scenario loading/camera setup from `experiments/common.py` into
  package services; experiment-specific presentation stays in `experiments/`.
- [x] Add typed `ReconstructionRequest`, `FrameReconstruction`, and
  `ReconstructionRun` results.
- [x] Provide a high-level `dfr.reconstruct(...)` that accepts dataset, frames,
  cameras, scale, reconstruction config, device, seed, and output config.
- [x] Keep `DensityReconstructor.process_frame()` as a compatibility layer until
  all active scripts migrate.
- [x] Separate metric computation from plotting and provide
  `dfr.evaluate(...)` with typed results.
- [x] Consolidate the remaining repeated angle-sweep/UE4 runner loops into
  typed package workflows.
- [x] Verify representative old/new runs agree within declared tolerances.

**Exit criteria:** one concise API/config drives load -> camera setup ->
reconstruct -> evaluate, and scenario runners no longer duplicate the pipeline.

### Phase 6 - Plotting Decomposition

- [x] Freeze a catalog of all 36 functions in `experiments/dfr_plot.py`, their
  callers, input data, and output files before moving them.
- [x] Classify each function as reusable package plotting, experiment-only
  figure, computation that belongs in analysis/evaluation, or obsolete.
- [x] Move shared style/save/layout logic from `experiments/plotting_utils.py`
  and duplicated scripts into `dfr.plotting`.
  - Complete for active supported plotting surfaces: `apply_academic_style`,
    3D axis styling, 3D view/layout helpers, and lightweight figure-saving
    defaults now live in `dfr.plotting`; `experiments` compatibility helpers
    delegate to the package. Supported analysis CLIs no longer call Matplotlib
    `savefig`, `tight_layout`, or `subplots_adjust` directly. Multiscale 3D
    density rendering and the remaining 3D density/GMM renderers have also
    moved to `dfr.plotting`. Legacy exploratory scripts still contain local
    layout calls and should migrate them when their owning figures are promoted.
- [ ] Make plotting functions accept result objects/axes and return Figure/Axes;
  saving is optional and uses the output manager.
  - Started: mode-count and DRA scale/model-order plotting now accept
    `ModeCurveResult` and `ScaleAnalysisResult` directly while preserving the
    existing array/legacy tuple APIs and Figure/Axes return contracts.
    `FrameReconstruction` now has a direct 3D reconstructed-GMM plot path; raw
    density-grid plots still require explicit density/tick arrays because the
    reconstruction result object does not store a reusable voxel density grid.
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

The next agent should continue Phase 6:

1. Continue moving low-risk reusable plot primitives identified in
   `experiments/DFR_PLOT_CATALOG.md`; next candidates are evaluation/noise plots
   after their computation has typed result objects, or publication/table
   figure split-outs.
2. Continue Phase 6 step 4 by adding typed adapters for evaluation plots
   (`EvaluationRun`/`FrameEvaluation`) or by extracting the next noise/dMOTA
   plotting primitive after its computation has a typed result object.
3. Migrate legacy `figs/` saving for `plot_camera_configurations` to an
   explicit output/artifact option after deciding whether these migrated
   wrappers remain supported experiment CLIs or only compatibility wrappers.
4. Add headless plotting smoke tests before migrating each high-value figure.
5. Keep the Phase 4 supported/legacy classification in
   `experiments/README.md` current when promoting another research study.

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
- **2026-07-06 - Version storage:** keep commits and release tags on the local
  machine. Do not push to GitHub unless the repository owner explicitly changes
  this decision. Reason: local version maintenance is sufficient and avoids
  exporting source code and result files.
- **2026-07-06 - Test discovery:** restrict pytest to `tests/`. Reason: default
  discovery imported `experiments/dataset_viewer_test.py`, which performs heavy
  interactive work during import and caused collection to hang.
- **2026-07-06 - Metric devices:** honor the existing `device` argument in
  `compute_metrics_batched_torch` and support CPU execution. Reason: this makes
  metric behavior testable without changing the density/overlap equations used
  on CUDA.
- **2026-07-06 - Dataset resolution:** canonical loading accepts a scenario
  name, YAML config, explicit data path, or `DatasetSpec`, and stores absolute
  paths before loading. Reason: downstream analysis/reconstruction must not
  change behavior with the process working directory.
- **2026-07-06 - Config-relative data:** registered scenario configs retain the
  legacy rule that `data_file` is relative to the project root; standalone YAML
  configs resolve relative data beside the config. Reason: preserve existing
  scenarios while making portable external configs intuitive.
- **2026-07-06 - Managed run identity:** use
  `outputs/<workflow>/<run-id>/` with required explicit resume/overwrite
  policies. Reason: accidental reuse must fail, while legitimate resumable
  analyses retain caches and verify their resolved scientific configuration.
- **2026-07-06 - Provenance schemas:** manifest and resolved-config schemas
  start at version 1. Reason: saved runs need a migration point independent of
  Python class layout or package version.
- **2026-07-06 - Transitional reconstruction CLI:** add a thin managed
  one-frame runner now, but defer the stable Python workflow API to Phase 5.
  Reason: users need a straightforward current command without freezing the
  eventual orchestration architecture prematurely.
- **2026-07-06 - Shared config scope:** common configs contain only reusable
  camera layout, frame/scale selection, evaluation-grid, dataset, output,
  training, reconstruction, and seed fields. Reason: paper-specific controls
  belong in experiment configs rather than the package-wide contract.
- **2026-07-06 - Multiframe DRA timing:** migrate `fit_dra_multiframe.py` during
  Phase 4 analysis extraction, not as an isolated Phase 3 path rewrite. Reason:
  its per-frame caches and fit objects should adopt the same typed result schema
  as the single-frame DRA implementation once that package API exists.
- **2026-07-06 - Analysis result boundary:** typed results contain arrays and
  explicit NPZ save/load only; experiment scripts decide managed paths, cache
  resume timing, and plotting. Reason: scientific computation must be reusable
  without silently writing or importing presentation code.
- **2026-07-06 - Legacy DRA compatibility:** keep the six-value tuple adapter
  and accept old caches when callers supply missing dataset/agent context.
  Reason: existing `dfr_plot.py` and expensive cached sweeps must remain usable
  during incremental migration.
- **2026-07-06 - Explicit analysis dispatch:** `dfr.analyze` requires a named
  `kind` and exactly one frame in its first version; mode scales use world
  units, while DRA scales use mean-NND multiples and dispatch visibly to CUDA.
  Reason: the convenient facade must not conceal an expensive analysis or
  silently invent a scale grid.
- **2026-07-06 - Reconstruction result boundary:** `dfr.reconstruct` always
  returns detached CPU arrays in typed per-frame results; managed persistence
  is optional and selected only with `OutputConfig`. Reason: researchers need
  immediate composable data without coupling computation to filesystem output.
- **2026-07-06 - Camera orientation:** generated encircling cameras auto-aim at
  each reconstructed frame, while explicit poses preserve their supplied
  quaternions. Reason: an explicit pose should not be silently reoriented.
- **2026-07-06 - Supported analysis surface:** supported Phase 4 CLIs are the
  two DRA workflows, 3PL/2PL manifold workflows, mechanistic derivation,
  synthetic benchmark, and mode-count validation. `power_law.py` and
  `reconstruction_scale_determination.py` preserve explicit legacy studies;
  `dfr_plot.py` is deferred to Phase 6 and cannot launch an implicit figure.
  Reason: a finite supported surface can have managed, tested contracts without
  pretending every exploratory notebook-like script is production workflow.
- **2026-07-06 - Evaluation semantics:** preserve the existing voxelized
  density-mass definitions: recall=TP/ground-truth mass,
  hallucination=FP/predicted mass, miss=FN/ground-truth mass, and
  dMOTA=1-(FN+FP)/ground-truth mass. Reason: typed APIs and device controls must
  not silently redefine published metrics.
- **2026-07-06 - Legacy camera quaternion:** the old runner's initial
  `[1, 0, 0, 0]` quaternion is not retained. Encircling cameras use valid xyzw
  identity before auto-aim; a characterization test proves post-auto-aim poses
  and projections match. Reason: the initial orientation is overwritten and
  should not propagate a misleading convention into the public API.
- **2026-07-06 - Publication table profiles:** retain Tables 2, 3, and 4 as
  named experiment presets over one package scenario runner; require an
  explicit CLI action and write reconstruction/evaluation to separate managed
  workflows. Reason: iteration, dataset, camera, and noise differences remain
  visible without keeping three 869-line executable copies.
- **2026-07-06 - External observation boundary:** measured flock detections
  and thresholded UE4 detections must not be routed through the canonical
  simulated-projection runner. Reason: both require asymmetric or time-varying
  camera state and externally observed 2D point sets; replacing them with
  `simulate_vision` would silently change the experiment.
- **2026-07-07 - External observation API:** use
  `ExternalObservationFrame` plus `reconstruct_observations(...)` for measured
  or image-derived detections. Reason: these workflows need one externally
  supplied 2D point set per camera, camera-system provenance, optional
  per-frame camera poses, and visibility masks, while still returning the
  standard `ReconstructionRun` and managed artifacts.
- **2026-07-07 - Phase 5 runner consolidation complete:** ordinary scenario
  reconstruction uses `ScenarioRunSpec`, and measured/time-varying external
  detections use `ExternalObservationFrame`. Remaining direct
  `DensityReconstructor` calls in angle-sweep are specialized studies rather
  than repeated ordinary scenario loops. Reason: deleting those kernels would
  remove distinct camera-angle, profiling, and convergence experiments rather
  than consolidating duplicate orchestration.
- **2026-07-08 - Plot layout migration boundary:** Phase 6 step 3 covers
  shared layout/style/save behavior for `dfr.plotting`, supported analysis
  CLIs, and experiment compatibility wrappers. Legacy exploratory scripts may
  keep local Matplotlib layout calls until their figures are promoted or split.
  Reason: replacing every historic `tight_layout` call in place would churn
  figure-specific legacy scripts without improving the public plotting API.

## Handoff Log

Add one newest-first entry per working session. Include commit(s), verification,
known failures, and the exact next step.

### 2026-07-08 - Phase 6 typed reconstruction GMM plot path

- Added `render_frame_reconstruction_gmm_3d` and
  `plot_frame_reconstruction_gmm_3d` so a typed `FrameReconstruction` can be
  rendered directly as 3D GMM wireframes, mean markers, and optional source
  agent positions.
- Exported the new helpers from `dfr.plotting` and documented their explicit
  save behavior in the README reconstruction section.
- Kept density-grid rendering array-based because `FrameReconstruction` stores
  the reconstructed GMM and original positions, not a reusable voxel grid or
  density tensor.
- Verification: focused density/analysis/entrypoint tests (27 passed);
  `compileall`; `git diff --check`; `pytest -m "not cuda"` (145 passed,
  7 deselected, 1 warning); `pytest -m cuda` (6 passed, 1 skipped,
  145 deselected).
- Next step: continue Phase 6 step 4 with typed evaluation plot adapters or
  extract the next noise/dMOTA plotting primitive after giving its computation
  a package result object.

### 2026-07-08 - Phase 6 typed analysis plot contracts started

- Extended `dfr.plotting.plot_mode_count_curve` to accept
  `ModeCurveResult` directly, inferring dataset/frame labels while preserving
  the existing array-based API.
- Extended `plot_dra_scale_model_order_surface` and
  `plot_dra_surface_grid` to accept `ScaleAnalysisResult` objects directly
  while retaining legacy tuple compatibility for migrated experiment wrappers.
- Documented typed analysis plotting in the README and added headless tests for
  typed result plotting plus validation of mixed result/array arguments.
- Verification: focused plotting/catalog/entrypoint tests (27 passed);
  `compileall`; `git diff --check`; `pytest -m "not cuda"` (143 passed,
  7 deselected, 1 warning); `pytest -m cuda` (6 passed, 1 skipped,
  143 deselected).
- Next step: continue Phase 6 step 4 with typed adapters for reconstruction or
  density plotting helpers whose stable package results already exist, or
  extract the next evaluation/noise plot after its computation has a typed
  result object.

### 2026-07-08 - Phase 6 shared layout helpers completed

- Added `set_3d_view`, `prepare_3d_axis`, and `apply_figure_layout` to
  `dfr.plotting.style` and exported them from `dfr.plotting`.
- Migrated reusable plotting modules and supported analysis CLIs to use package
  layout/view helpers instead of local `tight_layout`, `subplots_adjust`, or
  `view_init` calls.
- Added static entrypoint coverage so supported analysis figure commands keep
  using package layout helpers, plus direct helper tests for 2-value/3-value
  3D views and tight/adjust layout modes.
- Remaining legacy calls are intentionally left in exploratory or deferred
  scripts such as `experiments/dfr_plot.py`, `power_law.py`, and old
  interactive viewers until their owning figures are promoted.
- Verification: focused plotting/entrypoint suite (35 passed); `compileall`;
  `git diff --check`; `pytest -m "not cuda"` (140 passed, 7 deselected,
  1 warning); `pytest -m cuda` (6 passed, 1 skipped, 140 deselected).
- Next step: begin Phase 6 step 4 by tightening result-object/axes contracts
  for migrated plots and extracting the next evaluation/noise plotting
  primitive with headless tests.

### 2026-07-08 - Phase 6 density/GMM plotting utilities completed

- Added remaining 3D density/GMM rendering helpers to `dfr.plotting.density`:
  GMM wireframes, GMM mean overlays, GT-style density composites, and
  reconstructed-GMM composites.
- Exported the new density/GMM helpers from `dfr.plotting`.
- Converted `experiments.plotting_utils` rendering helpers into compatibility
  wrappers over `dfr.plotting`, preserving old function names and layer
  constants for legacy callers.
- Added headless tests for GMM wireframes, GMM means, reconstructed-GMM
  composites, validation behavior, and static wrapper delegation.
- Verification: focused density/entrypoint/style tests (17 passed);
  `compileall`; `git diff --check`; `pytest -m "not cuda"` (136 passed,
  7 deselected, 1 warning); `pytest -m cuda` (6 passed, 1 skipped,
  136 deselected).
- Next step: continue Phase 6 step 3 with repeated layout helpers, or begin
  the next evaluation/noise plot extraction after typing its computation.

### 2026-07-07 - Phase 6 multiscale density renderer

- Added `dfr.plotting.density` with reusable 3D density-shell rendering, agent
  overlays, single density-field plotting, and multiscale density series
  plotting.
- Migrated `experiments.dfr_plot.plot_jackdaw2_multiscale_density` to delegate
  rendering to `dfr.plotting.plot_multiscale_density_fields` while preserving
  legacy mode-count/density caches and `figs/` filenames.
- Added headless tests for single 3D density rendering, multiscale density
  series rendering, validation errors, and the legacy wrapper delegation.
- Verification: focused density/catalog/style tests (13 passed); `compileall`;
  `git diff --check`; `pytest -m "not cuda"` (133 passed, 7 deselected,
  1 warning); `pytest -m cuda` (6 passed, 1 skipped, 133 deselected).
- Next step: continue Phase 6 step 3 with remaining density/GMM helpers or
  start the next typed evaluation-plot extraction.

### 2026-07-07 - Phase 6 supported figure saving centralized

- Migrated remaining raw Matplotlib `savefig` calls in supported analysis
  entry points to `dfr.plotting.save_figure`: `fit_dra_multiframe`,
  `parameter_manifold`, `parameter_manifold_2pl`, `mechanistic_derivation`,
  and `validate_mode_counting`.
- Migrated the remaining manifold style hooks to `dfr.plotting.apply_academic_style`.
- Added static guards so supported analysis scripts do not reintroduce raw
  `.savefig(` calls and the manifold entry points keep using package style.
- Verification: focused entrypoint/style tests (11 passed); `compileall`;
  `git diff --check`; `pytest -m "not cuda"` (129 passed, 7 deselected,
  1 warning); `pytest -m cuda` (6 passed, 1 skipped, 129 deselected).
- Next step: continue Phase 6 step 3 with shared layout helpers or move the
  multiscale-density panel renderer.

### 2026-07-07 - Phase 6 shared style/save helpers started

- Added package-level `dfr.plotting.style_3d_axis` and `dfr.plotting.save_figure`
  beside the existing `apply_academic_style` helper.
- Migrated `experiments.plotting_utils._set_academic_style` and
  `_style_3d_ax` into compatibility delegates over `dfr.plotting`, preserving
  old import names for remaining legacy figure code.
- Switched `dfr.plotting.analysis` to use the shared academic style helper
  instead of local `plt.rcParams.update` dictionaries.
- Switched the supported `experiments.plot_dra_scale_model_order` figure save
  path to `dfr.plotting.save_figure`; managed artifact provenance remains the
  responsibility of `RunArtifacts` where full run metadata is needed.
- Added headless tests for academic-style overrides, 3D-axis styling,
  figure-saving parent creation, and static delegation guards.
- Verification: focused style/entrypoint/analysis-plot tests (16 passed);
  `compileall`; `git diff --check`; `pytest -m "not cuda"` (127 passed,
  7 deselected, 1 warning); `pytest -m cuda` (6 passed, 1 skipped,
  127 deselected).
- Next step: run full verification, then continue step 3 by replacing more
  repeated figure-save/layout calls or migrate the multiscale-density panel.

### 2026-07-07 - Phase 6 DRA surface plotting primitive

- Added reusable DRA scale/model-order surface renderers to
  `dfr.plotting.analysis`: one single-surface primitive and one managed
  multi-dataset grid helper. Both return Matplotlib figure/axes objects and
  leave saving to callers.
- Migrated `experiments.dfr_plot.plot_jackdaw2_dra_scale_model_order_surface`
  to delegate its 3D DRA surface and fitted-wireframe rendering to
  `dfr.plotting` while preserving its legacy CUDA/cache computation and
  `figs/` save behavior.
- Migrated the supported `experiments.plot_dra_scale_model_order.plot_surfaces`
  managed-run figure to use the same package grid renderer.
- Added headless tests for the single DRA surface, DRA grid, validation paths,
  legacy wrapper delegation, and supported DRA CLI delegation.
- Verification: focused plotting/catalog/entrypoint tests (18 passed);
  `compileall`; `git diff --check`; `pytest -m "not cuda"` (123 passed,
  7 deselected, 1 warning); `pytest -m cuda` (6 passed, 1 skipped,
  123 deselected).
- Next step: migrate the multiscale-density panel renderer or revisit legacy
  `figs/` saves for wrappers that already delegate to `dfr.plotting`.

### 2026-07-07 - Phase 6 mode-count plotting primitive

- Added `dfr.analysis.scales` with reusable NND-bound validation and adaptive
  representative-scale selection previously embedded in `experiments.dfr_plot`.
- Added `dfr.plotting.analysis.plot_mode_count_curve`, a headless
  data-first renderer for empirical mode-count curves that returns
  Matplotlib figure/axes objects and does not save by itself.
- Migrated `experiments.dfr_plot` private scale helpers to compatibility
  delegates and migrated `plot_jackdaw2_mode_count_curve` to use the package
  renderer while preserving its legacy CUDA/cache computation and `figs/`
  output behavior.
- Marked the Phase 6 classification task complete because
  `experiments/DFR_PLOT_CATALOG.md` classifies all 36 retained top-level
  functions and now records migration status for the moved helpers/wrapper.
- Added headless tests for NND-bound validation, adaptive scale selection,
  mode-count curve rendering, input validation, and legacy wrapper delegation.
- Verification: focused plotting/catalog tests (19 passed); `compileall`;
  `git diff --check`; `pytest -m "not cuda"` (118 passed, 7 deselected,
  1 warning); `pytest -m cuda` (6 passed, 1 skipped, 118 deselected).
- Next step: migrate the DRA-surface or multiscale-density plotting primitive,
  then revisit legacy `figs/` saves for migrated wrappers.

### 2026-07-07 - Phase 6 2D projection plotting primitives

- Added `dfr.plotting.projections` with transparent colormap construction,
  projection scatter plotting, image-plane density contours, and projected 2D
  GMM density/ellipse rendering.
- Migrated `experiments.dfr_plot.plot_jackdaw2_2d_gmm` to delegate the 2D GMM
  rendering to `dfr.plotting` while preserving its legacy model-loading and
  `figs/` save behavior.
- Migrated `experiments.dfr_plot.plot_jackdaw2_2d_observations` to delegate
  projection scatter and coarse-density rendering to `dfr.plotting` while
  preserving its legacy CUDA image computation and `figs/` save behavior.
- Added headless Matplotlib tests for projection scatter, density contours, and
  projected-GMM ellipse rendering; added static guards for the legacy wrapper
  delegation.
- Updated `experiments/DFR_PLOT_CATALOG.md`, README, and
  `experiments/README.md` to reflect the migrated 2D wrappers.
- Verification: `compileall`; focused plotting/catalog tests; `git diff
  --check`; `pytest -m "not cuda"` (113 passed, 7 deselected, 1 warning);
  `pytest -m cuda` (6 passed, 1 skipped, 113 deselected).
- Next step: migrate an analysis-backed plotting primitive, likely the
  mode-count curve, DRA surface, or multiscale-density panel after separating
  computation/cache reads from rendering.

### 2026-07-07 - Phase 6 trajectory plotting primitive

- Added `dfr.plotting.plot_trajectory_snapshot(...)`, a data-first 3D
  trajectory/final-position renderer with no save/display side effects.
- Migrated `experiments.dfr_plot.plot_single_scenario_new` to delegate
  rendering to `dfr.plotting` while preserving its legacy `figs/scene_traj_*`
  save behavior for compatibility.
- Removed the unused scenario-log directory creation from that legacy plotting
  wrapper.
- Added headless Matplotlib tests for the trajectory primitive and a static
  guard that the legacy wrapper delegates to the package primitive.
- Updated `experiments/DFR_PLOT_CATALOG.md`, README, and
  `experiments/README.md` to reflect the second migrated wrapper.
- Verification: `compileall`; focused plotting/catalog tests; `git diff
  --check`; `pytest -m "not cuda"` (108 passed, 7 deselected, 1 warning);
  `pytest -m cuda` (6 passed, 1 skipped, 108 deselected).
- Next step: migrate the 2D projection/GMM view primitives
  (`plot_jackdaw2_2d_gmm` and/or `plot_jackdaw2_2d_observations`).

### 2026-07-07 - Phase 6 first plotting primitive

- Added the initial `dfr.plotting` package with shared academic Matplotlib
  styling and a data-first `plot_camera_configurations(...)` primitive.
- Migrated `experiments.dfr_plot.plot_camera_configurations` to delegate
  rendering to `dfr.plotting` while preserving its legacy `figs/` PNG/PDF save
  behavior for compatibility.
- Added headless Matplotlib tests for the camera-configuration primitive and a
  static guard that the legacy wrapper delegates to the package primitive.
- Updated `experiments/DFR_PLOT_CATALOG.md`, README, and `experiments/README.md`
  to reflect the first migrated wrapper.
- Verification: `compileall`; focused plotting/catalog tests; `git diff
  --check`; `pytest -m "not cuda"` (104 passed, 7 deselected, 1 warning);
  `pytest -m cuda` (6 passed, 1 skipped, 104 deselected).
- Next step: migrate the next low-risk reusable primitive, likely
  `plot_single_scenario_new` for trajectory/camera layout or the 2D
  projection/GMM view functions.

### 2026-07-07 - Phase 6 dfr_plot catalog freeze

- Added `experiments/DFR_PLOT_CATALOG.md`, cataloging all 36 top-level
  functions in `experiments/dfr_plot.py` with line ranges, known callers,
  inputs/loads, outputs/side effects, classification, and migration notes.
- Added a regression test that parses `dfr_plot.py` and requires every
  top-level function to appear in the catalog.
- Updated README and `experiments/README.md` to point to the catalog.
- Verification: `compileall`; catalog synchronization test; `git diff --check`;
  `pytest -m "not cuda"` (100 passed, 7 deselected, 1 warning);
  `pytest -m cuda` (6 passed, 1 skipped, 100 deselected).
- Next step: create the initial `dfr.plotting` package and migrate one
  low-risk reusable plotting primitive with headless tests.

### 2026-07-07 - Phase 5 runner consolidation completed

- Migrated `experiments/run_scenarios_ue4.py` primary reconstruction from a
  hand-written per-frame `DensityReconstructor` loop to
  `ExternalObservationFrame` plus `reconstruct_observations(...)`.
- Preserved UE4 time-varying `CameraStateUE4` systems for numerical work while
  saving per-frame 7D camera-pose provenance in typed reconstruction results.
- Added explicit UE4 output controls (`--output-root`, `--run-id`, `--resume`,
  `--overwrite`, `--no-output`, `--seed`) and removed scenario-log creation
  from the active primary run.
- Removed the retained `_run_single_scenario_legacy` ordinary body from
  `run_scenarios_angle_sweep.py`; the ordinary path now only dispatches through
  `ScenarioRunSpec`.
- Updated README, specialized-runner inventory, and tests to reflect that
  Phase 5 runner consolidation is complete.
- Verification: `compileall`; `git diff --check`; `pytest -m "not cuda"` (99
  passed, 7 deselected, 1 warning); `pytest -m cuda` (6 passed, 1 skipped, 99
  deselected).
- Next step: start Phase 6 with a catalog of `experiments/dfr_plot.py`
  functions before moving plotting code.

### 2026-07-07 - Phase 5 external observations and flock primary migration

- Added `dfr.reconstruction.observations` with `ExternalObservationFrame`,
  `reconstruct_observations(...)`, an internal observation dataset adapter,
  managed artifact creation, aggregate `statistics.npz`, and shared
  `ReconstructionRun` results.
- Exported the external-observation workflow from `dfr` and
  `dfr.reconstruction`.
- Migrated the primary measured-flock `run` path to assemble calibrated
  detections into `ExternalObservationFrame` objects and call
  `reconstruct_observations`; it no longer writes scenario-log outputs from the
  active reconstruction loop.
- Left flock visualization, baseline, and historical metrics helpers as legacy
  consumers of scenario-log directories; UE4 remains the next external loop to
  migrate.
- Documented the new API in README and updated the specialized-runner
  ownership inventory.
- Verification: `compileall`; `git diff --check`; `pytest -m "not cuda"` (97
  passed, 7 deselected, 1 warning); `pytest -m cuda` (6 passed, 1 skipped, 97
  deselected).
- Next step: move the UE4 thresholded-centroid loop onto
  `ExternalObservationFrame`/`reconstruct_observations`, then remove the
  retained ordinary angle-sweep legacy body.

### 2026-07-06 - Phase 5 specialized runner boundary and dispatch

- Added `RUNNER_SPECIALIZATIONS.md` with observation sources, camera behavior,
  executable commands, current output status, and the exact package boundary
  for angle, flock, and UE4 workflows.
- Moved the ordinary angle-sweep scenario path onto `ScenarioRunSpec`; retained
  the baseline-angle, convergence, voxel, and profiling kernels as explicit
  studies rather than pretending they are ordinary scenario runs.
- Replaced hard-coded default execution in all three modules with required
  subcommands/options, so imports no longer launch CUDA/interactive work.
- Added validated `FlockInputConfig` and required primary flock data,
  calibration, and detection paths; removed its corrupt machine-specific path
  literals from the active run path.
- Made UE4 project root and all three image roots explicit, validated them,
  and removed import-time root log-file creation from flock and UE4.
- Verification: three CLI `--help` commands; `compileall`; `git diff --check`;
  `pytest -m "not cuda"` (92 passed); `pytest -m cuda` (6 passed, 1 skipped:
  optional small rasterizer unavailable).
- Next step: introduce the typed external-observation workflow needed to move
  measured flock or time-varying UE4 reconstruction into `dfr`, then delete the
  characterized legacy ordinary angle body.

### 2026-07-06 - Phase 5 publication table runner consolidation

- Added public `ScenarioRunSpec`, `run_scenario`, and `run_scenarios` services
  for named-dataset loading, frame ranges, ordered ground-truth scale caches,
  camera/training/reconstruction controls, seeded noise, and managed aggregate
  statistics.
- Moved the primary scenario adapter onto the shared service and removed its
  remaining local loading/scale/statistics orchestration.
- Replaced three 869-line Table 2/3/4 copies with side-effect-free seven-line
  compatibility entry points over typed publication profiles. Table 2 keeps
  100 iterations, Table 3 keeps 500, and Table 4 keeps the active starling
  projection-noise preset.
- Added explicit `reconstruct` and reconstruct-plus-evaluate (`run`) actions,
  documented commands, profile override controls, and managed output paths.
- Verification: all three CLI help commands; `compileall`; `git diff --check`;
  `pytest -m "not cuda"` (88 passed); `pytest -m cuda` (6 passed, 1 skipped:
  optional small rasterizer unavailable).
- Next step: inventory and migrate the flock, angle-sweep, and UE4 runner
  specializations, then finish the pretrained-evaluator CLI/output migration.

### 2026-07-06 - Phase 5 shared camera and primary runner migration

- Extended typed reconstruction requests with one explicit scale per frame,
  reproducible bounded projection noise, and the existing decoupled-backend
  switch so scenario sweeps do not need to bypass `dfr.reconstruct`.
- Added package bounded-noise handling and kept one camera ring across all
  selected frames, preserving legacy scenario geometry.
- Refactored `experiments.common` scenario/camera/metric helpers into thin
  compatibility adapters over package services.
- Characterized the old quaternion initialization and verified identical
  auto-aimed camera poses/projections on CPU.
- Migrated the active `run_scenarios.py` reconstruction path to typed package
  configs/results and managed output; removed its 207-line duplicated legacy
  implementation, reducing the script from 855 to 740 lines.
- Added an end-to-end CUDA comparison between direct `process_frame()` and the
  migrated managed workflow; means, radii, and weights agree within `1e-5`.
- Verification: canonical scenario/camera characterization; representative
  direct-vs-workflow CUDA agreement at `1e-5`; `compileall`;
  `git diff --check`; `pytest -m "not cuda"` (83 passed); `pytest -m cuda`
  (6 passed, 1 skipped: optional small rasterizer unavailable).
- Next step: extract one configurable shared runner and migrate the three table
  variants before flock/angle-specific behavior.

### 2026-07-06 - Phase 5 typed evaluation workflow

- Added side-effect-free isotropic-GMM evaluation and batched TP/FP/FN density
  integration under `dfr.evaluation.metrics`, with validated grids and explicit
  CPU/CUDA device selection.
- Added typed `EvaluationSummary`, `FrameEvaluation`, and `EvaluationRun`
  results exposing historical recall, miss, hallucination, and dMOTA equations.
- Added public `dfr.evaluate(...)` for an in-memory `ReconstructionRun` or a
  saved managed reconstruction directory, with optional explicit ground truth,
  bounds/config, and managed evaluation output.
- Kept `dfr.utils` metric functions as compatibility wrappers and migrated
  `compute_metrics_from_pretrained.py` to the package metric/aggregate APIs.
- Added CPU tests for identical densities, derived equations, in-memory and
  saved-run evaluation, explicit output, and validation behavior.
- Documented the evaluation workflow and saved-run usage in README.
- Verification: public API import; representative legacy aggregate check;
  `compileall`; `git diff --check`; `pytest -m "not cuda"` (78 passed);
  `pytest -m cuda` (6 passed, 1 skipped: optional small rasterizer unavailable).
- Next step: migrate shared camera setup and one representative scenario loop,
  then finish the legacy pretrained-evaluator CLI/output migration.

### 2026-07-06 - Phase 4 supported analysis entry points completed

- Added `experiments/README.md` as the detailed analysis command catalog with
  support classification, inputs, outputs, runtime expectations, examples,
  legacy caveats, and ownership boundaries for every analysis-oriented module.
- Migrated the 3PL/2PL manifold, mechanistic derivation, synthetic benchmark,
  and mode-count validation commands to managed analysis runs; supported CLIs
  no longer contain direct `figs/`, `results/`, or current-working-directory
  output assumptions.
- Added shared managed-analysis parser/artifact helpers under `dfr.analysis`
  instead of repeating output/run collision plumbing in every experiment.
- Extracted symmetric-2PL fitting into `dfr.analysis.manifold` with a typed
  result and golden recovery test; the experiment now owns presentation rather
  than reusable curve fitting.
- Made project roots, seeds, run IDs, resume/overwrite policies, trial counts,
  and agent counts explicit where applicable.
- Replaced hard-coded default execution in `power_law.py` and
  `reconstruction_scale_determination.py` with required experiment dispatch;
  disabled `dfr_plot.py`'s implicit animation pending Phase 6.
- Added contract tests that supported CLIs stay off legacy output roots, remain
  cataloged, and preserve explicit dispatch for legacy studies.
- Verification: all eight supported/legacy analysis `--help` commands;
  supported-output static contract; `compileall`; `git diff --check`;
  `pytest -m "not cuda"` (73 passed); `pytest -m cuda` (5 passed, 1 skipped:
  optional small rasterizer unavailable).
- Next step: resume Phase 5 with typed evaluation metrics and
  `dfr.evaluate(...)`.

### 2026-07-06 - Phase 5 typed reconstruction workflow

- Added validated `ReconstructionRequest`, `FrameReconstruction`, and
  `ReconstructionRun` contracts; frame results expose detached positions, GMM
  arrays, camera poses/projections, visibility, scale, timings, and summary.
- Added explicit/encircling camera-system construction under
  `dfr.reconstruction`, including the established adjacent two-camera layout.
- Added public `dfr.reconstruct(...)` with one or many frames, fixed/adaptive
  scale, typed training/reconstruction configs, device, seed, optional managed
  output, and scenario-config discovery from loaded dataset metadata.
- Preserved `DensityReconstructor.process_frame()` unchanged as the numerical
  compatibility layer and preserved single-frame artifact filenames.
- Reduced `reconstruct_one_frame.py` from 297 to 117 lines by making it a thin
  public-workflow CLI.
- Added CPU request/result/camera tests and passed the real one-iteration CUDA
  CLI workflow with managed artifacts.
- Verification: public import; reconstruction CLI help; `compileall`;
  `git diff --check`; `pytest -m "not cuda"` (68 passed); `pytest -m cuda`
  (5 passed, 1 skipped: optional small rasterizer unavailable).
- Next step: typed evaluation metrics and `dfr.evaluate(...)`, followed by one
  representative legacy scenario-runner migration.

### 2026-07-06 - Phase 4 parameter manifold and analysis facade

- Added side-effect-free centered-3PL evaluation, per-frame batch fitting,
  legacy three-file cache loading, median-NND calculation, inverse target-mode
  scale selection, Hill shape fitting, and vectorized shape projection under
  `dfr.analysis.manifold`.
- Refactored `parameter_manifold.py` to import the shared numerical functions;
  retained compatibility aliases for research scripts during migration.
- Added public `dfr.analyze(...)` dispatch for one-frame mode curves and
  CUDA DRA surfaces using `AnalysisConfig`, with no implicit result saving.
- Added a golden fit based on the historic jackdaw cache schema, cache/result
  round trips, inverse-scale and projection checks, facade validation, and
  explicit DRA dispatch tests.
- Documented facade semantics, lower-level manifold APIs, cache schema, script
  inputs, runtime expectations, and remaining historic output behavior.
- Verification: parameter-manifold CLI help; public API import; `compileall`;
  `git diff --check`; `pytest -m "not cuda"` (61 passed); `pytest -m cuda`
  (5 passed, 1 skipped: optional small rasterizer unavailable).
- Next step: migrate remaining analysis outputs/thin wrappers, or start typed
  reconstruction workflows in Phase 5.

### 2026-07-06 - Phase 4 mode and DRA analysis extraction

- Added `dfr.analysis` with data-only `ModeCurveResult`,
  `ScaleAnalysisResult`, and `ManifoldAnalysisResult`, including validated NPZ
  save/load and legacy DRA cache context handling.
- Added direct `count_modes`, `compute_mode_curve`, and
  `analyze_dataset_modes` APIs so mode analysis no longer requires an
  experiment import or manual Torch setup.
- Extracted nearest-neighbor scaling, model-order grids, batched CUDA DRA,
  resumable surface filling, design matrices, surface fits, row/column CV,
  frame sampling, frame/dataset CV, and pooled multiframe fitting.
- Refactored both DRA scripts to use package analysis; multiframe no longer
  imports the single-frame experiment. Both retain managed artifacts and legacy
  cache seeding.
- Reduced `plot_dra_scale_model_order.py` from 688 to 361 lines and removed
  duplicated fit/cross-validation implementations from the multiframe script.
- Fixed frame sampling for requested counts of one or two.
- Added deterministic result round trips, legacy cache validation, analytical
  golden coefficients, multiframe CV, direct mode-curve tests, and an extracted
  CUDA DRA identity test.
- Documented direct mode and DRA analysis APIs in README.
- Verification: both DRA CLI `--help` commands; no multiframe cross-experiment
  import; `compileall`; `git diff --check`; `pytest -m "not cuda"` (51 passed);
  `pytest -m cuda` (5 passed, 1 skipped: optional small rasterizer unavailable).
- Next step: parameter-manifold computation/cache extraction and the first
  general `dfr.analyze(...)` facade.

### 2026-07-06 - Phase 3 shared workflow configuration

- Added validated `CameraConfig` for encircling layouts or explicit
  `[x, y, z, qx, qy, qz, qw]` poses, plus `AnalysisConfig`,
  `EvaluationConfig`, and schema-versioned nested `RunConfig`.
- Added `DatasetSpec` and `OutputConfig` dict restoration; extended artifact
  serialization to prefer each config's stable `to_dict` contract.
- Preserved the historic `targetd_num_mode` field while exposing a correctly
  named `target_mode_count` compatibility property.
- Migrated the one-frame runner to `CameraConfig`/`RunConfig` and the DRA runner
  to `AnalysisConfig` without changing their CLIs.
- Added complete nested YAML round-trip tests, explicit-pose normalization,
  unknown-schema rejection, and validation tests for camera, analysis,
  evaluation, and run settings.
- Documented the typed configuration API in README.
- Verification: public config API import; both managed CLI `--help` commands;
  `compileall`; `git diff --check`; `pytest -m "not cuda"` (38 passed);
  `pytest -m cuda` (4 passed, 1 skipped: optional small rasterizer unavailable).
- Next step: Phase 4 `dfr.analysis` result types and DRA computation extraction.

### 2026-07-06 - Phase 3 artifact foundation and first migrations

- Added `OutputConfig` and `RunArtifacts` with project-root-relative output
  resolution, safe generated/explicit run IDs, canonical subdirectories, and
  path-traversal protection.
- Added schema-versioned `manifest.json` and resolved `config.yaml` with UTC
  timestamps, local Git commit, package version, device, metadata, and resume
  count.
- Added explicit collision/resume/overwrite behavior plus centralized atomic
  JSON, NPZ, checkpoint, and figure writers.
- Added serialization for dataclasses, paths, datetime/enum values, NumPy
  values/arrays, Torch devices/tensors, mappings, and sequences.
- Added `warn_legacy_output` for `figs/`, `results/`, and scenario log writes.
- Migrated `plot_dra_scale_model_order.py` to managed analysis runs: caches in
  `cache/`, fit arrays in `data/`, summaries in `metrics/`, and plots in
  `figures/`; it no longer requires the repository cwd.
- Added `experiments/reconstruct_one_frame.py`, a transitional explicit CLI for
  dataset/frame/camera/scale/iterations/voxel/seed settings. It saves final
  arrays, checkpoint, metrics, manifest, and config under one run.
- Removed an unnecessary tensor copy in voxel carving found by the new CUDA
  runner test.
- Verification: both migrated CLI `--help` commands; `compileall`;
  `git diff --check`; `pytest -m "not cuda"` (28 passed); `pytest -m cuda`
  (4 passed, 1 skipped: optional small rasterizer unavailable).
- Next step: finish Phase 3 with shared workflow config types and nested config
  round trips, then begin Phase 4 analysis extraction.

### 2026-07-06 - Phase 2 canonical dataset API

- Added the `dfr.data` package with a documented `Dataset` protocol,
  `DatasetSpec`, `ScenarioRegistry`, `resolve_dataset`, `load_dataset`, and
  shared frame selection.
- Published common loading symbols from `dfr/__init__.py`; `DatasetFactory`
  remains available for compatibility.
- Added length, validated frame access, optional velocity/timestamp state,
  source/coordinate metadata, and ground-truth position access to existing
  dataset objects.
- Added distinct errors for missing paths, unsupported extensions, malformed
  supported files, invalid frames, empty selections, and missing velocities.
- Documented and tested all loader schemas: NPY, both NPZ layouts, MATLAB,
  two-frame RTF, timestamp-grouped HDF5, and drone CSV.
- Migrated `experiments/common.py`, `plot_dra_scale_model_order.py`, and
  `fit_dra_multiframe.py` to the canonical loader without moving legacy loader
  implementations yet.
- Verified named scenario `boids` loads correctly while the process cwd is
  outside the repository.
- Verification: public API import; both migrated CLI `--help` commands;
  `compileall`; `git diff --check`; `pytest -m "not cuda"` (22 passed);
  `pytest -m cuda` (3 passed, 1 skipped: optional small rasterizer unavailable).
- Next step: Phase 3 output configuration, run artifacts, manifests, and the
  first analysis/reconstruction migrations.

### 2026-07-06 - Phase 1 project contract and CPU safety net

- Replaced the unfinished README with verified installation, loading, analysis,
  reconstruction/evaluation, output, testing, troubleshooting, and complete
  experiment-script documentation.
- Added `CONTRIBUTING.md`; expanded package metadata/dependencies and pytest
  configuration in `pyproject.toml`; synchronized `environment.txt`.
- Moved tests to `tests/`, isolated discovery from interactive experiment files,
  and added a small JSON trajectory fixture.
- Added 12 passing CPU characterization tests for loaders/frame filtering,
  velocity strategies, camera projection/culling, projection-only rendering,
  mode count, scale search, density metrics, typed configs, and checkpoint I/O.
- Replaced the oversized CUDA rasterizer tests with explicitly marked 64x64
  smoke tests and added a two-camera, one-iteration end-to-end reconstruction
  smoke test. The large rasterizer is installed; only the optional small
  rasterizer test skips because that variant is not installed.
- Fixed `compute_metrics_batched_torch` to use its requested device and return
  its accumulated Python floats without invalid `.item()` calls.
- Verification: metadata parse; `compileall` for `dfr`, `experiments`, and
  `tests`; `git diff --check`; `pytest -m "not cuda"` (12 passed);
  `pytest -m cuda` (3 passed, 1 skipped: optional small rasterizer unavailable).
- Next step: Phase 2 dataset protocol, registry/resolver, and public loading
  facade.

### 2026-07-06 - Baseline and architecture plan

- Preserved the current working tree in `7cde21e` and tagged it `v0.1.0`.
- Audited documentation, package/experiment size, duplicate function names, and
  generated-output paths.
- Created this roadmap; no refactor implementation has started.
- Confirmed that version maintenance is local-only; no GitHub push is pending.
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
