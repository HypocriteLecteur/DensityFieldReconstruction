# DFR Refactor Roadmap and Handoff

This file is the single source of truth for the refactor. Keep it current in
every refactor commit so another developer or agent can resume without relying
on chat history.

## Status

- **Current phase:** Phase 4 in progress. Mode, DRA, parameter-manifold, typed
  result, and first `dfr.analyze` APIs are complete; remaining work is thinning
  the long-tail analysis scripts and migrating their legacy output policies.
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
- [ ] Reduce all analysis scripts to CLI/config wrappers and document each script's
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

The next agent should finish Phase 4 or begin Phase 5:

1. Migrate the remaining active analysis scripts away from direct `figs/`,
   `results/`, and scenario-cache writes, prioritizing
   `parameter_manifold.py`, `parameter_manifold_2pl.py`, and
   `validate_mode_counting.py`.
2. Split parameter-manifold cache generation from the publication CLI so the
   experiment becomes a thin configuration/presentation wrapper.
3. Decide whether the long-tail exploratory scripts are active reproducibility
   entry points or historical code before spending time converting each one.
4. If proceeding to Phase 5, define typed reconstruction request/result objects
   and build `dfr.reconstruct(...)` on the already migrated one-frame runner.
5. Keep all CPU tests and the five available CUDA tests passing.

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

## Handoff Log

Add one newest-first entry per working session. Include commit(s), verification,
known failures, and the exact next step.

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
