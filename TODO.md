# DFR Refactor Roadmap and Handoff

This file is the single source of truth for the refactor. Keep it current in
every refactor commit so another developer or agent can resume without relying
on chat history.

## Status

- **Current phase:** Phase 8 compatibility cleanup is in progress. The
  compatibility inventory and archive policy now document active/archive
  boundaries and deletion rules before any removal. Phase 7 documented the
  intended top-level API, high-traffic package
  surfaces, and the main public class/function contracts for data loading,
  configuration, artifacts, reconstruction, evaluation, plotting, camera
  systems, external observations, scenario runners, analysis helpers, and model
  checkpointing. The docs layer includes a checked CPU toy workflow, workflow
  examples, command-verification notes, and a module ownership map. Phase
  6 plotting decomposition is complete: the
  `experiments/dfr_plot.py` function catalog is frozen in
  `experiments/DFR_PLOT_CATALOG.md`; reusable camera-configuration,
  trajectory-snapshot, 2D projection/GMM, mode-count curve, DRA
  scale/model-order surface, multiscale density, 3D density/GMM rendering,
  shared style/save/layout primitives, typed analysis-result plotting paths,
  typed frame-reconstruction GMM plotting, and typed evaluation-summary
  and metric-series plotting have moved to `dfr.plotting`. The first
  publication/table split-outs, `experiments.plot_publication_table2`,
  `experiments.plot_publication_time_efficiency`, and
  `experiments.plot_publication_noise_robustness`, now own hard-coded
  publication figures formerly embedded in `experiments/dfr_plot.py`. The
  remaining public `dfr_plot.py` functions have a documented support policy:
  delegated wrappers are compatibility-supported, while all other public
  functions are archive-only historical references until re-owned. Supported
  compatibility wrappers no longer create `figs/` outputs unless an explicit
  output directory is supplied; physical deletion of `dfr_plot.py` is deferred
  to Phase 8 compatibility cleanup.
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
- [x] Make plotting functions accept result objects/axes and return Figure/Axes;
  saving is optional and uses the output manager.
  - Complete for reusable package plotting primitives: mode-count and DRA
    scale/model-order plotting now accept
    `ModeCurveResult` and `ScaleAnalysisResult` directly while preserving the
    existing array/legacy tuple APIs and Figure/Axes return contracts.
    `FrameReconstruction` now has a direct 3D reconstructed-GMM plot path; raw
    density-grid plots still require explicit density/tick arrays because the
    reconstruction result object does not store a reusable voxel density grid.
    `EvaluationRun`, `FrameEvaluation`, and `EvaluationSummary` now have a
    direct summary-metric bar plot path, and `EvaluationRun`/frame sequences
    have a per-frame metric-series plot path. Grid/multiscale helpers return
    figure/axes collections rather than a single axes object.
    `experiments.dfr_plot.plot_camera_configurations` now follows the
    no-implicit-save rule too: it returns Figure/Axes by default and saves only
    when an explicit `output_dir` is supplied. The trajectory compatibility
    wrapper `plot_single_scenario_new` follows the same explicit-save rule, as
    do the Jackdaw2 2D GMM/projection, mode-count, multiscale-density, and DRA
    surface wrappers.
- [x] Split publication/table-specific figures into small, named experiment
  scripts rather than one replacement monolith.
  - Started with `experiments.plot_publication_time_efficiency`, which owns the
    hard-coded training-time scaling figure and provides an explicit CLI.
    `plot_table_time_efficiency` remains as a compatibility wrapper that
    delegates to the named script and saves only when an explicit `save_dir` is
    supplied.
    Continued with `experiments.plot_publication_table2`, which owns the
    hard-coded Table 2 capacity-scaling and recall/hallucination tradeoff
    figures while `plot_table_2_results` delegates. Completed with
    `experiments.plot_publication_noise_robustness`, which owns the hard-coded
    noise robustness publication figure while `plot_table_noise_robustness`
    delegates. The three publication compatibility wrappers now all save only
    when an explicit `save_dir` is supplied.
- [x] Add headless smoke tests for representative 2D, 3D, camera, scale, and
  evaluation plots.
  - Complete: camera layout, 2D projection/density/GMM, 3D trajectory,
    density/multiscale/reconstruction-GMM, scale/mode/DRA, evaluation summary,
    evaluation metric series, style/layout, legacy-wrapper delegation, and the
    smoke-test matrix are covered under the Agg backend.
- [x] Remove `experiments/dfr_plot.py` only after callers and documented
  commands migrate.
  - Resolved for Phase 6 by removing it from active command/API status rather
    than deleting it immediately: direct execution remains disabled, all
    supported wrappers are documented compatibility shims with explicit-save
    behavior, archive-only functions are not supported callers, and physical
    deletion is deferred to Phase 8 after compatibility wrappers are removed.

**Exit criteria:** no active plotting module is a grab bag, reusable plots are
importable, and every remaining experiment figure has a documented command.

### Phase 7 - Core Documentation and Public API

- [x] Add module docstrings and complete public docstrings for data, camera,
  model, reconstruction, scale, mode, evaluation, plotting, and artifact APIs.
  - Started with expanded module-level documentation for `dfr`, `dfr.data`,
    `dfr.analysis`, `dfr.reconstruction`, `dfr.evaluation`, `dfr.plotting`,
    `dfr.config`, `dfr.artifacts`, and `dfr.workflows`.
  - Continued with expanded public class/function docstrings for
    `dfr.data.base`, `dfr.data.loading`, `dfr.data.registry`, `dfr.data.spec`,
    `dfr.config`, `dfr.artifacts`, `dfr.reconstruction.pipeline`,
    `dfr.reconstruction.results`, `dfr.evaluation.pipeline`,
    `dfr.evaluation.results`, and core plotting primitives.
  - Continued into lower-level public APIs: `dfr.reconstruction.cameras`,
    `dfr.reconstruction.observations`, `dfr.reconstruction.scenarios`,
    `dfr.analysis.cli`, `dfr.analysis.dra`, `dfr.analysis.manifold`,
    `dfr.analysis.modes`, `dfr.analysis.results`, `dfr.analysis.scales`, and
    `dfr.model_checkpoint`.
- [x] Document units, coordinate systems, shapes/dtypes, device behavior,
  randomness, side effects, exceptions, and return contracts where relevant.
  - Started at the package boundary: top-level docs now define frame indices,
    world-coordinate position shapes, scale units, CUDA side effects, and
    no-implicit-artifact behavior.
  - Continued across public functions/classes: documented dataset array shapes,
    explicit camera pose order, managed output semantics, reconstruction CPU
    result arrays, voxel-evaluation bounds/resolution, plotting return values,
    and no-save behavior.
  - Continued into lower-level APIs: documented camera-system construction,
    image-plane projection/noise shapes, external-observation contracts,
    scenario frame-selection and scale-cache semantics, normalized DRA scale
    grids, CUDA/resume behavior, manifold-fit inputs/outputs, and checkpoint
    restore conventions.
- [x] Export only the intended common API from `dfr/__init__.py`; keep advanced
  APIs accessible from their submodules.
  - Started: `tests/test_public_api.py` guards the current intended
    top-level `dfr.__all__` surface and package contract docstring.
  - Added a high-traffic docstring contract guard for `load_dataset`,
    `resolve_dataset`, `reconstruct`, `evaluate`, and representative plotting
    APIs.
  - Added a lower-level public docstring guard for representative
    reconstruction, scenario, analysis, and manifold APIs.
- [x] Add runnable examples and API reference generation or a lightweight
  `docs/` tree.
- [x] Add a script catalog table to the README and a module ownership map to
  developer docs.
- [x] Test README/example commands in a clean environment where practical.
  - Added `examples/toy_workflow.py` and `tests/test_examples.py`, which run a
    CPU-safe load/analyze/plot example against a generated toy dataset.
  - Reviewed safe command snippets on 2026-07-10: ran the toy workflow script,
    supported `--help`/explicit-dispatch commands, and documented the remaining
    CUDA/data-dependent commands in `docs/COMMAND_VERIFICATION.md`.

**Exit criteria:** common tasks are discoverable from README/API docs without
reading implementation or experiment source.

### Phase 8 - Cleanup and Release

- [ ] Remove compatibility wrappers only after all active callers migrate.
  - Started with `docs/PHASE8_COMPATIBILITY_INVENTORY.md` and static tests:
    no active Python module imports `experiments.dfr_plot`, and
    `experiments.plotting_utils` is imported only by `experiments.dfr_plot`.
- [ ] Remove confirmed-dead copies and duplicated functions; use Git history
  rather than keeping backup files.
  - Completed for `density_field_reconstruction_copy/` and
    `experiments_legacy/`: they were ignored local-only backup copies, so a
    verified local ZIP was created before their approved deletion. Remaining
    duplicated-function cleanup is tracked by the `dfr_plot.py` archive
    surface and later Phase 8 inventory slices.
- [ ] Decide whether historical scripts belong in an archive branch, paper
  reproduction directory, or should remain at the stable tag only.
  - Inventory recommends deciding archive policy before deleting `dfr_plot.py`,
    `plotting_utils.py`, copied directories, or generated `figs/`/`results/`
    content.
  - The documented policy keeps local Git history and `v0.1.0` as the default
    archive, requires explicit owner direction for a separate archive branch
    or external generated-artifact backup, and requires one removal surface
    per commit.
- [ ] Confirm `figs/` and `results/` are no longer written by active code, then
  archive or remove them in a separately reviewed change.
  - Inventory separates supported managed-output paths from legacy/archive
    producers: `dfr_plot.py`, explicit-dispatch studies, legacy cache seeding,
    and scenario-log consumers remain to be retired or archived.
  - The policy treats `figs/` and `results/` as generated/historical artifacts
    and requires their removal in a separate commit after active producers are
    retired or redirected to `outputs/`.
- [ ] Run CPU tests, CUDA tests, representative analyses, and end-to-end
  reconstruction/evaluation.
- [ ] Review docs from a clean clone and verify output paths are reproducible.
- [ ] Tag the completed refactor as the next semantic version and write release
  notes with migration examples from `v0.1.0`.

## Immediate Next Actions

The next agent should continue Phase 8 compatibility cleanup:

1. Replace the `experiments.dfr_plot --list-functions` catalog path, then
   remove `experiments/dfr_plot.py` and `experiments/plotting_utils.py` as one
   compatibility/archive surface if the resulting documentation boundary is
   adequate.
2. Update `experiments/DFR_PLOT_CATALOG.md`, `experiments/README.md`,
   `docs/MODULE_OWNERSHIP.md`, and tests in the same commit as any removal.
3. Run focused catalog/inventory tests plus full CPU/CUDA tiers after every
   deletion or compatibility-boundary change.

## Decisions

- **2026-07-06 - Stable marker:** use annotated tag `v0.1.0` at `7cde21e`.
  Reason: package metadata already declares version 0.1.0 and no prior tags
  exist.
- **2026-07-10 - Phase 8 archive policy:** use local Git history and the
  `v0.1.0` tag as the default preservation mechanism for duplicate source,
  archive-only plotting code, and generated historical artifacts; keep an
  archive branch or external backup only with explicit owner direction.
  Reason: `main` should contain active, maintainable code rather than source
  copies or generated output, while the local stable tag remains a recoverable
  scientific reference.
- **2026-07-10 - Ignored copied-source exception:**
  `density_field_reconstruction_copy/` and `experiments_legacy/` were never
  Git-tracked, so preserve them in a verified local ZIP before deletion rather
  than relying on `v0.1.0`. Reason: the source ZIP for the stable tag does not
  include ignored local-only paths.
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
- **2026-07-08 - Plot result-object boundary:** Phase 6 step 4 is complete for
  reusable package plotting primitives that have stable typed package results.
  Raw density-grid and multiscale plots remain array/data-dict based because no
  current result object stores reusable voxel density grids; grid-style helpers
  return figure/axes collections. Reason: inventing pseudo-result wrappers for
  transient plotting arrays would add ceremony without improving workflow
  clarity.
- **2026-07-08 - `dfr_plot.py` compatibility boundary:** Only delegated
  wrappers documented under "Supported compatibility wrappers" in
  `experiments/DFR_PLOT_CATALOG.md` are compatibility-supported. All other
  public `dfr_plot.py` functions are archive-only historical references until
  someone re-owns them as a named CLI or package API with tests. Reason:
  preserving every old plotting function as active API would keep the grab-bag
  module alive indefinitely.
- **2026-07-08 - Camera-configuration wrapper saving:** Keep
  `experiments.dfr_plot.plot_camera_configurations` as a compatibility wrapper,
  but require an explicit `output_dir` for file writes. Reason: the reusable
  package primitive already has the correct no-save default, and preserving an
  implicit `figs/` write would conflict with the refactor output contract.
- **2026-07-08 - Publication wrapper saving:** Keep
  `plot_table_2_results`, `plot_table_time_efficiency`, and
  `plot_table_noise_robustness` as compatibility wrappers, but require an
  explicit `save_dir` for file writes and `show=True` for interactive display.
  Reason: the named `experiments.plot_publication_*` CLIs now own publication
  exports, so the compatibility wrappers should not silently create `figs/`.
- **2026-07-08 - Trajectory wrapper saving:** Keep
  `experiments.dfr_plot.plot_single_scenario_new` as a compatibility wrapper,
  but require an explicit `output_dir` for file writes. Reason: rendering
  already delegates to `dfr.plotting.plot_trajectory_snapshot`, so the wrapper
  should share the reusable plotting API's no-save default.
- **2026-07-09 - Jackdaw2 2D wrapper saving:** Keep
  `plot_jackdaw2_2d_gmm` and `plot_jackdaw2_2d_observations` as compatibility
  wrappers, but require an explicit `output_dir` for file writes. Reason: the
  reusable 2D projection/GMM rendering already lives in `dfr.plotting`, and
  the remaining legacy wrapper work is computation/loading rather than output
  path ownership.
- **2026-07-09 - Final supported wrapper saving:** Keep
  `plot_jackdaw2_mode_count_curve`, `plot_jackdaw2_multiscale_density`, and
  `plot_jackdaw2_dra_scale_model_order_surface` as compatibility wrappers, but
  require an explicit `output_dir` for figure writes. Reason: their reusable
  rendering lives in `dfr.plotting`; the remaining legacy work is cache/CUDA
  computation, which should not imply hidden `figs/` output.
- **2026-07-09 - Phase 6 completion boundary:** Do not physically delete
  `experiments/dfr_plot.py` during Phase 6. Treat it as a disabled direct CLI,
  documented compatibility/archive module, and defer physical deletion to
  Phase 8 after compatibility wrappers are removed. Reason: deleting it now
  would erase reviewed historical functions and break explicit compatibility
  imports, while leaving it as an active grab-bag has already been prevented by
  policy, tests, and documentation.

## Handoff Log

Add one newest-first entry per working session. Include commit(s), verification,
known failures, and the exact next step.

### 2026-07-10 - Phase 8 ignored backup-copy cleanup

- Confirmed `density_field_reconstruction_copy/` and `experiments_legacy/`
  are ignored, have no Git history, and are not referenced by active package,
  experiment, or example Python code.
- Created and verified the dedicated local archive
  `outputs/releases/DensityFieldReconstruction-phase8-legacy-copies-20260710.zip`
  and removed both backup directories; SHA-256:
  `B49E0856B468BB23C4A9D88948301E119AB8F137A724E8AAD8732E25611B84A9`.
- Added a guard against active source references and a post-cleanup absence
  test for the two copied directories.
- Verification: focused Phase 8 inventory/docs tests passed with 10 tests;
  `compileall dfr experiments tests examples` passed; the local archive contains
  41 copied-tree entries and both source directories are absent; `git diff
  --check` passed with Windows line-ending warnings only; `pytest -m "not
  cuda"` passed with 179 tests and 7 deselected; `pytest -m cuda` passed with
  6 tests, 1 skipped rasterizer-extension test, and 179 deselected.
- Next step: replace the `dfr_plot --list-functions` catalog path before
  removing the remaining plotting archive surface.

### 2026-07-10 - Phase 8 archive policy documented

- Added `docs/PHASE8_ARCHIVE_POLICY.md`, defining local Git history plus
  `v0.1.0` as the standard archive, deletion boundaries for compatibility
  plotting code, copied source directories, generated artifacts, and scenario
  logs, and required one-surface-per-commit verification rules.
- Linked the policy from `docs/README.md` and the Phase 8 inventory, and added
  static coverage that the policy retains its essential cleanup rules.
- Updated Phase 8 status, decisions, and next actions: removal of the two
  copied backup trees is the next low-risk deletion, but requires explicit
  approval before destructive filesystem work.
- Verification: focused Phase 8 inventory/docs tests passed with 8 tests;
  `compileall dfr experiments tests examples` passed; `git diff --check`
  passed with Windows line-ending warnings only; `pytest -m "not cuda"`
  passed with 177 tests and 7 deselected; `pytest -m cuda` passed with 6
  tests, 1 skipped rasterizer-extension test, and 177 deselected.
- Next step: after verification and commit, request approval for the
  deletion-only backup-copy cleanup slice.

### 2026-07-10 - Phase 8 compatibility inventory started

- Added `docs/PHASE8_COMPATIBILITY_INVENTORY.md`, documenting the first cleanup
  inventory for `experiments.dfr_plot`, `experiments.plotting_utils`, legacy
  `figs/`, `results/`, `scenarios/*/logs/` producers, and copied backup
  directories.
- Added `tests/test_phase8_inventory.py` with static guards that no active
  Python module imports `experiments.dfr_plot` and that
  `experiments.plotting_utils` remains isolated to `experiments.dfr_plot`.
- Linked the Phase 8 inventory from `docs/README.md`.
- Updated the Phase 8 checklist and Immediate Next Actions to continue with an
  archive/deletion policy decision before removing wrappers or copied trees.
- Verification: focused inventory/docs/catalog/entrypoint tests passed with 25
  tests; `compileall dfr experiments tests examples` passed; `git diff
  --check` passed with Windows line-ending warnings only; `pytest -m "not
  cuda"` passed with 176 tests and 7 deselected; `pytest -m cuda` passed with
  6 tests, 1 skipped rasterizer-extension test, and 176 deselected.
- Next step: decide the archive policy for `dfr_plot.py`, `plotting_utils.py`,
  backup copy directories, and generated legacy outputs; then remove one
  low-risk surface at a time with tests.

### 2026-07-10 - Phase 7 completed

- Completed the remaining docs-command review: ran the checked CPU toy workflow
  script plus low-risk `--help`/explicit-dispatch commands for supported
  analysis CLIs, reconstruction CLI, publication table/figure commands,
  legacy explicit-dispatch studies, `dfr_plot --list-functions`, and
  specialized angle/flock/UE4 runner help paths.
- Added `docs/COMMAND_VERIFICATION.md` and linked it from README/docs so the
  repository records which commands were actually executed and which examples
  remain intentionally CUDA/data/asset-dependent.
- Updated `tests/test_docs.py` to guard the command-verification notes.
- Marked all Phase 7 tasks complete and moved Immediate Next Actions to Phase
  8 compatibility cleanup.
- Verification: focused docs/examples/public/analysis tests passed with 19
  tests; `compileall dfr experiments tests examples` passed; `git diff
  --check` passed with Windows line-ending warnings only; `pytest -m "not
  cuda"` passed with 173 tests and 7 deselected; `pytest -m cuda` passed with
  6 tests, 1 skipped rasterizer-extension test, and 173 deselected.
- Next step: begin Phase 8 with an inventory of active callers/imports for
  compatibility wrappers and legacy output producers before deleting anything.

### 2026-07-09 - Phase 7 workflow docs and ownership map added

- Added `docs/README.md`, `docs/WORKFLOW.md`, and
  `docs/MODULE_OWNERSHIP.md`.
- Added `examples/toy_workflow.py`, a CPU-safe runnable example that creates a
  tiny dataset, loads it through `dfr.load_dataset`, computes a mode-count
  curve on CPU, and saves a figure explicitly through `dfr.plotting`.
- Linked the new docs from `README.md`; the root README already contains the
  script catalog table, and the new ownership map documents how package APIs,
  supported experiment wrappers, legacy scripts, and output roots relate.
- Added `tests/test_docs.py` and `tests/test_examples.py` to guard README doc
  links, workflow/output-policy coverage, ownership-map coverage, and the
  runnable toy example.
- Marked the Phase 7 lightweight docs/examples task and script-catalog/module
  ownership task complete. The remaining Phase 7 documentation task is to
  review/verify README/example command snippets where practical.
- Verification: focused docs/examples/public/analysis-entrypoint tests passed
  with 18 tests; `compileall dfr experiments tests examples` passed;
  `git diff --check` passed with Windows line-ending warnings only;
  `pytest -m "not cuda"` passed with 172 tests and 7 deselected;
  `pytest -m cuda` passed with 6 tests, 1 skipped rasterizer-extension test,
  and 172 deselected.
- Next step: review remaining README/example command snippets for copy/paste
  accuracy and either test or explicitly mark CUDA/data-dependent commands.
  If satisfactory, mark Phase 7 complete and begin Phase 8 compatibility
  cleanup.

### 2026-07-09 - Phase 7 lower-level public docstrings expanded

- Expanded public documentation for lower-level reconstruction and analysis
  APIs: camera-system construction, bounded projection noise, external
  observation reconstruction, scenario runners, analysis CLI helpers, mode
  counting, scale selection, DRA surfaces/fits, manifold fitting, typed
  analysis results, and Gaussian model checkpoint helpers.
- Documented key contracts at those call sites: world-coordinate positions,
  image-plane projection shapes, explicit camera pose order, scenario
  frame-selection semantics, ground-truth scale-cache ordering, normalized NND
  scale grids, CUDA requirements, resumable DRA rows, fitted-surface return
  dictionaries, manifold-fit diagnostic arrays, and checkpoint restore
  conventions.
- Extended `tests/test_public_api.py` with a lower-level docstring contract
  guard covering representative reconstruction, scenario, analysis, DRA, and
  manifold APIs.
- Verification: focused lower-level public/analysis/reconstruction tests
  passed with 63 tests; `compileall dfr experiments tests` passed;
  `git diff --check` passed with Windows line-ending warnings only;
  `pytest -m "not cuda"` passed with 168 tests and 7 deselected;
  `pytest -m cuda` passed with 6 tests, 1 skipped rasterizer-extension test,
  and 168 deselected.
- Known limitation: runnable examples/lightweight docs, the README script
  catalog table, and module ownership map still need Phase 7 coverage.
- Next step: add lightweight docs/examples for load -> analyze -> reconstruct
  -> evaluate -> plot, then update README/developer docs with script and
  ownership maps.

### 2026-07-09 - Phase 7 high-traffic public docstrings expanded

- Expanded public class/function docstrings for the highest-traffic Phase 7
  APIs: dataset protocol/loading/registry/specs, config dataclasses, managed
  artifacts, reconstruction/evaluation pipelines and typed results, and core
  plotting primitives.
- Documented key contracts directly at call sites: world-coordinate data
  shapes, frame-index semantics, explicit camera pose order, fixed/adaptive
  reconstruction scales, CUDA requirements, CPU result arrays,
  voxel-evaluation bounds/resolution, managed run resume/overwrite behavior,
  plotting return values, and no-save/no-output side effects.
- Extended `tests/test_public_api.py` with a high-traffic docstring contract
  guard covering `load_dataset`, `resolve_dataset`, `reconstruct`, `evaluate`,
  `plot_density_field_3d`, and `plot_projected_gmm_density`.
- Verification: focused public/data/config/artifact/reconstruction/evaluation/
  plotting tests passed with 82 tests; `compileall dfr experiments tests`
  passed; `git diff --check` passed with Windows line-ending warnings only;
  `pytest -m "not cuda"` passed with 167 tests and 7 deselected;
  `pytest -m cuda` passed with 6 tests, 1 skipped rasterizer-extension test,
  and 167 deselected.
- Known limitation: lower-level camera/model/analysis modules and runnable docs
  examples still need Phase 7 coverage.
- Next step: continue Phase 7 with remaining lower-level public docstrings and
  add a lightweight `docs/` tree or examples for load -> analyze -> reconstruct
  -> evaluate -> plot.

### 2026-07-09 - Phase 7 public API documentation started

- Expanded the top-level `dfr` package docstring to document the intended
  load -> analyze -> reconstruct -> evaluate API shape, dataset position
  shapes, world-coordinate units, scale conventions, no-implicit-artifact
  behavior, and CUDA expectations.
- Expanded module docstrings for `dfr.data`, `dfr.analysis`,
  `dfr.reconstruction`, `dfr.evaluation`, `dfr.plotting`, `dfr.config`,
  `dfr.artifacts`, and `dfr.workflows`.
- Expanded `dfr.workflows.analyze` with parameter/return documentation for the
  common analysis facade.
- Added `tests/test_public_api.py` to guard the intended top-level
  `dfr.__all__` surface and core package contract documentation.
- Verification: focused public API/import/config/artifact tests passed with 29
  tests; `compileall dfr experiments tests` passed; `git diff --check` passed
  with Windows line-ending warnings only; `pytest -m "not cuda"` passed with
  166 tests and 7 deselected; `pytest -m cuda` passed with 6 tests, 1 skipped
  rasterizer-extension test, and 166 deselected.
- Known limitation: detailed public class/function docstrings still need to be
  filled in across data loaders, configs, artifacts, reconstruction,
  evaluation, and plotting primitives.
- Next step: continue Phase 7 by documenting high-traffic classes/functions in
  `dfr.data.base`, `dfr.data.loading`, `dfr.config`, `dfr.artifacts`,
  `dfr.reconstruction.pipeline`, and `dfr.evaluation.pipeline`.

### 2026-07-09 - Phase 6 plotting decomposition completed

- Changed the final supported `experiments.dfr_plot` wrappers with hidden
  `figs/` writes:
  `plot_jackdaw2_mode_count_curve`,
  `plot_jackdaw2_multiscale_density`, and
  `plot_jackdaw2_dra_scale_model_order_surface` now return figures/results by
  default and save figures only when `output_dir` is supplied.
- Updated `experiments/DFR_PLOT_CATALOG.md` and `experiments/README.md` so all
  supported compatibility wrappers have documented explicit-save behavior.
- Updated `tests/test_plot_catalog.py` to assert no supported wrapper region
  keeps an active `figs/` save path.
- Marked Phase 6 complete and moved the physical deletion of
  `experiments/dfr_plot.py` to Phase 8 compatibility cleanup. Direct execution
  remains disabled, supported wrappers are documented, and archive-only
  functions must not receive new callers.
- Verification: focused plotting/catalog tests passed with 26 tests;
  `compileall dfr experiments tests` passed; `git diff --check` passed with
  Windows line-ending warnings only; `pytest -m "not cuda"` passed with 164
  tests and 7 deselected; `pytest -m cuda` passed with 6 tests, 1 skipped
  rasterizer-extension test, and 164 deselected.
- Known limitation: archive-only functions inside `experiments/dfr_plot.py`
  still contain historical `figs/`, root PNG, and cache writes; they are not
  active/supported commands and should be removed or revived through named
  APIs in Phase 8 or a separately reviewed migration.
- Next step: begin Phase 7 core documentation and public API work.

### 2026-07-09 - Phase 6 Jackdaw2 2D wrapper saves tightened

- Changed `experiments.dfr_plot.plot_jackdaw2_2d_gmm` and
  `plot_jackdaw2_2d_observations` from implicit `figs/scene_traj_*` writers to
  explicit-save compatibility wrappers: they return generated figures by
  default and save only when `output_dir` is supplied.
- Kept rendering delegated to `dfr.plotting.plot_projected_gmm_density`,
  `plot_projection_points`, `plot_density_image`, and `save_figure`.
- Updated `experiments/DFR_PLOT_CATALOG.md`, `experiments/README.md`, and this
  TODO with the new output behavior.
- Updated `tests/test_plot_catalog.py` to assert the wrappers no longer have
  active `figs/` defaults and use explicit `output_dir`/`save_figure`.
- Verification: focused catalog/projection/smoke tests passed with 13 tests;
  `compileall dfr experiments tests` passed; `git diff --check` passed with
  Windows line-ending warnings only; `pytest -m "not cuda"` passed with 164
  tests and 7 deselected; `pytest -m cuda` passed with 6 tests, 1 skipped
  rasterizer-extension test, and 164 deselected.
- Known limitation: supported mode-count, multiscale-density, and DRA-surface
  wrappers still preserve historical `figs/` writes until tightened.
- Next step: continue tightening supported compatibility wrappers with
  remaining implicit `figs/` writes, likely `plot_jackdaw2_mode_count_curve`,
  `plot_jackdaw2_multiscale_density`, or
  `plot_jackdaw2_dra_scale_model_order_surface`.

### 2026-07-08 - Phase 6 trajectory wrapper save tightened

- Changed `experiments.dfr_plot.plot_single_scenario_new` from an implicit
  `figs/scene_traj_<name>.png` writer to an explicit-save compatibility
  wrapper: it returns Figure/Axes by default and saves only when `output_dir`
  is supplied.
- Kept rendering delegated to `dfr.plotting.plot_trajectory_snapshot` and
  figure export delegated to `dfr.plotting.save_figure`.
- Updated `experiments/DFR_PLOT_CATALOG.md`, `experiments/README.md`, and this
  TODO with the new output behavior.
- Updated `tests/test_plot_catalog.py` to assert the wrapper no longer has the
  legacy `figs/` default and uses explicit `output_dir`/`save_figure`.
- Verification: focused catalog/trajectory/smoke tests passed with 12 tests;
  `compileall dfr experiments tests` passed; `git diff --check` passed with
  Windows line-ending warnings only; `pytest -m "not cuda"` passed with 164
  tests and 7 deselected; `pytest -m cuda` passed with 6 tests, 1 skipped
  rasterizer-extension test, and 164 deselected.
- Known limitation: supported Jackdaw2 2D projection/GMM, mode-count,
  multiscale-density, and DRA-surface wrappers still preserve historical
  `figs/` writes until tightened.
- Next step: continue tightening supported compatibility wrappers with
  remaining implicit `figs/` writes, likely the Jackdaw2 projection/GMM
  wrappers.

### 2026-07-08 - Phase 6 publication wrapper saves tightened

- Changed `experiments.dfr_plot.plot_table_2_results`,
  `plot_table_time_efficiency`, and `plot_table_noise_robustness` from
  implicit `figs/` writers/displayers to explicit compatibility wrappers:
  they return figures by default, save only when `save_dir` is supplied, and
  display only when `show=True`.
- Kept rendering delegated to the named publication scripts:
  `experiments.plot_publication_table2`,
  `experiments.plot_publication_time_efficiency`, and
  `experiments.plot_publication_noise_robustness`.
- Updated `experiments/DFR_PLOT_CATALOG.md`, `experiments/README.md`, and this
  TODO with the new output behavior.
- Updated publication wrapper characterization tests to assert explicit
  `save_dir`/`show` behavior and no active `figs/` default.
- Verification: focused publication/catalog tests passed with 18 tests;
  `compileall dfr experiments tests` passed; `git diff --check` passed with
  Windows line-ending warnings only; `pytest -m "not cuda"` passed with 164
  tests and 7 deselected; `pytest -m cuda` passed with 6 tests, 1 skipped
  rasterizer-extension test, and 164 deselected.
- Known limitation: heavier supported wrappers such as trajectory, 2D
  projection/GMM, mode-count, multiscale density, and DRA surface may still
  preserve historical `figs/` writes until tightened in later slices.
- Next step: continue tightening supported compatibility wrappers with
  remaining implicit `figs/` writes, likely `plot_single_scenario_new` or the
  Jackdaw2 projection wrappers.

### 2026-07-08 - Phase 6 camera-configuration wrapper save tightened

- Changed `experiments.dfr_plot.plot_camera_configurations` from an implicit
  `figs/camera_configurations.[png|pdf]` writer to an explicit-save
  compatibility wrapper: it returns Figure/Axes by default and saves only when
  `output_dir` is supplied.
- Kept rendering delegated to `dfr.plotting.plot_camera_configurations` and
  figure export delegated to `dfr.plotting.save_figure`.
- Updated `experiments/DFR_PLOT_CATALOG.md`, `experiments/README.md`, and this
  TODO with the new output behavior.
- Updated `tests/test_plot_catalog.py` to assert the wrapper no longer has the
  legacy `figs/` default and uses explicit `output_dir`/`save_figure`.
- Verification: focused catalog/camera/smoke tests passed with 12 tests;
  `compileall dfr experiments tests` passed; `git diff --check` passed with
  Windows line-ending warnings only; `pytest -m "not cuda"` passed with 164
  tests and 7 deselected; `pytest -m cuda` passed with 6 tests, 1 skipped
  rasterizer-extension test, and 164 deselected.
- Known limitation: other supported compatibility wrappers may still preserve
  historical `figs/` writes until tightened or replaced by named commands.
- Next step: continue tightening supported compatibility wrappers with
  remaining implicit `figs/` writes, or move to Phase 7 documentation if Phase
  6 compatibility cleanup is sufficient.

### 2026-07-08 - Phase 6 dfr_plot compatibility policy documented

- Reviewed the remaining public functions in `experiments/dfr_plot.py` and
  documented the operational support policy in
  `experiments/DFR_PLOT_CATALOG.md`.
- Classified 10 delegated functions as supported compatibility wrappers:
  trajectory, 2D projection/GMM, mode-count, multiscale density, DRA surface,
  camera configuration, and the three publication-table split-outs.
- Classified the remaining public `dfr_plot.py` functions as archive-only
  historical references. New callers should not import them directly; revive
  one only by moving it to a named CLI or package API with an explicit output
  contract and tests.
- Updated `experiments/README.md` and TODO status/decision text with the
  support boundary.
- Added a `tests/test_plot_catalog.py` check that every public
  `dfr_plot.py` function is covered by exactly one support-policy bucket.
- Verification: focused catalog tests passed with 8 tests; `compileall dfr
  experiments tests` passed; `git diff --check` passed with Windows
  line-ending warnings only; `pytest -m "not cuda"` passed with 164 tests and
  7 deselected; `pytest -m cuda` passed with 6 tests, 1 skipped
  rasterizer-extension test, and 164 deselected.
- Known limitation: `experiments/dfr_plot.py` still exists as a compatibility
  archive; actual removal remains a later Phase 6/8 cleanup after compatibility
  policy is accepted.
- Next step: migrate legacy `figs/` saving for supported wrappers, starting
  with `plot_camera_configurations`, or proceed to Phase 7 documentation if no
  further Phase 6 compatibility cleanup is desired.

### 2026-07-08 - Phase 6 publication noise robustness split-out completed

- Added `experiments/plot_publication_noise_robustness.py`, an explicit
  experiment CLI for the hard-coded noise-robustness publication figure
  formerly embedded in `experiments/dfr_plot.py`.
- Kept `plot_table_noise_robustness` as a compatibility wrapper that delegates
  to the named script and preserves the historical `figs/` output default.
- Marked the Phase 6 publication/table-specific split-out task complete:
  Table 2, time-efficiency, and noise-robustness publication figures now have
  small named experiment commands.
- Updated `README.md`, `experiments/README.md`, and
  `experiments/DFR_PLOT_CATALOG.md` so the command and migration boundary are
  discoverable.
- Added `tests/test_publication_noise_robustness.py` for headless plotting,
  save-helper behavior, and legacy wrapper delegation.
- Verification: focused noise/publication/catalog tests passed with 17 tests;
  the `experiments.plot_publication_noise_robustness --help` CLI passed;
  `compileall dfr experiments tests` passed; `git diff --check` passed with
  Windows line-ending warnings only; `pytest -m "not cuda"` passed with 163
  tests and 7 deselected; `pytest -m cuda` passed with 6 tests, 1 skipped
  rasterizer-extension test, and 163 deselected.
- Known limitation: `experiments/dfr_plot.py` remains as a compatibility
  archive with delegated wrappers and unreachable legacy bodies; removal is
  still blocked on the later Phase 6/8 migration policy.
- Next step: review the remaining `experiments/dfr_plot.py` wrappers and
  decide which compatibility entry points are still supported versus
  archive-only historical figures.

### 2026-07-08 - Phase 6 publication Table 2 split-out started

- Added `experiments/plot_publication_table2.py`, an explicit experiment CLI
  for the hard-coded Table 2 capacity-scaling and recall/hallucination
  tradeoff figures formerly embedded in `experiments/dfr_plot.py`.
- Kept `plot_table_2_results` as a compatibility wrapper that delegates to the
  named script and preserves the historical `figs/` output default.
- Updated `README.md`, `experiments/README.md`, and
  `experiments/DFR_PLOT_CATALOG.md` so the command and migration boundary are
  discoverable.
- Added `tests/test_publication_table2.py` for headless plotting, save-helper
  behavior, and legacy wrapper delegation.
- Verification: focused Table 2/publication/catalog tests passed with 14 tests;
  the `experiments.plot_publication_table2 --help` CLI passed; `compileall dfr
  experiments tests` passed; `git diff --check` passed with Windows
  line-ending warnings only; `pytest -m "not cuda"` passed with 160 tests and
  7 deselected; `pytest -m cuda` passed with 6 tests, 1 skipped
  rasterizer-extension test, and 160 deselected.
- Known limitation: the broader publication/table split-out task remains open;
  `plot_table_noise_robustness` is the next candidate in
  `experiments/dfr_plot.py`.
- Next step: continue the publication/table split-out with
  `plot_table_noise_robustness`.

### 2026-07-08 - Phase 6 publication time-efficiency split-out started

- Added `experiments/plot_publication_time_efficiency.py`, an explicit
  experiment CLI for the hard-coded training-time scaling publication figure
  formerly embedded in `experiments/dfr_plot.py`.
- Kept `plot_table_time_efficiency` as a compatibility wrapper that delegates
  to the named script and preserves the historical `figs/` output default.
- Updated `README.md`, `experiments/README.md`, and
  `experiments/DFR_PLOT_CATALOG.md` so the new command and migration boundary
  are discoverable.
- Added `tests/test_publication_time_efficiency.py` for headless plotting,
  save-helper behavior, and legacy wrapper delegation.
- Verification: focused publication/catalog tests passed with 10 tests; the
  `experiments.plot_publication_time_efficiency --help` CLI passed;
  `compileall dfr experiments tests` passed; `git diff --check` passed with
  Windows line-ending warnings only; `pytest -m "not cuda"` passed with 156
  tests and 7 deselected; `pytest -m cuda` passed with 6 tests, 1 skipped
  rasterizer-extension test, and 156 deselected.
- Known limitation: the broader publication/table split-out task remains open;
  `plot_table_2_results` and `plot_table_noise_robustness` are the next
  candidates in `experiments/dfr_plot.py`.
- Next step: continue the publication/table split-out with
  `plot_table_2_results` or `plot_table_noise_robustness`.

### 2026-07-08 - Phase 6 representative plotting smoke matrix completed

- Added `tests/test_plotting_smoke_matrix.py` to make the Phase 6 headless
  smoke-test matrix explicit and durable across camera, 2D, 3D, trajectory,
  scale, and evaluation plot families.
- Confirmed representative plot tests run under Matplotlib's Agg backend and
  cover public exports for camera configuration, projection/density/GMM,
  trajectory, density/multiscale/reconstruction-GMM, mode/DRA scale surfaces,
  evaluation summary, and evaluation metric-series plots.
- Marked Phase 6 step 5 complete.
- Verification: focused plotting/catalog/entrypoint suite (58 passed);
  `compileall`; `git diff --check`; `pytest -m "not cuda"` (153 passed,
  7 deselected, 1 warning); `pytest -m cuda` (6 passed, 1 skipped,
  153 deselected).
- Next step: begin the next unchecked Phase 6 task by splitting
  publication/table-specific figures into small named experiment scripts, with
  the hard-coded Table 2/time/noise figures as likely first candidates.

### 2026-07-08 - Phase 6 result-object plotting contracts completed

- Completed Phase 6 step 4 for reusable package plots: analysis plots accept
  `ModeCurveResult`/`ScaleAnalysisResult`, reconstruction GMM plots accept
  `FrameReconstruction`, and evaluation plots accept `EvaluationRun`,
  `FrameEvaluation`, `EvaluationSummary`, or ordered frame sequences.
- Added `plot_evaluation_metric_series` for per-frame recall, hallucination,
  dMOTA, miss, or metric subsets from typed evaluation results.
- Documented explicit save usage for evaluation summary and metric-series
  figures in the README.
- Recorded the boundary that raw density-grid/multiscale plots stay data-array
  based until a stable density-grid result object exists.
- Verification: focused plotting suite (44 passed); `compileall`;
  `git diff --check`; `pytest -m "not cuda"` (152 passed, 7 deselected,
  1 warning); `pytest -m cuda` (6 passed, 1 skipped, 152 deselected).
- Next step: begin Phase 6 step 5 by ensuring the representative plotting
  smoke-test matrix is complete, then continue extracting publication/noise
  figures behind typed computation/results.

### 2026-07-08 - Phase 6 typed evaluation summary plot path

- Added `dfr.plotting.evaluation` with `plot_evaluation_summary`, a compact
  metric bar chart that accepts `EvaluationRun`, `FrameEvaluation`, or
  `EvaluationSummary` directly and returns Figure/Axes without saving.
- Exported the helper from `dfr.plotting` and documented explicit save usage in
  the README evaluation section.
- Added headless tests for aggregate runs, individual frames, bare summaries,
  existing axes, metric subsets, and validation errors.
- Verification: focused plotting/evaluation checks (31 passed); `compileall`;
  `git diff --check`; `pytest -m "not cuda"` (149 passed, 7 deselected,
  1 warning); `pytest -m cuda` (6 passed, 1 skipped, 149 deselected).
- Next step: extract the next noise/dMOTA plotting primitive after its
  computation has a typed result object, or add per-frame evaluation
  metric-series plots if an active study needs them.

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
