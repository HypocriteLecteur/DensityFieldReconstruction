# Phase 8 Compatibility Inventory

Last scanned: 2026-07-10.

This inventory is the starting point for Phase 8 cleanup. It records what is
still compatibility/archive material, what appears to have no active caller,
and which legacy output producers must be handled before deleting directories
or wrappers.

## Compatibility-wrapper import status

Static scan result:

- No active Python module imports `experiments.dfr_plot`.
- The only supported executable interaction with `dfr_plot.py` is:

  ```powershell
  python -m experiments.dfr_plot --list-functions
  ```

- Documentation/tests mention `experiments.dfr_plot` for policy and catalog
  checks only.
- `experiments.plotting_utils` is imported only by `experiments.dfr_plot`.

Implication: Phase 8 can remove or archive `dfr_plot.py` and
`plotting_utils.py` by following `docs/PHASE8_ARCHIVE_POLICY.md`. Until then,
new code must not add imports from either module.

## Supported `dfr_plot.py` compatibility wrappers

These wrappers remain characterized by tests and documented in
`experiments/DFR_PLOT_CATALOG.md`. They are safe to remove only after the owner
accepts the named package/script replacements as the migration path:

- `plot_single_scenario_new`
- `plot_jackdaw2_2d_gmm`
- `plot_jackdaw2_2d_observations`
- `plot_jackdaw2_mode_count_curve`
- `plot_jackdaw2_multiscale_density`
- `plot_jackdaw2_dra_scale_model_order_surface`
- `plot_camera_configurations`
- `plot_table_2_results`
- `plot_table_time_efficiency`
- `plot_table_noise_robustness`

The replacement owners are `dfr.plotting` for reusable plotting primitives and
the explicit publication figure scripts for hard-coded table figures.

## Archive-only `dfr_plot.py` functions

The Phase 6 catalog classifies all other public functions in `dfr_plot.py` as
archive-only historical references. They have no supported active caller. If
one becomes scientifically active again, migrate it first to a named CLI or
package API with tests and an explicit output contract.

## Legacy output producers

### `figs/`

Active supported compatibility wrappers no longer default to `figs/`; they
save only when a caller supplies an explicit directory. Remaining `figs/`
writers are concentrated in:

- archive-only `experiments.dfr_plot` functions;
- legacy explicit-dispatch studies such as `experiments.power_law`;
- importable plotting helper functions inside `parameter_manifold.py` and
  `parameter_manifold_2pl.py` before `main()` resets their figure directory to
  managed artifacts.

Phase 8 should not delete `figs/` in the same commit as code removal. First
remove or archive producers, then separately decide whether committed/generated
figures are preserved only at `v0.1.0`, archived, or deleted.

### `results/`

New managed analysis writes go under `outputs/analysis/<run-id>/`. Remaining
`results/` references are legacy cache readers/producers, notably:

- `experiments.dfr_plot` mode/DRA cache paths;
- `experiments.fit_dra_multiframe.seed_existing_cache`, which may copy a
  matching legacy single-frame cache into a managed run cache;
- historical cache directories already present under `results/`.

Phase 8 can remove `results/` only after all cache-seeding and archive-only
study paths are either retired or redirected.

### `scenarios/*/logs/`

Primary migrated reconstruction paths use managed `outputs/reconstruction/`
runs. Remaining scenario-log readers/writers are legacy consumers or
specialized study paths:

- baseline/metric helpers in `experiments.run_scenarios.py`;
- angle-sweep baseline/profile/diagnostic paths;
- flock visualization/baseline/metrics helpers;
- UE4/flock historical visualization or metrics consumers;
- `experiments.visualize.py`, `inspect_3d_error.py`, and
  `run_post_processing.py`.

Do not add new scenario-log producers. Remove or archive readers only after a
managed-run replacement exists or the owner confirms the study is historical.

## Copied/backup directories

`density_field_reconstruction_copy/` and `experiments_legacy/` were ignored,
local-only backup trees rather than package imports. Phase 8 confirmed they
had no active package, experiment, or example Python reference, created the
verified archive named in `docs/PHASE8_ARCHIVE_POLICY.md`, and removed both on
2026-07-10. They are now protected by a post-cleanup absence test instead of
being kept in the active working tree.

## Recommended Phase 8 order

1. Add or keep static guards that prevent new active imports from
   `experiments.dfr_plot` and `experiments.plotting_utils`.
2. Apply `docs/PHASE8_ARCHIVE_POLICY.md` to `dfr_plot.py`,
   `plotting_utils.py`, backup directories, and generated `figs/`/`results/`
   content.
3. Remove one compatibility/archive surface at a time, updating
   `experiments/DFR_PLOT_CATALOG.md`, `experiments/README.md`,
   `docs/MODULE_OWNERSHIP.md`, and `TODO.md` in the same commit.
4. Run focused catalog/import tests plus the full CPU and CUDA tiers after each
   deletion.
