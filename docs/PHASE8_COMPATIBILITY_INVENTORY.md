# Phase 8 Compatibility Inventory

Last scanned: 2026-07-10.

This inventory is the starting point for Phase 8 cleanup. It records what is
still compatibility/archive material, what appears to have no active caller,
and which legacy output producers must be handled before deleting directories
or wrappers.

## Compatibility-wrapper import status

Removal result:

- `experiments/dfr_plot.py` and `experiments/plotting_utils.py` were removed
  together on 2026-07-10.
- No active Python module imports `experiments.dfr_plot`.
- No active Python module imports `experiments.plotting_utils`.
- The supported executable interaction with the frozen historical catalog is:

  ```powershell
  python -m experiments.plot_catalog --list-functions
  ```

- The 30 public names and their support classifications remain in
  `experiments/DFR_PLOT_CATALOG.md`; source-reading wrapper tests were replaced
  by catalog and package/publication-owner checks.

Implication: the historical implementation is preserved through local Git
history and `v0.1.0`, while active code uses `dfr.plotting`, named publication
scripts, and `experiments.plot_catalog`.

## Former supported compatibility wrappers

These historical wrapper names remain documented in
`experiments/DFR_PLOT_CATALOG.md`. Their replacement owners are the active
interfaces:

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

The reusable replacement owner is `dfr.plotting`; hard-coded table figures are
owned by the explicit publication figure scripts.

## Former archive-only functions

The Phase 6 catalog classifies all other public functions as archive-only
historical references. They have no supported active caller. If one becomes
scientifically active again, recover it from Git history first, then migrate it
to a named CLI or package API with tests and an explicit output contract.

## Legacy output producers

The following is the current literal-path static inventory after the retired
plot archive was removed on 2026-07-10. It distinguishes path references from
active supported writers; each remaining script still needs a separate
reader/writer classification before migration or deletion.

### `figs/`

Active supported compatibility wrappers no longer default to `figs/`; they
save only when a caller supplies an explicit directory. Remaining `figs/`
There are no remaining active `figs/` producers:

- `experiments.power_law` was the only literal writer and now creates managed
  analysis artifacts under `outputs/analysis/<run-id>/figures/`;
- `experiments.parameter_manifold` and `experiments.parameter_manifold_2pl`
  no longer use `Path("figs")` as an import-time fallback. Their
  figure-producing helpers require the managed CLI to set the artifact figure
  directory.

Phase 8 should not delete `figs/` in the same commit as code removal. First
separately inspect historical content, then decide whether committed/generated
figures are preserved only at `v0.1.0`, archived, or deleted.

### `results/`

New managed analysis writes go under `outputs/analysis/<run-id>/`. Remaining
`results/` references are legacy cache readers/producers, notably:

- `experiments.fit_dra_multiframe.seed_existing_cache`, which no longer reads
  `results/` by default. It accepts `--legacy-cache-root` only when a
  researcher explicitly wants to copy a matching historical cache into a
  managed run cache;
- historical cache directories already present under `results/`.

Phase 8 can remove `results/` only after all cache-seeding and archive-only
study paths are either retired or redirected.

### `scenarios/*/logs/`

Primary migrated reconstruction paths use managed `outputs/reconstruction/`
runs. Remaining scenario-log readers/writers are legacy consumers or
specialized study paths:

- baseline/metric helpers in `experiments.run_scenarios`;
- angle-sweep baseline/profile/diagnostic paths in
  `experiments.run_scenarios_angle_sweep`;
- flock visualization/baseline/metrics helpers and its optional cleanup path
  in `experiments.run_scenarios_flock`;
- `experiments.visualize`, `experiments.inspect_3d_error`,
  `experiments.run_post_processing`, and
  `experiments.reconstruction_scale_determination`;
- historical search/evaluation readers in `experiments.search_learning_parameters`,
  `experiments.search_regularization_parameters`, and
  `experiments.compute_metrics_from_pretrained`.

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

1. Keep static guards that ensure the retired plot modules remain absent and
   cannot return as active imports.
2. Apply `docs/PHASE8_ARCHIVE_POLICY.md` to remaining generated `figs/`/
   `results/` content and scenario-log readers/producers.
3. Remove one remaining compatibility/archive surface at a time, updating
   `experiments/DFR_PLOT_CATALOG.md`, `experiments/README.md`,
   `docs/MODULE_OWNERSHIP.md`, and `TODO.md` in the same commit.
4. Run focused catalog/import tests plus the full CPU and CUDA tiers after each
   deletion.
