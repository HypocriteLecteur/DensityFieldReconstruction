# Phase 8 Archive and Deletion Policy

Last updated: 2026-07-10.

This policy turns the Phase 8 compatibility inventory into deletion rules.
It is intentionally conservative: remove duplicate source and generated
artifacts from the active tree only after the replacement path, preservation
mechanism, and verification steps are documented.

## Preservation mechanism

The canonical archive for pre-refactor source is local Git history plus the
annotated `v0.1.0` stable tag. Do not keep duplicate source trees in `main`
only as backups. If the owner needs a paper-specific archive branch or an
external zip of generated artifacts, create that explicitly as a separate
task before deletion.

## Source archive surfaces

### `experiments.dfr_plot` and `experiments.plotting_utils`

Treat `experiments.dfr_plot` and `experiments.plotting_utils` as archive
surfaces during Phase 8:

- do not add new imports from either module;
- preserve the historical function list in `experiments/DFR_PLOT_CATALOG.md`;
- keep supported replacements documented in `dfr.plotting` or the named
  publication scripts;
- remove the `--list-functions` command dependency before deleting
  `experiments/dfr_plot.py`;
- remove the pair together unless an intermediate commit keeps
  `dfr_plot.py` working without `plotting_utils.py`.

If an archive-only function becomes scientifically active again, migrate it
first to a named package API or experiment CLI with tests and managed-output
behavior.

### Copied backup directories

`density_field_reconstruction_copy/` and `experiments_legacy/` are backup
copy candidates, not active package surfaces. They should be removed from
`main` after one focused no-import/no-command guard confirms that supported
code, tests, docs, and cataloged commands do not depend on them.

These two trees are intentionally ignored and were never stored in Git, so
`v0.1.0` cannot preserve their contents. Before their approved 2026-07-10
removal, create and verify a dedicated local archive. The archive for this
cleanup is
`outputs/releases/DensityFieldReconstruction-phase8-legacy-copies-20260710.zip`
(SHA-256 `B49E0856B468BB23C4A9D88948301E119AB8F137A724E8AAD8732E25611B84A9`).

## Generated artifact surfaces

### `figs/` and `results/`

Treat committed or working-tree content under `figs/` and `results/` as
generated or historical artifacts unless a file is explicitly reclassified as
a small curated fixture. Active maintained workflows write through the managed
`outputs/` contract. Do not migrate generated artifacts into the package tree.

Remove or archive generated artifacts in a separate commit after their active
producers are retired or redirected. If the owner wants a human-readable
backup of historical generated files, create an external zip before removal;
otherwise local Git history and `v0.1.0` are the preservation mechanism.

### `scenarios/*/logs/`

Treat `scenarios/*/logs/` readers and writers as legacy run-artifact surfaces.
Do not add new scenario-log producers. Remove producers only after an
equivalent managed-run replacement exists, or after the owner confirms the
study is historical and should remain available only through Git history.

## Required deletion workflow

For every Phase 8 removal:

1. Remove one surface per commit.
2. Update `TODO.md`, `docs/PHASE8_COMPATIBILITY_INVENTORY.md`, and any
   affected catalog or ownership docs in the same commit.
3. Keep or add a focused guard proving no supported caller imports or invokes
   the removed surface.
4. Run focused guards plus the full CPU and CUDA test tiers after the change.
5. Do not run destructive filesystem deletion without explicit approval and
   the normal sandbox escalation path.
