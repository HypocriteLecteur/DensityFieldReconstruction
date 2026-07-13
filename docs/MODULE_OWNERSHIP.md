# Module Ownership Map

Use this map when deciding where a refactor, helper, or new script should live.
The north star is: reusable computation belongs in `dfr/`; research-specific
or publication-specific orchestration belongs in `experiments/`; generated
artifacts belong in `outputs/`.

## Package modules

| Area | Owner module(s) | Responsibilities | Do not put here |
|---|---|---|---|
| Dataset loading | `dfr.data`, `dfr.dataset_io` | Resolve scenarios, load supported file formats, expose `Dataset` protocol and frame selection. | Plotting, reconstruction side effects, hard-coded study constants. |
| Workflow configuration | `dfr.config` | Serializable dataclasses for analysis, cameras, evaluation, and runs. | Experiment-specific defaults that are not reusable. |
| Managed artifacts | `dfr.artifacts` | `OutputConfig`, `RunArtifacts`, manifests, config serialization, safe category paths. | Scientific computation or Matplotlib styling. |
| Analysis computation | `dfr.analysis` | Mode counts, DRA scale/model-order surfaces, manifold fitting, typed analysis results. | Publication figure layout or implicit cache/output roots. |
| Reconstruction | `dfr.reconstruction` | Camera-system construction, typed reconstruction requests/results, scenario runners, external observation reconstruction. | One-off study loops or table-specific presets. |
| Evaluation | `dfr.evaluation` | Density-overlap metrics, typed evaluation runs, managed evaluation output. | Plot formatting or scenario selection. |
| Plotting | `dfr.plotting` | Reusable Matplotlib primitives that return figures/axes/artists and never save implicitly. | Data loading, CUDA reconstruction, managed-run orchestration. |
| Model checkpointing | `dfr.model_checkpoint` | Gaussian model checkpoint build/load/restore compatibility helpers. | Run-directory policy; use `RunArtifacts` for where files go. |

## Experiment modules

| Area | Owner module(s) | Current status |
|---|---|---|
| Managed analysis CLIs | `experiments.plot_dra_scale_model_order`, `fit_dra_multiframe`, `parameter_manifold*`, `mechanistic_derivation`, `synthetic_benchmark`, `validate_mode_counting` | Supported analysis commands. They should parse arguments, call `dfr.analysis`, and write through managed artifacts or `dfr.plotting.save_figure`. |
| One-frame reconstruction CLI | `experiments.reconstruct_one_frame` | Transitional wrapper over `dfr.reconstruct`. Keep CLI concerns here; reusable logic belongs in `dfr.reconstruction`. |
| Publication reconstruction tables | `experiments.publication_scenarios`, `run_scenarios_table_2/3/4` | Table-specific presets and command dispatch. Shared reconstruction should stay in `ScenarioRunSpec`/`run_scenario`. |
| Publication figures | `experiments.plot_publication_table2`, `plot_publication_time_efficiency`, `plot_publication_noise_robustness` | Explicit figure commands for hard-coded publication tables. General plotting primitives belong in `dfr.plotting`. |
| Legacy plot catalog | `experiments.plot_catalog`, `experiments.DFR_PLOT_CATALOG.md` | Standalone catalog command for historical public function names. The retired source remains available through local Git history and `v0.1.0`; it is not an active import surface. |
| Specialized workflows | `run_scenarios_angle_sweep`, `run_scenarios_flock`, `run_scenarios_ue4`, `RUNNER_SPECIALIZATIONS.md` | Explicit-dispatch special cases. Promote reusable camera/external-observation logic into `dfr.reconstruction` when generalized. |
| Retired legacy studies | Local Git history and `v0.1.0` | Power-law and reconstruction-scale scripts are archive-only. Re-own reusable analysis in `dfr.analysis` before adding a new CLI. |
| Animation workflow | `generate_scene_animations`, `dfr.evaluation.density` | Managed MP4 rendering over configured studies. Keep the density-grid computation reusable and file-free in `dfr.evaluation`; the CLI owns `outputs/animations/<run-id>/figures/`. |

## Output ownership

| Output location | Owner | Policy |
|---|---|---|
| `outputs/<workflow>/<run-id>/` | `dfr.artifacts` and managed CLIs | Canonical root for new generated artifacts. |
| `outputs/.../figures/` | Calling workflow via `RunArtifacts.save_figure` | Preferred managed figure destination. |
| Explicit standalone path | Caller via `dfr.plotting.save_figure` | Acceptable for docs, compatibility wrappers, and ad hoc exports. |
| `figs/` | Legacy scripts only | Do not add new producers. |
| `results/` | Legacy caches only | Do not add new producers. |
| `scenarios/*/logs/` | Legacy reconstruction logs | New reconstruction should use managed `outputs/reconstruction/`. |

## Refactor checklist

Before moving or adding code, answer:

1. Is this reusable computation? Put it in `dfr/` with typed inputs/results.
2. Is this a paper/table/study preset? Put orchestration in `experiments/` and
   call package APIs.
3. Does it write files? Require an explicit path, `OutputConfig`, or
   `RunArtifacts`.
4. Does it plot? Return `Figure`/`Axes` from `dfr.plotting`; save only at the
   caller edge.
5. Does it touch a legacy script? Update `experiments/README.md`,
   `experiments/DFR_PLOT_CATALOG.md` when relevant, and `TODO.md`.
