# `experiments/dfr_plot.py` Phase 6 Catalog

This catalog freezes the state of `experiments/dfr_plot.py` at the start of
Phase 6. Do not move or delete functions from that file until this table is
updated and the migration target has tests.

## Summary

- Top-level functions: 36.
- External Python callers found by static scan: none for public figure
  functions. The module is currently a legacy archive invoked only by explicit
  imports or by `python -m experiments.dfr_plot --list-functions`.
- Common inputs: named scenario configs under `scenarios/`, legacy
  `reconstruction_scale.npz` caches, legacy scenario logs/checkpoints, and
  hard-coded publication constants.
- Common outputs: `figs/`, root-level PNG/PDF files, local pickle/NPZ caches,
  and interactive `plt.show()` calls. These are legacy locations; migrated
  plotting must use `RunArtifacts.figures_dir`, `data/`, `metrics/`, or caller
  supplied paths.
- Migration rule: computation moves to `dfr.analysis` or `dfr.evaluation`;
  reusable plotting primitives move to `dfr.plotting`; publication-only figure
  assembly stays as small experiment scripts.

## Classification key

- `package-plot`: reusable plotting/rendering candidate for `dfr.plotting`.
- `package-compute`: reusable numerical work that should not remain in a plot
  module.
- `experiment-figure`: publication or paper-specific figure assembly.
- `experiment-study`: expensive scientific study coupled to reconstruction or
  evaluation; extract computation first.
- `legacy-helper`: helper kept only while legacy functions remain.
- `obsolete/manual`: exploratory function with no known caller and unclear
  repeatability; keep until reviewed, but do not promote without a fresh owner.

## Function inventory

| Function | Lines | Known callers | Inputs / loaded data | Outputs / side effects | Classification | Migration target / notes |
|---|---:|---|---|---|---|---|
| `_unpack` | 94-97 | Internal helper | `run_params` dict | None | legacy-helper | Replace with typed specs or remove when legacy functions migrate. |
| `_load_scenario` | 98-104 | Internal helper | `scenarios/<name>/config.yaml`, dataset path from config | Uses cwd-relative paths | legacy-helper | Replace with `dfr.load_dataset` / `DatasetSpec`. |
| `_step_range` | 105-110 | Internal helper | Dataset trajectory length and frame controls | None | legacy-helper | Replace with `dfr.data.select_frame_indices`. |
| `_build_cam_system` | 111-128 | Internal helper | Dataset, config, generated encircling cameras | CUDA camera system | legacy-helper | Already superseded by `dfr.reconstruction.build_camera_system`. |
| `scale_estimation` | 129-152 | None | `reconstruction_scale_estim*.npy` | Interactive plot; commented legacy `np.save` | obsolete/manual | Review against current analysis scale APIs; likely obsolete. |
| `plot_multiple_scenarios` | 153-157 | None | `DATASET_RUNS` | Interactive `plt.show()` | experiment-figure | Replace with explicit CLI if still needed. |
| `plot_single_scenario_new` | 158-180 | Internal from `plot_multiple_scenarios` | Named scenario positions | `figs/scene_traj_<name>.png` | package-plot | Migrated wrapper: trajectory rendering now delegates to `dfr.plotting.plot_trajectory_snapshot`; legacy wrapper still saves to `figs/`. |
| `plot_jackdaw2_density_field` | 181-325 | None | `jackdaw2`, `reconstruction_scale.npz` | Three density/GMM PNGs in `figs/` | experiment-figure | Split reusable density/GMM rendering into `dfr.plotting.reconstruction`; keep jackdaw2 composition as experiment script. |
| `plot_all_ground_truth_density_fields` | 326-403 | None | All `DATASET_RUNS`, `reconstruction_scale.npz` | Ground-truth density PNGs in `figs/` | experiment-figure | Generalize to dataset-density figure from loaded dataset + scale. |
| `plot_jackdaw2_2d_gmm` | 404-575 | None | `jackdaw2`, camera system, GMM reconstruction | Per-camera 2D GMM PNGs in `figs/` | package-plot | Migrated wrapper: image-plane GMM rendering delegates to `dfr.plotting.plot_projected_gmm_density`; computation/loading remains legacy. |
| `plot_jackdaw2_2d_observations` | 576-672 | None | `jackdaw2`, `reconstruction_scale.npz`, projections | Coarse/projection PNGs in `figs/` | package-plot | Migrated wrapper: projection scatter and density rendering delegate to `dfr.plotting`; computation/loading remains legacy. |
| `plot_single_scenario` | 673-723 | None | Named scenario positions | Commented legacy save; interactive plot | obsolete/manual | Superseded by `plot_single_scenario_new`/animation tooling unless owner says otherwise. |
| `overview_scaling_law` | 724-781 | None | Synthetic 2D Gaussian field | `figs/2d_gss.png`, interactive plot | experiment-figure | Publication schematic; make small explicit figure script if retained. |
| `plot_scale_space_curve` | 782-821 | None | Synthetic 1D curve | `figs/2d_gss_curve.png` | experiment-figure | Publication schematic; no package dependency needed. |
| `_validate_nnd_bounds` | 822-832 | Internal helper | Nearest-neighbor scaling inputs | Raises validation errors | package-compute | Move beside adaptive scale utilities if still needed. |
| `_select_adaptive_density_scales` | 833-911 | Internal helper | Positions, reference scale, target levels | Scale list | package-compute | Candidate for `dfr.analysis.scales`; add deterministic tests before moving. |
| `plot_jackdaw2_mode_count_curve` | 912-1088 | Internal from multiscale density | Mode-count cache NPZ or recomputed CUDA mode counts | Cache NPZ, mode-count curve figure | package-compute | Split mode-count computation/cache to analysis; curve plot to `dfr.plotting.analysis`. |
| `plot_jackdaw2_multiscale_density` | 1089-1254 | None | Density cache NPZ and mode-count cache | Cache NPZ, multiscale density figure | experiment-figure | Extract density panels renderer; keep jackdaw2-specific assembly. |
| `plot_jackdaw2_dra_scale_model_order_surface` | 1255-1395 | None | `jackdaw2`, generated scale/model grid | DRA surface PNG, interactive plot | package-compute | DRA computation is already partly in `dfr.analysis`; migrate plot to `dfr.plotting.analysis`. |
| `visual_hull_diagram` | 1396-1431 | None | Named scenario/camera projection | No save currently | obsolete/manual | Review with visual-hull studies; likely publication schematic. |
| `assumption_3_error` | 1432-1556 | None | Synthetic 2D surface | Root-level `<name>_error_*.png`, interactive plot | experiment-figure | Publication-specific; move to explicit script with managed figure output. |
| `visual_hull_tau_vs_visual_hull_ghost` | 1557-1674 | None | Synthetic visual-hull geometry | `figs/VH_diagram.png`, interactive plot | experiment-figure | Publication schematic; isolate from package computation. |
| `run_geometric_visual_hulls` | 1675-1784 | None | Scenario positions, `reconstruction_scale.npz` | Interactive plot | experiment-study | Extract visual-hull computation before plotting. |
| `plot_ratio_surface` | 1785-1861 | None | Scenario positions and derived grid | Interactive ratio surface | experiment-study | Needs owner; likely analysis computation + plot split. |
| `dra_metrics` | 1862-1886 | None | Hard-coded metrics example | Interactive plot; commented PDF save | obsolete/manual | Manual schematic; keep out of package unless revived. |
| `one_frame_parameter_search` | 1887-1951 | None | Scenario scale cache, one frame | CUDA reconstruction/evaluation side effects | experiment-study | Computation belongs in configurable runner/search workflow, not plotting. |
| `one_frame_convergence` | 1952-2076 | None | Scenario scale cache, one frame | Interactive/convergence plots; commented PDF/PNG saves | experiment-study | Extract convergence run/result type before plotting. |
| `one_frame_dMOTA_factor_analysis` | 2077-2197 | None | Scenario scale cache, pickle cache | Root `dmota_comparison.png`, pickle cache, interactive plot | package-compute | Evaluation sweep should move to `dfr.evaluation` or experiment CLI; plot reads typed results. |
| `one_frame_dMOTA_factor_analysis_2` | 2198-2485 | None | Generated/loaded pickle cache | Root `dra_target_modes_comparison.png`, pickle cache, interactive plot | package-compute | Same as above; unify target-mode factor sweep. |
| `one_frame_dMOTA_noise` | 2486-2633 | None | Scenario scale cache, pickle cache | Root `dra_noise_variance_comparison.png`, pickle cache, interactive plot | package-compute | Noise robustness sweep should reuse evaluation API and managed artifacts. |
| `one_frame_dMOTA_3d_noise` | 2634-2755 | None | Scenario scale cache, pickle cache | Root `dra_mapped_3d_noise_comparison.png`, pickle cache, interactive plot | package-compute | Same as noise sweep; separate metric computation from plot. |
| `plot_dra_and_loss` | 2756-3119 | None | `reconstruction_scale.npz`, reconstructed models/losses | DRA/loss and GT/reconstruction figures | experiment-figure | Use saved reconstruction/evaluation readers; extract reusable dual-axis plot. |
| `plot_camera_configurations` | 3120-3195 | None | Named scenario/camera generated poses | `figs/camera_configurations.[png|pdf]` | package-plot | Migrated wrapper: rendering now delegates to `dfr.plotting.plot_camera_configurations`; legacy wrapper still saves to `figs/`. |
| `plot_table_2_results` | 3196-3438 | None | Hard-coded publication metrics | Table 2 PNG/PDF, interactive plot | experiment-figure | Keep as publication-table script reading managed metrics. |
| `plot_table_time_efficiency` | 3439-3583 | None | Hard-coded time-efficiency metrics | Table-time PNG/PDF, interactive plot | experiment-figure | Keep as publication-table script reading managed metrics. |
| `plot_table_noise_robustness` | 3584-3772 | None | Hard-coded noise metrics | Table-noise PNG/PDF, interactive plot | experiment-figure | Keep as publication-table script reading managed metrics. |

## Migration order recommendation

1. Continue with low-risk reusable plot primitives. The
   `plot_camera_configurations`, `plot_single_scenario_new`,
   `plot_jackdaw2_2d_gmm`, and `plot_jackdaw2_2d_observations` rendering
   primitives have moved; next candidates are analysis-backed mode-count,
   DRA-surface, or multiscale-density plots.
2. Then move analysis-backed plots whose computation already has package
   analogs: mode-count curves, DRA surfaces, and multiscale density panels.
3. Defer the one-frame dMOTA/noise/convergence studies until their computation
   has explicit config/result objects and managed caches.
4. Leave publication-table and schematic figures as small experiment scripts
   unless they prove broadly reusable.

## Open questions for the owner

- Which paper figures are still active deliverables versus historical drafts?
- Should `figs/` outputs be regenerated from managed runs or preserved only at
  the `v0.1.0` baseline?
- Which dMOTA/noise sweeps are scientifically current enough to justify a
  package-level result type?
