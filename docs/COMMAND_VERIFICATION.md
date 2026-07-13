# Command Verification Notes

Last reviewed: 2026-07-10.

This page records which README/docs commands were actually run during Phase 7
documentation closure and which commands remain intentionally unexecuted
because they require CUDA, large local datasets, long runtimes, or external
image/detection assets.

## Executed successfully

CPU-safe runnable example:

```powershell
python examples/toy_workflow.py
```

Help/dispatch checks:

```powershell
python -m experiments.plot_dra_scale_model_order --help
python -m experiments.fit_dra_multiframe --help
python -m experiments.parameter_manifold --help
python -m experiments.parameter_manifold_2pl --help
python -m experiments.mechanistic_derivation --help
python -m experiments.synthetic_benchmark --help
python -m experiments.validate_mode_counting --help
python -m experiments.plot_catalog --list-functions
python -m experiments.reconstruct_one_frame --help
python -m experiments.run_scenarios_table_2 reconstruct --help
python -m experiments.run_scenarios_table_3 reconstruct --help
python -m experiments.run_scenarios_table_4 run --help
python -m experiments.plot_publication_table2 --help
python -m experiments.plot_publication_time_efficiency --help
python -m experiments.plot_publication_noise_robustness --help
python -m experiments.run_scenarios_angle_sweep --help
python -m experiments.run_scenarios_flock --help
python -m experiments.generate_scene_animations --help
python -m experiments.run_scenarios_ue4 --help
```

## Intentionally not executed in the docs review

The following README/docs examples are syntactically documented but require
large ignored datasets, CUDA reconstruction, compiled rasterizers, external
image/detection assets, or long-running study settings:

- `python -m experiments.plot_dra_scale_model_order --datasets jackdaw2 --output-root outputs --run-id jackdaw2-dra`
- `python -m experiments.fit_dra_multiframe --datasets jackdaw2 --frames-per-dataset 3 --output-root outputs --run-id jackdaw2-multiframe`
- `python -m experiments.run_scenarios_table_3 reconstruct --datasets starling --camera-counts 2 --run-id-prefix table3-smoke`
- `python -m experiments.run_scenarios_table_4 run --noise-levels 0 1 2 --camera-counts 2`
- `python -m experiments.reconstruct_one_frame --dataset jackdaw2 --frame 2800 --camera-count 2 --scale 1.0 --iterations 100 --run-id jackdaw2-frame-2800`
- `python -m experiments.run_scenarios`
- In-memory CUDA snippets using `dfr.reconstruct`, `dfr.run_scenario`,
  `dfr.evaluate(..., device="cuda")`, or `dfr.analyze(..., kind="dra")`

Before relying on one of these commands, verify that the named scenario's
`config.yaml`, ignored source dataset, CUDA device, and required rasterizer
extensions are present.
