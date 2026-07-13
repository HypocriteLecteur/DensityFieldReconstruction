# DFR v0.2.0 — Refactored Workflow Release

`v0.2.0` is the first local release after the usability and architecture
refactor. It replaces the mixed experiment-script workflow with a small public
Python API, typed configuration/results, and managed artifact directories.

## Use this instead

```python
import dfr

dataset = dfr.load_dataset("jackdaw2")
analysis = dfr.analyze(
    dataset,
    kind="modes",
    config=dfr.AnalysisConfig(frames=(2800,), scales=(0.5, 1.0, 1.5)),
)
run = dfr.reconstruct(
    dataset,
    frames=(2800,),
    cameras=dfr.CameraConfig.encircling(count=4),
    scale=1.0,
    output=dfr.OutputConfig(workflow="reconstruction", name="jackdaw2-demo"),
)
evaluation = dfr.evaluate(run, ground_truth=dataset)
```

For reproducible command-line work, prefer the documented `python -m
experiments...` entry points. They expose explicit `--project-root`,
`--output-root`, run-ID, and collision-policy options where they save work.

## Output migration

New artifacts are written beneath:

```text
outputs/<workflow>/<run-id>/
  config.yaml
  manifest.json
  data/ checkpoints/ metrics/ figures/ logs/ cache/
```

Do not add new writes to `figs/`, `results/`, or `scenarios/*/logs/`. Existing
historical contents remain untouched for scientific preservation; explicitly
seed an old DRA cache only with the documented legacy-cache option.

## Retired surfaces

The former `experiments/dfr_plot.py`, `experiments/plotting_utils.py`,
`power_law.py`, isolated diagnostics, duplicate source trees, and historical
scenario-log runner studies were retired. Their source remains recoverable
from the local `v0.1.0` tag and Git history. Use `python -m
experiments.plot_catalog --list-functions` to inspect the frozen historical
plot-function catalog without importing archived code.

## Compatibility notes

- Reusable plots now live in `dfr.plotting` and return Matplotlib objects;
  saving is explicit.
- Named datasets load through `dfr.load_dataset`; analysis, reconstruction, and
  evaluation return typed result objects.
- Measured flock and UE4 image detections use `reconstruct_observations` via
  their explicit specialized commands.
- The project remains locally versioned. `v0.1.0` is the pre-refactor stable
  baseline, and `v0.2.0` is the refactor release.
