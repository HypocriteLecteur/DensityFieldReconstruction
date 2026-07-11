"""Run ordinary named-scenario reconstructions through the managed package API.

Historical baseline, metrics, timing, and scenario-log helpers were retired in
Phase 8. Use the specialized study modules only when a historical workflow is
explicitly re-owned with a managed artifact contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from dfr import CameraConfig, OutputConfig, ScenarioRunSpec, run_scenario
from dfr.config import ReconstructionParams, TrainingParams


DATASET_RUNS = (
    {"name": "starling", "start_step": 0, "end_step": None, "step_length": 1},
    {"name": "swift", "start_step": 0, "end_step": None, "step_length": 200},
    {"name": "jackdaw", "start_step": 350, "end_step": 550, "step_length": 10},
    {"name": "jackdaw2", "start_step": 2700, "end_step": 3460, "step_length": 20},
)


def run_single_scenario(
    run_params: dict[str, Any],
    *,
    project_root: Path | None = None,
    output: OutputConfig | None = None,
    seed: int = 12345,
):
    """Run one ordinary scenario with managed reconstruction artifacts."""
    root = Path(project_root or Path.cwd()).expanduser().resolve()
    name = str(run_params["name"])
    run = run_scenario(
        ScenarioRunSpec(
            dataset=name,
            start=int(run_params["start_step"]),
            stop=run_params["end_step"],
            step=int(run_params["step_length"]),
            cameras=CameraConfig.encircling(count=2, device="cuda"),
            training=TrainingParams(lr_max_steps=500),
            reconstruction=ReconstructionParams(10, 0.5, 0.3, 32, 20),
            use_ground_truth_scales=True,
            seed=seed,
            projection_noise_std=float(run_params.get("noise_std", 0.0)),
            output=output,
        ),
        project_root=root,
    )
    return run


def create_parser() -> argparse.ArgumentParser:
    """Create the explicit managed ordinary-scenario CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("reconstruct",))
    parser.add_argument("--dataset", choices=tuple(item["name"] for item in DATASET_RUNS))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-id")
    parser.add_argument("--overwrite-run", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run selected ordinary scenarios through the managed reconstruction API."""
    args = create_parser().parse_args(argv)
    selected = [
        item for item in DATASET_RUNS if args.dataset is None or item["name"] == args.dataset
    ]
    if args.run_id is not None and len(selected) != 1:
        raise ValueError("--run-id requires selecting exactly one --dataset.")

    for params in selected:
        name = params["name"]
        output = OutputConfig(
            workflow="reconstruction",
            name=f"ordinary-{name}",
            root=args.output_root,
            run_id=args.run_id,
            project_root=args.project_root,
            overwrite=args.overwrite_run,
        )
        run = run_single_scenario(
            params, project_root=args.project_root, output=output, seed=args.seed
        )
        print(f"[{name}] managed outputs: {run.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
