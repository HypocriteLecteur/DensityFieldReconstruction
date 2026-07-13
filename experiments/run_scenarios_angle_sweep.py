"""Managed reconstruction entry point for the former camera-angle study.

The historical profiling, baseline, convergence, and scenario-log studies are
preserved in local Git history.  This module retains only its reusable ordinary
reconstruction preset and delegates execution to :func:`dfr.run_scenario`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from dfr import CameraConfig, OutputConfig, ScenarioRunSpec, run_scenario
from dfr.config import ReconstructionParams, TrainingParams
from dfr.data.registry import default_project_root


CAMERA_COUNT = 2
USE_DECOUPLED = False
USE_GROUND_TRUTH_SCALES = True

DATASET_RUNS = (
    {"name": "starling", "start_step": 0, "end_step": None, "step_length": 1},
    {"name": "swift", "start_step": 0, "end_step": None, "step_length": 200},
    {"name": "jackdaw", "start_step": 350, "end_step": 550, "step_length": 10},
    {"name": "jackdaw2", "start_step": 2700, "end_step": 3460, "step_length": 20},
)


def run_single_scenario(
    run_params: dict,
    *,
    project_root: Path | None = None,
    output: OutputConfig | None = None,
    seed: int = 12345,
):
    """Reconstruct one configured scenario through the shared package runner."""
    root = Path(project_root or default_project_root()).expanduser().resolve()
    return run_scenario(
        ScenarioRunSpec(
            dataset=str(run_params["name"]),
            start=int(run_params["start_step"]),
            stop=run_params["end_step"],
            step=int(run_params["step_length"]),
            cameras=CameraConfig.encircling(count=CAMERA_COUNT, device="cuda"),
            training=TrainingParams(
                xyz_lr_c=0.05,
                xyz_lr_final_c=0.9,
                radius_lr_c=0.05,
                radius_lr_final_c=0.9,
                weights_lr_c=0.10,
                weights_lr_final_c=0.7,
                xyz_reg=1.0,
                radius_reg=0.3,
                radius_cutoff_inv=0.5,
                lr_max_steps=100,
            ),
            reconstruction=ReconstructionParams(10, 0.5, 0.3, 32, 20),
            use_ground_truth_scales=USE_GROUND_TRUTH_SCALES,
            projection_noise_std=float(run_params.get("noise_std", 0.0)),
            use_decoupled=USE_DECOUPLED,
            seed=seed,
            output=output,
        ),
        project_root=root,
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the explicit managed-reconstruction command parser."""
    parser = argparse.ArgumentParser(
        description="Run managed reconstruction for a former camera-angle preset."
    )
    parser.add_argument("study", choices=("reconstruct",))
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    parser.add_argument("--dataset", choices=tuple(item["name"] for item in DATASET_RUNS))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=12345)
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--resume", action="store_true")
    policy.add_argument("--overwrite-run", action="store_true")
    parser.add_argument("--no-output", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one or all presets with managed output by default."""
    args = create_parser().parse_args(argv)
    selected = [
        item for item in DATASET_RUNS if args.dataset is None or item["name"] == args.dataset
    ]
    if args.run_id is not None and len(selected) != 1:
        raise ValueError("--run-id requires exactly one --dataset selection.")
    for params in selected:
        output = None
        if not args.no_output:
            output = OutputConfig(
                workflow="reconstruction",
                name=f"angle-sweep-{params['name']}",
                root=args.output_root,
                run_id=args.run_id,
                project_root=args.project_root,
                resume=args.resume,
                overwrite=args.overwrite_run,
            )
        run_single_scenario(
            params, project_root=args.project_root, output=output, seed=args.seed
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
