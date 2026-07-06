"""Reconstruct one configured dataset frame into a managed output run.

This thin CLI resolves arguments into the public :func:`dfr.reconstruct`
workflow. Generated data stays under ``outputs/reconstruction/<run-id>/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dfr import (
    CameraConfig,
    OutputConfig,
    RunArtifacts,
    load_dataset,
    reconstruct,
    resolve_dataset,
    select_frame_indices,
)
from dfr.config import ReconstructionParams
from dfr.reconstruction.pipeline import default_training_params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Scenario name or config YAML.")
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--camera-count", type=int, default=2)
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Fixed world scale; omit to use adaptive scale selection.",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--target-mode-count", type=int, default=10)
    parser.add_argument("--voxel-scale", type=float, default=0.5)
    parser.add_argument("--voxel-peak-threshold", type=float, default=0.3)
    parser.add_argument("--voxel-grid-max-size", type=int, default=32)
    parser.add_argument("--voxel-peaks-number", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-id", default=None)
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--resume", action="store_true")
    policy.add_argument("--overwrite-run", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.camera_count < 2:
        raise ValueError("camera-count must be at least 2.")
    if args.iterations < 1:
        raise ValueError("iterations must be positive.")
    if args.scale is not None and args.scale <= 0:
        raise ValueError("scale must be positive when supplied.")
    if args.target_mode_count < 1 or args.voxel_peaks_number < 1:
        raise ValueError("target-mode-count and voxel-peaks-number must be positive.")
    if args.voxel_scale <= 0 or args.voxel_grid_max_size < 2:
        raise ValueError("voxel settings must be positive and grid size at least 2.")
    if not 0 <= args.voxel_peak_threshold <= 1:
        raise ValueError("voxel-peak-threshold must be between 0 and 1.")
    if args.resume and args.run_id is None:
        raise ValueError("--resume requires an explicit --run-id.")


def run(args: argparse.Namespace) -> RunArtifacts:
    """Execute one frame and return its managed artifact paths."""
    _validate_args(args)
    project_root = args.project_root or Path(__file__).resolve().parents[1]
    spec = resolve_dataset(args.dataset, project_root=project_root)
    if spec.config_path is None:
        raise ValueError(
            "Reconstruction needs a scenario/config YAML with camera settings; "
            f"an explicit data file is insufficient: {spec.data_path}"
        )
    dataset = load_dataset(spec)
    frame = select_frame_indices(dataset, args.frame)[0]
    result = reconstruct(
        dataset,
        frames=frame,
        cameras=CameraConfig.encircling(count=args.camera_count, device="cuda"),
        scale=args.scale,
        training=default_training_params(args.iterations),
        reconstruction=ReconstructionParams(
            targetd_num_mode=args.target_mode_count,
            voxel_scale=args.voxel_scale,
            voxel_peak_threshold=args.voxel_peak_threshold,
            voxel_grid_max_size=args.voxel_grid_max_size,
            voxel_peaks_number=args.voxel_peaks_number,
        ),
        seed=args.seed,
        output=OutputConfig(
            workflow="reconstruction",
            name=f"{spec.name} frame {frame}",
            root=args.output_root,
            run_id=args.run_id,
            project_root=project_root,
            resume=args.resume,
            overwrite=args.overwrite_run,
        ),
        scenario_config=spec.config_path,
    )
    if result.artifacts is None:  # OutputConfig above makes this unreachable.
        raise RuntimeError("Managed reconstruction did not create artifacts.")
    return result.artifacts


def main() -> None:
    artifacts = run(build_parser().parse_args())
    print(f"Reconstruction saved to: {artifacts.run_dir}")


if __name__ == "__main__":
    main()
