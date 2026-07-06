"""Reconstruct one configured dataset frame into a managed output run.

This is a thin transitional CLI over the current reconstruction classes. It
keeps all generated data under ``outputs/reconstruction/<run-id>/`` and is
intended for quick experiments while the higher-level Phase 5 workflow API is
developed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dfr import (
    OutputConfig,
    RunArtifacts,
    load_dataset,
    resolve_dataset,
    select_frame_indices,
)
from dfr.camera_state import CameraState
from dfr.camera_system import MultiCameraSystem
from dfr.config import ReconstructionParams, TrainingParams
from dfr.density_field_reconstructor import DensityReconstructor
from dfr.model_checkpoint import build_checkpoint
from dfr.simulation_config import SimulationConfig
from dfr.utils import calculate_gmm_dissimilarity, generate_encircling_cameras


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


def _camera_system(dataset, frame: int, config, count: int) -> MultiCameraSystem:
    # The established two-camera configuration uses adjacent positions from a
    # four-camera ring, avoiding the degenerate opposite-camera pair.
    generated_count = 4 if count == 2 else count
    camera_positions, _ = generate_encircling_cameras(
        dataset,
        [frame],
        config.intrinsics_params,
        config.H,
        config.W,
        cam_num=generated_count,
        padding=1,
    )
    camera_positions = camera_positions[:count]
    identity_quaternions = np.tile(
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (count, 1)
    )
    camera_poses = np.hstack((camera_positions, identity_quaternions)).astype(
        np.float32
    )
    return MultiCameraSystem.create_homogeneous_system(
        state_class=CameraState,
        intrinsics=config.intrinsics_params,
        H=config.H,
        W=config.W,
        poses_or_RTs=camera_poses,
        near_clip=config.near_clip,
        far_clip=config.far_clip,
        size=config.size,
        device="cuda",
    )


def run(args: argparse.Namespace) -> RunArtifacts:
    """Execute one frame and return its managed artifact paths."""
    _validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("One-frame reconstruction requires CUDA.")

    project_root = args.project_root or Path(__file__).resolve().parents[1]
    spec = resolve_dataset(args.dataset, project_root=project_root)
    if spec.config_path is None:
        raise ValueError(
            "Reconstruction needs a scenario/config YAML with camera settings; "
            f"an explicit data file is insufficient: {spec.data_path}"
        )
    dataset = load_dataset(spec)
    frame = select_frame_indices(dataset, args.frame)[0]
    simulation_config = SimulationConfig(str(spec.config_path))
    positions = dataset.positions_at_time_step(frame).astype(np.float32, copy=False)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cameras = _camera_system(dataset, frame, simulation_config, args.camera_count)
    camera_poses, projections, _, visibility_masks = cameras.simulate_vision(
        positions,
        renderer="projection_only",
        is_auto_aim=True,
    )

    train_params = TrainingParams(
        xyz_lr_c=0.05,
        xyz_lr_final_c=0.9,
        radius_lr_c=0.05,
        radius_lr_final_c=0.9,
        weights_lr_c=0.10,
        weights_lr_final_c=0.7,
        xyz_reg=1.0,
        radius_reg=0.3,
        radius_cutoff_inv=0.5,
        lr_max_steps=args.iterations,
    )
    reconstruction_params = ReconstructionParams(
        targetd_num_mode=args.target_mode_count,
        voxel_scale=args.voxel_scale,
        voxel_peak_threshold=args.voxel_peak_threshold,
        voxel_grid_max_size=args.voxel_grid_max_size,
        voxel_peaks_number=args.voxel_peaks_number,
    )
    artifacts = RunArtifacts.create(
        OutputConfig(
            workflow="reconstruction",
            name=f"{spec.name} frame {frame}",
            root=args.output_root,
            run_id=args.run_id,
            project_root=project_root,
            resume=args.resume,
            overwrite=args.overwrite_run,
        ),
        resolved_config={
            "dataset": spec,
            "frame": frame,
            "camera_count": args.camera_count,
            "fixed_scale": args.scale,
            "seed": args.seed,
            "training": train_params,
            "reconstruction": reconstruction_params,
        },
        device="cuda",
        metadata={"entrypoint": "experiments.reconstruct_one_frame"},
    )

    reconstructor = DensityReconstructor(
        max_iter=args.iterations,
        W=simulation_config.W,
        H=simulation_config.H,
        far_clip=simulation_config.far_clip,
    )
    models, scale_spaces = reconstructor.process_frame(
        cameras,
        point_sets=projections,
        is_adaptive_scale=args.scale is None,
        scale=args.scale,
        positions=positions,
        train_params=train_params,
        reconstruction_params=reconstruction_params,
    )
    model = models[0]
    visible = np.logical_and.reduce(visibility_masks)
    dissimilarity = calculate_gmm_dissimilarity(
        positions[visible],
        reconstructor.scale,
        model._xyz,
        model._weights,
        model._radius,
    )

    projection_arrays = {
        f"projection_{index}": projection
        for index, projection in enumerate(projections)
    }
    artifacts.save_npz(
        "reconstruction.npz",
        overwrite=args.resume,
        positions=positions,
        means=model._xyz.detach().cpu().numpy(),
        radii=model._radius.detach().cpu().numpy(),
        weights=model._weights.detach().cpu().numpy(),
        camera_poses=np.asarray(camera_poses),
        scale=np.asarray(reconstructor.scale),
        **projection_arrays,
    )
    artifacts.save_checkpoint(
        "final_model.pth", build_checkpoint(model), overwrite=args.resume
    )
    artifacts.save_json(
        "summary.json",
        {
            "dataset": spec.name,
            "frame": frame,
            "agent_count": len(positions),
            "visible_agent_count": int(np.count_nonzero(visible)),
            "gaussian_count": model.num_gaussians,
            "scale": reconstructor.scale,
            "mean_training_loss": model.mean_loss,
            "density_dissimilarity": dissimilarity,
            "time_ms": reconstructor.time_metrics,
            "scale_space_shapes": [list(space.shape) for space in scale_spaces],
        },
        category="metrics",
        overwrite=args.resume,
    )
    return artifacts


def main() -> None:
    artifacts = run(build_parser().parse_args())
    print(f"Reconstruction saved to: {artifacts.run_dir}")


if __name__ == "__main__":
    main()
