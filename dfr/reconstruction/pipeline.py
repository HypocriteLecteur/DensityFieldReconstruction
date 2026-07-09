"""Composable high-level density reconstruction workflow."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from dfr.artifacts import OutputConfig, RunArtifacts
from dfr.config import CameraConfig, ReconstructionParams, RunConfig, TrainingParams
from dfr.data.base import Dataset
from dfr.data.frames import select_frame_indices
from dfr.data.spec import DatasetSpec
from dfr.model_checkpoint import build_checkpoint
from dfr.reconstruction.cameras import (
    add_bounded_projection_noise,
    build_camera_system,
)
from dfr.reconstruction.results import (
    FrameReconstruction,
    ReconstructionRequest,
    ReconstructionRun,
)
from dfr.simulation_config import SimulationConfig


def default_training_params(iterations: int = 100) -> TrainingParams:
    """Return the established one-frame optimization defaults.

    ``iterations`` becomes ``TrainingParams.lr_max_steps`` and must be
    positive. The remaining coefficients match the pre-refactor experiment
    defaults so new workflow calls reproduce the same training schedule unless
    callers supply an explicit :class:`TrainingParams`.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive.")
    return TrainingParams(
        xyz_lr_c=0.05,
        xyz_lr_final_c=0.9,
        radius_lr_c=0.05,
        radius_lr_final_c=0.9,
        weights_lr_c=0.10,
        weights_lr_final_c=0.7,
        xyz_reg=1.0,
        radius_reg=0.3,
        radius_cutoff_inv=0.5,
        lr_max_steps=iterations,
    )


def default_reconstruction_params() -> ReconstructionParams:
    """Return the established adaptive-scale and visual-hull defaults.

    These defaults target ten density modes and use the legacy visual-hull grid
    controls. Values are intentionally conservative for compatibility with the
    original one-frame experiment scripts.
    """
    return ReconstructionParams(
        targetd_num_mode=10,
        voxel_scale=0.5,
        voxel_peak_threshold=0.3,
        voxel_grid_max_size=32,
        voxel_peaks_number=20,
    )


def reconstruct(
    dataset: Dataset,
    *,
    frames,
    cameras: CameraConfig,
    scale: Optional[float] = None,
    frame_scales=None,
    training: Optional[TrainingParams] = None,
    reconstruction: Optional[ReconstructionParams] = None,
    device: Optional[str] = None,
    seed: int = 12345,
    projection_noise_std: float = 0.0,
    use_decoupled: bool = False,
    output: Optional[OutputConfig] = None,
    scenario_config: Optional[str | Path] = None,
) -> ReconstructionRun:
    """Reconstruct selected frames and optionally persist one managed run.

    Parameters
    ----------
    dataset:
        Loaded :class:`dfr.data.Dataset`; frame positions are read through
        ``positions_at_time_step`` as world-coordinate ``(agents, 3)`` arrays.
    frames:
        Frame selector accepted by :func:`dfr.data.select_frame_indices`.
    cameras:
        Camera layout and device settings. The current backend requires CUDA.
    scale, frame_scales:
        Optional fixed reconstruction scale(s) in world-coordinate units.
        Omit both to use adaptive scale selection. The two options are mutually
        exclusive.
    output:
        Optional :class:`dfr.artifacts.OutputConfig` with
        ``workflow="reconstruction"``. When omitted, the workflow is
        computation-only and writes nothing.

    Returns
    -------
    dfr.reconstruction.ReconstructionRun
        Typed per-frame reconstruction arrays on CPU plus optional managed
        artifact paths.

    Raises
    ------
    RuntimeError
        If CUDA is unavailable.
    FileNotFoundError
        If the scenario YAML required for camera calibration cannot be found.
    """
    selected_frames = select_frame_indices(dataset, frames)
    selected_cameras = cameras
    if device is not None and device != cameras.device:
        selected_cameras = replace(cameras, device=str(device))
    training = training or default_training_params()
    reconstruction_config = reconstruction or default_reconstruction_params()
    config_path = _scenario_config(dataset, scenario_config)
    request = ReconstructionRequest(
        dataset=dataset,
        frames=selected_frames,
        cameras=selected_cameras,
        training=training,
        reconstruction=reconstruction_config,
        scale=scale,
        frame_scales=(
            tuple(float(value) for value in frame_scales)
            if frame_scales is not None
            else None
        ),
        projection_noise_std=projection_noise_std,
        use_decoupled=use_decoupled,
        seed=seed,
        output=output,
        scenario_config=config_path,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Density reconstruction requires CUDA.")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    simulation = SimulationConfig(str(config_path))
    camera_system = build_camera_system(
        dataset, selected_frames, simulation, selected_cameras
    )
    artifacts = _create_artifacts(request) if output is not None else None
    results = []
    rng = np.random.default_rng(seed)
    for frame_index, frame in enumerate(selected_frames):
        result, model = _reconstruct_frame(
            request,
            frame_index,
            frame,
            simulation,
            camera_system,
            rng,
        )
        results.append(result)
        if artifacts is not None:
            _save_frame(artifacts, result, model, single=len(selected_frames) == 1)
    return ReconstructionRun(request=request, frames=tuple(results), artifacts=artifacts)


def _scenario_config(dataset: Dataset, explicit: Optional[str | Path]) -> Path:
    value = explicit or dataset.metadata.get("scenario_config")
    if value is None:
        raise ValueError(
            "Reconstruction requires a scenario/config YAML with camera settings."
        )
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Scenario config does not exist: {path}")
    return path


def _create_artifacts(request: ReconstructionRequest) -> RunArtifacts:
    if len(request.frames) == 1:
        # Preserve the transitional CLI's resolved-config schema so an
        # expensive pre-refactor single-frame run remains resumable.
        project_root = request.dataset.metadata.get("project_root")
        source_path = request.dataset.source_path
        if source_path is None:
            raise ValueError("Managed reconstruction requires a dataset source path.")
        spec = DatasetSpec(
            name=str(request.dataset.metadata.get("dataset_name") or "dataset"),
            data_path=Path(source_path),
            config_path=request.scenario_config,
            project_root=Path(project_root) if project_root else None,
        )
        run_config = RunConfig(
            dataset=spec,
            output=request.output,
            camera=request.cameras,
            training=request.training,
            reconstruction=request.reconstruction,
            seed=request.seed,
        )
        resolved_config = {
            "run": run_config,
            "frame": request.frames[0],
            "fixed_scale": request.scale_for_index(0),
        }
        if request.projection_noise_std != 0:
            resolved_config["projection_noise_std"] = request.projection_noise_std
        if request.use_decoupled:
            resolved_config["use_decoupled"] = True
    else:
        resolved_config = {"request": request}
    return RunArtifacts.create(
        request.output,
        resolved_config=resolved_config,
        device=request.cameras.device,
        metadata={"entrypoint": "dfr.reconstruct"},
    )


def _reconstruct_frame(request, frame_index, frame, simulation, camera_system, rng):
    # Lazy imports keep data-only result/config APIs usable without loading the
    # compiled reconstruction stack.
    from dfr.density_field_reconstructor import DensityReconstructor
    from dfr.utils import calculate_gmm_dissimilarity

    positions = request.dataset.positions_at_time_step(frame).astype(
        np.float32, copy=False
    )
    camera_poses, projections, _, visibility_masks = camera_system.simulate_vision(
        positions,
        renderer="projection_only",
        is_auto_aim=request.cameras.layout == "encircling",
    )
    projections = add_bounded_projection_noise(
        projections,
        camera_system,
        request.projection_noise_std,
        rng,
    )
    fixed_scale = request.scale_for_index(frame_index)
    reconstructor = DensityReconstructor(
        max_iter=request.training.lr_max_steps,
        W=simulation.W,
        H=simulation.H,
        far_clip=simulation.far_clip,
        use_decoupled=request.use_decoupled,
    )
    models, scale_spaces = reconstructor.process_frame(
        camera_system,
        point_sets=projections,
        is_adaptive_scale=fixed_scale is None,
        scale=fixed_scale,
        positions=positions,
        train_params=request.training,
        reconstruction_params=request.reconstruction,
    )
    model = models[0]
    visible = np.logical_and.reduce(visibility_masks)
    dissimilarity = None
    if np.any(visible):
        dissimilarity = float(
            calculate_gmm_dissimilarity(
                positions[visible],
                reconstructor.scale,
                model._xyz,
                model._weights,
                model._radius,
            )
        )
    result = FrameReconstruction(
        dataset_name=str(request.dataset.metadata.get("dataset_name") or "dataset"),
        frame=frame,
        positions=positions,
        means=model._xyz.detach().cpu().numpy(),
        radii=model._radius.detach().cpu().numpy(),
        weights=model._weights.detach().cpu().numpy(),
        camera_poses=np.asarray(camera_poses),
        projections=tuple(projections),
        visible_mask=visible,
        scale=float(reconstructor.scale),
        mean_training_loss=(
            None if model.mean_loss is None else float(model.mean_loss)
        ),
        density_dissimilarity=dissimilarity,
        time_ms=reconstructor.time_metrics,
        scale_space_shapes=tuple(tuple(space.shape) for space in scale_spaces),
    )
    return result, model


def _save_frame(
    artifacts: RunArtifacts,
    result: FrameReconstruction,
    model,
    *,
    single: bool,
) -> None:
    prefix = "" if single else f"frame_{result.frame:06d}/"
    projections = {
        f"projection_{index}": projection
        for index, projection in enumerate(result.projections)
    }
    artifacts.save_npz(
        f"{prefix}reconstruction.npz",
        overwrite=artifacts.output.resume,
        positions=result.positions,
        means=result.means,
        radii=result.radii,
        weights=result.weights,
        camera_poses=result.camera_poses,
        visible_mask=result.visible_mask,
        scale=np.asarray(result.scale),
        **projections,
    )
    artifacts.save_checkpoint(
        f"{prefix}final_model.pth",
        build_checkpoint(model),
        overwrite=artifacts.output.resume,
    )
    artifacts.save_json(
        f"{prefix}summary.json",
        result.summary(),
        category="metrics",
        overwrite=artifacts.output.resume,
    )
