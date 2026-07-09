"""Reconstruction workflow for measured or otherwise external observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import torch

from dfr.artifacts import OutputConfig, RunArtifacts
from dfr.config import CameraConfig, ReconstructionParams, TrainingParams
from dfr.reconstruction.pipeline import (
    _save_frame,
    default_reconstruction_params,
    default_training_params,
)
from dfr.reconstruction.results import (
    FrameReconstruction,
    ReconstructionRequest,
    ReconstructionRun,
)


@dataclass(frozen=True, slots=True)
class ExternalObservationFrame:
    """One externally observed frame ready for DFR reconstruction.

    This is the common boundary for measured flock detections, UE4 thresholded
    images, or any other source that already has 2-D detections/projections.
    Unlike :func:`dfr.reconstruct`, these projections are not produced by the
    simulator. The caller supplies the camera system that should interpret them.

    ``positions`` must be a world-coordinate ``(agents, 3)`` array.
    ``projections`` must contain one ``(visible_agents, 2)`` pixel-coordinate
    array per camera. ``camera_poses`` are read from ``camera_system`` when
    available, otherwise callers must supply ``(cameras, 7)`` poses in
    ``(x, y, z, qx, qy, qz, qw)`` order. ``visible_mask`` defaults to all true.
    """

    dataset_name: str
    frame: int
    positions: np.ndarray
    projections: tuple[np.ndarray, ...] | list[np.ndarray]
    camera_system: Any = field(repr=False, compare=False)
    camera_poses: Optional[np.ndarray] = None
    visible_mask: Optional[np.ndarray] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = self.dataset_name.strip()
        if not name:
            raise ValueError("ExternalObservationFrame.dataset_name must not be empty.")
        object.__setattr__(self, "dataset_name", name)
        object.__setattr__(self, "frame", int(self.frame))

        positions = _points("positions", self.positions)
        object.__setattr__(self, "positions", positions)

        projections = tuple(_projection(value) for value in self.projections)
        if not projections:
            raise ValueError("ExternalObservationFrame.projections must not be empty.")
        object.__setattr__(self, "projections", projections)

        camera_count = _camera_count(self.camera_system)
        if camera_count is not None and camera_count != len(projections):
            raise ValueError(
                "ExternalObservationFrame.projections must contain one array per "
                f"camera: got {len(projections)} for {camera_count} cameras."
            )

        camera_poses = (
            _camera_poses_from_system(self.camera_system)
            if self.camera_poses is None
            else np.asarray(self.camera_poses, dtype=np.float32)
        )
        if camera_poses.ndim != 2 or camera_poses.shape != (len(projections), 7):
            raise ValueError("camera_poses must have shape (cameras, 7).")
        object.__setattr__(self, "camera_poses", camera_poses)

        if self.visible_mask is None:
            visible = np.ones(len(positions), dtype=bool)
        else:
            visible = np.asarray(self.visible_mask, dtype=bool)
        if visible.shape != (len(positions),):
            raise ValueError("visible_mask must align with positions.")
        object.__setattr__(self, "visible_mask", visible)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return serializable provenance without embedding camera objects."""
        return {
            "dataset_name": self.dataset_name,
            "frame": self.frame,
            "agent_count": int(len(self.positions)),
            "visible_agent_count": int(np.count_nonzero(self.visible_mask)),
            "projection_counts": [int(len(value)) for value in self.projections],
            "camera_poses": self.camera_poses,
            "metadata": dict(self.metadata),
        }


def reconstruct_observations(
    observations: Iterable[ExternalObservationFrame],
    *,
    scale: Optional[float] = None,
    frame_scales: Optional[Iterable[float]] = None,
    training: Optional[TrainingParams] = None,
    reconstruction: Optional[ReconstructionParams] = None,
    seed: int = 12345,
    use_decoupled: bool = False,
    output: Optional[OutputConfig] = None,
) -> ReconstructionRun:
    """Reconstruct externally supplied projections into a typed run.

    ``observations`` may use either one static camera system or a different
    camera system per frame. This supports both measured flock footage and UE4
    renders where camera poses vary over time. ``scale`` and ``frame_scales``
    are fixed world-coordinate density scales; omit both to use adaptive scale
    selection. CUDA is required by the active reconstruction backend. No files
    are written unless ``output`` is supplied, in which case arrays,
    checkpoints, per-frame summaries, and aggregate statistics are saved under
    the managed reconstruction run directory.
    """

    selected = tuple(observations)
    if not selected:
        raise ValueError("At least one external observation frame is required.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    request = ReconstructionRequest(
        dataset=_ObservationDataset(selected),
        frames=tuple(frame.frame for frame in selected),
        cameras=CameraConfig.explicit(selected[0].camera_poses),
        training=training or default_training_params(),
        reconstruction=reconstruction or default_reconstruction_params(),
        scale=scale,
        frame_scales=(
            tuple(float(value) for value in frame_scales)
            if frame_scales is not None
            else None
        ),
        projection_noise_std=0.0,
        use_decoupled=use_decoupled,
        seed=seed,
        output=output,
        scenario_config=None,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Density reconstruction requires CUDA.")

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    artifacts = _create_artifacts(request, selected) if output is not None else None
    results: list[FrameReconstruction] = []
    for frame_index, observation in enumerate(selected):
        result, model = _reconstruct_observation(request, frame_index, observation)
        results.append(result)
        if artifacts is not None:
            _save_frame(artifacts, result, model, single=len(selected) == 1)
    run = ReconstructionRun(request=request, frames=tuple(results), artifacts=artifacts)
    _save_statistics(run)
    return run


class _ObservationDataset:
    def __init__(self, observations: tuple[ExternalObservationFrame, ...]) -> None:
        self._observations = observations
        self._by_frame = {item.frame: item.positions for item in observations}
        self._trajectories = _pad_positions([item.positions for item in observations])
        names = sorted({item.dataset_name for item in observations})
        self._metadata = {
            "dataset_name": names[0] if len(names) == 1 else "external-observations",
            "observation_source": "external",
            "dataset_names": names,
        }

    @property
    def trajectories(self) -> np.ndarray:
        return self._trajectories

    @property
    def velocities(self) -> np.ndarray:
        return np.empty((0, 0, 3), dtype=np.float32)

    @property
    def has_velocities(self) -> bool:
        return False

    @property
    def timestamps(self) -> None:
        return None

    @property
    def has_timestamps(self) -> bool:
        return False

    @property
    def source_path(self) -> None:
        return None

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    @property
    def coordinate_system(self) -> None:
        return None

    @property
    def ground_truth_positions(self) -> np.ndarray:
        return self._trajectories

    @property
    def frame_count(self) -> int:
        return len(self._observations)

    def __len__(self) -> int:
        return len(self._observations)

    def normalize_time_step(self, time_step: int) -> int:
        frame = int(time_step)
        if frame not in self._by_frame:
            raise IndexError(f"Observation frame is not available: {frame}")
        return frame

    def positions_at_time_step(self, time_step: int) -> np.ndarray:
        return self._by_frame[self.normalize_time_step(time_step)]

    def positions_at_time_step_mask(
        self, time_step: int
    ) -> tuple[np.ndarray, np.ndarray]:
        positions = self.positions_at_time_step(time_step)
        return positions, np.ones(len(positions), dtype=bool)


def _create_artifacts(
    request: ReconstructionRequest,
    observations: tuple[ExternalObservationFrame, ...],
) -> RunArtifacts:
    return RunArtifacts.create(
        request.output,
        resolved_config={
            "request": request,
            "external_observations": [observation.to_dict() for observation in observations],
        },
        device=request.cameras.device,
        metadata={
            "entrypoint": "dfr.reconstruct_observations",
            "observation_source": "external",
        },
    )


def _reconstruct_observation(
    request: ReconstructionRequest,
    frame_index: int,
    observation: ExternalObservationFrame,
):
    from dfr.density_field_reconstructor import DensityReconstructor
    from dfr.utils import calculate_gmm_dissimilarity

    fixed_scale = request.scale_for_index(frame_index)
    reconstructor = DensityReconstructor(
        max_iter=request.training.lr_max_steps,
        use_decoupled=request.use_decoupled,
        **_camera_render_kwargs(observation.camera_system),
    )
    models, scale_spaces = reconstructor.process_frame(
        observation.camera_system,
        point_sets=list(observation.projections),
        positions=observation.positions,
        initGMM=None,
        is_adaptive_scale=fixed_scale is None,
        scale=fixed_scale,
        is_store_intermediate=False,
        is_log=False,
        train_params=request.training,
        reconstruction_params=request.reconstruction,
    )
    model = models[0]
    dissimilarity = None
    if np.any(observation.visible_mask):
        dissimilarity = float(
            calculate_gmm_dissimilarity(
                observation.positions[observation.visible_mask],
                reconstructor.scale,
                model._xyz,
                model._weights,
                model._radius,
            )
        )
    result = FrameReconstruction(
        dataset_name=observation.dataset_name,
        frame=observation.frame,
        positions=observation.positions,
        means=model._xyz.detach().cpu().numpy(),
        radii=model._radius.detach().cpu().numpy(),
        weights=model._weights.detach().cpu().numpy(),
        camera_poses=observation.camera_poses,
        projections=observation.projections,
        visible_mask=observation.visible_mask,
        scale=float(reconstructor.scale),
        mean_training_loss=(
            None if model.mean_loss is None else float(model.mean_loss)
        ),
        density_dissimilarity=dissimilarity,
        time_ms=reconstructor.time_metrics,
        scale_space_shapes=tuple(tuple(space.shape) for space in scale_spaces),
    )
    return result, model


def _save_statistics(run: ReconstructionRun) -> None:
    if run.artifacts is None:
        return
    timing_names = sorted({key for frame in run.frames for key in frame.time_ms})
    run.artifacts.save_npz(
        "statistics.npz",
        overwrite=run.artifacts.output.resume,
        **{
            key: np.asarray([frame.time_ms.get(key, np.nan) for frame in run.frames])
            for key in timing_names
        },
        final_training_loss=np.asarray(
            [frame.mean_training_loss for frame in run.frames], dtype=float
        ),
        final_density_field_loss=np.asarray(
            [frame.density_dissimilarity for frame in run.frames], dtype=float
        ),
        final_gmm_num=np.asarray([frame.gaussian_count for frame in run.frames]),
        scale=np.asarray([frame.scale for frame in run.frames]),
    )


def _camera_count(camera_system: Any) -> Optional[int]:
    cameras = getattr(camera_system, "cameras", None)
    return None if cameras is None else len(cameras)


def _camera_render_kwargs(camera_system: Any) -> dict[str, Any]:
    cameras = getattr(camera_system, "cameras", None)
    if not cameras:
        return {}
    state = getattr(cameras[0], "state", None)
    if state is None:
        return {}
    kwargs = {}
    for name in ("W", "H", "far_clip"):
        value = getattr(state, name, None)
        if value is not None:
            kwargs[name] = value
    return kwargs


def _camera_poses_from_system(camera_system: Any) -> np.ndarray:
    cameras = getattr(camera_system, "cameras", None)
    if not cameras:
        raise ValueError(
            "camera_poses must be supplied when camera_system has no cameras."
        )
    poses = []
    for index, camera in enumerate(cameras):
        state = getattr(camera, "state", None)
        pose = getattr(state, "pose_np", None)
        if pose is None:
            raise ValueError(
                "camera_poses must be supplied when camera state has no pose_np "
                f"(camera {index})."
            )
        poses.append(np.asarray(pose, dtype=np.float32))
    return np.asarray(poses, dtype=np.float32)


def _points(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (points, 3).")
    return array


def _projection(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("Every projection must have shape (visible agents, 2).")
    return array


def _pad_positions(frames: list[np.ndarray]) -> np.ndarray:
    max_count = max(len(frame) for frame in frames)
    padded = np.full((len(frames), max_count, 3), np.nan, dtype=np.float32)
    for index, frame in enumerate(frames):
        padded[index, : len(frame)] = frame
    return padded
