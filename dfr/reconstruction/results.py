"""Typed requests and data-only results for reconstruction workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from dfr.artifacts import OutputConfig, RunArtifacts
from dfr.config import CameraConfig, ReconstructionParams, TrainingParams
from dfr.data.base import Dataset


@dataclass(frozen=True, slots=True)
class ReconstructionRequest:
    """Fully resolved controls for one dataset reconstruction run."""

    dataset: Dataset
    frames: tuple[int, ...]
    cameras: CameraConfig
    training: TrainingParams
    reconstruction: ReconstructionParams
    scale: Optional[float] = None
    frame_scales: Optional[tuple[float, ...]] = None
    projection_noise_std: float = 0.0
    use_decoupled: bool = False
    seed: int = 12345
    output: Optional[OutputConfig] = None
    scenario_config: Optional[Path] = None

    def __post_init__(self) -> None:
        frames = tuple(int(frame) for frame in self.frames)
        if not frames:
            raise ValueError("ReconstructionRequest.frames must not be empty.")
        object.__setattr__(self, "frames", frames)
        if self.scale is not None and self.scale <= 0:
            raise ValueError("ReconstructionRequest.scale must be positive.")
        if self.frame_scales is not None:
            values = tuple(float(value) for value in self.frame_scales)
            if self.scale is not None:
                raise ValueError("scale and frame_scales are mutually exclusive.")
            if len(values) != len(frames) or any(value <= 0 for value in values):
                raise ValueError(
                    "frame_scales must contain one positive scale per frame."
                )
            object.__setattr__(self, "frame_scales", values)
        if self.projection_noise_std < 0:
            raise ValueError("projection_noise_std must be non-negative.")
        if self.seed < 0:
            raise ValueError("ReconstructionRequest.seed must be non-negative.")
        if self.output is not None and self.output.workflow != "reconstruction":
            raise ValueError("Reconstruction output workflow must be 'reconstruction'.")
        if self.cameras.device != "cuda":
            raise ValueError("The current reconstruction backend requires device='cuda'.")
        if self.training.lr_max_steps < 1:
            raise ValueError("Training iterations must be positive.")
        params = self.reconstruction
        if params.target_mode_count < 1 or params.voxel_peaks_number < 1:
            raise ValueError("Mode and voxel-peak counts must be positive.")
        if params.voxel_scale <= 0 or params.voxel_grid_max_size < 2:
            raise ValueError("Voxel scale/grid controls are invalid.")
        if not 0 <= params.voxel_peak_threshold <= 1:
            raise ValueError("Voxel peak threshold must be between zero and one.")
        if self.scenario_config is not None:
            object.__setattr__(
                self, "scenario_config", Path(self.scenario_config).expanduser().resolve()
            )

    def to_dict(self) -> dict[str, Any]:
        """Return provenance-safe controls without serializing dataset arrays."""
        return {
            "dataset": {
                "name": self.dataset.metadata.get("dataset_name"),
                "source_path": (
                    str(self.dataset.source_path) if self.dataset.source_path else None
                ),
                "scenario_config": (
                    str(self.scenario_config) if self.scenario_config else None
                ),
            },
            "frames": list(self.frames),
            "cameras": self.cameras.to_dict(),
            "training": self.training.to_dict(),
            "reconstruction": self.reconstruction.to_dict(),
            "scale": self.scale,
            "frame_scales": (
                list(self.frame_scales) if self.frame_scales is not None else None
            ),
            "projection_noise_std": self.projection_noise_std,
            "use_decoupled": self.use_decoupled,
            "seed": self.seed,
        }

    def scale_for_index(self, index: int) -> Optional[float]:
        """Return the fixed scale for one selected frame, or None for adaptive."""
        return self.frame_scales[index] if self.frame_scales is not None else self.scale


@dataclass
class FrameReconstruction:
    """CPU arrays and metrics produced by reconstructing one dataset frame."""

    dataset_name: str
    frame: int
    positions: np.ndarray
    means: np.ndarray
    radii: np.ndarray
    weights: np.ndarray
    camera_poses: np.ndarray
    projections: tuple[np.ndarray, ...]
    visible_mask: np.ndarray
    scale: float
    mean_training_loss: Optional[float]
    density_dissimilarity: Optional[float]
    time_ms: Mapping[str, float]
    scale_space_shapes: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.dataset_name:
            raise ValueError("dataset_name must not be empty.")
        self.positions = _points("positions", self.positions)
        self.means = _points("means", self.means)
        self.radii = _column("radii", self.radii, len(self.means))
        self.weights = _column("weights", self.weights, len(self.means))
        self.camera_poses = np.asarray(self.camera_poses, dtype=np.float32)
        if self.camera_poses.ndim != 2 or self.camera_poses.shape[1] != 7:
            raise ValueError("camera_poses must have shape (cameras, 7).")
        self.projections = tuple(
            np.asarray(projection, dtype=np.float32) for projection in self.projections
        )
        if len(self.projections) != len(self.camera_poses):
            raise ValueError("projections must contain one array per camera.")
        if any(value.ndim != 2 or value.shape[1] != 2 for value in self.projections):
            raise ValueError("Every projection must have shape (visible agents, 2).")
        self.visible_mask = np.asarray(self.visible_mask, dtype=bool)
        if self.visible_mask.shape != (len(self.positions),):
            raise ValueError("visible_mask must align with positions.")
        if self.scale <= 0 or not np.isfinite(self.scale):
            raise ValueError("scale must be positive and finite.")
        self.time_ms = {str(key): float(value) for key, value in self.time_ms.items()}
        self.scale_space_shapes = tuple(
            tuple(int(value) for value in shape) for shape in self.scale_space_shapes
        )

    @property
    def gaussian_count(self) -> int:
        return len(self.means)

    def summary(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "frame": self.frame,
            "agent_count": len(self.positions),
            "visible_agent_count": int(np.count_nonzero(self.visible_mask)),
            "gaussian_count": self.gaussian_count,
            "scale": self.scale,
            "mean_training_loss": self.mean_training_loss,
            "density_dissimilarity": self.density_dissimilarity,
            "time_ms": dict(self.time_ms),
            "scale_space_shapes": [list(shape) for shape in self.scale_space_shapes],
        }


@dataclass(frozen=True, slots=True)
class ReconstructionRun:
    """Typed collection of reconstructed frames and optional managed artifacts."""

    request: ReconstructionRequest
    frames: tuple[FrameReconstruction, ...]
    artifacts: Optional[RunArtifacts] = None

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("ReconstructionRun.frames must not be empty.")
        actual = tuple(result.frame for result in self.frames)
        if actual != self.request.frames:
            raise ValueError("Frame results must align with the requested frame order.")

    @property
    def run_dir(self) -> Optional[Path]:
        return self.artifacts.run_dir if self.artifacts is not None else None


def _points(name: str, values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (points, 3).")
    return array


def _column(name: str, values, length: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    if len(array) != length:
        raise ValueError(f"{name} must align with reconstructed means.")
    return array
