"""Configurable dataset-scenario reconstruction runners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from dfr.artifacts import OutputConfig
from dfr.config import CameraConfig, ReconstructionParams, TrainingParams
from dfr.data.loading import load_dataset
from dfr.reconstruction.pipeline import (
    default_reconstruction_params,
    default_training_params,
    reconstruct,
)
from dfr.reconstruction.results import ReconstructionRun


@dataclass(frozen=True, slots=True)
class ScenarioRunSpec:
    """Declarative controls for one named-scenario reconstruction.

    ``stop=None`` selects through the final dataset frame. Ground-truth scale
    caches retain the historical contract: values correspond in order to the
    selected frames rather than to absolute dataset frame numbers.
    """

    dataset: str
    start: int = 0
    stop: Optional[int] = None
    step: int = 1
    cameras: CameraConfig = CameraConfig.encircling()
    training: Optional[TrainingParams] = None
    reconstruction: Optional[ReconstructionParams] = None
    use_ground_truth_scales: bool = True
    scale_cache_name: str = "reconstruction_scale.npz"
    scale_cache_key: str = "scales_gt"
    projection_noise_std: float = 0.0
    use_decoupled: bool = False
    seed: int = 12345
    output: Optional[OutputConfig] = None

    def __post_init__(self) -> None:
        if not self.dataset.strip():
            raise ValueError("ScenarioRunSpec.dataset must not be empty.")
        if self.start < 0:
            raise ValueError("ScenarioRunSpec.start must be non-negative.")
        if self.stop is not None and self.stop < 0:
            raise ValueError("ScenarioRunSpec.stop must be non-negative.")
        if self.step < 1:
            raise ValueError("ScenarioRunSpec.step must be positive.")
        if self.projection_noise_std < 0:
            raise ValueError("projection_noise_std must be non-negative.")
        if self.seed < 0:
            raise ValueError("ScenarioRunSpec.seed must be non-negative.")
        if not self.scale_cache_name or Path(self.scale_cache_name).name != self.scale_cache_name:
            raise ValueError("scale_cache_name must be one filename.")
        if not self.scale_cache_key:
            raise ValueError("scale_cache_key must not be empty.")
        if self.output is not None and self.output.workflow != "reconstruction":
            raise ValueError("Scenario output workflow must be 'reconstruction'.")

    def to_dict(self) -> dict:
        """Return a serialization-friendly representation of this run."""
        return {
            "dataset": self.dataset,
            "start": self.start,
            "stop": self.stop,
            "step": self.step,
            "cameras": self.cameras.to_dict(),
            "training": self.training.to_dict() if self.training else None,
            "reconstruction": (
                self.reconstruction.to_dict() if self.reconstruction else None
            ),
            "use_ground_truth_scales": self.use_ground_truth_scales,
            "scale_cache_name": self.scale_cache_name,
            "scale_cache_key": self.scale_cache_key,
            "projection_noise_std": self.projection_noise_std,
            "use_decoupled": self.use_decoupled,
            "seed": self.seed,
            "output": self.output.to_dict() if self.output else None,
        }


def run_scenario(
    spec: ScenarioRunSpec,
    *,
    project_root: str | Path,
) -> ReconstructionRun:
    """Load and reconstruct one scenario according to ``spec``."""
    root = Path(project_root).expanduser().resolve()
    dataset = load_dataset(spec.dataset, project_root=root)
    stop = len(dataset) if spec.stop is None else min(spec.stop, len(dataset))
    frames = tuple(range(spec.start, stop, spec.step))
    if not frames:
        raise ValueError(
            f"Scenario {spec.dataset!r} selects no frames: "
            f"start={spec.start}, stop={stop}, step={spec.step}."
        )
    frame_scales = (
        _load_frame_scales(dataset, frames, spec)
        if spec.use_ground_truth_scales
        else None
    )
    run = reconstruct(
        dataset,
        frames=frames,
        cameras=spec.cameras,
        frame_scales=frame_scales,
        training=spec.training or default_training_params(),
        reconstruction=spec.reconstruction or default_reconstruction_params(),
        seed=spec.seed,
        projection_noise_std=spec.projection_noise_std,
        use_decoupled=spec.use_decoupled,
        output=spec.output,
    )
    _save_statistics(run)
    return run


def run_scenarios(
    specs: Iterable[ScenarioRunSpec],
    *,
    project_root: str | Path,
) -> tuple[ReconstructionRun, ...]:
    """Run a sequence of scenario specifications in declaration order."""
    selected = tuple(specs)
    if not selected:
        raise ValueError("At least one scenario specification is required.")
    return tuple(run_scenario(spec, project_root=project_root) for spec in selected)


def _load_frame_scales(dataset, frames, spec: ScenarioRunSpec) -> tuple[float, ...]:
    config_path = dataset.metadata.get("scenario_config")
    if config_path is None:
        raise ValueError("Ground-truth scales require scenario-config metadata.")
    cache_path = Path(config_path).resolve().parent / spec.scale_cache_name
    if not cache_path.is_file():
        raise FileNotFoundError(f"Ground-truth scale cache does not exist: {cache_path}")
    with np.load(cache_path, allow_pickle=False) as cache:
        if spec.scale_cache_key not in cache:
            raise KeyError(
                f"Scale cache {cache_path} has no key {spec.scale_cache_key!r}."
            )
        values = np.asarray(cache[spec.scale_cache_key], dtype=float).reshape(-1)
    if len(values) < len(frames):
        raise ValueError(
            f"Ground-truth scale cache has {len(values)} values for "
            f"{len(frames)} selected frames."
        )
    selected = tuple(float(value) for value in values[: len(frames)])
    if any(not np.isfinite(value) or value <= 0 for value in selected):
        raise ValueError("Ground-truth scales must be finite and positive.")
    return selected


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
