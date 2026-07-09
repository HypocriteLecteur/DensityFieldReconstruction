"""Typed per-frame and aggregate density-evaluation results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from dfr.artifacts import RunArtifacts
from dfr.config import EvaluationConfig


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Density-overlap masses and derived metrics for one or more frames.

    Masses are non-negative voxel-integral values in the same density units as
    :func:`dfr.evaluation.compute_density_overlap_masses`. Derived metrics are
    dimensionless ratios: ``recall``, ``miss``, ``hallucination``, and
    ``dmota``. Aggregate summaries are formed by summing masses first and then
    recomputing the ratios.
    """

    true_positive_mass: float
    false_positive_mass: float
    false_negative_mass: float
    ground_truth_mass: float
    predicted_mass: float

    def __post_init__(self) -> None:
        values = (
            self.true_positive_mass,
            self.false_positive_mass,
            self.false_negative_mass,
            self.ground_truth_mass,
            self.predicted_mass,
        )
        if any(not np.isfinite(value) or value < 0 for value in values):
            raise ValueError("Evaluation masses must be finite and non-negative.")
        if self.ground_truth_mass <= 0:
            raise ValueError("ground_truth_mass must be positive.")

    @property
    def recall(self) -> float:
        return self.true_positive_mass / self.ground_truth_mass

    @property
    def miss(self) -> float:
        return self.false_negative_mass / self.ground_truth_mass

    @property
    def hallucination(self) -> float:
        return (
            self.false_positive_mass / self.predicted_mass
            if self.predicted_mass > 0
            else 0.0
        )

    @property
    def dmota(self) -> float:
        return 1.0 - (
            self.false_negative_mass + self.false_positive_mass
        ) / self.ground_truth_mass

    def to_dict(self) -> dict[str, float]:
        """Return masses and derived metrics as JSON-safe floats."""
        return {
            "true_positive_mass": self.true_positive_mass,
            "false_positive_mass": self.false_positive_mass,
            "false_negative_mass": self.false_negative_mass,
            "ground_truth_mass": self.ground_truth_mass,
            "predicted_mass": self.predicted_mass,
            "recall": self.recall,
            "miss": self.miss,
            "hallucination": self.hallucination,
            "dmota": self.dmota,
        }


@dataclass(frozen=True, slots=True)
class FrameEvaluation:
    """Evaluation summary and grid provenance for one reconstructed frame.

    ``bounds`` contains three world-coordinate ``(min, max)`` axis intervals
    used to build the evaluation grid. ``voxel_resolution`` is the positive
    world-coordinate grid spacing used for the overlap calculation.
    """

    dataset_name: str
    frame: int
    summary: EvaluationSummary
    bounds: tuple[tuple[float, float], ...]
    voxel_resolution: float

    def __post_init__(self) -> None:
        if not self.dataset_name:
            raise ValueError("dataset_name must not be empty.")
        if len(self.bounds) != 3 or any(len(axis) != 2 for axis in self.bounds):
            raise ValueError("bounds must contain three (min, max) pairs.")
        if any(
            not np.isfinite(value)
            for axis in self.bounds
            for value in axis
        ) or any(axis[1] <= axis[0] for axis in self.bounds):
            raise ValueError("bounds must be finite and satisfy min < max.")
        if self.voxel_resolution <= 0:
            raise ValueError("voxel_resolution must be positive.")

    def to_dict(self) -> dict:
        """Return frame metadata, bounds, resolution, and metric values."""
        return {
            "dataset": self.dataset_name,
            "frame": self.frame,
            "bounds": [list(axis) for axis in self.bounds],
            "voxel_resolution": self.voxel_resolution,
            **self.summary.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    """Typed evaluation of a reconstruction run.

    ``frames`` preserves per-frame metrics. ``artifacts`` is populated only
    when evaluation was called with an explicit output config.
    """

    frames: tuple[FrameEvaluation, ...]
    config: EvaluationConfig
    artifacts: Optional[RunArtifacts] = None

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("EvaluationRun.frames must not be empty.")

    @property
    def summary(self) -> EvaluationSummary:
        """Aggregate all frame masses, then recompute derived metrics."""
        return EvaluationSummary(
            true_positive_mass=sum(
                frame.summary.true_positive_mass for frame in self.frames
            ),
            false_positive_mass=sum(
                frame.summary.false_positive_mass for frame in self.frames
            ),
            false_negative_mass=sum(
                frame.summary.false_negative_mass for frame in self.frames
            ),
            ground_truth_mass=sum(
                frame.summary.ground_truth_mass for frame in self.frames
            ),
            predicted_mass=sum(frame.summary.predicted_mass for frame in self.frames),
        )

    @property
    def run_dir(self) -> Optional[Path]:
        return self.artifacts.run_dir if self.artifacts is not None else None
