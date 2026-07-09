"""Typed, persistence-friendly result objects for DFR analyses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


def _one_dimensional(name: str, value, dtype=None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    return array


@dataclass
class ModeCurveResult:
    """Mode counts measured over an ordered scale grid.

    ``scales`` is a strictly increasing positive 1D array. In low-level mode
    counting APIs these values are world-coordinate density scales; plotting
    helpers may label them as normalized NND scales when the caller has already
    normalized them. ``mode_counts`` is one non-negative integer per scale.
    Optional ``frame`` and ``dataset_name`` fields preserve provenance for
    plots, caches, and handoff files.
    """

    scales: np.ndarray
    mode_counts: np.ndarray
    frame: Optional[int] = None
    dataset_name: Optional[str] = None

    def __post_init__(self) -> None:
        self.scales = _one_dimensional("scales", self.scales, np.float64)
        self.mode_counts = _one_dimensional("mode_counts", self.mode_counts, np.int64)
        if self.scales.shape != self.mode_counts.shape:
            raise ValueError("scales and mode_counts must have identical shapes.")
        if np.any(~np.isfinite(self.scales)) or np.any(self.scales <= 0):
            raise ValueError("scales must contain positive finite values.")
        if np.any(np.diff(self.scales) <= 0):
            raise ValueError("scales must be strictly increasing.")
        if np.any(self.mode_counts < 0):
            raise ValueError("mode_counts must be non-negative.")

    def save_npz(self, path: str | Path) -> Path:
        """Save this result as a compact ``.npz`` cache and return its path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target,
            scales=self.scales,
            mode_counts=self.mode_counts,
            frame=-1 if self.frame is None else self.frame,
            dataset_name="" if self.dataset_name is None else self.dataset_name,
        )
        return target

    @classmethod
    def load_npz(cls, path: str | Path) -> "ModeCurveResult":
        """Load a result previously written by :meth:`save_npz`."""
        with np.load(path) as data:
            frame = int(data["frame"])
            dataset_name = str(data["dataset_name"])
            return cls(
                scales=data["scales"],
                mode_counts=data["mode_counts"],
                frame=None if frame < 0 else frame,
                dataset_name=dataset_name or None,
            )


@dataclass
class ScaleAnalysisResult:
    """DRA values over normalized scale and Gaussian model order for one frame.

    ``normalized_scales`` is a strictly increasing positive NND-normalized
    scale grid. ``scales`` converts it back to world-coordinate scale by
    multiplying by ``mean_nnd``. ``component_counts`` and
    ``model_order_percentages`` describe the model-order axis. ``dra`` is a
    ``(len(normalized_scales), len(component_counts))`` surface and may contain
    ``NaN`` cells while a resumable CUDA sweep is in progress.
    """

    dataset_name: str
    time_step: int
    normalized_scales: np.ndarray
    model_order_percentages: np.ndarray
    component_counts: np.ndarray
    dra: np.ndarray
    mean_nnd: float
    number_of_animals: int
    voxel_res_fraction: float

    def __post_init__(self) -> None:
        if not self.dataset_name:
            raise ValueError("dataset_name must not be empty.")
        self.normalized_scales = _one_dimensional(
            "normalized_scales", self.normalized_scales, np.float64
        )
        self.model_order_percentages = _one_dimensional(
            "model_order_percentages", self.model_order_percentages, np.float64
        )
        self.component_counts = _one_dimensional(
            "component_counts", self.component_counts, np.int64
        )
        self.dra = np.asarray(self.dra, dtype=np.float64)
        if self.model_order_percentages.shape != self.component_counts.shape:
            raise ValueError(
                "model_order_percentages and component_counts must have identical shapes."
            )
        expected_shape = (len(self.normalized_scales), len(self.component_counts))
        if self.dra.shape != expected_shape:
            raise ValueError(f"dra shape must be {expected_shape}, got {self.dra.shape}.")
        if np.any(~np.isfinite(self.normalized_scales)) or np.any(
            self.normalized_scales <= 0
        ):
            raise ValueError("normalized_scales must be positive and finite.")
        if np.any(np.diff(self.normalized_scales) <= 0):
            raise ValueError("normalized_scales must be strictly increasing.")
        if self.mean_nnd <= 0 or not np.isfinite(self.mean_nnd):
            raise ValueError("mean_nnd must be positive and finite.")
        if self.number_of_animals < 2:
            raise ValueError("number_of_animals must be at least 2.")
        if self.voxel_res_fraction <= 0:
            raise ValueError("voxel_res_fraction must be positive.")

    @property
    def scales(self) -> np.ndarray:
        return self.normalized_scales * self.mean_nnd

    @property
    def is_complete(self) -> bool:
        return bool(np.all(np.isfinite(self.dra)))

    def matches_grid(self, other: "ScaleAnalysisResult") -> bool:
        """Whether two results can safely share/resume one DRA grid."""
        return (
            self.dataset_name == other.dataset_name
            and self.time_step == other.time_step
            and self.number_of_animals == other.number_of_animals
            and self.voxel_res_fraction == other.voxel_res_fraction
            and self.mean_nnd == other.mean_nnd
            and np.array_equal(self.normalized_scales, other.normalized_scales)
            and np.array_equal(self.component_counts, other.component_counts)
            and np.array_equal(
                self.model_order_percentages, other.model_order_percentages
            )
        )

    def as_legacy_tuple(self) -> tuple:
        """Compatibility shape used by pre-Phase-4 experiment callers."""
        return (
            self.normalized_scales,
            self.model_order_percentages,
            self.component_counts,
            self.dra,
            self.mean_nnd,
            self.number_of_animals,
        )

    def save_npz(self, path: str | Path) -> Path:
        """Save this DRA surface as a compatibility-friendly ``.npz`` cache."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target,
            dataset_name=self.dataset_name,
            time_step=self.time_step,
            scales=self.scales,
            normalized_scales=self.normalized_scales,
            component_numbers=self.component_counts,
            model_order_percentages=self.model_order_percentages,
            mean_nnd=self.mean_nnd,
            number_of_animals=self.number_of_animals,
            dra=self.dra,
            voxel_res_fraction=self.voxel_res_fraction,
        )
        return target

    @classmethod
    def load_npz(
        cls,
        path: str | Path,
        *,
        dataset_name: Optional[str] = None,
        number_of_animals: Optional[int] = None,
    ) -> "ScaleAnalysisResult":
        """Load a modern or legacy DRA surface cache.

        Legacy caches may lack ``dataset_name`` or ``number_of_animals``; pass
        those values explicitly when needed.
        """
        with np.load(path) as data:
            stored_name = (
                str(data["dataset_name"])
                if "dataset_name" in data.files
                else dataset_name
            )
            if not stored_name:
                raise ValueError(
                    "Legacy scale cache has no dataset_name; pass dataset_name explicitly."
                )
            stored_animal_count = (
                int(data["number_of_animals"])
                if "number_of_animals" in data.files
                else number_of_animals
            )
            if stored_animal_count is None:
                raise ValueError(
                    "Legacy scale cache has no number_of_animals; pass it explicitly."
                )
            return cls(
                dataset_name=stored_name,
                time_step=int(data["time_step"]),
                normalized_scales=data["normalized_scales"],
                model_order_percentages=data["model_order_percentages"],
                component_counts=data["component_numbers"],
                dra=data["dra"],
                mean_nnd=float(data["mean_nnd"]),
                number_of_animals=stored_animal_count,
                voxel_res_fraction=float(data["voxel_res_fraction"]),
            )


@dataclass
class ManifoldAnalysisResult:
    """Generic fitted-parameter table for manifold analysis.

    ``parameters`` has shape ``(frames, parameters)`` and aligns with
    ``frame_ids``. ``parameter_names`` gives the column order. Optional
    ``dataset_names`` must have one label per frame and is useful when
    aggregating manifold fits across scenarios.
    """

    parameter_names: tuple[str, ...]
    parameters: np.ndarray
    frame_ids: np.ndarray
    dataset_names: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if not self.parameter_names:
            raise ValueError("parameter_names must not be empty.")
        self.parameters = np.asarray(self.parameters, dtype=np.float64)
        self.frame_ids = np.asarray(self.frame_ids, dtype=np.int64)
        if self.frame_ids.ndim != 1:
            raise ValueError("frame_ids must be a one-dimensional array.")
        if self.parameters.ndim != 2:
            raise ValueError("parameters must be a two-dimensional array.")
        if self.parameters.shape != (len(self.frame_ids), len(self.parameter_names)):
            raise ValueError(
                "parameters shape must be (number of frames, number of parameters)."
            )
        if self.dataset_names is not None:
            self.dataset_names = _one_dimensional(
                "dataset_names", self.dataset_names, str
            )
            if len(self.dataset_names) != len(self.frame_ids):
                raise ValueError("dataset_names must align with frame_ids.")

    def save_npz(self, path: str | Path) -> Path:
        """Save the parameter table as a portable ``.npz`` cache."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target,
            parameter_names=np.asarray(self.parameter_names),
            parameters=self.parameters,
            frame_ids=self.frame_ids,
            dataset_names=(
                np.asarray([], dtype=str)
                if self.dataset_names is None
                else self.dataset_names
            ),
        )
        return target

    @classmethod
    def load_npz(cls, path: str | Path) -> "ManifoldAnalysisResult":
        """Load a parameter table previously written by :meth:`save_npz`."""
        with np.load(path) as data:
            names = data["dataset_names"]
            return cls(
                parameter_names=tuple(str(value) for value in data["parameter_names"]),
                parameters=data["parameters"],
                frame_ids=data["frame_ids"],
                dataset_names=names if len(names) else None,
            )
