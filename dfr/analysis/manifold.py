"""Centered-3PL parameter-manifold fitting without plotting side effects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
from scipy.special import expm1

from dfr.analysis.results import ManifoldAnalysisResult


PARAMETER_NAMES = ("k", "sigma_half", "log10_gamma")
DEFAULT_BOUNDS = ((0.1, 1e-6, -2.0), (20.0, np.inf, 5.0))


def centered_3pl_excess(x, parameters, number_of_agents: int) -> np.ndarray:
    """Evaluate the centered 3PL curve above its asymptotic floor of one.

    ``parameters`` are ``(k, sigma_half, log10_gamma)``. Add one to the
    returned values to obtain mode counts.
    """
    k, sigma_half, log10_gamma = np.asarray(parameters, dtype=np.float64)
    if number_of_agents < 2:
        raise ValueError("number_of_agents must be at least 2.")
    if k <= 0 or sigma_half <= 0:
        raise ValueError("k and sigma_half must be positive.")
    gamma = 10.0**log10_gamma
    scaling = max(float(expm1(np.log(2.0) / max(gamma, 1e-6))), 1e-12)
    x = np.asarray(x, dtype=np.float64)
    log_ratio = np.clip(
        k * np.log(np.maximum(x / sigma_half, 1e-12)) + np.log(scaling),
        -500,
        500,
    )
    return (number_of_agents - 1.0) / np.power(1.0 + np.exp(log_ratio), gamma)


def median_nearest_neighbour_distance(positions) -> float:
    """Return the median nearest-neighbour distance for one point frame.

    ``positions`` may be 2D or 3D but must contain at least two points. A small
    positive floor prevents downstream log-scale curve fitting from receiving
    zero.
    """
    points = np.asarray(positions, dtype=np.float64)
    if points.ndim != 2 or len(points) < 2:
        raise ValueError("positions must contain at least two points.")
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    return max(float(np.median(np.min(distances, axis=1))), 1e-8)


@dataclass
class LegacyManifoldCache:
    """Validated view of the historic three-file parameter-manifold cache.

    ``mode_counts`` has shape ``(frames, scales)``. ``scale_ranges`` has shape
    ``(frames, 2)`` and stores per-frame world-coordinate scale intervals used
    to reconstruct logarithmic scale grids. ``nearest_neighbour_distances`` is
    optional legacy metadata aligned with frames.
    """

    mode_counts: np.ndarray
    scale_ranges: np.ndarray
    nearest_neighbour_distances: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.mode_counts = np.asarray(self.mode_counts, dtype=np.float64)
        self.scale_ranges = np.asarray(self.scale_ranges, dtype=np.float64)
        if self.mode_counts.ndim != 2 or self.mode_counts.shape[1] < 5:
            raise ValueError("mode_counts must be a 2D array with at least five scales.")
        if self.scale_ranges.shape != (len(self.mode_counts), 2):
            raise ValueError("scale_ranges must have shape (frames, 2).")
        if np.any(~np.isfinite(self.scale_ranges)) or np.any(self.scale_ranges <= 0):
            raise ValueError("scale_ranges must be positive and finite.")
        if np.any(self.scale_ranges[:, 1] <= self.scale_ranges[:, 0]):
            raise ValueError("Every scale range must satisfy start < stop.")
        if np.any(~np.isfinite(self.mode_counts)) or np.any(self.mode_counts < 0):
            raise ValueError("mode_counts must be finite and non-negative.")
        if self.nearest_neighbour_distances is not None:
            self.nearest_neighbour_distances = np.asarray(
                self.nearest_neighbour_distances, dtype=np.float64
            )
            if self.nearest_neighbour_distances.shape != (len(self.mode_counts),):
                raise ValueError("nearest-neighbour distances must align with frames.")

    @property
    def completed_rows(self) -> np.ndarray:
        """Mask rows containing at least one historic mode-count sample."""
        return np.any(self.mode_counts != 0, axis=1)


def load_legacy_manifold_cache(directory: str | Path) -> LegacyManifoldCache:
    """Load ``modes.npy``, ``scale_range.npy``, and optional ``nn_dists.npy``."""
    root = Path(directory)
    mode_path = root / "modes.npy"
    scale_path = root / "scale_range.npy"
    missing = [str(path) for path in (mode_path, scale_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing parameter-manifold cache: " + ", ".join(missing))
    nearest_path = root / "nn_dists.npy"
    return LegacyManifoldCache(
        mode_counts=np.load(mode_path, allow_pickle=False),
        scale_ranges=np.load(scale_path, allow_pickle=False),
        nearest_neighbour_distances=(
            np.load(nearest_path, allow_pickle=False) if nearest_path.is_file() else None
        ),
    )


@dataclass
class Centered3PLFitBatch:
    """Aligned per-frame centered-3PL fit details.

    ``result`` contains only successful parameter rows. ``success`` and
    ``fitted_curves`` remain aligned to the original input frame order, and
    ``scale_grids`` stores the reconstructed per-frame world-coordinate scale
    grid used for fitting.
    """

    result: ManifoldAnalysisResult
    success: np.ndarray
    fitted_curves: tuple[Optional[np.ndarray], ...]
    residual_variances: np.ndarray
    scale_grids: tuple[np.ndarray, ...]


@dataclass
class Symmetric2PLFitBatch:
    """Aligned success mask and compact symmetric-2PL parameter table.

    ``result`` contains successful ``(k, sigma_half)`` fits. ``success`` and
    ``aligned_parameters`` preserve the original input frame order for
    diagnostics and plotting.
    """

    result: ManifoldAnalysisResult
    success: np.ndarray
    aligned_parameters: np.ndarray


def symmetric_2pl_mode_count(
    x, k: float, sigma_half: float, number_of_agents: int
) -> np.ndarray:
    """Evaluate the symmetric two-parameter mode-count curve.

    ``x`` is a world-coordinate scale grid, ``sigma_half`` is the scale where
    the fitted curve reaches half its dynamic range, and ``number_of_agents``
    sets the asymptotic high-count limit.
    """
    if k <= 0 or sigma_half <= 0 or number_of_agents < 2:
        raise ValueError("k, sigma_half, and number_of_agents must be positive.")
    values = np.asarray(x, dtype=np.float64)
    return 1.0 + (number_of_agents - 1.0) / (
        1.0 + np.power(values / sigma_half, k)
    )


def fit_symmetric_2pl_curves(
    frame_ids,
    animal_counts,
    scale_ranges,
    mode_counts,
    *,
    dataset_name: Optional[str] = None,
    max_function_evaluations: int = 5000,
) -> Symmetric2PLFitBatch:
    """Fit the symmetric 2PL model to aligned historic mode-count curves.

    ``frame_ids``, ``animal_counts``, and ``scale_ranges`` are 1D/per-frame
    arrays. ``mode_counts`` is ``(frames, scales)``. Fitting failures are
    recorded in the returned ``success`` mask instead of raising.
    """
    frames = np.asarray(frame_ids, dtype=np.int64)
    counts = np.asarray(animal_counts, dtype=np.int64)
    ranges = np.asarray(scale_ranges, dtype=np.float64)
    modes = np.asarray(mode_counts, dtype=np.float64)
    if frames.ndim != 1 or len(frames) == 0:
        raise ValueError("frame_ids must be a non-empty 1D array.")
    if counts.shape != frames.shape or np.any(counts < 2):
        raise ValueError("animal_counts must align with frames and be at least 2.")
    if ranges.shape != (len(frames), 2) or np.any(ranges <= 0):
        raise ValueError("scale_ranges must have shape (frames, 2) and be positive.")
    if np.any(ranges[:, 1] <= ranges[:, 0]):
        raise ValueError("Every scale range must satisfy start < stop.")
    if modes.ndim != 2 or modes.shape[0] != len(frames) or modes.shape[1] < 3:
        raise ValueError("mode_counts must have shape (frames, at least three scales).")

    parameters = np.full((len(frames), 2), np.nan)
    success = np.zeros(len(frames), dtype=bool)
    for index, (number_of_agents, scale_range, observed) in enumerate(
        zip(counts, ranges, modes)
    ):
        scales = np.logspace(
            np.log10(max(scale_range[0], 1e-12)),
            np.log10(max(scale_range[1], 1e-11)),
            modes.shape[1],
        )
        try:
            fitted, _ = curve_fit(
                lambda x, k, sigma_half: symmetric_2pl_mode_count(
                    x, k, sigma_half, number_of_agents
                ),
                scales,
                observed,
                p0=(2.0, float(np.median(scales))),
                bounds=((0.1, 1e-6), (50.0, np.inf)),
                maxfev=max_function_evaluations,
            )
        except (RuntimeError, ValueError, FloatingPointError):
            continue
        parameters[index] = fitted
        success[index] = True
    names = (
        np.asarray([dataset_name] * int(success.sum()), dtype=str)
        if dataset_name is not None
        else None
    )
    result = ManifoldAnalysisResult(
        parameter_names=("k", "sigma_half"),
        parameters=parameters[success],
        frame_ids=frames[success],
        dataset_names=names,
    )
    return Symmetric2PLFitBatch(result, success, parameters)


def fit_centered_3pl_curves(
    frame_ids,
    animal_counts,
    scale_ranges,
    mode_counts,
    *,
    dataset_name: Optional[str] = None,
    saturation: float = 0.8,
    max_function_evaluations: int = 5000,
) -> Centered3PLFitBatch:
    """Fit the historic centered 3PL model independently to many frames.

    ``scale_ranges`` are per-frame world-coordinate intervals used to recreate
    logarithmic scale grids. The fit starts after the first sample below the
    requested saturation fraction of the animal count. Failed frames remain in
    aligned diagnostic arrays but are omitted from the compact result table.
    """
    frames = np.asarray(frame_ids, dtype=np.int64)
    counts = np.asarray(animal_counts, dtype=np.int64)
    ranges = np.asarray(scale_ranges, dtype=np.float64)
    modes = np.asarray(mode_counts, dtype=np.float64)
    if frames.ndim != 1 or len(frames) == 0:
        raise ValueError("frame_ids must be a non-empty 1D array.")
    if counts.shape != frames.shape or np.any(counts < 2):
        raise ValueError("animal_counts must align with frames and be at least 2.")
    if ranges.shape != (len(frames), 2) or np.any(ranges <= 0):
        raise ValueError("scale_ranges must have shape (frames, 2) and be positive.")
    if np.any(ranges[:, 1] <= ranges[:, 0]):
        raise ValueError("Every scale range must satisfy start < stop.")
    if modes.ndim != 2 or modes.shape[0] != len(frames) or modes.shape[1] < 5:
        raise ValueError("mode_counts must have shape (frames, at least five scales).")
    if not 0 < saturation < 1:
        raise ValueError("saturation must lie strictly between zero and one.")

    success = np.zeros(len(frames), dtype=bool)
    parameters = np.full((len(frames), len(PARAMETER_NAMES)), np.nan)
    residuals = np.full(len(frames), np.nan)
    fitted: list[Optional[np.ndarray]] = [None] * len(frames)
    grids: list[np.ndarray] = []
    for index, (number_of_agents, scale_range, observed) in enumerate(
        zip(counts, ranges, modes)
    ):
        scales = np.logspace(
            np.log10(max(scale_range[0], 1e-6)),
            np.log10(max(scale_range[1], 1e-5)),
            modes.shape[1],
        )
        grids.append(scales)
        boundary = int(np.argmax(observed <= saturation * number_of_agents))
        if boundary == 0:
            boundary = max(
                1,
                int(
                    np.argmax(
                        observed
                        <= min(saturation + 0.1, 0.99) * number_of_agents
                    )
                ),
            )
        x_data = scales[boundary:]
        y_data = observed[boundary:]
        if len(x_data) < 5:
            continue
        try:
            fitted_parameters, _ = curve_fit(
                lambda x, *values: centered_3pl_excess(x, values, number_of_agents),
                x_data,
                y_data,
                p0=(2.0, float(np.median(scales)), 0.0),
                sigma=np.maximum(y_data, 1.0),
                absolute_sigma=True,
                bounds=DEFAULT_BOUNDS,
                maxfev=max_function_evaluations,
            )
        except (RuntimeError, ValueError, FloatingPointError):
            continue
        parameters[index] = fitted_parameters
        fitted[index] = centered_3pl_excess(
            scales, fitted_parameters, number_of_agents
        ) + 1.0
        prediction = (
            centered_3pl_excess(x_data, fitted_parameters, number_of_agents) + 1.0
        )
        residuals[index] = float(np.mean(np.square(y_data - prediction)))
        success[index] = True

    names = None
    if dataset_name is not None:
        names = np.asarray([dataset_name] * int(success.sum()), dtype=str)
    result = ManifoldAnalysisResult(
        parameter_names=PARAMETER_NAMES,
        parameters=parameters[success],
        frame_ids=frames[success],
        dataset_names=names,
    )
    return Centered3PLFitBatch(
        result=result,
        success=success,
        fitted_curves=tuple(fitted),
        residual_variances=residuals,
        scale_grids=tuple(grids),
    )


def scale_for_mode_count(
    parameters, number_of_agents: int, target_modes: float
) -> float:
    """Invert a centered 3PL fit to recommend the scale for a target mode count.

    Returns a world-coordinate scale value for ``target_modes`` given
    ``(k, sigma_half, log10_gamma)`` parameters.
    """
    if not 1 < target_modes < number_of_agents:
        raise ValueError("target_modes must lie strictly between 1 and number_of_agents.")
    k, sigma_half, log10_gamma = np.asarray(parameters, dtype=np.float64)
    gamma = 10.0**log10_gamma
    scaling = max(float(expm1(np.log(2.0) / max(gamma, 1e-6))), 1e-12)
    fraction = (target_modes - 1.0) / (number_of_agents - 1.0)
    ratio_power = (fraction ** (-1.0 / gamma) - 1.0) / scaling
    return float(sigma_half * max(ratio_power, 0.0) ** (1.0 / k))


def fit_shape_curve(
    k_values, log_gamma_values
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the historic Hill shape curve and return parameters and dense samples.

    Inputs are aligned 1D arrays from centered-3PL fits. The return value is
    ``(parameters, log_gamma_grid, k_grid)`` for projection/visualization.
    """
    k_values = np.asarray(k_values, dtype=np.float64)
    log_gamma_values = np.asarray(log_gamma_values, dtype=np.float64)
    if (
        k_values.ndim != 1
        or k_values.shape != log_gamma_values.shape
        or len(k_values) < 5
    ):
        raise ValueError(
            "k_values and log_gamma_values must be aligned 1D arrays of length >= 5."
        )

    def hill_model(log_gamma, amplitude, center, scale, power, floor):
        return floor + amplitude / (
            1.0
            + np.power(np.maximum((log_gamma - center) / scale, 1e-10), power)
        )

    parameters, _ = curve_fit(
        hill_model,
        log_gamma_values,
        k_values,
        p0=(20, -1, 0.5, 2, 0),
        maxfev=10000,
    )
    log_gamma_grid = np.linspace(log_gamma_values.min(), log_gamma_values.max(), 500)
    k_grid = hill_model(log_gamma_grid, *parameters)
    return parameters, log_gamma_grid, k_grid


def project_to_shape_curve(k_values, log_gamma_values, log_gamma_grid, k_grid):
    """Project points to their nearest samples on a fitted shape curve.

    All inputs are aligned 1D arrays. The return value is
    ``(projected_k, projected_log_gamma)`` sampled from the fitted grid.
    """
    k_values = np.asarray(k_values, dtype=np.float64)
    log_gamma_values = np.asarray(log_gamma_values, dtype=np.float64)
    log_gamma_grid = np.asarray(log_gamma_grid, dtype=np.float64)
    k_grid = np.asarray(k_grid, dtype=np.float64)
    if k_values.shape != log_gamma_values.shape or k_values.ndim != 1:
        raise ValueError("k_values and log_gamma_values must be aligned 1D arrays.")
    if log_gamma_grid.shape != k_grid.shape or log_gamma_grid.ndim != 1:
        raise ValueError("shape-curve grids must be aligned 1D arrays.")
    distances = (
        np.square(k_values[:, None] - k_grid[None, :])
        + np.square(log_gamma_values[:, None] - log_gamma_grid[None, :])
    )
    indices = np.argmin(distances, axis=1)
    return k_grid[indices], log_gamma_grid[indices]
