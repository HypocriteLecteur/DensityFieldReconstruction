"""Reusable DRA scale/model-order computation and fitting."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from scipy.spatial import cKDTree

from dfr.analysis.results import ScaleAnalysisResult
from dfr.gaussian_mixture_reduction import GMR


FIT_MODELS = ("power", "power_interaction", "log_quadratic")


def validate_normalized_scales(values) -> np.ndarray:
    scales = np.asarray(values, dtype=np.float64)
    if (
        scales.ndim != 1
        or len(scales) < 2
        or not np.all(np.isfinite(scales))
        or np.any(scales <= 0)
        or np.any(np.diff(scales) <= 0)
    ):
        raise ValueError(
            "normalized scales must be a strictly increasing 1D array of at "
            "least two positive finite values."
        )
    return scales


def mean_nearest_neighbour_distance(positions: np.ndarray) -> float:
    """Return the mean distance from every point to its nearest other point."""
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (agents, 3).")
    if len(positions) < 2:
        raise ValueError("At least two agents are required to calculate NND.")
    distances, _ = cKDTree(positions).query(positions, k=2)
    return float(np.mean(distances[:, 1]))


def model_orders(
    number_of_agents: int, steps: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Return unique rounded component counts and requested percentages of N."""
    if number_of_agents < 2 or steps < 2:
        raise ValueError("number_of_agents and steps must both be at least 2.")
    maximum_fraction = 0.30 if number_of_agents < 100 else 0.10
    requested_fractions = np.linspace(0.01, maximum_fraction, steps)
    components = np.clip(
        np.rint(number_of_agents * requested_fractions).astype(int),
        1,
        number_of_agents,
    )
    if len(np.unique(components)) != steps:
        raise ValueError(
            f"The model-order sweep produces duplicate integer counts for "
            f"N={number_of_agents}: {components.tolist()}"
        )
    return components, 100.0 * requested_fractions


def create_scale_analysis(
    dataset_name: str,
    time_step: int,
    positions: np.ndarray,
    normalized_scales,
    voxel_res_fraction: float,
    model_order_steps: int = 10,
) -> ScaleAnalysisResult:
    """Create an empty typed DRA surface from one point frame."""
    positions = np.asarray(positions, dtype=np.float32)
    scales = validate_normalized_scales(normalized_scales)
    component_counts, percentages = model_orders(len(positions), model_order_steps)
    return ScaleAnalysisResult(
        dataset_name=dataset_name,
        time_step=int(time_step),
        normalized_scales=scales,
        model_order_percentages=percentages,
        component_counts=component_counts,
        dra=np.full((len(scales), len(component_counts)), np.nan, dtype=np.float64),
        mean_nnd=mean_nearest_neighbour_distance(positions),
        number_of_animals=len(positions),
        voxel_res_fraction=float(voxel_res_fraction),
    )


@torch.inference_mode()
def compute_dra_sweep(
    positions: np.ndarray,
    scale: float,
    reduced_models: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    bounds: np.ndarray,
    voxel_res: np.float64,
    batch_size: int,
) -> np.ndarray:
    """Evaluate several reduced mixtures in one shared voxel traversal."""
    device = torch.device("cuda")
    number_of_animals = len(positions)
    gt_means = torch.as_tensor(positions, device=device, dtype=torch.float32)
    gt_weights = torch.ones(number_of_animals, device=device)
    gt_sigmas = torch.full((number_of_animals,), scale, device=device)
    all_means = [gt_means]
    all_weights = [gt_weights]
    all_sigmas = [gt_sigmas]
    mixture_slices = []
    component_offset = number_of_animals
    for model_means, model_weights, model_sigmas in reduced_models:
        component_count = len(model_means)
        all_means.append(model_means)
        all_weights.append(model_weights.reshape(-1))
        all_sigmas.append(model_sigmas.reshape(-1))
        mixture_slices.append(slice(component_offset, component_offset + component_count))
        component_offset += component_count
    all_means = torch.cat(all_means)
    all_weights = torch.cat(all_weights)
    all_sigmas = torch.cat(all_sigmas)
    variances = all_sigmas.square()
    inverse_negative_two_variances = -0.5 / variances
    normalized_weights = all_weights / (2.0 * torch.pi * variances).pow(1.5)

    ticks = [
        torch.arange(axis[0], axis[1], voxel_res, device=device) for axis in bounds
    ]
    nx, ny, nz = (len(axis) for axis in ticks)
    total_voxels = nx * ny * nz
    absolute_error_sums = torch.zeros(len(reduced_models), device=device)
    for start in range(0, total_voxels, batch_size):
        stop = min(start + batch_size, total_voxels)
        flat = torch.arange(start, stop, device=device)
        ix = flat // (ny * nz)
        iy = (flat // nz) % ny
        iz = flat % nz
        coordinates = torch.stack((ticks[0][ix], ticks[1][iy], ticks[2][iz]), dim=-1)
        densities = torch.cdist(coordinates, all_means).square_()
        densities.mul_(inverse_negative_two_variances).exp_()
        densities.mul_(normalized_weights)
        ground_truth = densities[:, :number_of_animals].sum(dim=1)
        for index, model_slice in enumerate(mixture_slices):
            prediction = densities[:, model_slice].sum(dim=1)
            absolute_error_sums[index] += torch.sum(torch.abs(prediction - ground_truth))
    l1_masses = absolute_error_sums.cpu().numpy() * float(voxel_res**3)
    return 1.0 - l1_masses / number_of_animals


RowCallback = Callable[[ScaleAnalysisResult, int, dict], None]


def compute_scale_model_order_surface(
    positions: np.ndarray,
    result: ScaleAnalysisResult,
    *,
    batch_size: int = 200_000,
    row_callback: Optional[RowCallback] = None,
) -> ScaleAnalysisResult:
    """Fill missing cells of a DRA result; optional callback handles persistence."""
    if not torch.cuda.is_available():
        raise RuntimeError("DRA surface computation requires CUDA.")
    positions = np.asarray(positions, dtype=np.float32)
    if len(positions) != result.number_of_animals:
        raise ValueError("positions count does not match ScaleAnalysisResult.")
    data_min = positions.min(axis=0)
    data_max = positions.max(axis=0)
    voxel_res = np.float64(np.max(data_max - data_min) * result.voxel_res_fraction)
    means = torch.as_tensor(positions, device="cuda", dtype=torch.float32)
    weights = torch.ones((result.number_of_animals, 1), device="cuda")
    for scale_index, scale in enumerate(result.scales):
        missing_indices = np.flatnonzero(~np.isfinite(result.dra[scale_index]))
        if len(missing_indices) == 0:
            continue
        missing_components = result.component_counts[missing_indices]
        bounds = np.column_stack((data_min - 3.0 * scale, data_max + 3.0 * scale))
        radii = torch.full((result.number_of_animals, 1), float(scale), device="cuda")
        gmr_start = time.perf_counter()
        snapshots = GMR.runnalls_algorithm_simple_torch(
            means=means,
            radii=radii,
            weights=weights,
            L=int(missing_components.min()),
            DEVICE="cuda",
            snapshot_Ls=missing_components,
        )
        torch.cuda.synchronize()
        gmr_seconds = time.perf_counter() - gmr_start
        reduced_models = []
        for count in missing_components:
            reduced_means, reduced_weights, covariance = snapshots[int(count)]
            reduced_models.append(
                (
                    reduced_means,
                    reduced_weights,
                    torch.sqrt(covariance[:, 0, 0]).reshape(-1, 1),
                )
            )
        dra_start = time.perf_counter()
        result.dra[scale_index, missing_indices] = compute_dra_sweep(
            positions,
            float(scale),
            reduced_models,
            bounds,
            voxel_res,
            batch_size,
        )
        torch.cuda.synchronize()
        timing = {
            "gmr_seconds": gmr_seconds,
            "dra_seconds": time.perf_counter() - dra_start,
            "voxel_resolution": float(voxel_res),
            "component_indices": missing_indices.copy(),
        }
        if row_callback is not None:
            row_callback(result, scale_index, timing)
    return result


def fit_design_matrix(
    normalized_scale: np.ndarray,
    normalized_order: np.ndarray,
    model_name: str,
) -> np.ndarray:
    """Build features for log(1-DRA) from log scale and model order."""
    log_scale = np.log(np.asarray(normalized_scale))
    log_order = np.log(np.asarray(normalized_order))
    columns = [np.ones(log_scale.size), log_scale, log_order]
    if model_name in ("power_interaction", "log_quadratic"):
        columns.append(log_scale * log_order)
    if model_name == "log_quadratic":
        columns.extend((np.square(log_scale), np.square(log_order)))
    if model_name not in FIT_MODELS:
        raise ValueError(f"Unknown fit model: {model_name}")
    return np.column_stack(columns)


def fit_one_surface_model(
    normalized_scale: np.ndarray,
    normalized_order: np.ndarray,
    dra: np.ndarray,
    model_name: str,
) -> dict:
    design = fit_design_matrix(normalized_scale, normalized_order, model_name)
    dra = np.asarray(dra)
    log_error = np.log(np.maximum(1.0 - dra, 1e-8))
    coefficients = np.linalg.lstsq(design, log_error, rcond=None)[0]
    prediction = 1.0 - np.exp(design @ coefficients)
    residual = prediction - dra
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    denominator = np.sum((dra - dra.mean()) ** 2)
    r_squared = float(1.0 - np.sum(np.square(residual)) / denominator)
    return {
        "name": model_name,
        "coefficients": coefficients,
        "prediction": prediction,
        "rmse": rmse,
        "r_squared": r_squared,
    }


def cross_validated_rmse(scale_grid, order_grid, dra, model_name: str) -> float:
    """Hold out each complete scale row and model-order column in turn."""
    errors = []
    for scale_index in range(dra.shape[0]):
        mask = np.ones(dra.shape, dtype=bool)
        mask[scale_index, :] = False
        fitted = fit_one_surface_model(
            scale_grid[mask], order_grid[mask], dra[mask], model_name
        )
        prediction = 1.0 - np.exp(
            fit_design_matrix(scale_grid[scale_index], order_grid[scale_index], model_name)
            @ fitted["coefficients"]
        )
        errors.extend(prediction - dra[scale_index])
    for order_index in range(dra.shape[1]):
        mask = np.ones(dra.shape, dtype=bool)
        mask[:, order_index] = False
        fitted = fit_one_surface_model(
            scale_grid[mask], order_grid[mask], dra[mask], model_name
        )
        prediction = 1.0 - np.exp(
            fit_design_matrix(scale_grid[:, order_index], order_grid[:, order_index], model_name)
            @ fitted["coefficients"]
        )
        errors.extend(prediction - dra[:, order_index])
    return float(np.sqrt(np.mean(np.square(errors))))


def fit_dra_surface(
    normalized_scales: np.ndarray,
    components: np.ndarray,
    number_of_animals: int,
    dra: np.ndarray,
) -> dict:
    normalized_orders = components / number_of_animals
    scale_grid, order_grid = np.meshgrid(
        normalized_scales, normalized_orders, indexing="ij"
    )
    candidates = {}
    for name in FIT_MODELS:
        fitted = fit_one_surface_model(
            scale_grid.ravel(), order_grid.ravel(), dra.ravel(), name
        )
        fitted["prediction"] = fitted["prediction"].reshape(dra.shape)
        fitted["cv_rmse"] = cross_validated_rmse(scale_grid, order_grid, dra, name)
        candidates[name] = fitted
    best_name = min(candidates, key=lambda name: candidates[name]["cv_rmse"])
    return {
        "best_name": best_name,
        "normalized_orders": normalized_orders,
        "candidates": candidates,
    }


@dataclass
class DRAFrameSamples:
    dataset: str
    time_step: int
    number_of_animals: int
    mean_nnd: float
    scale: np.ndarray
    order: np.ndarray
    dra: np.ndarray

    @classmethod
    def from_result(cls, result: ScaleAnalysisResult) -> "DRAFrameSamples":
        normalized_orders = result.component_counts / result.number_of_animals
        scale_grid, order_grid = np.meshgrid(
            result.normalized_scales, normalized_orders, indexing="ij"
        )
        return cls(
            dataset=result.dataset_name,
            time_step=result.time_step,
            number_of_animals=result.number_of_animals,
            mean_nnd=result.mean_nnd,
            scale=scale_grid.ravel(),
            order=order_grid.ravel(),
            dra=result.dra.ravel(),
        )


def select_frames(start: int, stop: int, count: int, preferred: int) -> np.ndarray:
    """Include endpoints/preferred frame, then bisect the largest time gaps."""
    if stop <= start:
        raise ValueError(f"Empty frame interval [{start}, {stop}).")
    available = stop - start
    count = min(count, available)
    if count < 1:
        raise ValueError("Frame count must be positive.")
    if count == available:
        return np.arange(start, stop, dtype=int)
    preferred = min(max(preferred, start), stop - 1)
    if count == 1:
        return np.asarray([preferred], dtype=int)
    if count == 2:
        return np.asarray([start, stop - 1], dtype=int)
    selected = {start, stop - 1, preferred}
    while len(selected) < count:
        ordered = sorted(selected)
        _, left, right = max(
            (right - left, left, right) for left, right in zip(ordered, ordered[1:])
        )
        midpoint = (left + right) // 2
        if midpoint in selected:
            midpoint += 1
        selected.add(midpoint)
    return np.asarray(sorted(selected), dtype=int)


def concatenate_frames(frames: list[DRAFrameSamples]):
    if not frames:
        raise ValueError("At least one DRA frame is required.")
    return (
        np.concatenate([frame.scale for frame in frames]),
        np.concatenate([frame.order for frame in frames]),
        np.concatenate([frame.dra for frame in frames]),
    )


def grouped_cv_rmse(frames: list[DRAFrameSamples], model_name: str) -> float:
    errors = []
    for held_index, held in enumerate(frames):
        training = [frame for index, frame in enumerate(frames) if index != held_index]
        if not training:
            return float("nan")
        scale, order, dra = concatenate_frames(training)
        fitted = fit_one_surface_model(scale, order, dra, model_name)
        prediction = 1.0 - np.exp(
            fit_design_matrix(held.scale, held.order, model_name)
            @ fitted["coefficients"]
        )
        errors.extend(prediction - held.dra)
    return float(np.sqrt(np.mean(np.square(errors))))


def leave_one_dataset_out_rmse(
    frames: list[DRAFrameSamples], model_name: str
) -> float:
    errors = []
    datasets = sorted({frame.dataset for frame in frames})
    if len(datasets) < 2:
        return float("nan")
    for held_dataset in datasets:
        training = [frame for frame in frames if frame.dataset != held_dataset]
        held = [frame for frame in frames if frame.dataset == held_dataset]
        scale, order, dra = concatenate_frames(training)
        fitted = fit_one_surface_model(scale, order, dra, model_name)
        for frame in held:
            prediction = 1.0 - np.exp(
                fit_design_matrix(frame.scale, frame.order, model_name)
                @ fitted["coefficients"]
            )
            errors.extend(prediction - frame.dra)
    return float(np.sqrt(np.mean(np.square(errors))))


def fit_frames(
    frames: list[DRAFrameSamples], include_dataset_cv: bool = False
) -> dict:
    scale, order, dra = concatenate_frames(frames)
    candidates = {}
    for name in FIT_MODELS:
        fitted = fit_one_surface_model(scale, order, dra, name)
        fitted["frame_cv_rmse"] = grouped_cv_rmse(frames, name)
        if include_dataset_cv:
            fitted["dataset_cv_rmse"] = leave_one_dataset_out_rmse(frames, name)
        candidates[name] = fitted
    best_name = min(candidates, key=lambda name: candidates[name]["frame_cv_rmse"])
    return {"best_name": best_name, "candidates": candidates}
