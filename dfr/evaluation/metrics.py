"""Pure voxelized density-overlap metrics with explicit device selection."""

from __future__ import annotations

import numpy as np
import torch


def evaluate_isotropic_gmm(
    coordinates: torch.Tensor,
    means: torch.Tensor,
    weights: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a 3D isotropic Gaussian mixture at query coordinates."""
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape (samples, 3).")
    if means.ndim != 2 or means.shape[1] != 3 or len(means) == 0:
        raise ValueError("means must have shape (components, 3).")
    weights = weights.reshape(-1)
    sigmas = sigmas.reshape(-1)
    if len(weights) != len(means) or len(sigmas) != len(means):
        raise ValueError("weights and sigmas must align with means.")
    if torch.any(weights < 0):
        raise ValueError("weights must be non-negative.")
    if torch.any(sigmas <= 0):
        raise ValueError("sigmas must be positive.")
    squared_distances = torch.cdist(coordinates, means).square()
    variances = sigmas.square()
    normalization = (2.0 * torch.pi * variances).pow(1.5)
    component_densities = (weights / normalization) * torch.exp(
        -squared_distances / (2.0 * variances)
    )
    return torch.sum(component_densities, dim=1)


@torch.inference_mode()
def compute_density_overlap_masses(
    ground_truth_means,
    ground_truth_sigma: float,
    predicted_means,
    predicted_weights,
    predicted_sigmas,
    *,
    bounds,
    voxel_resolution: float,
    batch_size: int = 500_000,
    device: str | torch.device = "cuda",
) -> tuple[float, float, float]:
    """Integrate true-positive, false-positive, and false-negative density mass."""
    if ground_truth_sigma <= 0 or voxel_resolution <= 0 or batch_size < 1:
        raise ValueError("sigma, voxel_resolution, and batch_size must be positive.")
    selected_device = torch.device(device)
    truth_means = torch.as_tensor(
        ground_truth_means, dtype=torch.float32, device=selected_device
    )
    prediction_means = torch.as_tensor(
        predicted_means, dtype=torch.float32, device=selected_device
    )
    prediction_weights = torch.as_tensor(
        predicted_weights, dtype=torch.float32, device=selected_device
    ).reshape(-1)
    prediction_sigmas = torch.as_tensor(
        predicted_sigmas, dtype=torch.float32, device=selected_device
    ).reshape(-1)
    if truth_means.ndim != 2 or truth_means.shape[1] != 3 or len(truth_means) == 0:
        raise ValueError("ground_truth_means must be a non-empty (points, 3) array.")
    if prediction_means.ndim != 2 or prediction_means.shape[1] != 3:
        raise ValueError("predicted_means must have shape (components, 3).")
    if len(prediction_means) == 0:
        raise ValueError("At least one predicted Gaussian is required.")
    if len(prediction_weights) != len(prediction_means) or len(
        prediction_sigmas
    ) != len(prediction_means):
        raise ValueError("Predicted weights and sigmas must align with means.")
    if torch.any(prediction_weights < 0) or torch.any(prediction_sigmas <= 0):
        raise ValueError("Predicted weights must be non-negative and sigmas positive.")

    bounds_array = np.asarray(bounds, dtype=np.float64)
    if bounds_array.shape != (3, 2) or np.any(~np.isfinite(bounds_array)):
        raise ValueError("bounds must contain three finite (min, max) pairs.")
    if np.any(bounds_array[:, 1] <= bounds_array[:, 0]):
        raise ValueError("Every bound must satisfy min < max.")
    ticks = [
        torch.arange(start, stop, voxel_resolution, device=selected_device)
        for start, stop in bounds_array
    ]
    nx, ny, nz = (len(axis) for axis in ticks)
    if min(nx, ny, nz) == 0:
        raise ValueError("voxel_resolution produces an empty evaluation grid.")
    total_voxels = nx * ny * nz
    voxel_volume = float(voxel_resolution**3)
    truth_weights = torch.ones(len(truth_means), device=selected_device)
    truth_sigmas = torch.full(
        (len(truth_means),), ground_truth_sigma, device=selected_device
    )
    true_positive = false_positive = false_negative = 0.0
    for start_index in range(0, total_voxels, batch_size):
        stop_index = min(start_index + batch_size, total_voxels)
        flat = torch.arange(start_index, stop_index, device=selected_device)
        ix = flat // (ny * nz)
        iy = (flat // nz) % ny
        iz = flat % nz
        coordinates = torch.stack(
            (ticks[0][ix], ticks[1][iy], ticks[2][iz]), dim=-1
        )
        truth = evaluate_isotropic_gmm(
            coordinates, truth_means, truth_weights, truth_sigmas
        )
        prediction = evaluate_isotropic_gmm(
            coordinates,
            prediction_means,
            prediction_weights,
            prediction_sigmas,
        )
        true_positive += (
            torch.minimum(truth, prediction).sum().item() * voxel_volume
        )
        false_positive += (
            torch.clamp(prediction - truth, min=0).sum().item() * voxel_volume
        )
        false_negative += (
            torch.clamp(truth - prediction, min=0).sum().item() * voxel_volume
        )
    return true_positive, false_positive, false_negative


def automatic_evaluation_bounds(positions, scale: float) -> np.ndarray:
    """Return legacy-compatible point bounds padded by three density scales."""
    points = np.asarray(positions, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("positions must be a non-empty (points, 3) array.")
    if scale <= 0:
        raise ValueError("scale must be positive.")
    return np.column_stack(
        (points.min(axis=0) - 3 * scale, points.max(axis=0) + 3 * scale)
    )
