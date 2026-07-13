"""Reusable voxel-grid sampling for isotropic Gaussian density fields.

These helpers expose the grid convention used by the evaluation and animation
workflows without writing files.  The returned grid is a plain mapping so
existing experiment code can retain direct, explicit access to its axes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from dfr.evaluation.metrics import evaluate_isotropic_gmm


def build_isotropic_density_grid(
    positions: np.ndarray,
    scale: float,
    *,
    voxel_res_fraction: float = 2.5e-2,
    padding_scales: float = 3.0,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    """Create an explicit voxel grid enclosing an isotropic particle mixture.

    ``positions`` must contain at least one world-coordinate point shaped
    ``(agents, 3)``.  The grid spacing is the largest position extent times
    ``voxel_res_fraction`` and each side is padded by ``padding_scales`` times
    ``scale``.  Tick tensors live on ``device``; callers may move them to CPU
    after sampling when retaining a grid for plotting.
    """
    points = np.asarray(positions)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("positions must be a non-empty array shaped (agents, 3).")
    if not np.isfinite(points).all():
        raise ValueError("positions must contain only finite values.")
    if scale <= 0 or voxel_res_fraction <= 0 or padding_scales < 0:
        raise ValueError("scale and voxel_res_fraction must be positive; padding_scales cannot be negative.")

    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    extent = float(np.max(maximum - minimum))
    if extent <= 0:
        raise ValueError("positions must span a non-zero spatial extent.")
    voxel_resolution = extent * float(voxel_res_fraction)
    bounds = np.vstack(
        (minimum - padding_scales * scale, maximum + padding_scales * scale)
    ).T
    ticks = tuple(
        torch.arange(axis[0], axis[1], voxel_resolution, device=device)
        for axis in bounds
    )
    if any(axis.numel() == 0 for axis in ticks):
        raise ValueError("grid bounds and voxel resolution produced an empty axis.")
    nx, ny, nz = (int(axis.numel()) for axis in ticks)
    return {
        "x_ticks": ticks[0],
        "y_ticks": ticks[1],
        "z_ticks": ticks[2],
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "voxel_res": voxel_resolution,
        "total_voxels": nx * ny * nz,
    }


def sample_isotropic_density_grid(
    positions: np.ndarray,
    scale: float,
    grid: dict[str, Any],
    *,
    batch_size: int = 50_000,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Evaluate a unit-weight isotropic mixture on ``grid`` in batches.

    The returned one-dimensional float32 density tensor is stored on CPU so
    callers can cache it without retaining GPU memory.  Reshape it with the
    grid's ``nx``, ``ny``, and ``nz`` values for a three-dimensional field.
    """
    if scale <= 0 or batch_size < 1:
        raise ValueError("scale and batch_size must be positive.")
    points = np.asarray(positions)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("positions must be a non-empty array shaped (agents, 3).")

    try:
        x_ticks = grid["x_ticks"]
        y_ticks = grid["y_ticks"]
        z_ticks = grid["z_ticks"]
        nx, ny, nz = (int(grid[key]) for key in ("nx", "ny", "nz"))
        total = int(grid["total_voxels"])
    except KeyError as error:
        raise ValueError(f"grid is missing required key {error.args[0]!r}.") from error
    if total != nx * ny * nz:
        raise ValueError("grid total_voxels does not match its dimensions.")

    selected_device = torch.device(device)
    means = torch.as_tensor(points, dtype=torch.float32, device=selected_device)
    weights = torch.ones(points.shape[0], dtype=torch.float32, device=selected_device)
    sigmas = torch.full((points.shape[0],), float(scale), dtype=torch.float32, device=selected_device)
    density = torch.empty(total, dtype=torch.float32, device="cpu")

    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        indices = torch.arange(start, stop, device=selected_device)
        x_index = indices // (ny * nz)
        y_index = (indices // nz) % ny
        z_index = indices % nz
        coordinates = torch.stack(
            [x_ticks[x_index], y_ticks[y_index], z_ticks[z_index]], dim=-1
        )
        density[start:stop] = evaluate_isotropic_gmm(
            coordinates, means, weights, sigmas
        ).cpu()
    return density
