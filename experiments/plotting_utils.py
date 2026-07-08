"""Shared 3D rendering utilities for density field and GMM visualisation.

Extracted from ``experiments/dfr_plot.py`` and
``experiments/run_scenarios_angle_sweep.py`` to eliminate duplication between
``plot_jackdaw2_density_field``, ``_draw_gmm_frame``, and the new multi-scale
density figures.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from dfr.plotting import (
    DEFAULT_DENSITY_LAYERS,
    FIELD_DENSITY_LAYERS,
    apply_academic_style,
    render_agent_positions as _render_agent_positions,
    render_density_field_3d as _render_density_field_3d,
    render_density_shells as _render_density_shells,
    render_gmm_means as _render_gmm_means,
    render_gmm_wireframes as _render_gmm_wireframes,
    render_reconstructed_gmm_3d as _render_reconstructed_gmm_3d,
    style_3d_axis,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Matplotlib style helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _set_academic_style() -> None:
    """Apply publication-quality academic styling to matplotlib."""
    apply_academic_style(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "savefig.pad_inches": 0.1,
        }
    )


def _style_3d_ax(ax: plt.Axes) -> None:
    """Transparent panes, subtle grid, no edge colour on a 3D axis."""
    style_3d_axis(ax)


# ═══════════════════════════════════════════════════════════════════════════════
#  Voxel grid construction & density evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def build_voxel_grid(
    positions: np.ndarray,
    scale: float,
    voxel_res_factor: float = 2.5e-2,
    device: str = "cuda",
) -> dict:
    """Create the 3D voxel grid parameters for density evaluation.

    Parameters
    ----------
    positions : (N, 3) float32
        Agent positions.
    scale : float
        Gaussian sigma used to pad the bounding box (3σ on each side).
    voxel_res_factor : float
        Voxel width as a fraction of the largest spatial extent.
    device : str
        Torch device for the tick tensors.

    Returns
    -------
    dict with keys: x_ticks, y_ticks, z_ticks, nx, ny, nz, voxel_res,
    total_voxels.
    """
    min_c = np.min(positions, axis=0)
    max_c = np.max(positions, axis=0)
    extent = np.max(max_c - min_c)
    voxel_res = extent * voxel_res_factor
    bounds = np.vstack((min_c - 3 * scale, max_c + 3 * scale)).T

    x_ticks = torch.arange(bounds[0, 0], bounds[0, 1], voxel_res, device=device)
    y_ticks = torch.arange(bounds[1, 0], bounds[1, 1], voxel_res, device=device)
    z_ticks = torch.arange(bounds[2, 0], bounds[2, 1], voxel_res, device=device)

    return {
        "x_ticks": x_ticks,
        "y_ticks": y_ticks,
        "z_ticks": z_ticks,
        "nx": len(x_ticks),
        "ny": len(y_ticks),
        "nz": len(z_ticks),
        "voxel_res": voxel_res,
        "total_voxels": len(x_ticks) * len(y_ticks) * len(z_ticks),
    }


def compute_gt_density(
    positions: np.ndarray,
    scale: float,
    grid: dict,
    batch_size: int = 50000,
    device: str = "cuda",
) -> torch.Tensor:
    """Evaluate GT GMM (one isotropic Gaussian per agent) on the voxel grid.

    Returns a flat float32 tensor on CPU so it does not exhaust GPU memory.
    """
    from dfr.utils import eval_isotropic_gmm_torch

    N = positions.shape[0]
    gt_means = torch.from_numpy(positions).float().to(device)
    gt_weights = torch.full((N,), 1.0, device=device, dtype=torch.float)
    gt_sigmas = torch.full((N,), scale, device=device, dtype=torch.float)

    x_t = grid["x_ticks"]
    y_t = grid["y_ticks"]
    z_t = grid["z_ticks"]
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    total = grid["total_voxels"]

    density_flat = torch.empty(total, dtype=torch.float32, device="cpu")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        idx = torch.arange(start, end, device=device)
        ix = idx // (ny * nz)
        iy = (idx // nz) % ny
        iz = idx % nz
        coords = torch.stack([x_t[ix], y_t[iy], z_t[iz]], dim=-1)
        dens = eval_isotropic_gmm_torch(coords, gt_means, gt_weights, gt_sigmas)
        density_flat[start:end] = dens.cpu()

    return density_flat  # on CPU


# ═══════════════════════════════════════════════════════════════════════════════
#  3D scatter-based density rendering primitives
# ═══════════════════════════════════════════════════════════════════════════════

# Layer aliases kept for older experiment imports.
DEFAULT_LAYERS = DEFAULT_DENSITY_LAYERS
FIELD_LAYERS = FIELD_DENSITY_LAYERS


def _layer_thresholds(
    max_density: float, layers: Sequence[dict] | None = None,
) -> list[dict]:
    """Resolve layer config dicts, converting *thresh_frac* → absolute *thresh*."""
    if layers is None:
        layers = DEFAULT_LAYERS
    return [
        {**lyr, "thresh": lyr["thresh_frac"] * max_density} for lyr in layers
    ]


def render_density_shells(
    ax: plt.Axes,
    density_3d: np.ndarray,
    x_ticks_np: np.ndarray,
    y_ticks_np: np.ndarray,
    z_ticks_np: np.ndarray,
    max_density: float | None = None,
    layers: Sequence[dict] | None = None,
) -> None:
    """Compatibility wrapper for :func:`dfr.plotting.render_density_shells`."""
    _render_density_shells(
        ax,
        density_3d,
        x_ticks_np,
        y_ticks_np,
        z_ticks_np,
        max_density=max_density,
        layers=layers,
    )


def render_gmm_wireframes(
    ax: plt.Axes,
    means_np: np.ndarray,
    sigmas_np: np.ndarray,
    weights_np: np.ndarray,
    colour: str = "#4169e1",
    z_sort_pos: float = -5e8,
    sphere_res: int = 20,
) -> None:
    """Compatibility wrapper for :func:`dfr.plotting.render_gmm_wireframes`."""
    _render_gmm_wireframes(
        ax,
        means_np,
        sigmas_np,
        weights_np,
        colour=colour,
        z_sort_pos=z_sort_pos,
        sphere_res=sphere_res,
    )


def render_agent_positions(
    ax: plt.Axes,
    positions: np.ndarray,
    colour: str = "#1f2937",
    size: float = 25,
    alpha: float = 1.0,
    z_sort_pos: float = -1e9,
) -> None:
    """Compatibility wrapper for :func:`dfr.plotting.render_agent_positions`."""
    return _render_agent_positions(
        ax,
        positions,
        colour=colour,
        size=size,
        alpha=alpha,
        z_sort_pos=z_sort_pos,
    )


def render_gmm_means(
    ax: plt.Axes,
    means_np: np.ndarray,
    colour: str = "#4169e1",
    size: float = 14,
    alpha: float = 0.85,
    z_sort_pos: float = -6e8,
) -> None:
    """Compatibility wrapper for :func:`dfr.plotting.render_gmm_means`."""
    return _render_gmm_means(
        ax,
        means_np,
        colour=colour,
        size=size,
        alpha=alpha,
        z_sort_pos=z_sort_pos,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Composite renderers
# ═══════════════════════════════════════════════════════════════════════════════

def render_density_field_3d(
    ax: plt.Axes,
    density_3d: np.ndarray,
    x_ticks_np: np.ndarray,
    y_ticks_np: np.ndarray,
    z_ticks_np: np.ndarray,
    positions: np.ndarray,
    max_density: float | None = None,
    layers: Sequence[dict] | None = None,
) -> None:
    """Compatibility wrapper for :func:`dfr.plotting.render_density_field_3d`."""
    _render_density_field_3d(
        ax,
        density_3d,
        x_ticks_np,
        y_ticks_np,
        z_ticks_np,
        positions,
        max_density=max_density,
        layers=layers,
    )


def render_reconstructed_gmm_3d(
    ax: plt.Axes,
    density_3d: np.ndarray,
    x_ticks_np: np.ndarray,
    y_ticks_np: np.ndarray,
    z_ticks_np: np.ndarray,
    positions: np.ndarray,
    means_np: np.ndarray,
    sigmas_np: np.ndarray,
    weights_np: np.ndarray,
    max_density: float | None = None,
    gmm_colour: str = "#4169e1",
) -> None:
    """Compatibility wrapper for :func:`dfr.plotting.render_reconstructed_gmm_3d`."""
    _render_reconstructed_gmm_3d(
        ax,
        density_3d,
        x_ticks_np,
        y_ticks_np,
        z_ticks_np,
        positions,
        means_np,
        sigmas_np,
        weights_np,
        max_density=max_density,
        gmm_colour=gmm_colour,
    )
