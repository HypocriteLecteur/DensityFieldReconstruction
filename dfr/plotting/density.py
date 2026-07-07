"""3D density-field plotting primitives."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DENSITY_LAYERS = [
    {"thresh_frac": 0.10, "alpha_min": 0.45, "alpha_max": 0.95, "size": 8},
    {"thresh_frac": 0.02, "alpha_min": 0.25, "alpha_max": 0.80, "size": 6},
    {"thresh_frac": 0.002, "alpha_min": 0.08, "alpha_max": 0.50, "size": 4},
]


def render_density_shells(
    ax,
    density_3d,
    x_ticks,
    y_ticks,
    z_ticks,
    *,
    max_density: Optional[float] = None,
    layers: Optional[Sequence[dict]] = None,
) -> None:
    """Render a 3D density field as nested semi-transparent scatter shells."""
    density = _density(density_3d)
    x_axis, y_axis, z_axis = _ticks(x_ticks, y_ticks, z_ticks, density.shape)
    maximum = float(np.max(density)) if max_density is None else float(max_density)
    if not np.isfinite(maximum) or maximum <= 0:
        return

    norm = mcolors.PowerNorm(gamma=0.35, vmin=0, vmax=maximum)
    for layer in _density_layers(maximum, layers):
        mask = density >= layer["thresh"]
        if not np.any(mask):
            continue
        ix, iy, iz = np.where(mask)
        points = np.stack([x_axis[ix], y_axis[iy], z_axis[iz]], axis=-1)
        values = density[mask]
        colours = plt.cm.viridis(norm(values))
        alphas = (
            norm(values) * (layer["alpha_max"] - layer["alpha_min"])
            + layer["alpha_min"]
        )
        colours[:, 3] = np.clip(alphas, layer["alpha_min"], layer["alpha_max"])
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=colours,
            s=layer["size"],
            edgecolors="none",
            depthshade=False,
            rasterized=True,
        )


def render_agent_positions(
    ax,
    positions,
    *,
    colour: str = "#1f2937",
    size: float = 25,
    alpha: float = 1.0,
    z_sort_pos: float = -1e9,
):
    """Overlay 3D agent positions on top of density shells."""
    points = _positions(positions)
    collection = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colour,
        s=size,
        alpha=alpha,
        linewidths=0.8,
    )
    original_projection = collection.do_3d_projection

    def _force_projection(zpos=z_sort_pos, orig=original_projection, obj=collection):
        orig()
        obj._sort_zpos = zpos
        return obj._sort_zpos

    collection.do_3d_projection = _force_projection
    return collection


def plot_density_field_3d(
    density_3d,
    x_ticks,
    y_ticks,
    z_ticks,
    positions,
    *,
    normalized_scale: Optional[float] = None,
    mode_count: Optional[int] = None,
    ax=None,
    view: Optional[tuple[float, float, float]] = (33, -117, 0),
    figsize: tuple[float, float] = (10, 10),
    axis_off: bool = True,
    layers: Optional[Sequence[dict]] = None,
):
    """Plot one 3D density field with an optional scale/mode annotation."""
    density = _density(density_3d)
    x_axis, y_axis, z_axis = _ticks(x_ticks, y_ticks, z_ticks, density.shape)
    points = _positions(positions)
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    if view is not None:
        _set_view(ax, view)
    if axis_off:
        ax.set_axis_off()

    render_density_shells(
        ax,
        density,
        x_axis,
        y_axis,
        z_axis,
        layers=layers,
    )
    render_agent_positions(ax, points)
    if normalized_scale is not None and mode_count is not None:
        ax.text2D(
            0.02,
            0.98,
            f"{float(normalized_scale):.3f} x NND ({int(mode_count)} modes)",
            transform=ax.transAxes,
            fontsize=16,
            va="top",
        )
    fig.tight_layout(pad=0)
    return fig, ax


def plot_multiscale_density_fields(
    density_data: Iterable[dict],
    positions,
    normalized_scales,
    mode_counts,
    *,
    view: Optional[tuple[float, float, float]] = (33, -117, 0),
    figsize: tuple[float, float] = (10, 10),
) -> list[tuple[plt.Figure, plt.Axes]]:
    """Plot one 3D density figure for each selected scale."""
    scales = np.asarray(normalized_scales, dtype=float)
    counts = np.asarray(mode_counts, dtype=int).reshape(-1)
    data = list(density_data)
    if len(data) != len(scales) or len(data) != len(counts):
        raise ValueError(
            "density_data, normalized_scales, and mode_counts must have equal length."
        )
    figures = []
    for item, scale, count in zip(data, scales, counts):
        figures.append(
            plot_density_field_3d(
                item["density"],
                item["x_ticks"],
                item["y_ticks"],
                item["z_ticks"],
                positions,
                normalized_scale=scale,
                mode_count=int(count),
                view=view,
                figsize=figsize,
            )
        )
    return figures


def _density(values) -> np.ndarray:
    density = np.asarray(values, dtype=float)
    if density.ndim != 3:
        raise ValueError("density_3d must have shape (nx, ny, nz).")
    if np.any(~np.isfinite(density)):
        raise ValueError("density_3d must contain finite values.")
    return density


def _ticks(x_ticks, y_ticks, z_ticks, shape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = tuple(np.asarray(values, dtype=float).reshape(-1) for values in (x_ticks, y_ticks, z_ticks))
    if tuple(len(axis) for axis in axes) != tuple(shape):
        raise ValueError("tick arrays must match the density dimensions.")
    if any(np.any(~np.isfinite(axis)) for axis in axes):
        raise ValueError("tick arrays must contain finite values.")
    return axes


def _positions(values) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("positions must have shape (agents, 3).")
    if len(points) == 0:
        raise ValueError("positions must not be empty.")
    if np.any(~np.isfinite(points)):
        raise ValueError("positions must contain finite values.")
    return points


def _density_layers(max_density: float, layers: Optional[Sequence[dict]]) -> list[dict]:
    selected = DEFAULT_DENSITY_LAYERS if layers is None else layers
    resolved = []
    for layer in selected:
        resolved.append(
            {
                **layer,
                "thresh": float(layer["thresh_frac"]) * max_density,
            }
        )
    return resolved


def _set_view(ax, view: tuple[float, float, float]) -> None:
    elev, azim, roll = view
    try:
        ax.view_init(elev=elev, azim=azim, roll=roll)
    except TypeError:
        ax.view_init(elev=elev, azim=azim)
