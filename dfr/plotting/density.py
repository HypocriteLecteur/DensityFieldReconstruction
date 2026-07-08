"""3D density-field plotting primitives."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from dfr.plotting.style import apply_figure_layout, prepare_3d_axis


DEFAULT_DENSITY_LAYERS = [
    {"thresh_frac": 0.10, "alpha_min": 0.45, "alpha_max": 0.95, "size": 8},
    {"thresh_frac": 0.02, "alpha_min": 0.25, "alpha_max": 0.80, "size": 6},
    {"thresh_frac": 0.002, "alpha_min": 0.08, "alpha_max": 0.50, "size": 4},
]

FIELD_DENSITY_LAYERS = [
    {"thresh_frac": 0.10, "alpha_min": 0.18, "alpha_max": 0.55, "size": 8},
    {"thresh_frac": 0.02, "alpha_min": 0.10, "alpha_max": 0.40, "size": 6},
    {"thresh_frac": 0.002, "alpha_min": 0.04, "alpha_max": 0.22, "size": 4},
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


def render_gmm_wireframes(
    ax,
    means,
    sigmas,
    weights,
    *,
    colour: str = "#4169e1",
    z_sort_pos: float = -5e8,
    sphere_res: int = 20,
) -> list:
    """Draw isotropic GMM components as depth-ordered wireframe spheres."""
    means_array, sigmas_array, weights_array = _gmm_components(means, sigmas, weights)
    if sphere_res < 4:
        raise ValueError("sphere_res must be at least 4.")
    if len(means_array) == 0:
        return []

    weight_max = float(weights_array.max()) if weights_array.size else 1.0
    u = np.linspace(0, 2 * np.pi, sphere_res)
    v = np.linspace(0, np.pi, sphere_res)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))

    wireframes = []
    for mean, sigma, weight in zip(means_array, sigmas_array, weights_array):
        alpha = (
            max(0.15, min(0.70, float(weight) / weight_max))
            if weight_max > 0
            else 0.25
        )
        rgba = (*mcolors.to_rgb(colour), alpha)
        wireframe = ax.plot_wireframe(
            mean[0] + float(sigma) * sphere_x,
            mean[1] + float(sigma) * sphere_y,
            mean[2] + float(sigma) * sphere_z,
            color=rgba,
            rstride=2,
            cstride=2,
            linewidth=1.7,
        )
        _force_depth_order(wireframe, z_sort_pos)
        wireframes.append(wireframe)
    return wireframes


def render_gmm_means(
    ax,
    means,
    *,
    colour: str = "#4169e1",
    size: float = 14,
    alpha: float = 0.85,
    z_sort_pos: float = -6e8,
):
    """Overlay GMM mean positions, sorted in front of wireframes."""
    means_array = _positions(means, allow_empty=True)
    collection = ax.scatter(
        means_array[:, 0],
        means_array[:, 1],
        means_array[:, 2],
        c=colour,
        marker="o",
        s=size,
        alpha=alpha,
        edgecolors="none",
        depthshade=True,
    )
    _force_depth_order(collection, z_sort_pos)
    return collection


def render_density_field_3d(
    ax,
    density_3d,
    x_ticks,
    y_ticks,
    z_ticks,
    positions,
    *,
    max_density: Optional[float] = None,
    layers: Optional[Sequence[dict]] = None,
) -> None:
    """Render a GT-style density field: density shells plus agent overlay."""
    density = _density(density_3d)
    x_axis, y_axis, z_axis = _ticks(x_ticks, y_ticks, z_ticks, density.shape)
    points = _positions(positions)
    maximum = float(np.max(density)) if max_density is None else float(max_density)
    render_density_shells(
        ax,
        density,
        x_axis,
        y_axis,
        z_axis,
        max_density=maximum,
        layers=layers,
    )
    render_agent_positions(ax, points)


def render_reconstructed_gmm_3d(
    ax,
    density_3d,
    x_ticks,
    y_ticks,
    z_ticks,
    positions,
    means,
    sigmas,
    weights,
    *,
    max_density: Optional[float] = None,
    gmm_colour: str = "#4169e1",
) -> None:
    """Render density shells plus reconstructed isotropic GMM wireframes."""
    density = _density(density_3d)
    x_axis, y_axis, z_axis = _ticks(x_ticks, y_ticks, z_ticks, density.shape)
    points = _positions(positions)
    means_array, sigmas_array, weights_array = _gmm_components(means, sigmas, weights)
    maximum = float(np.max(density)) if max_density is None else float(max_density)
    render_density_shells(
        ax,
        density,
        x_axis,
        y_axis,
        z_axis,
        max_density=maximum,
        layers=FIELD_DENSITY_LAYERS,
    )
    render_gmm_wireframes(
        ax,
        means_array,
        sigmas_array,
        weights_array,
        colour=gmm_colour,
        z_sort_pos=-5e8,
    )
    render_gmm_means(ax, means_array, colour=gmm_colour, z_sort_pos=-6e8)
    render_agent_positions(ax, points, z_sort_pos=-1e9)


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

    prepare_3d_axis(ax, view=view, axis_off=axis_off)

    render_density_field_3d(
        ax,
        density,
        x_axis,
        y_axis,
        z_axis,
        points,
        layers=layers,
    )
    if normalized_scale is not None and mode_count is not None:
        ax.text2D(
            0.02,
            0.98,
            f"{float(normalized_scale):.3f} x NND ({int(mode_count)} modes)",
            transform=ax.transAxes,
            fontsize=16,
            va="top",
        )
    apply_figure_layout(fig, pad=0)
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


def _positions(values, *, allow_empty: bool = False) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("positions must have shape (agents, 3).")
    if len(points) == 0 and not allow_empty:
        raise ValueError("positions must not be empty.")
    if np.any(~np.isfinite(points)):
        raise ValueError("positions must contain finite values.")
    return points


def _gmm_components(means, sigmas, weights) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means_array = _positions(means, allow_empty=True)
    sigmas_array = np.asarray(sigmas, dtype=float).reshape(-1)
    weights_array = np.asarray(weights, dtype=float).reshape(-1)
    if sigmas_array.shape != (len(means_array),):
        raise ValueError("sigmas must contain one value per GMM mean.")
    if weights_array.shape != (len(means_array),):
        raise ValueError("weights must contain one value per GMM mean.")
    if np.any(~np.isfinite(sigmas_array)) or np.any(sigmas_array <= 0):
        raise ValueError("sigmas must contain positive finite values.")
    if np.any(~np.isfinite(weights_array)) or np.any(weights_array < 0):
        raise ValueError("weights must contain finite non-negative values.")
    return means_array, sigmas_array, weights_array


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

def _force_depth_order(artist, z_sort_pos: float) -> None:
    original_projection = artist.do_3d_projection

    def _force_projection(zpos=z_sort_pos, orig=original_projection, obj=artist):
        orig()
        obj._sort_zpos = zpos
        return obj._sort_zpos

    artist.do_3d_projection = _force_projection
