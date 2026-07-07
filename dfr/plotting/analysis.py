"""Analysis-result plotting primitives."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from dfr.analysis import select_adaptive_density_scales, validate_nnd_bounds


def plot_mode_count_curve(
    normalized_scales,
    mode_counts,
    *,
    dataset_name: Optional[str] = None,
    frame: Optional[int] = None,
    number_of_agents: Optional[int] = None,
    n_slices: int = 4,
    slice_relative_positions=None,
    nnd_bounds=None,
    ax=None,
):
    """Plot a log-log empirical mode-count curve.

    Parameters are data-only so callers can pass a ``ModeCurveResult``'s arrays,
    legacy cache arrays, or freshly computed values. The function returns the
    created ``(Figure, Axes)`` and never saves by itself.
    """
    scales = np.asarray(normalized_scales, dtype=float)
    counts = np.asarray(mode_counts, dtype=float)
    if scales.ndim != 1 or counts.shape != scales.shape:
        raise ValueError("normalized_scales and mode_counts must be equal-length 1D arrays.")
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("normalized_scales must contain positive finite values.")
    if np.any(~np.isfinite(counts)) or np.any(counts < 1):
        raise ValueError("mode_counts must contain finite values of at least one.")
    if np.any(np.diff(scales) <= 0):
        raise ValueError("normalized_scales must be strictly increasing.")
    if not isinstance(n_slices, (int, np.integer)) or n_slices < 1:
        raise ValueError("n_slices must be a positive integer.")

    lower, upper = validate_nnd_bounds(nnd_bounds or (scales[0], scales[-1]))
    if lower < scales[0] or upper > scales[-1]:
        raise ValueError("nnd_bounds must lie inside the normalized scale range.")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 13,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        }
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    else:
        fig = ax.figure

    label = _curve_label(dataset_name, frame, number_of_agents)
    ax.plot(scales, counts, color="#2c3e50", lw=2, label=label)

    selected_indices, selected_scales = select_adaptive_density_scales(
        scales,
        counts,
        n_selected=int(n_slices),
        relative_positions=slice_relative_positions,
    )
    slice_colours = plt.get_cmap(
        "tab10" if n_slices <= 10 else "turbo",
        n_slices,
    )(np.arange(n_slices))
    for i, (index, selected_scale, colour) in enumerate(
        zip(selected_indices, selected_scales, slice_colours),
        start=1,
    ):
        ax.plot(
            selected_scale,
            counts[index],
            marker="o",
            linestyle="none",
            markersize=9,
            markerfacecolor=colour,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=f"Slice {i} ({selected_scale:.3f} x NND)",
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_xlabel(r"Normalized scale ($\sigma / \mathrm{NND}$)")
    ax.set_ylabel("Number of Modes")

    if lower <= 1.0 <= upper:
        x_axis_transform = ax.get_xaxis_transform()
        ax.plot(
            1.0,
            0.055,
            marker="v",
            markersize=10,
            color="black",
            linestyle="none",
            transform=x_axis_transform,
            clip_on=False,
            zorder=4,
        )
        ax.text(
            1.0,
            0.105,
            "NND",
            color="black",
            fontsize=15,
            fontweight="semibold",
            ha="center",
            va="bottom",
            transform=x_axis_transform,
            zorder=4,
        )

    ax.legend(
        loc="best",
        ncol=2,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.0,
    )
    fig.tight_layout()
    return fig, ax


def plot_dra_scale_model_order_surface(
    normalized_scales,
    component_counts,
    dra,
    *,
    number_of_animals: int,
    fitted_dra=None,
    title: Optional[str] = None,
    ax=None,
    surface_alpha: float = 0.88,
    wireframe_label: Optional[str] = None,
    z_label: str = "DRA",
    z_label_as_text: bool = False,
    max_model_order_ticks: Optional[int] = None,
    include_component_counts_in_ticks: bool = False,
):
    """Plot one DRA scale/model-order surface and optional fitted wireframe."""
    scales, components, surface_values = _dra_surface_arrays(
        normalized_scales,
        component_counts,
        dra,
        number_of_animals=number_of_animals,
    )
    fitted_values = None
    if fitted_dra is not None:
        fitted_values = np.asarray(fitted_dra, dtype=float)
        if fitted_values.shape != surface_values.shape:
            raise ValueError("fitted_dra must have the same shape as dra.")

    order_percentages = 100.0 * components / int(number_of_animals)
    scale_grid, order_grid = np.meshgrid(scales, order_percentages, indexing="ij")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 11,
        }
    )

    if ax is None:
        fig = plt.figure(figsize=(10, 8), dpi=300)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    surface = ax.plot_surface(
        scale_grid,
        order_grid,
        surface_values,
        cmap="viridis",
        edgecolor="none",
        antialiased=True,
        alpha=float(surface_alpha),
    )
    if fitted_values is not None:
        ax.plot_wireframe(
            scale_grid,
            order_grid,
            fitted_values,
            color="black",
            linewidth=0.75,
            rstride=1,
            cstride=1,
            label=wireframe_label,
        )

    if title:
        ax.set_title(title)
    ax.set_xlabel(r"Normalized scale ($\sigma / \mathrm{NND}$)", labelpad=10)
    ax.set_ylabel("Model order / N (%)", labelpad=12)
    _set_dra_z_label(ax, z_label, z_label_as_text)
    ax.set_xlim(float(scales[0]), float(scales[-1]))
    _set_model_order_ticks(
        ax,
        order_percentages,
        components,
        max_ticks=max_model_order_ticks,
        include_component_counts=include_component_counts_in_ticks,
    )
    ax.view_init(elev=28, azim=-130)
    if wireframe_label:
        ax.legend(loc="upper left", frameon=False)
    return fig, ax, surface


def plot_dra_surface_grid(
    results: Mapping[str, tuple],
    fits: Mapping[str, dict],
    *,
    columns: int = 2,
):
    """Plot a grid of DRA scale/model-order surfaces from legacy result tuples."""
    if not results:
        raise ValueError("results must contain at least one DRA surface.")
    if columns < 1:
        raise ValueError("columns must be positive.")

    rows = int(np.ceil(len(results) / columns))
    figure = plt.figure(figsize=(7.5 * columns, 5.5 * rows))
    axes = []
    surface = None
    for plot_index, (dataset_name, result) in enumerate(results.items(), start=1):
        normalized_scales, _, components, dra, _, number_of_animals = result
        fit = fits[dataset_name]
        best = fit["candidates"][fit["best_name"]]
        title = (
            f"{dataset_name.capitalize()} ({fit['best_name']}, "
            f"$R^2$={best['r_squared']:.3f})"
        )
        axis = figure.add_subplot(rows, columns, plot_index, projection="3d")
        _, axis, surface = plot_dra_scale_model_order_surface(
            normalized_scales,
            components,
            dra,
            number_of_animals=number_of_animals,
            fitted_dra=best["prediction"],
            title=title,
            ax=axis,
            surface_alpha=0.88,
            max_model_order_ticks=None,
            include_component_counts_in_ticks=True,
        )
        axis.set_xlabel("Scale / mean NND")
        axis.set_zlabel("DRA")
        axes.append(axis)

    if surface is not None:
        figure.colorbar(surface, ax=figure.axes, shrink=0.62, pad=0.06, label="DRA")
    figure.subplots_adjust(
        left=0.02,
        right=0.88,
        bottom=0.04,
        top=0.94,
        wspace=0.03,
        hspace=0.12,
    )
    return figure, axes


def _curve_label(
    dataset_name: Optional[str],
    frame: Optional[int],
    number_of_agents: Optional[int],
) -> str:
    parts = []
    if dataset_name:
        parts.append(str(dataset_name))
    if frame is not None:
        parts.append(f"frame {int(frame)}")
    if number_of_agents is not None:
        parts.append(f"(N={int(number_of_agents)})")
    return " ".join(parts) if parts else "Mode count"


def _dra_surface_arrays(
    normalized_scales,
    component_counts,
    dra,
    *,
    number_of_animals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scales = np.asarray(normalized_scales, dtype=float)
    components = np.asarray(component_counts, dtype=float)
    values = np.asarray(dra, dtype=float)
    if scales.ndim != 1 or len(scales) < 2:
        raise ValueError("normalized_scales must be a 1D array with at least two values.")
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("normalized_scales must contain positive finite values.")
    if np.any(np.diff(scales) <= 0):
        raise ValueError("normalized_scales must be strictly increasing.")
    if components.ndim != 1 or len(components) == 0:
        raise ValueError("component_counts must be a non-empty 1D array.")
    if np.any(~np.isfinite(components)) or np.any(components <= 0):
        raise ValueError("component_counts must contain positive finite values.")
    if int(number_of_animals) < 1:
        raise ValueError("number_of_animals must be positive.")
    if values.shape != (len(scales), len(components)):
        raise ValueError(
            "dra must have shape (len(normalized_scales), len(component_counts))."
        )
    if np.any(~np.isfinite(values)):
        raise ValueError("dra must contain finite values.")
    return scales, components, values


def _set_dra_z_label(ax, label: str, as_text: bool) -> None:
    if not as_text:
        ax.set_zlabel(label)
        return
    ax.set_zlabel("")
    ax.text2D(
        -0.08,
        0.50,
        label,
        transform=ax.transAxes,
        rotation=90,
        fontsize=16,
        ha="center",
        va="center",
        clip_on=False,
    )


def _set_model_order_ticks(
    ax,
    percentages: np.ndarray,
    components: np.ndarray,
    *,
    max_ticks: Optional[int],
    include_component_counts: bool,
) -> None:
    if max_ticks is not None:
        tick_indices = np.unique(
            np.rint(np.linspace(0, len(percentages) - 1, min(max_ticks, len(percentages))))
            .astype(int)
        )
    else:
        tick_indices = np.arange(len(percentages), dtype=int)
    ticks = percentages[tick_indices]
    ax.set_yticks(ticks)
    if include_component_counts:
        labels = [
            f"{percentage:.1f}\n({int(component)})"
            for percentage, component in zip(ticks, components[tick_indices])
        ]
        ax.set_yticklabels(labels, fontsize=7)
    else:
        ax.set_yticklabels([f"{percentage:.1f}" for percentage in ticks])
