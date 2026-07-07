"""Analysis-result plotting primitives."""

from __future__ import annotations

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
