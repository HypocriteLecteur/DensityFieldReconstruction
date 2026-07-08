"""Trajectory plotting primitives."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from dfr.plotting.style import apply_figure_layout, prepare_3d_axis


def plot_trajectory_snapshot(
    trajectories,
    positions,
    *,
    ax=None,
    view: tuple[float, float, float] | None = (33, -117, 0),
    axis_off: bool = True,
    line_color: str = "tab:gray",
    line_alpha: float = 0.15,
    line_width: float = 1.2,
    point_color: str = "#1f2937",
    point_size: float = 25,
    point_alpha: float = 0.65,
):
    """Plot 3D agent trajectories with a final-position snapshot.

    Parameters
    ----------
    trajectories:
        Array with shape ``(frames, agents, 3)``.
    positions:
        Final or highlighted positions with shape ``(agents, 3)``.
    ax:
        Optional existing 3D axes. A new figure/axes pair is created when
        omitted.
    view:
        Optional ``(elev, azim, roll)`` camera view. Set to ``None`` to keep the
        axes' current view.
    axis_off:
        Hide axes and grid when true.

    Returns
    -------
    tuple
        ``(fig, ax)``. The function does not save or display the figure.
    """

    paths = _trajectories(trajectories)
    points = _positions(positions, paths.shape[1])
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    prepare_3d_axis(ax, view=view, axis_off=axis_off)
    apply_figure_layout(fig, pad=0)

    for index in range(paths.shape[1]):
        ax.plot(
            paths[:, index, 0],
            paths[:, index, 1],
            paths[:, index, 2],
            color=line_color,
            alpha=line_alpha,
            linewidth=line_width,
            zorder=1,
        )
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=point_color,
        s=point_size,
        alpha=point_alpha,
        edgecolors="none",
        depthshade=True,
        zorder=3,
    )
    return fig, ax

def _trajectories(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("trajectories must have shape (frames, agents, 3).")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("trajectories must contain at least one frame and agent.")
    return array


def _positions(values, agent_count: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("positions must have shape (agents, 3).")
    if len(array) != agent_count:
        raise ValueError("positions must contain one point per trajectory agent.")
    return array
