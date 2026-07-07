"""Camera-layout plotting primitives."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from dfr.plotting.style import apply_academic_style


DEFAULT_CAMERA_STYLES = {
    2: {"color": "#D55E00", "marker": "o", "label": r"$K = 2$"},
    3: {"color": "#0072B2", "marker": "o", "label": r"$K = 3$"},
    5: {"color": "#009E73", "marker": "o", "label": r"$K = 5$"},
}


def plot_camera_configurations(
    positions,
    camera_positions: Mapping[int, np.ndarray],
    *,
    center: Optional[np.ndarray] = None,
    swarm_radius: Optional[float] = None,
    orbit_radius: Optional[float] = None,
    styles: Optional[Mapping[int, Mapping[str, object]]] = None,
    ax=None,
    apply_style: bool = True,
):
    """Plot a top-down swarm view with one or more camera configurations.

    Parameters
    ----------
    positions:
        Agent positions with shape ``(agents, 2)`` or ``(agents, 3)``.
    camera_positions:
        Mapping from camera-count label to camera positions, each shaped
        ``(cameras, 2)`` or ``(cameras, 3)``.
    center:
        Optional swarm center. If omitted, the center of the position bounding
        box is used.
    swarm_radius:
        Optional radius of the dashed swarm extent circle. If omitted, the
        maximum Euclidean distance from ``center`` is used.
    orbit_radius:
        Optional shared camera-orbit radius. If omitted, it is inferred from
        the farthest supplied camera.
    styles:
        Optional per-camera-count style overrides. Each style may contain
        ``color``, ``marker``, and ``label`` keys.
    ax:
        Existing axes to draw on. When omitted, a new square figure is created.
    apply_style:
        Apply DFR's academic Matplotlib rcParams before creating a new figure.

    Returns
    -------
    tuple
        ``(fig, ax)``. The function does not save or display the figure.
    """

    points = _points("positions", positions)
    cameras = {
        int(count): _points(f"camera_positions[{count}]", values)
        for count, values in camera_positions.items()
    }
    if not cameras:
        raise ValueError("camera_positions must contain at least one configuration.")

    center_array = _center(points, center)
    radius = _swarm_radius(points, center_array, swarm_radius)
    orbit = _orbit_radius(cameras, center_array, orbit_radius)
    resolved_styles = _styles(cameras.keys(), styles)

    if apply_style:
        apply_academic_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 7.2))
    else:
        fig = ax.figure

    center_xy = center_array[:2]
    theta = np.linspace(0, 2 * np.pi, 600)
    ax.plot(
        center_xy[0] + orbit * np.cos(theta),
        center_xy[1] + orbit * np.sin(theta),
        color="#E5E7EB",
        linewidth=1.0,
        zorder=0,
    )
    ax.scatter(
        points[:, 0],
        points[:, 1],
        c="#9CA3AF",
        s=1.0,
        alpha=0.35,
        edgecolors="none",
        zorder=1,
    )
    ax.add_patch(
        plt.Circle(
            (center_xy[0], center_xy[1]),
            radius,
            fill=False,
            color="#CBD5E0",
            linewidth=0.8,
            linestyle=(0, (4, 5)),
            alpha=0.65,
            zorder=2,
        )
    )

    for count in sorted(cameras):
        camera_xy = cameras[count][:, :2]
        style = resolved_styles[count]
        color = str(style["color"])
        for camera in camera_xy:
            vector = camera - center_xy
            distance = float(np.linalg.norm(vector))
            if distance < 1e-9:
                continue
            direction = vector / distance
            start = center_xy + direction * (radius * 1.05)
            _draw_leader_line(ax, start, camera, color)
            _draw_legacy_offset_line(ax, count, vector, start, camera, color)

        ax.scatter(
            camera_xy[:, 0],
            camera_xy[:, 1],
            marker=str(style["marker"]),
            s=160,
            c=color,
            edgecolors="white",
            linewidths=1.8,
            zorder=12,
            label=str(style["label"]),
        )

    legend = ax.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        facecolor="white",
        framealpha=0.92,
        borderpad=0.6,
        handletextpad=0.5,
        labelspacing=0.4,
    )
    legend.set_zorder(20)

    pad = orbit * 0.25
    ax.set_xlim(center_xy[0] - orbit - pad, center_xy[0] + orbit + pad)
    ax.set_ylim(center_xy[1] - orbit - pad, center_xy[1] + orbit + pad)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


def _draw_leader_line(ax, start, camera, color: str) -> None:
    ax.plot(
        [start[0], camera[0]],
        [start[1], camera[1]],
        color=color,
        linewidth=1.0,
        linestyle=(0, (4, 5)),
        alpha=0.55,
        zorder=4,
    )


def _draw_legacy_offset_line(ax, count: int, vector, start, camera, color: str) -> None:
    """Preserve the legacy overlap-avoidance guide lines from dfr_plot."""
    if not np.isclose(vector[1], 0.0):
        return
    offset = {2: 20.0, 3: -20.0}.get(count)
    if offset is None:
        return
    ax.plot(
        [start[0], camera[0]],
        [start[1] + offset, camera[1] + offset],
        color=color,
        linewidth=1.0,
        linestyle=(0, (4, 5)),
        alpha=0.55,
        zorder=4,
    )


def _points(name: str, values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] not in {2, 3}:
        raise ValueError(f"{name} must have shape (points, 2) or (points, 3).")
    if len(array) == 0:
        raise ValueError(f"{name} must not be empty.")
    return array


def _center(points: np.ndarray, center) -> np.ndarray:
    if center is None:
        return (points.min(axis=0) + points.max(axis=0)) / 2.0
    array = np.asarray(center, dtype=np.float32).reshape(-1)
    if array.shape != (points.shape[1],):
        raise ValueError("center must have one coordinate per position dimension.")
    return array


def _swarm_radius(points: np.ndarray, center: np.ndarray, explicit) -> float:
    if explicit is not None:
        value = float(explicit)
    else:
        value = float(np.max(np.linalg.norm(points - center, axis=1)))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("swarm_radius must be positive and finite.")
    return value


def _orbit_radius(cameras: Mapping[int, np.ndarray], center: np.ndarray, explicit) -> float:
    if explicit is not None:
        value = float(explicit)
    else:
        center_xy = center[:2]
        value = max(
            float(np.max(np.linalg.norm(values[:, :2] - center_xy, axis=1)))
            for values in cameras.values()
        )
    if not np.isfinite(value) or value <= 0:
        raise ValueError("orbit_radius must be positive and finite.")
    return value


def _styles(counts, styles) -> dict[int, Mapping[str, object]]:
    selected = {}
    overrides = dict(styles or {})
    for count in counts:
        base = DEFAULT_CAMERA_STYLES.get(
            count,
            {"color": "#4B5563", "marker": "o", "label": rf"$K = {count}$"},
        )
        selected[count] = {**base, **dict(overrides.get(count, {}))}
    return selected
