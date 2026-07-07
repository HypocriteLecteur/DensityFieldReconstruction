"""Reusable plotting primitives for DFR workflows.

Package plotting functions return Matplotlib figure/axes objects and avoid
filesystem writes unless a caller explicitly saves the figure.
"""

from dfr.plotting.cameras import plot_camera_configurations
from dfr.plotting.projections import (
    plot_density_image,
    plot_projected_gmm_density,
    plot_projection_points,
    transparent_colormap,
)
from dfr.plotting.style import apply_academic_style
from dfr.plotting.trajectories import plot_trajectory_snapshot

__all__ = [
    "apply_academic_style",
    "plot_camera_configurations",
    "plot_density_image",
    "plot_projected_gmm_density",
    "plot_projection_points",
    "plot_trajectory_snapshot",
    "transparent_colormap",
]
