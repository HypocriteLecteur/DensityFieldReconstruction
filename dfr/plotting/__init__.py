"""Reusable plotting primitives for DFR workflows.

Package plotting functions return Matplotlib figure/axes objects and avoid
filesystem writes unless a caller explicitly saves the figure.
"""

from dfr.plotting.cameras import plot_camera_configurations
from dfr.plotting.analysis import (
    plot_dra_scale_model_order_surface,
    plot_dra_surface_grid,
    plot_mode_count_curve,
)
from dfr.plotting.density import (
    plot_density_field_3d,
    plot_multiscale_density_fields,
    render_agent_positions,
    render_density_shells,
)
from dfr.plotting.projections import (
    plot_density_image,
    plot_projected_gmm_density,
    plot_projection_points,
    transparent_colormap,
)
from dfr.plotting.style import apply_academic_style, save_figure, style_3d_axis
from dfr.plotting.trajectories import plot_trajectory_snapshot

__all__ = [
    "apply_academic_style",
    "plot_camera_configurations",
    "plot_density_image",
    "plot_density_field_3d",
    "plot_dra_scale_model_order_surface",
    "plot_dra_surface_grid",
    "plot_mode_count_curve",
    "plot_multiscale_density_fields",
    "plot_projected_gmm_density",
    "plot_projection_points",
    "plot_trajectory_snapshot",
    "render_agent_positions",
    "render_density_shells",
    "save_figure",
    "style_3d_axis",
    "transparent_colormap",
]
