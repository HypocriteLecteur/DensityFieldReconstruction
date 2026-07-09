"""Reusable plotting primitives for DFR workflows.

Plotting functions in this package accept data arrays or typed result objects
and return Matplotlib ``Figure``/``Axes`` objects.  They do not call
``plt.show`` and do not write files.  Use :func:`save_figure` or
``RunArtifacts.save_figure`` when a caller explicitly wants a figure on disk.

The plotting API is intentionally data-first: experiment scripts should load or
compute data, call these primitives, and handle output paths at the edge.  This
keeps reusable rendering independent from legacy ``figs/`` directories and
managed run artifacts.
"""

from dfr.plotting.cameras import plot_camera_configurations
from dfr.plotting.analysis import (
    plot_dra_scale_model_order_surface,
    plot_dra_surface_grid,
    plot_mode_count_curve,
)
from dfr.plotting.density import (
    DEFAULT_DENSITY_LAYERS,
    FIELD_DENSITY_LAYERS,
    plot_density_field_3d,
    plot_frame_reconstruction_gmm_3d,
    plot_multiscale_density_fields,
    render_agent_positions,
    render_density_shells,
    render_density_field_3d,
    render_frame_reconstruction_gmm_3d,
    render_gmm_means,
    render_gmm_wireframes,
    render_reconstructed_gmm_3d,
)
from dfr.plotting.evaluation import (
    plot_evaluation_metric_series,
    plot_evaluation_summary,
)
from dfr.plotting.projections import (
    plot_density_image,
    plot_projected_gmm_density,
    plot_projection_points,
    transparent_colormap,
)
from dfr.plotting.style import (
    apply_academic_style,
    apply_figure_layout,
    prepare_3d_axis,
    save_figure,
    set_3d_view,
    style_3d_axis,
)
from dfr.plotting.trajectories import plot_trajectory_snapshot

__all__ = [
    "apply_academic_style",
    "apply_figure_layout",
    "DEFAULT_DENSITY_LAYERS",
    "FIELD_DENSITY_LAYERS",
    "plot_camera_configurations",
    "plot_density_image",
    "plot_density_field_3d",
    "plot_frame_reconstruction_gmm_3d",
    "plot_dra_scale_model_order_surface",
    "plot_dra_surface_grid",
    "plot_evaluation_metric_series",
    "plot_evaluation_summary",
    "plot_mode_count_curve",
    "plot_multiscale_density_fields",
    "plot_projected_gmm_density",
    "plot_projection_points",
    "plot_trajectory_snapshot",
    "prepare_3d_axis",
    "render_agent_positions",
    "render_density_shells",
    "render_density_field_3d",
    "render_frame_reconstruction_gmm_3d",
    "render_gmm_means",
    "render_gmm_wireframes",
    "render_reconstructed_gmm_3d",
    "save_figure",
    "set_3d_view",
    "style_3d_axis",
    "transparent_colormap",
]
