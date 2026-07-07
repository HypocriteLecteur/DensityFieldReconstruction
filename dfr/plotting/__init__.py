"""Reusable plotting primitives for DFR workflows.

Package plotting functions return Matplotlib figure/axes objects and avoid
filesystem writes unless a caller explicitly saves the figure.
"""

from dfr.plotting.cameras import plot_camera_configurations
from dfr.plotting.style import apply_academic_style

__all__ = [
    "apply_academic_style",
    "plot_camera_configurations",
]
