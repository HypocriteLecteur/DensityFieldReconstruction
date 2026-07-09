"""Reusable analysis APIs and typed result objects.

This package contains computation-only analysis helpers.  Functions here
return typed result objects such as :class:`ModeCurveResult` and
:class:`ScaleAnalysisResult`; they do not save figures or artifacts unless a
caller explicitly uses managed artifact helpers from :mod:`dfr.artifacts`.

Scale conventions are function-specific:

- mode-count curves usually consume world-space density scales;
- DRA scale/model-order sweeps consume scales normalized by the frame's mean
  nearest-neighbour distance unless the lower-level function documents
  otherwise;
- manifold helpers operate on cached or fitted mode-count parameter curves.

CUDA is required for DRA surface computations and some mode-count paths that
use the legacy mean-shift implementation.  Pure fitting, result loading, and
configuration utilities are CPU-safe.
"""

from dfr.analysis.dra import (
    DRAFrameSamples,
    FIT_MODELS,
    concatenate_frames,
    compute_dra_sweep,
    compute_scale_model_order_surface,
    create_scale_analysis,
    fit_design_matrix,
    fit_dra_surface,
    fit_frames,
    fit_one_surface_model,
    mean_nearest_neighbour_distance,
    model_orders,
    grouped_cv_rmse,
    leave_one_dataset_out_rmse,
    select_frames,
)
from dfr.analysis.results import (
    ManifoldAnalysisResult,
    ModeCurveResult,
    ScaleAnalysisResult,
)
from dfr.analysis.modes import analyze_dataset_modes, compute_mode_curve, count_modes
from dfr.analysis.scales import (
    select_adaptive_density_scales,
    validate_nnd_bounds,
)
from dfr.analysis.cli import add_managed_output_arguments, create_analysis_artifacts
from dfr.analysis.manifold import (
    Centered3PLFitBatch,
    LegacyManifoldCache,
    PARAMETER_NAMES,
    Symmetric2PLFitBatch,
    centered_3pl_excess,
    fit_centered_3pl_curves,
    fit_symmetric_2pl_curves,
    fit_shape_curve,
    load_legacy_manifold_cache,
    median_nearest_neighbour_distance,
    project_to_shape_curve,
    scale_for_mode_count,
    symmetric_2pl_mode_count,
)

__all__ = [
    "DRAFrameSamples",
    "Centered3PLFitBatch",
    "FIT_MODELS",
    "LegacyManifoldCache",
    "ManifoldAnalysisResult",
    "ModeCurveResult",
    "PARAMETER_NAMES",
    "Symmetric2PLFitBatch",
    "ScaleAnalysisResult",
    "analyze_dataset_modes",
    "add_managed_output_arguments",
    "centered_3pl_excess",
    "concatenate_frames",
    "compute_scale_model_order_surface",
    "compute_mode_curve",
    "compute_dra_sweep",
    "count_modes",
    "create_scale_analysis",
    "create_analysis_artifacts",
    "fit_design_matrix",
    "fit_centered_3pl_curves",
    "fit_symmetric_2pl_curves",
    "fit_dra_surface",
    "fit_frames",
    "fit_one_surface_model",
    "fit_shape_curve",
    "grouped_cv_rmse",
    "leave_one_dataset_out_rmse",
    "load_legacy_manifold_cache",
    "mean_nearest_neighbour_distance",
    "median_nearest_neighbour_distance",
    "model_orders",
    "project_to_shape_curve",
    "scale_for_mode_count",
    "symmetric_2pl_mode_count",
    "select_frames",
    "select_adaptive_density_scales",
    "validate_nnd_bounds",
]
