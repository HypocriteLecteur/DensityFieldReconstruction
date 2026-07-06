"""Reusable analysis APIs and typed result objects."""

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

__all__ = [
    "DRAFrameSamples",
    "FIT_MODELS",
    "ManifoldAnalysisResult",
    "ModeCurveResult",
    "ScaleAnalysisResult",
    "analyze_dataset_modes",
    "concatenate_frames",
    "compute_scale_model_order_surface",
    "compute_mode_curve",
    "compute_dra_sweep",
    "count_modes",
    "create_scale_analysis",
    "fit_design_matrix",
    "fit_dra_surface",
    "fit_frames",
    "fit_one_surface_model",
    "grouped_cv_rmse",
    "leave_one_dataset_out_rmse",
    "mean_nearest_neighbour_distance",
    "model_orders",
    "select_frames",
]
