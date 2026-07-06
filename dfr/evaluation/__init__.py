"""Density-overlap metrics, typed results, and evaluation workflow."""

from dfr.evaluation.metrics import (
    automatic_evaluation_bounds,
    compute_density_overlap_masses,
    evaluate_isotropic_gmm,
)
from dfr.evaluation.pipeline import evaluate
from dfr.evaluation.results import EvaluationRun, EvaluationSummary, FrameEvaluation

__all__ = [
    "EvaluationRun",
    "EvaluationSummary",
    "FrameEvaluation",
    "automatic_evaluation_bounds",
    "compute_density_overlap_masses",
    "evaluate",
    "evaluate_isotropic_gmm",
]
