"""Density-overlap metrics, typed results, and evaluation workflow.

Evaluation compares reconstructed Gaussian mixtures against ground-truth
particle positions or density fields using voxelized density-overlap masses.
The high-level :func:`evaluate` function returns an :class:`EvaluationRun`
containing per-frame :class:`FrameEvaluation` records and an
:class:`EvaluationSummary`.

Voxel resolution and bounds are explicit through :class:`dfr.EvaluationConfig`.
All distances and voxel sizes use dataset world-coordinate units.  CUDA is
used when the selected device is CUDA; CPU evaluation is available for small
tests and smoke fixtures.
"""

from dfr.evaluation.metrics import (
    automatic_evaluation_bounds,
    compute_density_overlap_masses,
    evaluate_isotropic_gmm,
)
from dfr.evaluation.density import (
    build_isotropic_density_grid,
    sample_isotropic_density_grid,
)
from dfr.evaluation.pipeline import evaluate
from dfr.evaluation.results import EvaluationRun, EvaluationSummary, FrameEvaluation

__all__ = [
    "EvaluationRun",
    "EvaluationSummary",
    "FrameEvaluation",
    "automatic_evaluation_bounds",
    "build_isotropic_density_grid",
    "compute_density_overlap_masses",
    "evaluate",
    "evaluate_isotropic_gmm",
    "sample_isotropic_density_grid",
]
