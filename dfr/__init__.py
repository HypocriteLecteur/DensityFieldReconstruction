"""Public API for Density Field Reconstruction."""

from dfr.data import (
    Dataset,
    DatasetSpec,
    ScenarioRegistry,
    load_dataset,
    resolve_dataset,
    select_frame_indices,
)
from dfr.artifacts import OutputConfig, RunArtifacts
from dfr.config import AnalysisConfig, CameraConfig, EvaluationConfig, RunConfig
from dfr.workflows import analyze
from dfr.reconstruction.pipeline import reconstruct
from dfr.reconstruction.results import (
    FrameReconstruction,
    ReconstructionRequest,
    ReconstructionRun,
)
from dfr.evaluation.pipeline import evaluate
from dfr.evaluation.results import EvaluationRun, EvaluationSummary, FrameEvaluation

__version__ = "0.1.0"

__all__ = [
    "AnalysisConfig",
    "CameraConfig",
    "Dataset",
    "DatasetSpec",
    "EvaluationConfig",
    "EvaluationRun",
    "EvaluationSummary",
    "FrameEvaluation",
    "OutputConfig",
    "FrameReconstruction",
    "ReconstructionRequest",
    "ReconstructionRun",
    "RunArtifacts",
    "RunConfig",
    "ScenarioRegistry",
    "analyze",
    "evaluate",
    "load_dataset",
    "resolve_dataset",
    "reconstruct",
    "select_frame_indices",
]
