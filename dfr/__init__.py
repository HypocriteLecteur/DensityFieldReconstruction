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

__version__ = "0.1.0"

__all__ = [
    "AnalysisConfig",
    "CameraConfig",
    "Dataset",
    "DatasetSpec",
    "EvaluationConfig",
    "OutputConfig",
    "RunArtifacts",
    "RunConfig",
    "ScenarioRegistry",
    "analyze",
    "load_dataset",
    "resolve_dataset",
    "select_frame_indices",
]
