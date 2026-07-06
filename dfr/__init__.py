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

__version__ = "0.1.0"

__all__ = [
    "Dataset",
    "DatasetSpec",
    "OutputConfig",
    "RunArtifacts",
    "ScenarioRegistry",
    "load_dataset",
    "resolve_dataset",
    "select_frame_indices",
]
