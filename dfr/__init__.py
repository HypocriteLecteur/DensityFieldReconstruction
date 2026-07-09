"""Public API for Density Field Reconstruction.

The top-level :mod:`dfr` namespace is intentionally small and workflow
oriented.  It exposes the common research path:

``load_dataset`` -> ``analyze`` -> ``reconstruct`` -> ``evaluate``

plus the typed configuration/result objects needed to make those calls
reproducible.  Advanced numerical routines remain available from subpackages
such as :mod:`dfr.analysis`, :mod:`dfr.reconstruction`, :mod:`dfr.evaluation`,
and :mod:`dfr.plotting`.

Coordinate and data conventions
-------------------------------
Datasets expose positions as NumPy arrays shaped ``(agents, 3)`` for a single
frame, in the dataset's world-coordinate units.  Frame IDs are integer
trajectory indices.  Density scales, reconstruction radii, and evaluation
voxel sizes use the same world-coordinate units unless a function explicitly
states that scales are normalized by nearest-neighbour distance.

Side effects
------------
Package-level workflow functions return typed result objects and do not write artifacts
unless an explicit :class:`OutputConfig` or output path is supplied.
CUDA-backed reconstruction and some DRA analyses require a CUDA-capable PyTorch
runtime; CPU-safe analysis, loading, plotting, configuration, and artifact APIs
remain importable without CUDA extensions.

Example
-------
>>> import dfr
>>> dataset = dfr.load_dataset("jackdaw2")
>>> analysis = dfr.analyze(
...     dataset,
...     frames=[2800],
...     scales=(0.5, 1.0, 1.5),
...     kind="modes",
... )
"""

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
from dfr.reconstruction.observations import (
    ExternalObservationFrame,
    reconstruct_observations,
)
from dfr.reconstruction.scenarios import ScenarioRunSpec, run_scenario, run_scenarios
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
    "ExternalObservationFrame",
    "FrameEvaluation",
    "OutputConfig",
    "FrameReconstruction",
    "ReconstructionRequest",
    "ReconstructionRun",
    "RunArtifacts",
    "RunConfig",
    "ScenarioRegistry",
    "ScenarioRunSpec",
    "analyze",
    "evaluate",
    "load_dataset",
    "resolve_dataset",
    "reconstruct",
    "reconstruct_observations",
    "run_scenario",
    "run_scenarios",
    "select_frame_indices",
]
