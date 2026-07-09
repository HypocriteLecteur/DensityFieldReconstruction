"""High-level reconstruction requests, results, cameras, and pipeline.

The public reconstruction API centers on :func:`reconstruct`, which turns a
dataset, selected frames, camera configuration, reconstruction scale, and
training/reconstruction parameters into a :class:`ReconstructionRun`.

Camera poses are expressed as world-coordinate centers plus quaternions where
explicit poses are used.  The default encircling camera layout is generated
from the dataset geometry and scenario camera intrinsics.  GPU reconstruction
requires CUDA because the active density-field reconstructor and rasterizer
stack operate on CUDA tensors.

Workflow functions return typed results and write checkpoints, arrays, and
metadata only when an explicit :class:`dfr.OutputConfig` is provided.  Scenario
helpers remain available for reproducibility scripts, but new reusable logic
should live in package modules rather than experiment scripts.
"""

from dfr.reconstruction.cameras import add_bounded_projection_noise, build_camera_system
from dfr.reconstruction.pipeline import (
    default_reconstruction_params,
    default_training_params,
    reconstruct,
)
from dfr.reconstruction.results import (
    FrameReconstruction,
    ReconstructionRequest,
    ReconstructionRun,
)
from dfr.reconstruction.observations import (
    ExternalObservationFrame,
    reconstruct_observations,
)
from dfr.reconstruction.scenarios import (
    ScenarioRunSpec,
    run_scenario,
    run_scenarios,
)

__all__ = [
    "FrameReconstruction",
    "ExternalObservationFrame",
    "ReconstructionRequest",
    "ReconstructionRun",
    "ScenarioRunSpec",
    "build_camera_system",
    "add_bounded_projection_noise",
    "default_reconstruction_params",
    "default_training_params",
    "reconstruct",
    "reconstruct_observations",
    "run_scenario",
    "run_scenarios",
]
