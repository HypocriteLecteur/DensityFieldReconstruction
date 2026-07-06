"""Camera-system construction shared by reconstruction entry points."""

from __future__ import annotations

import numpy as np

from dfr.camera_state import CameraState
from dfr.camera_system import MultiCameraSystem
from dfr.config import CameraConfig
from dfr.data.base import Dataset
from dfr.simulation_config import SimulationConfig
from dfr.utils import generate_encircling_cameras


def build_camera_system(
    dataset: Dataset,
    frames: tuple[int, ...],
    simulation: SimulationConfig,
    config: CameraConfig,
) -> MultiCameraSystem:
    """Build explicit or generated homogeneous cameras for selected frames."""
    if config.layout == "explicit":
        poses = np.asarray(config.poses, dtype=np.float32)
    else:
        # For two cameras, preserve the established adjacent pair from a
        # four-camera ring instead of choosing a degenerate opposite pair.
        generated_count = 4 if config.count == 2 else config.count
        positions, _ = generate_encircling_cameras(
            dataset,
            frames,
            simulation.intrinsics_params,
            simulation.H,
            simulation.W,
            cam_num=generated_count,
            padding=config.padding,
            is_3d=config.is_3d,
        )
        orientations = np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            (config.count, 1),
        )
        poses = np.hstack((positions[: config.count], orientations)).astype(
            np.float32
        )
    return MultiCameraSystem.create_homogeneous_system(
        state_class=CameraState,
        intrinsics=simulation.intrinsics_params,
        H=simulation.H,
        W=simulation.W,
        poses_or_RTs=poses,
        near_clip=simulation.near_clip,
        far_clip=simulation.far_clip,
        size=simulation.size,
        device=config.device,
    )
