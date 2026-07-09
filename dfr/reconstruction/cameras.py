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
    """Build a homogeneous multi-camera system for reconstruction.

    Parameters
    ----------
    dataset, frames:
        Dataset and selected integer frame IDs used to infer an encircling
        camera orbit when ``config.layout == "encircling"``.
    simulation:
        Scenario camera calibration and clipping settings loaded from YAML.
    config:
        Camera count/layout/device settings. Explicit layouts provide poses in
        ``(x, y, z, qx, qy, qz, qw)`` world-coordinate order.

    Returns
    -------
    dfr.camera_system.MultiCameraSystem
        Homogeneous cameras sharing the scenario intrinsics, image size, clip
        planes, and requested device. For the historical two-camera encircling
        case, the adjacent pair from a four-camera ring is preserved.
    """
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


def add_bounded_projection_noise(
    projections: list[np.ndarray],
    camera_system: MultiCameraSystem,
    standard_deviation: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Add bounded Gaussian pixel noise to image-plane projections.

    ``projections`` is one ``(visible_agents, 2)`` pixel-coordinate array per
    camera. Samples that would move outside ``[0, W] x [0, H]`` are resampled
    until they land inside the corresponding camera image. A zero standard
    deviation returns defensive copies without changing coordinates.
    """
    if standard_deviation < 0:
        raise ValueError("standard_deviation must be non-negative.")
    if standard_deviation == 0:
        return [np.asarray(projection).copy() for projection in projections]
    noisy = []
    for projection, camera in zip(projections, camera_system.cameras):
        source = np.asarray(projection)
        result = source.copy()
        pending = np.ones(len(source), dtype=bool)
        attempts = 0
        while np.any(pending):
            attempts += 1
            if attempts > 10_000:
                raise RuntimeError("Could not sample bounded projection noise.")
            candidate = source[pending] + rng.normal(
                0.0, standard_deviation, size=(int(np.sum(pending)), 2)
            )
            inside = (
                (candidate[:, 0] >= 0)
                & (candidate[:, 0] <= camera.state.W)
                & (candidate[:, 1] >= 0)
                & (candidate[:, 1] <= camera.state.H)
            )
            accepted = np.flatnonzero(pending)[inside]
            result[accepted] = candidate[inside]
            pending[accepted] = False
        noisy.append(result)
    return noisy
