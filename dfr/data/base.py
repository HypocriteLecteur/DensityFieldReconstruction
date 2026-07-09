"""Stable structural interface for loaded DFR datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Dataset(Protocol):
    """Minimum in-memory dataset contract used by DFR workflows.

    Implementations expose trajectories as world-coordinate arrays shaped
    ``(frames, agents, 3)``. Optional velocity arrays use the same shape and
    optional timestamps use shape ``(frames,)``. Loaders may NaN-pad absent or
    invalid agents in the raw trajectory arrays; callers that need a clean
    frame should use :meth:`positions_at_time_step`, which returns only valid
    rows shaped ``(valid_agents, 3)``.

    ``time_step`` arguments are integer frame indices. Negative indices follow
    Python sequence semantics through :meth:`normalize_time_step`; out-of-range
    indices should raise :class:`IndexError`. Metadata is intentionally a loose
    mapping so loaders can preserve source-specific provenance, but common
    keys populated by :func:`dfr.load_dataset` include ``dataset_name``,
    ``scenario_config``, and ``project_root``.
    """

    @property
    def trajectories(self) -> np.ndarray: ...

    @property
    def velocities(self) -> np.ndarray: ...

    @property
    def has_velocities(self) -> bool: ...

    @property
    def timestamps(self) -> Optional[np.ndarray]: ...

    @property
    def has_timestamps(self) -> bool: ...

    @property
    def source_path(self) -> Optional[Path]: ...

    @property
    def metadata(self) -> Mapping[str, Any]: ...

    @property
    def coordinate_system(self) -> Optional[str]: ...

    @property
    def ground_truth_positions(self) -> np.ndarray: ...

    @property
    def frame_count(self) -> int: ...

    def __len__(self) -> int: ...

    def normalize_time_step(self, time_step: int) -> int: ...

    def positions_at_time_step(self, time_step: int) -> np.ndarray: ...

    def positions_at_time_step_mask(
        self, time_step: int
    ) -> tuple[np.ndarray, np.ndarray]: ...
