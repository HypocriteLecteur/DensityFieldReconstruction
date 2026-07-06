"""Stable structural interface for loaded DFR datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Dataset(Protocol):
    """Protocol shared by analysis and reconstruction workflows.

    Required position arrays have shape ``(frames, agents, 3)``. Optional
    velocity arrays use the same shape. Loaders may NaN-pad absent agents;
    ``positions_at_time_step`` returns only valid rows.
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
