"""Canonical dataset API for DFR.

Use :func:`load_dataset` for both registered scenarios and explicit data/config
paths.  Returned datasets satisfy :class:`Dataset`: callers can query the
number of frames, fetch ``(agents, 3)`` world-coordinate positions with
``positions_at_time_step(frame)``, and inspect optional metadata such as
timestamps, velocities, and ground-truth fields when a loader provides them.

The registry resolves paths relative to an explicit project root rather than
the process working directory.  :func:`select_frame_indices` normalizes integer,
slice, and iterable frame selectors for analysis and reconstruction workflows.
"""

from dfr.data.base import Dataset
from dfr.data.frames import FrameSelector, select_frame_indices
from dfr.data.loading import load_dataset
from dfr.data.registry import DatasetSource, ScenarioRegistry, resolve_dataset
from dfr.data.spec import DatasetSpec

__all__ = [
    "Dataset",
    "DatasetSource",
    "DatasetSpec",
    "FrameSelector",
    "ScenarioRegistry",
    "load_dataset",
    "resolve_dataset",
    "select_frame_indices",
]
