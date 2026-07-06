"""Canonical dataset API for DFR."""

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
