"""Shared frame-selection and validation helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional, Union

import numpy as np

from dfr.data.base import Dataset


FrameSelector = Optional[Union[int, slice, Iterable[int]]]


def select_frame_indices(
    dataset: Dataset, frames: FrameSelector = None
) -> tuple[int, ...]:
    """Normalize a frame selector to validated non-negative indices.

    ``frames`` may be None (all frames), one integer, a slice, or an iterable
    of integers. Order and repeated indices are preserved for explicit
    iterables. Empty selections are rejected.
    """
    if frames is None:
        selected = tuple(range(len(dataset)))
    elif isinstance(frames, (int, np.integer)):
        selected = (dataset.normalize_time_step(int(frames)),)
    elif isinstance(frames, slice):
        if frames.step == 0:
            raise ValueError("Frame slice step cannot be zero.")
        selected = tuple(range(*frames.indices(len(dataset))))
    else:
        if isinstance(frames, (str, bytes)):
            raise TypeError("Frame selection must not be a string.")
        try:
            selected = tuple(dataset.normalize_time_step(index) for index in frames)
        except TypeError as error:
            raise TypeError(
                "Frame selection must be an integer, slice, or iterable of integers."
            ) from error

    if not selected:
        raise ValueError("Frame selection is empty.")
    return selected
