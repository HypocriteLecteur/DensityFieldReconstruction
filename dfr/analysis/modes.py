"""Straightforward package API for mode counts and mode-count curves."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from dfr.analysis.results import ModeCurveResult
from dfr.data.base import Dataset
from dfr.mode_finding import mode_counting


def count_modes(
    positions: np.ndarray | torch.Tensor,
    scale: float,
    *,
    initial_modes: Optional[np.ndarray | torch.Tensor] = None,
    device: str | torch.device | None = None,
    max_iter: int = 1000,
    tolerance: float = 1e-2,
) -> int:
    """Count density modes for one point frame and world-space scale."""
    if scale <= 0:
        raise ValueError("scale must be positive.")
    if max_iter < 1 or tolerance <= 0:
        raise ValueError("max_iter and tolerance must be positive.")
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    points = torch.as_tensor(positions, dtype=torch.float32, device=selected_device)
    if points.ndim != 2 or points.shape[1] not in (2, 3) or len(points) == 0:
        raise ValueError("positions must be a non-empty (agents, 2|3) array.")
    modes = (
        points.clone()
        if initial_modes is None
        else torch.as_tensor(
            initial_modes, dtype=torch.float32, device=selected_device
        )
    )
    if modes.ndim != 2 or modes.shape[1] != points.shape[1] or len(modes) == 0:
        raise ValueError("initial_modes must be a non-empty array with matching dimension.")
    return int(
        mode_counting(
            points,
            modes,
            float(scale),
            max_iter=max_iter,
            tol=tolerance,
        )
    )


def compute_mode_curve(
    positions: np.ndarray | torch.Tensor,
    scales,
    *,
    frame: Optional[int] = None,
    dataset_name: Optional[str] = None,
    device: str | torch.device | None = None,
    max_iter: int = 1000,
    tolerance: float = 1e-2,
) -> ModeCurveResult:
    """Count modes independently at every scale and return typed curve data."""
    scale_values = np.asarray(scales, dtype=np.float64)
    # Validate ordering before starting potentially expensive computation.
    if (
        scale_values.ndim != 1
        or len(scale_values) == 0
        or np.any(~np.isfinite(scale_values))
        or np.any(scale_values <= 0)
        or np.any(np.diff(scale_values) <= 0)
    ):
        raise ValueError("scales must be a strictly increasing positive 1D array.")
    counts = [
        count_modes(
            positions,
            float(scale),
            device=device,
            max_iter=max_iter,
            tolerance=tolerance,
        )
        for scale in scale_values
    ]
    return ModeCurveResult(
        scales=scale_values,
        mode_counts=counts,
        frame=frame,
        dataset_name=dataset_name,
    )


def analyze_dataset_modes(
    dataset: Dataset,
    frame: int,
    scales,
    **kwargs,
) -> ModeCurveResult:
    """Load one dataset frame and compute its mode-count curve."""
    normalized_frame = dataset.normalize_time_step(frame)
    name = dataset.metadata.get("dataset_name")
    return compute_mode_curve(
        dataset.positions_at_time_step(normalized_frame),
        scales,
        frame=normalized_frame,
        dataset_name=name,
        **kwargs,
    )
