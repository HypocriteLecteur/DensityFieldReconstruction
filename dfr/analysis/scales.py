"""Scale-grid helpers shared by analysis and plotting workflows."""

from __future__ import annotations

import numpy as np


def validate_nnd_bounds(nnd_bounds) -> tuple[float, float]:
    """Return a validated ``(lower, upper)`` NND-normalised scale interval."""
    bounds = np.asarray(nnd_bounds, dtype=float)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
        raise ValueError("nnd_bounds must contain two finite numbers.")
    lower, upper = map(float, bounds)
    if lower <= 0 or upper <= lower:
        raise ValueError("nnd_bounds must satisfy 0 < lower < upper.")
    return lower, upper


def select_adaptive_density_scales(
    normalized_scales,
    mode_counts,
    *,
    n_selected: int = 4,
    relative_positions=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select representative scales from an empirical mode-count transition.

    The first and last sweep samples are reserved as visual/domain boundaries,
    so all returned indices lie strictly inside the scale range. When
    ``relative_positions`` is provided, those positions in the open interval
    ``(0, 1)`` override adaptive placement along the logarithmic scale range.
    """
    normalized_scales = np.asarray(normalized_scales, dtype=float)
    mode_counts = np.asarray(mode_counts, dtype=float)
    if normalized_scales.ndim != 1 or mode_counts.shape != normalized_scales.shape:
        raise ValueError(
            "normalized_scales and mode_counts must be equal-length 1D arrays."
        )
    if n_selected < 1 or len(normalized_scales) < n_selected + 2:
        raise ValueError(
            "The scale sweep must contain n_selected plus two boundary samples."
        )
    if np.any(normalized_scales <= 0) or np.any(mode_counts < 1):
        raise ValueError("Scales must be positive and mode counts must be at least one.")

    monotone_counts = np.minimum.accumulate(mode_counts)
    selected: set[int] = set()
    available = set(range(1, len(normalized_scales) - 1))
    log_scales = np.log(normalized_scales)

    if relative_positions is not None:
        relative_positions = np.asarray(relative_positions, dtype=float)
        if relative_positions.shape != (n_selected,):
            raise ValueError(
                "slice_relative_positions must contain exactly n_slices values."
            )
        if (
            not np.all(np.isfinite(relative_positions))
            or np.any(relative_positions <= 0)
            or np.any(relative_positions >= 1)
            or np.any(np.diff(relative_positions) <= 0)
        ):
            raise ValueError(
                "slice_relative_positions must be finite, strictly increasing, "
                "and strictly between 0 and 1."
            )
        targets = log_scales[0] + relative_positions * (
            log_scales[-1] - log_scales[0]
        )
        for target in targets:
            idx = min(available, key=lambda i: abs(log_scales[i] - target))
            selected.add(idx)
            available.remove(idx)

    elif monotone_counts[0] > monotone_counts[-1]:
        targets = np.geomspace(
            monotone_counts[0],
            monotone_counts[-1],
            n_selected + 2,
        )[1:-1]
        for target in targets:
            if len(selected) == n_selected:
                break
            idx = min(
                available,
                key=lambda i: abs(np.log(monotone_counts[i]) - np.log(target)),
            )
            selected.add(idx)
            available.remove(idx)

    fallback_targets = np.linspace(
        log_scales[0],
        log_scales[-1],
        n_selected + 2,
    )[1:-1]
    for target in fallback_targets:
        if len(selected) == n_selected:
            break
        idx = min(available, key=lambda i: abs(log_scales[i] - target))
        selected.add(idx)
        available.remove(idx)

    indices = np.asarray(sorted(selected), dtype=int)
    return indices, normalized_scales[indices]
