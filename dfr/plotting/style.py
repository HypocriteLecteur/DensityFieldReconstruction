"""Shared Matplotlib styling helpers for DFR figures."""

from __future__ import annotations

from typing import Mapping, Any

import matplotlib.pyplot as plt


ACADEMIC_STYLE = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times", "Times New Roman"],
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


def apply_academic_style(overrides: Mapping[str, Any] | None = None) -> None:
    """Apply the publication-style defaults used by DFR figures."""
    style = dict(ACADEMIC_STYLE)
    if overrides is not None:
        style.update(overrides)
    plt.rcParams.update(style)
