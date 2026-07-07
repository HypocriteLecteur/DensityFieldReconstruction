"""Shared Matplotlib styling, layout, and saving helpers for DFR figures."""

from __future__ import annotations

from pathlib import Path
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


def style_3d_axis(ax: plt.Axes) -> None:
    """Apply DFR's transparent-pane 3D axis styling in-place."""
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis._axinfo["grid"].update(
            {
                "color": (0.8, 0.8, 0.8, 0.5),
                "linewidth": 0.5,
            }
        )
        axis.pane.set_edgecolor("none")


def save_figure(
    figure,
    path: str | Path,
    *,
    dpi: int = 300,
    bbox_inches: str | None = "tight",
    pad_inches: float | None = None,
    transparent: bool | None = None,
    create_parent: bool = True,
    **savefig_kwargs: Any,
) -> Path:
    """Save a Matplotlib figure with DFR's default publication settings.

    Managed workflows should still prefer ``RunArtifacts.save_figure`` when
    they need overwrite protection and manifest-backed provenance. This helper
    is for reusable plotting primitives, compatibility wrappers, and
    experiment-local figure exports.
    """
    target = Path(path)
    if create_parent:
        target.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {"dpi": dpi}
    if bbox_inches is not None:
        kwargs["bbox_inches"] = bbox_inches
    if pad_inches is not None:
        kwargs["pad_inches"] = pad_inches
    if transparent is not None:
        kwargs["transparent"] = transparent
    kwargs.update(savefig_kwargs)
    figure.savefig(target, **kwargs)
    return target
