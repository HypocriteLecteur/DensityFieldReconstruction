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
    """Apply the publication-style defaults used by DFR figures.

    This mutates Matplotlib ``rcParams`` process-wide. Pass ``overrides`` for a
    small local adjustment before creating figures; callers that need isolated
    styling should use Matplotlib's own context managers around this helper.
    """
    style = dict(ACADEMIC_STYLE)
    if overrides is not None:
        style.update(overrides)
    plt.rcParams.update(style)


def style_3d_axis(ax: plt.Axes) -> None:
    """Apply DFR's transparent-pane 3D axis styling in-place.

    ``ax`` must be a Matplotlib 3D axes object. The function only changes axis
    panes/grid styling and returns ``None``.
    """
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis._axinfo["grid"].update(
            {
                "color": (0.8, 0.8, 0.8, 0.5),
                "linewidth": 0.5,
            }
        )
        axis.pane.set_edgecolor("none")


def set_3d_view(ax: plt.Axes, view: tuple[float, float] | tuple[float, float, float]) -> None:
    """Set a 3D Matplotlib view, tolerating older Matplotlib without ``roll``.

    ``view`` is ``(elev, azim)`` or ``(elev, azim, roll)`` in degrees. Older
    Matplotlib versions that do not support ``roll`` silently receive only
    elevation and azimuth.
    """
    if len(view) == 2:
        elev, azim = view
        roll = None
    elif len(view) == 3:
        elev, azim, roll = view
    else:
        raise ValueError("view must be (elev, azim) or (elev, azim, roll).")
    if roll is None:
        ax.view_init(elev=elev, azim=azim)
        return
    try:
        ax.view_init(elev=elev, azim=azim, roll=roll)
    except TypeError:
        ax.view_init(elev=elev, azim=azim)


def prepare_3d_axis(
    ax: plt.Axes,
    *,
    view: tuple[float, float] | tuple[float, float, float] | None = None,
    axis_off: bool = False,
) -> None:
    """Apply common 3D axis view and visibility settings.

    This is a lightweight convenience wrapper used by plotting primitives. It
    never creates figures, saves files, or changes global Matplotlib state.
    """
    if view is not None:
        set_3d_view(ax, view)
    if axis_off:
        ax.set_axis_off()


def apply_figure_layout(
    figure,
    *,
    pad: float | None = None,
    rect: tuple[float, float, float, float] | None = None,
    adjust: Mapping[str, Any] | None = None,
) -> None:
    """Apply a consistent tight-layout or subplot-adjust operation.

    When ``adjust`` is supplied, it is passed to ``figure.subplots_adjust``.
    Otherwise ``figure.tight_layout`` is called with optional ``pad`` and
    ``rect``. The helper mutates layout in-place and returns ``None``.
    """
    if adjust is not None:
        figure.subplots_adjust(**dict(adjust))
        return
    kwargs: dict[str, Any] = {}
    if pad is not None:
        kwargs["pad"] = pad
    if rect is not None:
        kwargs["rect"] = rect
    figure.tight_layout(**kwargs)


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
    experiment-local figure exports. Parent directories are created by default
    and the resolved target :class:`pathlib.Path` is returned.
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
