"""Evaluation-result plotting primitives."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from dfr.evaluation.results import EvaluationRun, EvaluationSummary, FrameEvaluation
from dfr.plotting.style import apply_academic_style, apply_figure_layout


DEFAULT_EVALUATION_METRICS = ("recall", "miss", "hallucination", "dmota")


def plot_evaluation_summary(
    evaluation: EvaluationRun | FrameEvaluation | EvaluationSummary,
    *,
    metrics: Sequence[str] = DEFAULT_EVALUATION_METRICS,
    ax=None,
    title: Optional[str] = None,
    ylim: tuple[float, float] | None = None,
    bar_color: str = "#4169e1",
):
    """Plot a typed evaluation result as a compact metric bar chart.

    The input may be an aggregate :class:`EvaluationRun`, one
    :class:`FrameEvaluation`, or a bare :class:`EvaluationSummary`. The function
    returns ``(Figure, Axes)`` and leaves saving to the caller.
    """
    summary, default_title = _evaluation_summary_and_title(evaluation)
    metric_names = _metric_names(metrics)
    values = np.asarray([getattr(summary, name) for name in metric_names], dtype=float)

    apply_academic_style(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    else:
        fig = ax.figure

    labels = [_metric_label(name) for name in metric_names]
    bars = ax.bar(labels, values, color=bar_color, alpha=0.85)
    ax.set_ylabel("Metric value")
    ax.set_title(default_title if title is None else title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(0.0, color="black", linewidth=0.8)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )
    apply_figure_layout(fig)
    return fig, ax


def _evaluation_summary_and_title(
    evaluation: EvaluationRun | FrameEvaluation | EvaluationSummary,
) -> tuple[EvaluationSummary, str]:
    if isinstance(evaluation, EvaluationRun):
        return (
            evaluation.summary,
            f"Evaluation summary ({len(evaluation.frames)} frame(s))",
        )
    if isinstance(evaluation, FrameEvaluation):
        return (
            evaluation.summary,
            f"{evaluation.dataset_name} frame {evaluation.frame} evaluation",
        )
    if isinstance(evaluation, EvaluationSummary):
        return evaluation, "Evaluation summary"
    raise TypeError(
        "evaluation must be an EvaluationRun, FrameEvaluation, or EvaluationSummary."
    )


def _metric_names(metrics: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(str(name).lower() for name in metrics)
    if not selected:
        raise ValueError("metrics must contain at least one metric name.")
    allowed = set(DEFAULT_EVALUATION_METRICS)
    unknown = sorted(set(selected) - allowed)
    if unknown:
        raise ValueError(f"Unsupported evaluation metrics: {unknown}.")
    return selected


def _metric_label(name: str) -> str:
    if name == "dmota":
        return "dMOTA"
    return name.capitalize()
