"""Publication figure for training-time scaling across iteration budgets.

This script is a Phase 6 split-out from ``experiments.dfr_plot``. It keeps the
paper-specific hard-coded table data in a small explicit command instead of the
legacy mixed plotting archive.

Examples
--------
    python -m experiments.plot_publication_time_efficiency --output-dir outputs/publication-figures
    python -m experiments.plot_publication_time_efficiency --no-save --show
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from dfr.plotting import apply_academic_style, apply_figure_layout, save_figure


DATASETS = ("Swift", "Starling", "Jackdaw", "Jackdaw 2")
METHODS = ("Ours-2", "Ours-3", "Ours-5")
ITERATIONS = (100, 200, 500)
TRAINING_TIMES_MS = {
    "Swift": {
        "Ours-2": {100: 116, 200: 213, 500: 559},
        "Ours-3": {100: 123, 200: 225, 500: 554},
        "Ours-5": {100: 119, 200: 234, 500: 551},
    },
    "Starling": {
        "Ours-2": {100: 109, 200: 203, 500: 513},
        "Ours-3": {100: 105, 200: 289, 500: 535},
        "Ours-5": {100: 122, 200: 214, 500: 722},
    },
    "Jackdaw": {
        "Ours-2": {100: 118, 200: 218, 500: 547},
        "Ours-3": {100: 112, 200: 234, 500: 574},
        "Ours-5": {100: 126, 200: 256, 500: 570},
    },
    "Jackdaw 2": {
        "Ours-2": {100: 128, 200: 229, 500: 556},
        "Ours-3": {100: 128, 200: 239, 500: 544},
        "Ours-5": {100: 126, 200: 230, 500: 564},
    },
}
METHOD_COLORS = {
    "Ours-2": "#D55E00",
    "Ours-3": "#0072B2",
    "Ours-5": "#009E73",
}
DATASET_MARKERS = {
    "Swift": "o",
    "Starling": "s",
    "Jackdaw": "D",
    "Jackdaw 2": "^",
}


def plot_time_efficiency(ax=None):
    """Plot training time versus optimizer iterations and return ``(fig, ax)``."""
    apply_academic_style(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
        }
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6.5))
    else:
        fig = ax.figure

    for dataset in DATASETS:
        for method in METHODS:
            times = [TRAINING_TIMES_MS[dataset][method][iteration] for iteration in ITERATIONS]
            ax.scatter(
                ITERATIONS,
                times,
                c=METHOD_COLORS[method],
                marker=DATASET_MARKERS[dataset],
                s=100,
                edgecolors="white",
                linewidths=0.7,
                alpha=0.92,
                zorder=4,
            )

    all_times = [
        TRAINING_TIMES_MS[dataset][method][iteration]
        for dataset in DATASETS
        for method in METHODS
        for iteration in ITERATIONS
    ]
    time_low, time_high = min(all_times), max(all_times)
    time_pad = (time_high - time_low) * 0.12

    average_100_iter = np.mean(
        [TRAINING_TIMES_MS[dataset][method][100] for dataset in DATASETS for method in METHODS]
    )
    x_line = np.array([80, 520])
    y_line = (average_100_iter / 100.0) * x_line
    ax.plot(
        x_line,
        y_line,
        color="#333333",
        linestyle="--",
        linewidth=1.4,
        alpha=0.50,
        zorder=2,
        label=r"ideal linear  $T(k) \propto k$",
    )

    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=METHOD_COLORS[method],
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=method,
        )
        for method in METHODS
    ]
    dataset_handles = [
        Line2D(
            [0],
            [0],
            marker=DATASET_MARKERS[dataset],
            color="w",
            markerfacecolor="#555555",
            markersize=9,
            label=dataset,
        )
        for dataset in DATASETS
    ]
    method_legend = ax.legend(
        handles=method_handles,
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        fontsize=13,
        title="Method",
        title_fontsize=14,
    )
    ax.add_artist(method_legend)
    ax.legend(
        handles=dataset_handles,
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        fontsize=13,
        title="Dataset",
        title_fontsize=14,
    )

    ax.set_xlim(80, 520)
    ax.set_xticks(ITERATIONS)
    ax.set_ylim(time_low - time_pad, time_high + time_pad)
    ax.set_xlabel("Training Iterations")
    ax.set_ylabel("Training Time (msec)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linestyle="-", alpha=0.18)
    ax.grid(True, which="minor", linestyle=":", alpha=0.06)
    apply_figure_layout(fig, pad=3.5)
    return fig, ax


def save_time_efficiency_figure(
    output_dir: str | Path,
    *,
    formats: Iterable[str] = ("png", "pdf"),
) -> list[Path]:
    """Create and save the time-efficiency figure to ``output_dir``."""
    fig, _ = plot_time_efficiency()
    saved = []
    for fmt in formats:
        path = Path(output_dir) / f"table_dra_vs_iters.{fmt}"
        saved.append(save_figure(fig, path, dpi=300, bbox_inches="tight"))
    plt.close(fig)
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "publication-figures",
        help="Directory for generated figure files.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        help="Figure formats to save.",
    )
    parser.add_argument("--no-save", action="store_true", help="Render without saving.")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig, _ = plot_time_efficiency()
    if not args.no_save:
        for fmt in args.formats:
            path = args.output_dir / f"table_dra_vs_iters.{fmt}"
            save_figure(fig, path, dpi=300, bbox_inches="tight")
            print(f"Saved figure: {path}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
