"""Publication figures for Table 2 reconstruction metrics.

This script is a Phase 6 split-out from the retired legacy plot archive. It keeps the
paper-specific hard-coded Table 2 metrics in a small explicit command instead
of the legacy mixed plotting archive.

Examples
--------
    python -m experiments.plot_publication_table2 --output-dir outputs/publication-figures
    python -m experiments.plot_publication_table2 --figure tradeoff --no-save --show
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from dfr.plotting import apply_academic_style, apply_figure_layout, save_figure


FigureName = Literal["capacity", "tradeoff", "all"]

DATASETS = ("Swift", "Starling", "Jackdaw", "Jackdaw 2")
OURS_METHODS = ("Ours-2", "Ours-3", "Ours-5")
ALL_METHODS = ("GMR-2",) + OURS_METHODS
CAMERA_COUNTS = (2, 3, 5)

# Structure: TABLE2_METRICS[dataset][method] = (recall, hallucination, DRA)
TABLE2_METRICS = {
    "Swift": {
        "GMR-2": (0.824, 0.038, 0.792),
        "Ours-2": (0.749, 0.245, 0.504),
        "Ours-3": (0.831, 0.168, 0.663),
        "Ours-5": (0.889, 0.107, 0.782),
    },
    "Starling": {
        "GMR-2": (0.904, 0.096, 0.808),
        "Ours-2": (0.644, 0.355, 0.289),
        "Ours-3": (0.889, 0.120, 0.768),
        "Ours-5": (0.901, 0.113, 0.786),
    },
    "Jackdaw": {
        "GMR-2": (0.903, 0.059, 0.847),
        "Ours-2": (0.700, 0.296, 0.405),
        "Ours-3": (0.821, 0.177, 0.645),
        "Ours-5": (0.873, 0.119, 0.755),
    },
    "Jackdaw 2": {
        "GMR-2": (0.938, 0.054, 0.884),
        "Ours-2": (0.801, 0.188, 0.614),
        "Ours-3": (0.871, 0.120, 0.752),
        "Ours-5": (0.906, 0.085, 0.822),
    },
}

METHOD_COLORS = {
    "GMR-2": "#333333",
    "Ours-2": "#D55E00",
    "Ours-3": "#0072B2",
    "Ours-5": "#009E73",
}
METHOD_MARKERS = {
    "GMR-2": "*",
    "Ours-2": "s",
    "Ours-3": "D",
    "Ours-5": "o",
}
DATASET_COLORS = {
    "Swift": "#1f77b4",
    "Starling": "#d62728",
    "Jackdaw": "#2ca02c",
    "Jackdaw 2": "#9467bd",
}
DATASET_MARKERS = {
    "Swift": "o",
    "Starling": "s",
    "Jackdaw": "D",
    "Jackdaw 2": "^",
}


def _apply_table2_style() -> None:
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


def plot_capacity_scaling(ax=None):
    """Plot DRA/DEA capacity scaling versus number of cameras.

    Returns
    -------
    tuple
        ``(fig, ax)`` for the rendered capacity-scaling figure.
    """
    _apply_table2_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6.5))
    else:
        fig = ax.figure

    for dataset in DATASETS:
        gmr_dra = TABLE2_METRICS[dataset]["GMR-2"][2]
        ax.axhline(
            y=gmr_dra,
            color=DATASET_COLORS[dataset],
            linestyle="--",
            linewidth=1.2,
            alpha=0.55,
            zorder=2,
        )

    for dataset in DATASETS:
        dra_values = [TABLE2_METRICS[dataset][method][2] for method in OURS_METHODS]
        ax.plot(
            CAMERA_COUNTS,
            dra_values,
            color=DATASET_COLORS[dataset],
            marker=DATASET_MARKERS[dataset],
            markersize=10,
            linewidth=2.2,
            markeredgewidth=0.8,
            markeredgecolor="white",
            label=dataset,
            zorder=5,
        )

    gmr_handle = Line2D(
        [0],
        [0],
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label="GMR-2 (baseline)",
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0, gmr_handle)
    labels.insert(0, "GMR-2 (baseline)")
    ax.legend(
        handles,
        labels,
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        fontsize=13,
    )

    ax.set_xlabel("Number of Cameras")
    ax.set_ylabel("DEA")
    ax.set_xticks(CAMERA_COUNTS)
    ax.set_xlim(1.5, 5.5)
    ax.set_ylim(0.15, 0.98)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", axis="y", linestyle="-", alpha=0.2)
    ax.grid(True, which="minor", axis="y", linestyle=":", alpha=0.08)
    apply_figure_layout(fig, pad=2.5)
    return fig, ax


def plot_recall_hallucination_tradeoff(ax=None):
    """Plot recall versus hallucination with iso-DRA reference curves."""
    _apply_table2_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6.5))
    else:
        fig = ax.figure

    hallucination_grid = np.linspace(0.001, 0.42, 200)
    iso_dra_levels = (0.3, 0.5, 0.7, 0.85)
    for dra_level in iso_dra_levels:
        recall_curve = dra_level * (1.0 - hallucination_grid) / (
            1.0 - 2.0 * hallucination_grid
        )
        visible = (recall_curve > 0.4) & (recall_curve < 1.05)
        if visible.any():
            ax.plot(
                hallucination_grid[visible],
                recall_curve[visible],
                color="#B0B0B0",
                linewidth=0.7,
                linestyle=":",
                alpha=0.55,
                zorder=1,
            )
            label_index = np.where(visible)[0][-1]
            ax.annotate(
                f"DRA={dra_level}",
                (hallucination_grid[label_index], recall_curve[label_index]),
                textcoords="offset points",
                xytext=(4, -2),
                fontsize=11,
                color="#888888",
                va="top",
                alpha=0.7,
            )

    for dataset in DATASETS:
        for method in ALL_METHODS:
            recall, hallucination, _ = TABLE2_METRICS[dataset][method]
            ax.scatter(
                hallucination,
                recall,
                c=METHOD_COLORS[method],
                marker=DATASET_MARKERS[dataset],
                s=140 if method == "GMR-2" else 110,
                edgecolors="white",
                linewidths=0.8 if method == "GMR-2" else 0.5,
                alpha=0.92,
                zorder=6 if method == "GMR-2" else 4,
            )

    ax.annotate(
        "better (lower hallucination)",
        xy=(0.02, 0.015),
        xycoords="axes fraction",
        fontsize=12,
        color="#888888",
        ha="left",
        va="bottom",
    )
    ax.annotate(
        "better (higher recall)",
        xy=(0.02, 0.975),
        xycoords="axes fraction",
        fontsize=12,
        color="#888888",
        ha="left",
        va="top",
    )

    method_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linewidth=2.5,
            marker=METHOD_MARKERS[method],
            markersize=8,
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=method,
        )
        for method in ALL_METHODS
    ]
    dataset_handles = [
        Line2D(
            [0],
            [0],
            marker=DATASET_MARKERS[dataset],
            color="w",
            markerfacecolor="#333333",
            markersize=9,
            label=dataset,
        )
        for dataset in DATASETS
    ]

    method_legend = ax.legend(
        handles=method_handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.10),
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
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        fontsize=13,
        title="Dataset",
        title_fontsize=14,
    )

    ax.set_xlabel("Hallucination")
    ax.set_ylabel("Recall")
    ax.set_xlim(-0.02, 0.48)
    ax.set_ylim(0.55, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linestyle="-", alpha=0.15)
    ax.grid(True, which="minor", linestyle=":", alpha=0.06)
    apply_figure_layout(fig, pad=2.5)
    return fig, ax


def plot_table2_figures() -> tuple[plt.Figure, plt.Figure]:
    """Render both Table 2 publication figures."""
    capacity_fig, _ = plot_capacity_scaling()
    tradeoff_fig, _ = plot_recall_hallucination_tradeoff()
    return capacity_fig, tradeoff_fig


def save_table2_figures(
    output_dir: str | Path,
    *,
    formats: Iterable[str] = ("png", "pdf"),
    figure: FigureName = "all",
) -> list[Path]:
    """Create and save selected Table 2 publication figures."""
    saved: list[Path] = []
    output_dir = Path(output_dir)

    if figure in ("capacity", "all"):
        capacity_fig, _ = plot_capacity_scaling()
        for fmt in formats:
            path = output_dir / f"table2_dea_capacity_scaling.{fmt}"
            saved.append(save_figure(capacity_fig, path, dpi=300, bbox_inches="tight"))
        plt.close(capacity_fig)

    if figure in ("tradeoff", "all"):
        tradeoff_fig, _ = plot_recall_hallucination_tradeoff()
        for fmt in formats:
            path = output_dir / f"table2_recall_hallu_tradeoff.{fmt}"
            saved.append(save_figure(tradeoff_fig, path, dpi=300, bbox_inches="tight"))
        plt.close(tradeoff_fig)

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
    parser.add_argument(
        "--figure",
        choices=["capacity", "tradeoff", "all"],
        default="all",
        help="Which Table 2 figure to render.",
    )
    parser.add_argument("--no-save", action="store_true", help="Render without saving.")
    parser.add_argument("--show", action="store_true", help="Display figures interactively.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures: list[plt.Figure] = []

    if args.figure in ("capacity", "all"):
        fig, _ = plot_capacity_scaling()
        figures.append(fig)
        if not args.no_save:
            for fmt in args.formats:
                path = args.output_dir / f"table2_dea_capacity_scaling.{fmt}"
                save_figure(fig, path, dpi=300, bbox_inches="tight")
                print(f"Saved figure: {path}")

    if args.figure in ("tradeoff", "all"):
        fig, _ = plot_recall_hallucination_tradeoff()
        figures.append(fig)
        if not args.no_save:
            for fmt in args.formats:
                path = args.output_dir / f"table2_recall_hallu_tradeoff.{fmt}"
                save_figure(fig, path, dpi=300, bbox_inches="tight")
                print(f"Saved figure: {path}")

    if args.show:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)


if __name__ == "__main__":
    main()
