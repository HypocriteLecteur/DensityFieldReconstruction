"""Publication figure for Table 4 noise robustness metrics.

This script is a Phase 6 split-out from ``experiments.dfr_plot``. It keeps the
paper-specific hard-coded noise-robustness metrics in a small explicit command
instead of the legacy mixed plotting archive.

Examples
--------
    python -m experiments.plot_publication_noise_robustness --output-dir outputs/publication-figures
    python -m experiments.plot_publication_noise_robustness --no-save --show
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from dfr.plotting import apply_academic_style, apply_figure_layout, save_figure


DATASETS = ("Swift", "Starling", "Jackdaw", "Jackdaw 2")
CAMERA_LABELS = ("2-cam", "3-cam", "5-cam")
NOISE_LEVELS = (5, 10, 20)

# Structure: NOISE_METRICS[dataset][camera_label][sigma_n] = (recall, DRA)
NOISE_METRICS = {
    "Swift": {
        "2-cam": {5: (0.744, 0.497), 10: (0.741, 0.491), 20: (0.747, 0.506)},
        "3-cam": {5: (0.833, 0.668), 10: (0.833, 0.669), 20: (0.826, 0.655)},
        "5-cam": {5: (0.888, 0.780), 10: (0.887, 0.777), 20: (0.874, 0.753)},
    },
    "Starling": {
        "2-cam": {5: (0.642, 0.282), 10: (0.648, 0.297), 20: (0.669, 0.222)},
        "3-cam": {5: (0.890, 0.770), 10: (0.875, 0.742), 20: (0.851, 0.703)},
        "5-cam": {5: (0.895, 0.779), 10: (0.896, 0.780), 20: (0.870, 0.734)},
    },
    "Jackdaw": {
        "2-cam": {5: (0.702, 0.408), 10: (0.693, 0.391), 20: (0.683, 0.377)},
        "3-cam": {5: (0.825, 0.652), 10: (0.814, 0.632), 20: (0.800, 0.603)},
        "5-cam": {5: (0.877, 0.763), 10: (0.875, 0.758), 20: (0.855, 0.711)},
    },
    "Jackdaw 2": {
        "2-cam": {5: (0.796, 0.609), 10: (0.796, 0.608), 20: (0.783, 0.585)},
        "3-cam": {5: (0.866, 0.744), 10: (0.862, 0.735), 20: (0.840, 0.692)},
        "5-cam": {5: (0.903, 0.815), 10: (0.888, 0.796), 20: (0.856, 0.737)},
    },
}

CAMERA_TO_METHOD = {"2-cam": "Ours-2", "3-cam": "Ours-3", "5-cam": "Ours-5"}
CLEAN_DRA = {
    "Swift": {"Ours-2": 0.504, "Ours-3": 0.663, "Ours-5": 0.782},
    "Starling": {"Ours-2": 0.289, "Ours-3": 0.768, "Ours-5": 0.786},
    "Jackdaw": {"Ours-2": 0.405, "Ours-3": 0.645, "Ours-5": 0.755},
    "Jackdaw 2": {"Ours-2": 0.614, "Ours-3": 0.752, "Ours-5": 0.822},
}

# Median nearest-neighbor distance per dataset, in pixels.
NND_PIXELS = {"Swift": 6.4, "Starling": 6.3, "Jackdaw": 8.1, "Jackdaw 2": 13.4}

CAMERA_COLORS = {
    "2-cam": "#D55E00",
    "3-cam": "#0072B2",
    "5-cam": "#009E73",
}
DATASET_MARKERS = {
    "Swift": "o",
    "Starling": "s",
    "Jackdaw": "D",
    "Jackdaw 2": "^",
}


def plot_noise_robustness(ax=None):
    """Plot normalized image noise versus relative DEA/DRA degradation.

    Returns
    -------
    tuple
        ``(fig, ax)`` for the rendered noise-robustness figure.
    """
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

    ax.axvline(
        x=1.0,
        color="#AAAAAA",
        linestyle="--",
        linewidth=1.2,
        alpha=0.55,
        zorder=1,
    )
    ax.axhline(
        y=1.0,
        color="#333333",
        linestyle="--",
        linewidth=1.0,
        alpha=0.45,
        zorder=1,
    )

    for camera_label in CAMERA_LABELS:
        for dataset in DATASETS:
            nnd = NND_PIXELS[dataset]
            base_dra = CLEAN_DRA[dataset][CAMERA_TO_METHOD[camera_label]]
            eta_values = [noise / nnd for noise in NOISE_LEVELS]
            relative_dra_values = [
                NOISE_METRICS[dataset][camera_label][noise][1] / base_dra
                for noise in NOISE_LEVELS
            ]
            ax.plot(
                eta_values,
                relative_dra_values,
                color=CAMERA_COLORS[camera_label],
                linewidth=0.9,
                alpha=0.45,
                zorder=2,
            )
            ax.scatter(
                eta_values,
                relative_dra_values,
                c=CAMERA_COLORS[camera_label],
                marker=DATASET_MARKERS[dataset],
                s=90,
                edgecolors="white",
                linewidths=0.6,
                alpha=0.92,
                zorder=4,
            )

    camera_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=CAMERA_COLORS[camera_label],
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=camera_label,
        )
        for camera_label in CAMERA_LABELS
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
    camera_legend = ax.legend(
        handles=camera_handles,
        loc="lower left",
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        fontsize=13,
        title="Cameras",
        title_fontsize=14,
    )
    ax.add_artist(camera_legend)
    ax.legend(
        handles=dataset_handles,
        loc="lower center",
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        fontsize=13,
        title="Dataset",
        title_fontsize=14,
    )

    ax.set_xlim(-0.05, 3.50)
    ax.set_ylim(0.75, 1.05)
    ax.set_xlabel(r"Normalized Noise  $\sigma_n\,/\,\mathrm{NND}$")
    ax.set_ylabel(r"DEA degradation $\%$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linestyle="-", alpha=0.18)
    ax.grid(True, which="minor", linestyle=":", alpha=0.06)
    apply_figure_layout(fig, pad=2.5)
    return fig, ax


def save_noise_robustness_figure(
    output_dir: str | Path,
    *,
    formats: Iterable[str] = ("png", "pdf"),
) -> list[Path]:
    """Create and save the noise-robustness figure to ``output_dir``."""
    fig, _ = plot_noise_robustness()
    saved: list[Path] = []
    for fmt in formats:
        path = Path(output_dir) / f"table_noise_robustness.{fmt}"
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
    fig, _ = plot_noise_robustness()
    if not args.no_save:
        for fmt in args.formats:
            path = args.output_dir / f"table_noise_robustness.{fmt}"
            save_figure(fig, path, dpi=300, bbox_inches="tight")
            print(f"Saved figure: {path}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
