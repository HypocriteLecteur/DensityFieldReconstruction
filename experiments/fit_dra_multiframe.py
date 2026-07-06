"""Fit shared DRA(scale/NND, model-order/N) laws across multiple frames.

The expensive DRA grid for every frame is cached independently. Candidate
functional forms are selected with leave-one-frame-out cross-validation, both
within each dataset and globally across all sampled datasets. All frame grids
span 0.5--1.5 times their frame-specific mean nearest-neighbour distance.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dfr import load_dataset as load_dfr_dataset
from experiments.plot_dra_scale_model_order import (
    FIT_MODELS,
    SWEEPS,
    SweepConfig,
    compute_surface,
    fit_design_matrix,
    fit_one_surface_model,
)


FRAME_RANGES = {
    "swift": (0, None),
    "jackdaw": (350, 550),
    "starling": (0, 2),
    "jackdaw2": (2700, 3460),
}

# Every multiframe fit uses the same NND-normalized scale support. Keeping this
# local to the experiment prevents changes to the single-frame sweep defaults.
MULTIFRAME_NORMALIZED_SCALES = np.linspace(0.5, 1.5, 11)


@dataclass
class FrameGrid:
    dataset: str
    time_step: int
    number_of_animals: int
    mean_nnd: float
    scale: np.ndarray
    order: np.ndarray
    dra: np.ndarray


def select_frames(start: int, stop: int, count: int, preferred: int) -> np.ndarray:
    """Include endpoints/preferred frame, then bisect the largest time gaps."""
    if stop <= start:
        raise ValueError(f"Empty frame interval [{start}, {stop}).")
    available = stop - start
    count = min(count, available)
    if count == available:
        return np.arange(start, stop, dtype=int)

    selected = {start, stop - 1, min(max(preferred, start), stop - 1)}
    while len(selected) < count:
        ordered = sorted(selected)
        gaps = [(right - left, left, right) for left, right in zip(ordered, ordered[1:])]
        _, left, right = max(gaps)
        midpoint = (left + right) // 2
        if midpoint in selected:
            midpoint += 1
        selected.add(midpoint)
    return np.asarray(sorted(selected), dtype=int)


def load_dataset(dataset_name: str):
    """Load a registered scenario through the canonical package API."""
    return load_dfr_dataset(dataset_name)


def seed_existing_cache(
    dataset_name: str,
    time_step: int,
    frame_dir: Path,
    force: bool,
    normalized_scales: np.ndarray,
) -> None:
    """Reuse a one-frame cache only when its normalized scale grid matches."""
    if force or time_step != SWEEPS[dataset_name].time_step:
        return
    source = (
        Path("results")
        / "dra_scale_model_order"
        / f"{dataset_name}_dra_scale_model_order.npz"
    )
    destination = frame_dir / f"{dataset_name}_dra_scale_model_order.npz"
    if source.exists() and not destination.exists():
        with np.load(source) as cache:
            if ("normalized_scales" in cache.files
                    and np.array_equal(cache["normalized_scales"], normalized_scales)):
                shutil.copy2(source, destination)


def flatten_frame_result(
    dataset_name: str,
    time_step: int,
    result: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int],
) -> FrameGrid:
    normalized_scales, _, components, dra, mean_nnd, number_of_animals = result
    normalized_orders = components / number_of_animals
    scale_grid, order_grid = np.meshgrid(
        normalized_scales, normalized_orders, indexing="ij"
    )
    return FrameGrid(
        dataset=dataset_name,
        time_step=time_step,
        number_of_animals=number_of_animals,
        mean_nnd=mean_nnd,
        scale=scale_grid.ravel(),
        order=order_grid.ravel(),
        dra=dra.ravel(),
    )


def concatenate_frames(frames: list[FrameGrid]):
    return (
        np.concatenate([frame.scale for frame in frames]),
        np.concatenate([frame.order for frame in frames]),
        np.concatenate([frame.dra for frame in frames]),
    )


def grouped_cv_rmse(frames: list[FrameGrid], model_name: str) -> float:
    """Test each full frame using a model trained on all other frames."""
    errors = []
    for held_index, held_frame in enumerate(frames):
        training_frames = [frame for index, frame in enumerate(frames) if index != held_index]
        if not training_frames:
            return float("nan")
        scale, order, dra = concatenate_frames(training_frames)
        fitted = fit_one_surface_model(scale, order, dra, model_name)
        prediction = 1.0 - np.exp(
            fit_design_matrix(held_frame.scale, held_frame.order, model_name)
            @ fitted["coefficients"]
        )
        errors.extend(prediction - held_frame.dra)
    return float(np.sqrt(np.mean(np.square(errors))))


def leave_one_dataset_out_rmse(frames: list[FrameGrid], model_name: str) -> float:
    errors = []
    for held_dataset in sorted({frame.dataset for frame in frames}):
        training = [frame for frame in frames if frame.dataset != held_dataset]
        held = [frame for frame in frames if frame.dataset == held_dataset]
        scale, order, dra = concatenate_frames(training)
        fitted = fit_one_surface_model(scale, order, dra, model_name)
        for held_frame in held:
            prediction = 1.0 - np.exp(
                fit_design_matrix(held_frame.scale, held_frame.order, model_name)
                @ fitted["coefficients"]
            )
            errors.extend(prediction - held_frame.dra)
    return float(np.sqrt(np.mean(np.square(errors))))


def fit_frames(frames: list[FrameGrid], include_dataset_cv: bool = False) -> dict:
    scale, order, dra = concatenate_frames(frames)
    candidates = {}
    for model_name in FIT_MODELS:
        fitted = fit_one_surface_model(scale, order, dra, model_name)
        fitted["frame_cv_rmse"] = grouped_cv_rmse(frames, model_name)
        if include_dataset_cv:
            fitted["dataset_cv_rmse"] = leave_one_dataset_out_rmse(
                frames, model_name
            )
        candidates[model_name] = fitted
    best_name = min(candidates, key=lambda name: candidates[name]["frame_cv_rmse"])
    return {"best_name": best_name, "candidates": candidates}


def save_samples(dataset_name: str, frames: list[FrameGrid], output_dir: Path) -> None:
    scale, order, dra = concatenate_frames(frames)
    frame_ids = np.concatenate(
        [np.full(len(frame.dra), frame.time_step, dtype=int) for frame in frames]
    )
    np.savez(
        output_dir / f"{dataset_name}_multiframe_samples.npz",
        normalized_scale=scale,
        normalized_order=order,
        dra=dra,
        time_step=frame_ids,
        sampled_frames=np.asarray([frame.time_step for frame in frames]),
        agent_counts=np.asarray([frame.number_of_animals for frame in frames]),
        mean_nnds=np.asarray([frame.mean_nnd for frame in frames]),
    )


def save_fit(name: str, fit: dict, output_dir: Path) -> None:
    best = fit["candidates"][fit["best_name"]]
    np.savez(
        output_dir / f"{name}_multiframe_fit.npz",
        model_name=fit["best_name"],
        coefficients=best["coefficients"],
        rmse=best["rmse"],
        frame_cv_rmse=best["frame_cv_rmse"],
        r_squared=best["r_squared"],
        dataset_cv_rmse=best.get("dataset_cv_rmse", np.nan),
    )
    payload = {
        "selected_model": fit["best_name"],
        "variables": {"s": "scale / mean NND", "q": "component count / N"},
        "equation": (
            "DRA = 1 - exp(b0 + b1*ln(s) + b2*ln(q) + "
            "b3*ln(s)*ln(q) + b4*ln(s)^2 + b5*ln(q)^2); "
            "omit trailing terms not present in the selected model"
        ),
        "coefficient_order": [
            "intercept", "ln(s)", "ln(q)", "ln(s)*ln(q)",
            "ln(s)^2", "ln(q)^2",
        ],
        "candidates": {
            model_name: {
                "coefficients": candidate["coefficients"].tolist(),
                "rmse": candidate["rmse"],
                "frame_cv_rmse": candidate["frame_cv_rmse"],
                "dataset_cv_rmse": candidate.get("dataset_cv_rmse"),
                "r_squared": candidate["r_squared"],
            }
            for model_name, candidate in fit["candidates"].items()
        },
    }
    with open(output_dir / f"{name}_multiframe_fit.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def plot_dataset_fits(
    dataset_frames: dict[str, list[FrameGrid]],
    fits: dict[str, dict],
    output_path: Path,
) -> None:
    figure = plt.figure(figsize=(15, 11))
    for plot_index, (dataset_name, frames) in enumerate(dataset_frames.items(), 1):
        axis = figure.add_subplot(2, 2, plot_index, projection="3d")
        for frame in frames:
            axis.scatter(
                frame.scale,
                100.0 * frame.order,
                frame.dra,
                s=5,
                alpha=0.28,
            )
        fit = fits[dataset_name]
        best = fit["candidates"][fit["best_name"]]
        scale_axis = np.linspace(
            MULTIFRAME_NORMALIZED_SCALES[0],
            MULTIFRAME_NORMALIZED_SCALES[-1],
            40,
        )
        orders = np.concatenate([frame.order for frame in frames])
        order_axis = np.linspace(orders.min(), orders.max(), 40)
        scale_grid, order_grid = np.meshgrid(scale_axis, order_axis, indexing="ij")
        prediction = 1.0 - np.exp(
            fit_design_matrix(
                scale_grid.ravel(), order_grid.ravel(), fit["best_name"]
            )
            @ best["coefficients"]
        )
        axis.plot_wireframe(
            scale_grid,
            100.0 * order_grid,
            prediction.reshape(scale_grid.shape),
            color="black",
            linewidth=0.55,
            rstride=3,
            cstride=3,
        )
        axis.set_title(
            f"{dataset_name.capitalize()}: {fit['best_name']}\n"
            f"frame-CV RMSE={best['frame_cv_rmse']:.3f}, $R^2$={best['r_squared']:.3f}"
        )
        axis.set_xlabel("Scale / mean NND")
        axis.set_ylabel("Model order / N (%)")
        axis.set_zlabel("DRA")
        axis.view_init(elev=27, azim=-130)
    figure.subplots_adjust(left=0.03, right=0.97, bottom=0.04, top=0.93, wspace=0.05)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _metrics_table(fit: dict, include_dataset_cv: bool = False) -> str:
    """Format every candidate model using a publication-friendly text table."""
    models = list(FIT_MODELS)
    rows = [
        ("RMSE", "rmse", 4),
        ("Frame-CV RMSE", "frame_cv_rmse", 4),
    ]
    if include_dataset_cv:
        rows.append(("Dataset-CV RMSE", "dataset_cv_rmse", 4))
    rows.append(("R²", "r_squared", 3))

    values = [["Metric", *models]]
    for label, key, decimals in rows:
        values.append([
            label,
            *[
                f"{fit['candidates'][model][key]:.{decimals}f}"
                for model in models
            ],
        ])

    widths = [
        max(len(row[column]) for row in values)
        for column in range(len(values[0]))
    ]
    top = "┌" + "┬".join("─" * (width + 2) for width in widths) + "┐"
    separator = "├" + "┼".join("─" * (width + 2) for width in widths) + "┤"
    bottom = "└" + "┴".join("─" * (width + 2) for width in widths) + "┘"

    def format_row(row: list[str], header: bool = False) -> str:
        cells = []
        for index, (cell, width) in enumerate(zip(row, widths)):
            cells.append(cell.center(width) if header or index == 0 else cell.ljust(width))
        return "│ " + " │ ".join(cells) + " │"

    lines = [top, format_row(values[0], header=True), separator]
    for row_index, row in enumerate(values[1:]):
        lines.append(format_row(row))
        if row_index < len(values) - 2:
            lines.append(separator)
    lines.append(bottom)
    return "\n".join(lines)


def format_results_report(
    dataset_frames: dict[str, list[FrameGrid]],
    fits: dict[str, dict],
    universal_fit: dict,
) -> str:
    """Return the complete per-dataset and pooled model-comparison report."""
    display_names = {
        "swift": "Swift",
        "jackdaw": "Jackdaw",
        "starling": "Starling",
        "jackdaw2": "Jackdaw2",
    }
    lines = ["---", "Per-Dataset Results", ""]
    for dataset_name, frames in dataset_frames.items():
        frame_ids = ", ".join(str(frame.time_step) for frame in frames)
        counts = [frame.number_of_animals for frame in frames]
        count_summary = (
            f"N={counts[0]}" if min(counts) == max(counts)
            else f"N={min(counts)}–{max(counts)}"
        )
        lines.extend([
            f"{display_names.get(dataset_name, dataset_name.capitalize())} "
            f"({len(frames)} frames: {frame_ids} — {count_summary})",
            "",
            _metrics_table(fits[dataset_name]),
            "",
        ])

    all_frames = [frame for frames in dataset_frames.values() for frame in frames]
    lines.extend([
        f"Universal (all {len(dataset_frames)} datasets pooled, "
        f"{len(all_frames)} frames)",
        "",
        _metrics_table(
            universal_fit,
            include_dataset_cv=all(
                "dataset_cv_rmse" in universal_fit["candidates"][model]
                for model in FIT_MODELS
            ),
        ),
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-per-dataset", type=int, default=5)
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(FRAME_RANGES), default=list(FRAME_RANGES)
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "dra_scale_model_order_multiframe_0p5_1p5",
    )
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--voxel-res-fraction", type=float, default=5e-3)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    # The report uses box-drawing characters and R²; force a portable UTF-8
    # console encoding on Windows while still writing the report as UTF-8.
    if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_frames = {}

    for dataset_name in args.datasets:
        dataset = load_dataset(dataset_name)
        start, configured_stop = FRAME_RANGES[dataset_name]
        stop = min(configured_stop or len(dataset.trajectories), len(dataset.trajectories))
        time_steps = select_frames(
            start,
            stop,
            args.frames_per_dataset,
            SWEEPS[dataset_name].time_step,
        )
        print(f"[{dataset_name}] sampled frames: {time_steps.tolist()}")
        frames = []
        for time_step in time_steps:
            frame_dir = args.output_dir / dataset_name / f"frame_{time_step:05d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            seed_existing_cache(
                dataset_name,
                int(time_step),
                frame_dir,
                args.force,
                MULTIFRAME_NORMALIZED_SCALES,
            )
            positions = dataset.positions_at_time_step(int(time_step))
            result = compute_surface(
                dataset_name=dataset_name,
                sweep=SweepConfig(int(time_step)),
                output_dir=frame_dir,
                force=args.force,
                voxel_res_fraction=args.voxel_res_fraction,
                batch_size=args.batch_size,
                positions=positions,
                normalized_scale_values=MULTIFRAME_NORMALIZED_SCALES,
            )
            frames.append(flatten_frame_result(dataset_name, int(time_step), result))
        dataset_frames[dataset_name] = frames
        save_samples(dataset_name, frames, args.output_dir)

    fits = {}
    for dataset_name, frames in dataset_frames.items():
        fits[dataset_name] = fit_frames(frames)
        save_fit(dataset_name, fits[dataset_name], args.output_dir)
        best = fits[dataset_name]["candidates"][fits[dataset_name]["best_name"]]
        print(
            f"[{dataset_name}] {fits[dataset_name]['best_name']}: "
            f"RMSE={best['rmse']:.4f}, frame-CV={best['frame_cv_rmse']:.4f}, "
            f"R2={best['r_squared']:.4f}"
        )

    all_frames = [frame for frames in dataset_frames.values() for frame in frames]
    universal_fit = fit_frames(
        all_frames, include_dataset_cv=len(dataset_frames) > 1
    )
    save_fit("universal", universal_fit, args.output_dir)
    best = universal_fit["candidates"][universal_fit["best_name"]]
    print(
        f"[universal] {universal_fit['best_name']}: RMSE={best['rmse']:.4f}, "
        f"frame-CV={best['frame_cv_rmse']:.4f}, "
        f"dataset-CV={best.get('dataset_cv_rmse', float('nan')):.4f}, "
        f"R2={best['r_squared']:.4f}"
    )
    report = format_results_report(dataset_frames, fits, universal_fit)
    print("\n" + report)
    with open(args.output_dir / "multiframe_results.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    plot_dataset_fits(
        dataset_frames,
        fits,
        args.output_dir / "multiframe_dra_fits.png",
    )


if __name__ == "__main__":
    main()
