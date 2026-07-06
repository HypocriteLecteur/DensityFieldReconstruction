"""Sweep GMM scale/model order, fit DRA surfaces, and plot managed results.

Reusable computation and fitting live in :mod:`dfr.analysis`. This module owns
the experiment's dataset choices, resumable cache, CLI, persistence, and plot.

Examples::

    python -m experiments.plot_dra_scale_model_order
    python -m experiments.plot_dra_scale_model_order --datasets jackdaw --force
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dfr import AnalysisConfig, OutputConfig, RunArtifacts, load_dataset
from dfr.analysis import (
    FIT_MODELS,
    ScaleAnalysisResult,
    compute_scale_model_order_surface,
    create_scale_analysis,
    fit_dra_surface,
)


@dataclass(frozen=True)
class SweepConfig:
    time_step: int


NORMALIZED_SCALES = np.linspace(1.2, 3.5, 11)
MODEL_ORDER_STEPS = 10
SWEEPS = {
    "swift": SweepConfig(1000),
    "jackdaw": SweepConfig(400),
    "starling": SweepConfig(0),
    "jackdaw2": SweepConfig(2800),
}


def load_positions(dataset_name: str, time_step: int) -> np.ndarray:
    dataset = load_dataset(dataset_name)
    return dataset.positions_at_time_step(time_step).astype(np.float32, copy=False)


def _cache_compatible(
    cached: ScaleAnalysisResult, expected: ScaleAnalysisResult
) -> bool:
    return cached.matches_grid(expected)


def cache_surface_fit(
    dataset_name: str,
    fit: dict,
    output_dir: Path,
    summary_dir: Path | None = None,
) -> None:
    """Save machine-readable fit arrays and a JSON metric summary."""
    summary_dir = output_dir if summary_dir is None else summary_dir
    best = fit["candidates"][fit["best_name"]]
    np.savez(
        output_dir / f"{dataset_name}_dra_surface_fit.npz",
        model_name=fit["best_name"],
        coefficients=best["coefficients"],
        prediction=best["prediction"],
        normalized_orders=fit["normalized_orders"],
        rmse=best["rmse"],
        cv_rmse=best["cv_rmse"],
        r_squared=best["r_squared"],
    )
    summary = {
        "selected_model": fit["best_name"],
        "equation": (
            "DRA(s,q) = 1 - exp(beta0 + beta1*ln(s) + beta2*ln(q) "
            "+ beta3*ln(s)*ln(q) + beta4*ln(s)^2 + beta5*ln(q)^2), "
            "s=scale/NND, q=L/N; omit unavailable trailing terms"
        ),
        "coefficient_order": [
            "intercept",
            "ln(s)",
            "ln(q)",
            "ln(s)*ln(q)",
            "ln(s)^2",
            "ln(q)^2",
        ],
        "candidates": {
            name: {
                "coefficients": candidate["coefficients"].tolist(),
                "rmse": candidate["rmse"],
                "cv_rmse": candidate["cv_rmse"],
                "r_squared": candidate["r_squared"],
            }
            for name, candidate in fit["candidates"].items()
        },
    }
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / f"{dataset_name}_dra_surface_fit.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=2)


def compute_surface(
    dataset_name: str,
    sweep: SweepConfig,
    output_dir: Path,
    force: bool,
    voxel_res_fraction: float,
    batch_size: int,
    positions: np.ndarray | None = None,
    normalized_scale_values: np.ndarray | None = None,
    cache_stem: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Compatibility wrapper around the typed package DRA computation."""
    positions = (
        load_positions(dataset_name, sweep.time_step)
        if positions is None
        else positions.astype(np.float32, copy=False)
    )
    normalized_scales = (
        NORMALIZED_SCALES
        if normalized_scale_values is None
        else normalized_scale_values
    )
    result = create_scale_analysis(
        dataset_name=dataset_name,
        time_step=sweep.time_step,
        positions=positions,
        normalized_scales=normalized_scales,
        voxel_res_fraction=voxel_res_fraction,
        model_order_steps=MODEL_ORDER_STEPS,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f"{cache_stem or dataset_name}_dra_scale_model_order.npz"
    if cache_path.exists() and not force:
        cached = ScaleAnalysisResult.load_npz(
            cache_path,
            dataset_name=dataset_name,
            number_of_animals=len(positions),
        )
        if _cache_compatible(cached, result):
            result = cached
            state = "complete" if result.is_complete else "partial"
            print(f"[{dataset_name}] loaded {state} cache: {cache_path}")
            if result.is_complete:
                return result.as_legacy_tuple()

    data_span = float(np.max(positions.max(axis=0) - positions.min(axis=0)))
    print(
        f"[{dataset_name}] frame={sweep.time_step}, N={len(positions)}, "
        f"mean NND={result.mean_nnd:.5g}, "
        f"voxel_res={data_span * voxel_res_fraction:.5g}, "
        f"L={result.component_counts.tolist()}"
    )

    def row_completed(current: ScaleAnalysisResult, scale_index: int, timing: dict):
        indices = timing["component_indices"]
        for component_index in indices:
            print(
                f"  scale/NND={current.normalized_scales[scale_index]:.3f}, "
                f"scale={current.scales[scale_index]:.3f}, "
                f"L={current.component_counts[component_index]:3d}: "
                f"DRA={current.dra[scale_index, component_index]:.5f}"
            )
        print(
            f"  row timing: GMR={timing['gmr_seconds']:.2f}s, "
            f"DRA={timing['dra_seconds']:.2f}s, "
            f"total={timing['gmr_seconds'] + timing['dra_seconds']:.2f}s"
        )
        current.save_npz(cache_path)

    compute_scale_model_order_surface(
        positions,
        result,
        batch_size=batch_size,
        row_callback=row_completed,
    )
    return result.as_legacy_tuple()


def plot_surfaces(
    results: dict[str, tuple],
    fits: dict[str, dict],
    output_path: Path,
    show: bool,
) -> None:
    figure = plt.figure(figsize=(15, 11))
    surface = None
    for plot_index, (dataset_name, result) in enumerate(results.items(), start=1):
        normalized_scales, _, components, dra, _, number_of_animals = result
        actual_percentages = 100.0 * components / number_of_animals
        axis = figure.add_subplot(2, 2, plot_index, projection="3d")
        scale_grid, component_grid = np.meshgrid(
            normalized_scales, actual_percentages, indexing="ij"
        )
        surface = axis.plot_surface(
            scale_grid,
            component_grid,
            dra,
            cmap="viridis",
            edgecolor="none",
            antialiased=True,
            alpha=0.88,
        )
        best = fits[dataset_name]["candidates"][fits[dataset_name]["best_name"]]
        axis.plot_wireframe(
            scale_grid,
            component_grid,
            best["prediction"],
            color="black",
            linewidth=0.65,
            rstride=1,
            cstride=1,
        )
        axis.set_title(
            f"{dataset_name.capitalize()} ({fits[dataset_name]['best_name']}, "
            f"$R^2$={best['r_squared']:.3f})"
        )
        axis.set_xlabel("Scale / mean NND")
        axis.set_ylabel("Model order / N (%)")
        axis.set_zlabel("DRA")
        axis.set_yticks(actual_percentages)
        axis.set_yticklabels(
            [
                f"{percentage:.1f}\n({component})"
                for percentage, component in zip(actual_percentages, components)
            ],
            fontsize=7,
        )
        axis.view_init(elev=28, azim=-130)
    if surface is not None:
        figure.colorbar(surface, ax=figure.axes, shrink=0.62, pad=0.06, label="DRA")
    figure.subplots_adjust(
        left=0.02, right=0.88, bottom=0.04, top=0.94, wspace=0.03, hspace=0.12
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    if show:
        plt.show()
    else:
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(SWEEPS), default=list(SWEEPS)
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-id", default="dra-scale-model-order")
    parser.add_argument("--overwrite-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Ignore cached data.")
    parser.add_argument("--voxel-res-fraction", type=float, default=5e-3)
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA-capable PyTorch setup.")
    if args.voxel_res_fraction <= 0 or args.batch_size <= 0:
        raise ValueError("Voxel resolution fraction and batch size must be positive.")
    project_root = Path(__file__).resolve().parents[1]
    analysis_config = AnalysisConfig(
        frames=tuple(SWEEPS[name].time_step for name in args.datasets),
        scales=tuple(float(value) for value in NORMALIZED_SCALES),
        device="cuda",
    )
    artifacts = RunArtifacts.create(
        OutputConfig(
            workflow="analysis",
            name="DRA scale and model order",
            root=args.output_root,
            run_id=args.run_id,
            project_root=project_root,
            resume=not args.overwrite_run,
            overwrite=args.overwrite_run,
        ),
        resolved_config={
            "datasets": args.datasets,
            "sweeps": {
                name: {"time_step": SWEEPS[name].time_step} for name in args.datasets
            },
            "analysis": analysis_config,
            "model_order_steps": MODEL_ORDER_STEPS,
            "fit_models": FIT_MODELS,
            "voxel_res_fraction": args.voxel_res_fraction,
            "batch_size": args.batch_size,
        },
        device="cuda",
        metadata={"entrypoint": "experiments.plot_dra_scale_model_order"},
    )
    results = {
        name: compute_surface(
            dataset_name=name,
            sweep=SWEEPS[name],
            output_dir=artifacts.cache_dir,
            force=args.force,
            voxel_res_fraction=args.voxel_res_fraction,
            batch_size=args.batch_size,
        )
        for name in args.datasets
    }
    fits = {}
    for name, result in results.items():
        normalized_scales, _, components, dra, _, number_of_animals = result
        fits[name] = fit_dra_surface(
            normalized_scales, components, number_of_animals, dra
        )
        cache_surface_fit(
            name, fits[name], artifacts.data_dir, summary_dir=artifacts.metrics_dir
        )
        best = fits[name]["candidates"][fits[name]["best_name"]]
        print(
            f"[{name}] fit={fits[name]['best_name']}, RMSE={best['rmse']:.5f}, "
            f"CV-RMSE={best['cv_rmse']:.5f}, R^2={best['r_squared']:.5f}"
        )
    plot_surfaces(
        results,
        fits,
        artifacts.figures_dir / "dra_scale_model_order_3d.png",
        show=args.show,
    )
    artifacts.save_json(
        "fit_index.json",
        {
            name: {
                "selected_model": fit["best_name"],
                "rmse": fit["candidates"][fit["best_name"]]["rmse"],
                "cv_rmse": fit["candidates"][fit["best_name"]]["cv_rmse"],
                "r_squared": fit["candidates"][fit["best_name"]]["r_squared"],
            }
            for name, fit in fits.items()
        },
        category="metrics",
        overwrite=True,
    )
    print(f"Managed run directory: {artifacts.run_dir}")


if __name__ == "__main__":
    main()
