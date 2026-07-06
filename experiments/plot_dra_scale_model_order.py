"""Sweep GMM scale and model order and plot DRA surfaces.

For one representative frame from each biological dataset, this script treats
the animal positions as a unit-weight isotropic GMM. It sweeps scale / mean NND
from 1.2 to 3.5, and model order / agent count from 1% to 10% (or 30% when
N < 100). At every parameter pair it reduces that GMM with Runnalls' algorithm,
then computes

    DRA = 1 - (false-positive mass + false-negative mass) / number of animals.

Run from the repository root, for example::

    python experiments/plot_dra_scale_model_order.py
    python experiments/plot_dra_scale_model_order.py --datasets jackdaw --force

CUDA is required by both the GMR implementation and the voxelized DRA metric.
Intermediate results are cached after every scale so an interrupted sweep can
be resumed.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree

from dfr.dataset_io import DatasetFactory
from dfr.gaussian_mixture_reduction import GMR
from dfr.simulation_config import SimulationConfig


@dataclass(frozen=True)
class SweepConfig:
    time_step: int


# Scale is expressed relative to the frame's mean nearest-neighbour distance.
NORMALIZED_SCALES = np.linspace(1.2, 3.5, 11)
MODEL_ORDER_STEPS = 10
FIT_MODELS = ("power", "power_interaction", "log_quadratic")


SWEEPS = {
    "swift": SweepConfig(1000),
    "jackdaw": SweepConfig(400),
    "starling": SweepConfig(0),
    "jackdaw2": SweepConfig(2800),
}


def load_positions(dataset_name: str, time_step: int) -> np.ndarray:
    scenario_dir = Path("scenarios") / dataset_name
    config = SimulationConfig(str(scenario_dir / "config.yaml"))
    dataset = DatasetFactory().get_dataset(config.data_file)
    return dataset.positions_at_time_step(time_step).astype(np.float32, copy=False)


def mean_nearest_neighbour_distance(positions: np.ndarray) -> float:
    """Return the mean distance from every agent to its nearest other agent."""
    if len(positions) < 2:
        raise ValueError("At least two agents are required to calculate NND.")
    distances, _ = cKDTree(positions).query(positions, k=2)
    return float(np.mean(distances[:, 1]))


def model_orders(number_of_agents: int) -> tuple[np.ndarray, np.ndarray]:
    """Return rounded component counts and requested percentages of N."""
    maximum_fraction = 0.30 if number_of_agents < 100 else 0.10
    requested_fractions = np.linspace(0.01, maximum_fraction, MODEL_ORDER_STEPS)
    components = np.clip(
        np.rint(number_of_agents * requested_fractions).astype(int),
        1,
        number_of_agents,
    )
    if len(np.unique(components)) != MODEL_ORDER_STEPS:
        raise ValueError(
            f"The model-order sweep produces duplicate integer counts for "
            f"N={number_of_agents}: {components.tolist()}"
        )
    requested_percentages = 100.0 * requested_fractions
    return components, requested_percentages


def cache_matches(
    cache: np.lib.npyio.NpzFile,
    time_step: int,
    scales: np.ndarray,
    normalized_scales: np.ndarray,
    components: np.ndarray,
    model_order_percentages: np.ndarray,
    mean_nnd: float,
    voxel_res_fraction: float,
) -> bool:
    return (
        "voxel_res_fraction" in cache.files
        and "normalized_scales" in cache.files
        and "model_order_percentages" in cache.files
        and "mean_nnd" in cache.files
        and int(cache["time_step"]) == time_step
        and np.array_equal(cache["scales"], scales)
        and np.array_equal(cache["normalized_scales"], normalized_scales)
        and np.array_equal(cache["component_numbers"], components)
        and np.array_equal(
            cache["model_order_percentages"], model_order_percentages
        )
        and float(cache["mean_nnd"]) == mean_nnd
        and float(cache["voxel_res_fraction"]) == voxel_res_fraction
        and cache["dra"].shape == (len(scales), len(components))
    )


def save_cache(
    path: Path,
    time_step: int,
    scales: np.ndarray,
    normalized_scales: np.ndarray,
    components: np.ndarray,
    model_order_percentages: np.ndarray,
    mean_nnd: float,
    dra: np.ndarray,
    voxel_res_fraction: float,
) -> None:
    np.savez(
        path,
        time_step=time_step,
        scales=scales,
        normalized_scales=normalized_scales,
        component_numbers=components,
        model_order_percentages=model_order_percentages,
        mean_nnd=mean_nnd,
        dra=dra,
        voxel_res_fraction=voxel_res_fraction,
    )


@torch.inference_mode()
def compute_dra_sweep(
    positions: np.ndarray,
    scale: float,
    reduced_models: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    bounds: np.ndarray,
    voxel_res: np.float64,
    batch_size: int,
) -> np.ndarray:
    """Evaluate all model orders in one shared voxel traversal.

    Ground-truth density and voxel coordinates are computed once per batch.
    Since FP + FN equals the L1 density error, only that sum is accumulated;
    keeping accumulators on the GPU also avoids synchronization per batch.
    """
    device = torch.device("cuda")
    number_of_animals = len(positions)
    gt_means = torch.as_tensor(positions, device=device, dtype=torch.float32)
    gt_weights = torch.ones(number_of_animals, device=device)
    gt_sigmas = torch.full((number_of_animals,), scale, device=device)

    # Concatenate every mixture so each voxel batch needs only one cdist and
    # one Gaussian-kernel evaluation. Slices below recover the individual
    # mixture densities from the shared component-density matrix.
    all_means = [gt_means]
    all_weights = [gt_weights]
    all_sigmas = [gt_sigmas]
    mixture_slices = []
    component_offset = number_of_animals
    for model_means, model_weights, model_sigmas in reduced_models:
        component_count = len(model_means)
        all_means.append(model_means)
        all_weights.append(model_weights.reshape(-1))
        all_sigmas.append(model_sigmas.reshape(-1))
        mixture_slices.append(
            slice(component_offset, component_offset + component_count)
        )
        component_offset += component_count
    all_means = torch.cat(all_means)
    all_weights = torch.cat(all_weights)
    all_sigmas = torch.cat(all_sigmas)
    variances = all_sigmas.square()
    inverse_negative_two_variances = -0.5 / variances
    normalized_weights = all_weights / (2.0 * torch.pi * variances).pow(1.5)

    ticks = [
        torch.arange(axis_bounds[0], axis_bounds[1], voxel_res, device=device)
        for axis_bounds in bounds
    ]
    nx, ny, nz = (len(axis_ticks) for axis_ticks in ticks)
    total_voxels = nx * ny * nz
    absolute_error_sums = torch.zeros(len(reduced_models), device=device)

    for start_index in range(0, total_voxels, batch_size):
        end_index = min(start_index + batch_size, total_voxels)
        flat_indices = torch.arange(start_index, end_index, device=device)
        ix = flat_indices // (ny * nz)
        iy = (flat_indices // nz) % ny
        iz = flat_indices % nz
        coordinates = torch.stack(
            (ticks[0][ix], ticks[1][iy], ticks[2][iz]), dim=-1
        )
        component_densities = torch.cdist(coordinates, all_means).square_()
        component_densities.mul_(inverse_negative_two_variances).exp_()
        component_densities.mul_(normalized_weights)
        gt_density = component_densities[:, :number_of_animals].sum(dim=1)

        for model_index, model_slice in enumerate(mixture_slices):
            predicted_density = component_densities[:, model_slice].sum(dim=1)
            absolute_error_sums[model_index] += torch.sum(
                torch.abs(predicted_density - gt_density)
            )

    l1_masses = absolute_error_sums.cpu().numpy() * float(voxel_res ** 3)
    return 1.0 - l1_masses / number_of_animals


def fit_design_matrix(
    normalized_scale: np.ndarray,
    normalized_order: np.ndarray,
    model_name: str,
) -> np.ndarray:
    """Build features for log(1 - DRA) as a function of log scale/order."""
    log_scale = np.log(np.asarray(normalized_scale))
    log_order = np.log(np.asarray(normalized_order))
    columns = [np.ones(log_scale.size), log_scale, log_order]
    if model_name in ("power_interaction", "log_quadratic"):
        columns.append(log_scale * log_order)
    if model_name == "log_quadratic":
        columns.extend((np.square(log_scale), np.square(log_order)))
    if model_name not in FIT_MODELS:
        raise ValueError(f"Unknown fit model: {model_name}")
    return np.column_stack(columns)


def fit_one_surface_model(
    normalized_scale: np.ndarray,
    normalized_order: np.ndarray,
    dra: np.ndarray,
    model_name: str,
) -> dict:
    design = fit_design_matrix(normalized_scale, normalized_order, model_name)
    log_error = np.log(np.maximum(1.0 - dra, 1e-8))
    coefficients = np.linalg.lstsq(design, log_error, rcond=None)[0]
    prediction = 1.0 - np.exp(design @ coefficients)
    residual = prediction - dra
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    denominator = np.sum((dra - dra.mean()) ** 2)
    r_squared = float(1.0 - np.sum(np.square(residual)) / denominator)
    return {
        "name": model_name,
        "coefficients": coefficients,
        "prediction": prediction,
        "rmse": rmse,
        "r_squared": r_squared,
    }


def cross_validated_rmse(
    scale_grid: np.ndarray,
    order_grid: np.ndarray,
    dra: np.ndarray,
    model_name: str,
) -> float:
    """Hold out each complete scale row and model-order column in turn."""
    errors = []
    for scale_index in range(dra.shape[0]):
        training_mask = np.ones(dra.shape, dtype=bool)
        training_mask[scale_index, :] = False
        fitted = fit_one_surface_model(
            scale_grid[training_mask],
            order_grid[training_mask],
            dra[training_mask],
            model_name,
        )
        held_out_design = fit_design_matrix(
            scale_grid[scale_index], order_grid[scale_index], model_name
        )
        prediction = 1.0 - np.exp(
            held_out_design @ fitted["coefficients"]
        )
        errors.extend(prediction - dra[scale_index])

    for order_index in range(dra.shape[1]):
        training_mask = np.ones(dra.shape, dtype=bool)
        training_mask[:, order_index] = False
        fitted = fit_one_surface_model(
            scale_grid[training_mask],
            order_grid[training_mask],
            dra[training_mask],
            model_name,
        )
        held_out_design = fit_design_matrix(
            scale_grid[:, order_index], order_grid[:, order_index], model_name
        )
        prediction = 1.0 - np.exp(
            held_out_design @ fitted["coefficients"]
        )
        errors.extend(prediction - dra[:, order_index])

    return float(np.sqrt(np.mean(np.square(errors))))


def fit_dra_surface(
    normalized_scales: np.ndarray,
    components: np.ndarray,
    number_of_animals: int,
    dra: np.ndarray,
) -> dict:
    normalized_orders = components / number_of_animals
    scale_grid, order_grid = np.meshgrid(
        normalized_scales, normalized_orders, indexing="ij"
    )
    candidates = {}
    for model_name in FIT_MODELS:
        fitted = fit_one_surface_model(
            scale_grid.ravel(), order_grid.ravel(), dra.ravel(), model_name
        )
        fitted["prediction"] = fitted["prediction"].reshape(dra.shape)
        fitted["cv_rmse"] = cross_validated_rmse(
            scale_grid, order_grid, dra, model_name
        )
        candidates[model_name] = fitted
    best_name = min(candidates, key=lambda name: candidates[name]["cv_rmse"])
    return {
        "best_name": best_name,
        "normalized_orders": normalized_orders,
        "candidates": candidates,
    }


def cache_surface_fit(dataset_name: str, fit: dict, output_dir: Path) -> None:
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
    with open(
        output_dir / f"{dataset_name}_dra_surface_fit.json",
        "w",
        encoding="utf-8",
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
    if positions is None:
        positions = load_positions(dataset_name, sweep.time_step)
    else:
        positions = positions.astype(np.float32, copy=False)
    number_of_animals = len(positions)
    mean_nnd = mean_nearest_neighbour_distance(positions)
    if normalized_scale_values is None:
        normalized_scales = NORMALIZED_SCALES.copy()
    else:
        normalized_scales = np.asarray(normalized_scale_values, dtype=np.float64)
        if (normalized_scales.ndim != 1 or len(normalized_scales) < 2
                or not np.all(np.isfinite(normalized_scales))
                or np.any(normalized_scales <= 0)
                or np.any(np.diff(normalized_scales) <= 0)):
            raise ValueError(
                "normalized_scale_values must be a strictly increasing 1D "
                "array of at least two positive finite scales."
            )
    scales = normalized_scales * mean_nnd
    components, model_order_percentages = model_orders(number_of_animals)
    cache_name = cache_stem or dataset_name
    cache_path = output_dir / f"{cache_name}_dra_scale_model_order.npz"
    dra = np.full((len(scales), len(components)), np.nan, dtype=np.float64)

    if cache_path.exists() and not force:
        with np.load(cache_path) as cache:
            if cache_matches(
                cache,
                sweep.time_step,
                scales,
                normalized_scales,
                components,
                model_order_percentages,
                mean_nnd,
                voxel_res_fraction,
            ):
                dra[:] = cache["dra"]
                if np.all(np.isfinite(dra)):
                    print(f"[{dataset_name}] loaded complete cache: {cache_path}")
                    return (
                        normalized_scales,
                        model_order_percentages,
                        components,
                        dra,
                        mean_nnd,
                        number_of_animals,
                    )
                print(f"[{dataset_name}] resuming partial cache: {cache_path}")

    data_min = positions.min(axis=0)
    data_max = positions.max(axis=0)
    data_span = float(np.max(data_max - data_min))
    # Keep a NumPy scalar because compute_metrics_batched_torch returns
    # accumulated NumPy scalars via .item().
    voxel_res = np.float64(data_span * voxel_res_fraction)
    means = torch.as_tensor(positions, device="cuda", dtype=torch.float32)
    weights = torch.ones((number_of_animals, 1), device="cuda")

    print(
        f"[{dataset_name}] frame={sweep.time_step}, N={number_of_animals}, "
        f"mean NND={mean_nnd:.5g}, voxel_res={voxel_res:.5g}, "
        f"L={components.tolist()}"
    )
    for scale_index, scale in enumerate(scales):
        # Three standard deviations on every side matches plot_dra_and_loss.
        bounds = np.column_stack(
            (data_min - 3.0 * scale, data_max + 3.0 * scale)
        )
        radii = torch.full(
            (number_of_animals, 1), float(scale), device="cuda"
        )
        missing_indices = np.flatnonzero(~np.isfinite(dra[scale_index]))
        if len(missing_indices) == 0:
            continue

        missing_components = components[missing_indices]
        gmr_start = time.perf_counter()
        snapshots = GMR.runnalls_algorithm_simple_torch(
            means=means,
            radii=radii,
            weights=weights,
            L=int(missing_components.min()),
            DEVICE="cuda",
            snapshot_Ls=missing_components,
        )
        torch.cuda.synchronize()
        gmr_seconds = time.perf_counter() - gmr_start
        reduced_models = []
        for component_number in missing_components:
            reduced_means, reduced_weights, reduced_covariances = snapshots[
                int(component_number)
            ]
            reduced_sigmas = torch.sqrt(
                reduced_covariances[:, 0, 0]
            ).reshape(-1, 1)
            reduced_models.append(
                (reduced_means, reduced_weights, reduced_sigmas)
            )

        dra_start = time.perf_counter()
        dra[scale_index, missing_indices] = compute_dra_sweep(
            positions=positions,
            scale=float(scale),
            reduced_models=reduced_models,
            bounds=bounds,
            voxel_res=voxel_res,
            batch_size=batch_size,
        )
        torch.cuda.synchronize()
        dra_seconds = time.perf_counter() - dra_start
        for component_index in missing_indices:
            component_number = components[component_index]
            print(
                f"  scale/NND={normalized_scales[scale_index]:.3f}, "
                f"scale={scale:.3f}, L={component_number:3d}: "
                f"DRA={dra[scale_index, component_index]:.5f}"
            )
        print(
            f"  row timing: GMR={gmr_seconds:.2f}s, "
            f"DRA={dra_seconds:.2f}s, total={gmr_seconds + dra_seconds:.2f}s"
        )

        # Persist each completed row; rerunning resumes at the first NaN.
        save_cache(
            cache_path,
            sweep.time_step,
            scales,
            normalized_scales,
            components,
            model_order_percentages,
            mean_nnd,
            dra,
            voxel_res_fraction,
        )

    return (
        normalized_scales,
        model_order_percentages,
        components,
        dra,
        mean_nnd,
        number_of_animals,
    )


def plot_surfaces(
    results: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int],
    ],
    fits: dict[str, dict],
    output_path: Path,
    show: bool,
) -> None:
    figure = plt.figure(figsize=(15, 11))
    surface = None

    for plot_index, (dataset_name, result) in enumerate(results.items(), start=1):
        (
            normalized_scales,
            model_order_percentages,
            components,
            dra,
            mean_nnd,
            number_of_animals,
        ) = result
        actual_order_percentages = 100.0 * components / number_of_animals
        axis = figure.add_subplot(2, 2, plot_index, projection="3d")
        scale_grid, component_grid = np.meshgrid(
            normalized_scales, actual_order_percentages, indexing="ij"
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
        best_fit = fits[dataset_name]["candidates"][
            fits[dataset_name]["best_name"]
        ]
        axis.plot_wireframe(
            scale_grid,
            component_grid,
            best_fit["prediction"],
            color="black",
            linewidth=0.65,
            rstride=1,
            cstride=1,
            label="Fitted surface",
        )
        axis.set_title(
            f"{dataset_name.capitalize()} "
            f"({fits[dataset_name]['best_name']}, "
            f"$R^2$={best_fit['r_squared']:.3f})"
        )
        axis.set_xlabel("Scale / mean NND")
        axis.set_ylabel("Model order / N (%)")
        axis.set_zlabel("DRA")
        axis.set_yticks(actual_order_percentages)
        axis.set_yticklabels(
            [f"{percentage:.1f}\n({component})" for percentage, component in zip(
                actual_order_percentages, components
            )],
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
        "--datasets",
        nargs="+",
        choices=tuple(SWEEPS),
        default=list(SWEEPS),
        help="Datasets to compute (default: all four).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "dra_scale_model_order",
    )
    parser.add_argument("--force", action="store_true", help="Ignore cached data.")
    parser.add_argument(
        "--voxel-res-fraction",
        type=float,
        default=5e-3,
        help="Voxel width as a fraction of the frame's largest spatial span.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200_000,
        help="Voxel batch size (200k is fastest on the tested 8 GB RTX 4060).",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if Path.cwd() != Path(__file__).resolve().parents[1]:
        raise RuntimeError("Run this script from the repository root.")
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA-capable PyTorch setup.")
    if args.voxel_res_fraction <= 0 or args.batch_size <= 0:
        raise ValueError("Voxel resolution fraction and batch size must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for dataset_name in args.datasets:
        results[dataset_name] = compute_surface(
            dataset_name=dataset_name,
            sweep=SWEEPS[dataset_name],
            output_dir=args.output_dir,
            force=args.force,
            voxel_res_fraction=args.voxel_res_fraction,
            batch_size=args.batch_size,
        )

    fits = {}
    for dataset_name, result in results.items():
        normalized_scales, _, components, dra, _, number_of_animals = result
        fits[dataset_name] = fit_dra_surface(
            normalized_scales, components, number_of_animals, dra
        )
        cache_surface_fit(dataset_name, fits[dataset_name], args.output_dir)
        best = fits[dataset_name]["candidates"][fits[dataset_name]["best_name"]]
        print(
            f"[{dataset_name}] fit={fits[dataset_name]['best_name']}, "
            f"RMSE={best['rmse']:.5f}, CV-RMSE={best['cv_rmse']:.5f}, "
            f"R^2={best['r_squared']:.5f}"
        )

    plot_surfaces(
        results,
        fits,
        args.output_dir / "dra_scale_model_order_3d.png",
        show=args.show,
    )


if __name__ == "__main__":
    main()
