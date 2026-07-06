"""
Validate mode_counting on synthetic point clouds with known ground-truth
cluster counts. Measures accuracy as a function of scale and N.

Generates isotropic Gaussian clusters in 3D with controlled separation,
runs the full mode-counting pipeline, and computes error metrics.
"""
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from dfr.analysis import (
    add_managed_output_arguments,
    count_modes,
    create_analysis_artifacts,
)


def generate_clusters(n_clusters, points_per_cluster, cluster_std, separation,
                      dim=3, rng=None):
    """Generate synthetic point cloud with known cluster structure.

    Args:
        n_clusters: number of ground-truth clusters
        points_per_cluster: points in each cluster
        cluster_std: isotropic std of each Gaussian cluster
        separation: distance between adjacent cluster centers (on a line)
        dim: ambient dimension (default 3)
        rng: numpy random generator

    Returns:
        positions: [N, dim] point cloud
        true_cluster_labels: [N] ground-truth cluster IDs
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N = n_clusters * points_per_cluster
    positions = np.zeros((N, dim))
    labels = np.zeros(N, dtype=int)

    # Cluster centers on a line with spacing `separation`
    centers = np.zeros((n_clusters, dim))
    centers[:, 0] = np.arange(n_clusters) * separation

    for c in range(n_clusters):
        start = c * points_per_cluster
        end = start + points_per_cluster
        positions[start:end] = centers[c] + rng.normal(0, cluster_std, (points_per_cluster, dim))
        labels[start:end] = c

    return positions, labels


def test_mode_counting_accuracy(positions, true_n_clusters, scales, tol):
    """Run mode_counting at multiple scales, compute error vs ground truth.

    Returns:
        mode_counts: list of mode counts at each scale
        errors: absolute error |predicted - true| at each scale
    """
    mode_counts = []
    errors = []
    for sc in scales:
        n = count_modes(
            positions, sc, device="cuda", max_iter=400, tolerance=tol
        )
        mode_counts.append(n)
        errors.append(abs(n - true_n_clusters))
    return mode_counts, errors


def run_validation(seed=42):
    """Systematic validation across different cluster configurations."""
    rng = np.random.default_rng(seed)
    n_total_range = [50, 100, 200, 500]
    separation_ratios = [1.0, 2.0, 5.0, 10.0]  # separation / cluster_std
    cluster_std = 1.0

    results = {}

    for n_total in n_total_range:
        for sr in separation_ratios:
            separation = sr * cluster_std
            n_clusters = max(2, n_total // 20)  # ~20 pts per cluster
            pts_per = n_total // n_clusters
            n_total_actual = n_clusters * pts_per

            positions, true_labels = generate_clusters(
                n_clusters, pts_per, cluster_std, separation, rng=rng)

            # Compute avg_nn_dist for scale range
            from scipy.spatial.distance import cdist as scipy_cdist
            d = scipy_cdist(positions, positions)
            np.fill_diagonal(d, 1e10)
            avg_nn_dist = max(float(np.median(np.min(d, axis=1))), 1e-8)
            tol = max(avg_nn_dist * 1e-3, 1e-8)

            # Test at multiple scales around the expected transition
            scales = np.logspace(np.log10(avg_nn_dist * 0.1),
                                 np.log10(avg_nn_dist * 20), 30)

            mode_counts, errors = test_mode_counting_accuracy(
                positions, n_clusters, scales, tol)

            key = (n_total_actual, sr)
            results[key] = {
                "n_total": n_total_actual,
                "n_clusters": n_clusters,
                "separation_ratio": sr,
                "avg_nn_dist": avg_nn_dist,
                "scales": scales,
                "mode_counts": mode_counts,
                "errors": errors,
                "best_scale_idx": int(np.argmin(errors)),
                "min_error": min(errors),
            }

    return results


def plot_validation(results, output_path: Path):
    """Plot validation results: accuracy curves and error heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Left: example accuracy curves ---
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(results)))
    for (key, res), color in zip(sorted(results.items()), colors):
        sc_norm = res["scales"] / res["avg_nn_dist"]
        modes_norm = np.array(res["mode_counts"]) / res["n_clusters"]
        label = f'N={res["n_total"]}, sep={res["separation_ratio"]}x'
        ax.semilogx(sc_norm, modes_norm, color=color, lw=1, alpha=0.7,
                     label=label if res["separation_ratio"] == 2.0 else "")
    ax.axhline(1.0, color='k', ls='--', lw=0.5)
    ax.set_xlabel("scale / avg_nn_dist")
    ax.set_ylabel("predicted / true clusters")
    ax.set_title("Mode counting accuracy vs scale")
    ax.legend(fontsize=6, frameon=False)

    # --- Right: minimum error vs separation ---
    ax = axes[1]
    sep_values = sorted(set(k[1] for k in results))
    n_values = sorted(set(k[0] for k in results))
    error_grid = np.zeros((len(n_values), len(sep_values)))
    for i, n in enumerate(n_values):
        for j, sr in enumerate(sep_values):
            key = (n, sr)
            if key in results:
                error_grid[i, j] = results[key]["min_error"] / results[key]["n_clusters"]

    im = ax.imshow(error_grid, aspect='auto', origin='lower',
                    extent=[sep_values[0]-0.5, sep_values[-1]+0.5,
                            n_values[0]-25, n_values[-1]+25],
                    cmap='RdYlGn_r', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="relative error")
    ax.set_xlabel("separation / cluster_std")
    ax.set_ylabel("N (total points)")
    ax.set_title("Minimum mode-count error\n(fraction of true clusters)")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.show()
    print(f"  -> Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    add_managed_output_arguments(parser)
    args = parser.parse_args()
    if args.no_display:
        plt.show = lambda: None
    artifacts = create_analysis_artifacts(
        args,
        name="validate mode counting",
        resolved_config={"analysis": "validate_mode_counting", "seed": args.seed},
        entrypoint="experiments.validate_mode_counting",
    )
    print("=" * 60)
    print("  Validating mode_counting on synthetic data")
    print("=" * 60)

    results = run_validation(seed=args.seed)

    # Summary table
    print(f"\n  {'N':>5} {'sep':>5} {'n_true':>7} {'best_scale/nn':>14} {'min_err':>8}")
    print("  " + "-" * 45)
    for key in sorted(results.keys()):
        res = results[key]
        best_sc = res["scales"][res["best_scale_idx"]] / res["avg_nn_dist"]
        print(f"  {res['n_total']:>5} {res['separation_ratio']:>5.1f}x "
              f"{res['n_clusters']:>7} {best_sc:>14.2f} {res['min_error']:>8}")

    # Overall stats
    all_errors = [r["min_error"] / r["n_clusters"] for r in results.values()]
    print(f"\n  Mean relative error: {np.mean(all_errors):.3f}")
    print(f"  Perfect recoveries: {sum(1 for e in all_errors if e == 0)}/{len(all_errors)}")

    records = []
    for result in results.values():
        records.append(
            {
                key: value
                for key, value in result.items()
                if key not in {"scales", "mode_counts", "errors"}
            }
        )
    artifacts.save_json(
        "validation_summary.json",
        {"records": records, "mean_relative_error": float(np.mean(all_errors))},
        category="metrics",
        overwrite=args.resume,
    )
    plot_validation(results, artifacts.figures_dir / "validate_mode_counting.png")
    print(f"  Outputs: {artifacts.run_dir}")


if __name__ == "__main__":
    main()
