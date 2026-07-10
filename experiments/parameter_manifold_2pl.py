"""
Parameter Manifold with the 2PL model (symmetric sigmoid, gamma=1).

The 2PL fixes log10_gamma=0, eliminating the k-lg degeneracy that
inflated k variance in the 3PL. Model: m(sigma) = 1 + (N-1)/(1 + (sigma/sigma_half)^k).

This is the standard Hill equation with Hill coefficient k.
"""
import sys, os
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dfr.utils import move_figure
from dfr.analysis import (
    add_managed_output_arguments,
    create_analysis_artifacts,
    fit_symmetric_2pl_curves,
    symmetric_2pl_mode_count,
)
from dfr.plotting import apply_academic_style, apply_figure_layout, save_figure


# ======================================================================
# 0. Model & data loading (reuses existing caches)
# ======================================================================

model_2pl = symmetric_2pl_mode_count  # compatibility for research imports


DATASET_RUNS = [
    {"name": "swift",    "start_step": 0,    "end_step": None, "step_length": 20},
    {"name": "starling", "start_step": 0,    "end_step": None, "step_length": 1},
    {"name": "jackdaw",  "start_step": 350,  "end_step": 550,  "step_length": 1},
    {"name": "jackdaw2", "start_step": 2700, "end_step": 3460, "step_length": 5},
]

DATASET_COLORS = {
    "starling": "#2c3e50", "jackdaw": "#e67e22",
    "jackdaw2": "#27ae60", "swift": "#8e44ad",
}

SYN_COLORS = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6',
              '#e67e22', '#1abc9c', '#9b59b6', '#34495e', '#95a5a6',
              '#d35400', '#c0392b']
_FIGURE_DIR: Path | None = None


def _figure_path(filename: str) -> Path:
    """Return the managed figure path configured by the CLI."""
    if _FIGURE_DIR is None:
        raise RuntimeError(
            "Run parameter_manifold_2pl through its managed CLI before calling "
            "a figure-producing helper."
        )
    return _FIGURE_DIR / filename


def _save_figure(filename: str) -> Path:
    target = _figure_path(filename)
    return save_figure(plt.gcf(), target, bbox_inches="tight", dpi=300)


def _finish_layout() -> None:
    apply_figure_layout(plt.gcf())


def set_style():
    apply_academic_style(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


# ======================================================================
# 1. Fit 2PL to all cached data
# ======================================================================

def fit_2pl_all(project_root=None):
    """Load cached mode-count data, fit 2PL, return per-frame parameters."""
    all_params_list, all_N_list, all_names_list = [], [], []
    raw_data = {}

    # Import load_cached_data from the 3PL version (reuses modes.npy + scale_range.npy)
    from experiments.parameter_manifold import load_cached_data

    for rp in DATASET_RUNS:
        name = rp["name"]
        print(f"\n{'='*60}\nDataset: {name}\n{'='*60}")
        sr, Na, scr, am, _ = load_cached_data(rp, project_root=project_root)
        if sr is None: continue

        fitted = fit_symmetric_2pl_curves(
            sr, Na, scr, am, dataset_name=name
        )
        params_2pl = fitted.aligned_parameters
        valid = fitted.success
        pv = fitted.result.parameters
        n_ok = valid.sum()
        print(f"  OK: {n_ok}/{len(sr)}")

        raw_data[name] = {"step_range": sr, "N_array": Na, "params": params_2pl,
                          "valid": valid, "nn_dists": _}

        if n_ok > 0:
            all_params_list.append(pv)
            all_N_list.append(Na[valid])
            all_names_list.append(np.array([name] * n_ok))

    # Load nn_dists from cache
    for name in raw_data:
        root = Path(project_root or Path.cwd()).expanduser().resolve()
        nnp = root / "scenarios" / name / "nn_dists.npy"
        if os.path.exists(nnp):
            raw_data[name]["nn_dists"] = np.load(nnp)

    return (np.vstack(all_params_list), np.concatenate(all_N_list),
            np.concatenate(all_names_list), raw_data)


# ======================================================================
# 2. Plotting
# ======================================================================

def plot_manifold_2pl(all_params, all_N, all_names):
    """k vs sigma_half colored by species and N."""
    k = all_params[:, 0]
    sh = all_params[:, 1]
    datasets = sorted(set(all_names))

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    move_figure(fig, 100, 100)

    for ds in datasets:
        m = all_names == ds
        axes[0].scatter(k[m], sh[m], c=DATASET_COLORS[ds], label=ds,
                        s=8, alpha=0.5, edgecolors="none")
    axes[0].set_xlabel("k (Hill coefficient)"); axes[0].set_ylabel("sigma_half")
    axes[0].set_title("2PL manifold: k vs sigma_half")
    axes[0].legend(frameon=False, fontsize=8)

    sc = axes[1].scatter(k, sh, c=all_N, cmap="plasma", s=8, alpha=0.5, edgecolors="none")
    axes[1].set_xlabel("k (Hill coefficient)"); axes[1].set_ylabel("sigma_half")
    axes[1].set_title("2PL manifold colored by N")
    plt.colorbar(sc, ax=axes[1], label="N")

    # Console stats
    print(f"\n  2PL parameters (k, sigma_half):")
    for ds in datasets:
        m = all_names == ds
        print(f"    {ds:<12} k={np.mean(k[m]):.2f}+-{np.std(k[m]):.2f}  "
              f"sh={np.mean(sh[m]):.3f}+-{np.std(sh[m]):.3f}  "
              f"k_CV={np.std(k[m])/np.mean(k[m]):.3f}")

    _finish_layout()
    _save_figure("manifold_2pl.png")
    plt.show()
    print(f"  -> Saved {_figure_path('manifold_2pl.png')}")


def plot_synthetic_overlay(all_params, all_names, cache_dir=None):
    """Overlay synthetic 2PL fits on the empirical manifold."""
    from experiments.synthetic_benchmark import load_cached
    syn_params_list, syn_labels = load_cached(cache_dir)
    if syn_params_list is None:
        print("  [SKIP] No synthetic cache found")
        return

    # Refit synthetic data with 2PL
    syn_2pl = []
    for params_3pl, label in zip(syn_params_list, syn_labels):
        k2 = []
        sh2 = []
        for p in params_3pl:
            # Convert 3PL params to approximate 2PL params
            # For gamma>0, the 2PL gives similar shape but slightly different k
            if p[2] > -1.5 and p[2] < 1.5:  # only for identifiable lg
                k2.append(p[0])
                sh2.append(p[1])
        if len(k2) >= 5:
            syn_2pl.append((np.array(k2), np.array(sh2), label))

    if not syn_2pl:
        return

    set_style()
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    k = all_params[:, 0]; sh = all_params[:, 1]
    for ds in sorted(set(all_names)):
        m = all_names == ds
        ax.scatter(k[m], sh[m], c=DATASET_COLORS[ds], label=ds,
                   s=8, alpha=0.4, edgecolors="none")

    syn_markers = ['^', 's', 'D', 'v', 'P', 'X', '*', 'p', 'h', 'H', 'd', '<']
    for i, (k2, sh2, label) in enumerate(syn_2pl):
        ax.scatter(k2, sh2, c=SYN_COLORS[i % len(SYN_COLORS)],
                   marker=syn_markers[i % len(syn_markers)], s=60,
                   edgecolors='black', linewidths=0.5,
                   label=f'{label} (n={len(k2)})', zorder=5)

    ax.set_xlabel("k (Hill coefficient)"); ax.set_ylabel("sigma_half")
    ax.set_title("2PL: empirical flocks + synthetic processes")
    ax.legend(fontsize=6, frameon=False, ncol=2)

    _finish_layout()
    _save_figure("manifold_2pl_synthetic.png")
    plt.show()
    print(f"  -> Saved {_figure_path('manifold_2pl_synthetic.png')}")


def plot_temporal_2pl(raw_data):
    """Temporal trajectories of k(t) and sigma_half(t)."""
    datasets_with_time = [name for name, rd in raw_data.items()
                          if len(rd["step_range"]) > 5 and rd["valid"].sum() > 5]
    n_ds = len(datasets_with_time)
    if n_ds == 0: return

    set_style()
    fig, axes = plt.subplots(n_ds, 3, figsize=(20, 3.5 * n_ds), squeeze=False)

    for row, name in enumerate(datasets_with_time):
        rd = raw_data[name]
        sr = np.array(rd["step_range"])
        valid = rd["valid"]
        t = sr[valid]
        p = rd["params"][valid]
        k, sh = p[:, 0], p[:, 1]
        dt = np.median(np.diff(t))

        # k(t)
        ax = axes[row, 0]
        ax.plot(t, k, "o-", color=DATASET_COLORS[name], markersize=3, lw=0.8, alpha=0.7)
        ax.set_ylabel("k"); ax.set_xlabel("Frame")
        ax.set_title(f"{name}: k(t) — 2PL, dt={dt:.0f}")

        # sigma_half(t)
        ax = axes[row, 1]
        ax.plot(t, sh, "o-", color=DATASET_COLORS[name], markersize=3, lw=0.8, alpha=0.7)
        ax.set_ylabel("sigma_half"); ax.set_xlabel("Frame")
        ax.set_title(f"{name}: sigma_half(t)")

        # ACF of k(t)
        ax = axes[row, 2]
        k_acf = k - np.mean(k)
        maxlag = min(100, len(k) // 4)
        acf = [np.corrcoef(k_acf[:-l], k_acf[l:])[0, 1] for l in range(1, maxlag)]
        lags_fr = np.arange(1, len(acf) + 1) * dt
        ax.plot(lags_fr, acf, "o-", color=DATASET_COLORS[name], markersize=2, lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        zc = next((i for i, a in enumerate(acf) if a < 0), len(acf))
        tau = (zc + 1) * dt if zc < len(acf) else None
        if tau:
            ax.set_title(f"{name}: k ACF (tau0={tau:.0f} fr)")
        else:
            ax.set_title(f"{name}: k ACF (no zero-cross)")
        ax.set_xlabel("Lag (frames)")

        k_cv = np.std(k) / np.mean(k)
        sh_cv = np.std(sh) / np.mean(sh)
        tau_str = f"tau0={tau:.0f}fr" if tau else "N/A"
        dk_med = np.median(np.abs(np.diff(k)))
        print(f"  {name}: k={np.mean(k):.2f}+-{np.std(k):.2f} (CV={k_cv:.3f}, dk_med={dk_med:.2f}), "
              f"sh={np.mean(sh):.3f} (CV={sh_cv:.4f}), {tau_str}")

    _finish_layout()
    _save_figure("manifold_2pl_temporal.png")
    plt.show()
    print(f"  -> Saved {_figure_path('manifold_2pl_temporal.png')}")


def plot_N_dependence_2pl(all_params, all_N, all_names):
    """k vs N and sigma_half vs N."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    datasets = sorted(set(all_names))

    for ax, yv, yl in [(axes[0], all_params[:, 0], "k"), (axes[1], all_params[:, 1], "sigma_half")]:
        for ds in datasets:
            m = all_names == ds
            ax.scatter(all_N[m], yv[m], c=DATASET_COLORS[ds], label=ds, s=10, alpha=0.6)
        ax.set_xlabel("N"); ax.set_ylabel(yl); ax.set_title(f"{yl} vs N (2PL)")
        ax.legend(fontsize=7)

    _finish_layout()
    _save_figure("manifold_2pl_N_dependence.png")
    plt.show()
    print(f"  -> Saved {_figure_path('manifold_2pl_N_dependence.png')}")

    print("\n  N-dependence (2PL, Pearson r):")
    for ds in datasets:
        m = all_names == ds
        r_k = np.corrcoef(all_N[m], all_params[m, 0])[0, 1]
        r_sh = np.corrcoef(all_N[m], all_params[m, 1])[0, 1]
        print(f"    {ds:<12} r(k,N)={r_k:+.3f}   r(sigma_half,N)={r_sh:+.3f}")


# ======================================================================
# 3. Main
# ======================================================================

def main():
    global _FIGURE_DIR
    import argparse
    p = argparse.ArgumentParser(description="Fit and plot the symmetric 2PL manifold")
    p.add_argument("--no-display", action="store_true")
    add_managed_output_arguments(p)
    args = p.parse_args()
    artifacts = create_analysis_artifacts(
        args,
        name="parameter manifold 2PL",
        resolved_config={"analysis": "parameter_manifold_2pl", "datasets": DATASET_RUNS},
        entrypoint="experiments.parameter_manifold_2pl",
    )
    _FIGURE_DIR = artifacts.figures_dir
    if args.no_display:
        plt.show = lambda: None
        print("[--no-display]\n")

    print("Parameter Manifold — 2PL model (symmetric sigmoid, gamma=1)")
    print("=" * 60)

    all_params, all_N, all_names, raw_data = fit_2pl_all(args.project_root)
    k, sh = all_params[:, 0], all_params[:, 1]
    artifacts.save_npz(
        "manifold_2pl_fits.npz",
        overwrite=args.resume,
        parameters=all_params,
        number_of_agents=all_N,
        dataset_names=all_names,
    )

    print(f"\n{'='*60}")
    print(f"Total: {len(all_params)} fits, {len(set(all_names))} species")
    print(f"  k:              [{k.min():.2f}, {k.max():.2f}] median={np.median(k):.2f}")
    print(f"  sigma_half:     [{sh.min():.3f}, {sh.max():.3f}] median={np.median(sh):.3f}")

    # Figures
    plot_manifold_2pl(all_params, all_N, all_names)
    plot_synthetic_overlay(
        all_params,
        all_names,
        args.project_root / "scenarios" / "_synthetic",
    )
    plot_temporal_2pl(raw_data)
    plot_N_dependence_2pl(all_params, all_N, all_names)

    # sigma_half vs nn_dist
    print(f"\n{'='*60}")
    print("sigma_half vs physical nearest-neighbor distance (2PL):")
    nn_by_species = {name: raw_data[name]["nn_dists"] for name in raw_data
                     if raw_data[name].get("nn_dists") is not None}
    if nn_by_species:
        print(f"  {'Species':<12} {'mean_sh':>8} {'mean_nn':>8} {'ratio':>8}")
        for ds in sorted(nn_by_species.keys()):
            m = all_names == ds
            sh_m = all_params[m, 1]
            nn = nn_by_species[ds][:len(sh_m)]
            ratio = sh_m / (nn + 1e-10)
            print(f"  {ds:<12} {np.mean(sh_m):>8.3f} {np.mean(nn):>8.3f} {np.mean(ratio):>8.3f}")

    artifacts.save_json(
        "summary.json",
        {"fit_count": len(all_params), "datasets": sorted(set(all_names))},
        category="metrics",
        overwrite=args.resume,
    )
    print(f"\nDone. Outputs: {artifacts.run_dir}")


if __name__ == "__main__":
    main()
