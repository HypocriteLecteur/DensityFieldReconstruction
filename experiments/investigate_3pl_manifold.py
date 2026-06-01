"""
Investigate the 3PL parameter manifold across biological swarm datasets.

For each dataset x time_step, fits the 3-parameter logistic model:
    m(sigma) = 1 + (N-1) / (1 + (sigma/x0)^k)^gamma

Then applies manifold learning (PCA -> t-SNE -> UMAP) to discover whether
the 3PL parameters (k, x0, gamma) lie on a lower-dimensional manifold,
and clusters them to identify distinct spatial configuration modes.

Also compares 3PL against 2PL (gamma=1) and a constrained 3PL where
gamma is a quadratic function of log10(x0) to avoid the ceiling problem.

Datasets: starling, jackdaw, jackdaw2, swift
"""

import sys
import os

sys.path.append(os.getcwd())

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from tqdm import tqdm
from collections import defaultdict

from dfr.simulation_config import SimulationConfig
from dfr.dataset_io import DatasetFactory
from experiments.power_law import power_3pl, move_figure


# ======================================================================
# 0. Model functions
# ======================================================================

def model_2pl(x, k, x0, A, D):
    """2-parameter logistic: symmetric on log-x scale (gamma=1 3PL).

    Computed in log-space to avoid overflow: (x/x0)^k = exp(k * log(x/x0)).
    """
    log_ratio = np.clip(k * np.log(x / x0), -500, 500)
    return (A - D) / (1 + np.exp(log_ratio)) + D


def model_3pl_constrained(x, k, x0, A, D, a, b, c):
    """3PL with gamma = exp(a * log10(x0)^2 + b * log10(x0) + c).

    Gamma is no longer a free parameter -- it's deterministically
    computed from x0 via a quadratic in log-space.  The exponential
    link keeps gamma > 0 naturally (no bound needed).

    a, b, c are global or per-dataset (shared across frames in that group).
    k, x0 are per-frame.
    """
    log_x0 = np.log10(np.maximum(x0, 1e-12))
    log_gamma = np.clip(a * log_x0**2 + b * log_x0 + c, -10, 10)
    gamma = np.exp(log_gamma)
    return (A - D) / (1 + (x / x0)**k)**gamma + D


MODEL_REGISTRY = {
    "3pl": {"fn": lambda x, k, x0, gamma, A, D: power_3pl(x, k, x0, gamma, A=A, D=D),
            "n_params": 3, "p0": lambda med: [2.0, med, 1.0],
            "bounds": ([0.1, 1e-6, 0.1], [20.0, np.inf, np.inf])},
    "2pl": {"fn": lambda x, k, x0, A, D: model_2pl(x, k, x0, A=A, D=D),
            "n_params": 2, "p0": lambda med: [2.0, med],
            "bounds": ([0.1, 1e-6], [20.0, np.inf])},
    "3plc": {"fn": lambda x, k, x0, A, D, a, b, c: model_3pl_constrained(x, k, x0, A=A, D=D, a=a, b=b, c=c),
             "n_params": 2, "p0": lambda med: [2.0, med],
             "bounds": ([0.1, 1e-6], [20.0, np.inf]),
             "is_constrained": True},
}


# ======================================================================
# 1. Load cached mode count data
# ======================================================================

DATASET_RUNS = [
    {"name": "swift",    "start_step": 0,    "end_step": None, "step_length": 200},
    {"name": "starling", "start_step": 0,    "end_step": None, "step_length": 1},
    {"name": "jackdaw",  "start_step": 350,  "end_step": 550,  "step_length": 10},
    {"name": "jackdaw2", "start_step": 2700, "end_step": 3460, "step_length": 20},
]


def load_cached_data(run_params):
    """Load cached modes.npy and scale_range.npy for a dataset.

    Returns:
        step_range: list of actual time step indices
        N_array:    shape (num_steps,) -- number of agents per step
        scale_range: shape (num_steps, 2) -- [s_start, s_end] per step
        all_modes:  shape (num_steps, 40) -- mode count at 40 log-spaced scales
    """
    name = run_params["name"]
    scenario_path = os.path.join(os.getcwd(), "scenarios", name)

    modes_path = os.path.join(scenario_path, "modes.npy")
    scale_range_path = os.path.join(scenario_path, "scale_range.npy")

    if not os.path.exists(modes_path) or not os.path.exists(scale_range_path):
        print(f"  [WARN] No cache found for '{name}' -- run reconstruction_scale_determination.py first")
        return None, None, None, None

    all_modes = np.load(modes_path)          # (num_steps, 40)
    scale_range = np.load(scale_range_path)  # (num_steps, 2)

    # Determine which time steps were used
    config_path = os.path.join(scenario_path, "config.yaml")
    config = SimulationConfig(config_path)
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    max_steps = dataset.trajectories.shape[0]
    end_step = run_params["end_step"]
    effective_end = end_step if end_step is not None and end_step <= max_steps else max_steps
    step_range = list(range(run_params["start_step"], effective_end, run_params["step_length"]))

    # Align all arrays to the same length
    n_cached_modes = all_modes.shape[0]
    n_cached_range = scale_range.shape[0]
    n_steps_expected = len(step_range)

    n_effective = min(n_cached_modes, n_cached_range, n_steps_expected)

    if n_cached_modes != n_cached_range or n_cached_range != n_steps_expected:
        print(f"  [INFO] Aligning: modes={n_cached_modes}, range={n_cached_range}, "
              f"steps_expected={n_steps_expected} -> using n={n_effective}")

    step_range = step_range[:n_effective]
    scale_range = scale_range[:n_effective]
    all_modes = all_modes[:n_effective]

    # Compute N for each step
    print(f"  Loading N for {name} ({len(step_range)} steps)...")
    N_array = np.array([dataset.positions_at_time_step(s).shape[0] for s in tqdm(step_range)])

    return step_range, N_array, scale_range, all_modes


# ======================================================================
# 2. Fit 3PL and 2PL per time step
# ======================================================================

def fit_all_models_all_steps(step_range, N_array, scale_range, all_modes,
                             model_names=("3pl", "2pl"),
                             num_test_scale=40):
    """Fit every model for every time step.

    Returns:
        comparison: dict of model_name -> {
            "params":   list of arrays (variable-length per model),
            "fitted":   list of arrays,
            "resid_var": np.array (n_steps,),
            "success":  np.array (n_steps,) bool,
            "n_params": int,
        }
        plus per-step aligned arrays for convenience.
    """
    n_steps = all_modes.shape[0]

    # Initialize per-model storage
    comp = {}
    for mname in model_names:
        n_p = MODEL_REGISTRY[mname]["n_params"]
        comp[mname] = {
            "params":    [None] * n_steps,
            "fitted":    [None] * n_steps,
            "resid_var": np.full(n_steps, np.nan),
            "success":   np.zeros(n_steps, dtype=bool),
            "n_params":  n_p,
        }
    # Track very large gamma (diagnostic only, no ceiling anymore)
    hit_large_gamma = np.zeros(n_steps, dtype=bool)

    for i in range(n_steps):
        N = N_array[i]
        s_start, s_end = scale_range[i]
        test_scales = np.logspace(np.log10(s_start), np.log10(s_end), num_test_scale)
        modes = all_modes[i]

        # Trim saturated region (same for all models)
        begin_idx = np.argmax(modes <= 0.9 * N)
        if begin_idx == 0:
            begin_idx = max(1, np.argmax(modes <= 0.99 * N))

        x_data = test_scales[begin_idx:]
        y_data = modes[begin_idx:]

        if len(x_data) < 5:
            continue

        median_scale = np.median(test_scales)

        for mname in model_names:
            reg = MODEL_REGISTRY[mname]
            try:
                popt, _ = curve_fit(
                    lambda x, *p: reg["fn"](x, *p, A=N, D=1),
                    x_data,
                    y_data,
                    p0=reg["p0"](median_scale),
                    sigma=np.maximum(y_data, 1),
                    absolute_sigma=True,
                    bounds=reg["bounds"],
                    maxfev=5000,
                )
                fitted = reg["fn"](test_scales, *popt, A=N, D=1)
                comp[mname]["params"][i] = popt
                comp[mname]["fitted"][i] = fitted
                comp[mname]["resid_var"][i] = np.mean((y_data - reg["fn"](x_data, *popt, A=N, D=1))**2)
                comp[mname]["success"][i] = True
            except (RuntimeError, ValueError):
                continue

        # Track very large gamma (diagnostic only)
        if comp["3pl"]["success"][i]:
            gamma_val = comp["3pl"]["params"][i][2]
            if gamma_val >= 200.0:
                hit_large_gamma[i] = True

    comp["_hit_ceiling"] = hit_large_gamma
    comp["_step_range"] = step_range
    comp["_N_array"] = N_array
    return comp


# ======================================================================
# 2b. Gamma-x0 quadratic relationship & constrained 3PL refit
# ======================================================================

def fit_gamma_x0_quadratic(comparison):
    """Fit gamma = exp(a * log10(x0)^2 + b * log10(x0) + c) to all successful 3PL fits.

    With gamma unconstrained (no ceiling), all converged points are used.

    Returns:
        (a, b, c): global parameters
        r2: R^2 of the quadratic fit in log-gamma space
        data: dict with log_x0, log_gamma arrays used for the fit
    """
    log_x0_vals = []
    log_gamma_vals = []

    n_steps = len(comparison["_step_range"])
    for i in range(n_steps):
        if not comparison["3pl"]["success"][i]:
            continue
        gamma = comparison["3pl"]["params"][i][2]
        x0 = comparison["3pl"]["params"][i][1]
        log_x0_vals.append(np.log10(x0))
        log_gamma_vals.append(np.log(gamma))

    if len(log_x0_vals) < 10:
        print("  [WARN] Too few fits for quadratic regression")
        return None, None, None

    log_x0_arr = np.array(log_x0_vals)
    log_gamma_arr = np.array(log_gamma_vals)

    # Quadratic fit: log(gamma) = a * log10(x0)^2 + b * log10(x0) + c
    X = np.column_stack([log_x0_arr**2, log_x0_arr, np.ones_like(log_x0_arr)])
    coeffs, residuals, rank, singular = np.linalg.lstsq(X, log_gamma_arr, rcond=None)

    a, b, c = coeffs[0], coeffs[1], coeffs[2]

    # R^2
    predicted = X @ coeffs
    ss_res = np.sum((log_gamma_arr - predicted)**2)
    ss_tot = np.sum((log_gamma_arr - np.mean(log_gamma_arr))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    gamma_vals = np.exp(log_gamma_arr)
    print(f"\n  Gamma-x0 quadratic fit (n={len(log_x0_arr)} points, gamma unconstrained):")
    print(f"    log(gamma) = {a:.4f} * log10(x0)^2  +  {b:.4f} * log10(x0)  +  {c:.4f}")
    print(f"    R^2 = {r2:.4f}")
    print(f"    gamma range: [{np.min(gamma_vals):.1f}, {np.max(gamma_vals):.1f}]")
    print(f"    x0 range:   [{np.min(10**log_x0_arr):.4f}, {np.max(10**log_x0_arr):.4f}]")

    return (a, b, c), r2, {"log_x0": log_x0_arr, "log_gamma": log_gamma_arr}


def fit_constrained_refit(step_range, N_array, scale_range, all_modes,
                          global_a, global_b, global_c, num_test_scale=40):
    """Refit each frame using the constrained 3PL where gamma = f(x0; a,b,c).

    Only (k, x0) are optimized per frame; gamma is computed from x0 via the
    quadratic.  This is a genuine 2-parameter model -- gamma cannot
    hit a ceiling because it's deterministically linked to x0.

    Returns:
        params: np.array (n_steps, 2) -- [k, x0]
        fitted_curves: list of arrays
    """
    n_steps = all_modes.shape[0]
    params = np.full((n_steps, 2), np.nan)
    fitted_curves = [None] * n_steps
    resid_var = np.full(n_steps, np.nan)
    success = np.zeros(n_steps, dtype=bool)

    for i in range(n_steps):
        N = N_array[i]
        s_start, s_end = scale_range[i]
        test_scales = np.logspace(np.log10(s_start), np.log10(s_end), num_test_scale)
        modes = all_modes[i]

        begin_idx = np.argmax(modes <= 0.9 * N)
        if begin_idx == 0:
            begin_idx = max(1, np.argmax(modes <= 0.99 * N))

        x_data = test_scales[begin_idx:]
        y_data = modes[begin_idx:]

        if len(x_data) < 5:
            continue

        median_scale = np.median(test_scales)
        reg = MODEL_REGISTRY["3plc"]

        try:
            popt, _ = curve_fit(
                lambda x, k, x0: reg["fn"](x, k, x0, A=N, D=1,
                                            a=global_a, b=global_b, c=global_c),
                x_data,
                y_data,
                p0=reg["p0"](median_scale),
                sigma=np.maximum(y_data, 1),
                absolute_sigma=True,
                bounds=reg["bounds"],
                maxfev=5000,
            )
            k_fit, x0_fit = popt
            fitted = reg["fn"](test_scales, k_fit, x0_fit, A=N, D=1,
                               a=global_a, b=global_b, c=global_c)
            params[i] = popt
            fitted_curves[i] = fitted
            resid_var[i] = np.mean((y_data - reg["fn"](x_data, k_fit, x0_fit, A=N, D=1,
                                                       a=global_a, b=global_b, c=global_c))**2)
            success[i] = True
        except (RuntimeError, ValueError):
            continue

    return {
        "params": [p.tolist() if p is not None else None for p in params],
        "fitted": fitted_curves,
        "resid_var": resid_var,
        "success": success,
        "n_params": 2,
    }


def print_per_dataset_quadratic(comparison, all_names):
    """Fit gamma-x0 quadratic separately per dataset and compare coefficients.

    This reveals whether the gamma-x0 relationship is universal or
    dataset-specific -- key to understanding why a single global quadratic
    may underperform on some datasets.
    """
    print(f"\n  -- Per-dataset gamma-x0 quadratic fits --")
    print(f"  {'Dataset':<12} {'n':<8} {'a':<10} {'b':<10} {'c':<10} {'R^2':<8}")
    print(f"  {'-'*55}")

    per_ds_coeffs = {}
    for ds in sorted(set(all_names)):
        mask_ds = np.array(all_names) == ds
        log_x0_vals, log_gamma_vals = [], []
        n_ceil_ds = 0
        n_steps = len(comparison["_step_range"])
        for i in range(n_steps):
            if not comparison["3pl"]["success"][i]:
                continue
            if not mask_ds[i]:
                continue
            gamma = comparison["3pl"]["params"][i][2]
            if gamma >= 200.0:
                n_ceil_ds += 1
                continue
            x0 = comparison["3pl"]["params"][i][1]
            log_x0_vals.append(np.log10(x0))
            log_gamma_vals.append(np.log(gamma))

        if len(log_x0_vals) < 5:
            print(f"  {ds:<12} {len(log_x0_vals):<8} (too few points, {n_ceil_ds} with gamma>=200)")
            continue

        log_x0_arr = np.array(log_x0_vals)
        log_gamma_arr = np.array(log_gamma_vals)
        X = np.column_stack([log_x0_arr**2, log_x0_arr, np.ones_like(log_x0_arr)])
        coeffs, _, _, _ = np.linalg.lstsq(X, log_gamma_arr, rcond=None)
        a_ds, b_ds, c_ds = coeffs[0], coeffs[1], coeffs[2]
        predicted = X @ coeffs
        ss_res = np.sum((log_gamma_arr - predicted)**2)
        ss_tot = np.sum((log_gamma_arr - np.mean(log_gamma_arr))**2)
        r2_ds = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        per_ds_coeffs[ds] = (a_ds, b_ds, c_ds, r2_ds)
        print(f"  {ds:<12} {len(log_x0_arr):<8} {a_ds:<10.4f} {b_ds:<10.4f} {c_ds:<10.4f} {r2_ds:<8.4f}")

    print(f"\n  Interpretation:")
    print(f"  - If per-dataset (a,b,c) differ substantially, the gamma-x0 relationship")
    print(f"    is NOT universal. A single global quadratic will fit some datasets poorly.")
    print(f"  - If one dataset has much lower R^2, its gamma is less predictable from x0 alone.")

    return per_ds_coeffs


def plot_gamma_x0_relationship(comparison, all_names, quad_fit):
    """Figure: gamma vs log10(x0) scatter, with quadratic fit and predicted values."""
    if quad_fit is None or quad_fit[0] is None:
        print("  Skipping gamma-x0 plot (no quadratic fit available).")
        return

    (a, b, c), r2, fit_data = quad_fit
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    move_figure(fig, 100, 100)

    # Collect all points
    log_x0_all, gamma_all, is_large = [], [], []
    n_steps = len(comparison["_step_range"])
    for i in range(n_steps):
        if not comparison["3pl"]["success"][i]:
            continue
        gamma = comparison["3pl"]["params"][i][2]
        x0 = comparison["3pl"]["params"][i][1]
        log_x0_all.append(np.log10(x0))
        gamma_all.append(gamma)
        is_large.append(gamma >= 200.0)

    log_x0_all = np.array(log_x0_all)
    gamma_all = np.array(gamma_all)
    is_large = np.array(is_large)

    # Quadratic fit curve
    x0_grid = np.logspace(log_x0_all.min() - 0.5, log_x0_all.max() + 0.5, 200)
    log_x0_grid = np.log10(x0_grid)
    pred_gamma_grid = np.exp(a * log_x0_grid**2 + b * log_x0_grid + c)

    # -- Left: scatter with quadratic fit --
    ax = axes[0]
    ax.scatter(log_x0_all[~is_large], gamma_all[~is_large],
               c="#3498db", s=20, alpha=0.5, label=f"gamma<200 (n={sum(~is_large)})",
               edgecolors="none")
    ax.scatter(log_x0_all[is_large], gamma_all[is_large],
               c="#e74c3c", s=25, alpha=0.8, marker="^", label=f"gamma>=200 (n={sum(is_large)})",
               edgecolors="black", linewidths=0.5)
    ax.plot(log_x0_grid, pred_gamma_grid, "k-", lw=2, label=f"Quadratic fit (R^2={r2:.3f})")
    ax.set_xlabel("log10(x0)")
    ax.set_ylabel("gamma")
    ax.set_title("gamma vs log10(x0)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_yscale("log")

    # -- Middle: same but colored by dataset --
    ax = axes[1]
    ds_data = {ds: {"lx": [], "g": [], "c": []} for ds in sorted(set(all_names))}
    for i in range(n_steps):
        if not comparison["3pl"]["success"][i]:
            continue
        ds = all_names[i] if i < len(all_names) else "unknown"
        if ds not in ds_data:
            ds_data[ds] = {"lx": [], "g": [], "c": []}
        gamma = comparison["3pl"]["params"][i][2]
        x0 = comparison["3pl"]["params"][i][1]
        ds_data[ds]["lx"].append(np.log10(x0))
        ds_data[ds]["g"].append(gamma)
        ds_data[ds]["c"].append(gamma >= 200.0)

    for ds in sorted(ds_data.keys()):
        lx = np.array(ds_data[ds]["lx"])
        g = np.array(ds_data[ds]["g"])
        large_mask = np.array(ds_data[ds]["c"])
        if len(lx) == 0:
            continue
        ax.scatter(lx[~large_mask], g[~large_mask], c=DATASET_COLORS.get(ds, "#888"),
                   label=f"{ds} (n={sum(~large_mask)})", s=15, alpha=0.6, edgecolors="none")
        if sum(large_mask) > 0:
            ax.scatter(lx[large_mask], g[large_mask], c=DATASET_COLORS.get(ds, "#888"),
                       s=25, alpha=0.8, marker="^", edgecolors="black", linewidths=0.5)
    ax.plot(log_x0_grid, pred_gamma_grid, "k-", lw=2)
    ax.set_xlabel("log10(x0)")
    ax.set_ylabel("gamma")
    ax.set_title("gamma vs log10(x0) by dataset")
    ax.legend(frameon=False, fontsize=7)
    ax.set_yscale("log")

    # -- Right: predicted vs actual gamma --
    ax = axes[2]
    pred_gamma = np.exp(a * log_x0_all**2 + b * log_x0_all + c)
    ax.scatter(gamma_all[~is_large], pred_gamma[~is_large],
               c="#3498db", s=15, alpha=0.5, edgecolors="none",
               label=f"gamma<200 (n={sum(~is_large)})")
    ax.scatter(gamma_all[is_large], pred_gamma[is_large],
               c="#e74c3c", s=25, alpha=0.8, marker="^", edgecolors="black", linewidths=0.5,
               label=f"gamma>=200 (n={sum(is_large)})")
    lo = min(gamma_all.min(), pred_gamma.min())
    hi = max(gamma_all[~is_large].max(), pred_gamma.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("Actual gamma (free 3PL fit)")
    ax.set_ylabel("Predicted gamma (from x0)")
    ax.set_title("Predicted vs actual gamma")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig("figs/gamma_x0_relationship.png", bbox_inches="tight", dpi=300)
    plt.show()
    print("  -> Saved figs/gamma_x0_relationship.png")


# ======================================================================
# 3. Manifold learning
# ======================================================================

def run_manifold_learning(features, labels_dict, feature_names=("k", "x0", "gamma")):
    """
    Apply PCA, t-SNE, and UMAP to the parameter space.
    features: (n_samples, n_features) -- log-transformed where appropriate
    labels_dict: dict of label_name -> array of shape (n_samples,) for coloring
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # --- PCA ---
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # --- t-SNE ---
    print("  Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(features) - 1),
                random_state=42, init="pca", learning_rate="auto")
    X_tsne = tsne.fit_transform(X_scaled)

    # --- UMAP ---
    print("  Running UMAP...")
    try:
        import umap
        umap_reducer = umap.UMAP(n_components=2, n_neighbors=min(15, len(features) - 1),
                                 min_dist=0.1, random_state=42)
        X_umap = umap_reducer.fit_transform(X_scaled)
        has_umap = True
    except ImportError:
        print("  [WARN] umap-learn not installed, skipping UMAP")
        X_umap = np.zeros((len(features), 2))
        has_umap = False

    return {
        "pca": X_pca,
        "tsne": X_tsne,
        "umap": X_umap,
        "has_umap": has_umap,
        "pca_model": pca,
        "scaler": scaler,
    }, labels_dict


# ======================================================================
# 4. Clustering
# ======================================================================

def run_clustering(embedding, min_cluster_size=5):
    """Cluster using HDBSCAN on the UMAP embedding."""
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=3,
            cluster_selection_epsilon=0.5,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(embedding)
        return labels, clusterer
    except ImportError:
        print("  [WARN] hdbscan not installed, falling back to Gaussian Mixture")
        from sklearn.mixture import GaussianMixture

        # Try 2 to 6 components, pick by BIC
        bic_scores = []
        models = []
        for n in range(2, 7):
            gmm = GaussianMixture(n_components=n, random_state=42, covariance_type="full")
            gmm.fit(embedding)
            bic_scores.append(gmm.bic(embedding))
            models.append(gmm)

        best_idx = np.argmin(bic_scores)
        best_model = models[best_idx]
        labels = best_model.predict(embedding)
        print(f"  Best GMM: {best_idx + 2} components (BIC={bic_scores[best_idx]:.1f})")
        return labels, best_model


# ======================================================================
# 5. Plotting
# ======================================================================

DATASET_COLORS = {
    "starling": "#2c3e50",
    "jackdaw":  "#e67e22",
    "jackdaw2": "#27ae60",
    "swift":    "#8e44ad",
}


def set_style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 12,
        "axes.labelsize": 12, "axes.titlesize": 13,
        "legend.fontsize": 9, "xtick.direction": "in", "ytick.direction": "in",
        "axes.grid": True, "grid.alpha": 0.3,
    })


def get_color_map(names):
    """Return array of colors matching the list of dataset names."""
    return np.array([DATASET_COLORS.get(n, cm.tab10(i % 10)) for i, n in enumerate(names)])


def plot_pca_scree(pca_model, scaler):
    """Figure: PCA explained variance."""
    set_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.5))
    move_figure(fig, 100, 100)

    ev = pca_model.explained_variance_ratio_
    cumsum = np.cumsum(ev)
    components = np.arange(1, len(ev) + 1)

    ax0.bar(components, ev * 100, color="#3498db", alpha=0.7, edgecolor="white")
    ax0.set_xlabel("Principal component")
    ax0.set_ylabel("Explained variance (%)")
    ax0.set_title("PCA scree plot")
    ax0.set_xticks(components)

    ax1.plot(components, cumsum * 100, "o-", color="#2c3e50", lw=2, markersize=8)
    ax1.axhline(90, color="#e74c3c", linestyle="--", lw=1, alpha=0.5, label="90%")
    ax1.axhline(95, color="#e74c3c", linestyle=":", lw=1, alpha=0.5, label="95%")
    ax1.set_xlabel("Number of components")
    ax1.set_ylabel("Cumulative explained variance (%)")
    ax1.set_title("Cumulative explained variance")
    ax1.legend()
    ax1.set_xticks(components)

    # Print detailed PCA decomposition
    n_features = len(ev)
    if n_features == 2:
        feature_names = ("k", "log10(x0)")
    else:
        feature_names = ("k", "log10(x0)", "gamma")
    print(f"\n  -- PCA component loadings (coefficients of standardized features) --")
    for i, (var_pct, coefs) in enumerate(zip(ev * 100, pca_model.components_)):
        terms = " + ".join(f"({c:+.4f})*z_{name}" for c, name in zip(coefs, feature_names))
        print(f"  PC{i+1} ({var_pct:.1f}% var):  {terms}")

    # Show in terms of original (unstandardized) features
    print("\n  -- StandardScaler parameters --")
    for name, mu, sigma in zip(feature_names, scaler.mean_, scaler.scale_):
        print(f"  z_{name} = ({name} - {mu:.4f}) / {sigma:.4f}")

    print("\n  -- PC1 formula in raw features (substitute z_i above) --")
    c1 = pca_model.components_[0]
    for name, ci, mu, sigma in zip(feature_names, c1, scaler.mean_, scaler.scale_):
        print(f"  term_{name}: ({ci:+.4f}) * ({name} - {mu:.4f}) / {sigma:.4f}  =  ({ci/sigma:.4f}) * ({name} - {mu:.4f})")

    plt.tight_layout()
    plt.savefig("figs/3pl_pca_scree.png", bbox_inches="tight", dpi=300)
    plt.show()
    print("  -> Saved figs/3pl_pca_scree.png")


def plot_embeddings(embeddings, names, N_array, labels, has_umap):
    """Figure: t-SNE and UMAP embeddings colored by dataset, N, and cluster."""
    set_style()
    n_embed = 2 if has_umap else 1
    fig, axes = plt.subplots(3, n_embed, figsize=(6.5 * n_embed, 15),
                             squeeze=False)
    move_figure(fig, 100, 100)

    embed_keys = ["tsne", "umap"] if has_umap else ["tsne"]
    embed_titles = {"tsne": "t-SNE", "umap": "UMAP"}

    for col, key in enumerate(embed_keys):
        X = embeddings[key]

        # Row 0: colored by dataset
        ax = axes[0, col]
        for ds_name in sorted(set(names)):
            mask = names == ds_name
            ax.scatter(X[mask, 0], X[mask, 1], c=DATASET_COLORS[ds_name],
                       label=ds_name, s=20, alpha=0.7, edgecolors="none")
        ax.set_title(f"{embed_titles[key]} -- by dataset")
        ax.legend(frameon=False, markerscale=2, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

        # Row 1: colored by N
        ax = axes[1, col]
        sc = ax.scatter(X[:, 0], X[:, 1], c=N_array, cmap="plasma", s=20, alpha=0.7, edgecolors="none")
        ax.set_title(f"{embed_titles[key]} -- by N")
        plt.colorbar(sc, ax=ax, label="N")
        ax.set_xticks([]); ax.set_yticks([])

        # Row 2: colored by cluster
        ax = axes[2, col]
        unique_labels = sorted(set(labels))
        n_clusters = len([l for l in unique_labels if l >= 0])
        cluster_cmap = cm.tab10 if n_clusters <= 10 else cm.tab20
        for lbl in unique_labels:
            mask = labels == lbl
            label_str = f"Cluster {lbl}" if lbl >= 0 else "Noise"
            color = cluster_cmap(lbl % 10) if lbl >= 0 else "#bdc3c7"
            ax.scatter(X[mask, 0], X[mask, 1], c=[color], label=label_str,
                       s=20, alpha=0.7, edgecolors="none")
        ax.set_title(f"{embed_titles[key]} -- by cluster ({n_clusters} clusters)")
        ax.legend(frameon=False, markerscale=2, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

    # Hide unused columns
    for col in range(n_embed, axes.shape[1]):
        for row in range(3):
            axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig("figs/3pl_embeddings.png", bbox_inches="tight", dpi=300)
    plt.show()
    print("  -> Saved figs/3pl_embeddings.png")


def plot_cluster_curves(results_by_dataset, labels, all_params, all_N, all_names):
    """Figure: per-cluster mean 3PL curve +/- std band."""
    set_style()
    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l >= 0])

    if n_clusters == 0:
        print("  No clusters found, skipping cluster curve plot.")
        return

    sigma_shared = np.logspace(-2, 2, 200)
    cmap = cm.tab10 if n_clusters <= 10 else cm.tab20

    fig, axes = plt.subplots(1, n_clusters, figsize=(5.5 * n_clusters, 4.5),
                             squeeze=False)
    move_figure(fig, 100, 100)

    for idx, lbl in enumerate([l for l in unique_labels if l >= 0]):
        ax = axes[0, idx]
        mask = labels == lbl
        n_members = mask.sum()

        # Plot individual curves (thin, transparent)
        for i in np.where(mask)[0][:50]:
            k, x0, gamma = all_params[i]
            N_i = all_N[i]
            curve = power_3pl(sigma_shared, k, x0, gamma, A=N_i, D=1)
            ax.plot(sigma_shared, curve, color=cmap(lbl % 10), lw=0.5, alpha=0.15)

        # Mean curve
        if n_members > 1:
            all_curves = []
            for i in np.where(mask)[0]:
                k, x0, gamma = all_params[i]
                N_i = all_N[i]
                all_curves.append(power_3pl(sigma_shared, k, x0, gamma, A=N_i, D=1))
            mean_curve = np.mean(all_curves, axis=0)
            std_curve = np.std(all_curves, axis=0)
            ax.fill_between(sigma_shared, mean_curve - std_curve, mean_curve + std_curve,
                            color=cmap(lbl % 10), alpha=0.2)
            ax.plot(sigma_shared, mean_curve, color=cmap(lbl % 10), lw=2.5)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\sigma$")
        ax.set_ylabel("# Modes")
        ax.set_title(f"Cluster {lbl} (n={n_members})")

        ds_counts = {n: (all_names[mask] == n).sum() for n in sorted(set(all_names))}
        ds_str = ", ".join(f"{n}:{c}" for n, c in ds_counts.items() if c > 0)
        ax.text(0.95, 0.05, ds_str, transform=ax.transAxes, fontsize=7,
                ha="right", va="bottom", color="gray")

    plt.tight_layout()
    plt.savefig("figs/3pl_cluster_curves.png", bbox_inches="tight", dpi=300)
    plt.show()
    print("  -> Saved figs/3pl_cluster_curves.png")


def plot_parallel_coordinates(all_params, labels, feature_names):
    """Figure: parameter distributions colored by cluster."""
    set_style()
    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l >= 0])
    if n_clusters == 0:
        return

    n_cols = len(feature_names)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), squeeze=False)
    move_figure(fig, 100, 100)

    cmap_arr = cm.tab10 if n_clusters <= 10 else cm.tab20

    for col in range(n_cols):
        ax = axes[0, col]
        for lbl in unique_labels:
            mask = labels == lbl
            vals = all_params[mask, col]
            if lbl >= 0:
                ax.hist(vals, bins=30, alpha=0.5, color=cmap_arr(lbl % 10),
                        label=f"Cluster {lbl}" if col == 0 else "_nolegend_")
            else:
                ax.hist(vals, bins=30, alpha=0.2, color="#bdc3c7",
                        label="Noise" if col == 0 else "_nolegend_")
        ax.set_xlabel(feature_names[col])
        if col == 0:
            ax.set_ylabel("Count")
            ax.legend(frameon=False, fontsize=7)

    fig.suptitle("Parameter distributions by cluster", y=1.01)
    tag = "3pl" if n_cols >= 3 else "2pl"
    plt.tight_layout()
    plt.savefig(f"figs/{tag}_parameter_distributions.png", bbox_inches="tight", dpi=300)
    plt.show()
    print(f"  -> Saved figs/{tag}_parameter_distributions.png")


def plot_2d_parameter_space(all_params, all_names, all_N, labels):
    """Figure: scatter matrix of (k, x0, gamma) parameter pairs."""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    move_figure(fig, 100, 100)

    pairs = [(0, 1), (0, 2), (1, 2)]  # (k, x0), (k, gamma), (x0, gamma)
    titles = ["k vs x0", "k vs gamma", "x0 vs gamma"]

    for ax_idx, (i, j) in enumerate(pairs):
        ax = axes[ax_idx]
        for ds_name in sorted(set(all_names)):
            mask = all_names == ds_name
            ax.scatter(all_params[mask, i], all_params[mask, j],
                       c=DATASET_COLORS[ds_name], label=ds_name, s=15, alpha=0.6, edgecolors="none")
        ax.set_xlabel(["k", "x0", "gamma"][i])
        ax.set_ylabel(["k", "x0", "gamma"][j])
        ax.set_title(titles[ax_idx])
        if ax_idx == 0:
            ax.legend(frameon=False, markerscale=2, fontsize=8)
            ax.set_yscale('log')
        if ax_idx == 2:
            ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig("figs/3pl_parameter_pairs.png", bbox_inches="tight", dpi=300)
    plt.show()
    print("  -> Saved figs/3pl_parameter_pairs.png")


def plot_model_comparison(comparison, all_names, model_names=("3pl", "2pl")):
    """Figure: model comparison -- gamma diagnostics and residual variance.

    When model_names contains '3plc', the constrained model is included
    in all comparison panels alongside 3PL and 2PL.
    """
    set_style()
    n_models = len(model_names)
    fig = plt.figure(figsize=(18, 10))
    move_figure(fig, 100, 100)

    # Only use steps where ALL models succeeded
    all_ok = np.ones(len(comparison["_step_range"]), dtype=bool)
    for mname in model_names:
        all_ok &= comparison[mname]["success"]
    n_common = all_ok.sum()
    print(f"\n  Steps where all models converged: {n_common}/{len(all_ok)}")

    rv = {mname: comparison[mname]["resid_var"][all_ok] for mname in model_names}

    names_ok = all_names[all_ok] if len(all_names) == len(all_ok) else all_names
    if len(names_ok) != n_common:
        names_ok = np.array(["unknown"] * n_common)

    datasets = sorted(set(names_ok))
    ds_colors = {ds: DATASET_COLORS.get(ds, cm.tab10(i % 10))
                 for i, ds in enumerate(datasets)}
    bar_colors = {"3pl": "#e74c3c", "2pl": "#3498db", "3plc": "#2ecc71"}
    win_labels = {"3pl": "3PL\n(free g)", "2pl": "2PL\n(g=1)", "3plc": "3PLc\n(per-ds)"}

    # -- Row 1, Col 1: gamma histogram --
    ax = fig.add_subplot(2, 3, 1)
    gamma_vals = np.array([comparison["3pl"]["params"][i][2]
                           for i in range(len(comparison["_step_range"]))
                           if comparison["3pl"]["success"][i]])
    n_large = np.sum(gamma_vals >= 200.0)
    ax.hist(gamma_vals, bins=40, color="#e74c3c", alpha=0.7, edgecolor="white")
    ax.axvline(200, color="black", linestyle="--", lw=1.5, label="gamma=200")
    ax.set_xlabel("gamma")
    ax.set_ylabel("Count")
    ax.set_title(f"3PL gamma distribution\n({n_large}/{len(gamma_vals)} with gamma>=200)")
    ax.legend(fontsize=7)

    # -- Row 1, Col 2: large-gamma fraction by dataset --
    ax = fig.add_subplot(2, 3, 2)
    ds_list = sorted(set(all_names))
    large_fracs, ds_labels = [], []
    for ds in ds_list:
        mask_ds = np.array(all_names) == ds
        n_3pl_ok = np.sum(comparison["3pl"]["success"] & mask_ds)
        n_large_ds = np.sum(comparison["_hit_ceiling"] & mask_ds)
        if n_3pl_ok > 0:
            large_fracs.append(n_large_ds / n_3pl_ok * 100)
            ds_labels.append(ds)
    ax.bar(range(len(ds_labels)), large_fracs,
           color=[DATASET_COLORS.get(d, "#888") for d in ds_labels],
           alpha=0.7, edgecolor="white")
    ax.set_xticks(range(len(ds_labels)))
    ax.set_xticklabels(ds_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("gamma>=200 (%)")
    ax.set_title("Large gamma by dataset")
    ax.set_ylim(0, 105)
    for i, (v, ds) in enumerate(zip(large_fracs, ds_labels)):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=8)

    # -- Row 1, Col 3: which model wins --
    ax = fig.add_subplot(2, 3, 3)
    win_counts = {m: 0 for m in model_names}
    for i in range(n_common):
        best = min(model_names, key=lambda m: rv[m][i])
        win_counts[best] += 1
    labels_display = [win_labels.get(m, m) for m in model_names]
    ax.bar(model_names, [win_counts[m] for m in model_names],
           color=[bar_colors.get(m, "#888") for m in model_names],
           alpha=0.7, edgecolor="white")
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(labels_display, fontsize=8)
    ax.set_ylabel("Number of steps")
    ax.set_title(f"Best model (lowest resid var)\n({n_common} steps)")
    for i, m in enumerate(model_names):
        ax.text(i, win_counts[m] + max(win_counts.values()) * 0.02, str(win_counts[m]),
                ha="center", fontsize=10)

    # -- Row 2, Col 1: scatter 3PL vs 2PL residual variance --
    ax = fig.add_subplot(2, 3, 4)
    rv_3pl_log = np.log10(np.maximum(rv["3pl"], 1e-12))
    rv_2pl_log = np.log10(np.maximum(rv["2pl"], 1e-12))
    for ds in datasets:
        mask_ds = names_ok == ds
        if mask_ds.sum() == 0:
            continue
        ax.scatter(rv_3pl_log[mask_ds], rv_2pl_log[mask_ds],
                   c=ds_colors[ds], label=ds, s=15, alpha=0.5, edgecolors="none")
    lo = min(rv_3pl_log.min(), rv_2pl_log.min())
    hi = max(rv_3pl_log.max(), rv_2pl_log.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("3PL log10(resid var)")
    ax.set_ylabel("2PL log10(resid var)")
    n_2pl = np.sum(rv_2pl_log < rv_3pl_log)
    n_3pl = np.sum(rv_3pl_log < rv_2pl_log)
    ax.set_title(f"3PL vs 2PL residual variance\n(2PL better: {n_2pl}, 3PL better: {n_3pl})")
    ax.legend(frameon=False, markerscale=2, fontsize=7)

    # -- Row 2, Col 2: boxplot (all models pooled) --
    ax = fig.add_subplot(2, 3, 5)
    box_data = [rv[m] for m in model_names]
    box_labels = [win_labels.get(m, m) for m in model_names]
    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, showfliers=False)
    for patch, mname in zip(bp["boxes"], model_names):
        patch.set_facecolor(bar_colors.get(mname, "#888"))
        patch.set_alpha(0.6)
    ax.set_ylabel("Residual variance")
    ax.set_title(f"All datasets pooled (n={n_common})")
    ax.set_yscale("log")
    ax.tick_params(axis="x", labelsize=8)

    # -- Row 2, Col 3: per-dataset mean residual variance --
    ax = fig.add_subplot(2, 3, 6)
    x_pos = np.arange(len(datasets))
    width = 0.8 / n_models
    hatches = ["", "//", ".."]
    for i, mname in enumerate(model_names):
        means = []
        for ds in datasets:
            mask_ds = names_ok == ds
            if mask_ds.sum() > 0:
                means.append(np.mean(rv[mname][mask_ds]))
            else:
                means.append(0)
        ax.bar(x_pos + i * width, means, width,
               label=win_labels.get(mname, mname),
               color=bar_colors.get(mname, "#888"), alpha=0.7,
               edgecolor="white", hatch=hatches[i % len(hatches)])
    ax.set_xticks(x_pos + width * (n_models - 1) / 2)
    ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean residual variance")
    ax.set_title("Mean residual variance by dataset")
    ax.legend(frameon=False, fontsize=8)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig("figs/model_comparison.png", bbox_inches="tight", dpi=300)
    plt.show()
    print("  -> Saved figs/model_comparison.png")


def print_model_summary(comparison, all_names, model_names=("3pl", "2pl")):
    """Print per-dataset and overall model comparison statistics."""
    all_ok = np.ones(len(comparison["_step_range"]), dtype=bool)
    for mname in model_names:
        all_ok &= comparison[mname]["success"]
    n_common = all_ok.sum()

    model_labels = {"3pl": "3PL (free g)", "2pl": "2PL (g=1)", "3plc": "3PLc (per-ds g=f(x0))"}

    print(f"\n{'='*80}")
    print(f"Model comparison ({len(model_names)} models)")
    print(f"{'='*80}")
    print(f"Steps where all converged: {n_common}")

    # Overall
    print(f"\n  {'Model':<24} {'#Params':<9} {'Mean RV':<14} {'Median RV':<14} {'Win %':<10}")
    print(f"  {'-'*65}")
    for mname in model_names:
        rv_ok = comparison[mname]["resid_var"][all_ok]
        wins = 0
        for i in range(n_common):
            best = min(model_names, key=lambda m: comparison[m]["resid_var"][all_ok][i])
            if best == mname:
                wins += 1
        win_pct = wins / n_common * 100 if n_common > 0 else 0
        label = model_labels.get(mname, mname)
        print(f"  {label:<24} {comparison[mname]['n_params']:<9} "
              f"{np.mean(rv_ok):<14.4f} {np.median(rv_ok):<14.4f} {win_pct:<10.1f}")

    # Per-dataset
    print(f"\n  -- Per dataset --")
    for ds in sorted(set(all_names)):
        mask_ds = np.array(all_names) == ds
        mask = all_ok & mask_ds
        n = mask.sum()
        if n == 0:
            continue
        print(f"\n  {ds} (n={n}):")
        print(f"    {'Model':<24} {'Mean RV':<14} {'Median RV':<14}")
        for mname in model_names:
            rv_m = comparison[mname]["resid_var"][mask]
            label = model_labels.get(mname, mname)
            print(f"    {label:<24} {np.mean(rv_m):<14.4f} {np.median(rv_m):<14.4f}")

    # Gamma stats
    if "3pl" in comparison:
        n_3pl_ok = comparison["3pl"]["success"].sum()
        n_large = comparison["_hit_ceiling"].sum()
        n_large_per_ds = {}
        for ds in sorted(set(all_names)):
            mask_ds = np.array(all_names) == ds
            n_large_per_ds[ds] = int(np.sum(comparison["_hit_ceiling"] & mask_ds))
        gamma_vals = np.array([comparison["3pl"]["params"][i][2]
                               for i in range(len(comparison["_step_range"]))
                               if comparison["3pl"]["success"][i]])
        print(f"\n  3PL gamma stats: range=[{np.min(gamma_vals):.1f}, {np.max(gamma_vals):.1f}], "
              f"median={np.median(gamma_vals):.1f}")
        print(f"  gamma>=200: {n_large}/{n_3pl_ok} = {n_large/n_3pl_ok*100:.1f}%")
        print(f"  By dataset: {n_large_per_ds}")


def print_summary(all_params, labels, all_names, all_N):
    """Print per-cluster summary statistics."""
    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l >= 0])

    print(f"\n{'='*80}")
    print(f"Clustering summary: {n_clusters} clusters, {sum(labels == -1)} noise points")
    print(f"{'='*80}")

    for lbl in unique_labels:
        mask = labels == lbl
        n_pts = mask.sum()
        if lbl == -1:
            label_str = "Noise"
        else:
            label_str = f"Cluster {lbl}"

        params_subset = all_params[mask]
        N_subset = all_N[mask]
        names_subset = all_names[mask]

        ds_counts = {n: (names_subset == n).sum() for n in sorted(set(all_names))}

        print(f"\n{label_str} ({n_pts} points):")
        print(f"  Datasets: {ds_counts}")
        print(f"  N:        mean={np.mean(N_subset):.0f} +- {np.std(N_subset):.0f}, "
              f"range=[{np.min(N_subset):.0f}, {np.max(N_subset):.0f}]")
        if n_pts >= 1:
            n_cols = params_subset.shape[1]
            print(f"  k:        mean={np.nanmean(params_subset[:, 0]):.3f} +- {np.nanstd(params_subset[:, 0]):.3f}")
            print(f"  x0:       mean={np.nanmean(params_subset[:, 1]):.4f} +- {np.nanstd(params_subset[:, 1]):.4f}")
            if n_cols >= 3:
                print(f"  gamma:    mean={np.nanmean(params_subset[:, 2]):.3f} +- {np.nanstd(params_subset[:, 2]):.3f}")


# ======================================================================
# 6. Main
# ======================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="3PL manifold analysis across datasets")
    parser.add_argument("--skip-fit", action="store_true",
                        help="Skip fitting (use existing params if available)")
    parser.add_argument("--model", type=str, default="3pl",
                        choices=["3pl", "2pl"],
                        help="Which model to use for manifold learning (default: 3pl)")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip plt.show() -- save figures to disk only, print results to console")
    args = parser.parse_args()
    if args.no_display:
        plt.show = lambda: None  # suppress all figure display
        print("[--no-display] Suppressing figure windows; saving to disk only.\n")

    # --- Step 1: Load cached data & fit 3PL + 2PL ---
    all_params_list = []
    all_N_list = []
    all_names_list = []
    all_step_list = []
    results_by_dataset = {}
    comp_by_dataset = {}
    raw_data_by_dataset = {}

    for run_params in DATASET_RUNS:
        name = run_params["name"]
        print(f"\n{'='*60}\nDataset: {name}\n{'='*60}")

        step_range, N_array, scale_range, all_modes = load_cached_data(run_params)
        if step_range is None:
            continue

        raw_data_by_dataset[name] = {
            "step_range": step_range,
            "N_array": N_array,
            "scale_range": scale_range,
            "all_modes": all_modes,
        }

        print(f"  Fitting 3PL + 2PL for {len(step_range)} time steps...")
        comp = fit_all_models_all_steps(step_range, N_array, scale_range, all_modes)
        comp_by_dataset[name] = comp

        model_key = args.model
        n_ok = comp[model_key]["success"].sum()
        print(f"  {model_key} successfully fitted: {n_ok}/{len(step_range)}")

        valid_mask = comp[model_key]["success"]
        n_valid = valid_mask.sum()

        if n_valid > 0:
            params_valid = np.array([comp[model_key]["params"][i] for i in range(len(valid_mask)) if valid_mask[i]])
            all_params_list.append(params_valid)
            all_N_list.append(N_array[valid_mask])
            all_names_list.append(np.array([name] * n_valid))
            all_step_list.append(np.array(step_range)[valid_mask])

            results_by_dataset[name] = {
                "step_range": np.array(step_range)[valid_mask],
                "N_array": N_array[valid_mask],
                "params": params_valid,
                "scale_range": scale_range[valid_mask],
                "all_modes": all_modes[valid_mask],
            }

    if not all_params_list:
        print("No data loaded. Exiting.")
        return

    all_params = np.vstack(all_params_list)
    all_N = np.concatenate(all_N_list)
    all_names = np.concatenate(all_names_list)
    all_steps = np.concatenate(all_step_list)

    n_total = len(all_params)
    print(f"\n{'='*60}")
    print(f"Total samples ({args.model}): {n_total}")
    print(f"Datasets: {sorted(set(all_names))}")

    n_features = all_params.shape[1]
    if n_features == 3:
        print(f"Parameter ranges:")
        print(f"  k:     [{np.nanmin(all_params[:, 0]):.4f}, {np.nanmax(all_params[:, 0]):.4f}]")
        print(f"  x0:    [{np.nanmin(all_params[:, 1]):.4f}, {np.nanmax(all_params[:, 1]):.4f}]")
        print(f"  gamma: [{np.nanmin(all_params[:, 2]):.4f}, {np.nanmax(all_params[:, 2]):.4f}]")
    else:
        print(f"Parameter ranges:")
        print(f"  k:  [{np.nanmin(all_params[:, 0]):.4f}, {np.nanmax(all_params[:, 0]):.4f}]")
        print(f"  x0: [{np.nanmin(all_params[:, 1]):.4f}, {np.nanmax(all_params[:, 1]):.4f}]")

    # --- Step 1b: Build merged comparison (3PL + 2PL) ---
    ds_names_sorted = sorted(comp_by_dataset.keys())

    all_names_for_comp = []
    for name in ds_names_sorted:
        n_steps_total = len(comp_by_dataset[name]["_step_range"])
        all_names_for_comp.append(np.array([name] * n_steps_total))
    all_names_full = np.concatenate(all_names_for_comp)

    merged_comp = {
        "_step_range": np.concatenate([comp_by_dataset[n]["_step_range"] for n in ds_names_sorted]),
        "_N_array": np.concatenate([comp_by_dataset[n]["_N_array"] for n in ds_names_sorted]),
        "_hit_ceiling": np.concatenate([comp_by_dataset[n]["_hit_ceiling"] for n in ds_names_sorted]),
    }
    for mname in ["3pl", "2pl"]:
        merged_comp[mname] = {
            "params":    sum([comp_by_dataset[n][mname]["params"] for n in ds_names_sorted], []),
            "fitted":    sum([comp_by_dataset[n][mname]["fitted"] for n in ds_names_sorted], []),
            "resid_var": np.concatenate([comp_by_dataset[n][mname]["resid_var"] for n in ds_names_sorted]),
            "success":   np.concatenate([comp_by_dataset[n][mname]["success"] for n in ds_names_sorted]),
            "n_params":  comp_by_dataset[ds_names_sorted[0]][mname]["n_params"],
        }

    # --- Step 1c: Gamma-x0 quadratic fit & constrained 3PL refit ---
    print(f"\n{'='*60}")
    print("Gamma-x0 quadratic relationship")
    print(f"{'='*60}")

    quad_result = fit_gamma_x0_quadratic(merged_comp)

    if quad_result is not None and quad_result[0] is not None:
        plot_gamma_x0_relationship(merged_comp, all_names_full, quad_result)
        global_a, global_b, global_c = quad_result[0]

        per_ds_coeffs = print_per_dataset_quadratic(merged_comp, all_names_full)

        # Run constrained refit per dataset using per-dataset coefficients
        print(f"\n  Running constrained 3PL refit (per-dataset gamma = f(x0))...")
        constrained_results = {}
        for name in ds_names_sorted:
            rd = raw_data_by_dataset[name]
            if name in per_ds_coeffs:
                a_ds, b_ds, c_ds, r2_ds = per_ds_coeffs[name]
                print(f"    {name}: using per-dataset (a={a_ds:.3f}, b={b_ds:.3f}, c={c_ds:.3f}, R^2={r2_ds:.3f})")
            else:
                a_ds, b_ds, c_ds = global_a, global_b, global_c
                print(f"    {name}: too few points, falling back to global (a={a_ds:.3f}, b={b_ds:.3f}, c={c_ds:.3f})")
            cr = fit_constrained_refit(
                rd["step_range"], rd["N_array"], rd["scale_range"], rd["all_modes"],
                a_ds, b_ds, c_ds,
            )
            constrained_results[name] = {
                "resid_var": cr["resid_var"],
                "success": cr["success"],
                "n_params": 2,
            }

        merged_comp["3plc"] = {
            "resid_var": np.concatenate([constrained_results[n]["resid_var"] for n in ds_names_sorted]),
            "success": np.concatenate([constrained_results[n]["success"] for n in ds_names_sorted]),
            "n_params": 2,
            "params": [],
            "fitted": [],
        }
        comparison_models = ("3pl", "2pl", "3plc")
    else:
        comparison_models = ("3pl", "2pl")

    # --- Step 1d: Model comparison ---
    print_model_summary(merged_comp, all_names_full, model_names=comparison_models)
    plot_model_comparison(merged_comp, all_names_full, model_names=comparison_models)

    # --- Step 2: Manifold learning on chosen model's parameters ---
    if n_features == 3:
        features = np.column_stack([
            all_params[:, 0],
            np.log10(all_params[:, 1]),
            all_params[:, 2],
        ])
        model_feature_names = ("k", "log10(x0)", "gamma")
    else:
        features = np.column_stack([
            all_params[:, 0],
            np.log10(all_params[:, 1]),
        ])
        model_feature_names = ("k", "log10(x0)")

    print(f"\n{'='*60}")
    print(f"Manifold learning on {args.model} parameter space: {model_feature_names}")
    print(f"{'='*60}")

    embeddings_dict, labels_dict = run_manifold_learning(
        features, {"dataset": all_names, "N": all_N, "step": all_steps},
        feature_names=model_feature_names,
    )

    # --- Step 3: Clustering on UMAP ---
    print(f"\n{'='*60}")
    print("Clustering on UMAP embedding")
    print(f"{'='*60}")

    if embeddings_dict["has_umap"]:
        cluster_embedding = embeddings_dict["umap"]
    else:
        cluster_embedding = embeddings_dict["tsne"]

    labels, clusterer = run_clustering(cluster_embedding)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = sum(labels == -1)
    print(f"  Found {n_clusters} clusters, {n_noise} noise points")

    # --- Step 4: Plotting ---
    print(f"\n{'='*60}")
    print("Generating figures...")
    print(f"{'='*60}")

    if n_features == 3:
        plot_2d_parameter_space(all_params, all_names, all_N, labels)

    plot_pca_scree(embeddings_dict["pca_model"], embeddings_dict["scaler"])
    plot_embeddings(embeddings_dict, all_names, all_N, labels, embeddings_dict["has_umap"])

    if n_features == 3:
        plot_cluster_curves(results_by_dataset, labels, all_params, all_N, all_names)

    plot_parallel_coordinates(all_params, labels,
                              feature_names=("k", "x0", "gamma") if n_features == 3 else ("k", "x0"))

    # --- Step 5: Summary ---
    print_summary(all_params, labels, all_names, all_N)


if __name__ == "__main__":
    main()
