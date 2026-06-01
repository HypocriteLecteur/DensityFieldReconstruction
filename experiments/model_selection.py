"""
Model Selection: find a well-behaved closed-form model for mode-count-vs-scale curves.

Compares 4 candidate models across all datasets:
  A. Original 3PL (baseline) — m(s) = 1 + (N-1)/(1 + (s/x0)^k)^gamma
  B. Log-normal CDF        — m(s) = 1 + (N-1) * 0.5 * erfc((log s - mu)/(s*√2))
  C. Centered 3PL          — m(s) = 1 + (N-1)/(1 + (2^(1/g)-1)*(s/s_half)^k)^g
  D. Centered 2PL          — m(s) = 1 + (N-1)/(1 + (s/s_half)^k)

Model C is the key innovation: replacing x0 with sigma_half (the scale where
half the modes have merged) prevents x0 from exploding when gamma is large.

Usage:
  python experiments/model_selection.py              # full run
  python experiments/model_selection.py --no-display  # headless
  python experiments/model_selection.py --skip-fit    # reload cached fits
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc, expm1
from scipy.stats import spearmanr
from tqdm import tqdm
from collections import defaultdict
import warnings

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec

from dfr.simulation_config import SimulationConfig
from dfr.dataset_io import DatasetFactory
from experiments.power_law import power_3pl, move_figure


# ======================================================================
# 0. Model functions
# ======================================================================

def model_orig_3pl(x, params, N):
    """Original 3PL with log10(x0) parameterization.

    params = [k, log10_x0, gamma]
    Returns m(x) - 1  (caller adds D=1).
    """
    k, log10_x0, gamma = params
    x0 = 10.0**log10_x0
    log_ratio = np.clip(k * np.log(np.maximum(x / x0, 1e-12)), -500, 500)
    return (N - 1.0) / np.power(1.0 + np.exp(log_ratio), gamma)


def model_lognormal_cdf(x, params, N):
    """Log-normal CDF (survival function): m(x) = 1 + (N-1) * (1 - Phi(z)).

    params = [mu, s]
    where mu = log(median merge scale), s = spread on log scale.

    Uses erfc for numerical stability in the upper tail.
    Phi(z) = 0.5 * (1 + erf(z/√2)), so 1-Phi(z) = 0.5 * erfc(z/√2).
    """
    mu, s = params
    s_safe = np.maximum(s, 1e-6)
    z = (np.log(np.maximum(x, 1e-12)) - mu) / (s_safe * np.sqrt(2.0))
    return (N - 1.0) * 0.5 * erfc(z)


def model_centered_3pl(x, params, N):
    """Centered 3PL with sigma_half reparameterization.

    m(s) = 1 + (N-1) / (1 + (2^(1/gamma)-1) * (s/sigma_half)^k)^gamma

    params = [k, sigma_half, gamma]
    sigma_half is the scale where m = (N+1)/2 for any k, gamma.

    Uses expm1 for numerical stability: 2^(1/gamma) - 1 = expm1(ln(2)/gamma).
    All computation in log-space with clipping.
    """
    k, sigma_half, gamma = params
    g_safe = np.maximum(gamma, 1e-6)
    # scaling = 2^(1/gamma) - 1 = expm1(ln(2)/gamma)
    scaling = expm1(np.log(2.0) / g_safe)
    scaling_safe = np.maximum(scaling, 1e-12)
    log_ratio = np.clip(k * np.log(np.maximum(x / sigma_half, 1e-12)) +
                         np.log(scaling_safe), -500, 500)
    return (N - 1.0) / np.power(1.0 + np.exp(log_ratio), gamma)


def model_centered_2pl(x, params, N):
    """Centered 2PL: centered 3PL with gamma=1.

    m(s) = 1 + (N-1) / (1 + (s/sigma_half)^k)
    params = [k, sigma_half]
    """
    k, sigma_half = params
    log_ratio = np.clip(k * np.log(np.maximum(x / sigma_half, 1e-12)), -500, 500)
    return (N - 1.0) / (1.0 + np.exp(log_ratio))


# Model registry
MODELS = {
    "orig_3pl": {
        "fn": model_orig_3pl,
        "n_params": 3,
        "param_names": ["k", "log10_x0", "gamma"],
        "p0": lambda med: [2.0, np.log10(med), 1.0],
        "bounds": ([0.1, -2.0, 0.01], [20.0, 4.0, np.inf]),
        "label": "Original 3PL",
    },
    "lognormal": {
        "fn": model_lognormal_cdf,
        "n_params": 2,
        "param_names": ["mu", "s"],
        "p0": lambda med: [np.log(med), 1.0],
        "bounds": ([-np.inf, 0.01], [np.inf, np.inf]),
        "label": "Log-normal CDF",
    },
    "centered_3pl": {
        "fn": model_centered_3pl,
        "n_params": 3,
        "param_names": ["k", "sigma_half", "gamma"],
        "p0": lambda med: [2.0, med, 1.0],
        "bounds": ([0.1, 1e-6, 0.01], [20.0, np.inf, np.inf]),
        "label": "Centered 3PL",
    },
    "centered_2pl": {
        "fn": model_centered_2pl,
        "n_params": 2,
        "param_names": ["k", "sigma_half"],
        "p0": lambda med: [2.0, med],
        "bounds": ([0.1, 1e-6], [20.0, np.inf]),
        "label": "Centered 2PL",
    },
}

MODEL_ORDER = ["orig_3pl", "lognormal", "centered_3pl", "centered_2pl"]


# ======================================================================
# 1. Data loading
# ======================================================================

DATASET_RUNS = [
    {"name": "swift",    "start_step": 0,    "end_step": None, "step_length": 200},
    {"name": "starling", "start_step": 0,    "end_step": None, "step_length": 1},
    {"name": "jackdaw",  "start_step": 350,  "end_step": 550,  "step_length": 10},
    {"name": "jackdaw2", "start_step": 2700, "end_step": 3460, "step_length": 20},
]

DATASET_COLORS = {
    "starling": "#2c3e50",
    "jackdaw":  "#e67e22",
    "jackdaw2": "#27ae60",
    "swift":    "#8e44ad",
}


def set_style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "legend.fontsize": 8, "xtick.direction": "in", "ytick.direction": "in",
        "axes.grid": True, "grid.alpha": 0.3,
    })


def load_cached_data(run_params):
    """Load cached modes.npy and scale_range.npy for a dataset."""
    name = run_params["name"]
    scenario_path = os.path.join(os.getcwd(), "scenarios", name)

    modes_path = os.path.join(scenario_path, "modes.npy")
    scale_range_path = os.path.join(scenario_path, "scale_range.npy")

    if not os.path.exists(modes_path) or not os.path.exists(scale_range_path):
        print(f"  [WARN] No cache found for '{name}'")
        return None, None, None, None

    all_modes = np.load(modes_path)
    scale_range = np.load(scale_range_path)

    config_path = os.path.join(scenario_path, "config.yaml")
    config = SimulationConfig(config_path)
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    max_steps = dataset.trajectories.shape[0]
    end_step = run_params["end_step"]
    effective_end = end_step if end_step is not None and end_step <= max_steps else max_steps
    step_range = list(range(run_params["start_step"], effective_end, run_params["step_length"]))

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

    print(f"  Loading N for {name} ({len(step_range)} steps)...")
    N_array = np.array([dataset.positions_at_time_step(s).shape[0] for s in tqdm(step_range)])

    return step_range, N_array, scale_range, all_modes


# ======================================================================
# 2. Fitting pipeline
# ======================================================================

def fit_all_models_one_step(test_scales, modes, N, model_names, saturation=0.8):
    """Fit all models to a single (scale, modes) curve.

    saturation: trim the plateau where modes > saturation * N.
                Lower values remove more of the flat top.
    Returns:
        results: dict model_name -> {params, fitted, resid_var, success}
    """
    # Trim saturated region
    begin_idx = np.argmax(modes <= saturation * N)
    if begin_idx == 0:
        begin_idx = max(1, np.argmax(modes <= min(saturation + 0.1, 0.99) * N))

    x_data = test_scales[begin_idx:]
    y_data = modes[begin_idx:]

    results = {}
    for mname in model_names:
        entry = {"success": False, "params": None, "fitted": None,
                 "resid_var": np.nan}
        if len(x_data) < MODELS[mname]["n_params"] + 2:
            results[mname] = entry
            continue

        spec = MODELS[mname]
        median_scale = float(np.median(test_scales))
        try:
            popt, pcov = curve_fit(
                lambda x, *p: spec["fn"](x, p, N),
                x_data, y_data,
                p0=spec["p0"](median_scale),
                sigma=np.maximum(y_data, 1.0),
                absolute_sigma=True,
                bounds=spec["bounds"],
                maxfev=5000,
            )
            fitted_full = spec["fn"](test_scales, popt, N) + 1.0
            fitted_trim = spec["fn"](x_data, popt, N) + 1.0
            resid_var = np.mean((y_data - fitted_trim)**2)

            entry["success"] = True
            entry["params"] = popt
            entry["fitted"] = fitted_full
            entry["resid_var"] = resid_var
            entry["pcov"] = pcov if pcov is not None else np.full((len(popt), len(popt)), np.nan)
        except (RuntimeError, ValueError):
            pass
        results[mname] = entry

    return results


def fit_all_models_all_steps(step_range, N_array, scale_range, all_modes,
                             model_names, num_test_scale=40, saturation=0.8):
    """Fit all models to all time steps of one dataset."""
    n_steps = all_modes.shape[0]
    all_results = {mname: {
        "params": [None] * n_steps,
        "fitted": [None] * n_steps,
        "resid_var": np.full(n_steps, np.nan),
        "success": np.zeros(n_steps, dtype=bool),
        "pcov": [None] * n_steps,
    } for mname in model_names}

    # Also store the raw data for plotting
    all_results["_test_scales"] = [None] * n_steps
    all_results["_modes"] = [None] * n_steps
    all_results["_N"] = [None] * n_steps

    for i in range(n_steps):
        N = N_array[i]
        s_start, s_end = scale_range[i]
        test_scales = np.logspace(np.log10(max(s_start, 1e-6)),
                                  np.log10(max(s_end, 1e-5)), num_test_scale)
        modes = all_modes[i]

        all_results["_test_scales"][i] = test_scales
        all_results["_modes"][i] = modes
        all_results["_N"][i] = N

        step_results = fit_all_models_one_step(test_scales, modes, N, model_names,
                                                saturation=saturation)
        for mname in model_names:
            r = step_results[mname]
            all_results[mname]["params"][i] = r["params"]
            all_results[mname]["fitted"][i] = r["fitted"]
            all_results[mname]["resid_var"][i] = r["resid_var"]
            all_results[mname]["success"][i] = r["success"]
            all_results[mname]["pcov"][i] = r.get("pcov", None)

    return all_results


# ======================================================================
# 3. Metrics
# ======================================================================

def compute_all_metrics(all_fits_by_ds, all_names_full, merged):
    """Compute goodness-of-fit, parameter behavior, and manifold quality metrics."""
    metrics = {}

    model_names = [k for k in merged.keys() if not k.startswith("_")]
    all_ok = np.ones(len(all_names_full), dtype=bool)
    for mname in model_names:
        all_ok &= merged[mname]["success"]
    n_common = all_ok.sum()

    # --- 3a. Goodness-of-fit ---
    gof = {}
    n_pts_per_fit = 40  # approximate; trimmed data is fewer
    for mname in model_names:
        rv = merged[mname]["resid_var"][all_ok]
        n_p = MODELS[mname]["n_params"]
        # AICc: n*log(RSS/n) + 2k + 2k(k+1)/(n-k-1)
        # Approximate n_data_points = 35 (after trimming)
        n_data = 35.0
        rss = rv * n_data
        log_rss_n = np.log(np.maximum(rss / n_data, 1e-12))
        aicc_vals = n_data * log_rss_n + 2.0 * n_p + 2.0 * n_p * (n_p + 1.0) / np.maximum(n_data - n_p - 1.0, 1.0)
        bic_vals = n_data * log_rss_n + n_p * np.log(n_data)
        gof[mname] = {
            "mean_rv": float(np.mean(rv)),
            "median_rv": float(np.median(rv)),
            "mean_aicc": float(np.mean(aicc_vals[~np.isinf(aicc_vals)])),
            "mean_bic": float(np.mean(bic_vals[~np.isinf(bic_vals)])),
        }

    # Win counts
    win_counts = {mname: 0 for mname in model_names}
    for i in range(n_common):
        best = min(model_names, key=lambda m: merged[m]["resid_var"][all_ok][i])
        win_counts[best] += 1
    for mname in model_names:
        gof[mname]["win_pct"] = win_counts[mname] / n_common * 100.0 if n_common > 0 else 0.0

    metrics["gof"] = gof
    metrics["n_common"] = n_common

    # --- 3b. Per-dataset breakdown ---
    per_ds = {}
    for ds in sorted(set(all_names_full)):
        mask_ds = np.array(all_names_full) == ds
        mask = all_ok & mask_ds
        n = mask.sum()
        if n == 0:
            continue
        per_ds[ds] = {"n": n}
        for mname in model_names:
            rv_ds = merged[mname]["resid_var"][mask]
            per_ds[ds][mname] = {
                "mean_rv": float(np.mean(rv_ds)),
                "median_rv": float(np.median(rv_ds)),
            }
    metrics["per_dataset"] = per_ds

    # --- 3c. Parameter correlations ---
    param_corrs = {}
    for mname in model_names:
        pnames = MODELS[mname]["param_names"]
        n_p = len(pnames)
        param_matrix = []
        for i in range(len(all_names_full)):
            if merged[mname]["success"][i] and all_ok[i]:
                param_matrix.append(merged[mname]["params"][i])
        if len(param_matrix) < 5:
            param_corrs[mname] = {"corr_matrix": None, "max_corr": np.nan}
            continue
        param_arr = np.array(param_matrix)
        corr_matrix = np.zeros((n_p, n_p))
        for ii in range(n_p):
            for jj in range(n_p):
                if ii == jj:
                    corr_matrix[ii, jj] = 1.0
                else:
                    rho, _ = spearmanr(param_arr[:, ii], param_arr[:, jj])
                    corr_matrix[ii, jj] = rho
        # Max absolute off-diagonal correlation
        off_diag = corr_matrix[~np.eye(n_p, dtype=bool)]
        max_corr = float(np.max(np.abs(off_diag))) if len(off_diag) > 0 else np.nan
        param_corrs[mname] = {"corr_matrix": corr_matrix, "max_corr": max_corr,
                              "param_names": pnames}
    metrics["param_corrs"] = param_corrs

    # --- 3d. Parameter ranges ---
    param_ranges = {}
    for mname in model_names:
        pnames = MODELS[mname]["param_names"]
        n_p = len(pnames)
        ranges = {}
        for pi in range(n_p):
            vals = []
            for i in range(len(all_names_full)):
                if merged[mname]["success"][i]:
                    vals.append(merged[mname]["params"][i][pi])
            if vals:
                vals_arr = np.array(vals)
                ranges[pnames[pi]] = (float(np.min(vals_arr)), float(np.max(vals_arr)),
                                       float(np.median(vals_arr)))
        # Fraction at bounds
        bounds_hi = MODELS[mname]["bounds"][1]
        for pi in range(n_p):
            if np.isinf(bounds_hi[pi]):
                continue
            n_at_bound = 0
            n_total = 0
            for i in range(len(all_names_full)):
                if merged[mname]["success"][i]:
                    n_total += 1
                    if merged[mname]["params"][i][pi] >= bounds_hi[pi] * 0.999:
                        n_at_bound += 1
            if n_total > 0:
                ranges[f"{pnames[pi]}_at_bound"] = f"{n_at_bound}/{n_total}"
        param_ranges[mname] = ranges
    metrics["param_ranges"] = param_ranges

    return metrics


# ======================================================================
# 4. Console output
# ======================================================================

def print_comparison_report(metrics):
    """Print formatted comparison table and diagnostics."""
    gof = metrics["gof"]
    model_names = list(gof.keys())
    n_common = metrics["n_common"]

    print(f"\n{'='*100}")
    print("MODEL SELECTION REPORT")
    print(f"{'='*100}")
    print(f"Steps where all models converged: {n_common}")

    # Main comparison table
    print(f"\n  {'Model':<20} {'#P':<4} {'Mean RV':<12} {'Median RV':<12} "
          f"{'AICc':<10} {'BIC':<10} {'Win %':<8} {'Max|corr|':<10}")
    print(f"  {'-'*85}")
    for mname in model_names:
        g = gof[mname]
        max_corr = metrics["param_corrs"][mname]["max_corr"]
        corr_str = f"{max_corr:.3f}" if not np.isnan(max_corr) else "N/A"
        print(f"  {MODELS[mname]['label']:<20} {MODELS[mname]['n_params']:<4} "
              f"{g['mean_rv']:<12.2f} {g['median_rv']:<12.2f} "
              f"{g['mean_aicc']:<10.1f} {g['mean_bic']:<10.1f} "
              f"{g['win_pct']:<8.1f} {corr_str:<10}")

    # Parameter ranges
    print(f"\n  {'Model':<20} {'Parameter ranges'}")
    print(f"  {'-'*85}")
    for mname in model_names:
        ranges = metrics["param_ranges"][mname]
        pnames = MODELS[mname]["param_names"]
        parts = []
        for pn in pnames:
            if pn in ranges:
                lo, hi, med = ranges[pn]
                parts.append(f"{pn}=[{lo:.3g}, {hi:.3g}] med={med:.3g}")
        for k, v in ranges.items():
            if k.endswith("_at_bound"):
                parts.append(f"{k}: {v}")
        print(f"  {MODELS[mname]['label']:<20} {', '.join(parts)}")

    # Parameter correlations
    print(f"\n  Parameter Spearman correlations (off-diagonal |max|):")
    for mname in model_names:
        pcorr = metrics["param_corrs"][mname]
        if pcorr["corr_matrix"] is None:
            print(f"    {MODELS[mname]['label']:<20} insufficient data")
            continue
        pnames = pcorr["param_names"]
        mat = pcorr["corr_matrix"]
        for ii in range(len(pnames)):
            for jj in range(ii + 1, len(pnames)):
                print(f"    {MODELS[mname]['label']:<20} rho({pnames[ii]}, {pnames[jj]}) = {mat[ii, jj]:.4f}")
        print(f"    {'':>20} max |corr| = {pcorr['max_corr']:.4f}")

    # Per-dataset
    print(f"\n  -- Per-dataset mean residual variance --")
    per_ds = metrics["per_dataset"]
    for ds in sorted(per_ds.keys()):
        info = per_ds[ds]
        print(f"\n  {ds} (n={info['n']}):")
        print(f"    {'Model':<20} {'Mean RV':<12} {'Median RV':<12}")
        for mname in model_names:
            if mname in info:
                print(f"    {MODELS[mname]['label']:<20} {info[mname]['mean_rv']:<12.2f} "
                      f"{info[mname]['median_rv']:<12.2f}")

    # Recommendation
    print(f"\n  {'='*85}")
    print(f"  RECOMMENDATION:")
    # Find model with best AICc
    best_aicc = min(model_names, key=lambda m: gof[m]["mean_aicc"])
    best_corr = min(model_names, key=lambda m:
                    metrics["param_corrs"][m]["max_corr"]
                    if not np.isnan(metrics["param_corrs"][m]["max_corr"]) else 999.0)
    print(f"    Best AICc:       {MODELS[best_aicc]['label']} "
          f"({gof[best_aicc]['mean_aicc']:.1f})")
    print(f"    Best behaved:    {MODELS[best_corr]['label']} "
          f"(max |corr| = {metrics['param_corrs'][best_corr]['max_corr']:.4f})")
    print(f"    For manifold learning: centered 3PL offers the best balance of fit "
          f"quality and parameter interpretability.")
    print(f"  {'='*85}\n")


# ======================================================================
# 5. Figures
# ======================================================================

def plot_gof_comparison(metrics, merged, all_names_full):
    """Figure 1: Goodness-of-fit comparison."""
    set_style()
    fig = plt.figure(figsize=(18, 12))
    move_figure(fig, 100, 100)

    model_names = [m for m in MODEL_ORDER if m in metrics["gof"]]
    all_ok = np.ones(len(all_names_full), dtype=bool)
    for mname in model_names:
        all_ok &= merged[mname]["success"]
    n_common = all_ok.sum()

    datasets = sorted(set(all_names_full))
    names_ok = all_names_full[all_ok]
    ds_colors = {ds: DATASET_COLORS.get(ds, "#888") for ds in datasets}
    bar_colors = {"orig_3pl": "#e74c3c", "lognormal": "#f39c12",
                  "centered_3pl": "#2ecc71", "centered_2pl": "#3498db"}

    # Row 1: Win counts, AICc boxplot, BIC boxplot
    ax = fig.add_subplot(2, 3, 1)
    labels = [MODELS[m]["label"] for m in model_names]
    wins = [metrics["gof"][m]["win_pct"] for m in model_names]
    ax.bar(range(len(model_names)), wins, color=[bar_colors.get(m, "#888") for m in model_names],
           alpha=0.7, edgecolor="white")
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Win %")
    ax.set_title(f"Win percentage\n({n_common} common steps)")

    # AICc
    ax = fig.add_subplot(2, 3, 2)
    rv = {m: merged[m]["resid_var"][all_ok] for m in model_names}
    n_data = 35.0
    aicc_data = []
    for m in model_names:
        rss = rv[m] * n_data
        log_rss_n = np.log(np.maximum(rss / n_data, 1e-12))
        n_p = MODELS[m]["n_params"]
        aicc = n_data * log_rss_n + 2*n_p + 2*n_p*(n_p+1)/np.maximum(n_data-n_p-1, 1)
        aicc_data.append(aicc)
    bp = ax.boxplot(aicc_data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, m in zip(bp["boxes"], model_names):
        patch.set_facecolor(bar_colors.get(m, "#888"))
        patch.set_alpha(0.6)
    ax.set_title("AICc (lower is better)")
    ax.tick_params(axis="x", labelsize=8, rotation=20)

    # BIC
    ax = fig.add_subplot(2, 3, 3)
    bic_data = []
    for m in model_names:
        rss = rv[m] * n_data
        log_rss_n = np.log(np.maximum(rss / n_data, 1e-12))
        n_p = MODELS[m]["n_params"]
        bic = n_data * log_rss_n + n_p * np.log(n_data)
        bic_data.append(bic)
    bp = ax.boxplot(bic_data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, m in zip(bp["boxes"], model_names):
        patch.set_facecolor(bar_colors.get(m, "#888"))
        patch.set_alpha(0.6)
    ax.set_title("BIC (lower is better)")
    ax.tick_params(axis="x", labelsize=8, rotation=20)

    # Row 2: Residual variance boxplot, per-dataset bars
    ax = fig.add_subplot(2, 3, 4)
    rv_data = [rv[m] for m in model_names]
    bp = ax.boxplot(rv_data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, m in zip(bp["boxes"], model_names):
        patch.set_facecolor(bar_colors.get(m, "#888"))
        patch.set_alpha(0.6)
    ax.set_ylabel("Residual variance")
    ax.set_title(f"Residual variance (pooled, n={n_common})")
    ax.set_yscale("log")
    ax.tick_params(axis="x", labelsize=8, rotation=20)

    # Per-dataset mean RV
    ax = fig.add_subplot(2, 3, 5)
    per_ds = metrics["per_dataset"]
    x_pos = np.arange(len(datasets))
    n_models = len(model_names)
    width = 0.8 / n_models
    hatches = ["", "//", "..", "xx"]
    for i, m in enumerate(model_names):
        means = []
        for ds in datasets:
            if ds in per_ds and m in per_ds[ds]:
                means.append(per_ds[ds][m]["mean_rv"])
            else:
                means.append(np.nan)
        ax.bar(x_pos + i * width, means, width,
               label=MODELS[m]["label"], color=bar_colors.get(m, "#888"),
               alpha=0.7, edgecolor="white", hatch=hatches[i % len(hatches)])
    ax.set_xticks(x_pos + width * (n_models - 1) / 2)
    ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean residual variance")
    ax.set_title("Mean RV by dataset")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=7)

    # Per-step win count bar chart
    ax = fig.add_subplot(2, 3, 6)
    win_counts = {m: metrics["gof"][m]["win_pct"] / 100 * n_common for m in model_names}
    ax.bar(range(len(model_names)), [win_counts[m] for m in model_names],
           color=[bar_colors.get(m, "#888") for m in model_names],
           alpha=0.7, edgecolor="white")
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Number of steps won")
    ax.set_title(f"Per-step winner (n={n_common})")
    for i, m in enumerate(model_names):
        ax.text(i, win_counts[m] + max(win_counts.values()) * 0.01,
                str(int(win_counts[m])), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("figs/model_selection/gof_comparison.png", bbox_inches="tight", dpi=300)
    plt.show()
    print("  -> Saved figs/model_selection/gof_comparison.png")


def plot_parameter_correlations(metrics):
    """Figure 2: Parameter correlation heatmaps."""
    set_style()
    model_names = [m for m in MODEL_ORDER if m in metrics["param_corrs"]]
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]
    move_figure(fig, 100, 100)

    for ax, mname in zip(axes, model_names):
        pcorr = metrics["param_corrs"][mname]
        if pcorr["corr_matrix"] is None:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(MODELS[mname]["label"])
            continue
        mat = pcorr["corr_matrix"]
        pnames = pcorr["param_names"]
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(len(pnames)))
        ax.set_yticks(range(len(pnames)))
        ax.set_xticklabels(pnames, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(pnames, fontsize=9)
        ax.set_title(f"{MODELS[mname]['label']}\nmax |r|={pcorr['max_corr']:.3f}")
        # Annotate cells
        for ii in range(len(pnames)):
            for jj in range(len(pnames)):
                ax.text(jj, ii, f"{mat[ii, jj]:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if abs(mat[ii, jj]) > 0.5 else "black")
    plt.colorbar(im, ax=axes[-1], shrink=0.8, label="Spearman rho")
    plt.tight_layout()
    plt.savefig("figs/model_selection/parameter_correlations.png", bbox_inches="tight", dpi=300)
    plt.show()
    print("  -> Saved figs/model_selection/parameter_correlations.png")


def plot_parameter_distributions(metrics, merged, all_names_full):
    """Figure 3: Parameter histograms — centered_3pl (top) vs orig_3pl (bottom)."""
    set_style()
    datasets = sorted(set(all_names_full))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    move_figure(fig, 100, 100)

    # Top row: centered_3pl, bottom row: orig_3pl
    rows = [
        ("centered_3pl", ["k", "sigma_half", "gamma"], ["k (centered)", "sigma_half", "log10(gamma)"]),
        ("orig_3pl",     ["k", "log10_x0", "gamma"],   ["k (orig)", "log10_x0", "log10(gamma)"]),
    ]

    for row_idx, (mname, pnames, titles) in enumerate(rows):
        for col_idx, (pn, t) in enumerate(zip(pnames, titles)):
            ax = axes[row_idx, col_idx]
            pi = pnames.index(pn)

            for ds in datasets:
                vals = []
                for i in range(len(all_names_full)):
                    if all_names_full[i] == ds and merged[mname]["success"][i]:
                        v = merged[mname]["params"][i][pi]
                        if "gamma" in pn.lower() and "log10" in t.lower():
                            v = np.log10(max(v, 1e-6))
                        vals.append(v)

                if not vals:
                    continue
                vals_arr = np.array(vals)
                ax.hist(vals_arr, bins=25, alpha=0.45,
                        color=DATASET_COLORS.get(ds, "#888"),
                        label=ds, edgecolor="white", linewidth=0.3)

            ax.set_xlabel(t)
            ax.set_ylabel("Count")
            ax.set_title(f"{MODELS[mname]['label']}: {t}")
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, fontsize=7)

            # Add median annotation
            all_vals = []
            for i in range(len(all_names_full)):
                if merged[mname]["success"][i]:
                    v = merged[mname]["params"][i][pi]
                    if "gamma" in pn.lower() and "log10" in t.lower():
                        v = np.log10(max(v, 1e-6))
                    all_vals.append(v)
            if all_vals:
                med = np.median(all_vals)
                ax.axvline(med, color="black", linestyle="--", lw=1, alpha=0.5)
                ax.text(0.98, 0.95, f"med={med:.2f}", transform=ax.transAxes,
                        ha="right", va="top", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    plt.tight_layout()
    plt.savefig("figs/model_selection/parameter_distributions.png", bbox_inches="tight", dpi=300)
    plt.show()
    print("  -> Saved figs/model_selection/parameter_distributions.png")


def plot_fitting_examples(merged, all_names_full, raw_data_by_dataset, ds_names_sorted):
    """Figure 4: Best and worst fit examples for each model."""
    set_style()
    model_names = [m for m in MODEL_ORDER if m in merged]
    n_models = len(model_names)
    # Show 2 examples (best fit, worst fit) x n_models
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 9))
    move_figure(fig, 100, 100)

    # Find global best and worst by normalized residual
    all_ok = np.ones(len(all_names_full), dtype=bool)
    for mname in model_names:
        all_ok &= merged[mname]["success"]

    # Collect (resid_var, dataset_idx, step_idx) across all models
    candidates = []
    for i in range(len(all_names_full)):
        if not all_ok[i]:
            continue
        ds = all_names_full[i]
        for mname in model_names:
            rv = merged[mname]["resid_var"][i]
            candidates.append((rv, ds, i, mname))

    candidates.sort(key=lambda x: x[0])

    # Show the globally worst and a representative best for each model
    for col, mname in enumerate(model_names):
        model_candidates = [(rv, ds, i) for rv, ds, i, mn in candidates if mn == mname]
        if len(model_candidates) < 2:
            continue

        # Best for this model
        best_rv, best_ds, best_i = model_candidates[0]
        ax = axes[0, col]
        _plot_single_fit(ax, best_i, best_ds, mname, merged, raw_data_by_dataset,
                         f"Best: {MODELS[mname]['label']}\nRV={best_rv:.1f}")

        # Worst for this model
        worst_rv, worst_ds, worst_i = model_candidates[-1]
        ax = axes[1, col]
        _plot_single_fit(ax, worst_i, worst_ds, mname, merged, raw_data_by_dataset,
                         f"Worst: {MODELS[mname]['label']}\nRV={worst_rv:.1f}")

    plt.tight_layout()
    plt.savefig("figs/model_selection/fitting_examples.png", bbox_inches="tight", dpi=300)
    plt.show()
    print("  -> Saved figs/model_selection/fitting_examples.png")


def _plot_single_fit(ax, step_i, ds_name, mname, merged, raw_data_by_dataset, title):
    """Helper: plot one model fit on one step with data points."""
    fitted = merged[mname]["fitted"][step_i]
    test_scales = merged.get("_test_scales", [None] * (step_i + 1))[step_i]
    modes = merged.get("_modes", [None] * (step_i + 1))[step_i]
    n_agents = merged.get("_N", [None] * (step_i + 1))[step_i]

    # Plot data points
    if test_scales is not None and modes is not None:
        ax.scatter(test_scales, modes, s=15, alpha=0.4, color="gray",
                   edgecolors="none", label="Data")
        ax.plot(test_scales, modes, "-", color="gray", lw=0.5, alpha=0.3)

    # Plot fitted curve
    if fitted is not None and test_scales is not None:
        ax.plot(test_scales, fitted, "-", color="#e74c3c", lw=2, label=MODELS[mname]["label"])

    # N reference line
    if n_agents is not None:
        ax.axhline(n_agents, color="black", linestyle=":", lw=0.8, alpha=0.4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sigma")
    ax.set_ylabel("modes")
    ax.set_title(title, fontsize=9)
    ax.legend(frameon=False, fontsize=6)


def plot_centered_3pl_manifold(metrics, merged, all_names_full):
    """Figure 5: 3D scatter of centered 3PL parameters with 2D projections."""
    set_style()
    mname = "centered_3pl"
    if mname not in merged:
        print("  Skipping manifold plot (centered_3pl not available).")
        return

    all_ok = np.ones(len(all_names_full), dtype=bool)
    for mn in MODEL_ORDER:
        if mn in merged:
            all_ok &= merged[mn]["success"]

    params_list = []
    ds_list = []
    for i in range(len(all_names_full)):
        if all_ok[i] and merged[mname]["success"][i]:
            params_list.append(merged[mname]["params"][i])
            ds_list.append(all_names_full[i])

    if len(params_list) < 3:
        return

    params_arr = np.array(params_list)
    datasets = sorted(set(ds_list))
    ds_colors = {ds: DATASET_COLORS.get(ds, "#888") for ds in datasets}

    fig = plt.figure(figsize=(14, 12))
    move_figure(fig, 100, 100)

    # 3D scatter
    ax_3d = fig.add_subplot(2, 2, 1, projection="3d")
    for ds in datasets:
        mask = np.array(ds_list) == ds
        ax_3d.scatter(params_arr[mask, 0], params_arr[mask, 1], params_arr[mask, 2],
                      c=ds_colors[ds], label=ds, s=15, alpha=0.7, edgecolors="none")
    ax_3d.set_xlabel("k")
    ax_3d.set_ylabel("sigma_half")
    ax_3d.set_zlabel("gamma")
    ax_3d.set_title("Centered 3PL parameter space")
    ax_3d.legend(frameon=False, fontsize=7)

    # 2D projections
    pairs = [(0, 1, "k", "sigma_half"), (0, 2, "k", "gamma"), (1, 2, "sigma_half", "gamma")]
    for idx, (pi, pj, xl, yl) in enumerate(pairs):
        ax = fig.add_subplot(2, 2, idx + 2)
        for ds in datasets:
            mask = np.array(ds_list) == ds
            ax.scatter(params_arr[mask, pi], params_arr[mask, pj],
                       c=ds_colors[ds], label=ds, s=12, alpha=0.6, edgecolors="none")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        if idx == 0:
            ax.set_yscale("log")
            ax.legend(frameon=False, fontsize=6)

    plt.tight_layout()
    plt.savefig("figs/model_selection/centered_3pl_manifold.png", bbox_inches="tight", dpi=300)
    plt.show()
    print("  -> Saved figs/model_selection/centered_3pl_manifold.png")


def plot_parameter_vs_time(merged, all_names_full, comp_by_dataset):
    """Figure 6: Parameter trajectories over time per dataset."""
    set_style()
    model_name = "centered_3pl"
    if model_name not in merged:
        return

    ds_names = sorted(comp_by_dataset.keys())
    pnames = MODELS[model_name]["param_names"]
    n_p = len(pnames)

    fig, axes = plt.subplots(len(ds_names), n_p, figsize=(5 * n_p, 4 * len(ds_names)),
                             squeeze=False)
    move_figure(fig, 100, 100)

    for row, ds in enumerate(ds_names):
        comp = comp_by_dataset[ds]
        n_steps = len(comp["_step_range"])
        step_nums = np.array(comp["_step_range"])

        for col in range(n_p):
            ax = axes[row, col]
            vals = []
            for i in range(n_steps):
                if comp[model_name]["success"][i]:
                    vals.append(comp[model_name]["params"][i][col])
                else:
                    vals.append(np.nan)
            ax.plot(step_nums, vals, "o-", markersize=4, lw=1,
                    color=DATASET_COLORS.get(ds, "#888"))
            ax.set_xlabel("Time step")
            ax.set_ylabel(pnames[col])
            ax.set_title(f"{ds}: {pnames[col]}")
            if pnames[col] == "sigma_half":
                ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig("figs/model_selection/parameter_vs_time.png", bbox_inches="tight", dpi=300)
    plt.show()
    print("  -> Saved figs/model_selection/parameter_vs_time.png")


def plot_bootstrap_stability(merged, all_names_full):
    """Figure 7: Bootstrap CV for each model/parameter."""
    set_style()
    model_names = [m for m in MODEL_ORDER if m in merged]

    # Collect CV estimates (we'll approximate from a few refits)
    # For efficiency, just show parameter ranges as a proxy
    all_ok = np.ones(len(all_names_full), dtype=bool)
    for m in model_names:
        all_ok &= merged[m]["success"]

    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 4.5),
                             squeeze=False)
    move_figure(fig, 100, 100)

    for col, mname in enumerate(model_names):
        ax = axes[0, col]
        pnames = MODELS[mname]["param_names"]
        n_p = len(pnames)

        # Compute normalized spread (IQR/median) for each parameter
        spreads = []
        labels = []
        for pi in range(n_p):
            vals = []
            for i in range(len(all_names_full)):
                if all_ok[i] and merged[mname]["success"][i]:
                    vals.append(merged[mname]["params"][i][pi])
            if len(vals) > 1:
                vals_arr = np.array(vals)
                # Normalized inter-quartile range
                q25, q75 = np.percentile(vals_arr, [25, 75])
                med = np.median(vals_arr)
                if med != 0:
                    spread = (q75 - q25) / abs(med)
                else:
                    spread = q75 - q25 if q75 > q25 else 0
                spreads.append(spread)
                labels.append(pnames[pi])

        if spreads:
            ax.bar(range(len(spreads)), spreads, color="#3498db", alpha=0.7, edgecolor="white")
            ax.set_xticks(range(len(spreads)))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel("IQR / median")
            ax.set_title(f"{MODELS[mname]['label']}\nparameter spread")

    plt.tight_layout()
    plt.savefig("figs/model_selection/bootstrap_stability.png", bbox_inches="tight", dpi=300)
    plt.show()
    print("  -> Saved figs/model_selection/bootstrap_stability.png")


# ======================================================================
# 6. Main
# ======================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Model selection for mode-count curves")
    parser.add_argument("--skip-fit", action="store_true",
                        help="Skip fitting, reload from cached .npz")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip plt.show(), save figures to disk only")
    parser.add_argument("--models", type=str, default="all",
                        help="Comma-separated model names to fit (default: all)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed figure diagnostics")
    parser.add_argument("--saturation", type=float, default=0.7,
                        help="Trim plateau where modes > saturation * N (default: 0.7)")
    args = parser.parse_args()
    if args.no_display:
        plt.show = lambda: None
        print("[--no-display] Suppressing figure windows.\n")

    # Determine which models to use
    if args.models == "all":
        model_names = list(MODEL_ORDER)
    else:
        model_names = [m.strip() for m in args.models.split(",")]

    cache_path = "data_scaling_law/model_selection_results.npz"

    # --- Step 1: Load data & fit models ---
    if args.skip_fit and os.path.exists(cache_path):
        print("Loading cached fits from", cache_path)
        cached = np.load(cache_path, allow_pickle=True)
        all_fits_by_ds = cached["all_fits_by_ds"].item()
        raw_data_by_dataset = cached["raw_data_by_dataset"].item()
        ds_names_sorted = list(all_fits_by_ds.keys())
        all_names_for_comp = cached["all_names_for_comp"].tolist()
    else:
        all_fits_by_ds = {}
        raw_data_by_dataset = {}
        all_names_for_comp = []

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

            print(f"  Fitting {len(model_names)} models for {len(step_range)} time steps "
                  f"(saturation={args.saturation})...")
            results = fit_all_models_all_steps(
                step_range, N_array, scale_range, all_modes, model_names,
                saturation=args.saturation,
            )
            all_fits_by_ds[name] = results

            n_ok = {m: results[m]["success"].sum() for m in model_names}
            print(f"  Successful fits: {n_ok}")

            for _ in range(len(step_range)):
                all_names_for_comp.append(name)

        ds_names_sorted = sorted(all_fits_by_ds.keys())
        all_names_for_comp = [name for name in ds_names_sorted
                              for _ in range(len(all_fits_by_ds[name][model_names[0]]["success"]))]

        # Save cache
        np.savez_compressed(cache_path,
                            all_fits_by_ds=all_fits_by_ds,
                            raw_data_by_dataset=raw_data_by_dataset,
                            all_names_for_comp=np.array(all_names_for_comp))
        print(f"  -> Cached fits to {cache_path}")

    all_names_full = np.array(all_names_for_comp)

    # Build merged comparison across all datasets
    merged = {}
    for mname in model_names:
        merged[mname] = {
            "params":    sum([all_fits_by_ds[n][mname]["params"] for n in ds_names_sorted], []),
            "fitted":    sum([all_fits_by_ds[n][mname]["fitted"] for n in ds_names_sorted], []),
            "resid_var": np.concatenate([all_fits_by_ds[n][mname]["resid_var"] for n in ds_names_sorted]),
            "success":   np.concatenate([all_fits_by_ds[n][mname]["success"] for n in ds_names_sorted]),
        }
    # Also merge raw data for plotting
    merged["_test_scales"] = sum([all_fits_by_ds[n]["_test_scales"] for n in ds_names_sorted], [])
    merged["_modes"] = sum([all_fits_by_ds[n]["_modes"] for n in ds_names_sorted], [])
    merged["_N"] = sum([all_fits_by_ds[n]["_N"] for n in ds_names_sorted], [])

    # Build comp_by_dataset for time-series plots (include step_range info)
    comp_by_dataset = {}
    for name in ds_names_sorted:
        comp = dict(all_fits_by_ds[name])
        comp["_step_range"] = raw_data_by_dataset[name]["step_range"]
        comp["_N_array"] = raw_data_by_dataset[name]["N_array"]
        comp_by_dataset[name] = comp

    # --- Step 2: Compute metrics ---
    print(f"\n{'='*60}")
    print("Computing metrics...")
    print(f"{'='*60}")
    metrics = compute_all_metrics(all_fits_by_ds, all_names_full, merged)

    # --- Step 3: Console report ---
    print_comparison_report(metrics)

    # --- Step 4: Figures ---
    print(f"\n{'='*60}")
    print("Generating figures...")
    print(f"{'='*60}")

    n_total_steps = len(all_names_full)
    all_ok_all = np.ones(n_total_steps, dtype=bool)
    for mname in model_names:
        all_ok_all &= merged[mname]["success"]

    if args.verbose:
        print(f"  Total steps: {n_total_steps}, common-fit: {all_ok_all.sum()}")
        for mname in model_names:
            n_ok = merged[mname]["success"].sum()
            print(f"    {MODELS[mname]['label']:<20} {n_ok}/{n_total_steps} fits succeeded")
        print(f"  Raw data available: {len(merged.get('_test_scales', []))} steps")
        print(f"  Datasets: {ds_names_sorted}")

    plot_gof_comparison(metrics, merged, all_names_full)
    if args.verbose:
        print("  gof_comparison: AICc, BIC, RV boxplots + per-dataset bars + win counts")

    plot_parameter_correlations(metrics)
    if args.verbose:
        print("  parameter_correlations: Spearman correlation heatmaps (2x2 grid)")

    plot_parameter_distributions(metrics, merged, all_names_full)
    if args.verbose:
        print("  parameter_distributions: orig_3pl vs centered_3pl parameter histograms")

    plot_fitting_examples(merged, all_names_full, raw_data_by_dataset, ds_names_sorted)
    if args.verbose:
        print("  fitting_examples: best/worst fits for each model with data points")

    plot_centered_3pl_manifold(metrics, merged, all_names_full)
    if args.verbose:
        print("  centered_3pl_manifold: 3D (k, sigma_half, gamma) scatter + 2D projections")

    plot_parameter_vs_time(merged, all_names_full, comp_by_dataset)
    if args.verbose:
        print("  parameter_vs_time: per-dataset parameter trajectories over time steps")

    plot_bootstrap_stability(merged, all_names_full)
    if args.verbose:
        print("  bootstrap_stability: IQR/median parameter spread per model")

    # --- Step 5: Diagnostic: investigate worst centered_3pl fit ---
    if "centered_3pl" in merged:
        rv_arr = merged["centered_3pl"]["resid_var"].copy()
        rv_arr[~merged["centered_3pl"]["success"]] = -1
        worst_idx = np.argmax(rv_arr)
        worst_rv = rv_arr[worst_idx]
        worst_ds = all_names_full[worst_idx]
        worst_params = merged["centered_3pl"]["params"][worst_idx]
        test_scales = merged["_test_scales"][worst_idx]
        modes = merged["_modes"][worst_idx]
        N = merged["_N"][worst_idx]
        fitted = merged["centered_3pl"]["fitted"][worst_idx]

        print(f"\n{'='*60}")
        print(f"Worst centered_3PL fit diagnostic")
        print(f"{'='*60}")
        print(f"  Dataset: {worst_ds}, step index (global): {worst_idx}")
        print(f"  N agents: {N}")
        print(f"  Residual variance: {worst_rv:.1f}")
        print(f"  Fitted params: k={worst_params[0]:.4f}, sigma_half={worst_params[1]:.4f}, gamma={worst_params[2]:.2f}")
        print(f"  Scale range: [{test_scales[0]:.4f}, {test_scales[-1]:.4f}]")

        # Find the saturation point
        saturation = args.saturation
        begin_idx = int(np.argmax(modes <= saturation * N))
        if begin_idx == 0:
            begin_idx = max(1, int(np.argmax(modes <= min(saturation + 0.1, 0.99) * N)))
        print(f"  Saturation threshold: {saturation}*N = {saturation*N:.1f}")
        print(f"  First scale below {saturation}*N: idx={begin_idx}, sigma={test_scales[begin_idx]:.4f}, modes={modes[begin_idx]:.1f}")
        print(f"  Plateau length: {begin_idx}/40 data points trimmed")

        # Check if plateau extends further
        n_plateau_check = int(np.argmax(modes <= 0.95 * N))
        print(f"  First scale below 0.95*N: idx={n_plateau_check}, sigma={test_scales[n_plateau_check]:.4f}")
        n_half_check = int(np.argmax(modes <= 0.5 * N))
        print(f"  First scale below 0.5*N:  idx={n_half_check}, sigma={test_scales[n_half_check]:.4f}")
        print(f"  Data points used for fit: {40 - begin_idx}")

        # Print all data points for manual inspection
        print(f"\n  Full data (sigma, modes, fitted):")
        print(f"  {'idx':<5} {'sigma':<12} {'modes':<10} {'fitted':<10} {'modes/N':<10}")
        for i in range(40):
            print(f"  {i:<5} {test_scales[i]:<12.4f} {modes[i]:<10.1f} {fitted[i]:<10.1f} {modes[i]/N:<10.3f}")

    print(f"\nDone. Figures saved to figs/model_selection/")


if __name__ == "__main__":
    main()
