"""
Parameter Manifold Investigation: classify spatial configurations of biological swarms
via the centered 3PL model.

Model:  m(sigma) = 1 + (N-1) / (1 + (2^(1/gamma)-1) * (sigma/sigma_half)^k)^gamma

Parameters (all compact and well-behaved):
  k           — steepness at the half-mode scale
  sigma_half  — scale where half the modes have merged
  log10_gamma — asymmetry (log10 space), gamma = 10^log10_gamma

Pipeline: fit -> PCA -> t-SNE -> UMAP -> cluster -> classify

Usage:
  python experiments/parameter_manifold.py              # full run
  python experiments/parameter_manifold.py --no-display  # headless, figures to disk
"""

import sys, os
from pathlib import Path

import numpy as np
from scipy.special import expm1
from tqdm import tqdm
import warnings
import torch

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dfr import load_dataset
from dfr.mode_finding import mode_counting, mode_counting_modified, find_scale_interval
from dfr.utils import move_figure
from dfr.analysis import (
    PARAMETER_NAMES,
    add_managed_output_arguments,
    centered_3pl_excess,
    create_analysis_artifacts,
    fit_centered_3pl_curves,
    fit_shape_curve,
    load_legacy_manifold_cache,
    median_nearest_neighbour_distance,
    project_to_shape_curve,
)
from dfr.plotting import apply_academic_style, apply_figure_layout, save_figure


# ======================================================================
# 0. Model
# ======================================================================

model_centered_3pl = centered_3pl_excess  # compatibility for research imports
compute_avg_nn_dist = median_nearest_neighbour_distance
PARAM_NAMES = list(PARAMETER_NAMES)
_FIGURE_DIR = Path("figs")


def _save_figure(filename: str, *, dpi: int = 300) -> Path:
    target = _FIGURE_DIR / filename
    return save_figure(plt.gcf(), target, bbox_inches="tight", dpi=dpi)


def _finish_layout() -> None:
    apply_figure_layout(plt.gcf())


# ======================================================================
# 1. Data loading
# ======================================================================

DATASET_RUNS = [
    {"name": "swift",    "start_step": 0,    "end_step": None, "step_length": 20,  "min_N": 50},
    {"name": "starling", "start_step": 0,    "end_step": None, "step_length": 1,   "min_N": 50},
    {"name": "jackdaw",  "start_step": 350,  "end_step": 550,  "step_length": 1,   "min_N": 50},
    {"name": "jackdaw2", "start_step": 2700, "end_step": 3460, "step_length": 5,   "min_N": 50},
]

DATASET_COLORS = {
    "starling": "#2c3e50", "jackdaw": "#e67e22",
    "jackdaw2": "#27ae60", "swift": "#8e44ad",
}


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


def load_cached_data(run_params, project_root=None):
    """Load cached modes.npy, scale_range.npy, and nn_dists.npy for a dataset.
    If cache is missing, compute the scaling law on-the-fly and save it.
    """
    name = run_params["name"]
    num_test_scale = 40
    save_every = 50
    nn_dists = None  # populated if cache exists or computed fresh
    root = Path(project_root or Path.cwd()).expanduser().resolve()
    sp = root / "scenarios" / name
    os.makedirs(sp, exist_ok=True)

    mp = os.path.join(sp, "modes.npy")
    srp = os.path.join(sp, "scale_range.npy")
    nnp = os.path.join(sp, "nn_dists.npy")

    dataset = load_dataset(name, project_root=root)

    max_steps = dataset.trajectories.shape[0]
    end = run_params["end_step"]
    eff_end = end if end is not None and end <= max_steps else max_steps
    step_range = list(range(run_params["start_step"], eff_end, run_params["step_length"]))

    # Filter by minimum N (remove frames with too few agents for meaningful mode-counting)
    min_N = run_params.get("min_N", 0)
    if min_N > 0:
        n_before = len(step_range)
        step_range = [s for s in step_range if dataset.positions_at_time_step(s).shape[0] >= min_N]
        if len(step_range) < n_before:
            print(f"  [FILTER] min_N={min_N}: kept {len(step_range)}/{n_before} steps")

    cache_exists = os.path.exists(mp) and os.path.exists(srp)

    # Detect incomplete (partial) saves: rows that are all zero are unprocessed
    if cache_exists:
        existing_cache = load_legacy_manifold_cache(sp)
        existing_modes = existing_cache.mode_counts
        existing_scales = existing_cache.scale_ranges
        # A valid mode count is never 0 — find last complete row
        row_sums = existing_modes.sum(axis=1)
        zero_rows = np.where(row_sums == 0)[0]
        if len(zero_rows) > 0:
            last_complete = zero_rows[0]  # first all-zero row
            print(f"  [RESUME] Partial cache found: {last_complete}/{len(step_range)} steps complete")
            if last_complete == 0:
                cache_exists = False  # nothing usable, start fresh
            # If only a few rows missing, we'll resume Phase 2 below
        elif existing_modes.shape[0] != len(step_range):
            print(f"  [INFO] Cache shape mismatch, recomputing")
            cache_exists = False

    if not cache_exists:
        print(f"  [CACHE MISS] Computing scaling law for '{name}' on-the-fly ({len(step_range)} steps)...")
        # --- Phase 1: find scale_range for each time step ---
        scale_range = np.zeros((len(step_range), 2))
        for i, s in enumerate(tqdm(step_range, desc=f"  Scale range [{name}]")):
            pos_np = dataset.positions_at_time_step(s)
            pos = torch.from_numpy(pos_np).cuda().float()
            N = pos.shape[0]
            avg_nn_dist = compute_avg_nn_dist(pos_np)
            tol = max(avg_nn_dist * 1e-3, 1e-8)

            def f(sc):
                if sc < avg_nn_dist * 0.1:
                    return N
                return mode_counting(pos, pos.clone(), sc, max_iter=200, tol=tol)
            s_start, s_end = find_scale_interval(
                f, N, s_initial_guess=avg_nn_dist * 5, atol=max(avg_nn_dist * 1e-2, 1e-8))
            s_start = max(s_start, max(avg_nn_dist * 1e-2, 1e-8))
            scale_range[i] = [s_start, s_end]

        np.save(srp, scale_range)

        # --- Phase 2: mode counting at each scale for each time step ---
        all_modes = np.zeros((len(step_range), num_test_scale))
        for i, s in enumerate(tqdm(step_range, desc=f"  Mode counting [{name}]")):
            pos_np = dataset.positions_at_time_step(s)
            pos = torch.from_numpy(pos_np).cuda().float()
            N = pos.shape[0]
            avg_nn_dist = compute_avg_nn_dist(pos_np)
            tol = max(avg_nn_dist * 1e-3, 1e-8)

            s_start, s_end = scale_range[i]

            # Extend s_start downward if plateau is truncated
            probe_s = float(s_start)
            probe_modes, _ = mode_counting_modified(pos, pos.clone(), probe_s,
                                                      max_iter=200, tol=tol)
            if probe_modes < 0.9 * N:
                for _ in range(10):
                    s_start /= 2.0
                    probe_modes, _ = mode_counting_modified(pos, pos.clone(), s_start,
                                                              max_iter=200, tol=tol)
                    if probe_modes >= 0.95 * N:
                        break
                s_start = max(s_start, max(avg_nn_dist * 1e-2, 1e-8))
                scale_range[i, 0] = s_start

            test_scales = np.logspace(np.log10(max(s_start, 1e-12)),
                                      np.log10(max(s_end, 1e-11)), num_test_scale)

            modes_pos = None
            prev_mode_num = N
            for j, sc in enumerate(test_scales):
                if sc < avg_nn_dist * 0.5:
                    all_modes[i, j] = N
                    continue
                if prev_mode_num <= 1:
                    all_modes[i, j] = 1
                    continue

                if prev_mode_num > 0.95 * N:
                    mi = 100
                elif prev_mode_num > 0.5 * N:
                    mi = 200
                else:
                    mi = 400

                curr_pos = modes_pos.clone() if modes_pos is not None else pos.clone()
                mode_num, tmp = mode_counting_modified(pos, curr_pos, sc,
                                                        max_iter=mi, tol=tol)
                modes_pos = torch.from_numpy(tmp).cuda().float()
                all_modes[i, j] = mode_num
                prev_mode_num = mode_num

            # Incremental save every `save_every` steps (and on final step)
            if (i + 1) % save_every == 0 or i == len(step_range) - 1:
                np.save(mp, all_modes)
                np.save(srp, scale_range)
                if (i + 1) % save_every == 0:
                    print(f"  [partial save] {i+1}/{len(step_range)} steps", end="\r")

        np.save(mp, all_modes)
        np.save(srp, scale_range)  # final save (re-save in case s_start was extended)

        # Also cache nn_dists for downstream analysis (sigma_half vs physical spacing)
        if not os.path.exists(nnp):
            nn_dists = np.array([compute_avg_nn_dist(dataset.positions_at_time_step(s))
                                 for s in tqdm(step_range, desc=f"  NN dist [{name}]")])
            np.save(nnp, nn_dists)
        else:
            nn_dists = np.load(nnp, allow_pickle=False)

        print(f"\n  [CACHE SAVED] {name}: modes={all_modes.shape}, scales={scale_range.shape}")

    else:
        cached = load_legacy_manifold_cache(sp)
        all_modes = cached.mode_counts
        scale_range = cached.scale_ranges
        nn_dists = cached.nearest_neighbour_distances
        if nn_dists is None:
            print(f"  Computing NN distances for '{name}' ({len(step_range)} steps)...")
            nn_dists = np.array([compute_avg_nn_dist(dataset.positions_at_time_step(s))
                                 for s in tqdm(step_range, desc=f"  NN dist [{name}]")])
            np.save(nnp, nn_dists)

        # Detect partial save: rows with all zeros are unprocessed
        row_sums = all_modes.sum(axis=1)
        zero_rows = np.where(row_sums == 0)[0]
        if len(zero_rows) > 0 and zero_rows[0] > 0:
            last_complete = zero_rows[0]
            print(f"  [RESUME] Partial cache: {last_complete}/{len(step_range)} steps done, resuming Phase 2")
            # Truncate to completed rows
            step_range_done = step_range[:last_complete]
            all_modes_done = all_modes[:last_complete]

            # Resume Phase 2 from last_complete onwards
            resume_start = last_complete
            for i, s in enumerate(tqdm(step_range[resume_start:], desc=f"  Mode counting [{name}] (resume)")):
                i_abs = resume_start + i
                pos_np = dataset.positions_at_time_step(s)
                pos = torch.from_numpy(pos_np).cuda().float()
                N = pos.shape[0]
                avg_nn_dist = compute_avg_nn_dist(pos_np)
                tol = max(avg_nn_dist * 1e-3, 1e-8)
                s_start, s_end = scale_range[i_abs]

                test_scales = np.logspace(np.log10(max(s_start, 1e-12)),
                                          np.log10(max(s_end, 1e-11)), num_test_scale)
                modes_pos = None; prev_mode_num = N
                for j, sc in enumerate(test_scales):
                    if sc < avg_nn_dist * 0.5:
                        all_modes[i_abs, j] = N; continue
                    if prev_mode_num <= 1:
                        all_modes[i_abs, j] = 1; continue
                    if prev_mode_num > 0.95 * N: mi = 100
                    elif prev_mode_num > 0.5 * N: mi = 200
                    else: mi = 400
                    curr_pos = modes_pos.clone() if modes_pos is not None else pos.clone()
                    mode_num, tmp = mode_counting_modified(pos, curr_pos, sc, max_iter=mi, tol=tol)
                    modes_pos = torch.from_numpy(tmp).cuda().float()
                    all_modes[i_abs, j] = mode_num
                    prev_mode_num = mode_num

                if (i_abs + 1) % save_every == 0 or i_abs == len(step_range) - 1:
                    np.save(mp, all_modes)
                    np.save(srp, scale_range)
                    if (i_abs + 1) % save_every == 0:
                        print(f"  [partial save] {i_abs+1}/{len(step_range)} steps", end="\r")

            np.save(mp, all_modes)
            np.save(srp, scale_range)
            print(f"\n  [CACHE SAVED] {name}: modes={all_modes.shape}, scales={scale_range.shape}")

        elif all_modes.shape[0] != scale_range.shape[0] or scale_range.shape[0] != len(step_range):
            print(f"  [INFO] Shape mismatch, recomputing would be needed")
            n_eff = min(all_modes.shape[0], scale_range.shape[0], len(step_range))
            step_range = step_range[:n_eff]
            scale_range = scale_range[:n_eff]
            all_modes = all_modes[:n_eff]

    print(f"  Loading N for {name} ({len(step_range)} steps)...")
    N_array = np.array([dataset.positions_at_time_step(s).shape[0] for s in tqdm(step_range)])

    return step_range, N_array, scale_range, all_modes, nn_dists


# ======================================================================
# 2. Fitting
# ======================================================================

def fit_all_steps(step_range, N_array, scale_range, all_modes,
                  saturation=0.8, num_test_scale=40):
    """Compatibility adapter over :func:`dfr.analysis.fit_centered_3pl_curves`."""
    if all_modes.shape[1] != num_test_scale:
        raise ValueError("num_test_scale must match the cached mode-count width.")
    batch = fit_centered_3pl_curves(
        step_range,
        N_array,
        scale_range,
        all_modes,
        saturation=saturation,
    )
    params = [None] * len(step_range)
    for source_index, values in zip(
        np.flatnonzero(batch.success), batch.result.parameters
    ):
        params[source_index] = values
    return {
        "params": params,
        "fitted": list(batch.fitted_curves),
        "resid_var": batch.residual_variances,
        "success": batch.success,
        "test_scales": batch.scale_grids[-1],
    }


# ======================================================================
# 3. Manifold learning
# ======================================================================

def run_manifold_learning(features):
    """PCA -> t-SNE -> UMAP on standardized features."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    print("  Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(features) - 1),
                random_state=42, init="pca", learning_rate="auto")
    X_tsne = tsne.fit_transform(X_scaled)

    print("  Running UMAP...")
    try:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=min(15, len(features) - 1),
                            min_dist=0.1, random_state=42)
        X_umap = reducer.fit_transform(X_scaled)
        has_umap = True
    except ImportError:
        print("  [WARN] umap-learn not installed")
        X_umap = np.zeros((len(features), 2))
        has_umap = False

    return {"pca": X_pca, "tsne": X_tsne, "umap": X_umap,
            "has_umap": has_umap, "pca_model": pca, "scaler": scaler}


# ======================================================================
# 4. Clustering
# ======================================================================

def run_clustering(embedding):
    """HDBSCAN on UMAP embedding, fallback to GMM."""
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3,
                                    cluster_selection_epsilon=0.5, metric="euclidean")
        labels = clusterer.fit_predict(embedding)
        return labels, "HDBSCAN"
    except ImportError:
        from sklearn.mixture import GaussianMixture
        bics, models = [], []
        for n in range(2, 7):
            gmm = GaussianMixture(n_components=n, random_state=42, covariance_type="full")
            gmm.fit(embedding)
            bics.append(gmm.bic(embedding))
            models.append(gmm)
        best = models[np.argmin(bics)]
        print(f"  Best GMM: {np.argmin(bics) + 2} components")
        return best.predict(embedding), "GMM"


# ======================================================================
# 5. Plotting
# ======================================================================

def plot_pca_scree(pca_model, scaler):
    """PCA scree plot with component loadings."""
    set_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.5))
    move_figure(fig, 100, 100)

    ev = pca_model.explained_variance_ratio_
    cumsum = np.cumsum(ev)
    comps = np.arange(1, len(ev) + 1)

    ax0.bar(comps, ev * 100, color="#3498db", alpha=0.7, edgecolor="white")
    ax0.set_xlabel("Principal component"); ax0.set_ylabel("Explained variance (%)")
    ax0.set_title("PCA scree plot"); ax0.set_xticks(comps)

    ax1.plot(comps, cumsum * 100, "o-", color="#2c3e50", lw=2, markersize=8)
    ax1.axhline(90, color="#e74c3c", ls="--", lw=1, alpha=0.5, label="90%")
    ax1.axhline(95, color="#e74c3c", ls=":", lw=1, alpha=0.5, label="95%")
    ax1.set_xlabel("Number of components"); ax1.set_ylabel("Cumulative variance (%)")
    ax1.set_title("Cumulative explained variance"); ax1.legend(); ax1.set_xticks(comps)

    # Console: component loadings
    print(f"\n  PCA loadings ({len(ev)} components):")
    for i, (vp, coefs) in enumerate(zip(ev * 100, pca_model.components_)):
        terms = " + ".join(f"({c:+.4f})*z_{n}" for c, n in zip(coefs, PARAM_NAMES))
        print(f"    PC{i+1} ({vp:.1f}%):  {terms}")
    print(f"\n  StandardScaler:")
    for n, mu, s in zip(PARAM_NAMES, scaler.mean_, scaler.scale_):
        print(f"    z_{n} = ({n} - {mu:.4f}) / {s:.4f}")

    _finish_layout()
    _save_figure("manifold_pca_scree.png")
    plt.show()
    print(f"  -> Saved {_FIGURE_DIR / 'manifold_pca_scree.png'}")


def plot_embeddings(embeddings, names, N_array, labels):
    """t-SNE and UMAP embeddings colored by dataset, N, and cluster."""
    set_style()
    has_umap = embeddings["has_umap"]
    n_embed = 2 if has_umap else 1
    fig, axes = plt.subplots(3, n_embed, figsize=(6.5 * n_embed, 15), squeeze=False)
    move_figure(fig, 100, 100)

    keys = ["tsne", "umap"] if has_umap else ["tsne"]
    titles = {"tsne": "t-SNE", "umap": "UMAP"}

    for col, key in enumerate(keys):
        X = embeddings[key]

        # By dataset
        ax = axes[0, col]
        for ds in sorted(set(names)):
            m = names == ds
            ax.scatter(X[m, 0], X[m, 1], c=DATASET_COLORS[ds], label=ds,
                       s=20, alpha=0.7, edgecolors="none")
        ax.set_title(f"{titles[key]} — by dataset")
        ax.legend(frameon=False, markerscale=2, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

        # By N
        ax = axes[1, col]
        sc = ax.scatter(X[:, 0], X[:, 1], c=N_array, cmap="plasma",
                        s=20, alpha=0.7, edgecolors="none")
        ax.set_title(f"{titles[key]} — by N")
        plt.colorbar(sc, ax=ax, label="N")
        ax.set_xticks([]); ax.set_yticks([])

        # By cluster
        ax = axes[2, col]
        ul = sorted(set(labels))
        nc = len([l for l in ul if l >= 0])
        cmap = cm.tab10 if nc <= 10 else cm.tab20
        for lbl in ul:
            m = labels == lbl
            c = cmap(lbl % 10) if lbl >= 0 else "#bdc3c7"
            ls = f"Cluster {lbl}" if lbl >= 0 else "Noise"
            ax.scatter(X[m, 0], X[m, 1], c=[c], label=ls, s=20, alpha=0.7, edgecolors="none")
        ax.set_title(f"{titles[key]} — by cluster ({nc} clusters)")
        ax.legend(frameon=False, markerscale=2, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

    for col in range(n_embed, axes.shape[1]):
        for row in range(3):
            axes[row, col].set_visible(False)

    _finish_layout()
    _save_figure("manifold_embeddings.png")
    plt.show()
    print(f"  -> Saved {_FIGURE_DIR / 'manifold_embeddings.png'}")


def plot_parameter_space(all_params, all_names):
    """3D scatter + 2D projections of the parameter manifold."""
    set_style()
    datasets = sorted(set(all_names))

    fig = plt.figure(figsize=(14, 12))
    move_figure(fig, 100, 100)

    # 3D
    ax3 = fig.add_subplot(2, 2, 1, projection="3d")
    for ds in datasets:
        m = all_names == ds
        ax3.scatter(all_params[m, 0], all_params[m, 1], all_params[m, 2],
                    c=DATASET_COLORS[ds], label=ds, s=15, alpha=0.7, edgecolors="none")
    ax3.set_xlabel("k"); ax3.set_ylabel("sigma_half"); ax3.set_zlabel("log10_gamma")
    ax3.set_title("Parameter manifold (3D)"); ax3.legend(frameon=False, fontsize=7)

    # 2D projections
    pairs = [(0, 1, "k", "sigma_half"), (0, 2, "k", "log10_gamma"),
             (1, 2, "sigma_half", "log10_gamma")]
    for idx, (pi, pj, xl, yl) in enumerate(pairs):
        ax = fig.add_subplot(2, 2, idx + 2)
        for ds in datasets:
            m = all_names == ds
            ax.scatter(all_params[m, pi], all_params[m, pj],
                       c=DATASET_COLORS[ds], label=ds, s=12, alpha=0.6, edgecolors="none")
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        if idx == 0:
            ax.set_yscale("log"); ax.legend(frameon=False, fontsize=6)

    _finish_layout()
    _save_figure("manifold_parameter_space.png")
    plt.show()
    print(f"  -> Saved {_FIGURE_DIR / 'manifold_parameter_space.png'}")


def plot_cluster_curves(all_params, all_N, all_names, labels):
    """Per-cluster mean curve with std band."""
    set_style()
    ul = sorted(set(labels))
    nc = len([l for l in ul if l >= 0])
    if nc == 0:
        return

    sigma_shared = np.logspace(-1.5, 1.5, 200)
    cmap = cm.tab10 if nc <= 10 else cm.tab20

    fig, axes = plt.subplots(1, nc, figsize=(5.5 * nc, 4.5), squeeze=False)
    move_figure(fig, 100, 100)

    for idx, lbl in enumerate([l for l in ul if l >= 0]):
        ax = axes[0, idx]
        mask = labels == lbl
        n_members = mask.sum()

        curves = []
        for i in np.where(mask)[0]:
            k, sh, lg = all_params[i]
            g = 10.0 ** lg
            scaling = expm1(np.log(2.0) / max(g, 1e-6))
            scaling_s = max(scaling, 1e-12)
            lr = np.clip(k * np.log(np.maximum(sigma_shared / sh, 1e-12)) + np.log(scaling_s),
                         -500, 500)
            c = 1.0 + (all_N[i] - 1.0) / np.power(1.0 + np.exp(lr), g)
            curves.append(c)
            if len(curves) <= 50:
                ax.plot(sigma_shared, c, color=cmap(lbl % 10), lw=0.5, alpha=0.15)

        if n_members > 1:
            mc = np.mean(curves, axis=0)
            sc = np.std(curves, axis=0)
            ax.fill_between(sigma_shared, mc - sc, mc + sc, color=cmap(lbl % 10), alpha=0.2)
            ax.plot(sigma_shared, mc, color=cmap(lbl % 10), lw=2.5)

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("sigma"); ax.set_ylabel("# Modes")
        ax.set_title(f"Cluster {lbl} (n={n_members})")

        ds_counts = {n: (all_names[mask] == n).sum() for n in sorted(set(all_names))}
        ds_str = ", ".join(f"{n}:{c}" for n, c in ds_counts.items() if c > 0)
        ax.text(0.95, 0.05, ds_str, transform=ax.transAxes, fontsize=7,
                ha="right", va="bottom", color="gray")

    _finish_layout()
    _save_figure("manifold_cluster_curves.png")
    plt.show()
    print(f"  -> Saved {_FIGURE_DIR / 'manifold_cluster_curves.png'}")


def plot_param_distributions(all_params, all_names, labels):
    """Parameter histograms by cluster."""
    set_style()
    ul = sorted(set(labels))
    nc = len([l for l in ul if l >= 0])
    if nc == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    move_figure(fig, 100, 100)
    cmap_arr = cm.tab10 if nc <= 10 else cm.tab20

    for col in range(3):
        ax = axes[col]
        for lbl in ul:
            m = labels == lbl
            vals = all_params[m, col]
            if lbl >= 0:
                ax.hist(vals, bins=25, alpha=0.5, color=cmap_arr(lbl % 10),
                        label=f"Cluster {lbl}" if col == 0 else "_nolegend_")
            else:
                ax.hist(vals, bins=25, alpha=0.2, color="#bdc3c7",
                        label="Noise" if col == 0 else "_nolegend_")
        ax.set_xlabel(PARAM_NAMES[col])
        if col == 0:
            ax.set_ylabel("Count"); ax.legend(frameon=False, fontsize=7)

    _finish_layout()
    _save_figure("manifold_param_distributions.png")
    plt.show()
    print(f"  -> Saved {_FIGURE_DIR / 'manifold_param_distributions.png'}")


# ======================================================================
# 6. Console summary
# ======================================================================

def print_manifold_summary(all_params, all_N, all_names, labels):
    """Per-cluster parameter statistics."""
    ul = sorted(set(labels))
    nc = len([l for l in ul if l >= 0])
    n_noise = int(sum(labels == -1))

    print(f"\n{'='*80}")
    print(f"Manifold summary: {nc} clusters, {n_noise} noise points")
    print(f"{'='*80}")

    for lbl in ul:
        m = labels == lbl
        n_pts = m.sum()
        label_str = f"Cluster {lbl}" if lbl >= 0 else "Noise"

        ps = all_params[m]
        Ns = all_N[m]
        ns = all_names[m]
        ds_counts = {n: (ns == n).sum() for n in sorted(set(all_names))}

        print(f"\n{label_str} ({n_pts} points):")
        print(f"  Datasets: {ds_counts}")
        print(f"  N:        mean={np.mean(Ns):.0f} +- {np.std(Ns):.0f}, "
              f"range=[{np.min(Ns):.0f}, {np.max(Ns):.0f}]")
        if n_pts >= 1:
            for ci, pn in enumerate(PARAM_NAMES):
                print(f"  {pn:<12} mean={np.mean(ps[:, ci]):.4f} +- {np.std(ps[:, ci]):.4f}")


# ======================================================================
# 6b. Intrinsic manifold: shape curve + scale axis
# ======================================================================

def plot_intrinsic_manifold(all_params, all_names, all_N, shape_fit, k_proj,
                            synthetic_params=None, synthetic_labels=None):
    """The intrinsic 2D manifold: shape coordinate (k_proj) vs sigma_half.

    This replaces the learned UMAP embedding with physically interpretable axes:
      - X-axis: projected k on the k--log10_gamma shape curve (steepness)
      - Y-axis: sigma_half (characteristic scale of substructure)
    """
    popt, lg_grid, k_grid = shape_fit
    a, d, s, p, c = popt

    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    move_figure(fig, 100, 100)
    datasets = sorted(set(all_names))

    # --- Left: the fitted shape curve ---
    ax = axes[0]
    for ds in datasets:
        m = all_names == ds
        ax.scatter(all_params[m, 0], all_params[m, 2], c=DATASET_COLORS[ds],
                   label=ds, s=15, alpha=0.6, edgecolors="none")
    ax.plot(k_grid, lg_grid, "k-", lw=2.5,
            label=f"Hill: k = {c:.2f} + {a:.2f}/(1+((lg-({d:.2f}))/{s:.2f})^{p:.2f})")

    # Overlay synthetic point processes for comparison
    if synthetic_params is not None and synthetic_labels is not None:
        syn_colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']
        syn_markers = ['^', 's', 'D', 'v', 'P']
        for i, (params, label) in enumerate(zip(synthetic_params, synthetic_labels)):
            if len(params) > 0:
                ax.scatter(params[:, 0], params[:, 2], c=syn_colors[i % len(syn_colors)],
                           marker=syn_markers[i % len(syn_markers)], s=50,
                           edgecolors='black', linewidths=0.5,
                           label=f'{label} (n={len(params)})', zorder=5)

    ax.set_xlabel("k")
    ax.set_ylabel("log10_gamma")
    ax.set_title("Shape curve: k vs log10_gamma")
    ax.legend(frameon=False, fontsize=6)

    # --- Center: intrinsic manifold (shape coord vs sigma_half) ---
    ax = axes[1]
    sigma_half = all_params[:, 1]
    for ds in datasets:
        m = all_names == ds
        ax.scatter(k_proj[m], sigma_half[m], c=DATASET_COLORS[ds],
                   label=ds, s=20, alpha=0.7, edgecolors="none")
    ax.set_xlabel("k_proj (projected k on shape curve)")
    ax.set_ylabel("sigma_half (characteristic scale)")
    ax.set_title("Intrinsic 2D manifold\n(shape coordinate vs scale)")
    ax.legend(frameon=False, fontsize=7)

    # --- Right: intrinsic manifold colored by N ---
    ax = axes[2]
    sc = ax.scatter(k_proj, sigma_half, c=all_N, cmap="plasma",
                    s=20, alpha=0.7, edgecolors="none")
    ax.set_xlabel("k_proj (projected k)")
    ax.set_ylabel("sigma_half")
    ax.set_title("Colored by N (# agents)")
    plt.colorbar(sc, ax=ax, label="N")

    # Print summary
    def hill_model(lg, a, d, s, p, c):
        return c + a / (1.0 + np.power(np.maximum((lg - d) / s, 1e-10), p))
    k_all = all_params[:, 0]
    lg_all = all_params[:, 2]
    k_pred = hill_model(lg_all, *popt)
    r2 = 1 - np.sum((k_pred - k_all)**2) / np.sum((k_all - np.mean(k_all))**2)
    print(f"\n  Shape curve: Hill model, R^2 = {r2:.4f}")
    print(f"    k = {popt[4]:.2f} + {popt[0]:.2f}/(1+((lg-({popt[1]:.2f}))/{popt[2]:.2f})^{popt[3]:.2f})")
    print(f"  Per-dataset mean (k_proj, sigma_half):")
    for ds in datasets:
        m = all_names == ds
        print(f"    {ds:<12} k_proj={np.mean(k_proj[m]):.2f}, sigma_half={np.mean(sigma_half[m]):.3f}")

    _finish_layout()
    _save_figure("manifold_intrinsic.png")
    plt.show()
    print(f"  -> Saved {_FIGURE_DIR / 'manifold_intrinsic.png'}")


def plot_shape_curve_mode_curves(shape_fit, all_params, N_typical=300):
    """Visualize 3PL mode-count curves sampled along the shape curve.

    Each point on the (k, log10_gamma) curve defines a full 3PL model.
    Sampling along the curve shows how the mode-count-vs-scale shape
    evolves with position on the manifold (fixing sigma_half).

    Args:
        shape_fit: (popt, lg_grid, k_grid) from fit_shape_curve
        all_params: [n_fits, 3] array of (k, sigma_half, log10_gamma)
        N_typical: fixed N for mode count curves
    """
    popt, lg_grid, k_grid = shape_fit
    a, d, s, p, c = popt

    # Pick a fixed sigma_half (median across all fits)
    sigma_half_fixed = np.median(all_params[:, 1])

    # Sample points along the shape curve
    lg_samples = np.linspace(lg_grid.min(), lg_grid.max(), 6)

    def hill_model(lg, a, d, s, p, c):
        return c + a / (1.0 + np.power(np.maximum((lg - d) / s, 1e-10), p))
    k_samples = hill_model(lg_samples, *popt)

    # Generate mode count curves
    sigma_range = np.logspace(-2, 2, 300)
    colors = cm.viridis(np.linspace(0.05, 0.95, len(lg_samples)))

    set_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5.5))
    move_figure(fig, 100, 600)

    # --- Left: shape curve with sampled points ---
    ax0.plot(k_grid, lg_grid, "k-", lw=2)
    ax0.scatter(k_samples, lg_samples, c=colors, s=80, zorder=5, edgecolors="white", linewidths=1)
    for i, (lg_i, k_i) in enumerate(zip(lg_samples, k_samples)):
        ax0.annotate(f"{chr(65+i)}", (k_i, lg_i), textcoords="offset points",
                     xytext=(8, -4), fontsize=9, fontweight="bold", color=colors[i])
    ax0.set_xlabel("k"); ax0.set_ylabel("log10_gamma")
    ax0.set_title(f"Shape curve with sample points\n(sigma_half = {sigma_half_fixed:.2f})")

    # --- Right: mode count curves for each sample ---
    for i, (lg_i, k_i) in enumerate(zip(lg_samples, k_samples)):
        gamma_i = 10.0 ** lg_i
        mc = 1.0 + (N_typical - 1.0) / np.power(
            1.0 + ((2.0 ** (1.0 / max(gamma_i, 1e-6)) - 1.0) *
                   (sigma_range / sigma_half_fixed) ** k_i),
            gamma_i)
        ax1.loglog(sigma_range, mc, color=colors[i], lw=2,
                   label=f"{chr(65+i)}: k={k_i:.1f}, lg={lg_i:.2f}")

    ax1.set_xlabel("sigma / sigma_half")
    ax1.set_ylabel("# Modes")
    ax1.set_title(f"3PL mode-count curves along shape curve\n(N={N_typical}, sigma_half={sigma_half_fixed:.2f})")
    ax1.legend(frameon=False, fontsize=7)

    _finish_layout()
    _save_figure("manifold_shape_mode_curves.png")
    plt.show()
    print(f"  -> Saved {_FIGURE_DIR / 'manifold_shape_mode_curves.png'}")


# ======================================================================
# 7. Scientific analysis: temporal dynamics & N-dependence
# ======================================================================

def plot_temporal_trajectories(raw_data, all_params, all_names):
    """Plot k_proj(t) and sigma_half(t) for each dataset with dense time sampling.

    Reveals how flocks move through the intrinsic manifold over time —
    transitions, trends, and stability of behavioral states.
    """
    set_style()
    datasets_with_time = [name for name, rd in raw_data.items()
                          if len(rd["step_range"]) > 5 and rd["valid"].sum() > 5]
    n_ds = len(datasets_with_time)
    if n_ds == 0:
        return

    fig, axes = plt.subplots(n_ds, 3, figsize=(20, 3.5 * n_ds), squeeze=False)
    move_figure(fig, 100, 600)

    for row, name in enumerate(datasets_with_time):
        rd = raw_data[name]
        sr = np.array(rd["step_range"])
        valid = rd["valid"]
        t = sr[valid]
        params = np.array([rd["params"][i] for i in range(len(valid)) if valid[i]])

        k = params[:, 0]
        sh = params[:, 1]
        dt = np.median(np.diff(t))

        # k_proj over time
        ax = axes[row, 0]
        ax.plot(t, k, "o-", color=DATASET_COLORS[name], markersize=3, lw=0.8, alpha=0.7)
        ax.set_ylabel("k (steepness)"); ax.set_xlabel("Frame")
        ax.set_title(f"{name}: k(t) — steepness over time, dt={dt:.0f}")

        # sigma_half over time
        ax = axes[row, 1]
        ax.plot(t, sh, "o-", color=DATASET_COLORS[name], markersize=3, lw=0.8, alpha=0.7)
        ax.set_ylabel("sigma_half (scale)"); ax.set_xlabel("Frame")
        ax.set_title(f"{name}: sigma_half(t) — characteristic scale")

        # ACF of k(t)
        ax = axes[row, 2]
        k_acf = k - np.mean(k)
        maxlag = min(100, len(k) // 4)
        acf_vals = [np.corrcoef(k_acf[:-l], k_acf[l:])[0, 1] for l in range(1, maxlag)]
        lags_fr = np.arange(1, len(acf_vals) + 1) * dt
        ax.plot(lags_fr, acf_vals, "o-", color=DATASET_COLORS[name], markersize=2, lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        # Zero-crossing
        zc = next((i for i, a in enumerate(acf_vals) if a < 0), len(acf_vals))
        tau_frames = (zc + 1) * dt if zc < len(acf_vals) else None
        if tau_frames:
            ax.axvline(tau_frames, color="gray", ls="--", lw=0.5)
            ax.set_title(f"{name}: k ACF (tau0={tau_frames:.0f} fr)")
        else:
            ax.set_title(f"{name}: k ACF (no zero-cross)")
        ax.set_xlabel("Lag (frames)"); ax.set_ylabel("Autocorrelation")

        # Annotate variability
        tau_str = f"tau0={tau_frames:.0f}fr" if tau_frames else "tau0=N/A"
        print(f"  {name}: k CV={np.std(k)/np.mean(k):.3f}, sigma_half CV={np.std(sh)/np.mean(sh):.3f}, {tau_str}")

    _finish_layout()
    _save_figure("manifold_temporal.png")
    plt.show()
    print(f"  -> Saved {_FIGURE_DIR / 'manifold_temporal.png'}")


def plot_N_dependence(all_params, all_N, all_names, shape_fit):
    """Analyze how k_proj and sigma_half depend on flock size N.

    Within each species, does the manifold position shift systematically
    with N? Tests the hypothesis that larger flocks are steeper or denser.
    """
    popt, lg_grid, k_grid = shape_fit

    def hill_model(lg, a, d, s, p, c):
        return c + a / (1.0 + np.power(np.maximum((lg - d) / s, 1e-10), p))

    k_proj_all = np.zeros(len(all_params))
    lg_all = all_params[:, 2]
    for i in range(len(all_params)):
        k_pred = hill_model(lg_all[i], *popt)
        k_proj_all[i] = k_pred

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    move_figure(fig, 100, 600)
    datasets = sorted(set(all_names))

    for ax, y_val, y_label in [
        (axes[0], all_params[:, 1], "sigma_half"),
        (axes[1], k_proj_all, "k_proj")]:
        for ds in datasets:
            m = all_names == ds
            ax.scatter(all_N[m], y_val[m], c=DATASET_COLORS[ds],
                       label=ds, s=10, alpha=0.6, edgecolors="none")
        ax.set_xlabel("N (flock size)"); ax.set_ylabel(y_label)
        ax.set_title(f"{y_label} vs N")
        ax.legend(frameon=False, fontsize=7)

    _finish_layout()
    _save_figure("manifold_N_dependence.png")
    plt.show()
    print(f"  -> Saved {_FIGURE_DIR / 'manifold_N_dependence.png'}")

    # Console: correlation coefficients
    print("\n  N-dependence (Pearson r):")
    for ds in datasets:
        m = all_names == ds
        r_sh = np.corrcoef(all_N[m], all_params[m, 1])[0, 1]
        r_k = np.corrcoef(all_N[m], k_proj_all[m])[0, 1]
        print(f"    {ds:<12} r(k_proj, N)={r_k:+.3f}   r(sigma_half, N)={r_sh:+.3f}")


# ======================================================================
# 8. Cluster stability via bootstrapping
# ======================================================================

def bootstrap_cluster_stability(all_params, n_boot=100):
    """Bootstrap resampling to assess cluster stability.

    Resamples the 3PL fit parameters with replacement, re-runs HDBSCAN
    directly on standardized 3D parameter space (not UMAP), and measures
    per-cluster preservation. This is more conservative than UMAP-HDBSCAN
    and reveals which clusters are robust to sampling noise.

    Returns: stability dict {cluster_id: [fraction_preserved, ...]}
    """
    from sklearn.preprocessing import StandardScaler
    try:
        import hdbscan
    except ImportError:
        print("  [SKIP] hdbscan not installed")
        return {}

    scaler = StandardScaler()
    X = scaler.fit_transform(all_params)
    n_total = len(all_params)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3,
                                cluster_selection_epsilon=0.5, metric="euclidean")
    ref_labels = clusterer.fit_predict(X)
    ref_clusters = [c for c in sorted(set(ref_labels)) if c >= 0]
    n_noise = int(sum(ref_labels == -1))

    stability = {c: [] for c in ref_clusters}
    for _ in range(n_boot):
        idx = np.random.choice(n_total, n_total, replace=True)
        labels_boot = clusterer.fit_predict(X[idx])
        for ref_c in ref_clusters:
            ref_mask = ref_labels == ref_c
            boot_in_ref = labels_boot[ref_mask]
            boot_in_ref = boot_in_ref[boot_in_ref >= 0]
            if len(boot_in_ref) > 0:
                majority = int(np.bincount(boot_in_ref).max())
                stability[ref_c].append(majority / ref_mask.sum())

    print(f"\n  Bootstrap stability ({n_boot} resamples, raw param space):")
    print(f"    Reference: {len(ref_clusters)} clusters + {n_noise} noise (UMAP-HDBSCAN: more clusters)")
    print(f"    {'Cluster':>8} {'Size':>6} {'Stability':>10}")

    all_scores = []
    for ref_c in sorted(stability.keys()):
        scores = stability[ref_c]
        mean_s = np.mean(scores)
        all_scores.extend(scores)
        n = (ref_labels == ref_c).sum()
        flag = " !!" if mean_s < 0.5 else ""
        print(f"    {ref_c:>8} {n:>6} {mean_s:>10.3f}{flag}")

    print(f"    Overall: {np.mean(all_scores):.3f} +- {np.std(all_scores):.3f}")

    # Flag unstable
    n_unstable = sum(1 for s in stability.values() if np.mean(s) < 0.5)
    if n_unstable > 0:
        print(f"    WARNING: {n_unstable}/{len(ref_clusters)} clusters are unstable (<0.5)")

    # Compare with GMM (BIC selection)
    from sklearn.mixture import GaussianMixture
    bics, gmm_models = [], []
    for n in range(2, 16):
        gmm = GaussianMixture(n_components=n, random_state=42, covariance_type="full")
        gmm.fit(X)
        bics.append(gmm.bic(X))
        gmm_models.append(gmm)
    best_n = np.argmin(bics) + 2
    gmm_labels = gmm_models[best_n - 2].predict(X)
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(ref_labels, gmm_labels)
    print(f"    GMM (BIC): {best_n} components, HDBSCAN vs GMM ARI={ari:.3f}")
    print(f"    Note: cluster counts are method-dependent (7 vs {best_n}), ARI={ari:.1f} indicates different structures")

    return stability


def compute_synthetic_params(cache_dir=None):
    """Load precomputed synthetic 3PL fits from cache, or generate if missing.

    Returns (synthetic_params, synthetic_labels) for plot_intrinsic_manifold.
    Cache: scenarios/_synthetic/synthetic_params.npz + synthetic_labels.npy
    """
    from experiments.synthetic_benchmark import load_cached, run_all

    synthetic_params, synthetic_labels = load_cached(cache_dir)
    if synthetic_params is None:
        print("  [CACHE MISS] Generating synthetic 3PL fits (this may take a few minutes)...")
        synthetic_params, synthetic_labels = run_all(
            n_trials=30, N=200, cache_dir=cache_dir
        )
    else:
        print(f"  [CACHE HIT] Loaded {sum(len(p) for p in synthetic_params)} "
              f"synthetic fits across {len(synthetic_params)} process types")
    return synthetic_params, synthetic_labels


# ======================================================================
# 9. Main
# ======================================================================

def main():
    global _FIGURE_DIR
    import argparse
    p = argparse.ArgumentParser(description="Parameter manifold investigation")
    p.add_argument("--no-display", action="store_true", help="Skip plt.show()")
    p.add_argument("--saturation", type=float, default=0.8,
                   help="Trim plateau at saturation * N (default: 0.8)")
    p.add_argument("--seed", type=int, default=12345)
    add_managed_output_arguments(p)
    args = p.parse_args()
    artifacts = create_analysis_artifacts(
        args,
        name="parameter manifold 3PL",
        resolved_config={
            "analysis": "parameter_manifold_3pl",
            "datasets": DATASET_RUNS,
            "saturation": args.saturation,
            "seed": args.seed,
        },
        entrypoint="experiments.parameter_manifold",
    )
    _FIGURE_DIR = artifacts.figures_dir
    np.random.seed(args.seed)
    if args.no_display:
        plt.show = lambda: None
        print("[--no-display] Headless mode.\n")

    # --- Fit ---
    all_params_list, all_N_list, all_names_list = [], [], []
    raw_data = {}

    for rp in DATASET_RUNS:
        name = rp["name"]
        print(f"\n{'='*60}\nDataset: {name}\n{'='*60}")
        sr, Na, scr, am, nn_dists = load_cached_data(
            rp, project_root=args.project_root
        )
        if sr is None:
            continue

        raw_data[name] = {"step_range": sr, "N_array": Na, "scale_range": scr,
                          "all_modes": am, "nn_dists": nn_dists}

        print(f"  Fitting centered_3pl_log ({len(sr)} steps, sat={args.saturation})...")
        res = fit_all_steps(sr, Na, scr, am, saturation=args.saturation)
        n_ok = res["success"].sum()
        print(f"  OK: {n_ok}/{len(sr)}")

        raw_data[name]["valid"] = res["success"]
        raw_data[name]["params"] = res["params"]

        valid = res["success"]
        if valid.sum() > 0:
            pv = np.array([res["params"][i] for i in range(len(valid)) if valid[i]])
            all_params_list.append(pv)
            all_N_list.append(Na[valid])
            all_names_list.append(np.array([name] * valid.sum()))

    if not all_params_list:
        print("No data. Exiting."); return

    all_params = np.vstack(all_params_list)
    all_N = np.concatenate(all_N_list)
    all_names = np.concatenate(all_names_list)
    n_total = len(all_params)

    print(f"\n{'='*60}")
    print(f"Total: {n_total} fits")
    print(f"Datasets: {sorted(set(all_names))}")
    for ci, pn in enumerate(PARAM_NAMES):
        print(f"  {pn:<14} [{np.min(all_params[:, ci]):.4f}, {np.max(all_params[:, ci]):.4f}] "
              f"median={np.median(all_params[:, ci]):.4f}")

    # --- Manifold learning ---
    print(f"\n{'='*60}\nManifold learning on ({', '.join(PARAM_NAMES)})\n{'='*60}")
    features = all_params  # already in good ranges: k, sigma_half, log10_gamma
    embeddings = run_manifold_learning(features)

    # --- Clustering ---
    print(f"\n{'='*60}\nClustering on UMAP\n{'='*60}")
    emb = embeddings["umap"] if embeddings["has_umap"] else embeddings["tsne"]
    labels, method = run_clustering(emb)
    nc = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  {method}: {nc} clusters, {int(sum(labels == -1))} noise")
    artifacts.save_npz(
        "manifold_fits.npz",
        overwrite=args.resume,
        parameters=all_params,
        number_of_agents=all_N,
        dataset_names=all_names,
        cluster_labels=labels,
    )

    # --- Cluster stability ---
    bootstrap_cluster_stability(all_params)

    # --- Shape curve fitting ---
    print(f"\n{'='*60}\nFitting shape curve (k vs log10_gamma)\n{'='*60}")
    k_vals = all_params[:, 0]
    lg_vals = all_params[:, 2]
    shape_fit = fit_shape_curve(k_vals, lg_vals)
    _, lg_grid, k_grid = shape_fit
    k_proj, lg_proj = project_to_shape_curve(k_vals, lg_vals, lg_grid, k_grid)
    artifacts.save_npz(
        "shape_projection.npz",
        overwrite=args.resume,
        shape_parameters=shape_fit[0],
        log_gamma_grid=lg_grid,
        k_grid=k_grid,
        projected_k=k_proj,
        projected_log_gamma=lg_proj,
    )

    # --- Figures ---
    print(f"\n{'='*60}\nGenerating figures\n{'='*60}")
    plot_pca_scree(embeddings["pca_model"], embeddings["scaler"])
    # Synthetic point processes for shape curve comparison
    synthetic_params, synthetic_labels = compute_synthetic_params(
        args.project_root / "scenarios" / "_synthetic"
    )
    plot_intrinsic_manifold(all_params, all_names, all_N, shape_fit, k_proj,
                            synthetic_params, synthetic_labels)
    plot_shape_curve_mode_curves(shape_fit, all_params)
    plot_parameter_space(all_params, all_names)
    plot_embeddings(embeddings, all_names, all_N, labels)
    plot_cluster_curves(all_params, all_N, all_names, labels)
    plot_param_distributions(all_params, all_names, labels)

    # --- Scientific analyses ---
    print(f"\n{'='*60}\nScientific analyses\n{'='*60}")
    plot_temporal_trajectories(raw_data, all_params, all_names)
    plot_N_dependence(all_params, all_N, all_names, shape_fit)

    # --- sigma_half vs physical nearest-neighbor distance ---
    print("\n  sigma_half vs physical nearest-neighbor distance:")
    nn_by_species = {name: raw_data[name]["nn_dists"] for name in sorted(set(all_names))
                     if raw_data[name].get("nn_dists") is not None}
    if nn_by_species:
        print(f"    {'Species':<12} {'mean_sh':>8} {'mean_nn':>8} {'ratio':>8} {'r(sh,nn)':>10}")
        all_sh, all_nn_cross = [], []
        for ds in sorted(nn_by_species.keys()):
            m = all_names == ds
            sh = all_params[m, 1]
            nn = nn_by_species[ds][:len(sh)]
            ratio = sh / (nn + 1e-10)
            r = np.corrcoef(sh, nn)[0, 1]
            all_sh.extend(sh); all_nn_cross.extend(nn)
            print(f"    {ds:<12} {np.mean(sh):>8.3f} {np.mean(nn):>8.3f} {np.mean(ratio):>8.3f} {r:>10.3f}")
        r_cross = np.corrcoef(all_sh, all_nn_cross)[0, 1]
        print(f"    {'Cross-species':<12} r(sigma_half, avg_nn) = {r_cross:.3f}")

        # Bootstrap 95% CIs for the ratio within each species
        print("\n    Ratio bootstrap (95% CI, 1000 resamples):")
        for ds in sorted(nn_by_species.keys()):
            m = all_names == ds
            sh = all_params[m, 1]
            nn = nn_by_species[ds][:len(sh)]
            ratios = sh / (nn + 1e-10)
            bs_means = [np.mean(np.random.choice(ratios, len(ratios), replace=True))
                        for _ in range(1000)]
            ci_lo, ci_hi = np.percentile(bs_means, [2.5, 97.5])
            overlap_msg = ""
            for ds2 in sorted(nn_by_species.keys()):
                if ds2 <= ds: continue
                m2 = all_names == ds2
                sh2 = all_params[m2, 1]
                nn2 = nn_by_species[ds2][:len(sh2)]
                ratios2 = sh2 / (nn2 + 1e-10)
                bs_means2 = [np.mean(np.random.choice(ratios2, len(ratios2), replace=True))
                             for _ in range(1000)]
                ci2_lo, ci2_hi = np.percentile(bs_means2, [2.5, 97.5])
                if ci_lo <= ci2_hi and ci2_lo <= ci_hi:
                    overlap_msg += f"  overlaps {ds2}"
            print(f"      {ds:<12} {np.mean(ratios):.4f} [{ci_lo:.4f}, {ci_hi:.4f}]{overlap_msg}")

    # --- Summary ---
    print_manifold_summary(all_params, all_N, all_names, labels)
    artifacts.save_json(
        "summary.json",
        {
            "fit_count": n_total,
            "datasets": sorted(set(all_names)),
            "cluster_method": method,
            "cluster_count": nc,
        },
        category="metrics",
        overwrite=args.resume,
    )
    print(f"\nDone. Outputs: {artifacts.run_dir}")


if __name__ == "__main__":
    main()
