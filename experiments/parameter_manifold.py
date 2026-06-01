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
sys.path.append(os.getcwd())

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expm1
from tqdm import tqdm
import warnings

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dfr.simulation_config import SimulationConfig
from dfr.dataset_io import DatasetFactory
from experiments.power_law import move_figure


# ======================================================================
# 0. Model
# ======================================================================

def model_centered_3pl(x, params, N):
    """Centered 3PL with sigma_half and log10_gamma.

    params = [k, sigma_half, log10_gamma]
    Returns m(x) - 1 (caller adds D=1).
    """
    k, sigma_half, log10_gamma = params
    gamma = 10.0 ** log10_gamma
    g_safe = max(gamma, 1e-6)
    scaling = expm1(np.log(2.0) / g_safe)
    scaling_safe = max(scaling, 1e-12)
    log_ratio = np.clip(k * np.log(np.maximum(x / sigma_half, 1e-12)) +
                         np.log(scaling_safe), -500, 500)
    return (N - 1.0) / np.power(1.0 + np.exp(log_ratio), gamma)


PARAM_NAMES = ["k", "sigma_half", "log10_gamma"]
P0 = lambda med: [2.0, med, 0.0]
BOUNDS = ([0.1, 1e-6, -2.0], [20.0, np.inf, 5.0])


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
    "starling": "#2c3e50", "jackdaw": "#e67e22",
    "jackdaw2": "#27ae60", "swift": "#8e44ad",
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
    sp = os.path.join(os.getcwd(), "scenarios", name)

    mp = os.path.join(sp, "modes.npy")
    srp = os.path.join(sp, "scale_range.npy")
    if not os.path.exists(mp) or not os.path.exists(srp):
        print(f"  [WARN] No cache for '{name}'")
        return None, None, None, None

    all_modes = np.load(mp)
    scale_range = np.load(srp)

    config = SimulationConfig(os.path.join(sp, "config.yaml"))
    dataset = DatasetFactory().get_dataset(config.data_file)

    max_steps = dataset.trajectories.shape[0]
    end = run_params["end_step"]
    eff_end = end if end is not None and end <= max_steps else max_steps
    step_range = list(range(run_params["start_step"], eff_end, run_params["step_length"]))

    n_eff = min(all_modes.shape[0], scale_range.shape[0], len(step_range))
    if all_modes.shape[0] != scale_range.shape[0] or scale_range.shape[0] != len(step_range):
        print(f"  [INFO] Aligning -> n={n_eff}")

    step_range = step_range[:n_eff]
    scale_range = scale_range[:n_eff]
    all_modes = all_modes[:n_eff]

    print(f"  Loading N for {name} ({len(step_range)} steps)...")
    N_array = np.array([dataset.positions_at_time_step(s).shape[0] for s in tqdm(step_range)])

    return step_range, N_array, scale_range, all_modes


# ======================================================================
# 2. Fitting
# ======================================================================

def fit_all_steps(step_range, N_array, scale_range, all_modes,
                  saturation=0.8, num_test_scale=40):
    """Fit centered_3pl_log to all time steps of one dataset."""
    n_steps = all_modes.shape[0]
    params = [None] * n_steps
    fitted = [None] * n_steps
    resid_var = np.full(n_steps, np.nan)
    success = np.zeros(n_steps, dtype=bool)

    for i in range(n_steps):
        N = N_array[i]
        s_start, s_end = scale_range[i]
        test_scales = np.logspace(np.log10(max(s_start, 1e-6)),
                                  np.log10(max(s_end, 1e-5)), num_test_scale)
        modes = all_modes[i]

        # Trim plateau
        bi = int(np.argmax(modes <= saturation * N))
        if bi == 0:
            bi = max(1, int(np.argmax(modes <= min(saturation + 0.1, 0.99) * N)))

        x_data = test_scales[bi:]
        y_data = modes[bi:]

        if len(x_data) < 5:
            continue

        try:
            popt, _ = curve_fit(
                lambda x, *p: model_centered_3pl(x, p, N),
                x_data, y_data,
                p0=P0(float(np.median(test_scales))),
                sigma=np.maximum(y_data, 1.0),
                absolute_sigma=True, bounds=BOUNDS, maxfev=5000,
            )
            params[i] = popt
            fitted[i] = model_centered_3pl(test_scales, popt, N) + 1.0
            resid_var[i] = np.mean((y_data - (model_centered_3pl(x_data, popt, N) + 1.0))**2)
            success[i] = True
        except (RuntimeError, ValueError):
            continue

    return {
        "params": params,
        "fitted": fitted,
        "resid_var": resid_var,
        "success": success,
        "test_scales": test_scales,  # stored from last step (for plotting)
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

    plt.tight_layout()
    plt.savefig("figs/manifold_pca_scree.png", bbox_inches="tight", dpi=300)
    plt.show()
    print("  -> Saved figs/manifold_pca_scree.png")


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

    plt.tight_layout()
    plt.savefig("figs/manifold_embeddings.png", bbox_inches="tight", dpi=300)
    plt.show()
    print("  -> Saved figs/manifold_embeddings.png")


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

    plt.tight_layout()
    plt.savefig("figs/manifold_parameter_space.png", bbox_inches="tight", dpi=300)
    plt.show()
    print("  -> Saved figs/manifold_parameter_space.png")


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

    plt.tight_layout()
    plt.savefig("figs/manifold_cluster_curves.png", bbox_inches="tight", dpi=300)
    plt.show()
    print("  -> Saved figs/manifold_cluster_curves.png")


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

    plt.tight_layout()
    plt.savefig("figs/manifold_param_distributions.png", bbox_inches="tight", dpi=300)
    plt.show()
    print("  -> Saved figs/manifold_param_distributions.png")


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
# 7. Main
# ======================================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description="Parameter manifold investigation")
    p.add_argument("--no-display", action="store_true", help="Skip plt.show()")
    p.add_argument("--saturation", type=float, default=0.8,
                   help="Trim plateau at saturation * N (default: 0.8)")
    args = p.parse_args()
    if args.no_display:
        plt.show = lambda: None
        print("[--no-display] Headless mode.\n")

    # --- Fit ---
    all_params_list, all_N_list, all_names_list = [], [], []
    raw_data = {}

    for rp in DATASET_RUNS:
        name = rp["name"]
        print(f"\n{'='*60}\nDataset: {name}\n{'='*60}")
        sr, Na, scr, am = load_cached_data(rp)
        if sr is None:
            continue

        raw_data[name] = {"step_range": sr, "N_array": Na, "scale_range": scr, "all_modes": am}

        print(f"  Fitting centered_3pl_log ({len(sr)} steps, sat={args.saturation})...")
        res = fit_all_steps(sr, Na, scr, am, saturation=args.saturation)
        n_ok = res["success"].sum()
        print(f"  OK: {n_ok}/{len(sr)}")

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

    # --- Figures ---
    print(f"\n{'='*60}\nGenerating figures\n{'='*60}")
    plot_pca_scree(embeddings["pca_model"], embeddings["scaler"])
    plot_parameter_space(all_params, all_names)
    plot_embeddings(embeddings, all_names, all_N, labels)
    plot_cluster_curves(all_params, all_N, all_names, labels)
    plot_param_distributions(all_params, all_names, labels)

    # --- Summary ---
    print_manifold_summary(all_params, all_N, all_names, labels)
    print(f"\nDone.")


if __name__ == "__main__":
    main()
