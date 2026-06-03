"""
Precompute 3PL fits for synthetic spatial point processes.
Cached to scenarios/_synthetic/synthetic_params.npz so parameter_manifold
doesn't recompute them every run.

Processes span the spectrum: regular → random → clustered
  - Poisson (homogeneous)
  - Hard-core (minimum distance, repulsive)
  - Lattice + jitter (ordered)
  - Thomas (Gaussian clusters with Poisson parents)
  - Matern (uniform disk clusters with Poisson parents)
  - Log-Gaussian Cox (doubly stochastic, clustered)
"""
import sys, os
sys.path.append(os.getcwd())

import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm

from experiments.parameter_manifold import model_centered_3pl


# ======================================================================
# Point process generators
# ======================================================================

def poisson(N, volume=1000.0, rng=None):
    """Homogeneous Poisson process."""
    if rng is None: rng = np.random.default_rng()
    L = volume ** (1/3)
    return rng.uniform(0, L, (N, 3))


def hard_core(N, min_dist, volume=1000.0, max_attempts=100, rng=None):
    """Hard-core (Matérn type II) process with minimum inter-point distance."""
    if rng is None: rng = np.random.default_rng()
    L = volume ** (1/3)
    points = []
    for _ in range(N * max_attempts):
        p = rng.uniform(0, L, 3)
        if all(np.linalg.norm(p - q) > min_dist for q in points):
            points.append(p)
        if len(points) >= N:
            break
    pts = np.array(points[:N])
    if len(pts) < N:
        pts = poisson(N, volume, rng=rng)  # fallback
    return pts


def lattice_jitter(N, jitter_frac=0.3, volume=1000.0, rng=None):
    """Regular lattice with random jitter (ordered)."""
    if rng is None: rng = np.random.default_rng()
    L = volume ** (1/3)
    n_per_dim = int(np.ceil(N ** (1/3)))
    spacing = L / n_per_dim
    grid = np.mgrid[0:n_per_dim, 0:n_per_dim, 0:n_per_dim].reshape(3, -1).T * spacing
    grid = grid[:N] + spacing * 0.5
    jitter = rng.normal(0, spacing * jitter_frac, grid.shape)
    return grid + jitter


def thomas(N, n_parents, cluster_std, volume=1000.0, rng=None):
    """Thomas (Poisson cluster) process."""
    if rng is None: rng = np.random.default_rng()
    L = volume ** (1/3)
    parents = rng.uniform(0, L, (n_parents, 3))
    pts_per = N // n_parents
    points = []
    for c in range(n_parents):
        n_pts = pts_per + (1 if c < N % n_parents else 0)
        points.append(parents[c] + rng.normal(0, cluster_std, (n_pts, 3)))
    return np.vstack(points)


def matern(N, n_parents, cluster_radius, volume=1000.0, rng=None):
    """Matern cluster process: uniform disk clusters."""
    if rng is None: rng = np.random.default_rng()
    L = volume ** (1/3)
    parents = rng.uniform(0, L, (n_parents, 3))
    pts_per = N // n_parents
    points = []
    for c in range(n_parents):
        n_pts = pts_per + (1 if c < N % n_parents else 0)
        # Uniform within sphere of radius cluster_radius
        r = cluster_radius * rng.random(n_pts) ** (1/3)
        dirs = rng.normal(0, 1, (n_pts, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        points.append(parents[c] + dirs * r[:, None])
    return np.vstack(points)


def lgcp(N, n_grid=20, volume=1000.0, rng=None):
    """Log-Gaussian Cox process: doubly stochastic, clustered."""
    if rng is None: rng = np.random.default_rng()
    L = volume ** (1/3)
    # Generate log-Gaussian random field on a grid
    grid_1d = np.linspace(0, L, n_grid)
    X, Y, Z = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
    coords = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    # Gaussian field with exponential covariance
    dist2 = np.sum((coords[:, None] - coords[None, :]) ** 2, axis=-1)
    K = np.exp(-np.sqrt(dist2) / (L * 0.3))
    L_chol = np.linalg.cholesky(K + 1e-6 * np.eye(K.shape[0]))
    field = L_chol @ rng.normal(0, 1, K.shape[0])
    field = np.exp(field - 0.5)  # log-Gaussian: mean 1
    # Sample points proportional to field intensity
    cell_vol = (L / n_grid) ** 3
    intensity = field * cell_vol
    # Normalize to expected N points, then Poisson sample
    intensity = intensity / intensity.sum() * N
    n_per_cell = rng.poisson(intensity)
    points = []
    for i, n_cell in enumerate(n_per_cell):
        if n_cell > 0:
            cell_origin = coords[i]
            cell_points = cell_origin + rng.uniform(0, L / n_grid, (int(n_cell), 3))
            points.append(cell_points)
    pts = np.vstack(points) if points else poisson(N, volume, rng=rng)
    # Downsample or upsample to exactly N
    if len(pts) > N:
        pts = pts[rng.choice(len(pts), N, replace=False)]
    elif len(pts) < N:
        extra = poisson(N - len(pts), volume, rng=rng)
        pts = np.vstack([pts, extra])
    return pts


# ======================================================================
# 3PL fitting
# ======================================================================

def compute_mode_curve(positions, scales):
    """Mode count at each scale."""
    import torch
    from dfr.mode_finding import mode_counting
    from scipy.spatial.distance import cdist as scipy_cdist
    pos = torch.from_numpy(positions).cuda().float()
    d = scipy_cdist(positions, positions)
    np.fill_diagonal(d, 1e10)
    avg_nn = max(float(np.median(np.min(d, axis=1))), 1e-8)
    tol = max(avg_nn * 1e-3, 1e-8)
    return np.array([mode_counting(pos, pos.clone(), s, max_iter=400, tol=tol)
                     for s in scales])


def fit_3pl(scales, mode_counts, N):
    """Fit 3PL to mode-count curve. Returns (k, sigma_half, log10_gamma)."""
    def fn(x, k, sh, lg):
        return 1.0 + model_centered_3pl(x, [k, sh, lg], N)
    try:
        popt, _ = curve_fit(fn, scales, mode_counts, p0=[2.0, np.median(scales), 0.0],
                            bounds=([0.1, 1e-6, -2], [20, np.inf, 5]), maxfev=5000)
        return popt
    except Exception:
        return None


# ======================================================================
# Main: generate and cache
# ======================================================================

def run_all(n_trials=30, N=200):
    """Generate synthetic data, fit 3PL, save to cache."""
    rng = np.random.default_rng(42)
    scales = np.logspace(-1, 1.5, 40)

    configs = [
        ("Poisson", lambda r: poisson(N, rng=r)),
        ("Hard-core", lambda r: hard_core(N, min_dist=1.5, rng=r)),
        ("Lattice-jitter", lambda r: lattice_jitter(N, jitter_frac=0.3, rng=r)),
        ("Thomas cs=0.5", lambda r: thomas(N, n_parents=10, cluster_std=0.5, rng=r)),
        ("Thomas cs=1.0", lambda r: thomas(N, n_parents=10, cluster_std=1.0, rng=r)),
        ("Thomas cs=2.0", lambda r: thomas(N, n_parents=10, cluster_std=2.0, rng=r)),
        ("Thomas cs=4.0", lambda r: thomas(N, n_parents=10, cluster_std=4.0, rng=r)),
        ("Matern r=0.5", lambda r: matern(N, n_parents=10, cluster_radius=0.5, rng=r)),
        ("Matern r=1.0", lambda r: matern(N, n_parents=10, cluster_radius=1.0, rng=r)),
        ("Matern r=2.0", lambda r: matern(N, n_parents=10, cluster_radius=2.0, rng=r)),
        ("Matern r=4.0", lambda r: matern(N, n_parents=10, cluster_radius=4.0, rng=r)),
        ("LGCP", lambda r: lgcp(N, rng=r)),
    ]

    all_params = []
    all_labels = []

    for label, generator in configs:
        params_list = []
        for _ in tqdm(range(n_trials), desc=f"  {label}"):
            pos = generator(rng)
            mc = compute_mode_curve(pos, scales)
            p = fit_3pl(scales, mc, N)
            if p is not None:
                params_list.append(p)
        if params_list:
            arr = np.array(params_list)
            all_params.append(arr)
            all_labels.append(label)
            print(f"    {label}: {len(arr)}/{n_trials} fits, "
                  f"k={arr[:,0].mean():.2f}+-{arr[:,0].std():.2f}, "
                  f"lg={arr[:,2].mean():.3f}+-{arr[:,2].std():.3f}")

    # Save cache
    cache_dir = os.path.join(os.getcwd(), "scenarios", "_synthetic")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "synthetic_params.npz")
    np.savez(cache_file, *all_params)
    # Save labels separately
    label_file = os.path.join(cache_dir, "synthetic_labels.npy")
    np.save(label_file, np.array(all_labels, dtype=object))
    print(f"\n  Cached {sum(len(p) for p in all_params)} fits across {len(all_params)} processes")
    print(f"  -> {cache_file}")

    return all_params, all_labels


def load_cached():
    """Load precomputed synthetic params. Returns (all_params, all_labels)."""
    cache_dir = os.path.join(os.getcwd(), "scenarios", "_synthetic")
    cache_file = os.path.join(cache_dir, "synthetic_params.npz")
    label_file = os.path.join(cache_dir, "synthetic_labels.npy")
    if not os.path.exists(cache_file):
        return None, None
    data = np.load(cache_file, allow_pickle=True)
    all_labels = list(np.load(label_file, allow_pickle=True))
    all_params = [data[k] for k in data.files]
    return all_params, all_labels


if __name__ == "__main__":
    print("=" * 60)
    print("  Precomputing synthetic 3PL fits")
    print("=" * 60)
    run_all(n_trials=30, N=200)
