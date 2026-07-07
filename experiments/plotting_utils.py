"""Shared 3D rendering utilities for density field and GMM visualisation.

Extracted from ``experiments/dfr_plot.py`` and
``experiments/run_scenarios_angle_sweep.py`` to eliminate duplication between
``plot_jackdaw2_density_field``, ``_draw_gmm_frame``, and the new multi-scale
density figures.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

from dfr.plotting import apply_academic_style, style_3d_axis


# ═══════════════════════════════════════════════════════════════════════════════
#  Matplotlib style helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _set_academic_style() -> None:
    """Apply publication-quality academic styling to matplotlib."""
    apply_academic_style(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "savefig.pad_inches": 0.1,
        }
    )


def _style_3d_ax(ax: plt.Axes) -> None:
    """Transparent panes, subtle grid, no edge colour on a 3D axis."""
    style_3d_axis(ax)


# ═══════════════════════════════════════════════════════════════════════════════
#  Voxel grid construction & density evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def build_voxel_grid(
    positions: np.ndarray,
    scale: float,
    voxel_res_factor: float = 2.5e-2,
    device: str = "cuda",
) -> dict:
    """Create the 3D voxel grid parameters for density evaluation.

    Parameters
    ----------
    positions : (N, 3) float32
        Agent positions.
    scale : float
        Gaussian sigma used to pad the bounding box (3σ on each side).
    voxel_res_factor : float
        Voxel width as a fraction of the largest spatial extent.
    device : str
        Torch device for the tick tensors.

    Returns
    -------
    dict with keys: x_ticks, y_ticks, z_ticks, nx, ny, nz, voxel_res,
    total_voxels.
    """
    min_c = np.min(positions, axis=0)
    max_c = np.max(positions, axis=0)
    extent = np.max(max_c - min_c)
    voxel_res = extent * voxel_res_factor
    bounds = np.vstack((min_c - 3 * scale, max_c + 3 * scale)).T

    x_ticks = torch.arange(bounds[0, 0], bounds[0, 1], voxel_res, device=device)
    y_ticks = torch.arange(bounds[1, 0], bounds[1, 1], voxel_res, device=device)
    z_ticks = torch.arange(bounds[2, 0], bounds[2, 1], voxel_res, device=device)

    return {
        "x_ticks": x_ticks,
        "y_ticks": y_ticks,
        "z_ticks": z_ticks,
        "nx": len(x_ticks),
        "ny": len(y_ticks),
        "nz": len(z_ticks),
        "voxel_res": voxel_res,
        "total_voxels": len(x_ticks) * len(y_ticks) * len(z_ticks),
    }


def compute_gt_density(
    positions: np.ndarray,
    scale: float,
    grid: dict,
    batch_size: int = 50000,
    device: str = "cuda",
) -> torch.Tensor:
    """Evaluate GT GMM (one isotropic Gaussian per agent) on the voxel grid.

    Returns a flat float32 tensor on CPU so it does not exhaust GPU memory.
    """
    from dfr.utils import eval_isotropic_gmm_torch

    N = positions.shape[0]
    gt_means = torch.from_numpy(positions).float().to(device)
    gt_weights = torch.full((N,), 1.0, device=device, dtype=torch.float)
    gt_sigmas = torch.full((N,), scale, device=device, dtype=torch.float)

    x_t = grid["x_ticks"]
    y_t = grid["y_ticks"]
    z_t = grid["z_ticks"]
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    total = grid["total_voxels"]

    density_flat = torch.empty(total, dtype=torch.float32, device="cpu")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        idx = torch.arange(start, end, device=device)
        ix = idx // (ny * nz)
        iy = (idx // nz) % ny
        iz = idx % nz
        coords = torch.stack([x_t[ix], y_t[iy], z_t[iz]], dim=-1)
        dens = eval_isotropic_gmm_torch(coords, gt_means, gt_weights, gt_sigmas)
        density_flat[start:end] = dens.cpu()

    return density_flat  # on CPU


# ═══════════════════════════════════════════════════════════════════════════════
#  3D scatter-based density rendering primitives
# ═══════════════════════════════════════════════════════════════════════════════

# Default density-shell layer configuration matching the existing figures.
DEFAULT_LAYERS = [
    {"thresh_frac": 0.10, "alpha_min": 0.45, "alpha_max": 0.95, "size": 8},
    {"thresh_frac": 0.02, "alpha_min": 0.25, "alpha_max": 0.80, "size": 6},
    {"thresh_frac": 0.002, "alpha_min": 0.08, "alpha_max": 0.50, "size": 4},
]

# Lower-alpha variant for when wireframes are overlaid.
FIELD_LAYERS = [
    {"thresh_frac": 0.10, "alpha_min": 0.18, "alpha_max": 0.55, "size": 8},
    {"thresh_frac": 0.02, "alpha_min": 0.10, "alpha_max": 0.40, "size": 6},
    {"thresh_frac": 0.002, "alpha_min": 0.04, "alpha_max": 0.22, "size": 4},
]


def _layer_thresholds(
    max_density: float, layers: Sequence[dict] | None = None,
) -> list[dict]:
    """Resolve layer config dicts, converting *thresh_frac* → absolute *thresh*."""
    if layers is None:
        layers = DEFAULT_LAYERS
    return [
        {**lyr, "thresh": lyr["thresh_frac"] * max_density} for lyr in layers
    ]


def render_density_shells(
    ax: plt.Axes,
    density_3d: np.ndarray,
    x_ticks_np: np.ndarray,
    y_ticks_np: np.ndarray,
    z_ticks_np: np.ndarray,
    max_density: float | None = None,
    layers: Sequence[dict] | None = None,
) -> None:
    """Render a 3D density field as nested semi-transparent scatter shells.

    Voxels above each layer's absolute density threshold are drawn as scatter
    points coloured by ``viridis`` with ``PowerNorm(gamma=0.35)``.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.Axes3D
    density_3d : (nx, ny, nz) float64
    x_ticks_np, y_ticks_np, z_ticks_np : 1D ndarray
        Voxel-centre coordinates on each axis (NumPy, on CPU).
    max_density : float, optional
        Max density for normalisation.  Defaults to ``density_3d.max()``.
    layers : list of dict, optional
        Each dict: thresh_frac, alpha_min, alpha_max, size.  Defaults to
        ``DEFAULT_LAYERS``.
    """
    if max_density is None:
        max_density = float(density_3d.max())

    resolved = _layer_thresholds(max_density, layers)
    norm = mcolors.PowerNorm(gamma=0.35, vmin=0, vmax=max_density)

    for layer in resolved:
        mask = density_3d >= layer["thresh"]
        if not mask.any():
            continue
        ix, iy, iz = np.where(mask)
        pts = np.stack([x_ticks_np[ix], y_ticks_np[iy], z_ticks_np[iz]], axis=-1)
        vals = density_3d[mask]
        colours = plt.cm.viridis(norm(vals))
        alphas = (
            norm(vals) * (layer["alpha_max"] - layer["alpha_min"])
            + layer["alpha_min"]
        )
        colours[:, 3] = np.clip(alphas, layer["alpha_min"], layer["alpha_max"])
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=colours, s=layer["size"], edgecolors="none",
            depthshade=False, rasterized=True,
        )


def render_gmm_wireframes(
    ax: plt.Axes,
    means_np: np.ndarray,
    sigmas_np: np.ndarray,
    weights_np: np.ndarray,
    colour: str = "#4169e1",
    z_sort_pos: float = -5e8,
    sphere_res: int = 20,
) -> None:
    """Draw each isotropic GMM component as a wireframe sphere.

    Alpha is scaled by component weight relative to the max weight.  Each
    wireframe artist has its ``do_3d_projection`` monkey-patched so it sorts
    at a fixed ``_sort_zpos``, placing it in front of the density-field scatter
    but behind higher-priority overlays (agent positions / GMM means).

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.Axes3D
    means_np : (K, 3) float64
    sigmas_np : (K,) float64
        Isotropic sigma (radius) per component.
    weights_np : (K,) float64
    colour : str
        Matplotlib colour for the wireframes.
    z_sort_pos : float
        Fixed ``_sort_zpos`` for depth ordering.
    sphere_res : int
        Angular resolution of the sphere mesh.
    """
    K = means_np.shape[0]
    w_max = float(weights_np.max()) if weights_np.size > 0 else 1.0

    # Sphere mesh template (unit sphere)
    u = np.linspace(0, 2 * np.pi, sphere_res)
    v = np.linspace(0, np.pi, sphere_res)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones(np.size(u)), np.cos(v))

    wireframe_objs = []
    for j in range(K):
        r = float(sigmas_np[j])
        alpha = (
            max(0.15, min(0.70, float(weights_np[j]) / w_max))
            if w_max > 0
            else 0.25
        )
        rgba = (*mcolors.to_rgb(colour), alpha)
        wf = ax.plot_wireframe(
            means_np[j, 0] + r * sx,
            means_np[j, 1] + r * sy,
            means_np[j, 2] + r * sz,
            color=rgba,
            rstride=2, cstride=2, linewidth=1.7,
        )
        wireframe_objs.append(wf)

    # Monkey-patch each wireframe for depth ordering.
    for wf in wireframe_objs:
        _orig_wf = wf.do_3d_projection

        def _patch(orig=_orig_wf, obj=wf, zpos=z_sort_pos):
            orig()
            obj._sort_zpos = zpos
            return obj._sort_zpos

        wf.do_3d_projection = _patch


def render_agent_positions(
    ax: plt.Axes,
    positions: np.ndarray,
    colour: str = "#1f2937",
    size: float = 25,
    alpha: float = 1.0,
    z_sort_pos: float = -1e9,
) -> None:
    """Overlay agent positions as scatter points forced to render on top.

    The returned collection has its ``do_3d_projection`` monkey-patched so
    ``_sort_zpos`` is fixed at *z_sort_pos* (default ``-1e9``, i.e. in front
    of everything).
    """
    coll = ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c=colour, s=size, alpha=alpha, linewidths=0.8,
    )
    _orig = coll.do_3d_projection

    def _force(zpos=z_sort_pos, orig=_orig, obj=coll):
        orig()
        obj._sort_zpos = zpos
        return obj._sort_zpos

    coll.do_3d_projection = _force
    return coll


def render_gmm_means(
    ax: plt.Axes,
    means_np: np.ndarray,
    colour: str = "#4169e1",
    size: float = 14,
    alpha: float = 0.85,
    z_sort_pos: float = -6e8,
) -> None:
    """Overlay GMM mean positions as small markers, sorted behind agents but
    in front of wireframes."""
    coll = ax.scatter(
        means_np[:, 0], means_np[:, 1], means_np[:, 2],
        c=colour, marker="o", s=size, alpha=alpha,
        edgecolors="none", depthshade=True,
    )
    _orig = coll.do_3d_projection

    def _force(zpos=z_sort_pos, orig=_orig, obj=coll):
        orig()
        obj._sort_zpos = zpos
        return obj._sort_zpos

    coll.do_3d_projection = _force


# ═══════════════════════════════════════════════════════════════════════════════
#  Composite renderers
# ═══════════════════════════════════════════════════════════════════════════════

def render_density_field_3d(
    ax: plt.Axes,
    density_3d: np.ndarray,
    x_ticks_np: np.ndarray,
    y_ticks_np: np.ndarray,
    z_ticks_np: np.ndarray,
    positions: np.ndarray,
    max_density: float | None = None,
    layers: Sequence[dict] | None = None,
) -> None:
    """Render a GT-style density field: nested density shells + agent overlay.

    This is the common rendering used by Figure 1 of
    ``plot_jackdaw2_density_field`` and the multi-scale density figures.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.Axes3D
        Should already have ``view_init`` applied and ``set_axis_off()`` called.
    density_3d : (nx, ny, nz) float64
    x_ticks_np, y_ticks_np, z_ticks_np : 1D ndarray
    positions : (N, 3) float32/float64
    max_density : float, optional
    layers : list of dict, optional
    """
    if max_density is None:
        max_density = float(density_3d.max())

    render_density_shells(
        ax, density_3d, x_ticks_np, y_ticks_np, z_ticks_np,
        max_density=max_density, layers=layers,
    )
    render_agent_positions(ax, positions)


def render_reconstructed_gmm_3d(
    ax: plt.Axes,
    density_3d: np.ndarray,
    x_ticks_np: np.ndarray,
    y_ticks_np: np.ndarray,
    z_ticks_np: np.ndarray,
    positions: np.ndarray,
    means_np: np.ndarray,
    sigmas_np: np.ndarray,
    weights_np: np.ndarray,
    max_density: float | None = None,
    gmm_colour: str = "#4169e1",
) -> None:
    """Render a reconstructed GMM: density shells + wireframe ellipsoids +
    GMM mean markers + agent overlay.

    This is the composite used by Figure 2 of
    ``plot_jackdaw2_density_field`` and by ``_draw_gmm_frame``.
    """
    if max_density is None:
        max_density = float(density_3d.max())

    # 1. Density shells (reduced alpha so wireframes show through)
    render_density_shells(
        ax, density_3d, x_ticks_np, y_ticks_np, z_ticks_np,
        max_density=max_density, layers=FIELD_LAYERS,
    )

    # 2. GMM wireframe ellipsoids
    render_gmm_wireframes(
        ax, means_np, sigmas_np, weights_np,
        colour=gmm_colour, z_sort_pos=-5e8,
    )

    # 3. GMM mean markers
    render_gmm_means(ax, means_np, colour=gmm_colour, z_sort_pos=-6e8)

    # 4. Agent positions on top
    render_agent_positions(ax, positions, z_sort_pos=-1e9)
