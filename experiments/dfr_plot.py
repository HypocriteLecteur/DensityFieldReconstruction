import logging
import sys
import os
import shutil
import pickle
import tempfile

sys.path.append(os.getcwd())  # To get around relative import issues. I hate Python.

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from dfr.simulation_config import SimulationConfig
from dfr.dataset_io import DatasetFactory
from dfr.camera_system import MultiCameraSystem
from dfr.density_field_reconstructor import DensityReconstructor
from dfr.camera_state import CameraState
from dfr.utils import calculate_gmm_dissimilarity, generate_encircling_cameras, compute_metrics_batched_torch
from dfr.visualizer import MultiGMMPlotter
from dfr.gaussian_mixture_reduction import GMR
from dfr.mode_finding import find_target_scale, mode_counting, model_4pl_scale_at_x_constant, analytic_solution
from experiments.power_law import move_figure
from experiments.reconstruction_scale_determination import compute_scaling_law
from dfr.density_field_model import GaussianModel

# ── Module-level constants ────────────────────────────────────────────────────
CAM_NUM = 2
LOG_NAME = 'base_reg_cam_2'

DATASET_RUNS = [
    {
        'name': 'swift',
        'log_name': LOG_NAME,
        'start_step': 0,
        'end_step': None,
        'step_length': 200,
    },
    {
        'name': 'starling',
        'log_name': LOG_NAME,
        'start_step': 0,
        'end_step': None,
        'step_length': 1,
    },
    {
        'name': 'jackdaw',
        'log_name': LOG_NAME,
        'start_step': 350,
        'end_step': 550,
        'step_length': 10,
    },
    {
        'name': 'jackdaw2',
        'log_name': LOG_NAME,
        'start_step': 2700,
        'end_step': 3460,
        'step_length': 20,
    },
]

# ── Shared parameter presets ──────────────────────────────────────────────────
_OPTIMIZED_TRAIN_PARAMS = {
    'xyz_lr_c': 0.11550156892954913,     'xyz_lr_final_c': 0.015263086280830469,
    'radius_lr_c': 0.09585436467026787,  'radius_lr_final_c': 0.02420618007560584,
    'weights_lr_c': 0.19814963583342243, 'weights_lr_final_c': 0.7979132269720964,
    'xyz_reg': 0.21978381872642633,       'radius_reg': 0.6083537781516261,
    'radius_cutoff_inv': 0.6013595613763145,
    'lr_max_steps': 100,
}

_DEFAULT_TRAIN_PARAMS = {
    'xyz_lr_c': 0.05, 'xyz_lr_final_c': 0.9,
    'radius_lr_c': 0.05, 'radius_lr_final_c': 0.9,
    'weights_lr_c': 0.10, 'weights_lr_final_c': 0.7,
    'xyz_reg': 1.0, 'radius_reg': 0.3,
    'radius_cutoff_inv': 0.5, 'lr_max_steps': 500,
}

_BASE_RECONSTRUCTION_PARAMS = {
    'targetd_num_mode': 10,
    'voxel_scale': 0.5, 'voxel_peak_threshold': 0.3,
    'voxel_grid_max_size': 32, 'voxel_peaks_number': 2 * 10,
}

# ── Helper functions ──────────────────────────────────────────────────────────
def _unpack(rp):
    """Unpack a run_params dict into (name, log_name, start, end, step)."""
    return rp['name'], rp['log_name'], rp['start_step'], rp['end_step'], rp.get('step_length', 1)

def _load_scenario(name):
    """Return (config, dataset, scenario_path) for a named scenario."""
    sp = os.path.join(os.getcwd(), "scenarios", name)
    config = SimulationConfig(os.path.join(sp, "config.yaml"))
    dataset = DatasetFactory().get_dataset(config.data_file)
    return config, dataset, sp

def _step_range(dataset, start_step, end_step, step_length):
    """Return (max_steps, effective_end, range(start, effective_end, step_length))."""
    mx = dataset.trajectories.shape[0]
    eff_end = end_step if end_step is not None and end_step <= mx else mx
    return mx, eff_end, range(start_step, eff_end, step_length)

def _build_cam_system(dataset, step_range, config, cam_num, far_clip=200, device='cuda'):
    """Build a MultiCameraSystem, handling the 2-camera special case (generate 4, keep 2)."""
    if cam_num == 2:
        cam_positions, _ = generate_encircling_cameras(dataset, step_range, config.intrinsics_params, config.H, config.W, cam_num=4, padding=1)
        cam_poses = np.hstack((cam_positions[:2], np.tile(np.array([1, 0, 0, 0]), (2, 1)))).astype(np.float32)
    else:
        cam_positions, _ = generate_encircling_cameras(dataset, step_range, config.intrinsics_params, config.H, config.W, cam_num=cam_num, padding=1)
        cam_poses = np.hstack((cam_positions, np.tile(np.array([1, 0, 0, 0]), (cam_num, 1)))).astype(np.float32)
    return MultiCameraSystem.create_homogeneous_system(
        state_class=CameraState, intrinsics=config.intrinsics_params,
        H=config.H, W=config.W, poses_or_RTs=cam_poses,
        near_clip=config.near_clip, far_clip=far_clip, size=config.size, device=device,
    )

def _set_academic_style():
    """Apply publication-quality academic styling to matplotlib."""
    plt.rcParams.update({
        "font.family": "serif",  "mathtext.fontset": "cm",
        "font.size": 12, "axes.labelsize": 14, "axes.titlesize": 14,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
    })

def _style_3d_ax(ax):
    """Transparent panes, subtle grid, no edge color on a 3D axis."""
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis._axinfo["grid"].update({"color": (0.8, 0.8, 0.8, 0.5), "linewidth": 0.5})
        axis.pane.set_edgecolor('none')

# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting / analysis functions
# ═══════════════════════════════════════════════════════════════════════════════

def scale_estimation():
    for run_params in DATASET_RUNS:
        name, _, start_step, end_step, step_length = _unpack(run_params)
        config, dataset, scenario_path = _load_scenario(name)
        _, eff_end, step_range = _step_range(dataset, start_step, end_step, step_length)

        scale_range, all_modes, params = compute_scaling_law(dataset, step_range, scenario_path)
        scales_estim = np.load(scenario_path + '/reconstruction_scale_estim.npy')
        scales_estim_after_training = np.load(scenario_path + '/reconstruction_scale_estim_after_training.npy')

        N_ = [dataset.positions_at_time_step(step).shape[0] for step in step_range]
        k_, x0_ = params[:, 0], params[:, 1]
        scales_gt = [model_4pl_scale_at_x_constant(10, A=1, B=N, k=k, x0=x0) for (N, k, x0) in zip(N_, k_, x0_)]
        # np.save(scenario_path + '/reconstruction_scale.npy', np.array(scales_gt))

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        move_figure(fig, 2800, 100)
        ax.plot(np.array(step_range), scales_gt, label='gt')
        ax.plot(np.array(step_range), scales_estim, label='estim')
        ax.plot(np.array(step_range), scales_estim_after_training, label='estim(after)')
        ax.legend()
    plt.show()

def plot_multiple_scenarios():
    for run_params in DATASET_RUNS:
        plot_single_scenario_new(run_params)
    plt.show()

def plot_single_scenario_new(run_params):
    name, log_name, start_step, end_step, step_length = _unpack(run_params)
    config, dataset, scenario_path = _load_scenario(name)
    _, effective_end_step, _ = _step_range(dataset, start_step, end_step, step_length)

    log_file_path = os.path.join(scenario_path, *["logs", log_name])
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)

    # Get positions and active agents mask at the exact end step
    target_idx = effective_end_step - 1
    positions, masks = dataset.positions_at_time_step_mask(target_idx)
    trajectories = dataset.trajectories[:, masks, :]

    # Setup Figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    move_figure(fig, 100, 100)
    ax.view_init(elev=33, azim=-117, roll=0)

    # --- AESTHETICS: Clean, no axis, no grid ---
    ax.set_axis_off()
    fig.tight_layout(pad=0)

    N = trajectories.shape[1]

    # 1. Plot the full trajectories from start_step to end_step
    for i in range(N):
        ax.plot(
            trajectories[start_step:effective_end_step, i, 0],
            trajectories[start_step:effective_end_step, i, 1],
            trajectories[start_step:effective_end_step, i, 2],
            color='tab:gray', alpha=0.15, linewidth=1.2, zorder=1,
        )

    # 2. Plot the end positions of the swarm
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c='#1f2937', s=25, alpha=0.65, edgecolors='none', depthshade=True, zorder=3)

    # Optional: Save with zero padding to keep the image strictly focused on the data
    fig.savefig(f"figs/scene_traj_{name}.png", transparent=True, bbox_inches='tight', pad_inches=0)


def plot_jackdaw2_density_field():
    """
    Produce two separate publication figures for the jackdaw2 dataset at the
    same time step used in plot_single_scenario_new.

    Figure 1 — Ground-truth density field: agent positions convolved with an
    isotropic Gaussian kernel, rendered as nested density shells.

    Figure 2 — MV-DFR reconstructed GMM: the density field *and* the individual
    Gaussian components overlaid as wireframe ellipsoids, so the viewer sees
    both the field and the GMM representation.
    """
    from experiments.run_scenarios_angle_sweep import _build_grid, _precompute_gt_density
    from dfr.utils import eval_isotropic_gmm_torch
    import matplotlib.colors as mcolors

    # ── Shared setup ───────────────────────────────────────────────────────
    name = 'jackdaw2'
    start_step, end_step, step_length = 2700, 3460, 20
    config, dataset, scenario_path = _load_scenario(name)
    _, effective_end_step, step_range = _step_range(dataset, start_step, end_step, step_length)

    target_idx = effective_end_step - 1
    positions = dataset.positions_at_time_step(target_idx)
    print(f"Time step: {target_idx},  N={positions.shape[0]}")

    gt_data = np.load(os.path.join(scenario_path, 'reconstruction_scale.npz'))
    gt_scales = gt_data['scales_gt']
    scale = float(gt_scales[-1])
    print(f"Scale: {scale:.4f}")

    # ── Voxel grid & GT density ────────────────────────────────────────────
    print("Building voxel grid and evaluating GT density...")
    grid = _build_grid(positions, scale, voxel_res_factor=2.5e-2)
    density_flat = _precompute_gt_density(positions, scale, grid)
    density_3d = density_flat.numpy().reshape(grid['nx'], grid['ny'], grid['nz'])
    print(f"Grid: {grid['nx']}×{grid['ny']}×{grid['nz']} = {grid['total_voxels']:,} voxels")

    x_t = grid['x_ticks']
    y_t = grid['y_ticks']
    z_t = grid['z_ticks']
    nx, ny, nz = grid['nx'], grid['ny'], grid['nz']
    total = grid['total_voxels']

    x_t_np = x_t.cpu().numpy()
    y_t_np = y_t.cpu().numpy()
    z_t_np = z_t.cpu().numpy()
    dm = density_3d.max()

    # ── Load MV-DFR reconstructed model ────────────────────────────────────
    log_name = 'base_reg_cam_2'
    log_file_path = os.path.join(scenario_path, "logs", log_name)
    last_training_step = list(step_range)[-1]
    checkpoint_path = os.path.join(log_file_path, f"t_{last_training_step:03d}",
                                   "checkpoint_level_0.pth")
    print(f"Loading reconstructed model from: {checkpoint_path}")
    training_history = GaussianModel.load_training_history(checkpoint_path)
    model_recon = GaussianModel.load_iter(training_history, iter=99)
    K = model_recon._xyz.shape[0]
    print(f"Reconstructed model: {K} components")

    recon_means = model_recon._xyz.detach()
    recon_weights = model_recon._weights.detach().squeeze(-1)
    recon_sigmas = model_recon._radius.detach().squeeze(-1)

    # ── Evaluate reconstructed GMM density on same grid ────────────────────
    print("Evaluating reconstructed GMM density on grid...")
    recon_density_flat = torch.empty(total, dtype=torch.float32, device='cpu')
    batch_size = 50000
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        idx = torch.arange(start, end, device='cuda')
        ix = idx // (ny * nz)
        iy = (idx // nz) % ny
        iz = idx % nz
        coords = torch.stack([x_t[ix], y_t[iy], z_t[iz]], dim=-1)
        dens = eval_isotropic_gmm_torch(coords, recon_means, recon_weights, recon_sigmas)
        recon_density_flat[start:end] = dens.cpu()

    recon_density_3d = recon_density_flat.numpy().reshape(nx, ny, nz)
    rdm = recon_density_3d.max()
    print(f"Reconstructed max density: {rdm:.4f}  (GT max: {dm:.4f})")

    # ── Shared rendering parameters ────────────────────────────────────────
    layers = [
        {'thresh': dm * 0.10, 'alpha_min': 0.45, 'alpha_max': 0.95, 'size': 8},
        {'thresh': dm * 0.02, 'alpha_min': 0.25, 'alpha_max': 0.80, 'size': 6},
        {'thresh': dm * 0.002,'alpha_min': 0.08, 'alpha_max': 0.50, 'size': 4},
    ]
    view = dict(elev=33, azim=-117, roll=0)

    # ═══════════════════════════════════════════════════════════════════════
    #  Figure 1 — Ground Truth Density Field
    # ═══════════════════════════════════════════════════════════════════════
    fig1 = plt.figure(figsize=(10, 10))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.view_init(**view)
    ax1.set_axis_off()

    norm_gt = mcolors.PowerNorm(gamma=0.35, vmin=0, vmax=dm)

    for layer in layers:
        mask = density_3d >= layer['thresh']
        if not mask.any():
            continue
        ix, iy, iz = np.where(mask)
        pts = np.stack([x_t_np[ix], y_t_np[iy], z_t_np[iz]], axis=-1)
        vals = density_3d[mask]
        colors = plt.cm.viridis(norm_gt(vals))
        alphas = norm_gt(vals) * (layer['alpha_max'] - layer['alpha_min']) + layer['alpha_min']
        colors[:, 3] = np.clip(alphas, layer['alpha_min'], layer['alpha_max'])
        ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                    c=colors, s=layer['size'], edgecolors='none',
                    depthshade=False, rasterized=True)

    # Agent positions forced on top
    agent_coll_gt = ax1.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c='#1f2937', s=25, alpha=1.0, linewidths=0.8)
    _orig_gt = agent_coll_gt.do_3d_projection
    def _force_gt():
        _orig_gt()
        agent_coll_gt._sort_zpos = -1e9
        return agent_coll_gt._sort_zpos
    agent_coll_gt.do_3d_projection = _force_gt

    fig1.tight_layout(pad=0)
    out_gt = f"figs/scene_traj_{name}_density_gt.png"
    fig1.savefig(out_gt, transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig1)
    print(f"Saved {out_gt}")

    # ═══════════════════════════════════════════════════════════════════════
    #  Figure 2 — MV-DFR Reconstructed GMM
    #  Renders the *field* as nested density shells AND overlays wireframe
    #  ellipsoids for each Gaussian component so the GMM structure is visible.
    # ═══════════════════════════════════════════════════════════════════════
    fig2 = plt.figure(figsize=(10, 10))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.view_init(**view)
    ax2.set_axis_off()

    # --- 2a. Density field (reduced alpha so wireframes show through) -------
    # Use the same colour scale as GT but with lower alpha range so the
    # overlaid GMM wireframes remain visible.
    norm_rec = mcolors.PowerNorm(gamma=0.35, vmin=0, vmax=dm)
    field_layers = [
        {'thresh': dm * 0.10, 'alpha_min': 0.18, 'alpha_max': 0.55, 'size': 8},
        {'thresh': dm * 0.02, 'alpha_min': 0.10, 'alpha_max': 0.40, 'size': 6},
        {'thresh': dm * 0.002,'alpha_min': 0.04, 'alpha_max': 0.22, 'size': 4},
    ]

    for layer in field_layers:
        mask = recon_density_3d >= layer['thresh']
        if not mask.any():
            continue
        ix, iy, iz = np.where(mask)
        pts = np.stack([x_t_np[ix], y_t_np[iy], z_t_np[iz]], axis=-1)
        vals = recon_density_3d[mask]
        colors = plt.cm.viridis(norm_rec(vals))
        alphas = norm_rec(vals) * (layer['alpha_max'] - layer['alpha_min']) + layer['alpha_min']
        colors[:, 3] = np.clip(alphas, layer['alpha_min'], layer['alpha_max'])
        ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                    c=colors, s=layer['size'], edgecolors='none',
                    depthshade=False, rasterized=True)

    # --- 2b. GMM component wireframe ellipsoids -----------------------------
    # Each isotropic Gaussian is drawn as a wireframe sphere at radius = σ
    # (1 standard deviation).  Their do_3d_projection is monkey-patched to
    # force them in front of the density-field scatter.
    means_np = recon_means.cpu().numpy()
    sigmas_np = recon_sigmas.cpu().numpy()
    weights_np = recon_weights.cpu().numpy()
    w_max = weights_np.max() if weights_np.size > 0 else 1.0

    gmm_color = '#4169e1'  # royalblue — distinct from the viridis density field

    # Sphere mesh template (unit sphere)
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones(np.size(u)), np.cos(v))

    wireframe_objs = []
    for j in range(K):
        r = float(sigmas_np[j])
        alpha = max(0.15, min(0.70, weights_np[j] / w_max)) if w_max > 0 else 0.25
        rgba = (*mcolors.to_rgb(gmm_color), alpha)
        wf = ax2.plot_wireframe(
            means_np[j, 0] + r * sx,
            means_np[j, 1] + r * sy,
            means_np[j, 2] + r * sz,
            color=rgba,
            rstride=2, cstride=2, linewidth=1.7,
        )
        wireframe_objs.append(wf)

    # Monkey-patch each wireframe so it sorts in front of the density field
    # but behind the agent positions (which use _sort_zpos = -1e9) and GMM
    # mean markers (which use -6e8).  Use default-argument binding to capture
    # the correct `obj` per loop iteration; use a fixed value so repeated
    # calls to do_3d_projection (one per render) don't accumulate the offset.
    for wf in wireframe_objs:
        _orig_wf = wf.do_3d_projection
        def _patch(orig=_orig_wf, obj=wf):
            orig()
            obj._sort_zpos = -5e8
            return obj._sort_zpos
        wf.do_3d_projection = _patch

    # GMM means as small scatter markers (also forced in front of density)
    means_coll = ax2.scatter(means_np[:, 0], means_np[:, 1], means_np[:, 2],
                             c=gmm_color, marker='o', s=14, alpha=0.85,
                             edgecolors='none', depthshade=True)
    _orig_means = means_coll.do_3d_projection
    def _force_means():
        _orig_means()
        means_coll._sort_zpos = -6e8
        return means_coll._sort_zpos
    means_coll.do_3d_projection = _force_means

    # --- 2c. Agent positions forced on top of everything --------------------
    agent_coll_rec = ax2.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c='#1f2937', s=25, alpha=1.0, linewidths=0.8)
    _orig_rec = agent_coll_rec.do_3d_projection
    def _force_rec():
        _orig_rec()
        agent_coll_rec._sort_zpos = -1e9
        return agent_coll_rec._sort_zpos
    agent_coll_rec.do_3d_projection = _force_rec

    fig2.tight_layout(pad=0)
    out_rec = f"figs/scene_traj_{name}_density_recon.png"
    fig2.savefig(out_rec, transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig2)
    print(f"Saved {out_rec}")

    # ═══════════════════════════════════════════════════════════════════════
    #  Figure 3 — GMM wireframes only (no density field)
    # ═══════════════════════════════════════════════════════════════════════
    fig3 = plt.figure(figsize=(10, 10))
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.view_init(**view)
    ax3.set_axis_off()

    for j in range(K):
        r = float(sigmas_np[j])
        alpha = max(0.15, min(0.70, weights_np[j] / w_max)) if w_max > 0 else 0.25
        rgba = (*mcolors.to_rgb(gmm_color), alpha)
        ax3.plot_wireframe(
            means_np[j, 0] + r * sx,
            means_np[j, 1] + r * sy,
            means_np[j, 2] + r * sz,
            color=rgba,
            rstride=2, cstride=2, linewidth=1.7,
        )

    ax3.scatter(means_np[:, 0], means_np[:, 1], means_np[:, 2],
                c=gmm_color, marker='o', s=14, alpha=0.85,
                edgecolors='none', depthshade=True)

    fig3.tight_layout(pad=0)
    out_wf = f"figs/scene_traj_{name}_gmm_wireframes.png"
    fig3.savefig(out_wf, transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig3)
    print(f"Saved {out_wf}")


def plot_jackdaw2_2d_gmm(density_cutoff=1e-2, num_levels=8):
    """
    Project the MV-DFR reconstructed GMM into each camera view using the local
    affine approximation, evaluate the resulting 2D density on the image grid,
    and render it as filled contours.

    For each 3D isotropic Gaussian (mean μ_w, radius σ, weight w):
      1. Transform to camera coordinates:  μ_c = R·μ_w + T
      2. Pinhole projection:  (u₀, v₀) = project(μ_c, K)
      3. Jacobian J = ∂(u,v)/∂(X,Y,Z) evaluated at μ_c
      4. 2D covariance:  Σ₂ = σ² · J·Jᵀ
      5. The 2D density is the sum of weighted bivariate Gaussians N((u₀,v₀), Σ₂).

    Output — one figure per camera, each containing:
      - filled + line contours of the 2D GMM density (royalblue, transparent bg)
      - 1σ ellipses for each Gaussian component (royalblue)
      - projected mean positions (royalblue scatter)
    """
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm
    from matplotlib.patches import Ellipse
    import matplotlib.colors as mcolors

    # ── Shared setup (same as plot_jackdaw2_density_field) ──────────────────
    name = 'jackdaw2'
    start_step, end_step, step_length = 2700, 3460, 20
    config, dataset, scenario_path = _load_scenario(name)
    _, effective_end_step, step_range = _step_range(dataset, start_step, end_step, step_length)

    target_idx = effective_end_step - 1
    positions = dataset.positions_at_time_step(target_idx)
    print(f"Time step: {target_idx},  N={positions.shape[0]}")

    # --- Load MV-DFR reconstructed model -----------------------------------
    log_name = 'base_reg_cam_2'
    log_file_path = os.path.join(scenario_path, "logs", log_name)
    last_training_step = list(step_range)[-1]
    checkpoint_path = os.path.join(log_file_path, f"t_{last_training_step:03d}",
                                   "checkpoint_level_0.pth")
    print(f"Loading reconstructed model from: {checkpoint_path}")
    training_history = GaussianModel.load_training_history(checkpoint_path)
    model_recon = GaussianModel.load_iter(training_history, iter=99)
    K = model_recon._xyz.shape[0]
    print(f"Reconstructed model: {K} components")

    means_w = model_recon._xyz.detach()          # (K, 3)
    sigmas = model_recon._radius.detach().squeeze(-1)  # (K,)
    weights = model_recon._weights.detach().squeeze(-1)  # (K,)

    # --- Camera system (aim at swarm — same as training) -------------------
    cam_system = _build_cam_system(dataset, step_range, config, CAM_NUM)
    cam_system.aim_all_at_swarm(positions)

    # --- Colormap: transparent → royalblue ---------------------------------
    top = np.array([0.255, 0.412, 0.882, 1.0])   # opaque royalblue
    bot = np.array([1.0,   1.0,   1.0,   0.0])    # transparent white
    n = 256
    alphas_c = np.linspace(0, 1, n)
    rgb = bot[:3] + (top[:3] - bot[:3]) * alphas_c[:, None]
    rgba = np.hstack([rgb, alphas_c[:, None]])
    cmap_transp = LinearSegmentedColormap.from_list('transp_blue', rgba)

    for i, cam in enumerate(cam_system.cameras):
        H_i, W_i = cam.state.H, cam.state.W
        device = means_w.device

        # Camera extrinsics / intrinsics on GPU
        R = cam.state.R  # (3, 3)
        T = cam.state.T  # (3,)
        fx = cam.state.intrinsics_params[0, 0].item()
        fy = cam.state.intrinsics_params[1, 1].item()
        cx = cam.state.intrinsics_params[0, 2].item()
        cy = cam.state.intrinsics_params[1, 2].item()

        # --- Project each 3D Gaussian to a 2D affine Gaussian ---------------
        # Step 1: world → camera
        mu_c = (R @ means_w.T).T + T                   # (K, 3)
        X, Y, Z = mu_c[:, 0], mu_c[:, 1], mu_c[:, 2]  # each (K,)

        # Filter components behind or too close to the camera
        visible = (Z > cam.state.near_clip) & (Z < cam.state.far_clip)
        # Also filter components that project far outside the image
        u0_all = fx * X / Z + cx
        v0_all = fy * Y / Z + cy
        in_bounds = (u0_all > -0.5 * W_i) & (u0_all < 1.5 * W_i) & \
                    (v0_all > -0.5 * H_i) & (v0_all < 1.5 * H_i)
        keep = visible & in_bounds

        if keep.sum() == 0:
            print(f"  Camera {i+1}: no visible components — skipping")
            continue

        # Subset to visible components
        X, Y, Z = X[keep], Y[keep], Z[keep]
        u0 = u0_all[keep]
        v0 = v0_all[keep]
        s2 = (sigmas[keep] ** 2)
        w_vis = weights[keep]
        K_vis = keep.sum().item()
        print(f"  Camera {i+1}: {K_vis}/{K} components visible")

        # Step 3: Jacobian J = ∂(u,v)/∂(X,Y,Z) at μ_c
        # J = [[fx/Z,   0,   -fx*X/Z²],
        #      [  0,  fy/Z,  -fy*Y/Z²]]
        invZ = 1.0 / Z
        J11 = fx * invZ
        J13 = -fx * X * invZ * invZ
        J22 = fy * invZ
        J23 = -fy * Y * invZ * invZ

        # Step 4: 2D covariance Σ₂ = σ² · J·Jᵀ  (isotropic 3D → anisotropic 2D)
        # J = [[J11, 0, J13], [0, J22, J23]]
        # J·Jᵀ = [[J11²+J13²,  J13·J23], [J13·J23,  J22²+J23²]]
        cov_xx = s2 * (J11 * J11 + J13 * J13)          # (K_vis,)
        cov_xy = s2 * (J13 * J23)                       # (K_vis,)
        cov_yy = s2 * (J22 * J22 + J23 * J23)          # (K_vis,)

        # Determinant & inverse for density evaluation
        det_cov = cov_xx * cov_yy - cov_xy * cov_xy    # (K_vis,)
        inv_xx = cov_yy / det_cov                       # (K_vis,)
        inv_xy = -cov_xy / det_cov                      # (K_vis,)
        inv_yy = cov_xx / det_cov                       # (K_vis,)
        norm_2d = w_vis / (2.0 * np.pi * torch.sqrt(det_cov.clamp(min=1e-30)))

        # --- Evaluate 2D GMM density on the image grid ----------------------
        y_px = torch.arange(H_i, device=device, dtype=torch.float32)
        x_px = torch.arange(W_i, device=device, dtype=torch.float32)
        YY, XX = torch.meshgrid(y_px, x_px, indexing='ij')  # (H, W)

        density_2d = torch.zeros(H_i, W_i, device=device)
        batch = 4  # process components in small batches to avoid OOM
        for start in range(0, K_vis, batch):
            end = min(start + batch, K_vis)
            du = XX.unsqueeze(0) - u0[start:end, None, None]  # (B, H, W)
            dv = YY.unsqueeze(0) - v0[start:end, None, None]  # (B, H, W)
            quad = (inv_xx[start:end, None, None] * du * du
                    + 2.0 * inv_xy[start:end, None, None] * du * dv
                    + inv_yy[start:end, None, None] * dv * dv)
            contrib = norm_2d[start:end, None, None] * torch.exp(-0.5 * quad)
            density_2d += contrib.sum(dim=0)

        density_np = density_2d.cpu().numpy()
        u0_np = u0.cpu().numpy()
        v0_np = v0.cpu().numpy()
        cov_xx_np = cov_xx.cpu().numpy()
        cov_xy_np = cov_xy.cpu().numpy()
        cov_yy_np = cov_yy.cpu().numpy()

        vmax = density_np.max()
        min_level = vmax * density_cutoff
        levels = np.geomspace(min_level, vmax, num_levels)
        print(f"  density range: [{density_np.min():.4g}, {vmax:.4g}], "
              f"levels: [{levels[0]:.4g} .. {levels[-1]:.4g}]")

        # --- Figure: 2D GMM density contours + component ellipses -----------
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        # Filled + line contours
        norm = PowerNorm(gamma=0.40, vmin=0, vmax=vmax)
        ax.contourf(np.arange(W_i), np.arange(H_i), density_np,
                    levels=levels, cmap=cmap_transp, norm=norm,
                    antialiased=True, zorder=1)

        # 1σ ellipses — black dashed, alpha ∝ weight
        w_np = w_vis.cpu().numpy()
        w_rel = w_np / w_np.max() if w_np.max() > 0 else np.ones_like(w_np)
        for j in range(K_vis):
            cxx, cxy, cyy = cov_xx_np[j], cov_xy_np[j], cov_yy_np[j]
            trace = cxx + cyy
            det = cxx * cyy - cxy * cxy
            if det <= 0:
                continue
            eig1 = 0.5 * (trace + np.sqrt(max(trace * trace - 4.0 * det, 0)))
            eig2 = 0.5 * (trace - np.sqrt(max(trace * trace - 4.0 * det, 0)))
            if eig1 <= 0 or eig2 <= 0:
                continue
            a = np.sqrt(eig1)
            b = np.sqrt(eig2)
            angle = np.degrees(np.arctan2(cxy, eig1 - cyy))
            alpha_j = 0.15 + 0.70 * w_rel[j]
            ell = Ellipse((u0_np[j], v0_np[j]), 2 * a, 2 * b, angle=angle,
                          facecolor='none', edgecolor='black',
                          linestyle='--', linewidth=1.0,
                          alpha=float(alpha_j), zorder=3)
            ax.add_patch(ell)

        ax.set_xlim(0, W_i - 1)
        ax.set_ylim(H_i - 1, 0)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout(pad=0)
        fig.savefig(f"figs/scene_traj_{name}_cam{i+1}_gmm2d.png",
                    dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)
        print(f"Camera {i+1}: saved 2D GMM projection")


def plot_jackdaw2_2d_observations(density_cutoff=1e-2, num_levels=8):
    """
    Produce four clean, publication-ready figures showing the 2D observations
    for the jackdaw2 dataset (same time step as the 3D density plots).

    Parameters
    ----------
    density_cutoff : float
        Fraction of vmax below which density is treated as zero.
        Default 1e-3 → lowest contour at 0.1 % of peak.
    num_levels : int
        Number of log-spaced contour levels.  Default 16.

    Output (4 separate figures, no titles / ticks / borders):
      - Camera 1  — 2D projected positions (royalblue scatter)
      - Camera 1  — coarse-grained density contour (transparent bg)
      - Camera 2  — 2D projected positions (royalblue scatter)
      - Camera 2  — coarse-grained density contour (transparent bg)

    The coarse-grained images use the reconstruction scale from gt_scales so
    the Gaussian-blob widths match the target resolution.
    """
    from dfr.camera_system import convolution_cupy_wrapper
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm

    name = 'jackdaw2'
    start_step, end_step, step_length = 2700, 3460, 20
    config, dataset, scenario_path = _load_scenario(name)
    _, effective_end_step, step_range = _step_range(dataset, start_step, end_step, step_length)

    target_idx = effective_end_step - 1
    positions = dataset.positions_at_time_step(target_idx)
    print(f"Time step: {target_idx},  N={positions.shape[0]}")

    # --- Reconstruction scale (same as used for GT density) ------------------
    gt_data = np.load(os.path.join(scenario_path, 'reconstruction_scale.npz'))
    gt_scales = gt_data['scales_gt']
    scale = float(gt_scales[-1])
    print(f"Scale: {scale:.4f}")

    # --- Camera system (same as training: 2 encircling cameras) -------------
    cam_system = _build_cam_system(dataset, step_range, config, CAM_NUM)

    # --- 2D projections ----------------------------------------------------
    _, point_sets, _, _ = cam_system.simulate_vision(positions, renderer='projection_only')

    # --- Coarse-grained images at reconstruction scale ----------------------
    center = np.mean(positions, axis=0)
    coarse_images = []
    for i, cam in enumerate(cam_system.cameras):
        points_2d = torch.tensor(point_sets[i], dtype=torch.float32).cuda()
        dist_cam = float(np.linalg.norm(cam.state.camera_center - center))
        # pixel-space radius matching the target reconstruction scale
        r_px = float(scale / dist_cam * cam.state.intrinsics_params[0, 0].item())
        H, W = cam.state.H, cam.state.W
        img = convolution_cupy_wrapper(points_2d, r_px, H, W, sigma_multiple=4.0)
        coarse_images.append(img.cpu().numpy())

    # --- Colormap: transparent → royalblue ---------------------------------
    # Build a custom colormap whose lowest value is fully transparent so the
    # figure background shows through low-density regions.
    top = np.array([0.255, 0.412, 0.882, 1.0])   # opaque royalblue
    bot = np.array([1.0,   1.0,   1.0,   0.0])    # transparent white
    n = 256
    alphas = np.linspace(0, 1, n)
    rgb = bot[:3] + (top[:3] - bot[:3]) * alphas[:, None]
    rgba = np.hstack([rgb, alphas[:, None]])
    cmap_transp = LinearSegmentedColormap.from_list('transp_blue', rgba)

    # --- Produce four separate, clean figures -------------------------------
    for i in range(len(cam_system.cameras)):
        proj = point_sets[i]
        H_i, W_i = cam_system.cameras[i].state.H, cam_system.cameras[i].state.W
        img = coarse_images[i]
        vmax = img.max()

        # log-spaced contour levels — dense near zero, sparse near max
        min_level = vmax * density_cutoff
        levels = np.geomspace(min_level, vmax, num_levels)

        # ---- Figure A: 2D projected positions (scatter) -------------------
        fig_s = plt.figure(figsize=(8, 6))
        ax_s = fig_s.add_subplot(111)
        ax_s.scatter(proj[:, 0], proj[:, 1],
                     c='royalblue', s=10, alpha=0.65, edgecolors='none')
        ax_s.set_xlim(0, W_i)
        ax_s.set_ylim(H_i, 0)
        ax_s.set_aspect('equal')
        ax_s.set_xticks([])
        ax_s.set_yticks([])
        for spine in ax_s.spines.values():
            spine.set_visible(False)
        fig_s.tight_layout(pad=0)
        fig_s.savefig(f"figs/scene_traj_{name}_cam{i+1}_projections.png",
                      dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig_s)

        # ---- Figure B: filled contour plot of the 2D density ---------------
        fig_i = plt.figure(figsize=(8, 6))
        ax_i = fig_i.add_subplot(111)
        y_px = np.arange(H_i)
        x_px = np.arange(W_i)
        norm = PowerNorm(gamma=0.40, vmin=0, vmax=vmax)
        ax_i.contourf(x_px, y_px, img, levels=levels,
                      cmap=cmap_transp, norm=norm, antialiased=True)
        ax_i.contour(x_px, y_px, img, levels=levels,
                     colors='#4169e1', linewidths=0.3, alpha=0.5)
        ax_i.set_xlim(0, W_i - 1)
        ax_i.set_ylim(H_i - 1, 0)
        ax_i.set_aspect('equal')
        ax_i.set_xticks([])
        ax_i.set_yticks([])
        for spine in ax_i.spines.values():
            spine.set_visible(False)
        fig_i.tight_layout(pad=0)
        fig_i.savefig(f"figs/scene_traj_{name}_cam{i+1}_coarse.png",
                      dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig_i)


def plot_single_scenario(run_params):
    name, log_name, start_step, end_step, step_length = _unpack(run_params)
    config, dataset, scenario_path = _load_scenario(name)
    _, effective_end_step, _ = _step_range(dataset, start_step, end_step, step_length)

    step_range = np.arange(start_step, effective_end_step, step_length)

    positions, masks = dataset.positions_at_time_step_mask(step_range[-1])
    trajectories = dataset.trajectories[:, masks, :]

    log_file_path = os.path.join(scenario_path, *["logs", log_name])
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)

    checkpoint_path = os.path.join(log_file_path, f"t_{step_range[-1]:03d}", f"checkpoint_level_0.pth")
    training_history = GaussianModel.load_training_history(checkpoint_path)
    GM_1 = GaussianModel.load_iter(training_history, 99)

    # gmm_visualizer = MultiGMMPlotter()
    # gmm_visualizer.add_gmm(GM_1._xyz.detach().cpu().numpy(), GM_1._radius.detach().cpu().numpy(), GM_1._weights.detach().cpu().numpy())
    # gmm_visualizer.update()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    move_figure(fig, 100, 100)
    ax.view_init(elev=33, azim=-117, roll=0)

    N = trajectories.shape[1]

    # 1. Define a tail length to prevent the "spaghetti" effect.
    # Adjust this scalar based on your dataset's framerate and desired visual trail.
    tail_length = 40
    # The step_range[-1] is your current target frame. We only want the history leading up to it.
    current_idx = step_range[-1]
    start_idx = max(0, current_idx - tail_length)

    for i in range(N):
        ax.plot(
            trajectories[start_idx:current_idx, i, 0],
            trajectories[start_idx:current_idx, i, 1],
            trajectories[start_idx:current_idx, i, 2],
            color='tab:gray', alpha=0.15, linewidth=0.6, zorder=1,
        )

    # 3. Plot the current positions of the swarm
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c='#1f2937', s=6, alpha=0.65, edgecolors='none', depthshade=True, zorder=3)

    plt.show()
    # fig.savefig(f"figs/scene_sample_{name}.png", transparent=True, bbox_inches='tight')
    plt.show()

def overview_scaling_law():
    torch.manual_seed(123456)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # 1. Generate points in a 2D SQUARE domain [-1, 1]^2 (Z is eliminated)
    num_points = 50  # Lowered point count so individual peaks are easy to see
    points = (torch.rand((num_points, 2), device=device) * 2.0) - 1.0

    # 2. Define the 2D grid
    grid_resolution = 200
    x = np.linspace(-1, 1, grid_resolution)
    y = np.linspace(-1, 1, grid_resolution)
    X, Y = np.meshgrid(x, y)
    grid_coords = torch.tensor(np.vstack([X.ravel(), Y.ravel()]).T, dtype=torch.float32, device=device)

    # 3. Define the scales
    scales = [0.25, 0.1, 0.05]
    stack_spacing = 2.5

    fig = plt.figure(figsize=(12, 10), dpi=300)
    move_figure(fig, 2800, 100)
    ax = fig.add_subplot(111, projection='3d')
    colormaps = ['viridis', 'viridis', 'viridis']

    for i, sigma in enumerate(scales):
        sq_dists = torch.cdist(grid_coords, points, p=2.0).pow(2)
        kernel_vals = torch.exp(-sq_dists / (2 * sigma**2))
        # CORRECTED FOR 2D: normalization is 1 / (2 * pi * sigma^2)
        normalization_factor = 1.0 / (num_points * (2 * torch.pi * sigma**2))
        densities = torch.sum(kernel_vals, dim=1) * normalization_factor
        density_grid = densities.cpu().numpy().reshape(grid_resolution, grid_resolution)
        density_grid = density_grid / density_grid.max()
        Z_plot = (i * stack_spacing) + density_grid
        ax.plot_surface(X, Y, Z_plot, cmap=colormaps[i % len(colormaps)], alpha=0.85, linewidth=0, antialiased=True)
        ax.text(-1.2, -1.2, i * stack_spacing - 0.3, f"σ = {sigma}", color='black', fontsize=12)

    # Plot Ground Truth
    top_z_base = len(scales) * stack_spacing
    ax.plot_surface(X, Y, np.full_like(X, top_z_base), color='gray', alpha=0.15)
    points_cpu = points.cpu().numpy()
    ax.scatter(points_cpu[:, 0], points_cpu[:, 1], top_z_base, color='black', s=25, alpha=1.0, edgecolors='none', zorder=10)

    ax.view_init(elev=15, azim=-60)
    # ax.set_xlabel('X Domain')
    # ax.set_ylabel('Y Domain')
    # ax.set_zlabel('True Density (Stacked)')
    # ax.set_title('2D Gaussian Scale Space (Coarse to Fine + GT)', pad=20)
    ax.set_zticklabels([])

    plt.tight_layout()
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(True, which='major', linestyle=':', alpha=0.5)
    fig.savefig(f"figs/2d_gss.png", transparent=True, bbox_inches='tight')
    plt.show()

def plot_scale_space_curve():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 14,
        "axes.labelsize": 14, "legend.fontsize": 12,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    })

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    # move_figure(fig, 2800, 100)

    n_points = 50
    target_scales = np.array([0.25, 0.1, 0.05])
    test_scales = np.logspace(-2, 0, 40)

    y_values = analytic_solution(test_scales / 2, n_points)
    target_y_values = analytic_solution(target_scales / 2, n_points)
    zeros = np.zeros_like(target_scales)

    ax.plot(test_scales, y_values, color='#2c3e50', lw=2, label='Analytic Solution')

    line_color = '#e74c3c'
    ax.hlines(y=target_y_values, xmin=zeros, xmax=target_scales, colors=line_color, linestyles='--', alpha=0.8)
    ax.vlines(x=target_scales, ymin=zeros, ymax=target_y_values, colors=line_color, linestyles='--', alpha=0.8)

    offset_multiplier = 1.15
    for scale, y_val in zip(target_scales, target_y_values):
        ax.text(scale * offset_multiplier, y_val, f"σ = {scale}",
                color='black', fontsize=12, verticalalignment='center')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_ylabel('Number of Modes', fontsize=12)
    ax.set_xlabel(r'Scale ($\sigma$)', fontsize=12)

    fig.tight_layout()
    fig.savefig("figs/2d_gss_curve.png", transparent=True, bbox_inches='tight')
    # plt.show()

def visual_hull_diagram():
    run_params = {'name': 'swift', 'log_name': LOG_NAME, 'start_step': 0, 'end_step': None, 'step_length': 200}
    name, log_name, start_step, end_step, step_length = _unpack(run_params)
    config, dataset, scenario_path = _load_scenario(name)
    _, _, step_range = _step_range(dataset, start_step, end_step, step_length)

    cam_positions, cam_radius = generate_encircling_cameras(dataset, step_range, config.intrinsics_params, config.H, config.W, cam_num=4, padding=1)
    cam_poses = np.hstack((cam_positions[:2], np.tile(np.array([1, 0, 0, 0]), (2, 1)))).astype(np.float32)

    cam_system = MultiCameraSystem.create_homogeneous_system(
        state_class=CameraState, intrinsics=config.intrinsics_params,
        H=config.H, W=config.W, poses_or_RTs=cam_poses,
        near_clip=config.near_clip, far_clip=config.far_clip, size=config.size, device='cuda',
    )
    reconstruction_params = {
        'targetd_num_mode': 10, 'voxel_scale': 0.5, 'voxel_peak_threshold': 0.3,
        'voxel_grid_max_size': 32, 'voxel_peaks_number': 2 * 10,
    }
    train_params = {
        'xyz_lr_c': 0.05, 'xyz_lr_final_c': 0.9,
        'radius_lr_c': 0.05, 'radius_lr_final_c': 0.9,
        'weights_lr_c': 0.10, 'weights_lr_final_c': 0.7,
        'xyz_reg': 1.0, 'radius_reg': 0.3, 'radius_cutoff_inv': 0.5, 'lr_max_steps': 500,
    }
    density_reconstructor = DensityReconstructor(max_iter=train_params['lr_max_steps'])

    time_step = step_range[0]
    positions = dataset.positions_at_time_step(time_step)
    poses, projections, _, masks = cam_system.simulate_vision(positions, renderer='projection_only')

    model, scale_spaces = \
        density_reconstructor.process_frame(cam_system, point_sets=projections, positions=positions,
                                            initGMM=None, is_adaptive_scale=True, scale=None,
                                            debug=True, train_params=train_params,
                                            reconstruction_params=reconstruction_params)

def assumption_3_error():
    def compute_exact_density(U_grid, V_grid, mu_3d, r):
        mu_x, mu_y, mu_z = mu_3d
        var = r**2
        norm = 1.0 / ((2 * np.pi * var) ** 1.5)
        z_min = max(0.001, mu_z - 6 * r)
        z_max = mu_z + 6 * r
        z_array = np.linspace(z_min, z_max, 200)
        U_exp, V_exp = U_grid[..., np.newaxis], V_grid[..., np.newaxis]
        z_exp = z_array[np.newaxis, np.newaxis, :]
        dx, dy, dz = U_exp * z_exp - mu_x, V_exp * z_exp - mu_y, z_exp - mu_z
        integrand = norm * np.exp(-(dx**2 + dy**2 + dz**2) / (2 * var)) * (z_exp**2)
        trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
        return trapz(integrand, x=z_array, axis=-1)

    def compute_affine_density(U_grid, V_grid, mu_3d, r):
        u0, v0, z0 = mu_3d[0] / mu_3d[2], mu_3d[1] / mu_3d[2], mu_3d[2]
        J = np.array([[1/z0, 0, -u0/z0], [0, 1/z0, -v0/z0]])
        Sigma_2d = (r**2) * (J @ J.T)
        inv_Sigma, det_Sigma = np.linalg.inv(Sigma_2d), np.linalg.det(Sigma_2d)
        dU, dV = U_grid - u0, V_grid - v0
        exponent = -0.5 * (dU**2 * inv_Sigma[0,0] + 2 * dU * dV * inv_Sigma[0,1] + dV**2 * inv_Sigma[1,1])
        return (1.0 / (2 * np.pi * np.sqrt(det_Sigma))) * np.exp(exponent)

    def compute_ortho_density(U_grid, V_grid, mu_3d, r, z_bar=1.0, f=1.0):
        u0, v0 = mu_3d[0] / mu_3d[2], mu_3d[1] / mu_3d[2]
        sigma_prime = (f / z_bar) * r
        var_2d = sigma_prime**2
        dU, dV = U_grid - u0, V_grid - v0
        return (1.0 / (2 * np.pi * var_2d)) * np.exp(-0.5 * (dU**2 + dV**2) / var_2d)

    def compute_hybrid_affine_density(U_grid, V_grid, mu_3d, r, z_bar=1.0):
        # 1. Find the 2D projection center (same ray as the exact/affine models)
        u0, v0 = mu_3d[0] / mu_3d[2], mu_3d[1] / mu_3d[2]
        # 2. Treat as if sitting at depth z_bar using the affine model.
        J = np.array([[1/z_bar, 0, -u0/z_bar], [0, 1/z_bar, -v0/z_bar]])
        Sigma_2d = (r**2) * (J @ J.T)
        inv_Sigma, det_Sigma = np.linalg.inv(Sigma_2d), np.linalg.det(Sigma_2d)
        dU, dV = U_grid - u0, V_grid - v0
        exponent = -0.5 * (dU**2 * inv_Sigma[0,0] + 2 * dU * dV * inv_Sigma[0,1] + dV**2 * inv_Sigma[1,1])
        return (1.0 / (2 * np.pi * np.sqrt(det_Sigma))) * np.exp(exponent)

    focal_length = 1.0
    z_bar = 1.0
    z_vals = np.linspace(0.8, 1.2, 20)
    u_vals = np.linspace(0.0, 0.5, 20)
    # r_vals = np.linspace(0.01, 0.2, 4)
    r_vals = [0.1]

    u_flat, z_flat, r_flat = [], [], []
    err_affine_flat, err_ortho_flat, err_hybrid_flat = [], [], []

    for Z in z_vals:
        for U in u_vals:
            for r in r_vals:
                mu_3d = np.array([U * Z, 0.0, Z])
                sigma_prime = (focal_length / z_bar) * r
                grid_range = 4 * sigma_prime
                grid_res = sigma_prime / 8.0
                u_centers = np.arange(U - grid_range, U + grid_range, grid_res)
                v_centers = np.arange(0.0 - grid_range, 0.0 + grid_range, grid_res)
                U_grid, V_grid = np.meshgrid(u_centers, v_centers)

                D_exact = compute_exact_density(U_grid, V_grid, mu_3d, r)
                D_affine = compute_affine_density(U_grid, V_grid, mu_3d, r)
                D_ortho = compute_ortho_density(U_grid, V_grid, mu_3d, r, z_bar, focal_length)
                D_hybrid = compute_hybrid_affine_density(U_grid, V_grid, mu_3d, r, z_bar)

                pixel_area = grid_res**2
                e_aff = np.sum(np.abs(D_exact - D_affine)) * pixel_area
                e_ortho = np.sum(np.abs(D_exact - D_ortho)) * pixel_area
                e_hybrid = np.sum(np.abs(D_exact - D_hybrid)) * pixel_area

                u_flat.append(U); z_flat.append(Z); r_flat.append(r)
                err_affine_flat.append(e_aff); err_ortho_flat.append(e_ortho); err_hybrid_flat.append(e_hybrid)

    u_flat = np.array(u_flat); z_flat = np.array(z_flat); r_flat = np.array(r_flat)
    err_affine_flat = np.array(err_affine_flat); err_ortho_flat = np.array(err_ortho_flat); err_hybrid_flat = np.array(err_hybrid_flat)

    _set_academic_style()

    def plot_3d_surface_error(err_flat, name):
        fig1 = plt.figure(figsize=(7, 6))
        move_figure(fig1, 2800, 100)
        ax1 = fig1.add_subplot(111, projection='3d')
        min_r = np.min(r_flat)
        mask = (r_flat == min_r)
        grid_shape = (len(z_vals), len(u_vals))
        angles = (np.atan(u_flat[mask]) * 180 / np.pi).reshape(grid_shape)
        z_plot = z_flat[mask].reshape(grid_shape)
        err_plot = err_flat[mask].reshape(grid_shape)
        surf1 = ax1.plot_surface(angles, z_plot, err_plot, cmap='viridis', edgecolor='none', alpha=0.9)
        ax1.set_xlabel(r'$\theta$ (Deg)'); ax1.set_ylabel(r'Depth $Z$'); ax1.set_zlabel(r'Error $\mathcal{E}$')
        _style_3d_ax(ax1)
        cb1 = fig1.colorbar(surf1, ax=ax1, shrink=0.5, pad=0.1)
        cb1.set_label(r'L1 Error', rotation=270, labelpad=20)
        fig1.tight_layout()
        fig1.savefig(f'{name}_error_surface.png')

    # plot_3d_surface_error(err_affine_flat, 'affine')
    # plot_3d_surface_error(err_ortho_flat, 'ortho')
    # plot_3d_surface_error(err_hybrid_flat, 'hybrid')

    def plot_2d_map_error(err_flat, name):
        fig1 = plt.figure(figsize=(7, 6))
        move_figure(fig1, 2800, 100)
        ax1 = fig1.add_subplot(111)
        min_r = np.min(r_flat)
        mask = (r_flat == min_r)
        grid_shape = (len(z_vals), len(u_vals))
        angles = (np.atan(u_flat[mask]) * 180 / np.pi).reshape(grid_shape)
        z_plot = z_flat[mask].reshape(grid_shape)
        err_plot = err_flat[mask].reshape(grid_shape)
        map1 = ax1.contourf(angles, z_plot - z_bar, err_plot, levels=50, cmap='viridis')
        ax1.set_xlabel(r'$\theta$ (Deg)'); ax1.set_ylabel(r'$\delta$z')
        cb1 = fig1.colorbar(map1, ax=ax1, pad=0.05)
        # cb1.set_label(r'Error $\mathcal{E}$', rotation=90, labelpad=20)
        fig1.tight_layout()
        fig1.savefig(f'{name}_error_2d_map.png')

    plot_2d_map_error(err_affine_flat, 'affine')
    plot_2d_map_error(err_ortho_flat, 'ortho')
    # plot_2d_map_error(err_hybrid_flat, 'hybrid')
    plt.show()

def visual_hull_tau_vs_visual_hull_ghost():
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times", "Times New Roman"],
        "mathtext.fontset": "cm",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
    })

    c1 = np.array([0.0, 1.0])
    c2 = np.array([12.0, 1.0])
    points = np.array([
        [3.6, 4.7], [4.2, 5.9], [4.9, 7.5], [4.6, 4.1],
        [5.2, 5.1], [5.9, 6.5], [6.6, 8.2], [5.6, 3.7],
        [6.2, 4.7], [7.0, 6.1], [7.8, 7.5], [6.9, 3.2],
        [7.5, 4.5], [8.4, 5.7],
    ])

    res = 1200
    x_min, x_max = -1, 13
    y_min, y_max = 0, 11.5
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)
    X, Y = np.meshgrid(x, y)
    grid_pts = np.c_[X.ravel(), Y.ravel()]

    def proj_angle(pts, camera):
        vec = pts - camera
        return np.arctan2(vec[:, 1], vec[:, 0])

    tau = 0.05
    theta_p1 = proj_angle(points, c1)
    theta_p2 = proj_angle(points, c2)
    theta_g1 = proj_angle(grid_pts, c1)
    theta_g2 = proj_angle(grid_pts, c2)

    in_any_cone1 = np.zeros(len(grid_pts), dtype=bool)
    in_any_cone2 = np.zeros(len(grid_pts), dtype=bool)
    vh_neigh = np.zeros(len(grid_pts), dtype=bool)

    for i in range(len(points)):
        cone1_i = np.abs(theta_g1 - theta_p1[i]) <= tau
        cone2_i = np.abs(theta_g2 - theta_p2[i]) <= tau
        in_any_cone1 |= cone1_i
        in_any_cone2 |= cone2_i
        vh_neigh |= (cone1_i & cone2_i)

    vh_tau = in_any_cone1 & in_any_cone2
    vh_ghost = vh_tau & (~vh_neigh)
    VH_neigh_img = vh_neigh.reshape(X.shape)
    VH_ghost_img = vh_ghost.reshape(X.shape)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    # fig.patch.set_facecolor('#F8F9FA'); ax.set_facecolor('#F8F9FA')
    # gx, gy = np.meshgrid(np.arange(0, 13, 0.5), np.arange(0, 12, 0.5))
    # ax.scatter(gx, gy, color='#E5E7EB', s=2, zorder=0)

    color_ghost = '#E78875'; color_neigh = '#7CB1A1'; color_rays = '#6B7A93'
    fill_alpha = 0.35; hatch_alpha = 0.85

    # --- Region 1: Ghosts (Spurious Geometries) ----
    ax.contourf(X, Y, VH_ghost_img, levels=[0.5, 1.5], colors=[color_ghost], alpha=fill_alpha, zorder=2)
    with plt.rc_context({'hatch.color': color_ghost, 'hatch.linewidth': 0.8}):
        ax.contourf(X, Y, VH_ghost_img, levels=[0.5, 1.5], colors=[(1, 1, 1, 0)], hatches=['////'], alpha=hatch_alpha, zorder=3)
    ax.contour(X, Y, VH_ghost_img, levels=[0.5], colors=[color_ghost], linewidths=1.5, alpha=1.0, zorder=4)

    # --- Region 2: Neighborhood (Actual Set) ---
    ax.contourf(X, Y, VH_neigh_img, levels=[0.5, 1.5], colors=[color_neigh], alpha=fill_alpha, zorder=5)
    with plt.rc_context({'hatch.color': color_neigh, 'hatch.linewidth': 0.8}):
        ax.contourf(X, Y, VH_neigh_img, levels=[0.5, 1.5], colors=[(1, 1, 1, 0)], hatches=['\\\\\\\\'], alpha=hatch_alpha, zorder=6)
    ax.contour(X, Y, VH_neigh_img, levels=[0.5], colors=[color_neigh], linewidths=1.5, alpha=1.0, zorder=7)

    # Plot actual agents
    # ax.scatter(points[:, 0], points[:, 1], c=color_neigh, edgecolor='None', s=250, alpha=0.3, zorder=8)
    ax.scatter(points[:, 0], points[:, 1], c='#1A1A1A', edgecolor='white', linewidth=1.2, s=70, zorder=9, label='Actual Agents ($p_i$)')

    # Draw Visual Cones, Cameras, and Individual Rays
    min_t1, max_t1 = np.min(theta_p1) - tau, np.max(theta_p1) + tau
    min_t2, max_t2 = np.min(theta_p2) - tau, np.max(theta_p2) + tau
    ray_length = 20

    for (c, min_t, max_t) in [(c1, min_t1, max_t1), (c2, min_t2, max_t2)]:
        ax.plot([c[0], c[0] + ray_length * np.cos(min_t)], [c[1], c[1] + ray_length * np.sin(min_t)], color=color_rays, lw=2, zorder=1)
        ax.plot([c[0], c[0] + ray_length * np.cos(max_t)], [c[1], c[1] + ray_length * np.sin(max_t)], color=color_rays, lw=2, zorder=1)

    # Individual connecting rays
    for p in points:
        ax.plot([c1[0], p[0]], [c1[1], p[1]], color=color_rays, alpha=0.15, lw=1.0, zorder=1)
        ax.plot([c2[0], p[0]], [c2[1], p[1]], color=color_rays, alpha=0.15, lw=1.0, zorder=1)

    # Camera Centers
    ax.plot(c1[0], c1[1], marker='s', color='#1A1A1A', markersize=10, markeredgecolor='white', markeredgewidth=1.5, zorder=10)
    ax.plot(c2[0], c2[1], marker='s', color='#1A1A1A', markersize=10, markeredgecolor='white', markeredgewidth=1.5, zorder=10)
    # ax.text(c1[0] - 0.4, c1[1] + 0.1, '$C_1$', fontsize=16, fontweight='bold', ha='right', color='#1A1A1A')
    # ax.text(c2[0] + 0.4, c2[1] + 0.1, '$C_2$', fontsize=16, fontweight='bold', ha='left', color='#1A1A1A')

    # Legend
    # fc_ghost = mcolors.to_rgba(color_ghost, alpha=fill_alpha); ec_ghost = mcolors.to_rgba(color_ghost, alpha=hatch_alpha)
    # fc_neigh = mcolors.to_rgba(color_neigh, alpha=fill_alpha); ec_neigh = mcolors.to_rgba(color_neigh, alpha=hatch_alpha)
    # legend_elements = [
    #     Patch(facecolor=fc_neigh, edgecolor=ec_neigh, hatch='\\\\\\\\', linewidth=1.5, label='$VH_{neigh}$ (Neighborhood Set)'),
    #     Patch(facecolor=fc_ghost, edgecolor=ec_ghost, hatch='////', linewidth=1.5, label='$VH_{ghost}$ (Spurious Geometries)'),
    #     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1A1A1A', markeredgecolor='white', markersize=10, label='Actual Agents ($p_i$)'),
    # ]
    # ax.legend(handles=legend_elements, loc='upper left', fontsize=13, framealpha=0.95, edgecolor='#DDDDDD')

    ax.set_xlim(-0.5, 12.5); ax.set_ylim(0.0, 11.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(f"figs/VH_diagram.png", transparent=True, bbox_inches='tight')
    plt.show()

def run_geometric_visual_hulls():
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import Patch

    run_params = DATASET_RUNS[0]
    name, _, start_step, end_step, step_length = _unpack(run_params)
    config, dataset, scenario_path = _load_scenario(name)
    _, _, step_range = _step_range(dataset, start_step, end_step, step_length)

    gt_data = np.load(scenario_path + '/reconstruction_scale.npz')
    gt_scales = gt_data['scales_gt']

    idx = 5
    time_step = step_range[idx]
    positions = dataset.positions_at_time_step(time_step)
    N = positions.shape[0]

    cam_positions, _ = generate_encircling_cameras(dataset, step_range, config.intrinsics_params, config.H, config.W, cam_num=4, padding=1)
    swarm_center = np.mean(positions, axis=0)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times", "Times New Roman"],
        "mathtext.fontset": "cm",
        "axes.edgecolor": "#333333", "axes.linewidth": 1.0,
    })

    grid_res = 60
    padding = 2.0
    min_pt = np.min(positions, axis=0) - padding
    max_pt = np.max(positions, axis=0) + padding

    x = np.linspace(min_pt[0], max_pt[0], grid_res)
    y = np.linspace(min_pt[1], max_pt[1], grid_res)
    z = np.linspace(min_pt[2], max_pt[2], grid_res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    grid_pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    V = grid_pts.shape[0]
    K = cam_positions.shape[0]

    tau = 0.02; cos_tau = np.cos(tau)
    in_vh_tau = np.ones(V, dtype=bool)
    in_vh_neigh_agents = np.ones((V, N), dtype=bool)

    print(f"Evaluating Voxel Grid for time step {time_step}...")
    for k in range(K):
        c = cam_positions[k]
        vec_g = grid_pts - c
        dir_g = vec_g / np.linalg.norm(vec_g, axis=1, keepdims=True)
        vec_p = positions - c
        dir_p = vec_p / np.linalg.norm(vec_p, axis=1, keepdims=True)
        cos_theta = np.dot(dir_g, dir_p.T)
        cone_mask = cos_theta >= cos_tau
        in_vh_tau &= np.any(cone_mask, axis=1)
        in_vh_neigh_agents &= cone_mask

    vh_neigh = np.any(in_vh_neigh_agents, axis=1)
    vh_ghost = in_vh_tau & (~vh_neigh)
    VH_neigh_img = vh_neigh.reshape(X.shape)
    VH_ghost_img = vh_ghost.reshape(X.shape)
    VH_tau_img = in_vh_tau.reshape(X.shape)

    vol_tau = np.sum(in_vh_tau); vol_neigh = np.sum(vh_neigh); vol_ghost = np.sum(vh_ghost)
    ratio = vol_neigh / vol_tau if vol_tau > 0 else 0
    print(f"VH_tau Voxels: {vol_tau} | VH_neigh Voxels: {vol_neigh} | Ghost Voxels: {vol_ghost}")
    print(f"Ratio VH_neigh / VH_tau: {ratio:.4f}")

    color_ghost = '#E78875'; color_neigh = '#7CB1A1'; fill_alpha = 0.35

    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(r"3D Decomposition of $\mathcal{VH}_{\tau}$", pad=20)

    colors = np.empty(X.shape, dtype=object)
    colors[VH_neigh_img] = color_neigh; colors[VH_ghost_img] = color_ghost

    dx = (max_pt[0] - min_pt[0]) / (grid_res - 1)
    dy = (max_pt[1] - min_pt[1]) / (grid_res - 1)
    dz = (max_pt[2] - min_pt[2]) / (grid_res - 1)
    x_edge = np.linspace(min_pt[0] - dx/2, max_pt[0] + dx/2, grid_res + 1)
    y_edge = np.linspace(min_pt[1] - dy/2, max_pt[1] + dy/2, grid_res + 1)
    z_edge = np.linspace(min_pt[2] - dz/2, max_pt[2] + dz/2, grid_res + 1)
    X_edge, Y_edge, Z_edge = np.meshgrid(x_edge, y_edge, z_edge, indexing='ij')

    ax.voxels(X_edge, Y_edge, Z_edge, VH_tau_img, facecolors=colors, edgecolor='k', linewidth=0.1, alpha=fill_alpha)
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='#1A1A1A', edgecolor='white', linewidth=0.8, s=50, zorder=9, label='Actual Agents ($p_i$)')
    ax.scatter(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2], marker='s', color='#1A1A1A', edgecolor='white', s=80, label='Cameras ($C_k$)')

    for c in cam_positions:
        ax.plot([c[0], swarm_center[0]], [c[1], swarm_center[1]], [c[2], swarm_center[2]], color='#6B7A93', alpha=0.3, linestyle='--', linewidth=1)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(min_pt[0], max_pt[0]); ax.set_ylim(min_pt[1], max_pt[1]); ax.set_zlim(min_pt[2], max_pt[2])
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

    legend_elements = [
        Patch(facecolor=color_neigh, alpha=fill_alpha, edgecolor='k', label=r'$\mathcal{VH}_{neigh}$ (Neighborhood Set)'),
        Patch(facecolor=color_ghost, alpha=fill_alpha, edgecolor='k', label=r'$\mathcal{VH}_{ghost}$ (Spurious Geometries)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1A1A1A', markeredgecolor='white', markersize=8, label='Agents ($p_i$)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#1A1A1A', markeredgecolor='white', markersize=8, label='Cameras ($C_k$)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.95, edgecolor='#DDDDDD')
    ax.text2D(0.05, 0.95, rf'Ratio ($\mathcal{{VH}}_{{neigh}} / \mathcal{{VH}}_{{\tau}}$): {ratio:.4f}',
              transform=ax.transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()

def plot_ratio_surface(run_params, scales, cam_nums, base_tau=0.05, idx=5, grid_res=50):
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    """
    Evaluates the ratio (VH_neigh / VH_tau) across a grid of scale factors and camera counts.
    Plots a 3D surface where X=scale, Y=cam_num, Z=ratio.
    """
    name, _, start_step, end_step, step_length = _unpack(run_params)
    config, dataset, scenario_path = _load_scenario(name)
    _, _, step_range = _step_range(dataset, start_step, end_step, step_length)

    time_step = step_range[idx]
    positions = dataset.positions_at_time_step(time_step)
    N = positions.shape[0]

    padding = 2.0
    min_pt = np.min(positions, axis=0) - padding
    max_pt = np.max(positions, axis=0) + padding

    x = np.linspace(min_pt[0], max_pt[0], grid_res)
    y = np.linspace(min_pt[1], max_pt[1], grid_res)
    z = np.linspace(min_pt[2], max_pt[2], grid_res)
    X_grid, Y_grid, Z_grid = np.meshgrid(x, y, z, indexing='ij')
    grid_pts = np.vstack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()]).T
    V = grid_pts.shape[0]

    Ratios = np.zeros((len(cam_nums), len(scales)))
    print(f"Evaluating surface grid ({len(cam_nums)}x{len(scales)} iterations)...")

    for i, cam_num in enumerate(cam_nums):
        cam_positions, _ = generate_encircling_cameras(dataset, step_range, config.intrinsics_params, config.H, config.W, cam_num=cam_num, padding=1, is_3d=True)
        K = cam_positions.shape[0]

        grid_rays = []
        agent_rays = []
        for c in cam_positions:
            vec_g = grid_pts - c
            grid_rays.append(vec_g / np.linalg.norm(vec_g, axis=1, keepdims=True))
            vec_p = positions - c
            agent_rays.append(vec_p / np.linalg.norm(vec_p, axis=1, keepdims=True))

        for j, scale in enumerate(scales):
            tau_current = base_tau * scale
            cos_tau = np.cos(tau_current)
            in_vh_tau = np.ones(V, dtype=bool)
            in_vh_neigh_agents = np.ones((V, N), dtype=bool)

            for k in range(K):
                cos_theta = np.dot(grid_rays[k], agent_rays[k].T)
                cone_mask = cos_theta >= cos_tau
                in_vh_tau &= np.any(cone_mask, axis=1)
                in_vh_neigh_agents &= cone_mask

            vh_neigh = np.any(in_vh_neigh_agents, axis=1)
            vol_tau = np.sum(in_vh_tau); vol_neigh = np.sum(vh_neigh)
            Ratios[i, j] = vol_neigh / vol_tau if vol_tau > 0 else 0
            print(f"Cam: {cam_num} | Scale: {scale:.2f} | Ratio: {Ratios[i, j]:.4f}")

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Computer Modern Roman", "Times"],
        "axes.edgecolor": "#333333",
    })

    fig = plt.figure(figsize=(12, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    X_surf, Y_surf = np.meshgrid(scales, cam_nums)
    surf = ax.plot_surface(X_surf, Y_surf, Ratios, cmap=cm.viridis, edgecolor='k', linewidth=0.5, alpha=0.8, antialiased=True)
    ax.set_title(r"Ratio $\mathcal{VH}_{neigh} / \mathcal{VH}_{\tau}$ vs Scale & Camera Count", pad=20, fontsize=16)
    ax.set_xlabel("Scale (Tolerance Multiplier)", labelpad=10, fontsize=12)
    ax.set_ylabel("Number of Cameras ($C_k$)", labelpad=10, fontsize=12)
    ax.set_zlabel("Ratio", labelpad=10, fontsize=12)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1, label='Ratio')
    ax.view_init(elev=30, azim=225)
    plt.tight_layout()
    plt.show()
    return X_surf, Y_surf, Ratios

def dra_metrics():
    def generate_gaussian(x, mu, sigma, amplitude=1.0):
        """Generates a 1D Gaussian curve."""
        return amplitude * np.exp(-0.5 * ((x - mu) / sigma)**2)

    x = np.linspace(-5, 7, 1000)
    D_GT = generate_gaussian(x, mu=0.0, sigma=1.2, amplitude=1.0)
    D_P = generate_gaussian(x, mu=1.8, sigma=1.0, amplitude=0.9)
    TP_boundary = np.minimum(D_GT, D_P)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

    ax.fill_between(x, 0, TP_boundary, facecolor='#A9DFBF', edgecolor='#1E8449', hatch='///', alpha=0.8, zorder=1)
    ax.fill_between(x, TP_boundary, D_GT, facecolor='#AED6F1', edgecolor='#2874A6', hatch='\\\\\\', alpha=0.8, zorder=1)
    ax.fill_between(x, TP_boundary, D_P, facecolor='#F5B7B1', edgecolor='#B03A2E', hatch='xxx', alpha=0.8, zorder=1)

    ax.plot(x, D_GT, color='#2874A6', linestyle='-', linewidth=2.5, zorder=3)
    ax.plot(x, D_P, color='#B03A2E', linestyle='-', linewidth=2.5, zorder=3)
    ax.axis('off')
    ax.plot([x.min(), x.max()], [0, 0], color='black', linewidth=1.5, zorder=4)

    plt.tight_layout()
    # plt.savefig("tp_fp_fn_gaussians.pdf", format='pdf', bbox_inches='tight')
    plt.show()

def one_frame_parameter_search(n_trials=50):
    import optuna

    run_params = DATASET_RUNS[0]
    name, _, start_step, end_step, step_length = _unpack(run_params)
    config, dataset, scenario_path = _load_scenario(name)
    _, _, step_range = _step_range(dataset, start_step, end_step, step_length)

    cam_system = _build_cam_system(dataset, step_range, config, CAM_NUM)

    reconstruction_params = {
        'targetd_num_mode': 10, 'voxel_scale': 0.5, 'voxel_peak_threshold': 0.3,
        'voxel_grid_max_size': 32, 'voxel_peaks_number': 2 * 10,
    }

    gt_data = np.load(scenario_path + '/reconstruction_scale.npz')
    gt_scales = gt_data['scales_gt']

    idx = 5
    time_step = step_range[idx]
    positions = dataset.positions_at_time_step(time_step)
    poses, projections, _, masks = cam_system.simulate_vision(positions, renderer='projection_only')

    output_dir = os.path.join(os.getcwd(), f"t_{time_step:03d}")

    def objective(trial):
        train_params = {
            'xyz_lr_c': trial.suggest_float('xyz_lr_c', 1e-3, 0.2, log=True),
            'xyz_lr_final_c': trial.suggest_float('xyz_lr_final_c', 1e-2, 0.5, log=True),
            'radius_lr_c': trial.suggest_float('radius_lr_c', 1e-3, 0.2, log=True),
            'radius_lr_final_c': trial.suggest_float('radius_lr_final_c', 1e-2, 1.0, log=True),
            'weights_lr_c': trial.suggest_float('weights_lr_c', 1e-3, 0.5, log=True),
            'weights_lr_final_c': trial.suggest_float('weights_lr_final_c', 1e-2, 1.0, log=True),
            'xyz_reg': trial.suggest_float('xyz_reg', 0.0, 1.0),
            'radius_reg': trial.suggest_float('radius_reg', 0.0, 1.0),
            'radius_cutoff_inv': trial.suggest_float('radius_cutoff_inv', 0.1, 1.0),
            'lr_max_steps': 100,
        }
        density_reconstructor = DensityReconstructor(max_iter=train_params['lr_max_steps'], use_decoupled=False)
        try:
            model, scale_spaces = density_reconstructor.process_frame(
                cam_system, point_sets=projections, positions=positions,
                initGMM=None, is_adaptive_scale=False, scale=gt_scales[idx],
                is_store_intermediate=False, is_log=False,
                output_dir=output_dir, debug=False,
                train_params=train_params, reconstruction_params=reconstruction_params,
            )
            return model[0].mean_loss
        except Exception as e:
            print(f"Trial pruned due to exception: {e}")
            raise optuna.TrialPruned()

    study = optuna.create_study(direction="minimize")
    print(f"Starting parameter search for {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)

    print("\n" + "="*30)
    print("BEST PARAMETERS FOUND")
    print("="*30)
    print(f"Minimum Loss: {study.best_value}")
    best_train_params = study.best_params
    for key, value in best_train_params.items():
        print(f"'{key}': {value},")
    return best_train_params

def one_frame_convergence():
    run_params = DATASET_RUNS[0]
    name, _, start_step, end_step, step_length = _unpack(run_params)
    config, dataset, scenario_path = _load_scenario(name)
    _, _, step_range = _step_range(dataset, start_step, end_step, step_length)

    CAM_NUM_local = 3  # Note: shadows module-level CAM_NUM
    cam_system = _build_cam_system(dataset, step_range, config, CAM_NUM_local)

    reconstruction_params = {
        'targetd_num_mode': 10, 'voxel_scale': 0.5, 'voxel_peak_threshold': 0.3,
        'voxel_grid_max_size': 32, 'voxel_peaks_number': 2 * 10,
    }
    # train_params_old = {
    #     'xyz_lr_c': 0.05, 'xyz_lr_final_c': 0.2,
    #     'radius_lr_c': 0.05, 'radius_lr_final_c': 0.5,
    #     'weights_lr_c': 0.10, 'weights_lr_final_c': 0.5,
    #     'xyz_reg': 0, 'radius_reg': 0, 'radius_cutoff_inv': 0.5, 'lr_max_steps': 1000,
    # }
    train_params = {
        'xyz_lr_c': 0.11550156892954913, 'xyz_lr_final_c': 0.015263086280830469,
        'radius_lr_c': 0.09585436467026787, 'radius_lr_final_c': 0.02420618007560584,
        'weights_lr_c': 0.19814963583342243, 'weights_lr_final_c': 0.7979132269720964,
        'xyz_reg': 0.21978381872642633, 'radius_reg': 0.6083537781516261,
        'radius_cutoff_inv': 0.6013595613763145, 'lr_max_steps': 100,
    }
    density_reconstructor = DensityReconstructor(max_iter=train_params['lr_max_steps'], use_decoupled=False)

    gt_data = np.load(scenario_path + '/reconstruction_scale.npz')
    gt_scales = gt_data['scales_gt']

    idx = 5
    time_step = step_range[idx]
    total_num = []
    positions = dataset.positions_at_time_step(time_step)
    N = positions.shape[0]
    total_num.append(positions.shape[0])
    # poses, _, images, masks = cam_system.simulate_vision(positions, renderer='gaussian')
    poses, projections, _, masks = cam_system.simulate_vision(positions, renderer='projection_only')

    output_dir = os.path.join(os.getcwd(), f"t_{time_step:03d}")

    model, scale_spaces = \
        density_reconstructor.process_frame(cam_system, point_sets=projections, positions=positions,
                                            initGMM=None, is_adaptive_scale=False, scale=gt_scales[idx],
                                            is_store_intermediate=True, is_log=True,
                                            output_dir=os.path.join(os.getcwd(), f"t_{time_step:03d}"),
                                            debug=False, train_params=train_params,
                                            reconstruction_params=reconstruction_params)

    # training_history = GaussianModel.load_training_history(os.path.join(output_dir, f"checkpoint_level_0.pth"))
    # model = GaussianModel.load_iter(training_history, 99)

    r_means, r_weights, r_covs = GMR.runnalls_algorithm_simple_torch(
        means=torch.from_numpy(positions),
        radii=torch.full((N, 1), gt_scales[idx], device='cuda', dtype=torch.float),
        weights=torch.full((N, 1), 1.0, device='cuda', dtype=torch.float),
        L=20, DEVICE='cuda',
    )
    r_radius = torch.sqrt(r_covs[:, 0, 0]).reshape((-1, 1))

    from gaussian_rasterizer_simple_large import rasterize_gaussians
    gmr_images = []
    for camera in cam_system.cameras:
        gmr_images.append(rasterize_gaussians(r_means, r_radius, r_weights, camera.state.R, camera.state.T, camera.state.K, camera.state.H, camera.state.W, False))
    gmr_losses = [torch.sum(scale_spaces[i][0] - gmr_images[i]).item() for i in range(CAM_NUM_local)]

    # gmm_visualizer = MultiGMMPlotter()
    # gmm_visualizer.add_gmm(model[0]._xyz.detach().cpu().numpy(), model[0]._radius.detach().cpu().numpy(), model[0]._weights.detach().cpu().numpy())
    # gmm_visualizer.update(real_means=positions)
    # move_figure(gmm_visualizer.fig, 100, 100)
    # gmm_visualizer.ax.view_init(elev=33, azim=-117, roll=0)
    # gmm_visualizer.fig.savefig("gmm_diagram.png", transparent=True, bbox_inches='tight')

    # fig = plt.figure(); ax = fig.add_subplot(111)
    # ax.plot(np.arange(train_params['lr_max_steps'])[::2], model[0].metrics_history['loss_history'][::2])
    # ax.plot(np.arange(train_params['lr_max_steps'])[1::2], model[0].metrics_history['loss_history'][1::2])
    # ax.set_yscale('log')

    print(f"training loss: {model[0].mean_loss}")

    min_coords = np.min(positions, axis=0); max_coords = np.max(positions, axis=0)
    bounds = np.vstack((min_coords - 3 * gt_scales[idx], max_coords + 3 * gt_scales[idx])).T
    voxel_res = np.max(max_coords - min_coords) * 5e-3

    total_tp_mass, total_fp_mass, total_fn_mass = \
        compute_metrics_batched_torch(means1_np=positions, sigma1=gt_scales[idx],
                                      pred_means=model[0]._xyz, pred_weights=model[0]._weights, pred_sigmas=model[0]._radius,
                                      bounds=bounds, voxel_res=voxel_res, batch_size=50000, device='cuda')
    recall = total_tp_mass / N
    hallucination = total_fp_mass / model[0]._weights.sum().item()
    dMOTA = 1 - (total_fn_mass + total_fp_mass) / N
    print(f"recall: {recall}, hallu: {hallucination}, dMOTA: {dMOTA}")

    total_tp_mass, total_fp_mass, total_fn_mass = \
        compute_metrics_batched_torch(means1_np=positions, sigma1=gt_scales[idx],
                                      pred_means=r_means, pred_weights=r_weights, pred_sigmas=r_radius,
                                      bounds=bounds, voxel_res=voxel_res, batch_size=50000, device='cuda')
    recall = total_tp_mass / N
    hallucination = total_fp_mass / model[0]._weights.sum().item()
    dMOTA = 1 - (total_fn_mass + total_fp_mass) / N
    print(f"gmr recall: {recall}, hallu: {hallucination}, dMOTA: {dMOTA}")

    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['mathtext.fontset'] = 'cm'; plt.rcParams['axes.labelsize'] = 12
    # plt.rcParams['xtick.labelsize'] = 10; plt.rcParams['ytick.labelsize'] = 10
    # plt.rcParams['legend.fontsize'] = 10; plt.rcParams['figure.dpi'] = 300
    # fig, ax = plt.subplots(figsize=(6, 4))
    # steps = np.arange(train_params['lr_max_steps'])
    # loss_history = model[0].metrics_history['loss_history']
    # ax.plot(steps[::2], loss_history[::2], color='#1f77b4', linewidth=1.5, label='Loss Cam 1')
    # ax.plot(steps[1::2], loss_history[1::2], color='#d62728', linewidth=1.5, linestyle='--', alpha=0.85, label='Loss Cam 2')
    # ax.set_yscale('log'); ax.set_xlabel('Training Step'); ax.set_ylabel('Training Loss')
    # ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    # ax.grid(True, which="major", axis="y", linestyle="-", alpha=0.2)
    # ax.grid(True, which="minor", axis="y", linestyle=":", alpha=0.1)
    # ax.legend(frameon=False, loc='best')
    # fig.tight_layout(); plt.savefig('loss_curve.pdf', format='pdf', bbox_inches='tight')

    # fig2 = plt.figure(); ax2 = fig2.add_subplot(111)
    # ax2.plot(np.arange(train_params['lr_max_steps']), model[0].metrics_history['grad_norm_xyz_history'], label='xyz')
    # ax2.plot(np.arange(train_params['lr_max_steps']), model[0].metrics_history['grad_norm_radius_history'], label='radius')
    # ax2.plot(np.arange(train_params['lr_max_steps']), model[0].metrics_history['grad_norm_weights_history'], label='weights')
    # ax2.set_yscale('log'); ax2.legend(); plt.show()

def one_frame_dMOTA_factor_analysis(force_recalculate=False):
    """
    Args:
        force_recalculate (bool): Set to True to ignore the cache and force a new calculation.
    """
    cache_filename = 'dmota_cache.pkl'
    cam_nums = [2, 3, 5, 7, 9]
    comp_nums = list(range(10, 41))

    if not force_recalculate and os.path.exists(cache_filename):
        print(f"✅ Found cached data in '{cache_filename}'. Loading results...")
        with open(cache_filename, 'rb') as f:
            cached_data = pickle.load(f)
            gmr_dmota_results = cached_data['gmr']
            model_dmota_results = cached_data['model']
    else:
        print("⏳ No cache found (or recalculation forced). Starting heavy computations...")

        run_params = DATASET_RUNS[0]
        name, _, start_step, end_step, step_length = _unpack(run_params)
        config, dataset, scenario_path = _load_scenario(name)
        _, _, step_range = _step_range(dataset, start_step, end_step, step_length)

        gt_data = np.load(scenario_path + '/reconstruction_scale.npz')
        gt_scales = gt_data['scales_gt']

        idx = 5
        time_step = step_range[idx]
        positions = dataset.positions_at_time_step(time_step)
        N = positions.shape[0]

        min_coords = np.min(positions, axis=0); max_coords = np.max(positions, axis=0)
        bounds = np.vstack((min_coords - 3 * gt_scales[idx], max_coords + 3 * gt_scales[idx])).T
        voxel_res = np.max(max_coords - min_coords) * 5e-3

        train_params = {
            'xyz_lr_c': 0.11550156892954913, 'xyz_lr_final_c': 0.015263086280830469,
            'radius_lr_c': 0.09585436467026787, 'radius_lr_final_c': 0.02420618007560584,
            'weights_lr_c': 0.19814963583342243, 'weights_lr_final_c': 0.7979132269720964,
            'xyz_reg': 0.21978381872642633, 'radius_reg': 0.6083537781516261,
            'radius_cutoff_inv': 0.6013595613763145, 'lr_max_steps': 1000,
        }

        reconstruction_params_base = {
            'targetd_num_mode': 10, 'voxel_scale': 0.5, 'voxel_peak_threshold': 0.3,
            'voxel_grid_max_size': 32,
        }

        log_file_path = os.getcwd()

        # Evaluate baseline GMR
        print("Evaluating baseline GMR model...")
        gmr_dmota_results = []
        for comp_num in comp_nums:
            r_means, r_weights, r_covs = GMR.runnalls_algorithm_simple_torch(
                means=torch.from_numpy(positions),
                radii=torch.full((N, 1), gt_scales[idx], device='cuda', dtype=torch.float),
                weights=torch.full((N, 1), 1.0, device='cuda', dtype=torch.float),
                L=comp_num, DEVICE='cuda',
            )
            r_radius = torch.sqrt(r_covs[:, 0, 0]).reshape((-1, 1))
            _, total_fp_mass, total_fn_mass = compute_metrics_batched_torch(
                means1_np=positions, sigma1=gt_scales[idx],
                pred_means=r_means, pred_weights=r_weights, pred_sigmas=r_radius,
                bounds=bounds, voxel_res=voxel_res, batch_size=50000, device='cuda',
            )
            gmr_dmota_results.append(1 - (total_fn_mass + total_fp_mass) / N)

        # Evaluate Main Model over CAM_NUMs
        model_dmota_results = {cam_num: [] for cam_num in cam_nums}
        for cam_num in cam_nums:
            print(f"Testing CAM_NUM = {cam_num}...")
            cam_system = _build_cam_system(dataset, step_range, config, cam_num)
            poses, projections, _, masks = cam_system.simulate_vision(positions, renderer='projection_only')
            density_reconstructor = DensityReconstructor(max_iter=train_params['lr_max_steps'], use_decoupled=False)

            for comp_num in comp_nums:
                reconstruction_params = reconstruction_params_base.copy()
                reconstruction_params['voxel_peaks_number'] = comp_num

                model, scale_spaces = density_reconstructor.process_frame(
                    cam_system, point_sets=projections, positions=positions,
                    initGMM=None, is_adaptive_scale=False, scale=gt_scales[idx],
                    is_store_intermediate=True, is_log=True,
                    output_dir=os.path.join(log_file_path, f"t_{time_step:03d}"),
                    debug=False, train_params=train_params,
                    reconstruction_params=reconstruction_params,
                )
                _, total_fp_mass, total_fn_mass = compute_metrics_batched_torch(
                    means1_np=positions, sigma1=gt_scales[idx],
                    pred_means=model[0]._xyz, pred_weights=model[0]._weights, pred_sigmas=model[0]._radius,
                    bounds=bounds, voxel_res=voxel_res, batch_size=50000, device='cuda',
                )
                model_dmota_results[cam_num].append(1 - (total_fn_mass + total_fp_mass) / N)

        print(f"💾 Saving computed results to '{cache_filename}'...")
        with open(cache_filename, 'wb') as f:
            pickle.dump({'gmr': gmr_dmota_results, 'model': model_dmota_results}, f)

    # Plotting
    print("🎨 Generating plot...")
    plt.figure(figsize=(6, 4))
    plt.rcParams['font.family'] = 'serif'; plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['axes.labelsize'] = 12; plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10; plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 300

    plt.plot(comp_nums, gmr_dmota_results, label='GMR-2', color='black', linewidth=2.5, linestyle='--', zorder=10)
    colormap = plt.cm.get_cmap('viridis', len(cam_nums))
    for i, cam_num in enumerate(cam_nums):
        plt.plot(comp_nums, model_dmota_results[cam_num], label=f'Ours-{cam_num}',
                 color=colormap(i), marker='o', markersize=4, linewidth=1.5)

    plt.xlabel('Component Number'); plt.ylabel('DRA')
    plt.gca().spines['top'].set_visible(False); plt.gca().spines['right'].set_visible(False)
    plt.grid(True, which="major", axis="y", linestyle="-", alpha=0.3)
    plt.legend(loc='lower right', ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig('dmota_comparison.png', bbox_inches='tight')
    plt.show()

def one_frame_dMOTA_factor_analysis_2(force_recalculate=False):
    """
    Args:
        force_recalculate (bool): Set to True to ignore the cache and force a new calculation.
    """
    cache_filename = 'dmota_cache_2.pkl'
    cam_nums = [2, 3, 5, 7, 9]
    target_modes = list(range(5, 26))
    fixed_comp_nums = 20

    if not force_recalculate and os.path.exists(cache_filename):
        print(f"✅ Found cached data in '{cache_filename}'. Loading results...")
        with open(cache_filename, 'rb') as f:
            cached_data = pickle.load(f)
            gmr_dmota_results = cached_data['gmr']
            model_dmota_results = cached_data['model']
            computed_scales = cached_data['scales']
    else:
        print("⏳ No cache found (or recalculation forced). Starting heavy computations...")

        run_params = DATASET_RUNS[0]
        name, _, start_step, end_step, step_length = _unpack(run_params)
        config, dataset, scenario_path = _load_scenario(name)
        _, _, step_range = _step_range(dataset, start_step, end_step, step_length)

        idx = 5
        time_step = step_range[idx]
        positions = dataset.positions_at_time_step(time_step)
        N = positions.shape[0]

        min_coords = np.min(positions, axis=0); max_coords = np.max(positions, axis=0)
        voxel_res = np.max(max_coords - min_coords) * 5e-3

        train_params = {
            'xyz_lr_c': 0.11550156892954913, 'xyz_lr_final_c': 0.015263086280830469,
            'radius_lr_c': 0.09585436467026787, 'radius_lr_final_c': 0.02420618007560584,
            'weights_lr_c': 0.19814963583342243, 'weights_lr_final_c': 0.7979132269720964,
            'xyz_reg': 0.21978381872642633, 'radius_reg': 0.6083537781516261,
            'radius_cutoff_inv': 0.6013595613763145, 'lr_max_steps': 1000,
        }

        reconstruction_params_base = {
            'voxel_scale': 0.5, 'voxel_peak_threshold': 0.3, 'voxel_grid_max_size': 32,
        }

        log_file_path = os.getcwd()

        # Calculate Exact Scales for Target Modes
        print("Calculating exact scales to hit target modes...")
        pos_gpu = torch.from_numpy(positions).cuda().float()
        nn_dist = torch.cdist(pos_gpu, pos_gpu) + torch.eye(pos_gpu.shape[0], device='cuda') * 1e10
        avg_nn_dist = torch.median(torch.min(nn_dist, dim=1).values).item()
        f = lambda s: mode_counting(pos_gpu, pos_gpu.clone(), s, max_iter=2000, tol=avg_nn_dist*5e-4)
        computed_scales = [find_target_scale(f, tm, 0, 5) for tm in target_modes]

        # Evaluate baseline GMR over Target Modes
        print("Evaluating baseline GMR model across target modes...")
        gmr_dmota_results = []
        for tm, current_scale in zip(target_modes, computed_scales):
            bounds = np.vstack((min_coords - 3 * current_scale, max_coords + 3 * current_scale)).T
            r_means, r_weights, r_covs = GMR.runnalls_algorithm_simple_torch(
                means=torch.from_numpy(positions),
                radii=torch.full((N, 1), current_scale, device='cuda', dtype=torch.float),
                weights=torch.full((N, 1), 1.0, device='cuda', dtype=torch.float),
                L=fixed_comp_nums, DEVICE='cuda',
            )
            r_radius = torch.sqrt(r_covs[:, 0, 0]).reshape((-1, 1))
            _, total_fp_mass, total_fn_mass = compute_metrics_batched_torch(
                means1_np=positions, sigma1=current_scale,
                pred_means=r_means, pred_weights=r_weights, pred_sigmas=r_radius,
                bounds=bounds, voxel_res=voxel_res, batch_size=50000, device='cuda',
            )
            gmr_dmota_results.append(1 - (total_fn_mass + total_fp_mass) / N)

        # Evaluate Main Model over CAM_NUMs and Target Modes
        model_dmota_results = {cam_num: [] for cam_num in cam_nums}
        for cam_num in cam_nums:
            print(f"Testing CAM_NUM = {cam_num}...")
            cam_system = _build_cam_system(dataset, step_range, config, cam_num)
            poses, projections, _, masks = cam_system.simulate_vision(positions, renderer='projection_only')
            density_reconstructor = DensityReconstructor(max_iter=train_params['lr_max_steps'], use_decoupled=False)

            for tm, current_scale in zip(target_modes, computed_scales):
                bounds = np.vstack((min_coords - 3 * current_scale, max_coords + 3 * current_scale)).T
                reconstruction_params = reconstruction_params_base.copy()
                reconstruction_params['targetd_num_mode'] = tm
                reconstruction_params['voxel_peaks_number'] = fixed_comp_nums

                model, scale_spaces = density_reconstructor.process_frame(
                    cam_system, point_sets=projections, positions=positions,
                    initGMM=None, is_adaptive_scale=False, scale=current_scale,
                    is_store_intermediate=True, is_log=True,
                    output_dir=os.path.join(log_file_path, f"t_{time_step:03d}"),
                    debug=False, train_params=train_params,
                    reconstruction_params=reconstruction_params,
                )
                _, total_fp_mass, total_fn_mass = compute_metrics_batched_torch(
                    means1_np=positions, sigma1=current_scale,
                    pred_means=model[0]._xyz, pred_weights=model[0]._weights, pred_sigmas=model[0]._radius,
                    bounds=bounds, voxel_res=voxel_res, batch_size=50000, device='cuda',
                )
                model_dmota_results[cam_num].append(1 - (total_fn_mass + total_fp_mass) / N)

        print(f"💾 Saving computed results to '{cache_filename}'...")
        with open(cache_filename, 'wb') as f:
            pickle.dump({'gmr': gmr_dmota_results, 'model': model_dmota_results, 'scales': computed_scales}, f)

    # Plotting
    print("🎨 Generating plot...")
    plt.figure(figsize=(6, 4))
    plt.rcParams['font.family'] = 'serif'; plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['axes.labelsize'] = 12; plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10; plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 300

    plt.plot(target_modes, gmr_dmota_results, label='GMR-2', color='black', linewidth=2.5, linestyle='--', zorder=10)
    colormap = plt.cm.get_cmap('viridis', len(cam_nums))
    for i, cam_num in enumerate(cam_nums):
        plt.plot(target_modes, model_dmota_results[cam_num], label=f'Ours-{cam_num}',
                 color=colormap(i), marker='o', markersize=4, linewidth=1.5)

    plt.xlabel('Number of Modes'); plt.ylabel('DRA')
    plt.xticks(target_modes[::2])
    plt.gca().spines['top'].set_visible(False); plt.gca().spines['right'].set_visible(False)
    plt.grid(True, which="major", axis="y", linestyle="-", alpha=0.3)
    plt.legend(loc='lower right', ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig('dra_target_modes_comparison.png', bbox_inches='tight')
    plt.show()

def one_frame_dMOTA_noise(force_recalculate=False):
    """
    Args:
        force_recalculate (bool): Set to True to ignore the cache and force a new calculation.
    """
    cache_filename = 'dra_noise_variance_cache.pkl'
    cam_nums = [2, 3, 5, 7, 9]
    noise_levels = list(range(5, 21, 3))  # [5, 8, 11, 14, 17, 20]
    num_trials = 10
    fixed_comp_num = 20

    if not force_recalculate and os.path.exists(cache_filename):
        print(f"✅ Found cached data in '{cache_filename}'. Loading results...")
        with open(cache_filename, 'rb') as f:
            cached_data = pickle.load(f)
            gmr_dmota = cached_data['gmr']
            model_dmota_raw = cached_data['model_raw']
    else:
        print("⏳ No cache found (or recalculation forced). Starting heavy computations...")

        run_params = DATASET_RUNS[0]
        name, _, start_step, end_step, step_length = _unpack(run_params)
        config, dataset, scenario_path = _load_scenario(name)
        _, _, step_range = _step_range(dataset, start_step, end_step, step_length)

        gt_data = np.load(scenario_path + '/reconstruction_scale.npz')
        gt_scales = gt_data['scales_gt']

        idx = 5
        time_step = step_range[idx]
        positions = dataset.positions_at_time_step(time_step)
        N = positions.shape[0]

        min_coords = np.min(positions, axis=0); max_coords = np.max(positions, axis=0)
        current_scale = gt_scales[idx]
        bounds = np.vstack((min_coords - 3 * current_scale, max_coords + 3 * current_scale)).T
        voxel_res = np.max(max_coords - min_coords) * 5e-3

        train_params = {
            'xyz_lr_c': 0.11550156892954913, 'xyz_lr_final_c': 0.015263086280830469,
            'radius_lr_c': 0.09585436467026787, 'radius_lr_final_c': 0.02420618007560584,
            'weights_lr_c': 0.19814963583342243, 'weights_lr_final_c': 0.7979132269720964,
            'xyz_reg': 0.21978381872642633, 'radius_reg': 0.6083537781516261,
            'radius_cutoff_inv': 0.6013595613763145, 'lr_max_steps': 100,
        }

        reconstruction_params_base = {
            'targetd_num_mode': 10, 'voxel_scale': 0.5, 'voxel_peak_threshold': 0.3,
            'voxel_grid_max_size': 32, 'voxel_peaks_number': fixed_comp_num,
        }

        log_file_path = os.getcwd()

        # Evaluate baseline GMR (Calculated Once)
        print("Evaluating baseline GMR model (Invariant to 2D noise)...")
        r_means, r_weights, r_covs = GMR.runnalls_algorithm_simple_torch(
            means=torch.from_numpy(positions),
            radii=torch.full((N, 1), current_scale, device='cuda', dtype=torch.float),
            weights=torch.full((N, 1), 1.0, device='cuda', dtype=torch.float),
            L=fixed_comp_num, DEVICE='cuda',
        )
        r_radius = torch.sqrt(r_covs[:, 0, 0]).reshape((-1, 1))
        _, total_fp_mass, total_fn_mass = compute_metrics_batched_torch(
            means1_np=positions, sigma1=current_scale,
            pred_means=r_means, pred_weights=r_weights, pred_sigmas=r_radius,
            bounds=bounds, voxel_res=voxel_res, batch_size=50000, device='cuda',
        )
        gmr_dmota = 1 - (total_fn_mass + total_fp_mass) / N

        # Evaluate Main Model over Trials
        model_dmota_raw = {cam_num: {n: [] for n in noise_levels} for cam_num in cam_nums}
        for cam_num in cam_nums:
            print(f"\nTesting CAM_NUM = {cam_num}...")
            cam_system = _build_cam_system(dataset, step_range, config, cam_num)
            poses, base_projections, _, masks = cam_system.simulate_vision(positions, renderer='projection_only')
            density_reconstructor = DensityReconstructor(max_iter=train_params['lr_max_steps'], use_decoupled=False)

            for noise_std in noise_levels:
                print(f"  -> Noise Std: {noise_std} ({num_trials} trials)")
                for trial in range(num_trials):
                    np.random.seed(42 + trial + int(noise_std * 100) + cam_num)

                    noisy_projections = []
                    for cam_idx in range(len(base_projections)):
                        max_w = cam_system.cameras[cam_idx].state.W
                        max_h = cam_system.cameras[cam_idx].state.H
                        new_projections = base_projections[cam_idx].copy()
                        needs_noise = np.ones(new_projections.shape[0], dtype=bool)

                        while np.any(needs_noise):
                            num_needs = np.sum(needs_noise)
                            noise = np.random.normal(0, noise_std, size=(num_needs, 2))
                            candidate_proj = base_projections[cam_idx][needs_noise] + noise
                            in_bounds = (candidate_proj[:, 0] >= 0) & (candidate_proj[:, 0] <= max_w) & \
                                        (candidate_proj[:, 1] >= 0) & (candidate_proj[:, 1] <= max_h)
                            valid_indices = np.where(needs_noise)[0][in_bounds]
                            new_projections[valid_indices] = candidate_proj[in_bounds]
                            needs_noise[valid_indices] = False
                        noisy_projections.append(new_projections)

                    trial_out_dir = os.path.join(log_file_path, f"t_{time_step:03d}_cam{cam_num}_n{noise_std}_tr{trial}")
                    model, scale_spaces = density_reconstructor.process_frame(
                        cam_system, point_sets=noisy_projections, positions=positions,
                        initGMM=None, is_adaptive_scale=False, scale=current_scale,
                        is_store_intermediate=False, is_log=False,
                        output_dir=trial_out_dir, debug=False,
                        train_params=train_params, reconstruction_params=reconstruction_params_base,
                    )
                    _, total_fp_mass, total_fn_mass = compute_metrics_batched_torch(
                        means1_np=positions, sigma1=current_scale,
                        pred_means=model[0]._xyz, pred_weights=model[0]._weights, pred_sigmas=model[0]._radius,
                        bounds=bounds, voxel_res=voxel_res, batch_size=50000, device='cuda',
                    )
                    model_dmota_raw[cam_num][noise_std].append(1 - (total_fn_mass + total_fp_mass) / N)

        print(f"\n💾 Saving computed results to '{cache_filename}'...")
        with open(cache_filename, 'wb') as f:
            pickle.dump({'gmr': gmr_dmota, 'model_raw': model_dmota_raw}, f)

    # Plotting Mean and Std Fill
    print("🎨 Generating plot with variance...")
    plt.figure(figsize=(7, 5))
    plt.rcParams['font.family'] = 'serif'; plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['axes.labelsize'] = 12; plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10; plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 300

    plt.axhline(y=gmr_dmota, label='GMR-2 Baseline', color='black', linewidth=2.5, linestyle='--', zorder=10)

    colormap = plt.cm.get_cmap('viridis', len(cam_nums))
    for i, cam_num in enumerate(cam_nums):
        means = [np.mean(model_dmota_raw[cam_num][n_std]) for n_std in noise_levels]
        stds = [np.std(model_dmota_raw[cam_num][n_std]) for n_std in noise_levels]
        means, stds = np.array(means), np.array(stds)
        color = colormap(i)
        plt.plot(noise_levels, means, label=f'Ours-{cam_num}', color=color, marker='o', markersize=4, linewidth=1.5)
        plt.fill_between(noise_levels, means - stds, means + stds, color=color, alpha=0.15, edgecolor='none')

    plt.xlabel('Noise Standard Deviation (Pixels)')
    plt.ylabel('DRA')
    plt.xticks(noise_levels)
    plt.gca().spines['top'].set_visible(False); plt.gca().spines['right'].set_visible(False)
    plt.grid(True, which="major", axis="y", linestyle="-", alpha=0.3)
    plt.legend(loc='lower left', ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig('dra_noise_variance_comparison.png', bbox_inches='tight')
    plt.show()

def one_frame_dMOTA_3d_noise(force_recalculate=False):
    """
    Args:
        force_recalculate (bool): Set to True to ignore the cache and force a new calculation.
    """
    cache_filename = 'dra_mapped_3d_noise_cache.pkl'
    noise_levels_2d = list(range(5, 21, 3))  # [5, 8, 11, 14, 17, 20]
    num_trials = 10
    fixed_comp_num = 20

    if not force_recalculate and os.path.exists(cache_filename):
        print(f"✅ Found cached data in '{cache_filename}'. Loading results...")
        with open(cache_filename, 'rb') as f:
            cached_data = pickle.load(f)
            noisy_full_raw = cached_data['noisy_full']
            noisy_gmr_raw = cached_data['noisy_gmr']
            noise_levels_3d = cached_data['noise_levels_3d']
    else:
        print("⏳ No cache found. Starting mapped 3D noise computations...")

        run_params = DATASET_RUNS[0]
        name, _, start_step, end_step, step_length = _unpack(run_params)
        config, dataset, scenario_path = _load_scenario(name)
        _, _, step_range = _step_range(dataset, start_step, end_step, step_length)

        gt_data = np.load(scenario_path + '/reconstruction_scale.npz')
        gt_scales = gt_data['scales_gt']

        idx = 5
        time_step = step_range[idx]
        positions = dataset.positions_at_time_step(time_step)
        N = positions.shape[0]

        # Convert 2D Pixel Noise to 3D Spatial Noise
        cam_positions, _ = generate_encircling_cameras(dataset, step_range, config.intrinsics_params, config.H, config.W, cam_num=4, padding=1)
        swarm_center = np.mean(positions, axis=0)
        D = np.linalg.norm(cam_positions[0] - swarm_center)
        focal_length = config.intrinsics_params[0, 0].item() if torch.is_tensor(config.intrinsics_params) else config.intrinsics_params[0, 0]
        noise_levels_3d = [n_2d * (D / focal_length) for n_2d in noise_levels_2d]

        print(f"Calculated Swarm Distance (D): {D:.2f}")
        print(f"Focal Length (f): {focal_length:.2f}")
        for n2, n3 in zip(noise_levels_2d, noise_levels_3d):
            print(f"  Mapped 2D {n2}px -> 3D {n3:.5f} units")

        min_coords = np.min(positions, axis=0); max_coords = np.max(positions, axis=0)
        current_scale = gt_scales[idx]
        max_noise_3d = np.max(noise_levels_3d)
        bounds = np.vstack((min_coords - 3 * current_scale - max_noise_3d,
                            max_coords + 3 * current_scale + max_noise_3d)).T
        voxel_res = np.max(max_coords - min_coords) * 5e-3

        # Evaluate 3D Noise over Trials
        noisy_full_raw = {n: [] for n in noise_levels_2d}
        noisy_gmr_raw = {n: [] for n in noise_levels_2d}

        for n_2d, n_3d in zip(noise_levels_2d, noise_levels_3d):
            print(f"Testing Equivalent 2D Noise = {n_2d}px ({num_trials} trials)...")
            for trial in range(num_trials):
                np.random.seed(42 + trial + int(n_2d * 100))
                noise_3d_array = np.random.normal(0, n_3d, size=positions.shape)
                noisy_positions = positions + noise_3d_array

                # EVAL 1: Full Perturbed Density Field (Unreduced)
                pred_means_full = torch.from_numpy(noisy_positions).cuda().float()
                pred_weights_full = torch.ones((N, 1), device='cuda', dtype=torch.float)
                pred_sigmas_full = torch.full((N, 1), current_scale, device='cuda', dtype=torch.float)
                _, total_fp_full, total_fn_full = compute_metrics_batched_torch(
                    means1_np=positions, sigma1=current_scale,
                    pred_means=pred_means_full, pred_weights=pred_weights_full, pred_sigmas=pred_sigmas_full,
                    bounds=bounds, voxel_res=voxel_res, batch_size=50000, device='cuda',
                )
                noisy_full_raw[n_2d].append(1 - (total_fn_full + total_fp_full) / N)

                # EVAL 2: GMR Reduced Perturbed Density Field
                r_means, r_weights, r_covs = GMR.runnalls_algorithm_simple_torch(
                    means=torch.from_numpy(noisy_positions),
                    radii=torch.full((N, 1), current_scale, device='cuda', dtype=torch.float),
                    weights=torch.full((N, 1), 1.0, device='cuda', dtype=torch.float),
                    L=fixed_comp_num, DEVICE='cuda',
                )
                r_radius = torch.sqrt(r_covs[:, 0, 0]).reshape((-1, 1))
                _, total_fp_gmr, total_fn_gmr = compute_metrics_batched_torch(
                    means1_np=positions, sigma1=current_scale,
                    pred_means=r_means, pred_weights=r_weights, pred_sigmas=r_radius,
                    bounds=bounds, voxel_res=voxel_res, batch_size=50000, device='cuda',
                )
                noisy_gmr_raw[n_2d].append(1 - (total_fn_gmr + total_fp_gmr) / N)

        print(f"💾 Saving computed results to '{cache_filename}'...")
        with open(cache_filename, 'wb') as f:
            pickle.dump({'noisy_full': noisy_full_raw, 'noisy_gmr': noisy_gmr_raw, 'noise_levels_3d': noise_levels_3d}, f)

    # Plotting Mean and Std Fill
    print("🎨 Generating plot with variance...")
    plt.figure(figsize=(7, 5))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['figure.dpi'] = 300

    def plot_with_variance(data_dict, label, color, marker):
        means = [np.mean(data_dict[n]) for n in noise_levels_2d]
        stds = [np.std(data_dict[n]) for n in noise_levels_2d]
        means, stds = np.array(means), np.array(stds)
        plt.plot(noise_levels_2d, means, label=label, color=color, marker=marker, markersize=5, linewidth=2)
        plt.fill_between(noise_levels_2d, means - stds, means + stds, color=color, alpha=0.15, edgecolor='none')

    plot_with_variance(noisy_full_raw, 'Perturbed Full Field', color='#1f77b4', marker='o')
    plot_with_variance(noisy_gmr_raw, f'Perturbed GMR Field ({fixed_comp_num} modes)', color='#ff7f0e', marker='s')

    plt.xlabel('Equivalent 2D Noise Standard Deviation (Pixels)')
    plt.ylabel('DRA')
    plt.xticks(noise_levels_2d)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(True, which="major", axis="y", linestyle="-", alpha=0.3)
    plt.legend(loc='lower left', frameon=False)
    plt.tight_layout()
    plt.savefig('dra_mapped_3d_noise_comparison.png', bbox_inches='tight')
    plt.show()

def plot_dra_and_loss(run_params=None, baseline_deg=90, eval_every=1,
                      save_animation_to=None, animation_every=2, animation_fps=10,
                      animation_dpi=150):
    """Plot DRA (dMOTA) and rendering loss on dual y-axes over 100 training iters.

    Runs one reconstruction frame, saves per-iteration checkpoints, then
    evaluates the dMOTA metric against the ground-truth 3D density at every
    *eval_every* iteration and overlays it with the rendering loss curve.

    Left y-axis (log):  rendering loss (alternating cam 1 / cam 2)
    Right y-axis:       dMOTA  = 1 − (FP + FN) / N

    Parameters
    ----------
    run_params : dict or None
        Dataset run entry.  Defaults to the jackdaw dataset at its first
        available time step.
    baseline_deg : float
        Angular separation between the two cameras (degrees).
    eval_every : int
        Evaluate dMOTA every N training iterations.
    save_animation_to : str or None
        If a file path (e.g. ``"figs/density_anim.mp4"``), generate a 3D
        density-field animation showing the GMM evolving across training
        iterations and save it to this path.  Requires ``imageio`` or
        ``ffmpeg`` on PATH.
    animation_every : int
        Render every N-th training iteration (default 2 → ~50 frames).
    animation_fps : int
        Frame rate of the output animation.
    animation_dpi : int
        DPI for the individual animation frames.
    """
    from experiments.run_scenarios_angle_sweep import (
        _build_angled_cam_system, _precompute_gt_density, _build_grid, _compute_metrics_cached,
    )

    # ── defaults ──────────────────────────────────────────────────────
    if run_params is None:
        run_params = {"name": "jackdaw", "start_step": 350, "end_step": 360, "step_length": 10}

    name = run_params["name"]
    start_step = run_params["start_step"]
    end_step = run_params["end_step"]
    step_length = run_params["step_length"]
    config, dataset, scenario_path = _load_scenario(name)

    step_list = list(range(start_step, end_step, step_length))

    # Global bounding sphere → camera distance D
    all_positions = np.vstack([dataset.positions_at_time_step(t) for t in step_list])
    center = (all_positions.min(axis=0) + all_positions.max(axis=0)) / 2.0
    max_radius = np.max(np.linalg.norm(all_positions - center, axis=1))
    fx = config.intrinsics_params[0, 0]; fy = config.intrinsics_params[1, 1]
    cx = config.intrinsics_params[0, 2]; cy = config.intrinsics_params[1, 2]
    min_half_fov = min(np.arctan2(cx, fx), np.arctan2(config.W - cx, fx),
                       np.arctan2(cy, fy), np.arctan2(config.H - cy, fy))
    D = max_radius / np.sin(min_half_fov)

    gt_data = np.load(os.path.join(scenario_path, "reconstruction_scale.npz"))
    gt_scales = gt_data["scales_gt"]

    idx = 0
    time_step = step_list[idx]
    positions = dataset.positions_at_time_step(time_step)
    gt_scale = gt_scales[idx]

    train_iters = 100
    train_params = {
        "xyz_lr_c": 0.05, "xyz_lr_final_c": 0.9,
        "radius_lr_c": 0.05, "radius_lr_final_c": 0.9,
        "weights_lr_c": 0.10, "weights_lr_final_c": 0.7,
        "xyz_reg": 1.0, "radius_reg": 0.3, "radius_cutoff_inv": 0.5, "lr_max_steps": train_iters,
    }
    reconstruction_params = {
        "targetd_num_mode": 10, "voxel_scale": 0.5, "voxel_peak_threshold": 0.3,
        "voxel_grid_max_size": 32, "voxel_peaks_number": 2 * 10,
    }

    cam_system = _build_angled_cam_system(center, D, baseline_deg, config)

    # ── Pre-compute GT density grid ───────────────────────────────────
    print(f"Pre-computing GT density grid (N={positions.shape[0]})…")
    grid = _build_grid(positions, gt_scale)
    gt_density = _precompute_gt_density(positions, gt_scale, grid)

    # ── Run training with checkpointing ───────────────────────────────
    print(f"Running training for {train_iters} iterations…")
    dr = DensityReconstructor(max_iter=train_params["lr_max_steps"], use_decoupled=False)
    _, projections, _, _ = cam_system.simulate_vision(positions, renderer="projection_only")

    with tempfile.TemporaryDirectory() as tmpdir:
        model, _ = dr.process_frame(
            cam_system, point_sets=projections, positions=positions,
            initGMM=None, is_adaptive_scale=False, scale=gt_scale,
            is_store_intermediate=True, is_log=True,
            output_dir=tmpdir, debug=False,
            train_params=train_params, reconstruction_params=reconstruction_params,
        )

        # ── Extract loss history ──────────────────────────────────────
        loss_history = model[0].metrics_history["loss_history"]
        steps = np.arange(len(loss_history))

        # ── Load checkpoints & compute dMOTA ──────────────────────────
        history_path = os.path.join(tmpdir, "checkpoint_level_0.pth")
        training_history = GaussianModel.load_training_history(history_path)

        N = positions.shape[0]
        eval_iters = list(range(0, train_iters, eval_every))
        if train_iters - 1 not in eval_iters:
            eval_iters.append(train_iters - 1)
        dmota_vals = np.full(len(eval_iters), np.nan)

        for e_idx, it in enumerate(eval_iters):
            ckpt = training_history[it + 1]
            tp, fp, fn = _compute_metrics_cached(
                ckpt["_xyz"], ckpt["_weights"], ckpt["_radius"], gt_density, grid,
            )
            dmota_vals[e_idx] = 1.0 - (fn + fp) / N

    # ── Shared GMM frame renderer (used by both animation & static figure) ──
    def _draw_gmm_frame(
        ax, means, weights, sigmas,
        x_t_np, y_t_np, z_t_np, nx, ny, nz, total_voxels,
        dm, positions,
        gmm_color='#4169e1',
    ):
        """Render one GMM frame on *ax*: density shells + wireframes + agents.

        All wireframe alpha values are scaled by component weight and baked
        into RGBA tuples — identical rendering for both static PNGs and
        animation frames.
        """
        from dfr.utils import eval_isotropic_gmm_torch
        import matplotlib.colors as mcolors

        # --- Evaluate GMM density on the voxel grid (batched on GPU) -------
        x_t = torch.as_tensor(x_t_np, device="cuda")
        y_t = torch.as_tensor(y_t_np, device="cuda")
        z_t = torch.as_tensor(z_t_np, device="cuda")

        recon_flat = torch.empty(total_voxels, dtype=torch.float32, device="cpu")
        batch_size = 50000
        for start in range(0, total_voxels, batch_size):
            end = min(start + batch_size, total_voxels)
            idx = torch.arange(start, end, device="cuda")
            ix = idx // (ny * nz)
            iy = (idx // nz) % ny
            iz = idx % nz
            coords = torch.stack([x_t[ix], y_t[iy], z_t[iz]], dim=-1)
            dens = eval_isotropic_gmm_torch(coords, means, weights, sigmas)
            recon_flat[start:end] = dens.cpu()
        recon_3d = recon_flat.numpy().reshape(nx, ny, nz)

        # --- Density shells -------------------------------------------------
        norm_rec = mcolors.PowerNorm(gamma=0.35, vmin=0, vmax=dm)
        field_layers = [
            {'thresh': dm * 0.10, 'alpha_min': 0.18, 'alpha_max': 0.55, 'size': 8},
            {'thresh': dm * 0.02, 'alpha_min': 0.10, 'alpha_max': 0.40, 'size': 6},
            {'thresh': dm * 0.002,'alpha_min': 0.04, 'alpha_max': 0.22, 'size': 4},
        ]
        for layer in field_layers:
            mask = recon_3d >= layer['thresh']
            if not mask.any():
                continue
            ix_, iy_, iz_ = np.where(mask)
            pts = np.stack([x_t_np[ix_], y_t_np[iy_], z_t_np[iz_]], axis=-1)
            vals = recon_3d[mask]
            colors = plt.cm.viridis(norm_rec(vals))
            alphas = (norm_rec(vals)
                      * (layer['alpha_max'] - layer['alpha_min'])
                      + layer['alpha_min'])
            colors[:, 3] = np.clip(alphas, layer['alpha_min'], layer['alpha_max'])
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=colors, s=layer['size'], edgecolors='none',
                       depthshade=False, rasterized=True)

        # --- GMM wireframe ellipsoids (alpha scaled by weights) -------------
        means_np = means.detach().cpu().numpy()
        sigmas_np = sigmas.detach().cpu().numpy()
        weights_np = weights.detach().cpu().numpy()
        w_max = weights_np.max() if weights_np.size > 0 else 1.0
        K = means_np.shape[0]

        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        sx = np.outer(np.cos(u), np.sin(v))
        sy = np.outer(np.sin(u), np.sin(v))
        sz = np.outer(np.ones(np.size(u)), np.cos(v))

        wireframe_objs = []
        for j in range(K):
            r = float(sigmas_np[j])
            alpha = max(0.15, min(0.70, weights_np[j] / w_max)) if w_max > 0 else 0.25
            rgba = (*mcolors.to_rgb(gmm_color), alpha)
            wf = ax.plot_wireframe(
                means_np[j, 0] + r * sx,
                means_np[j, 1] + r * sy,
                means_np[j, 2] + r * sz,
                color=rgba,
                rstride=2, cstride=2, linewidth=1.7,
            )
            wireframe_objs.append(wf)

        for wf in wireframe_objs:
            _orig_wf = wf.do_3d_projection
            def _patch(orig=_orig_wf, obj=wf):
                orig()
                obj._sort_zpos = -5e8
                return obj._sort_zpos
            wf.do_3d_projection = _patch

        # --- GMM mean markers (forced in front of density) ------------------
        means_coll = ax.scatter(
            means_np[:, 0], means_np[:, 1], means_np[:, 2],
            c=gmm_color, marker='o', s=14, alpha=0.85,
            edgecolors='none', depthshade=True)
        _orig_means = means_coll.do_3d_projection
        def _force_means():
            _orig_means()
            means_coll._sort_zpos = -6e8
            return means_coll._sort_zpos
        means_coll.do_3d_projection = _force_means

        # --- Agent positions (forced on top of everything) ------------------
        agent_coll = ax.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            c='#1f2937', s=25, alpha=1.0, linewidths=0.8)
        _orig_agent = agent_coll.do_3d_projection
        def _force_agent():
            _orig_agent()
            agent_coll._sort_zpos = -1e9
            return agent_coll._sort_zpos
        agent_coll.do_3d_projection = _force_agent

    # ── 3D density-field animation ────────────────────────────────────
    if save_animation_to is not None:
        print(f"Generating 3D density-field animation "
              f"({len(range(0, train_iters, animation_every))} frames)…")

        # --- Precompute grid coordinate batches (same for every frame) ---
        x_t = grid["x_ticks"]
        y_t = grid["y_ticks"]
        z_t = grid["z_ticks"]
        nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
        total_voxels = grid["total_voxels"]

        x_t_np = x_t.cpu().numpy()
        y_t_np = y_t.cpu().numpy()
        z_t_np = z_t.cpu().numpy()

        # --- Colour scale anchored to GT density max --------------------
        dm = gt_density.max().item()

        view = dict(elev=33, azim=-117, roll=0)
        gmm_color = "#4169e1"

        # --- Determine which iterations to render -----------------------
        anim_iters = list(range(0, train_iters, animation_every))
        if train_iters - 1 not in anim_iters:
            anim_iters.append(train_iters - 1)

        frame_dir = tempfile.mkdtemp()
        frame_paths = []

        for frame_idx, it in enumerate(anim_iters):
            ckpt = training_history[it + 1]
            means = ckpt["_xyz"]
            weights = ckpt["_weights"].squeeze(-1)
            sigmas = ckpt["_radius"].squeeze(-1)

            # --- Build one frame (shared renderer) ----------------------------
            fig_f = plt.figure(figsize=(10, 10))
            ax_f = fig_f.add_subplot(111, projection="3d")
            ax_f.view_init(**view)
            ax_f.set_axis_off()

            _draw_gmm_frame(
                ax_f, means, weights, sigmas,
                x_t_np, y_t_np, z_t_np, nx, ny, nz, total_voxels,
                dm, positions, gmm_color=gmm_color,
            )

            # Iteration counter
            ax_f.text2D(0.02, 0.98, f"Iter {it}",
                        transform=ax_f.transAxes,
                        fontsize=14, fontweight="bold", va="top")

            fig_f.tight_layout(pad=0)
            fp = os.path.join(frame_dir, f"frame_{frame_idx:04d}.png")
            fig_f.patch.set_facecolor("white")
            fig_f.patch.set_alpha(1.0)
            ax_f.set_facecolor("white")

            fig_f.savefig(
                fp,
                dpi=animation_dpi,
                bbox_inches="tight",
                pad_inches=0,
                transparent=False,
                facecolor="white",
            )
            plt.close(fig_f)
            frame_paths.append(fp)

            if (frame_idx + 1) % 10 == 0 or frame_idx == len(anim_iters) - 1:
                print(f"  Rendered {frame_idx + 1}/{len(anim_iters)} animation frames")

        # --- Combine frames into a video file ---------------------------
        print(f"Combining {len(frame_paths)} frames → {save_animation_to} …")
        try:
            import imageio
            writer = imageio.get_writer(save_animation_to, fps=animation_fps)
            for fp in frame_paths:
                writer.append_data(imageio.imread(fp))
            writer.close()
        except ImportError:
            import subprocess
            cmd = [
                "ffmpeg", "-y", "-framerate", str(animation_fps),
                "-i", os.path.join(frame_dir, "frame_%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                save_animation_to,
            ]
            subprocess.run(cmd, check=True)

        # Clean up temporary frame directory
        shutil.rmtree(frame_dir, ignore_errors=True)
        print(f"Animation saved → {save_animation_to}")

    # ── GT density field figure (same style as plot_jackdaw2_density_field) ──
    print("Rendering GT density field figure…")
    import matplotlib.colors as mcolors

    out_dir = os.path.join(os.getcwd(), "figs")
    os.makedirs(out_dir, exist_ok=True)

    # Extract numpy arrays from the precomputed grid
    x_t = grid["x_ticks"]
    y_t = grid["y_ticks"]
    z_t = grid["z_ticks"]
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    x_t_np = x_t.cpu().numpy() if hasattr(x_t, "cpu") else np.array(x_t)
    y_t_np = y_t.cpu().numpy() if hasattr(y_t, "cpu") else np.array(y_t)
    z_t_np = z_t.cpu().numpy() if hasattr(z_t, "cpu") else np.array(z_t)
    density_3d = gt_density.numpy().reshape(nx, ny, nz)
    dm = density_3d.max()

    layers_gt = [
        {'thresh': dm * 0.10, 'alpha_min': 0.45, 'alpha_max': 0.95, 'size': 8},
        {'thresh': dm * 0.02, 'alpha_min': 0.25, 'alpha_max': 0.80, 'size': 6},
        {'thresh': dm * 0.002,'alpha_min': 0.08, 'alpha_max': 0.50, 'size': 4},
    ]
    view_gt = dict(elev=33, azim=-117, roll=0)

    fig_gt = plt.figure(figsize=(10, 10))
    ax_gt = fig_gt.add_subplot(111, projection='3d')
    ax_gt.view_init(**view_gt)
    ax_gt.set_axis_off()

    norm_gt = mcolors.PowerNorm(gamma=0.35, vmin=0, vmax=dm)

    for layer in layers_gt:
        mask = density_3d >= layer['thresh']
        if not mask.any():
            continue
        ix, iy, iz = np.where(mask)
        pts = np.stack([x_t_np[ix], y_t_np[iy], z_t_np[iz]], axis=-1)
        vals = density_3d[mask]
        colors = plt.cm.viridis(norm_gt(vals))
        alphas = norm_gt(vals) * (layer['alpha_max'] - layer['alpha_min']) + layer['alpha_min']
        colors[:, 3] = np.clip(alphas, layer['alpha_min'], layer['alpha_max'])
        ax_gt.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                      c=colors, s=layer['size'], edgecolors='none',
                      depthshade=False, rasterized=True)

    # Agent positions forced on top
    agent_coll_gt = ax_gt.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c='#1f2937', s=25, alpha=1.0, linewidths=0.8)
    _orig_gt = agent_coll_gt.do_3d_projection
    def _force_gt():
        _orig_gt()
        agent_coll_gt._sort_zpos = -1e9
        return agent_coll_gt._sort_zpos
    agent_coll_gt.do_3d_projection = _force_gt

    fig_gt.tight_layout(pad=0)
    out_gt = os.path.join(out_dir, f"dra_loss_{name}_{baseline_deg}deg_density_gt.png")
    fig_gt.savefig(out_gt, transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig_gt)
    print(f"Saved → {out_gt}")

    # ── Reconstructed GMM figure (same style as plot_jackdaw2_density_field Fig 2) ──
    print("Rendering reconstructed GMM figure…")

    recon_means = model[0]._xyz.detach()
    recon_weights = model[0]._weights.detach().squeeze(-1)
    recon_sigmas = model[0]._radius.detach().squeeze(-1)
    print(f"  Reconstructed model: {recon_means.shape[0]} components")

    total_voxels = grid["total_voxels"]

    fig_rec = plt.figure(figsize=(10, 10))
    ax_rec = fig_rec.add_subplot(111, projection='3d')
    ax_rec.view_init(**view_gt)
    ax_rec.set_axis_off()

    _draw_gmm_frame(
        ax_rec, recon_means, recon_weights, recon_sigmas,
        x_t_np, y_t_np, z_t_np, nx, ny, nz, total_voxels,
        dm, positions,
    )

    fig_rec.tight_layout(pad=0)
    out_rec = os.path.join(out_dir, f"dra_loss_{name}_{baseline_deg}deg_density_recon.png")
    fig_rec.savefig(out_rec, transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig_rec)
    print(f"Saved → {out_rec}")

    # ── Style ─────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
        "figure.dpi": 300,
    })

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_cam1 = "#1f77b4"; color_cam2 = "#d62728"; color_dmota = "#2ca02c"

    ax1.plot(steps[::2], loss_history[::2], color=color_cam1, linewidth=1.2, label="Loss Cam 1")
    ax1.plot(steps[1::2], loss_history[1::2], color=color_cam2, linewidth=1.2, linestyle="--", alpha=0.85, label="Loss Cam 2")
    ax1.set_yscale("log")
    ax1.set_xlabel("Training Step"); ax1.set_ylabel("Loss", color="#333333")
    ax1.tick_params(axis="y", labelcolor="#333333")
    ax1.set_ylim(bottom=max(1e-4, np.min(loss_history) * 0.5))

    ax2 = ax1.twinx()
    ax2.plot(np.array(eval_iters), dmota_vals, color=color_dmota, linewidth=2.2, markeredgewidth=1.5, label="DRA", zorder=5)
    ax2.set_ylabel("DRA", color=color_dmota)
    ax2.tick_params(axis="y", labelcolor=color_dmota)
    ax2.set_ylim(max(0.0, np.nanmin(dmota_vals) - 0.05), min(1.0, np.nanmax(dmota_vals) + 0.05))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", frameon=False, ncol=1)

    ax1.spines["top"].set_visible(False)
    ax1.grid(True, which="major", axis="y", linestyle="-", alpha=0.15)
    ax1.grid(True, which="minor", axis="y", linestyle=":", alpha=0.08)
    ax1.set_xlim(0, train_iters)

    fig.tight_layout()
    out_dir = os.path.join(os.getcwd(), "figs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"dra_loss_{name}_{baseline_deg}deg.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved → {out_path}")
    return fig, (ax1, ax2)

def plot_camera_configurations(dataset_name="swift"):
    """Generate a publication-quality figure showing encircling camera
    configurations for 2, 3, and 5 cameras overlaid on a single panel.

    Displays a top-down (XY-plane) view with the swarm point cloud, a
    shared camera orbit circle, and camera positions distinguished by
    colour and marker shape.  The camera generation logic matches
    ``experiments/run_scenarios.py`` exactly (the 2‑camera case generates
    4 positions and keeps the first two, yielding a 90° baseline).

    Parameters
    ----------
    dataset_name : str
        Name of the scenario dataset used to determine realistic swarm
        extent and camera distance.
    """
    config, dataset, scenario_path = _load_scenario(dataset_name)
    max_steps = dataset.trajectories.shape[0]
    step_range = range(0, max_steps, max(1, max_steps // 5))

    all_positions = np.vstack([dataset.positions_at_time_step(t) for t in step_range])
    center = (all_positions.min(axis=0) + all_positions.max(axis=0)) / 2.0
    max_radius = np.max(np.linalg.norm(all_positions - center, axis=1))
    positions = dataset.positions_at_time_step(step_range[0])

    # ── Academic styling ─────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times", "Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 10,
        "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    })

    CONFIG_STYLES = {
        2: {"color": "#D55E00", "marker": "o", "label": r"$K = 2$"},
        3: {"color": "#0072B2", "marker": "o", "label": r"$K = 3$"},
        5: {"color": "#009E73", "marker": "o", "label": r"$K = 5$"},
    }

    C_SWARM = "#9CA3AF"; C_ORBIT = "#D1D5DB"; C_BOUNDING = "#CBD5E0"; C_DIR = "#CBD5E0"
    C_ARC = "#D55E00"; C_ARC_TEXT = "#A04000"

    # ── Pre‑compute all camera positions ─────────────────────────
    cam_nums = [2, 3, 5]
    all_cameras = {}
    D = None
    for cam_num in cam_nums:
        if cam_num == 2:
            raw_positions, D = generate_encircling_cameras(dataset, step_range, config.intrinsics_params, config.H, config.W, cam_num=4, padding=1)
            all_cameras[cam_num] = raw_positions[:2]
        else:
            raw_positions, D = generate_encircling_cameras(dataset, step_range, config.intrinsics_params, config.H, config.W, cam_num=cam_num, padding=1)
            all_cameras[cam_num] = raw_positions

    # ── Build single‑panel figure ────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.2, 7.2))

    # Light shared orbit circle
    theta_full = np.linspace(0, 2 * np.pi, 600)
    ax.plot(center[0] + D * np.cos(theta_full), center[1] + D * np.sin(theta_full),
            color="#E5E7EB", linewidth=1.0, zorder=0)

    # Swarm points (top‑down projection)
    ax.scatter(positions[:, 0], positions[:, 1], c=C_SWARM, s=1.0, alpha=0.35, edgecolors="none", zorder=1)

    # Swarm bounding circle
    ax.add_patch(plt.Circle((center[0], center[1]), max_radius, fill=False, color="#CBD5E0",
                            linewidth=0.8, linestyle=(0, (4, 5)), alpha=0.65, zorder=2))

    # Per‑configuration: leader lines + markers
    for cam_num in cam_nums:
        style = CONFIG_STYLES[cam_num]
        cp = all_cameras[cam_num]
        for cam in cp:
            v = cam[:2] - center[:2]
            dist = np.linalg.norm(v)
            if dist < 1e-9:
                continue
            u = v / dist
            p_start = center[:2] + u * (max_radius * 1.05)
            ax.plot([p_start[0], cam[0]], [p_start[1], cam[1]],
                    color=style["color"], linewidth=1.0, linestyle=(0, (4, 5)), alpha=0.55, zorder=4)
            if cam_num == 2 and v[1] == 0:
                ax.plot([p_start[0], cam[0]], [p_start[1]+20, cam[1]+20],
                        color=style["color"], linewidth=1.0, linestyle=(0, (4, 5)), alpha=0.55, zorder=4)
            if cam_num == 3 and v[1] == 0:
                ax.plot([p_start[0], cam[0]], [p_start[1]-20, cam[1]-20],
                        color=style["color"], linewidth=1.0, linestyle=(0, (4, 5)), alpha=0.55, zorder=4)

        ax.scatter(cp[:, 0], cp[:, 1], marker=style["marker"], s=160,
                   c=style["color"], edgecolors="white", linewidths=1.8,
                   zorder=12, label=style["label"])

    # ── Legend ──────────────────────────────────────────────────
    legend = ax.legend(loc="lower right", frameon=True, fancybox=False,
                       edgecolor="#CCCCCC", facecolor="white",
                       framealpha=0.92, borderpad=0.6,
                       handletextpad=0.5, labelspacing=0.4)
    legend.set_zorder(20)

    # ── Axis limits & clean‑up ──────────────────────────────────
    pad = D * 0.25
    ax.set_xlim(center[0] - D - pad, center[0] + D + pad)
    ax.set_ylim(center[1] - D - pad, center[1] + D + pad)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    # ── Save & return ────────────────────────────────────────────
    out_dir = os.path.join(os.getcwd(), "figs")
    os.makedirs(out_dir, exist_ok=True)
    for fmt in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"camera_configurations.{fmt}"),
                    dpi=300, bbox_inches="tight", transparent=True)
    return fig, ax


if __name__ == "__main__":
    # scale_estimation()

    # plot_multiple_scenarios()

    # overview_scaling_law()
    # plot_scale_space_curve()

    # visual_hull_diagram()

    # assumption_3_error()

    # visual_hull_tau_vs_visual_hull_ghost()
    # run_geometric_visual_hulls()
    # run_params = DATASET_RUNS[0]

    # # Define the axes configurations
    # scale_range = np.linspace(0.3, 2.0, 10)
    # cam_num_range = np.array([3, 4, 6, 8, 10, 12, 15])

    # # Keep grid_res relatively low (e.g., 40-50) or this loop will take a long time
    # X_out, Y_out, Z_out = plot_ratio_surface(
    #     run_params,
    #     scales=scale_range,
    #     cam_nums=cam_num_range,
    #     base_tau=0.02,
    #     idx=5,
    #     grid_res=50
    # )

    # dra_metrics()

    # best_params = one_frame_parameter_search(n_trials=200)

    # one_frame_convergence()

    # one_frame_dMOTA_factor_analysis(force_recalculate=False)

    # one_frame_dMOTA_factor_analysis_2(force_recalculate=True)

    # one_frame_dMOTA_noise(force_recalculate=True)

    # one_frame_dMOTA_3d_noise()

    plot_dra_and_loss(
        save_animation_to="figs/density_anim.mp4",
        animation_every=1,   # ~50 frames
        animation_fps=10,    # 5-second animation
    )

    # plot_jackdaw2_density_field()

    # plot_jackdaw2_2d_observations()

    # plot_jackdaw2_2d_gmm()