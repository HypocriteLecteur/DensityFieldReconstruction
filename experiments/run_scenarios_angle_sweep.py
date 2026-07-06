import logging
import sys
import os
import shutil
from tqdm import tqdm
import glob
import argparse
from pathlib import Path



import time
import torch
import numpy as np
from dfr.simulation_config import SimulationConfig
from dfr.dataset_io import DatasetFactory
from dfr.camera_system import MultiCameraSystem
from dfr.density_field_reconstructor import DensityReconstructor
from dfr.density_field_model import GaussianModel
from dfr.camera_state import CameraState
from dfr.utils import calculate_gmm_dissimilarity, generate_encircling_cameras, compute_metrics_batched_torch
from dfr.visualizer import MultiGMMPlotter
from dfr.gaussian_mixture_reduction import GMR
from gaussian_rasterizer_simple_large import rasterize_gaussians
from dfr.utils import move_figure
from scipy.spatial import cKDTree
from experiments.common import setup_logger, setup_camera_system, print_global_metrics
from dfr import CameraConfig, OutputConfig, ScenarioRunSpec, run_scenario
from dfr.config import ReconstructionParams, TrainingParams

import matplotlib.pyplot as plt

# Setup logger
logger = setup_logger(__name__)

IS_LOGGING = False
CLEAN_LOGS = False

USE_DECOUPLED = False
USE_GT_SCALE = True

CAM_NUM = 2
LOG_NAME = 'base_reg_cam_2'

DATASET_RUNS = [
    {
        'name': 'starling',
        'log_name': LOG_NAME,
        'start_step': 0,
        'end_step': None,
        'step_length': 1,
    },
    {
        'name': 'swift',
        'log_name': LOG_NAME,
        'start_step': 0,
        'end_step': None,
        'step_length': 200,
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

VIS_LOG_NAME = 'base'
VIS_LOG_NAME2 = 'base_reg'
DATASET_VIS = [
    {
        'name': 'starling',
        'log_name': VIS_LOG_NAME,
        'log_name2': VIS_LOG_NAME2,
    },
    {
        'name': 'boids_multi',
        'log_name': VIS_LOG_NAME,
        'log_name2': VIS_LOG_NAME2,
    },
    {
        'name': 'clutter',
        'log_name': VIS_LOG_NAME,
        'log_name2': VIS_LOG_NAME2,
    },
]

def run_multi_scenarios():
    for run_params in DATASET_RUNS:
        run_single_scenario(run_params)

def run_single_scenario(run_params, *, project_root=None, output=None, seed=12345):
    """Run the ordinary scenario path through the shared package runner."""
    if CLEAN_LOGS:
        raise ValueError(
            "CLEAN_LOGS is unsupported for managed runs; use "
            "OutputConfig(overwrite=True)."
        )
    root = Path(project_root or Path.cwd()).expanduser().resolve()
    name = run_params['name']
    if output is None and IS_LOGGING:
        output = OutputConfig(
            workflow="reconstruction",
            name=f"angle-sweep {name}",
            run_id=f"angle-sweep-{name}-{run_params['log_name']}",
            project_root=root,
        )
    return run_scenario(
        ScenarioRunSpec(
            dataset=name,
            start=int(run_params['start_step']),
            stop=run_params['end_step'],
            step=int(run_params['step_length']),
            cameras=CameraConfig.encircling(count=CAM_NUM, device="cuda"),
            training=TrainingParams(
                xyz_lr_c=0.05,
                xyz_lr_final_c=0.9,
                radius_lr_c=0.05,
                radius_lr_final_c=0.9,
                weights_lr_c=0.10,
                weights_lr_final_c=0.7,
                xyz_reg=1.0,
                radius_reg=0.3,
                radius_cutoff_inv=0.5,
                lr_max_steps=100,
            ),
            reconstruction=ReconstructionParams(10, 0.5, 0.3, 32, 20),
            use_ground_truth_scales=USE_GT_SCALE,
            projection_noise_std=float(run_params.get('noise_std', 0.0)),
            use_decoupled=USE_DECOUPLED,
            seed=seed,
            output=output,
        ),
        project_root=root,
    )


def _run_single_scenario_legacy(run_params):
    # 1. Parameter extraction and Logging Setup
    name = run_params['name']
    log_name = run_params['log_name']
    start_step = run_params['start_step']
    end_step = run_params['end_step']
    step_length = run_params['step_length']

    noise_std = run_params.get('noise_std', 0.0)

    logger.info(f"Running scenario {name}")

    scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
    config_path = os.path.join(scenario_path, "config.yaml")

    if CLEAN_LOGS:
        if os.path.exists(os.path.join(scenario_path, "logs")):
            shutil.rmtree(os.path.join(scenario_path, "logs"))

        files_to_delete = glob.glob(os.path.join(scenario_path, 'metrics_*.npz'))
        for file_path in files_to_delete:
            try:
                os.remove(file_path)  # Deletes the file
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")
        return

    log_file_path = os.path.join(scenario_path, *["logs", log_name])
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)

    # 2. Initialize Metrics (must be re-initialized for each run)
    time_metrics = {
        'simulate_vision_time': [],
        'estimate_swarm_center': [],
        'adaptive_scale_selection': [],
        'generate_scale_space': [],
        'estimate_scale_space_peaks': [],
        'setup_gaussian_scale_space': [],
        'train_gaussian_scale_space': [],
    }
    loss_metrics = {
        'final_training_loss': [],
        'final_density_field_loss': [],
        'final_gmm_num': [],
        'scale': []
    }

    # 3. Load Dataset
    config = SimulationConfig(config_path)
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    max_steps = dataset.trajectories.shape[0]
    effective_end_step = end_step if end_step is not None and end_step <= max_steps else max_steps

    if start_step >= effective_end_step:
        logger.info(f"Skipping {name}: start_step ({start_step}) >= end_step ({effective_end_step}).")
        return

    step_range = range(start_step, effective_end_step, step_length)

    # Camera Configurations
    if CAM_NUM == 2:
        cam_positions, cam_radius = generate_encircling_cameras(dataset, step_range, config.intrinsics_params, config.H, config.W, cam_num=4, padding=1)
        cam_poses = np.hstack((cam_positions[:2], np.tile(np.array([1, 0, 0, 0]), (2, 1)))).astype(np.float32)
    else:
        cam_positions, cam_radius = generate_encircling_cameras(dataset, step_range, config.intrinsics_params, config.H, config.W, cam_num=CAM_NUM, padding=1)
        cam_poses = np.hstack((cam_positions, np.tile(np.array([1, 0, 0, 0]), (CAM_NUM, 1)))).astype(np.float32)

    # 4. System Initialization
    cam_system = MultiCameraSystem.create_homogeneous_system(
        state_class=CameraState,
        intrinsics=config.intrinsics_params,
        H=config.H, W=config.W,
        poses_or_RTs=cam_poses,
        near_clip=config.near_clip, far_clip=config.far_clip,
        size=config.size,
        device='cuda')
    reconstruction_params = {
        'targetd_num_mode': 10,
        # voxel method
        'voxel_scale': 0.5,
        'voxel_peak_threshold': 0.3,
        'voxel_grid_max_size': 32,
        'voxel_peaks_number': 2 * 10
    }
    train_params = {
        'xyz_lr_c': 0.05,
        'xyz_lr_final_c': 0.9,
        'radius_lr_c': 0.05,
        'radius_lr_final_c': 0.9,
        'weights_lr_c': 0.10,
        'weights_lr_final_c': 0.7,
        'xyz_reg': 1.0,
        'radius_reg': 0.3,
        'radius_cutoff_inv': 0.5,
        'lr_max_steps': 100
    }
    density_reconstructor = DensityReconstructor(max_iter=train_params['lr_max_steps'], use_decoupled=USE_DECOUPLED)

    if USE_GT_SCALE:
        gt_data = np.load(scenario_path + '/reconstruction_scale.npz')
        gt_scales = gt_data['scales_gt']

    # 5. Simulation Loop
    total_num = []
    for idx, time_step in enumerate(tqdm(step_range, desc=f"Processing {name}")):
        positions = dataset.positions_at_time_step(time_step)
        total_num.append(positions.shape[0])
        # poses, _, images, masks = cam_system.simulate_vision(positions, renderer='gaussian')
        poses, projections, _, masks = cam_system.simulate_vision(positions, renderer='projection_only')

        # add noise
        for cam_idx in range(len(projections)):
            max_w = cam_system.cameras[cam_idx].state.W
            max_h = cam_system.cameras[cam_idx].state.H

            new_projections = projections[cam_idx].copy()
            # Track which points still need valid noise
            needs_noise = np.ones(new_projections.shape[0], dtype=bool)

            while np.any(needs_noise):
                num_needs = np.sum(needs_noise)
                noise = np.random.normal(0, noise_std, size=(num_needs, 2))

                # Apply noise only to the original coordinates of the points that need it
                candidate_proj = projections[cam_idx][needs_noise] + noise

                # Check bounds
                in_bounds = (candidate_proj[:, 0] >= 0) & (candidate_proj[:, 0] <= max_w) & \
                            (candidate_proj[:, 1] >= 0) & (candidate_proj[:, 1] <= max_h)

                # Map the valid candidates back to their original indices
                valid_indices = np.where(needs_noise)[0][in_bounds]
                new_projections[valid_indices] = candidate_proj[in_bounds]

                # Mark these as done
                needs_noise[valid_indices] = False

            projections[cam_idx] = new_projections

        if USE_GT_SCALE:
            model, scale_spaces = \
            density_reconstructor.process_frame(cam_system, point_sets=projections, positions=positions,
                                                initGMM=None,
                                                is_adaptive_scale=False, scale=gt_scales[idx],
                                                is_store_intermediate=IS_LOGGING, is_log=IS_LOGGING,
                                                output_dir=os.path.join(log_file_path, f"t_{time_step:03d}"),
                                                debug=False,
                                                train_params=train_params,
                                                reconstruction_params=reconstruction_params)
        else:
            model, scale_spaces = \
            density_reconstructor.process_frame(cam_system, point_sets=projections, positions=positions,
                                                initGMM=None,
                                                is_adaptive_scale=True, scale=None,
                                                is_store_intermediate=IS_LOGGING, is_log=IS_LOGGING,
                                                output_dir=os.path.join(log_file_path, f"t_{time_step:03d}"),
                                                debug=False,
                                                train_params=train_params,
                                                reconstruction_params=reconstruction_params)

        # gmm_visualizer = MultiGMMPlotter()
        # gmm_visualizer.add_gmm(model[0]._xyz.detach().cpu().numpy(), model[0]._radius.detach().cpu().numpy(), model[0]._weights.detach().cpu().numpy())
        # gmm_visualizer.update()
        # move_figure(gmm_visualizer.fig, 2800, 100)
        # gmm_visualizer.ax.view_init(elev=33, azim=-117, roll=0)
        # # gmm_visualizer.fig.savefig("gmm_diagram.png", transparent=True, bbox_inches='tight')
        # plt.show()

        # 6. Collect Metrics
        for metric_name, value in density_reconstructor.time_metrics.items():
            time_metrics[metric_name].append(value)

        loss_metrics['final_training_loss'].append(model[0].mean_loss)
        loss_metrics['final_gmm_num'].append(model[0]._xyz.shape[0])
        loss_metrics['scale'].append(density_reconstructor.scale)

        is_visible = np.ones((positions.shape[0],), dtype=np.bool)
        for i in range(len(poses)):
            is_visible = is_visible & masks[i]
        loss_metrics['final_density_field_loss'].append(
            calculate_gmm_dissimilarity(
                positions[is_visible],
                density_reconstructor.scale,
                model[0]._xyz,
                model[0]._weights,
                model[0]._radius, use_decoupled=USE_DECOUPLED))

    # 7. Logging and Data Saving
    logger.info(f"Results for {name}:")
    if time_metrics['train_gaussian_scale_space']:
        mean_time = np.mean(np.array(time_metrics['train_gaussian_scale_space']))
        logger.info(f"Mean 'train_gaussian_scale_space' time: {mean_time:.2f} ms")
    else:
        logger.info("No time steps processed.")

    save_data = {**{k: np.array(v) for k, v in time_metrics.items()},
             **{k: np.array(v) for k, v in loss_metrics.items()}}

    save_path = os.path.join(log_file_path, "statistics.npz")
    if IS_LOGGING:
        np.savez(save_path, **save_data)
        logger.info(f"Statistics saved to: {save_path}")
    logger.info(f"Finished scenario {name}")

def run_multi_scenarios_baseline():
    for run_params in DATASET_RUNS:
        run_single_scenario_baseline(run_params)

def run_single_scenario_baseline(run_params):
    # 1. Parameter extraction and Logging Setup
    name = run_params['name']
    log_name = run_params['log_name']
    start_step = run_params['start_step']
    end_step = run_params['end_step']
    step_length = run_params['step_length']

    logger.info(f"Running scenario {name}")

    scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
    config_path = os.path.join(scenario_path, "config.yaml")

    log_file_path = os.path.join(scenario_path, *["logs", log_name])
    if not os.path.exists(log_file_path):
        ValueError(f'Log for {log_name} does not exist')

    log_file_path = os.path.join(scenario_path, *["logs", log_name])
    log_data = np.load(os.path.join(log_file_path, "statistics.npz"))
    scale_history = log_data['scale']
    gmm_num_history = log_data['final_gmm_num']

    # 2. Initialize Metrics (must be re-initialized for each run)
    loss_metrics = {
        'final_training_loss': [],
        'final_density_field_loss': [],
    }

    # 3. Load Dataset
    config = SimulationConfig(config_path)
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    max_steps = dataset.trajectories.shape[0]
    effective_end_step = end_step if end_step is not None and end_step <= max_steps else max_steps

    if start_step >= effective_end_step:
        logger.info(f"Skipping {name}: start_step ({start_step}) >= end_step ({effective_end_step}).")
        return

    # 4. System Initialization
    cam_system = MultiCameraSystem.create_homogeneous_system(
        state_class=CameraState,
        intrinsics=config.intrinsics_params,
        H=config.H, W=config.W,
        poses_or_RTs=config.cam_poses,
        near_clip=config.near_clip, far_clip=config.far_clip,
        size=config.size,
        device='cuda')

    # 5. Simulation Loop
    step_range = range(start_step, effective_end_step, step_length)

    for idx, time_step in enumerate(tqdm(step_range, desc=f"Processing {name}")):
        save_path = os.path.join(log_file_path, f"t_{time_step:03d}", f"baseline_level_{0}.pth")

        positions = dataset.positions_at_time_step(time_step)
        poses, _, images, masks = cam_system.simulate_vision(positions, renderer='gaussian')
        is_visible = np.ones((positions.shape[0],), dtype=np.bool)
        for i in range(len(poses)):
            is_visible = is_visible & masks[i]

        scale = scale_history[idx]
        gmm_num = gmm_num_history[idx]

        N = positions[is_visible].shape[0]

        r_means, r_weights, r_covs = GMR.runnalls_algorithm_simple_torch(
            means=torch.from_numpy(positions[is_visible]),
            radii=torch.full((N, 1), scale, device='cuda', dtype=torch.float),
            weights=torch.full((N, 1), 1.0, device='cuda', dtype=torch.float),
            L=gmm_num, DEVICE='cuda'
        )
        r_weights = r_weights.reshape((-1, 1))
        r_radius = torch.sqrt(r_covs[:, 0, 0]).reshape((-1, 1))

        final_means, final_unnorm_weights, final_covs = GMR.optimize_ise_isotropic(
            orig_means=torch.from_numpy(positions[is_visible]),
            orig_covs=(torch.eye(3, device='cuda') * (scale ** 2)).unsqueeze(0).expand(N, 3, 3),
            orig_weights=torch.full((N, 1), 1.0, device='cuda', dtype=torch.float),
            reduced_means=r_means,
            reduced_covs=r_covs,
            reduced_weights=r_weights,
            num_iterations=200,
            lr_mu_pct = 0.05,  # Percentage of data scale (e.g., 5%)
            lr_var = 0.05,     # Fixed LR for log-variances
            lr_weight = 0.05,  # Fixed LR for softmax logits
            DEVICE='cuda'
        )
        final_unnorm_weights = final_unnorm_weights.reshape((-1, 1))
        final_radius = torch.sqrt(final_covs[:, 0, 0]).reshape((-1, 1))

        checkpoint = {
            '_xyz': final_means.detach().clone(),
            '_radius': final_radius.detach().clone(),
            '_weights': final_unnorm_weights.detach().clone(),
            }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)

        # 6. Collect Metrics
        images_baseline = [
            rasterize_gaussians(
            final_means,
            final_radius,
            final_unnorm_weights,
            cam.state.R,
            cam.state.T,
            cam.state.K,
            cam.state.H,
            cam.state.W,
            False)
            for cam in cam_system.cameras
        ]

        # blur the images
        scale_spaces = []

        for i, (cam, img) in enumerate(zip(cam_system.cameras, images)):
            dist_cam = np.linalg.norm(cam.state.camera_center - np.mean(positions[is_visible], axis=0))

            pixel_scales = torch.tensor(
                [scale] / dist_cam * cam.state.intrinsics_params[0, 0].item(),
                device='cuda',
                dtype=torch.float32
            )

            scale_space = DensityReconstructor.generate_scale_space_img_rfft(img.to(dtype=torch.float32), pixel_scales)
            scale_spaces.append(scale_space)

        losses = [torch.sum(torch.abs(scale_spaces[i] - images_baseline[i])).item() for i in range(len(scale_spaces))]

        loss_metrics['final_training_loss'].append(losses[-1] + losses[0])

        loss_metrics['final_density_field_loss'].append(
            calculate_gmm_dissimilarity(
                positions[is_visible],
                scale,
                final_means,
                final_unnorm_weights,
                final_radius, use_decoupled=USE_DECOUPLED))

    # 7. Logging and Data Saving
    logger.info(f"Results for {name}:")

    save_data = {**{k: np.array(v) for k, v in loss_metrics.items()}}

    save_path = os.path.join(log_file_path, "statistics_baseline.npz")
    if IS_LOGGING:
        np.savez(save_path, **save_data)
        logger.info(f"Statistics saved to: {save_path}")
    logger.info(f"Finished scenario {name}")


# =============================================================================
# Helper functions for the baseline-angle sweep experiment
# =============================================================================

# ---- Camera setup ----

def _build_angled_cam_system(center, D, baseline_deg, config):
    """Create a 2-camera system with cam1 at 0° and cam2 at `baseline_deg` on
    the same circle of radius D around `center`."""
    cam_positions = []
    for angle_rad in [0.0, np.deg2rad(baseline_deg)]:
        px = center[0] + D * np.cos(angle_rad)
        py = center[1] + D * np.sin(angle_rad)
        pz = center[2]
        cam_positions.append(np.array([px, py, pz]))

    cam_poses = np.hstack((
        np.array(cam_positions),
        np.tile(np.array([1, 0, 0, 0]), (2, 1)),
    )).astype(np.float32)

    return MultiCameraSystem.create_homogeneous_system(
        state_class=CameraState,
        intrinsics=config.intrinsics_params,
        H=config.H, W=config.W,
        poses_or_RTs=cam_poses,
        near_clip=config.near_clip, far_clip=config.far_clip,
        size=config.size, device='cuda',
    )


# ---- GT density caching (avoids re-evaluating GT GMM on every angle) ----

def _build_grid(positions, gt_scale, voxel_res_factor=2.5e-2, device='cuda'):
    """Create the 3D voxel grid parameters shared by GT and pred evaluations."""
    min_c = np.min(positions, axis=0)
    max_c = np.max(positions, axis=0)
    extent = np.max(max_c - min_c)
    voxel_res = extent * voxel_res_factor
    bounds = np.vstack((min_c - 3 * gt_scale, max_c + 3 * gt_scale)).T

    x_ticks = torch.arange(bounds[0, 0], bounds[0, 1], voxel_res, device=device)
    y_ticks = torch.arange(bounds[1, 0], bounds[1, 1], voxel_res, device=device)
    z_ticks = torch.arange(bounds[2, 0], bounds[2, 1], voxel_res, device=device)

    return {
        'x_ticks': x_ticks, 'y_ticks': y_ticks, 'z_ticks': z_ticks,
        'nx': len(x_ticks), 'ny': len(y_ticks), 'nz': len(z_ticks),
        'voxel_res': voxel_res, 'total_voxels': len(x_ticks) * len(y_ticks) * len(z_ticks),
    }


def _precompute_gt_density(positions, gt_scale, grid, batch_size=50000, device='cuda'):
    """
    Evaluate the GT GMM (one isotropic Gaussian per 3D point) on the full voxel
    grid.  Returns a flat float32 tensor of density values, stored on CPU to
    avoid exhausting GPU memory across many frames.
    """
    N = positions.shape[0]
    gt_means = torch.from_numpy(positions).float().to(device)
    gt_weights = torch.full((N,), 1.0, device=device, dtype=torch.float)
    gt_sigmas = torch.full((N,), gt_scale, device=device, dtype=torch.float)

    x_t = grid['x_ticks']
    y_t = grid['y_ticks']
    z_t = grid['z_ticks']
    nx, ny, nz = grid['nx'], grid['ny'], grid['nz']
    total = grid['total_voxels']

    density_flat = torch.empty(total, dtype=torch.float32, device='cpu')

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        idx = torch.arange(start, end, device=device)
        ix = idx // (ny * nz)
        iy = (idx // nz) % ny
        iz = idx % nz
        coords = torch.stack([x_t[ix], y_t[iy], z_t[iz]], dim=-1)

        from dfr.utils import eval_isotropic_gmm_torch
        dens = eval_isotropic_gmm_torch(coords, gt_means, gt_weights, gt_sigmas)
        density_flat[start:end] = dens.cpu()

    return density_flat  # on CPU


def _compute_metrics_cached(pred_means, pred_weights, pred_radius,
                            gt_density_flat, grid, batch_size=50000, device='cuda'):
    """
    Compute TP/FP/FN using a *pre-computed* GT density grid (on CPU).
    Only evaluates the predicted GMM on-the-fly — the GT is a cheap lookup.
    """
    from dfr.utils import eval_isotropic_gmm_torch

    x_t = grid['x_ticks']
    y_t = grid['y_ticks']
    z_t = grid['z_ticks']
    nx, ny, nz = grid['nx'], grid['ny'], grid['nz']
    voxel_volume = grid['voxel_res'] ** 3
    total = grid['total_voxels']

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        idx = torch.arange(start, end, device=device)
        ix = idx // (ny * nz)
        iy = (idx // nz) % ny
        iz = idx % nz
        coords = torch.stack([x_t[ix], y_t[iy], z_t[iz]], dim=-1)

        # GT density: cheap CPU→GPU slice
        dens_gt = gt_density_flat[start:end].to(device)

        # Pred density: evaluate GMM (K ≈ 10-30 components → fast)
        dens_pred = eval_isotropic_gmm_torch(
            coords,
            pred_means.reshape(-1, 3),
            pred_weights.reshape(-1),
            pred_radius.reshape(-1),
        )

        total_tp += torch.sum(torch.minimum(dens_gt, dens_pred)).item() * voxel_volume
        total_fp += torch.sum(torch.clamp(dens_pred - dens_gt, min=0)).item() * voxel_volume
        total_fn += torch.sum(torch.clamp(dens_gt - dens_pred, min=0)).item() * voxel_volume

    return total_tp, total_fp, total_fn


def _compute_frame_metrics(positions, gt_scale, pred_means, pred_weights, pred_radius,
                           gt_cache=None):
    """
    Compute TP/FP/FN for one frame.  If `gt_cache` is a dict with pre-computed
    GT density + grid, use the fast cached path.
    """
    if gt_cache is not None:
        return _compute_metrics_cached(
            pred_means, pred_weights, pred_radius,
            gt_cache['density'], gt_cache['grid'],
        )
    # Fallback: original path (evaluates both GMMs)
    min_c = np.min(positions, axis=0)
    max_c = np.max(positions, axis=0)
    bounds = np.vstack((min_c - 3 * gt_scale, max_c + 3 * gt_scale)).T
    voxel_res = np.max(max_c - min_c) * 2.5e-2
    return compute_metrics_batched_torch(
        means1_np=positions, sigma1=gt_scale,
        pred_means=pred_means, pred_weights=pred_weights,
        pred_sigmas=pred_radius,
        bounds=bounds, voxel_res=voxel_res,
        batch_size=50000, device='cuda',
    )


def _build_gt_cache_for_frames(step_list, dataset, gt_scales, voxel_res_factor=2.5e-2):
    """
    Pre-compute GT density grids for every frame in `step_list`.
    Returns a list of dicts: [{'density': tensor, 'grid': dict}, …].
    Call once before the angle loop.
    """
    cache = []
    logger.info(f"Pre-computing GT density grids for {len(step_list)} frames "
                f"(voxel_res_factor={voxel_res_factor})…")
    for idx, time_step in enumerate(tqdm(step_list, desc="Caching GT densities")):
        positions = dataset.positions_at_time_step(time_step)
        grid = _build_grid(positions, gt_scales[idx], voxel_res_factor)
        density = _precompute_gt_density(positions, gt_scales[idx], grid)
        cache.append({'density': density, 'grid': grid})
    logger.info(f"GT cache built: {len(cache)} frames, "
                f"~{cache[0]['density'].numel() * 4 / 1024**2:.0f} MB/frame")
    return cache


def _build_gmr_init_gmm(positions, scale, target_L, device='cuda'):
    """
    Build a GMR-initialized GaussianModel from ground-truth 3D positions.

    1. Runnalls' algorithm: reduce N points → L components
    2. ISE optimisation: refine the reduced mixture
    3. Wrap in a GaussianModel suitable for `initGMM=[gm]`.
    """
    N = positions.shape[0]
    gt_means = torch.from_numpy(positions).float().to(device)
    gt_radii = torch.full((N, 1), scale, device=device, dtype=torch.float)
    gt_weights = torch.full((N, 1), 1.0, device=device, dtype=torch.float)

    # Step 1: Runnalls' reduction  N → L
    r_means, r_weights, r_covs = GMR.runnalls_algorithm_simple_torch(
        means=gt_means, radii=gt_radii, weights=gt_weights,
        L=target_L, DEVICE=device,
    )
    r_weights = r_weights.reshape((-1, 1))
    r_radius = torch.sqrt(r_covs[:, 0, 0]).reshape((-1, 1))

    # Step 2: ISE optimisation
    final_means, final_weights, final_covs = GMR.optimize_ise_isotropic(
        orig_means=gt_means,
        orig_covs=(torch.eye(3, device=device) * (scale ** 2)).unsqueeze(0).expand(N, 3, 3),
        orig_weights=gt_weights,
        reduced_means=r_means,
        reduced_covs=r_covs,
        reduced_weights=r_weights,
        num_iterations=200,
        lr_mu_pct=0.05,
        lr_var=0.05,
        lr_weight=0.05,
        DEVICE=device,
    )
    final_weights = final_weights.reshape((-1, 1))
    final_radius = torch.sqrt(final_covs[:, 0, 0]).reshape((-1, 1))

    # Wrap in a lightweight GaussianModel container (H/W don't matter here —
    # create_from_guess overwrites them via the rasterizer list anyway)
    gm = GaussianModel(H=1000, W=1000)
    gm._xyz = torch.nn.Parameter(final_means.detach().clone().requires_grad_(True))
    gm._radius = torch.nn.Parameter(final_radius.detach().clone().requires_grad_(True))
    gm._weights = torch.nn.Parameter(final_weights.detach().clone().requires_grad_(True))
    return gm


def _aggregate_global_metrics(tp_list, fp_list, fn_list, N_list, w_list):
    """Sum per-frame TP/FP/FN → global recall, hallucination, dMOTA."""
    s_tp = np.sum(tp_list)
    s_fp = np.sum(fp_list)
    s_fn = np.sum(fn_list)
    s_N  = np.sum(N_list)
    s_w  = np.sum(w_list)
    rec = s_tp / s_N if s_N > 0 else 0.0
    hal = s_fp / s_w if s_w > 0 else 0.0
    dm  = 1.0 - (s_fn + s_fp) / s_N if s_N > 0 else 0.0
    return rec, hal, dm


# =============================================================================
# Quick profiler: time each phase on a single angle + single timestep
# =============================================================================

def profile_bottleneck():
    """
    Run one reconstruction on a single frame and report wall-clock time for
    every phase so we can identify the dominant bottleneck.
    """
    dataset_name = 'jackdaw'
    time_step   = 400          # single frame
    baseline_deg = 90          # orthogonal

    scenario_path = os.path.join(os.getcwd(), "scenarios", dataset_name)
    config = SimulationConfig(os.path.join(scenario_path, "config.yaml"))
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    # Compute camera radius D (same as sweep)
    positions = dataset.positions_at_time_step(time_step)
    # Use a few frames for radius estimation
    all_positions = []
    for t in [350, 400, 450, 500, 550]:
        all_positions.append(dataset.positions_at_time_step(t))
    all_positions = np.vstack(all_positions)
    min_b = all_positions.min(axis=0)
    max_b = all_positions.max(axis=0)
    center = (min_b + max_b) / 2.0
    max_radius = np.max(np.linalg.norm(all_positions - center, axis=1))
    fx = config.intrinsics_params[0, 0]
    fy = config.intrinsics_params[1, 1]
    cx = config.intrinsics_params[0, 2]
    cy = config.intrinsics_params[1, 2]
    min_half_fov = min(np.arctan2(cx, fx), np.arctan2(config.W - cx, fx),
                       np.arctan2(cy, fy), np.arctan2(config.H - cy, fy))
    D = max_radius / np.sin(min_half_fov)

    gt_data = np.load(os.path.join(scenario_path, 'reconstruction_scale.npz'))
    gt_scales_all = gt_data['scales_gt']
    # Convert timestep → index (scales were saved with start_step / step_length stride)
    scale_idx = (time_step - 350) // 10   # start_step=350, step_length=10
    gt_scale = gt_scales_all[scale_idx]
    print(f"  Using gt_scale[{scale_idx}] = {gt_scale:.3f}")

    train_params = {
        'xyz_lr_c': 0.05, 'xyz_lr_final_c': 0.9,
        'radius_lr_c': 0.05, 'radius_lr_final_c': 0.9,
        'weights_lr_c': 0.10, 'weights_lr_final_c': 0.7,
        'xyz_reg': 1.0, 'radius_reg': 0.3,
        'radius_cutoff_inv': 0.5, 'lr_max_steps': 500,
    }
    reconstruction_params = {
        'targetd_num_mode': 10,
        'voxel_scale': 0.5, 'voxel_peak_threshold': 0.3,
        'voxel_grid_max_size': 32, 'voxel_peaks_number': 2 * 10,
    }

    density_reconstructor = DensityReconstructor(
        max_iter=train_params['lr_max_steps'],
        use_decoupled=USE_DECOUPLED,
    )

    cam_system = _build_angled_cam_system(center, D, baseline_deg, config)

    # ---- Phase timing ----
    times = {}

    t0 = time.perf_counter()
    _, projections, _, _ = cam_system.simulate_vision(
        positions, renderer='projection_only',
    )
    torch.cuda.synchronize()
    times['simulate_vision'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    model, _ = density_reconstructor.process_frame(
        cam_system, point_sets=projections, positions=positions,
        initGMM=None,
        is_adaptive_scale=False, scale=gt_scale,
        is_store_intermediate=False, is_log=False,
        output_dir=None, debug=False,
        train_params=train_params,
        reconstruction_params=reconstruction_params,
    )
    torch.cuda.synchronize()
    times['process_frame (total)'] = time.perf_counter() - t0

    # Sub-phases from density_reconstructor.time_metrics (scalar floats, ms)
    for k, v in density_reconstructor.time_metrics.items():
        if v:
            times[f'  |-- {k}'] = v  # already in ms

    n_comp = model[0]._xyz.shape[0]

    t0 = time.perf_counter()
    tp, fp, fn = _compute_frame_metrics(
        positions, gt_scale,
        model[0]._xyz, model[0]._weights, model[0]._radius,
    )
    torch.cuda.synchronize()
    times['compute_metrics'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = _build_gmr_init_gmm(positions, gt_scale, max(int(n_comp), 2))
    torch.cuda.synchronize()
    times['GMR init (Runnalls + ISE)'] = time.perf_counter() - t0

    # ---- Report ----
    print("\n" + "=" * 65)
    print(f"BOTTLENECK PROFILE  |  {dataset_name}  frame {time_step}  "
          f"baseline={baseline_deg}°  N={positions.shape[0]}  K={n_comp}")
    print("=" * 65)
    total = times['process_frame (total)'] + times['compute_metrics'] + times['GMR init (Runnalls + ISE)']
    for label, dt in times.items():
        if label.startswith('  |--'):
            # time_metrics values are in ms
            pct = dt / (total * 1000) * 100 if total > 0 else 0
            print(f"  {label:<48s} {dt:8.1f} ms  ({pct:5.1f}%)")
        else:
            pct = dt / total * 100 if total > 0 else 0
            print(f"  {label:<48s} {dt:8.2f} s  ({pct:5.1f}%)")
    print("-" * 65)
    print(f"  {'TOTAL':<42s} {total:8.2f} s")
    print(f"  Voxel grid for metrics: ~{_estimate_voxel_count(positions, gt_scale):,} voxels")
    print("=" * 65 + "\n")


def _estimate_voxel_count(positions, scale):
    """Rough estimate of the 3D grid size used by compute_metrics_batched_torch."""
    min_c = np.min(positions, axis=0)
    max_c = np.max(positions, axis=0)
    extent = np.max(max_c - min_c)
    voxel_res = extent * 2.5e-2
    span = (max_c - min_c) + 6 * scale  # 3σ padding each side
    return int(np.prod(np.ceil(span / voxel_res)))


# =============================================================================
# Voxel-resolution sensitivity test — how coarse can we go before metrics drift?
# =============================================================================

def test_voxel_coarsening():
    """
    Run reconstruction on ONE frame at ONE angle, then compute the three metrics
    using a range of voxel resolutions.  Plots metrics + runtime vs voxel count
    to reveal the accuracy/speed trade-off.
    """
    dataset_name = 'jackdaw'
    time_step   = 400
    baseline_deg = 90

    scenario_path = os.path.join(os.getcwd(), "scenarios", dataset_name)
    config = SimulationConfig(os.path.join(scenario_path, "config.yaml"))
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    positions = dataset.positions_at_time_step(time_step)
    all_positions = []
    for t in [350, 400, 450, 500, 550]:
        all_positions.append(dataset.positions_at_time_step(t))
    all_positions = np.vstack(all_positions)
    min_b = all_positions.min(axis=0)
    max_b = all_positions.max(axis=0)
    center = (min_b + max_b) / 2.0
    max_radius = np.max(np.linalg.norm(all_positions - center, axis=1))
    fx = config.intrinsics_params[0, 0]; fy = config.intrinsics_params[1, 1]
    cx = config.intrinsics_params[0, 2]; cy = config.intrinsics_params[1, 2]
    min_half_fov = min(np.arctan2(cx, fx), np.arctan2(config.W - cx, fx),
                       np.arctan2(cy, fy), np.arctan2(config.H - cy, fy))
    D = max_radius / np.sin(min_half_fov)

    gt_data = np.load(os.path.join(scenario_path, 'reconstruction_scale.npz'))
    gt_scale = gt_data['scales_gt'][(time_step - 350) // 10]

    train_params = {
        'xyz_lr_c': 0.05, 'xyz_lr_final_c': 0.9,
        'radius_lr_c': 0.05, 'radius_lr_final_c': 0.9,
        'weights_lr_c': 0.10, 'weights_lr_final_c': 0.7,
        'xyz_reg': 1.0, 'radius_reg': 0.3,
        'radius_cutoff_inv': 0.5, 'lr_max_steps': 500,
    }
    reconstruction_params = {
        'targetd_num_mode': 10,
        'voxel_scale': 0.5, 'voxel_peak_threshold': 0.3,
        'voxel_grid_max_size': 32, 'voxel_peaks_number': 2 * 10,
    }

    dr = DensityReconstructor(max_iter=train_params['lr_max_steps'],
                              use_decoupled=USE_DECOUPLED)
    cam_system = _build_angled_cam_system(center, D, baseline_deg, config)

    # Run reconstruction once
    print("Running reconstruction for coarsening test…")
    _, projections, _, _ = cam_system.simulate_vision(positions, renderer='projection_only')
    model, _ = dr.process_frame(
        cam_system, point_sets=projections, positions=positions,
        initGMM=None, is_adaptive_scale=False, scale=gt_scale,
        is_store_intermediate=False, is_log=False,
        output_dir=None, debug=False,
        train_params=train_params,
        reconstruction_params=reconstruction_params,
    )
    pred_means = model[0]._xyz
    pred_weights = model[0]._weights
    pred_radius = model[0]._radius

    # ---- Sweep voxel resolutions ----
    factors = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
    voxel_counts = []
    times = []
    recalls, halls, dmotas = [], [], []

    # "Ground truth" metrics at the finest resolution
    N = positions.shape[0]

    for factor in factors:
        grid = _build_grid(positions, gt_scale, voxel_res_factor=factor)
        density = _precompute_gt_density(positions, gt_scale, grid)

        t0 = time.perf_counter()
        tp, fp, fn = _compute_metrics_cached(
            pred_means, pred_weights, pred_radius, density, grid)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        rec = tp / N if N > 0 else 0.0
        hal = fp / pred_weights.sum().item() if pred_weights.sum().item() > 0 else 0.0
        dm  = 1.0 - (fn + fp) / N if N > 0 else 0.0

        voxel_counts.append(grid['total_voxels'])
        times.append(elapsed)
        recalls.append(rec)
        halls.append(hal)
        dmotas.append(dm)

        print(f"  factor={factor:.3f}  voxels={grid['total_voxels']:,}  "
              f"time={elapsed:.3f}s  R={rec:.4f}  H={hal:.4f}  dMOTA={dm:.4f}")

    # ---- Plot ----
    voxels_arr = np.array(voxel_counts)
    times_arr = np.array(times)
    ref_idx = 0  # finest resolution as reference

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    # Top row: metrics vs voxels
    for ax, yvals, ylabel, title in [
        (axes[0, 0], recalls, 'Recall', 'Recall vs Voxel Count'),
        (axes[0, 1], halls,   'Hallucination', 'Hallucination vs Voxel Count'),
        (axes[0, 2], dmotas,  'dMOTA', 'dMOTA vs Voxel Count'),
    ]:
        ax.semilogx(voxels_arr, yvals, 'o-', linewidth=2, markersize=8)
        ax.axhline(y=yvals[ref_idx], color='gray', linestyle=':', alpha=0.5,
                   label=f'Finest ({voxels_arr[ref_idx]:,} vox)')
        ax.set_xlabel('Voxel count')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Bottom row: delta from reference + time
    axes[1, 0].semilogx(voxels_arr, np.abs(np.array(recalls) - recalls[ref_idx]),
                        'o-', color='#2ca02c', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Voxel count')
    axes[1, 0].set_ylabel('|Δ Recall|')
    axes[1, 0].set_title('Recall error vs finest resolution')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].semilogx(voxels_arr, np.abs(np.array(dmotas) - dmotas[ref_idx]),
                        'o-', color='#1f77b4', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Voxel count')
    axes[1, 1].set_ylabel('|Δ dMOTA|')
    axes[1, 1].set_title('dMOTA error vs finest resolution')
    axes[1, 1].grid(True, alpha=0.3)

    ax_t = axes[1, 2]
    ax_t.loglog(voxels_arr, times_arr, 's-', color='#d62728', linewidth=2, markersize=10)
    ax_t.set_xlabel('Voxel count')
    ax_t.set_ylabel('Compute time (s)')
    ax_t.set_title('Metrics computation time')
    ax_t.grid(True, alpha=0.3)
    # Annotate current default
    current = np.argmin(np.abs(np.array(factors) - 0.005))
    ax_t.annotate(f'Current default\n({voxels_arr[current]:,} vox)',
                  (voxels_arr[current], times_arr[current]),
                  textcoords="offset points", xytext=(15, 15),
                  arrowprops=dict(arrowstyle='->'), fontsize=9)

    fig.suptitle(
        f'Effect of Voxel Grid Resolution on Metric Accuracy & Speed\n'
        f'{dataset_name} frame {time_step}, N={N}, baseline={baseline_deg}°',
        fontsize=14, fontweight='bold')
    plt.tight_layout()

    sweep_dir = os.path.join(scenario_path, "logs", "baseline_sweep")
    fig.savefig(os.path.join(sweep_dir, "voxel_coarsening_sensitivity.png"),
                dpi=150, bbox_inches='tight')
    print(f"\nSaved to {sweep_dir}/voxel_coarsening_sensitivity.png")

    return voxel_counts, recalls, halls, dmotas, times


# =============================================================================
# Training convergence — track metrics *during* Adam optimisation
# =============================================================================

def run_training_convergence():
    """
    Track metrics *during* Adam optimisation across multiple timesteps and
    baseline angles (10°–90°, every 20°).  For each (angle, frame) pair,
    checkpoints are saved every iteration; metrics are evaluated every
    `eval_every` iterations and then **averaged across frames**.

    Plots a 5-row × 3-col grid:
      - rows    = baseline angles  (10°, 30°, 50°, 70°, 90°)
      - columns = metrics           (Recall, Hallucination, dMOTA)
    Each subplot shows Default-init vs GMR-init convergence curves
    (mean ± std across timesteps as shaded band).
    """
    import tempfile

    dataset_name = 'jackdaw'
    start_step = 350
    end_step   = 550
    step_length = 10           # 20 frames available

    # Use every 4th frame → 5 frames
    step_list = list(range(start_step, end_step, step_length))
    time_steps = step_list[::4]    # e.g. [350, 390, 430, 470, 510]
    n_frames = len(time_steps)

    angles      = np.arange(10, 91, 20)   # [10, 30, 50, 70, 90]
    train_iters = 100
    eval_every  = 25                       # sample every 25 iterations

    scenario_path = os.path.join(os.getcwd(), "scenarios", dataset_name)
    config = SimulationConfig(os.path.join(scenario_path, "config.yaml"))
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    # Shared camera radius D (aggregate over selected frames)
    all_positions = []
    for t in step_list:
        all_positions.append(dataset.positions_at_time_step(t))
    all_positions = np.vstack(all_positions)
    min_b = all_positions.min(axis=0); max_b = all_positions.max(axis=0)
    center = (min_b + max_b) / 2.0
    max_radius = np.max(np.linalg.norm(all_positions - center, axis=1))
    fx = config.intrinsics_params[0, 0]; fy = config.intrinsics_params[1, 1]
    cx = config.intrinsics_params[0, 2]; cy = config.intrinsics_params[1, 2]
    min_half_fov = min(np.arctan2(cx, fx), np.arctan2(config.W - cx, fx),
                       np.arctan2(cy, fy), np.arctan2(config.H - cy, fy))
    D = max_radius / np.sin(min_half_fov)

    gt_data = np.load(os.path.join(scenario_path, 'reconstruction_scale.npz'))
    gt_scales_all = gt_data['scales_gt']

    train_params = {
        'xyz_lr_c': 0.05, 'xyz_lr_final_c': 0.9,
        'radius_lr_c': 0.05, 'radius_lr_final_c': 0.9,
        'weights_lr_c': 0.10, 'weights_lr_final_c': 0.7,
        'xyz_reg': 1.0, 'radius_reg': 0.3,
        'radius_cutoff_inv': 0.5, 'lr_max_steps': train_iters,
    }
    reconstruction_params = {
        'targetd_num_mode': 10,
        'voxel_scale': 0.5, 'voxel_peak_threshold': 0.3,
        'voxel_grid_max_size': 32, 'voxel_peaks_number': 2 * 10,
    }

    # ---- Pre-compute GT density grids for all frames ----
    print(f"Pre-computing GT density grids for {n_frames} frames…")
    gt_caches = []
    for frame_idx, ts in enumerate(time_steps):
        positions = dataset.positions_at_time_step(ts)
        scale_idx = (ts - start_step) // step_length
        gt_scale = gt_scales_all[scale_idx]
        grid = _build_grid(positions, gt_scale)
        density = _precompute_gt_density(positions, gt_scale, grid)
        gt_caches.append({
            'positions': positions,
            'gt_scale': gt_scale,
            'scale_idx': scale_idx,
            'grid': grid,
            'density': density,
            'N': positions.shape[0],
        })

    # ---- Pre-compute K (component count) per (angle, frame) for GMR init ----
    # We run a quick default reconstruction without checkpointing.
    print(f"Discovering component counts K for GMR init "
          f"({len(angles)} angles × {n_frames} frames)…")
    K_lookup = {}   # K_lookup[angle][frame_idx] = int

    for baseline_deg in angles:
        K_lookup[baseline_deg] = {}
        cam_system = _build_angled_cam_system(center, D, baseline_deg, config)
        for frame_idx, gc in enumerate(gt_caches):
            positions = gc['positions']
            gt_scale = gc['gt_scale']
            _, projections, _, _ = cam_system.simulate_vision(
                positions, renderer='projection_only')
            dr_temp = DensityReconstructor(
                max_iter=train_params['lr_max_steps'],
                use_decoupled=USE_DECOUPLED)
            model_temp, _ = dr_temp.process_frame(
                cam_system, point_sets=projections, positions=positions,
                initGMM=None,
                is_adaptive_scale=False, scale=gt_scale,
                is_store_intermediate=False, is_log=False,
                output_dir=None, debug=False,
                train_params=train_params,
                reconstruction_params=reconstruction_params)
            K_lookup[baseline_deg][frame_idx] = max(
                int(model_temp[0]._xyz.shape[0]), 2)
            del dr_temp, model_temp
            torch.cuda.empty_cache()

    # ---- Main loop: for each (angle, init_type), run training + eval ----
    # Structure: results[angle][init_name]['iters'] = shared array
    #            results[angle][init_name]['recall'] = (n_frames, n_iters) array
    all_results = {}

    eval_iter_nums = list(range(0, train_iters, eval_every))
    if train_iters - 1 not in eval_iter_nums:
        eval_iter_nums.append(train_iters - 1)
    n_evals = len(eval_iter_nums)

    for baseline_deg in tqdm(angles, desc="Angles"):
        all_results[baseline_deg] = {}
        cam_system = _build_angled_cam_system(center, D, baseline_deg, config)

        for init_name in ['Default', 'GMR']:
            # Accumulate per-frame metrics: (n_frames, n_evals)
            rec_all = np.full((n_frames, n_evals), np.nan)
            hal_all = np.full((n_frames, n_evals), np.nan)
            dm_all  = np.full((n_frames, n_evals), np.nan)

            for frame_idx, gc in enumerate(tqdm(
                gt_caches, desc=f'{baseline_deg}° {init_name}', leave=False)):
                positions = gc['positions']
                gt_scale  = gc['gt_scale']
                gt_density = gc['density']
                grid = gc['grid']

                # Build GMR init if needed
                gmr_init = None
                if init_name == 'GMR':
                    target_L = K_lookup[baseline_deg][frame_idx]
                    gmr_init = [_build_gmr_init_gmm(positions, gt_scale, target_L)]

                with tempfile.TemporaryDirectory() as tmpdir:
                    dr = DensityReconstructor(
                        max_iter=train_params['lr_max_steps'],
                        use_decoupled=USE_DECOUPLED)

                    _, projections, _, _ = cam_system.simulate_vision(
                        positions, renderer='projection_only')

                    dr.process_frame(
                        cam_system, point_sets=projections, positions=positions,
                        initGMM=gmr_init,
                        is_adaptive_scale=False, scale=gt_scale,
                        is_store_intermediate=True, is_log=False,
                        output_dir=tmpdir, debug=False,
                        train_params=train_params,
                        reconstruction_params=reconstruction_params)

                    # Load history and evaluate
                    history_path = os.path.join(tmpdir, "checkpoint_level_0.pth")
                    training_history = GaussianModel.load_training_history(
                        history_path)

                    N = positions.shape[0]
                    for e_idx, iter_num in enumerate(eval_iter_nums):
                        ckpt = training_history[iter_num + 1]  # +1 offset
                        tp, fp, fn = _compute_metrics_cached(
                            ckpt['_xyz'], ckpt['_weights'], ckpt['_radius'],
                            gt_density, grid)
                        w_sum = ckpt['_weights'].sum().item()
                        rec_all[frame_idx, e_idx] = tp / N if N > 0 else 0.0
                        hal_all[frame_idx, e_idx] = (fp / w_sum
                                                     if w_sum > 0 else 0.0)
                        dm_all[frame_idx, e_idx] = (1.0 - (fn + fp) / N
                                                    if N > 0 else 0.0)

            # Average across frames
            rec_mean = np.nanmean(rec_all, axis=0)
            rec_std  = np.nanstd(rec_all, axis=0)
            hal_mean = np.nanmean(hal_all, axis=0)
            hal_std  = np.nanstd(hal_all, axis=0)
            dm_mean  = np.nanmean(dm_all, axis=0)
            dm_std   = np.nanstd(dm_all, axis=0)

            all_results[baseline_deg][init_name] = {
                'iters': np.array(eval_iter_nums),
                'recall_mean': rec_mean, 'recall_std': rec_std,
                'hallucination_mean': hal_mean, 'hallucination_std': hal_std,
                'dMOTA_mean': dm_mean, 'dMOTA_std': dm_std,
            }

    # ---- Plot: 5 rows (angles) × 3 cols (metrics) ----
    n_angles = len(angles)
    fig, axes = plt.subplots(n_angles, 3, figsize=(20, 4.5 * n_angles))

    colors = {'Default': '#d62728', 'GMR': '#2ca02c'}

    for row, baseline_deg in enumerate(angles):
        for init_name in ['Default', 'GMR']:
            r = all_results[baseline_deg][init_name]
            it = r['iters']
            c = colors[init_name]
            lbl_def = f'{init_name}'

            # Recall
            ax = axes[row, 0]
            ax.plot(it, r['recall_mean'], color=c, linewidth=2, label=lbl_def)
            ax.fill_between(it,
                            r['recall_mean'] - r['recall_std'],
                            r['recall_mean'] + r['recall_std'],
                            color=c, alpha=0.12)
            # Hallucination
            ax = axes[row, 1]
            ax.plot(it, r['hallucination_mean'], color=c, linewidth=2,
                    label=lbl_def)
            ax.fill_between(it,
                            r['hallucination_mean'] - r['hallucination_std'],
                            r['hallucination_mean'] + r['hallucination_std'],
                            color=c, alpha=0.12)
            # dMOTA
            ax = axes[row, 2]
            ax.plot(it, r['dMOTA_mean'], color=c, linewidth=2, label=lbl_def)
            ax.fill_between(it,
                            r['dMOTA_mean'] - r['dMOTA_std'],
                            r['dMOTA_mean'] + r['dMOTA_std'],
                            color=c, alpha=0.12)

        for col, ylabel, title in [
            (0, 'Recall (Coverage)', 'Recall'),
            (1, 'Hallucination (FP Rate)', 'Hallucination'),
            (2, 'dMOTA', 'dMOTA'),
        ]:
            ax = axes[row, col]
            ax.set_xlabel('Training iteration')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{title}  —  baseline = {baseline_deg}°')
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.legend(fontsize=8)

    fig.suptitle(
        f'Training Convergence: Default Init vs GMR Init\n'
        f'{dataset_name}  |  {n_frames} frames (steps {time_steps[0]}–'
        f'{time_steps[-1]})  |  mean ± 1σ across frames',
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout()

    sweep_dir = os.path.join(scenario_path, "logs", "baseline_sweep")
    fig.savefig(os.path.join(sweep_dir, "training_convergence.png"),
                dpi=150, bbox_inches='tight')
    print(f"\nSaved to {sweep_dir}/training_convergence.png")

    return all_results



# =============================================================================
# Convergence diagnosis — identify WHY one frame converges slowly
# =============================================================================

def diagnose_slow_convergence():
    """
    Run Default-init reconstruction at 90° baseline for each of the 5 frames
    individually, tracking per-iteration metrics.  Identify which frame(s)
    converge slowly and print diagnostic info to explain why.
    """
    import tempfile
    
    dataset_name = 'jackdaw'
    start_step = 350
    end_step   = 550
    step_length = 10
    step_list   = list(range(start_step, end_step, step_length))
    time_steps  = step_list[::4]               # 5 frames
    baseline_deg = 90
    train_iters  = 500

    scenario_path = os.path.join(os.getcwd(), "scenarios", dataset_name)
    config = SimulationConfig(os.path.join(scenario_path, "config.yaml"))
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    # Shared camera radius D
    all_positions = []
    for t in step_list:
        all_positions.append(dataset.positions_at_time_step(t))
    all_positions = np.vstack(all_positions)
    min_b = all_positions.min(axis=0); max_b = all_positions.max(axis=0)
    center = (min_b + max_b) / 2.0
    max_radius = np.max(np.linalg.norm(all_positions - center, axis=1))
    fx = config.intrinsics_params[0, 0]; fy = config.intrinsics_params[1, 1]
    cx = config.intrinsics_params[0, 2]; cy = config.intrinsics_params[1, 2]
    min_half_fov = min(np.arctan2(cx, fx), np.arctan2(config.W - cx, fx),
                       np.arctan2(cy, fy), np.arctan2(config.H - cy, fy))
    D = max_radius / np.sin(min_half_fov)

    gt_data = np.load(os.path.join(scenario_path, 'reconstruction_scale.npz'))
    gt_scales_all = gt_data['scales_gt']

    train_params = {
        'xyz_lr_c': 0.05, 'xyz_lr_final_c': 0.9,
        'radius_lr_c': 0.05, 'radius_lr_final_c': 0.9,
        'weights_lr_c': 0.10, 'weights_lr_final_c': 0.7,
        'xyz_reg': 1.0, 'radius_reg': 0.3,
        'radius_cutoff_inv': 0.5, 'lr_max_steps': train_iters,
    }
    reconstruction_params = {
        'targetd_num_mode': 10,
        'voxel_scale': 0.5, 'voxel_peak_threshold': 0.3,
        'voxel_grid_max_size': 32, 'voxel_peaks_number': 2 * 10,
    }

    cam_system = _build_angled_cam_system(center, D, baseline_deg, config)

    # ---- Run each frame + collect diagnostics ----
    frame_data = []

    for frame_idx, ts in enumerate(time_steps):
        positions = dataset.positions_at_time_step(ts)
        scale_idx = (ts - start_step) // step_length
        gt_scale = gt_scales_all[scale_idx]

        # Pre-compute GT density
        grid = _build_grid(positions, gt_scale)
        gt_density = _precompute_gt_density(positions, gt_scale, grid)

        # ---- Run with checkpointing ----
        with tempfile.TemporaryDirectory() as tmpdir:
            dr = DensityReconstructor(
                max_iter=train_params['lr_max_steps'],
                use_decoupled=USE_DECOUPLED)
            _, projections, _, _ = cam_system.simulate_vision(
                positions, renderer='projection_only')
            dr.process_frame(
                cam_system, point_sets=projections, positions=positions,
                initGMM=None,
                is_adaptive_scale=False, scale=gt_scale,
                is_store_intermediate=True, is_log=False,
                output_dir=tmpdir, debug=False,
                train_params=train_params,
                reconstruction_params=reconstruction_params)

            history_path = os.path.join(tmpdir, "checkpoint_level_0.pth")
            training_history = GaussianModel.load_training_history(history_path)

            # Evaluate per-iteration metrics (every iteration, not sampled)
            iters = np.arange(train_iters)
            recalls = np.full(train_iters, np.nan)
            halls   = np.full(train_iters, np.nan)
            dmotas  = np.full(train_iters, np.nan)
            gmm_counts = np.full(train_iters, np.nan)

            N = positions.shape[0]
            for i in range(train_iters):
                ckpt = training_history[i + 1]
                tp, fp, fn = _compute_metrics_cached(
                    ckpt['_xyz'], ckpt['_weights'], ckpt['_radius'],
                    gt_density, grid)
                w_sum = ckpt['_weights'].sum().item()
                recalls[i] = tp / N if N > 0 else 0.0
                halls[i]   = fp / w_sum if w_sum > 0 else 0.0
                dmotas[i]  = 1.0 - (fn + fp) / N if N > 0 else 0.0
                gmm_counts[i] = ckpt['_xyz'].shape[0]

        # ---- Frame diagnostics ----
        extent = np.max(np.max(positions, axis=0) - np.min(positions, axis=0))
        # Triangulation quality: rough estimate from projections
        _, projections, _, masks = cam_system.simulate_vision(
            positions, renderer='projection_only')
        vis_frac = np.mean(np.array([m.mean() for m in masks]))

        # Convergence speed: iterations to reach 95% of final dMOTA
        final_dm = dmotas[-1]
        target = 0.95 * final_dm
        converged_at = np.argmax(dmotas >= target) if np.any(dmotas >= target) else train_iters

        # Initial dMOTA (after iter 0)
        init_dm = dmotas[0]
        # Improvement during training
        delta_dm = final_dm - init_dm

        frame_data.append({
            'ts': ts,
            'N': N,
            'gt_scale': gt_scale,
            'extent': extent,
            'vis_frac': vis_frac,
            'converged_at': converged_at,
            'init_dMOTA': init_dm,
            'final_dMOTA': final_dm,
            'delta_dMOTA': delta_dm,
            'iters': iters,
            'recalls': recalls,
            'halls': halls,
            'dmotas': dmotas,
            'gmm_counts': gmm_counts,
        })

        print(f"  frame {ts}: N={N:4d}  scale={gt_scale:.3f}  extent={extent:.1f}  "
              f"vis_frac={vis_frac:.3f}  converged_at={converged_at:4d}  "
              f"init_dMOTA={init_dm:.4f}  final_dMOTA={final_dm:.4f}  "
              f"Δ={delta_dm:+.4f}")

    # ---- Plot: per-frame convergence curves ----
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_steps)))

    for fi, fd in enumerate(frame_data):
        lbl = (f"t={fd['ts']} (N={fd['N']}, conv@{fd['converged_at']}, "
               f"ΔdMOTA={fd['delta_dMOTA']:+.3f})")

        axes[0, 0].plot(fd['iters'], fd['recalls'], color=colors[fi], linewidth=2, label=lbl)
        axes[0, 1].plot(fd['iters'], fd['halls'],   color=colors[fi], linewidth=2, label=lbl)
        axes[0, 2].plot(fd['iters'], fd['dmotas'],  color=colors[fi], linewidth=2, label=lbl)

        # Bottom row: zoomed first 100 iters
        n_zoom = 100
        axes[1, 0].plot(fd['iters'][:n_zoom], fd['recalls'][:n_zoom], color=colors[fi], linewidth=2, label=lbl)
        axes[1, 1].plot(fd['iters'][:n_zoom], fd['halls'][:n_zoom],   color=colors[fi], linewidth=2, label=lbl)
        axes[1, 2].plot(fd['iters'][:n_zoom], fd['dmotas'][:n_zoom],  color=colors[fi], linewidth=2, label=lbl)

        # GMM count
        axes[0, 0].plot(fd['iters'], fd['gmm_counts'] / np.max(fd['gmm_counts']),
                        '--', color=colors[fi], alpha=0.4, linewidth=1)

    for row, title_prefix in [(0, 'Full training'), (1, 'First 100 iterations')]:
        for col, ylabel, title in [
            (0, 'Recall (Coverage)', 'Recall'),
            (1, 'Hallucination (FP Rate)', 'Hallucination'),
            (2, 'dMOTA', 'dMOTA'),
        ]:
            ax = axes[row, col]
            ax.set_xlabel('Training iteration')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{title_prefix}: {title}')
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.legend(fontsize=7, loc='lower right')

    fig.suptitle(
        f'Per-Frame Convergence Diagnosis  |  {dataset_name}  baseline={baseline_deg}°  Default init\n'
        f'Dashed line = normalized GMM component count (right axis on Recall plot)',
        fontsize=13, fontweight='bold')
    plt.tight_layout()

    sweep_dir = os.path.join(scenario_path, "logs", "baseline_sweep")
    fig.savefig(os.path.join(sweep_dir, "convergence_diagnosis.png"),
                dpi=150, bbox_inches='tight')
    print(f"\nSaved to {sweep_dir}/convergence_diagnosis.png")

    return frame_data

def run_baseline_angle_sweep():
    """
    Investigate the effect of inter-camera baseline angle on reconstruction
    metrics, and whether poor performance at small angles is *inherent*
    (insufficient parallax) or due to bad *initialisation*.

    Compares two initialisations at each angle:
      - "Default":  standard farthest-point-sampled voxel initialisation
      - "GMR init":  Gaussian Mixture Reduction on the GT 3D positions,
                     then ISE-refined, with the same component count as the
                     default run converged to.

    Two cameras at constant radius D; cam1 fixed at 0°, cam2 at `baseline_angle`
    on the same XY circle.  Both auto-aim at the swarm centre.
    """
    # =========================================================================
    # Configuration
    # =========================================================================
    dataset_name = 'jackdaw'
    start_step = 350
    end_step   = 360
    step_length = 10           # → 20 timesteps

    angle_min  = 10            # degrees
    angle_max  = 90           # degrees
    angle_step = 10            # degrees

    # =========================================================================
    # 1. Load dataset & compute shared camera radius D
    # =========================================================================
    scenario_path = os.path.join(os.getcwd(), "scenarios", dataset_name)
    config_path   = os.path.join(scenario_path, "config.yaml")
    config = SimulationConfig(config_path)
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    max_steps = dataset.trajectories.shape[0]
    effective_end_step = (end_step if end_step is not None and end_step <= max_steps
                          else max_steps)
    step_list = list(range(start_step, effective_end_step, step_length))
    logger.info(f"Angle sweep: {len(step_list)} timesteps for {dataset_name} "
                f"(steps {start_step}–{effective_end_step}, stride {step_length})")

    # Aggregate all points → global bounding sphere
    all_positions = []
    for t in step_list:
        all_positions.append(dataset.positions_at_time_step(t))
    all_positions = np.vstack(all_positions)

    min_bounds = all_positions.min(axis=0)
    max_bounds = all_positions.max(axis=0)
    center = (min_bounds + max_bounds) / 2.0
    max_radius = np.max(np.linalg.norm(all_positions - center, axis=1))
    safe_radius = max_radius * 1.0       # padding = 1

    fx = config.intrinsics_params[0, 0]
    fy = config.intrinsics_params[1, 1]
    cx = config.intrinsics_params[0, 2]
    cy = config.intrinsics_params[1, 2]
    theta_x_left   = np.arctan2(cx, fx)
    theta_x_right  = np.arctan2(config.W - cx, fx)
    theta_y_top    = np.arctan2(cy, fy)
    theta_y_bottom = np.arctan2(config.H - cy, fy)
    min_half_fov = min(theta_x_left, theta_x_right, theta_y_top, theta_y_bottom)
    D = safe_radius / np.sin(min_half_fov)

    logger.info(f"Scene centre: {center}, bounding radius: {max_radius:.2f}, "
                f"camera distance D: {D:.2f}")

    # =========================================================================
    # 2. Load ground-truth scales
    # =========================================================================
    gt_data = np.load(os.path.join(scenario_path, 'reconstruction_scale.npz'))
    gt_scales = gt_data['scales_gt']
    if len(gt_scales) < len(step_list):
        logger.warning(f"gt_scales has {len(gt_scales)} entries but step_list "
                       f"has {len(step_list)} — truncating step_list")
        step_list = step_list[:len(gt_scales)]

    # =========================================================================
    # 3. Shared reconstruction objects
    # =========================================================================
    train_params = {
        'xyz_lr_c': 0.05, 'xyz_lr_final_c': 0.9,
        'radius_lr_c': 0.05, 'radius_lr_final_c': 0.9,
        'weights_lr_c': 0.10, 'weights_lr_final_c': 0.7,
        'xyz_reg': 1.0, 'radius_reg': 0.3,
        'radius_cutoff_inv': 0.5, 'lr_max_steps': 100,
    }
    reconstruction_params = {
        'targetd_num_mode': 10,
        'voxel_scale': 0.5, 'voxel_peak_threshold': 0.3,
        'voxel_grid_max_size': 32, 'voxel_peaks_number': 2 * 10,
    }
    density_reconstructor = DensityReconstructor(
        max_iter=train_params['lr_max_steps'],
        use_decoupled=USE_DECOUPLED,
    )

    # =========================================================================
    # 4. Sweep baseline angles — BOTH initialisations per angle
    # =========================================================================
    angles = np.arange(angle_min, angle_max + 1, angle_step)

    res_def = {'angle': [], 'recall': [], 'hallucination': [], 'dMOTA': []}
    res_gmr = {'angle': [], 'recall': [], 'hallucination': [], 'dMOTA': []}

    sweep_dir = os.path.join(scenario_path, "logs", "baseline_sweep")
    os.makedirs(sweep_dir, exist_ok=True)
    cache_def = os.path.join(sweep_dir, "angle_sweep_default.npz")
    cache_gmr = os.path.join(sweep_dir, "angle_sweep_gmr.npz")

    # Pre-compute GT density grids ONCE per frame (shared across all angles)
    gt_cache = _build_gt_cache_for_frames(step_list, dataset, gt_scales)

    for baseline_deg in tqdm(angles, desc="Sweeping baseline angles"):
        cam_system = _build_angled_cam_system(center, D, baseline_deg, config)

        # ---- 4a. DEFAULT initialisation (standard pipeline) ----
        tp_d, fp_d, fn_d, N_d, w_d = [], [], [], [], []
        gmm_nums = []           # record per-frame component count for GMR init

        for idx, time_step in enumerate(step_list):
            positions = dataset.positions_at_time_step(time_step)
            _, projections, _, _ = cam_system.simulate_vision(
                positions, renderer='projection_only',
            )

            model, _ = density_reconstructor.process_frame(
                cam_system, point_sets=projections, positions=positions,
                initGMM=None,
                is_adaptive_scale=False, scale=gt_scales[idx],
                is_store_intermediate=False, is_log=False,
                output_dir=None, debug=False,
                train_params=train_params,
                reconstruction_params=reconstruction_params,
            )

            n_comp = model[0]._xyz.shape[0]
            gmm_nums.append(n_comp)

            tp, fp, fn = _compute_frame_metrics(
                positions, gt_scales[idx],
                model[0]._xyz, model[0]._weights, model[0]._radius,
                gt_cache=gt_cache[idx],
            )
            tp_d.append(tp); fp_d.append(fp); fn_d.append(fn)
            N_d.append(positions.shape[0])
            w_d.append(model[0]._weights.sum().item())

        # ---- 4b. GMR initialisation (from GT positions) ----
        tp_g, fp_g, fn_g, N_g, w_g = [], [], [], [], []

        for idx, time_step in enumerate(step_list):
            positions = dataset.positions_at_time_step(time_step)
            _, projections, _, _ = cam_system.simulate_vision(
                positions, renderer='projection_only',
            )

            # Build GMR-init model with same component count as default
            target_L = max(int(gmm_nums[idx]), 2)
            init_gm = _build_gmr_init_gmm(positions, gt_scales[idx], target_L)

            model, _ = density_reconstructor.process_frame(
                cam_system, point_sets=projections, positions=positions,
                initGMM=[init_gm],
                is_adaptive_scale=False, scale=gt_scales[idx],
                is_store_intermediate=False, is_log=False,
                output_dir=None, debug=False,
                train_params=train_params,
                reconstruction_params=reconstruction_params,
            )

            tp, fp, fn = _compute_frame_metrics(
                positions, gt_scales[idx],
                model[0]._xyz, model[0]._weights, model[0]._radius,
                gt_cache=gt_cache[idx],
            )
            tp_g.append(tp); fp_g.append(fp); fn_g.append(fn)
            N_g.append(positions.shape[0])
            w_g.append(model[0]._weights.sum().item())

        # ---- 4c. Aggregate → global metrics for both inits ----
        rec_d, hal_d, dm_d = _aggregate_global_metrics(tp_d, fp_d, fn_d, N_d, w_d)
        rec_g, hal_g, dm_g = _aggregate_global_metrics(tp_g, fp_g, fn_g, N_g, w_g)

        res_def['angle'].append(baseline_deg)
        res_def['recall'].append(rec_d)
        res_def['hallucination'].append(hal_d)
        res_def['dMOTA'].append(dm_d)

        res_gmr['angle'].append(baseline_deg)
        res_gmr['recall'].append(rec_g)
        res_gmr['hallucination'].append(hal_g)
        res_gmr['dMOTA'].append(dm_g)

        logger.info(
            f"  angle={baseline_deg:4.0f}°  "
            f"Default: R={rec_d:.4f} H={hal_d:.4f} dMOTA={dm_d:.4f}  |  "
            f"GMR: R={rec_g:.4f} H={hal_g:.4f} dMOTA={dm_g:.4f}"
        )

        # Incremental saves (resumable)
        np.savez(cache_def, **{k: np.array(v) for k, v in res_def.items()})
        np.savez(cache_gmr, **{k: np.array(v) for k, v in res_gmr.items()})

    # =========================================================================
    # 5. Plot — default vs GMR on shared axes
    # =========================================================================
    ang   = np.array(res_def['angle'])
    rec_d = np.array(res_def['recall'])
    hal_d = np.array(res_def['hallucination'])
    dm_d  = np.array(res_def['dMOTA'])

    rec_g = np.array(res_gmr['recall'])
    hal_g = np.array(res_gmr['hallucination'])
    dm_g  = np.array(res_gmr['dMOTA'])

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    def _plot_one(ax, x, y_def, y_gmr, ylabel, title):
        ax.plot(x, y_def, 'o-', color='#d62728', linewidth=2, markersize=8,
                markerfacecolor='white', markeredgewidth=2, label='Default init')
        ax.plot(x, y_gmr, 's--', color='#2ca02c', linewidth=2, markersize=8,
                markerfacecolor='white', markeredgewidth=2, label='GMR init (GT pos.)')
        ax.set_xlabel('Baseline Angle (degrees)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=90, color='gray', linestyle=':', alpha=0.4,
                   label='Orthogonal (90°)')
        ax.legend(fontsize=9)
        all_y = np.concatenate([y_def, y_gmr])
        y_min, y_max = np.min(all_y), np.max(all_y)
        y_pad = max(0.02, (y_max - y_min) * 0.12)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    _plot_one(axes[0], ang, rec_d, rec_g, 'Recall (Coverage)', 'Recall')
    _plot_one(axes[1], ang, hal_d, hal_g, 'Hallucination (FP Rate)', 'Hallucination')
    _plot_one(axes[2], ang, dm_d,  dm_g,  'dMOTA', 'dMOTA')

    fig.suptitle(
        f'Effect of Baseline Angle: Default Init vs. GMR Init (from GT positions)\n'
        f'Dataset: {dataset_name}  |  {len(step_list)} timesteps  |  '
        f'2 cameras, radius = {D:.1f}',
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout()

    fig_path = os.path.join(sweep_dir, "angle_vs_metrics_default_vs_gmr.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    logger.info(f"Figure saved to: {fig_path}")

    # LaTeX-friendly table
    print("\nangle\tRec(Def)\tRec(GMR)\tHall(Def)\tHall(GMR)\tdMOTA(Def)\tdMOTA(GMR)")
    print("-" * 85)
    for a, rd, rg, hd, hg, dd, dg in zip(ang, rec_d, rec_g, hal_d, hal_g, dm_d, dm_g):
        print(f"{a:.0f}\t{rd:.4f}\t{rg:.4f}\t{hd:.4f}\t{hg:.4f}\t{dd:.4f}\t{dg:.4f}")

    return {'default': res_def, 'gmr': res_gmr}


def plot_time_multi_scenarios():
    for run_params in DATASET_VIS:
        plot_time_single_scenarios(run_params)

def plot_time_single_scenarios(run_params):
    name = run_params['name']
    log_name = run_params['log_name']
    scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
    log_file_path = os.path.join(scenario_path, *["logs", log_name])

    log_data = np.load(os.path.join(log_file_path, "statistics.npz"))

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    time_stamp = np.arange(log_data['estimate_swarm_center'].shape[0])
    ax.plot(time_stamp, log_data['estimate_swarm_center'], label='estimate_swarm_center')
    ax.plot(time_stamp, log_data['adaptive_scale_selection'], label='adaptive_scale_selection')
    ax.plot(time_stamp, log_data['generate_scale_space'], label='generate_scale_space')
    ax.plot(time_stamp, log_data['estimate_scale_space_peaks'], label='estimate_scale_space_peaks')
    ax.plot(time_stamp, log_data['setup_gaussian_scale_space'], label='setup_gaussian_scale_space')
    ax.plot(time_stamp, log_data['train_gaussian_scale_space'], label='train_gaussian_scale_space')

    plt.title(f'time for scenario {name}')
    plt.xlabel('time step')
    plt.yscale('log')
    plt.legend()

def print_global_metrics(label, metric_data):
    # 1. Sum up the raw counts/masses across all frames
    sum_tp = np.sum(metric_data['tp'])
    sum_fp = np.sum(metric_data['fp'])
    sum_fn = np.sum(metric_data['fn'])
    total_N = np.sum(metric_data['N'])
    total_weights = np.sum(metric_data['w'])

    # 3. Compute final global metrics
    global_recall = sum_tp / total_N if total_N > 0 else 0.0
    global_miss = sum_fn / total_N if total_N > 0 else 0.0
    global_hallucination = sum_fp / total_weights if total_weights > 0 else 0.0
    global_dmota = 1.0 - ((sum_fn + sum_fp) / total_N) if total_N > 0 else 0.0
    global_weight_err = np.mean(np.abs(metric_data['w'] - metric_data['N']) / metric_data['N'])

    # 4. Print results (comparing global vs. simple mean for reference)
    error_str = f"{global_recall:.3f} & {global_hallucination:.3f} & {global_dmota:.3f} &"
    # error_str = f"{global_recall:.3f} & {global_hallucination:.3f} & {global_dmota:.3f} & {metric_data['train_time']:.0f} & {global_weight_err*100:.1f} &"
    # error_str = f"{global_dmota:.3f} & {metric_data['train_time']:.0f} &"
    # error_str = f"{global_recall:.3f} & {global_dmota:.3f} &"
    return error_str

def compute_metrics_multi_scenarios():
    gt_error = []
    estim_error = []
    for run_params in DATASET_RUNS:
        compute_metrics_single_scenario(run_params, gt_error, estim_error)

    print('--------- GT ----------')
    print(*gt_error)
    # for i in range(4):
    #     if i == 3:
    #         print(gt_error[i][:-1] + r'\\')
    #     else:
    #         print(gt_error[i])
    # print('--------- ESTIM ----------')
    # print(*estim_error)
    # for i in range(4):
    #     if i == 3:
    #         print(estim_error[i][:-1] + r'\\')
    #     else:
    #         print(estim_error[i])

def compute_metrics_single_scenario(run_params, gt_error, estim_error):
    force_update = False

    name = run_params['name']
    log_name = run_params['log_name']
    start_step = run_params['start_step']
    end_step = run_params['end_step']
    step_length = run_params['step_length']

    scenario_path = os.path.join(os.getcwd(), *["scenarios", name])

    log_file_path = os.path.join(scenario_path, *["logs", log_name])

    config_path = os.path.join(scenario_path, "config.yaml")
    config = SimulationConfig(config_path)
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    max_steps = dataset.trajectories.shape[0]
    effective_end_step = end_step if end_step is not None and end_step <= max_steps else max_steps

    step_range = range(start_step, effective_end_step, step_length)

    gt_data = np.load(scenario_path + '/reconstruction_scale.npz')
    scale_history = gt_data['scales_gt']

    metrics = {
        'tp': [],
        'fp': [],
        'fn': [],
        'N': [],
        'w': [],
        'coverage_recall': [],
        'miss': [],
        'hallucination': [],
        'dMOTA': [],
        'train_time': 0
    }

    metrics_estim = {
        'tp': [],
        'fp': [],
        'fn': [],
        'N': [],
        'w': [],
        'coverage_recall': [],
        'miss': [],
        'hallucination': [],
        'dMOTA': [],
        'train_time': np.mean(np.load(os.path.join(log_file_path, f"statistics.npz"))['train_gaussian_scale_space']).item()
    }

    if force_update or not os.path.exists(os.path.join(scenario_path, f"metrics_estim_{log_name}.npz")):
        for idx, time_step in enumerate(tqdm(step_range, desc=f"Processing {name}")):
            time_step_path = os.path.join(log_file_path, f"t_{time_step:03d}")

            training_history = GaussianModel.load_training_history(os.path.join(time_step_path, f"checkpoint_level_0.pth"))
            model_estim = GaussianModel.load_iter(training_history, iter=99)

            model_gt = torch.load(os.path.join(time_step_path, f"baseline_level_0.pth"))

            positions = dataset.positions_at_time_step(time_step)
            N = positions.shape[0]

            min_coords = np.min(positions, axis=0)
            max_coords = np.max(positions, axis=0)

            bounds = np.vstack((min_coords - 3 * scale_history[idx], max_coords + 3 * scale_history[idx])).T # add three sigma padding
            voxel_res = np.max(max_coords - min_coords) * 5e-3
            voxel_num = np.prod((max_coords - min_coords) / voxel_res)
            total_tp_mass, total_fp_mass, total_fn_mass = \
            compute_metrics_batched_torch(means1_np=positions, sigma1=scale_history[idx],
                                        pred_means=model_estim._xyz, pred_weights=model_estim._weights, pred_sigmas=model_estim._radius,
                                        bounds=bounds, voxel_res=voxel_res, batch_size=50000, device='cuda')
            metrics_estim['tp'].append(total_tp_mass)
            metrics_estim['fp'].append(total_fp_mass)
            metrics_estim['fn'].append(total_fn_mass)
            metrics_estim['N'].append(N)
            metrics_estim['w'].append(model_estim._weights.sum().item())

            metrics_estim['coverage_recall'].append(total_tp_mass / N)
            metrics_estim['miss'].append(total_fn_mass / N)
            metrics_estim['hallucination'].append(total_fp_mass / model_estim._weights.sum().item())
            metrics_estim['dMOTA'].append(1 - (total_fn_mass + total_fp_mass) / N)

            total_tp_mass, total_fp_mass, total_fn_mass = \
                    compute_metrics_batched_torch(means1_np=positions, sigma1=scale_history[idx],
                                        pred_means=model_gt['_xyz'], pred_weights=model_gt['_weights'], pred_sigmas=model_gt['_radius'],
                                        bounds=bounds, voxel_res=voxel_res, batch_size=50000, device='cuda')
            metrics['tp'].append(total_tp_mass)
            metrics['fp'].append(total_fp_mass)
            metrics['fn'].append(total_fn_mass)
            metrics['N'].append(N)
            metrics['w'].append(model_gt['_weights'].sum().item())

            metrics['coverage_recall'].append(total_tp_mass / N)
            metrics['miss'].append(total_fn_mass / N)
            metrics['hallucination'].append(total_fp_mass / model_gt['_weights'].sum().item())
            metrics['dMOTA'].append(1 - (total_fn_mass + total_fp_mass) / N)

        metrics_estim = {k: np.array(v) for k, v in metrics_estim.items()}
        metrics = {k: np.array(v) for k, v in metrics.items()}
        np.savez(os.path.join(scenario_path, f"metrics_estim_{log_name}.npz"), **metrics_estim)
        np.savez(os.path.join(scenario_path, f"metrics_{log_name}.npz"), **metrics)
    else:
        metrics = np.load(os.path.join(scenario_path, f"metrics_{log_name}.npz"))
        metrics_estim = np.load(os.path.join(scenario_path, f"metrics_estim_{log_name}.npz"))

        metrics_estim = {k: np.array(v) for k, v in metrics_estim.items()}
        metrics = {k: np.array(v) for k, v in metrics.items()}

        metrics['train_time'] = 0
        metrics_estim['train_time'] = np.mean(np.load(os.path.join(log_file_path, f"statistics.npz"))['train_gaussian_scale_space']).item()

    gt_error.append(print_global_metrics('gt', metrics))
    estim_error.append(print_global_metrics('estim', metrics_estim))

def calculate_projection_median_nn_distance_multi_scenarios():
    results = {}
    for run_params in DATASET_RUNS:
        mean_nn, median_nn = calculate_projection_median_nn_distance_single_scenario(run_params)
        if mean_nn is not None:
            results[run_params['name']] = {'mean': mean_nn, 'median': median_nn}

    # Optionally print overall statistics across all scenarios
    return results

def calculate_projection_median_nn_distance_single_scenario(run_params):
    # 1. Parameter extraction and Logging Setup
    name = run_params['name']
    log_name = run_params['log_name']
    start_step = run_params['start_step']
    end_step = run_params['end_step']
    step_length = run_params['step_length']

    logger.info(f"Running scenario {name}")

    scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
    config_path = os.path.join(scenario_path, "config.yaml")
    log_file_path = os.path.join(scenario_path, *["logs", log_name])
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)

    # 3. Load Dataset
    config = SimulationConfig(config_path)
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    max_steps = dataset.trajectories.shape[0]
    effective_end_step = end_step if end_step is not None and end_step <= max_steps else max_steps

    if start_step >= effective_end_step:
        logger.info(f"Skipping {name}: start_step ({start_step}) >= end_step ({effective_end_step}).")
        return None, None

    step_range = range(start_step, effective_end_step, step_length)

    # Camera Configurations
    if CAM_NUM == 2:
        cam_positions, cam_radius = generate_encircling_cameras(dataset, step_range, config.intrinsics_params, config.H, config.W, cam_num=4, padding=1)
        cam_poses = np.hstack((cam_positions[:2], np.tile(np.array([1, 0, 0, 0]), (2, 1)))).astype(np.float32)
    else:
        cam_positions, cam_radius = generate_encircling_cameras(dataset, step_range, config.intrinsics_params, config.H, config.W, cam_num=CAM_NUM, padding=1)
        cam_poses = np.hstack((cam_positions, np.tile(np.array([1, 0, 0, 0]), (CAM_NUM, 1)))).astype(np.float32)

    # 4. System Initialization
    cam_system = MultiCameraSystem.create_homogeneous_system(
        state_class=CameraState,
        intrinsics=config.intrinsics_params,
        H=config.H, W=config.W,
        poses_or_RTs=cam_poses,
        near_clip=config.near_clip, far_clip=config.far_clip,
        size=config.size,
        device='cuda')

    # 5. Simulation Loop
    nn_dist = []
    for idx, time_step in enumerate(tqdm(step_range, desc=f"Processing {name}")):
        positions = dataset.positions_at_time_step(time_step)

        # simulate_vision returns projections and masks across all cameras
        poses, projections, _, masks = cam_system.simulate_vision(positions, renderer='projection_only')

        # Ensure projections and masks are numpy arrays for scipy
        if torch.is_tensor(projections):
            projections = projections.cpu().numpy()
        if torch.is_tensor(masks):
            masks = masks.cpu().numpy()

        # Iterate over each camera's projections
        for cam_idx in range(len(projections)):
            # Filter projections using the visibility mask
            valid_projections = projections[cam_idx]

            # Need at least 2 points to calculate a nearest neighbor distance
            if len(valid_projections) >= 2:
                # Build a KDTree for fast spatial queries
                tree = cKDTree(valid_projections)

                # Query the 2 nearest neighbors (k=2).
                # The 1st neighbor is the point itself (distance 0),
                # the 2nd neighbor is the actual nearest neighbor.
                distances, _ = tree.query(valid_projections, k=2)

                # Extract the distance to the nearest neighbor (index 1)
                nearest_distances = distances[:, 1]
                nn_dist.extend(nearest_distances.tolist())

    # 6. Calculate Final Metrics
    if len(nn_dist) > 0:
        mean_nn_dist = np.mean(nn_dist)
        median_nn_dist = np.median(nn_dist)
        logger.info(f"Scenario: {name} | Mean NN Dist: {mean_nn_dist:.4f} px | Median NN Dist: {median_nn_dist:.4f} px")

        # Optional: Save to file alongside other metrics
        np.savez(os.path.join(scenario_path, f"nn_distances_{log_name}.npz"),
                 distances=np.array(nn_dist),
                 mean=mean_nn_dist,
                 median=median_nn_dist)

        return mean_nn_dist, median_nn_dist
    else:
        logger.warning(f"No valid projections found for scenario {name} to calculate NN distance.")
        return None, None

def create_parser():
    parser = argparse.ArgumentParser(
        description="Explicit entry points for camera-angle reconstruction studies."
    )
    parser.add_argument(
        "study",
        choices=(
            "reconstruct",
            "profile",
            "voxel-coarsening",
            "training-convergence",
            "diagnose-convergence",
            "baseline-angle-sweep",
            "projection-nn",
            "metrics",
        ),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", choices=tuple(item['name'] for item in DATASET_RUNS))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--no-display", action="store_true")
    return parser


def main(argv=None):
    args = create_parser().parse_args(argv)
    if args.study == "reconstruct":
        selected = [
            item for item in DATASET_RUNS
            if args.dataset is None or item['name'] == args.dataset
        ]
        for params in selected:
            run_single_scenario(
                params, project_root=args.project_root, seed=args.seed
            )
    elif args.study == "profile":
        profile_bottleneck()
    elif args.study == "voxel-coarsening":
        test_voxel_coarsening()
    elif args.study == "training-convergence":
        run_training_convergence()
    elif args.study == "diagnose-convergence":
        diagnose_slow_convergence()
    elif args.study == "baseline-angle-sweep":
        run_baseline_angle_sweep()
    elif args.study == "projection-nn":
        calculate_projection_median_nn_distance_multi_scenarios()
    else:
        compute_metrics_multi_scenarios()
    if not args.no_display:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# for cam_num in [5]:
# for cam_num in [2, 3, 5]:
#     CAM_NUM = cam_num
#     LOG_NAME = 'base_reg_cam_' + str(CAM_NUM)

#     DATASET_RUNS = [
#         {
#             'name': 'swift',
#             'log_name': LOG_NAME,
#             'start_step': 0,
#             'end_step': None,
#             'step_length': 200,
#         },
#         {
#             'name': 'starling',
#             'log_name': LOG_NAME,
#             'start_step': 0,
#             'end_step': None,
#             'step_length': 1,
#         },
#         {
#             'name': 'jackdaw',
#             'log_name': LOG_NAME,
#             'start_step': 350,
#             'end_step': 550,
#             'step_length': 10,
#         },
#         {
#             'name': 'jackdaw2',
#             'log_name': LOG_NAME,
#             'start_step': 2700,
#             'end_step': 3460,
#             'step_length': 20,
#         },
#     ]

#     run_multi_scenarios()
    # run_multi_scenarios_baseline()

    # compute_metrics_multi_scenarios()

# NOISE_LEVELS = [20.0]
# np.random.seed(123456789)

# for noise_std in NOISE_LEVELS:
#     for cam_num in [2, 3, 5]:
#     # for cam_num in [5]:
#         CAM_NUM = cam_num
#         # Append both camera number and noise level to the log path
#         LOG_NAME = f'base_reg_cam_{CAM_NUM}_noise_{noise_std}'

#         DATASET_RUNS = [
#             {
#                 'name': 'swift',
#                 'log_name': LOG_NAME,
#                 'start_step': 0,
#                 'end_step': None,
#                 'step_length': 200,
#                 'noise_std': noise_std, # Pass the parameter
#             },
#             {
#                 'name': 'starling',
#                 'log_name': LOG_NAME,
#                 'start_step': 0,
#                 'end_step': None,
#                 'step_length': 1,
#                 'noise_std': noise_std,
#             },
#             {
#                 'name': 'jackdaw',
#                 'log_name': LOG_NAME,
#                 'start_step': 350,
#                 'end_step': 550,
#                 'step_length': 10,
#                 'noise_std': noise_std,
#             },
#             {
#                 'name': 'jackdaw2',
#                 'log_name': LOG_NAME,
#                 'start_step': 2700,
#                 'end_step': 3460,
#                 'step_length': 20,
#                 'noise_std': noise_std,
#             },
#         ]

#         # run_multi_scenarios()

#         # run_multi_scenarios_baseline()
#         compute_metrics_multi_scenarios()
