import logging
import sys
import os
from tqdm import tqdm
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
from dfr import CameraConfig, OutputConfig, load_dataset, reconstruct
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

def run_single_scenario(run_params, *, output=None, seed=12345):
    """Run one scenario through the typed package reconstruction workflow."""
    if CLEAN_LOGS:
        raise ValueError(
            "CLEAN_LOGS is unsupported for managed runs; use OutputConfig(overwrite=True)."
        )
    name = run_params['name']
    start_step = int(run_params['start_step'])
    step_length = int(run_params['step_length'])
    project_root = Path.cwd().resolve()
    dataset = load_dataset(name, project_root=project_root)
    stop = run_params['end_step']
    stop = min(int(stop), len(dataset)) if stop is not None else len(dataset)
    frames = tuple(range(start_step, stop, step_length))
    if not frames:
        logger.info(f"Skipping {name}: selected frame range is empty.")
        return None

    frame_scales = None
    if USE_GT_SCALE:
        scale_path = project_root / "scenarios" / name / "reconstruction_scale.npz"
        with np.load(scale_path, allow_pickle=False) as scale_data:
            available_scales = np.asarray(scale_data['scales_gt'], dtype=float)
        if len(available_scales) < len(frames):
            raise ValueError(
                f"Ground-truth scale cache has {len(available_scales)} values "
                f"for {len(frames)} selected frames."
            )
        frame_scales = tuple(available_scales[:len(frames)])

    if output is None and IS_LOGGING:
        output = OutputConfig(
            workflow="reconstruction",
            name=f"{name} {run_params['log_name']}",
            run_id=f"{name}-{run_params['log_name']}",
            project_root=project_root,
        )
    run = reconstruct(
        dataset,
        frames=frames,
        cameras=CameraConfig.encircling(count=CAM_NUM, device="cuda"),
        frame_scales=frame_scales,
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
            lr_max_steps=500,
        ),
        reconstruction=ReconstructionParams(
            targetd_num_mode=10,
            voxel_scale=0.5,
            voxel_peak_threshold=0.3,
            voxel_grid_max_size=32,
            voxel_peaks_number=20,
        ),
        seed=seed,
        projection_noise_std=float(run_params.get('noise_std', 0.0)),
        use_decoupled=USE_DECOUPLED,
        output=output,
    )
    logger.info(f"Results for {name}: {len(run.frames)} frames reconstructed.")
    if run.artifacts is not None:
        timing_names = sorted(
            {key for frame in run.frames for key in frame.time_ms}
        )
        run.artifacts.save_npz(
            "statistics.npz",
            overwrite=run.artifacts.output.resume,
            **{
                key: np.asarray([frame.time_ms.get(key, np.nan) for frame in run.frames])
                for key in timing_names
            },
            final_training_loss=np.asarray(
                [frame.mean_training_loss for frame in run.frames], dtype=float
            ),
            final_density_field_loss=np.asarray(
                [frame.density_dissimilarity for frame in run.frames], dtype=float
            ),
            final_gmm_num=np.asarray([frame.gaussian_count for frame in run.frames]),
            scale=np.asarray([frame.scale for frame in run.frames]),
        )
        logger.info(f"Managed outputs: {run.run_dir}")
    return run


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

if __name__ == "__main__":
    # calculate_projection_median_nn_distance_multi_scenarios()
    run_multi_scenarios()
    # run_multi_scenarios_baseline()

    # compute_metrics_multi_scenarios()
    # plot_time_multi_scenarios()

    plt.show()

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
