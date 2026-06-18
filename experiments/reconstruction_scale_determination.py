import logging
import sys
import os
import shutil
from tqdm import tqdm


import time
import torch
import numpy as np
from dfr.simulation_config import SimulationConfig
from dfr.dataset_io import DatasetFactory
from dfr.camera_system import MultiCameraSystem
from dfr.density_field_reconstructor import DensityReconstructor
from dfr.density_field_model import GaussianModel
from dfr.camera_state import CameraState
from dfr.utils import calculate_gmm_dissimilarity
from dfr.visualizer import MultiGMMPlotter
from dfr.gaussian_mixture_reduction import GMR
from dfr.mode_finding import mode_counting, mode_counting_modified, find_scale_interval, analytic_solution, model_4pl_scale_at_x_constant
from dfr.mode_finding import find_target_scale, analytic_solution_scale_at_x_constant
from gaussian_rasterizer_simple_large import rasterize_gaussians
from experiments.power_law import curve_fit, power_2pl, power_3pl
from dfr.utils import move_figure
from scipy.optimize import minimize_scalar

import matplotlib.pyplot as plt

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('run_experiments.log', mode='w')
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

IS_LOGGING = True

LOG_NAME = 'base_reg_cam_3_adaptive'

DATASET_RUNS = [
    {
        'name': 'swift',
        'log_name': LOG_NAME,
        'start_step': 0,
        'end_step': None,
        'step_length': 200,
    },
    # {
    #     'name': 'starling',
    #     'log_name': LOG_NAME,
    #     'start_step': 0,
    #     'end_step': None,
    #     'step_length': 1,
    # },
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

def calculate_volume_from_gmm_torch(weights, means, radius, kernel_variance):
    """
    Calculates the volume of the original point set from PyTorch GMM tensors.
    
    Args:
        weights: Tensor of shape [N, 1]
        means: Tensor of shape [N, 3]
        radius: Tensor of shape [N, 1] (Assuming this is the standard deviation)
        kernel_variance: float, the sigma^2 of the initial convolution kernel
        
    Returns:
        volume: A 0-dimensional tensor (scalar) representing the volume.
    """
    # Optional but recommended: Ensure weights sum to exactly 1
    weights = weights / weights.sum()
    
    # 1. Global Mean
    # weights [N, 1] broadcasts with means [N, 3] -> sum yields [3]
    global_mean = torch.sum(weights * means, dim=0) 
    
    # 2. Global Covariance
    # Center the means
    diff = means - global_mean # Shape: [N, 3]
    
    # Batch outer product of the differences: diff[i] @ diff[i]^T
    # diff.unsqueeze(2) is [N, 3, 1], diff.unsqueeze(1) is [N, 1, 3]
    outer_products = diff.unsqueeze(2) @ diff.unsqueeze(1) # Shape: [N, 3, 3]
    
    # Weighted sum of outer products (spread of the means)
    # weights.unsqueeze(2) is [N, 1, 1] to broadcast against [N, 3, 3]
    cov_means = torch.sum(weights.unsqueeze(2) * outer_products, dim=0) # Shape: [3, 3]
    
    # Pooled internal variance (since components are isotropic)
    internal_var = torch.sum(weights * (radius ** 2)) # Scalar
    
    # 3. Assemble covariance and deconvolve
    device = means.device
    dtype = means.dtype
    identity = torch.eye(3, device=device, dtype=dtype)
    
    global_cov = cov_means + internal_var * identity
    original_cov = global_cov - kernel_variance * identity
    
    # 4. Eigenvalues
    # Use eigvalsh instead of eigvals since covariance matrices are symmetric. 
    # It is faster and more numerically stable on the GPU.
    eigenvalues = torch.linalg.eigvalsh(original_cov)
    
    # Clamp below to 0 to prevent NaNs in torch.sqrt due to float precision loss
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    
    # 5. Volume Calculation
    radii = 2 * torch.sqrt(eigenvalues)
    volume = (4.0 / 3.0) * torch.pi * torch.prod(radii)
    
    return volume

def run_multi_scenarios_mode_counting():
    for run_params in DATASET_RUNS:
        run_single_scenario_mode_counting(run_params)

def compute_scaling_law(dataset, step_range, save_path):
    force_update = False
    mc_func = mode_counting
    mc_mod_func = mode_counting_modified

    paths = {
        'range': save_path + '/scale_range.npy',
        'modes': save_path + '/modes.npy',
    }

    # 2. Setup Parameters
    num_test_scale = 40

    # Process Scales: Reuse range from the first trial for all others to save time
    if force_update or not os.path.exists(paths["range"]):
        # Only compute for the first trial (or average them if needed, but here we reuse)
        scale_range = []
        for time_step in tqdm(step_range, desc=f"Processing Scale Range"):
            pos_gpu = torch.from_numpy(dataset.positions_at_time_step(time_step)).cuda().float()
            nn_dist = torch.cdist(pos_gpu, pos_gpu) + torch.eye(pos_gpu.shape[0], device='cuda') * 1e10
            avg_nn_dist = torch.median(torch.min(nn_dist, dim=1).values).item()
            
            f = lambda s: mc_func(pos_gpu, pos_gpu.clone(), s, max_iter=500, tol=avg_nn_dist*1e-3)
            s_start, s_end = find_scale_interval(f, pos_gpu.shape[0], s_initial_guess=avg_nn_dist*5, atol=avg_nn_dist*1e-2)
            # Clamp: DBSCAN requires eps > 0, and s_start=0 means binary search failed
            s_start = max(s_start, avg_nn_dist * 1e-2)
            scale_range.append([s_start, s_end])
        scale_range = np.array(scale_range)
        np.save(paths["range"], scale_range)
    else:
        scale_range = np.load(paths["range"])

    # Compute Modes: Now iterating over trials
    if force_update or not os.path.exists(paths["modes"]):
        all_modes = np.zeros((len(step_range), num_test_scale))
        num_low_extensions = 0

        for i, time_step in enumerate(tqdm(step_range, desc=f"Computing #mode")):
            pos_gpu = torch.from_numpy(dataset.positions_at_time_step(time_step)).cuda().float()
            nn_dist = torch.cdist(pos_gpu, pos_gpu) + torch.eye(pos_gpu.shape[0], device='cuda') * 1e10
            avg_nn_dist = torch.median(torch.min(nn_dist, dim=1).values).item()

            s_start, s_end = scale_range[i]
            N = pos_gpu.shape[0]

            # --- Fix: extend s_start downward if plateau is truncated ---
            # Check if the first scale is too coarse: if modes[0] is far from N,
            # the scale range missed the plateau. Extend s_start until we capture it.
            modes_pos = None
            test_s_start = s_start
            test_scales = np.logspace(np.log10(test_s_start), np.log10(s_end), num_test_scale)

            # Quick probe at the current s_start (use small max_iter: near N = fast convergence)
            probe_s = float(test_s_start)
            curr_pos_first = pos_gpu.clone()
            probe_modes, _ = mc_mod_func(pos_gpu, curr_pos_first, probe_s, max_iter=200, tol=avg_nn_dist*1e-3)

            # If modes at s_start are already far below N, extend downward
            if probe_modes < 0.9 * N:
                # Binary-search style: halve s_start until modes come close to N
                for _ in range(10):
                    test_s_start /= 2.0
                    probe_modes, _ = mc_mod_func(pos_gpu, pos_gpu.clone(), test_s_start,
                                                  max_iter=200, tol=avg_nn_dist*1e-3)
                    if probe_modes >= 0.95 * N:
                        break
                # Update the stored scale_range
                scale_range[i, 0] = max(test_s_start, avg_nn_dist * 1e-2)
                num_low_extensions += 1

            # Recompute full logspace with (possibly extended) s_start
            test_scales = np.logspace(np.log10(scale_range[i, 0]), np.log10(s_end), num_test_scale)

            modes_pos = None
            prev_mode_num = N
            for idx, s in enumerate(test_scales):
                # Optimization: skip at very small scales where no merging happens
                if s < avg_nn_dist * 0.5:
                    all_modes[i, idx] = N
                    continue

                # Optimization: if we already reached 1 mode, all subsequent scales are 1
                if prev_mode_num <= 1:
                    all_modes[i, idx] = 1
                    continue

                # Adaptive max_iter: fewer iters when near N (minimal movement)
                if prev_mode_num > 0.95 * N:
                    mi = 100
                elif prev_mode_num > 0.5 * N:
                    mi = 200
                else:
                    mi = 400

                curr_pos = modes_pos.clone() if modes_pos is not None else pos_gpu.clone()
                mode_num, tmp = mc_mod_func(pos_gpu, curr_pos, s, max_iter=mi, tol=avg_nn_dist*1e-3)
                modes_pos = torch.from_numpy(tmp).cuda().float()
                all_modes[i, idx] = mode_num
                prev_mode_num = mode_num

        if num_low_extensions > 0:
            print(f"  [FIX] Extended s_start downward for {num_low_extensions}/{len(step_range)} steps "
                  f"(scale range was truncated, missing the plateau)")
            np.save(paths["range"], scale_range)
        np.save(paths["modes"], all_modes)
    else:
        all_modes = np.load(paths["modes"])

    params = np.zeros((len(step_range), 3)) 
    for i, time_step in enumerate(step_range):
        s_start, s_end = scale_range[i]
        test_scales = np.logspace(np.log10(s_start), np.log10(s_end), num_test_scale)

        N = dataset.positions_at_time_step(time_step).shape[0]

        begin_idx = np.argmax(all_modes[i, :] <= 0.9 * N)
        x_data = test_scales[begin_idx:]
        y_data = all_modes[i, begin_idx:]

        # try:
        #     popt, _ = curve_fit(
        #         lambda x, k, x0: power_2pl(x, k, x0, A=N, D=1), 
        #         x_data, 
        #         y_data, 
        #         p0=(2, 1),
        #         # Using y_data as sigma maintains the Poisson-like weighting from your original code
        #         sigma=y_data, 
        #         absolute_sigma=True, 
        #         bounds=([0]*2, [np.inf]*2)
        #     )
        # except RuntimeError:
        #     print(f"Warning: Curve fit failed for pooled density index {i}")
        #     popt = [np.nan, np.nan]
            
        # params[i] = popt

        try:
            popt, _ = curve_fit(
                lambda x, k, x0, gamma: power_3pl(x, k, x0, gamma, A=N, D=1), 
                x_data, 
                y_data, 
                p0=(2, 1, 1),
                # Using y_data as sigma maintains the Poisson-like weighting from your original code
                sigma=y_data, 
                absolute_sigma=True, 
                bounds=([0]*3, [10]*3)
            )
        except RuntimeError:
            print(f"Warning: Curve fit failed for pooled density index {i}")
            popt = [np.nan]*3
        params[i] = popt

    return scale_range, all_modes, params

def run_single_scenario_mode_counting(run_params):
    # 1. Parameter extraction and Logging Setup
    name = run_params['name']
    start_step = run_params['start_step']
    end_step = run_params['end_step']
    step_length = run_params['step_length']

    logger.info(f"Running scenario {name}")

    scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
    config_path = os.path.join(scenario_path, "config.yaml")

    # 3. Load Dataset
    config = SimulationConfig(config_path) 
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    max_steps = dataset.trajectories.shape[0]
    effective_end_step = end_step if end_step is not None and end_step <= max_steps else max_steps
    
    if start_step >= effective_end_step:
        logger.info(f"Skipping {name}: start_step ({start_step}) >= end_step ({effective_end_step}).")
        return

    # 5. Simulation Loop
    step_range = range(start_step, effective_end_step, step_length)

    scale_range, all_modes, params = compute_scaling_law(dataset, np.array(step_range), save_path=scenario_path)

    # test_scales = np.logspace(np.log10(scale_range[0, 0]), np.log10(scale_range[0, 1]), 40)
    # plt.plot(test_scales, all_modes[0])
    # N = dataset.positions_at_time_step(start_step).shape[0]
    # print(N)
    # plt.plot(test_scales, power_2pl(test_scales, 0.5, 0.1, A=N, D=1))
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()

def run_multi_scenarios_gt_scale():
    for run_params in DATASET_RUNS:
        name = run_params['name']
        start_step = run_params['start_step']
        end_step = run_params['end_step']
        step_length = run_params['step_length']

        scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
        config_path = os.path.join(scenario_path, "config.yaml")

        # 3. Load Dataset
        config = SimulationConfig(config_path) 
        factory = DatasetFactory()
        dataset = factory.get_dataset(config.data_file)

        max_steps = dataset.trajectories.shape[0]
        effective_end_step = end_step if end_step is not None and end_step <= max_steps else max_steps

        step_range = range(start_step, effective_end_step, step_length)

        scale_range, all_modes, params = compute_scaling_law(dataset, step_range, scenario_path)
        # scales_estim = np.load(scenario_path + '/reconstruction_scale_estim.npy')

        N_ = [dataset.positions_at_time_step(step).shape[0] for step in step_range]
        k_ = params[:, 0]
        x0_ = params[:, 1]

        num_gmm_gt = [10 for N in N_]
        # num_gmm_gt = [max(10, int(0.01*N)) for N in N_]
        scales_gt = []

        # 
        num_test_scale = 40
        for idx, time_step in enumerate(tqdm(step_range)):
            pos_gpu = torch.from_numpy(dataset.positions_at_time_step(time_step)).cuda().float()
            nn_dist = torch.cdist(pos_gpu, pos_gpu) + torch.eye(pos_gpu.shape[0], device='cuda') * 1e10
            avg_nn_dist = torch.median(torch.min(nn_dist, dim=1).values).item()
            f = lambda s: mode_counting(pos_gpu, pos_gpu.clone(), s, max_iter=2000, tol=avg_nn_dist*5e-4)

            test_scales = np.logspace(np.log10(scale_range[idx, 0]), np.log10(scale_range[idx, 1]), num_test_scale)

            s_low = test_scales[np.argmin(all_modes[idx] > 10).item() - 1].item()
            s_high = test_scales[np.argmax(all_modes[idx] < 10).item()].item()

            scale_gt = find_target_scale(f, 10, s_low, s_high)
            scales_gt.append(scale_gt)
            if f(scale_gt) != 10:
                raise ValueError("find scale fails.")

        gt = {
            'num_gmm_gt': np.array(num_gmm_gt),
            'scales_gt': np.array(scales_gt)
        }
        np.savez(scenario_path + '/reconstruction_scale.npz', **gt)

        # Setup Plotting
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        move_figure(fig, 2800, 100)

        ax.plot(np.array(step_range), scales_gt, label='gt')
        # ax.plot(np.array(step_range), scales_estim, label='estim')
        ax.legend()
    
    plt.show()

def run_multi_scenarios_scale_estimation():
    for run_params in DATASET_RUNS:
        run_single_scenario_scale_estimation(run_params)

def run_single_scenario_scale_estimation(run_params):
    force_update = True

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

    # 2. Initialize Metrics (must be re-initialized for each run)
    scales = []

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
    if force_update:
        # 4. System Initialization
        cam_system = MultiCameraSystem.create_homogeneous_system(
            state_class=CameraState,
            intrinsics=config.intrinsics_params,
            H=config.H, W=config.W, 
            poses_or_RTs=config.cam_poses,
            near_clip=config.near_clip, far_clip=config.far_clip, 
            size=config.size,
            device='cuda')
        density_reconstructor = DensityReconstructor(max_iter=config.iter, use_decoupled=False)
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

        # 5. Simulation Loop
        for time_step in (pbar := tqdm(step_range, desc=f"Processing {name}")):
            positions = dataset.positions_at_time_step(time_step)
            poses, _, images, masks = cam_system.simulate_vision(positions, renderer='gaussian')

            model, scale_spaces = \
            density_reconstructor.process_frame(cam_system, images, positions=positions,
                                                initGMM=None,
                                                is_adaptive_scale=True, estimate_scale_only=True,
                                                is_store_intermediate=False, is_log=False,
                                                output_dir=os.path.join(log_file_path, f"t_{time_step:03d}"),
                                                debug=False,
                                                train_params=train_params,
                                                reconstruction_params=reconstruction_params)

            scales.append(density_reconstructor.scale)
        scales = np.array(scales)

        save_path = os.path.join(scenario_path, "reconstruction_scale_estim.npy")
        if IS_LOGGING:
            np.save(save_path, scales)
            logger.info(f"Statistics saved to: {save_path}")
        logger.info(f"Finished scenario {name}")
    else:
        save_path = os.path.join(scenario_path, "reconstruction_scale_estim.npy")
        scales = np.load(save_path)

    gt_data = np.load(scenario_path + '/reconstruction_scale.npz')
    gt_scales = gt_data['scales_gt']

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    move_figure(fig, 2800, 100)

    ax.plot(np.array(step_range), gt_scales, label='gt')
    ax.plot(np.array(step_range), scales, label='estim')
    ax.legend()

def run_multi_scenarios_scale_estimation_after_training():
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    move_figure(fig, 100, 100)

    start_idx = 0

    mean_true_estim_error = []
    std_true_estim_error = []

    mean_pretrain_estim_error = []
    std_pretrain_estim_error = []

    mean_trained_estim_error = []
    std_trained_estim_error = []

    mean_true2_estim_error = []
    std_true2_estim_error = []

    scales_gt, scales_beform_train, scales = np.array([]), np.array([]), np.array([])
    for run_params in DATASET_RUNS:
        scales_gt_, scales_beform_train_, scales_, scales_gt_gmm_, scales_gt_gmm_2_ = run_single_scenario_scale_estimation_after_training(run_params)
        scales_gt = np.hstack((scales_gt, scales_gt_))
        scales_beform_train = np.hstack((scales_beform_train, scales_beform_train_))
        scales = np.hstack((scales, scales_))

        mean_true_estim_error.append(np.mean(np.abs((scales_gt_gmm_ - scales_gt_)) / scales_gt_).item())
        std_true_estim_error.append(np.std(np.abs((scales_gt_gmm_ - scales_gt_)) / scales_gt_).item())

        mean_pretrain_estim_error.append(np.mean(np.abs((scales_beform_train_ - scales_gt_)) / scales_gt_).item())
        std_pretrain_estim_error.append(np.std(np.abs((scales_beform_train_ - scales_gt_)) / scales_gt_).item())

        mean_trained_estim_error.append(np.mean(np.abs((scales_ - scales_gt_)) / scales_gt_).item())
        std_trained_estim_error.append(np.std(np.abs((scales_ - scales_gt_)) / scales_gt_).item())

        mean_true2_estim_error.append(np.mean(np.abs((scales_gt_gmm_2_ - scales_gt_)) / scales_gt_).item())
        std_true2_estim_error.append(np.std(np.abs((scales_gt_gmm_2_ - scales_gt_)) / scales_gt_).item())

        ax.plot(np.arange(scales_gt_.shape[0]) + start_idx, scales_gt_, color='blue', label='groundtruth')
        ax.plot(np.arange(scales_gt_.shape[0]) + start_idx, scales_beform_train_, color='orange', label='pretrain_estim')
        ax.plot(np.arange(scales_gt_.shape[0]) + start_idx, scales_, color='green', label='trained_estim')
        ax.plot(np.arange(scales_gt_.shape[0]) + start_idx, scales_gt_gmm_, color='cyan', label='groundtruth_gmm')
        ax.plot(np.arange(scales_gt_.shape[0]) + start_idx, scales_gt_gmm_2_, color='black', label='groundtruth_gmm_2')
        start_idx = start_idx + scales_gt_.shape[0]
    ax.legend()
    plt.show()

    # print(mean_pretrain_estim_error)
    # print(std_pretrain_estim_error)
    print(" & ".join(rf"{a*100:.1f} $\pm$ {b*100:.1f}" for a, b in zip(mean_true_estim_error, std_true_estim_error)))
    print()
    print(" & ".join(rf"{a*100:.1f} $\pm$ {b*100:.1f}" for a, b in zip(mean_pretrain_estim_error, std_pretrain_estim_error)))
    print()
    print(" & ".join(rf"{a*100:.1f} $\pm$ {b*100:.1f}" for a, b in zip(mean_trained_estim_error, std_trained_estim_error)))
    print()
    print(" & ".join(rf"{a*100:.1f} $\pm$ {b*100:.1f}" for a, b in zip(mean_true2_estim_error, std_true2_estim_error)))
    # print(mean_trained_estim_error)
    # print(std_trained_estim_error)

def run_single_scenario_scale_estimation_after_training(run_params):
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
    
    config = SimulationConfig(config_path) 
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    max_steps = dataset.trajectories.shape[0]
    effective_end_step = end_step if end_step is not None and end_step <= max_steps else max_steps
    
    if start_step >= effective_end_step:
        logger.info(f"Skipping {name}: start_step ({start_step}) >= end_step ({effective_end_step}).")
        return

    step_range = range(start_step, effective_end_step, step_length)
    scale_range, all_modes, params = compute_scaling_law(dataset, np.array(step_range), save_path=scenario_path)
    
    scales_gt = np.load(os.path.join(scenario_path, "reconstruction_scale.npz"))['scales_gt']

    save_path = os.path.join(scenario_path, "reconstruction_scale_estim_after_training.npy")
    save_path2 = os.path.join(scenario_path, "reconstruction_scale_estim_true_gmm.npy")
    # if not os.path.exists(save_path):
    if True:
        scales = []
        scales_gt_gmm = []
        scales_gt_gmm_2 = []
        for idx, time_step in enumerate(tqdm(step_range, desc=f"Processing {name}")):
            positions = dataset.positions_at_time_step(time_step)
            N = positions.shape[0]
            cov_matrix = np.cov(positions, rowvar=False)
            # Use eigh for symmetric matrices; returns eigenvalues and eigenvectors
            eigenvalues_true, _ = np.linalg.eigh(cov_matrix)
            cov_ = calculate_gmm_covariance(torch.from_numpy(positions.astype(np.float32)).cuda(), 
                                            torch.ones((N, 1), dtype=torch.float32, device='cuda'), 
                                            torch.ones((N, 1), dtype=torch.float32, device='cuda') * scales_gt[idx])
            eigenvalues_true_blur, _ = torch.linalg.eigh(cov_)
            eigenvalues_true_blur = eigenvalues_true_blur - scales_gt[idx]**2

            time_step_file_path = os.path.join(log_file_path, f"t_{time_step:03d}")
            model_path = os.path.join(time_step_file_path, "checkpoint_level_0.pth")
            training_history = GaussianModel.load_training_history(model_path)

            model_estim = GaussianModel.load_iter(training_history, 99)

            cov = calculate_gmm_covariance(model_estim._xyz, model_estim._weights, model_estim._radius)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            eigenvalues = torch.clamp(eigenvalues - scales_gt[idx] ** 2, min=scales_gt[idx] ** 2)
            radii_ = 2 * torch.sqrt(eigenvalues)
            # effective_volume = ((4/3) * np.pi * torch.prod(radii_)).item() * 1.5378
            effective_volume = torch.prod(radii_).item() / 0.14877254
            effective_N = torch.sum(model_estim._weights).item()

            s_start, s_end = scale_range[idx]
            test_scales = np.logspace(np.log10(s_start), np.log10(s_end), 40)

            scales.append(analytic_solution_scale_at_x_constant(10, N=effective_N, d=3, V=effective_volume))
            # scales.append(analytic_solution_scale_at_x_constant(10, N=effective_N, d=2, V=effective_area))

            model_gt = torch.load(os.path.join(scenario_path, "logs", log_name, f"t_{time_step:03d}", "baseline_level_0.pth"))
            effective_volume = np.prod(2 * np.sqrt(eigenvalues_true)).item() / 0.14877254
            # effective_volume = torch.sum(4/3 * torch.pi * (1.7 * model_gt['_radius'])**3).item()
            # effective_N = torch.sum(model_gt['_weights']).item()
            scales_gt_gmm.append(analytic_solution_scale_at_x_constant(10, N=effective_N, d=3, V=effective_volume))

            cov = calculate_gmm_covariance(model_gt['_xyz'], model_gt['_weights'], model_gt['_radius'])
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            eigenvalues = torch.clamp(eigenvalues - scales_gt[idx] ** 2, min=scales_gt[idx] ** 2)
            radii_ = 2 * torch.sqrt(eigenvalues)
            # effective_volume = ((4/3) * np.pi * torch.prod(radii_)).item()
            effective_volume = torch.prod(radii_).item() / 0.14877254

            scales_gt_gmm_2.append(analytic_solution_scale_at_x_constant(10, N=effective_N, d=3, V=effective_volume * 1.5378))

            # fig, ax = plt.subplots(figsize=(12, 8))
            # move_figure(fig, 100, 100)
            # ax.scatter(test_scales, all_modes[idx])
            # # N = positions.shape[0]
            # # plt.plot(test_scales, power_2pl(test_scales, params[idx][0], params[idx][1], A=N, D=1))
            # ax.plot(test_scales, analytic_solution(test_scales/effective_volume**(1/3), N=effective_N, d=3, pbc=False), color='orange')
            # # plt.plot(test_scales, analytic_solution(test_scales/(density_reconstructor.A*2)**(1/2), N=N, d=2, pbc=False))
            # ax.hlines(10, test_scales[0], test_scales[-1])
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.show()
        
        np.save(save_path, scales)
        np.save(save_path2, scales_gt_gmm)
    else:
        scales = np.load(save_path)
        scales_gt_gmm = np.load(save_path2)

    scales_beform_train = np.load(os.path.join(scenario_path, "reconstruction_scale_estim.npy"))
    return scales_gt, scales_beform_train, scales, scales_gt_gmm, scales_gt_gmm_2

def run_multi_scenarios_effective_volume_estimation():
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    move_figure(fig, 2800, 100)

    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111)
    move_figure(fig2, 2800, 100)

    start_idx = 0

    mean_pos_error = []
    std_pos_error = []

    mean_gmm_error = []
    std_gmm_error = []

    mean_estim_error = []
    std_estim_error = []

    dim = 0 # 0 for 2d, 1 for 3d

    for run_params in DATASET_RUNS:
        effective_volume_gt_, effective_volume_pos_, effective_volume_gmm_, effective_volume_estim_ = run_single_scenario_effective_volume_estimation(run_params)

        def formula_3d(x):
            return 3.03728557 * x ** 0.94563168
        
        def formula_2d(x):
            return 1.20175111 * x ** 0.98303737
        
        formula = formula_2d

        pos_error = np.abs((formula(effective_volume_pos_[dim]) - effective_volume_gt_[dim])) / effective_volume_gt_[dim]
        mean_pos_error.append(np.mean(pos_error).item())
        std_pos_error.append(np.std(pos_error).item())

        gmm_error = np.abs((formula(effective_volume_gmm_[dim]) - effective_volume_gt_[dim])) / effective_volume_gt_[dim]
        mean_gmm_error.append(np.mean(gmm_error).item())
        std_gmm_error.append(np.std(gmm_error).item())

        estim_error = np.abs((formula(effective_volume_estim_[dim]) - effective_volume_gt_[dim])) / effective_volume_gt_[dim]
        mean_estim_error.append(np.mean(estim_error).item())
        std_estim_error.append(np.std(estim_error).item())

        ax.plot(np.arange(len(effective_volume_gt_[dim])) + start_idx, effective_volume_gt_[dim], color='blue', label='GT')
        ax.plot(np.arange(len(effective_volume_gt_[dim])) + start_idx, formula(effective_volume_pos_[dim]), color='orange', label='POS')
        ax.plot(np.arange(len(effective_volume_gt_[dim])) + start_idx, formula(effective_volume_gmm_[dim]), color='green', label='GMM')
        ax.plot(np.arange(len(effective_volume_gt_[dim])) + start_idx, formula(effective_volume_estim_[dim]), color='cyan', label='ESTIM')
        ax.vlines([start_idx], 0, 1e4)

        ax2.scatter(effective_volume_estim_[dim], effective_volume_gt_[dim])
        ax2.plot(effective_volume_estim_[dim], formula(effective_volume_estim_[dim]))
        
        start_idx = start_idx + len(effective_volume_gt_[dim])
    ax.legend()
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    plt.show()

    print('------POS-------')
    print(" & ".join(rf"{a*100:.1f} $\pm$ {b*100:.1f}" for a, b in zip(mean_pos_error, std_pos_error)))
    print()
    print('------GMM-------')
    print(" & ".join(rf"{a*100:.1f} $\pm$ {b*100:.1f}" for a, b in zip(mean_gmm_error, std_gmm_error)))
    print()
    print('------ESTIM-------')
    print(" & ".join(rf"{a*100:.1f} $\pm$ {b*100:.1f}" for a, b in zip(mean_estim_error, std_estim_error)))

def run_single_scenario_effective_volume_estimation(run_params):
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
    
    config = SimulationConfig(config_path) 
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    max_steps = dataset.trajectories.shape[0]
    effective_end_step = end_step if end_step is not None and end_step <= max_steps else max_steps
    
    if start_step >= effective_end_step:
        logger.info(f"Skipping {name}: start_step ({start_step}) >= end_step ({effective_end_step}).")
        return

    step_range = range(start_step, effective_end_step, step_length)
    scale_range, all_modes, params = compute_scaling_law(dataset, np.array(step_range), save_path=scenario_path)

    gt_data = np.load(scenario_path + '/reconstruction_scale.npz')
    gt_scales = gt_data['scales_gt']

    # if not os.path.exists(save_path):
    if True:
        effective_volume_gt = [[], []]
        effective_volume_pos = [[], []]
        effective_volume_gmm = [[], []]
        effective_volume_estim = [[], []]
        for idx, time_step in enumerate(tqdm(step_range, desc=f"Processing {name}")):
            positions = dataset.positions_at_time_step(time_step)
            N = positions.shape[0]

            s_start, s_end = scale_range[idx]
            test_scales = np.logspace(np.log10(s_start), np.log10(s_end), 40)

            mode_count = power_2pl(test_scales, *params[idx], A=N, D=1)
            
            f_2d = lambda x: np.sum(np.abs(np.log(analytic_solution(test_scales / x**(1/2), N=N, d=2)) - np.log(mode_count)))
            f_3d = lambda x: np.sum(np.abs(np.log(analytic_solution(test_scales / x**(1/3), N=N, d=3)) - np.log(mode_count)))

            result_2d = minimize_scalar(f_2d, bounds=(0.001, 1e7))
            result_3d = minimize_scalar(f_3d, bounds=(0.001, 1e7))

            if result_2d.fun.item() < result_3d.fun.item():
                effective_volume_gt[0].append(result_2d.x.item())

                # POS VOLUME ESTIMATION
                cov_matrix = np.cov(positions, rowvar=False)
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                radii = 2 * np.sqrt(eigenvalues)
                sorted_radii = np.sort(radii)
                effective_volume_pos[0].append(np.prod(sorted_radii[1:]).item() * np.pi)

                # TRUE GMM VOLUME ESTIMATION
                model_gt = torch.load(os.path.join(scenario_path, "logs", log_name, f"t_{time_step:03d}", "baseline_level_0.pth"))
                cov = calculate_gmm_covariance(model_gt['_xyz'], model_gt['_weights'], model_gt['_radius'])
                eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                eigenvalues = torch.clamp(eigenvalues - gt_scales[idx] ** 2, min=gt_scales[idx] ** 2)
                radii_ = 2 * torch.sqrt(eigenvalues)
                sorted_radii = torch.sort(radii_)[0]
                tmp = np.pi * torch.prod(radii_[1:]).item()
                effective_volume_gmm[0].append(tmp)

                # ESTIM GMM VOLUME ESTIMATION
                time_step_file_path = os.path.join(log_file_path, f"t_{time_step:03d}")
                model_path = os.path.join(time_step_file_path, "checkpoint_level_0.pth")
                training_history = GaussianModel.load_training_history(model_path)
                model_estim = GaussianModel.load_iter(training_history, 99)
                cov = calculate_gmm_covariance(model_estim._xyz, model_estim._weights, model_estim._radius)
                eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                eigenvalues = torch.clamp(eigenvalues - gt_scales[idx] ** 2, min=gt_scales[idx] ** 2)
                radii_ = 2 * torch.sqrt(eigenvalues)
                sorted_radii = torch.sort(radii_)[0]
                tmp = np.pi * torch.prod(sorted_radii[1:]).item()
                effective_volume_estim[0].append(tmp)

            else:
                effective_volume_gt[1].append(result_3d.x.item())

                # POS VOLUME ESTIMATION
                cov_matrix = np.cov(positions, rowvar=False)
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                radii = 2 * np.sqrt(eigenvalues)
                # effective_volume_pos[1].append(np.prod(radii).item() / 0.14877254)
                tmp = (4/3) * np.pi * np.prod(radii).item()
                effective_volume_pos[1].append(tmp)
                # effective_volume_pos[1].append((4/3) * np.pi * np.prod(radii).item())

                # TRUE GMM VOLUME ESTIMATION
                model_gt = torch.load(os.path.join(scenario_path, "logs", log_name, f"t_{time_step:03d}", "baseline_level_0.pth"))
                cov = calculate_gmm_covariance(model_gt['_xyz'], model_gt['_weights'], model_gt['_radius'])
                eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                eigenvalues = torch.clamp(eigenvalues - gt_scales[idx] ** 2, min=gt_scales[idx] ** 2)
                radii_ = 2 * torch.sqrt(eigenvalues)
                # effective_volume_gmm[1].append(torch.prod(radii_).item() / 0.14877254)
                tmp = (4/3) * np.pi * torch.prod(radii_).item()
                effective_volume_gmm[1].append(tmp)

                # ESTIM GMM VOLUME ESTIMATION
                time_step_file_path = os.path.join(log_file_path, f"t_{time_step:03d}")
                model_path = os.path.join(time_step_file_path, "checkpoint_level_0.pth")
                training_history = GaussianModel.load_training_history(model_path)
                model_estim = GaussianModel.load_iter(training_history, 99)
                cov = calculate_gmm_covariance(model_estim._xyz, model_estim._weights, model_estim._radius)
                eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                eigenvalues = torch.clamp(eigenvalues - gt_scales[idx] ** 2, min=gt_scales[idx] ** 2)
                radii_ = 2 * torch.sqrt(eigenvalues)
                tmp = (4/3) * np.pi * torch.prod(radii_).item()
                effective_volume_estim[1].append(tmp)
                # effective_volume_estim[1].append((4/3) * np.pi * torch.prod(radii_).item())
    effective_volume_gt = [np.array(arr) for arr in effective_volume_gt]
    effective_volume_pos = [np.array(arr) for arr in effective_volume_pos]
    effective_volume_gmm = [np.array(arr) for arr in effective_volume_gmm]
    effective_volume_estim = [np.array(arr) for arr in effective_volume_estim]

    return effective_volume_gt, effective_volume_pos, effective_volume_gmm, effective_volume_estim

def examine_scale_estimation_single_time_step():
    run_params = DATASET_RUNS[0]

    time_step = 350

    name = run_params['name']
    start_step = run_params['start_step']
    end_step = run_params['end_step']
    step_length = run_params['step_length']

    idx = int((time_step - start_step) / step_length)

    scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
    config_path = os.path.join(scenario_path, "config.yaml")

    config = SimulationConfig(config_path) 
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    max_steps = dataset.trajectories.shape[0]
    effective_end_step = end_step if end_step is not None and end_step <= max_steps else max_steps
    step_range = range(start_step, effective_end_step, step_length)

    cam_system = MultiCameraSystem.create_homogeneous_system(
        state_class=CameraState,
        intrinsics=config.intrinsics_params,
        H=config.H, W=config.W, 
        poses_or_RTs=config.cam_poses,
        near_clip=config.near_clip, far_clip=config.far_clip, 
        size=config.size,
        device='cuda')
    density_reconstructor = DensityReconstructor(max_iter=config.iter, use_decoupled=False)
    reconstruction_params = {
        'targetd_num_mode': 10,
        # voxel method
        'voxel_scale': 0.5,
        'voxel_peak_threshold': 0.03,
        'voxel_grid_max_size': 32,
        'voxel_peaks_number': 30
    }

    positions = dataset.positions_at_time_step(time_step)
    poses, _, images, masks = cam_system.simulate_vision(positions, renderer='gaussian')

    model, scale_spaces = \
        density_reconstructor.process_frame(cam_system, images, positions=positions,
                                            initGMM=None,
                                            is_adaptive_scale=True, estimate_scale_only=True,
                                            debug=False,
                                            reconstruction_params=reconstruction_params)
    
    scale_range, all_modes, params = compute_scaling_law(dataset, np.array(step_range), save_path=scenario_path)

    s_start, s_end = scale_range[idx]
    test_scales = np.logspace(np.log10(s_start), np.log10(s_end), 40)

    # 

    plt.scatter(test_scales, all_modes[idx])
    N = positions.shape[0]
    plt.plot(test_scales, power_2pl(test_scales, params[idx][0], params[idx][1], A=N, D=1))
    print(density_reconstructor.volume)
    print(density_reconstructor.radii)
    plt.plot(test_scales, analytic_solution(test_scales/density_reconstructor.volume**(1/3), N=N, d=3, pbc=False))
    plt.plot(test_scales, analytic_solution(test_scales/(density_reconstructor.A*2)**(1/2), N=N, d=2, pbc=False))
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

def calculate_gmm_covariance(means: torch.Tensor, weights: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    """
    Calculates the global covariance matrix of an isotropic GMM.
    
    Args:
        means: Tensor of shape [N, 3] containing the centers.
        weights: Tensor of shape [N, 1] containing the mixture weights.
        radius: Tensor of shape [N, 1] containing the isotropic standard deviation.
        
    Returns:
        global_cov: Tensor of shape [3, 3] representing the global covariance matrix.
    """
    # 1. Ensure weights sum to 1
    w = weights / weights.sum()
    
    # 2. Calculate the global mean
    # Broadcasting w [N, 1] over means [N, 3]
    global_mean = (w * means).sum(dim=0, keepdim=True) # Shape: [1, 3]
    
    # 3. Calculate the covariance from the spread of the component means
    # Center the means
    centered_means = means - global_mean # Shape: [N, 3]
    
    # Compute: Sum of w_i * (mu_i - mu) * (mu_i - mu)^T
    # Using matrix multiplication for efficiency: (3, N) @ (N, 3) -> (3, 3)
    cov_from_means = (w * centered_means).T @ centered_means
    
    # 4. Calculate the covariance from the individual isotropic components
    # Assuming 'radius' represents standard deviation, so variance is radius^2.
    # (If 'radius' is already variance, just use 'radius' instead of 'radius ** 2')
    variances = radius ** 2
    
    # Weighted sum of variances
    global_var_scalar = (w * variances).sum()
    
    # Construct the isotropic covariance matrix part
    cov_from_components = global_var_scalar * torch.eye(3, device=means.device, dtype=means.dtype)
    
    # 5. Total Covariance
    global_cov = cov_from_means + cov_from_components
    
    return global_cov

def calculate_gmm_3rd_moment(means: torch.Tensor, weights: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    """
    Calculates the 3rd central moment tensor (unnormalized skewness) of an isotropic GMM.
    
    Args:
        means: Tensor of shape [N, 3] containing the centers.
        weights: Tensor of shape [N, 1] containing the mixture weights.
        radius: Tensor of shape [N, 1] containing the isotropic standard deviation.
        
    Returns:
        third_moment: Tensor of shape [3, 3, 3].
    """
    # 1. Normalize weights and find the global mean
    w = weights / weights.sum()
    global_mean = (w * means).sum(dim=0, keepdim=True) # [1, 3]
    
    # 2. Calculate the centered means (d_i)
    d = means - global_mean # [N, 3]
    
    # 3. Term 1: Skewness from the arrangement of the component means
    # Sum over N: w_i * d_ia * d_ib * d_ic
    # w.squeeze(-1) ensures shape is [N]
    term1 = torch.einsum('ni,nj,nk,n->ijk', d, d, d, w.squeeze(-1))
    
    # 4. Term 2: Cross terms from the component variances
    variances = radius ** 2 # [N, 1]
    
    # Pre-calculate the weighted sum vector: S = sum_i(w_i * r_i^2 * d_i)
    # Shape of S will be [3]
    S = (w * variances * d).sum(dim=0)
    
    I = torch.eye(3, device=means.device, dtype=means.dtype) # [3, 3]
    
    # Construct the tensor components: S_a*delta_bc + S_b*delta_ac + S_c*delta_ab
    term2_a = torch.einsum('i,jk->ijk', S, I)
    term2_b = torch.einsum('j,ik->ijk', S, I)
    term2_c = torch.einsum('k,ij->ijk', S, I)
    
    # 5. Total 3rd Central Moment
    third_moment = term1 + term2_a + term2_b + term2_c
    
    return third_moment

from scipy.spatial import KDTree
def run_multi_scenarios_effective_volume_find_formula():
    fig = plt.figure(figsize=(10, 6))
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)
    move_figure(fig, 2800, 100)

    fig2 = plt.figure(figsize=(10, 6))
    # ax2 = fig2.add_subplot(111, projection='3d')
    ax2 = fig2.add_subplot(111)
    move_figure(fig2, 2800, 100)

    # Store data grouped by dataset
    datasets_2d_X = []
    datasets_2d_y = []
    datasets_3d_X = []
    datasets_3d_y = []

    for run_params in DATASET_RUNS:
        N_area, N_volume, effective_area, effective_volume, avg_nn_dist_area, avg_nn_dist_volume, ellipsoid_area, ellipsoid_volume, moments_area, moments_volume, blurred_volume, blurred_volume_estim = run_single_scenario_effective_volume_find_formula(run_params)

        # 2D Data
        input_data_2d = np.hstack((N_area.reshape((-1, 1)), avg_nn_dist_area.reshape((-1, 1)), ellipsoid_area.reshape((-1, 1)), moments_area.reshape((-1, 4))))
        output_data_2d = effective_area.reshape((-1, 1))
        
        # Only append if this dataset actually contains 2D data
        if input_data_2d.shape[0] > 0:
            datasets_2d_X.append(input_data_2d)
            datasets_2d_y.append(output_data_2d)

        # 3D Data
        input_data_3d = np.hstack((N_volume.reshape((-1, 1)), avg_nn_dist_volume.reshape((-1, 1)), ellipsoid_volume.reshape((-1, 1)), moments_volume.reshape((-1, 6)), blurred_volume.reshape((-1, 1)), blurred_volume_estim.reshape((-1, 1))))
        output_data_3d = effective_volume.reshape((-1, 1))
        
        # Only append if this dataset actually contains 3D data
        if input_data_3d.shape[0] > 0:
            datasets_3d_X.append(input_data_3d)
            datasets_3d_y.append(output_data_3d)

    def mean_of_mean_percentage_error(params, X_grouped, y_grouped, formula_func):
        """
        X_grouped: list of numpy arrays (each array is a dataset's inputs)
        y_grouped: list of numpy arrays (each array is a dataset's targets)
        formula_func: A python function representing the PySR equation structure
        params: the coefficients to optimize
        """
        dataset_errors = []
        
        for X_ds, y_ds in zip(X_grouped, y_grouped):
            # Generate predictions using the current parameters
            preds = formula_func(X_ds, *params)
            
            # Calculate percentage error for this dataset
            # Adding a small epsilon to avoid division by zero
            perc_error = np.abs((preds.ravel() - y_ds.ravel()) / (y_ds.ravel() + 1e-8))
            
            # Mean percentage error FOR THIS DATASET
            dataset_errors.append(np.mean(perc_error))
            
        # Return the MEAN of the dataset means
        if not dataset_errors:
            return np.nan 
            
        return np.mean(dataset_errors)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.model_selection import cross_val_score
    from itertools import combinations

    y = np.vstack(datasets_2d_y)
    X = np.vstack(datasets_2d_X)

    # y = np.vstack(datasets_3d_y)
    # X = np.vstack(datasets_3d_X)

    # print("--- 1. Individual Feature Power (Mutual Information) ---")
    # mi_scores = mutual_info_regression(X, y)

    # for i, score in enumerate(mi_scores):
    #     print(f"Feature {i}: Score = {score:.4f}")
    # print("Higher score means stronger individual predictive power.\n")

    # print("--- 2. Relative Importance (Random Forest) ---")
    # rf = RandomForestRegressor(n_estimators=100, random_state=42)
    # rf.fit(X, y)

    # for i, importance in enumerate(rf.feature_importances_):
    #     print(f"Feature {i}: Importance = {importance:.4f}")
    # print("Measures how useful the feature is when all features are present.\n")

    # print("--- 3. Evaluating All 255 Combinations (Exhaustive Search) ---")
    # best_score = -np.inf
    # best_combo = None

    # for k in range(1, X.shape[1] + 1):
    #     # Iterate through all combinations of size k
    #     for combo in combinations(range(X.shape[1]), k):
            
    #         # Subset the input data
    #         X_subset = X[:, combo]
            
    #         # Evaluate using 5-fold cross-validation
    #         # Using negative Mean Squared Error (closer to 0 is better)
    #         scores = cross_val_score(
    #             rf, 
    #             X_subset, 
    #             y, 
    #             cv=5, 
    #             scoring='neg_mean_squared_error',
    #             n_jobs=4 # Uses all CPU cores to speed this up
    #         )
            
    #         mean_score = np.mean(scores)
            
    #         # Update best score if this combination is better
    #         if mean_score > best_score:
    #             best_score = mean_score
    #             best_combo = combo

    # print(f"✅ The absolute best combination of features is: {best_combo}")
    # print(f"🏆 Best Cross-Validated Score (Neg MSE): {best_score:.4f}")

    feature_indices = [2]
    X_chosen = X[:, feature_indices]
    # X_chosen = X

    # ---------------------------------------------------------
    # 2. Configure and Run PySR
    # ---------------------------------------------------------
    from pysr import PySRRegressor
    model = PySRRegressor(
        elementwise_loss="loss(x, y) = abs(x - y) / (y + 1e-8)",

        niterations=1000, 
        
        # The mathematical building blocks the algorithm is allowed to use
        binary_operators=["+", "*", "-", "/", "^"],
        unary_operators=[
            "exp",
            "log",
        ],
        extra_sympy_mappings={"inv": lambda x: 1/x},
        constraints={'^': (-1, 1)},
        # Limit complexity so you don't get massive, unreadable equations
        maxsize=15,
        maxdepth=10,
        
        # Safely limiting CPU cores to avoid Windows memory crashes
        procs=4,
        
        # Update the console to show you the progress
        verbosity=1 ,
        random_state=42,
        progress=False
    )

    model.fit(X_chosen, y)

    # # # 1.100e-01  1.925e-02  y = (X9 / 0.49741) ^ 0.97803

    # # ---------------------------------------------------------
    # # 3. View the Results
    # # ---------------------------------------------------------
    # print("\n--- Model Fitting Complete ---")

    # # PySR creates a 'Pareto Front' - a list of equations balancing simplicity and accuracy.
    # # It automatically selects the one with the best trade-off.
    # print("\n🏆 Best Equation Found:")
    # print(model.sympy())

    # print("\n📊 Full Equation Leaderboard (Complexity vs. Accuracy):")
    # print(model.equations_)

    # ---------------------------------------------------------
    # Example: Fine-Tuning the 3D Formula
    # ---------------------------------------------------------
    print("\n--- Fine Tuning 3D Formula against Custom Metric ---")

    def candidate_formula_3d(X, c0, c1):
        feature_9 = X[:, 2] # Adjust index based on your exact feature selection
        return c0*(feature_9)**c1

    # Initial guess based on PySR's output
    initial_guess_3d = [1.5464514, 1]

    # Optimize!
    from scipy.optimize import minimize
    res_3d = minimize(
        mean_of_mean_percentage_error, 
        x0=initial_guess_3d, 
        args=(datasets_3d_X, datasets_3d_y, candidate_formula_3d),
        method='Nelder-Mead' # Nelder-Mead is robust for non-differentiable absolute errors
    )

    if res_3d.success:
        print(f"✅ Optimized params: {res_3d.x}")
        print(f"📊 Final Mean of Mean Percentage Error (3D): {res_3d.fun * 100:.2f}%")
    else:
        print("Optimization failed:", res_3d.message)

def run_single_scenario_effective_volume_find_formula(run_params):
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
    
    config = SimulationConfig(config_path) 
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    max_steps = dataset.trajectories.shape[0]
    effective_end_step = end_step if end_step is not None and end_step <= max_steps else max_steps
    
    if start_step >= effective_end_step:
        logger.info(f"Skipping {name}: start_step ({start_step}) >= end_step ({effective_end_step}).")
        return

    step_range = range(start_step, effective_end_step, step_length)
    scale_range, all_modes, params = compute_scaling_law(dataset, np.array(step_range), save_path=scenario_path)
    
    save_path = os.path.join(scenario_path, "reconstruction_scale_estim_after_training.npz")
    
    # if True:
    if not os.path.exists(save_path):
        N_area = []
        N_volume = []
        effective_area = []
        effective_volume = []
        avg_nn_dist_area = []
        avg_nn_dist_volume = []
        ellipsoid_volume = []
        ellipsoid_area = []
        
        # Renamed to reflect the principal axes projection
        moments_area = []
        moments_volume = []

        blurred_volume = []
        blurred_volume_estim = []

        gt_data = np.load(scenario_path + '/reconstruction_scale.npz')
        gt_scales = gt_data['scales_gt']

        for idx, time_step in enumerate(tqdm(step_range, desc=f"Processing {name}")):
            time_step_file_path = os.path.join(log_file_path, f"t_{time_step:03d}")

            model_path = os.path.join(time_step_file_path, "checkpoint_level_0.pth")
            training_history = GaussianModel.load_training_history(model_path)
            model = GaussianModel.load_iter(training_history, 99)
    
            positions = dataset.positions_at_time_step(time_step)
            N = positions.shape[0]

            tree = KDTree(positions)
            dists, _ = tree.query(positions, k=2)
            avg_nn_dist = np.mean(dists[:, 1]).item()

            s_start, s_end = scale_range[idx]
            test_scales = np.logspace(np.log10(s_start), np.log10(s_end), 40)

            mode_count = power_2pl(test_scales, *params[idx], A=N, D=1)
            
            f_2d = lambda x: np.sum(np.abs(np.log(analytic_solution(test_scales / x**(1/2), N=N, d=2)) - np.log(mode_count)))
            f_3d = lambda x: np.sum(np.abs(np.log(analytic_solution(test_scales / x**(1/3), N=N, d=3)) - np.log(mode_count)))

            result_2d = minimize_scalar(f_2d, bounds=(0.001, 1e7))
            result_3d = minimize_scalar(f_3d, bounds=(0.001, 1e7))

            # --- EIGEN DECOMPOSITION ---
            cov_matrix = np.cov(positions, rowvar=False)
            # Use eigh for symmetric matrices; returns eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # np.maximum guards against tiny negative floats due to precision errors
            radii = 2 * np.sqrt(np.maximum(eigenvalues, 0))

            # --- FAST APPROXIMATION FOR PRINCIPAL THIRD MOMENT ---
            mean_pos = np.mean(positions, axis=0)
            centered_positions = positions - mean_pos
            
            # Project centered data onto the principal axes (eigenvectors)
            projected_positions = np.dot(centered_positions, eigenvectors)
            
            # Calculate the 3rd central moment along the new orthogonal axes
            principal_third_moment = np.mean(projected_positions**3, axis=0)

            if result_2d.fun.item() < result_3d.fun.item():
                effective_area.append(result_2d.x.item())
                N_area.append(N)
                avg_nn_dist_area.append(avg_nn_dist)
                sorted_radii = np.sort(radii)
                sorted_idx = np.argsort(radii)
                ellipsoid_area.append(np.pi * sorted_radii[1] * sorted_radii[2])
                moments_area.append(np.hstack((sorted_radii[1:], principal_third_moment[[sorted_idx[1], sorted_idx[2]]])))
            else:
                effective_volume.append(result_3d.x.item())
                N_volume.append(N)
                avg_nn_dist_volume.append(avg_nn_dist)
                sorted_radii = np.sort(radii)
                sorted_idx = np.argsort(radii)
                ellipsoid_volume.append(4/3 * np.pi * np.prod(radii))
                moments_volume.append(np.hstack((sorted_radii, principal_third_moment[sorted_idx])))
                r_means, r_weights, r_covs = GMR.runnalls_algorithm_simple_torch(
                    means=torch.from_numpy(positions),
                    radii=torch.full((N, 1), gt_scales[idx], device='cuda', dtype=torch.float),
                    weights=torch.full((N, 1), 1.0, device='cuda', dtype=torch.float),
                    L=10, DEVICE='cuda'
                )
                r_weights = r_weights.reshape((-1, 1))
                r_radius = torch.sqrt(r_covs[:, 0, 0]).reshape((-1, 1))
                cov = calculate_gmm_covariance(r_means, r_weights.reshape((-1, 1)), r_radius) - gt_scales[idx] ** 2 * torch.eye(3, device='cuda', dtype=torch.float32)
                eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                eigenvalues = torch.clamp(eigenvalues, min=0.0)
                radii_ = 2 * torch.sqrt(eigenvalues)
                blurred_volume.append(((4/3) * np.pi * torch.prod(radii_)).item())

                cov = calculate_gmm_covariance(model._xyz, model._weights, model._radius) - gt_scales[idx] ** 2 * torch.eye(3, device='cuda', dtype=torch.float32)
                eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                eigenvalues = torch.clamp(eigenvalues, min=0.0)
                radii_ = 2 * torch.sqrt(eigenvalues)
                blurred_volume_estim.append(((4/3) * np.pi * torch.prod(radii_)).item())
        
        N_area = np.array(N_area)
        N_volume = np.array(N_volume)
        effective_area = np.array(effective_area)
        avg_nn_dist_area = np.array(avg_nn_dist_area)
        effective_volume = np.array(effective_volume)
        avg_nn_dist_volume = np.array(avg_nn_dist_volume)
        ellipsoid_area = np.array(ellipsoid_area)
        ellipsoid_volume = np.array(ellipsoid_volume)
        moments_area = np.array(moments_area)
        moments_volume = np.array(moments_volume)
        blurred_volume = np.array(blurred_volume)
        blurred_volume_estim = np.array(blurred_volume_estim)

        np.savez(
            save_path, 
            N_area=N_area, 
            N_volume=N_volume, 
            effective_area=effective_area, 
            effective_volume=effective_volume, 
            avg_nn_dist_area=avg_nn_dist_area, 
            avg_nn_dist_volume=avg_nn_dist_volume, 
            ellipsoid_area=ellipsoid_area, 
            ellipsoid_volume=ellipsoid_volume,
            moments_area=moments_area,
            moments_volume=moments_volume,
            blurred_volume=blurred_volume,
            blurred_volume_estim=blurred_volume_estim
        )
    else:
        data = np.load(save_path, allow_pickle=True)
        
        N_area = data['N_area']
        N_volume = data['N_volume']
        effective_area = data['effective_area']
        effective_volume = data['effective_volume']
        avg_nn_dist_area = data['avg_nn_dist_area']
        avg_nn_dist_volume = data['avg_nn_dist_volume']
        ellipsoid_area = data['ellipsoid_area']
        ellipsoid_volume = data['ellipsoid_volume']
        moments_area = data['moments_area']
        moments_volume = data['moments_volume']
        blurred_volume = data['blurred_volume']
        blurred_volume_estim = data['blurred_volume_estim']

    return N_area, N_volume, effective_area, effective_volume, avg_nn_dist_area, avg_nn_dist_volume, ellipsoid_area, ellipsoid_volume, moments_area, moments_volume, blurred_volume, blurred_volume_estim

if __name__ == "__main__":
    # run_multi_scenarios_mode_counting()
    # run_multi_scenarios_gt_scale()

    # run_multi_scenarios_scale_estimation()
    # plt.show()

    # run_multi_scenarios_scale_estimation_after_training()
    run_multi_scenarios_effective_volume_estimation()

    # run_multi_scenarios_effective_volume_find_formula()

    # examine_scale_estimation_single_time_step()