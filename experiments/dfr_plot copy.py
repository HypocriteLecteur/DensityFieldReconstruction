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
from dfr.camera_state import CameraState
from dfr.utils import calculate_gmm_dissimilarity, generate_encircling_cameras, compute_metrics_batched_torch
from dfr.visualizer import MultiGMMPlotter
from dfr.gaussian_mixture_reduction import GMR
from dfr.mode_finding import find_target_scale, mode_counting, model_4pl_scale_at_x_constant, analytic_solution
from gaussian_rasterizer_simple_large import rasterize_gaussians
from dfr.utils import move_figure
from experiments.reconstruction_scale_determination import compute_scaling_law
from dfr.density_field_model import GaussianModel

import matplotlib.pyplot as plt

CAM_NUM = 2
LOG_NAME = 'base_reg_cam_2'

DATASET_RUNS = [
    # {
    #     'name': 'swift',
    #     'log_name': LOG_NAME,
    #     'start_step': 0,
    #     'end_step': None,
    #     'step_length': 200,
    # },
    # {
    #     'name': 'starling',
    #     'log_name': LOG_NAME,
    #     'start_step': 0,
    #     'end_step': None,
    #     'step_length': 1,
    # },
    # {
    #     'name': 'jackdaw',
    #     'log_name': LOG_NAME,
    #     'start_step': 350,
    #     'end_step': 550,
    #     'step_length': 10,
    # },
    {
        'name': 'jackdaw2',
        'log_name': LOG_NAME,
        'start_step': 2700,
        'end_step': 3460,
        'step_length': 20,
    },
]

def scale_estimation():
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

        num_test_scale = 40

        scale_range, all_modes, params = compute_scaling_law(dataset, step_range, scenario_path)
        scales_estim = np.load(scenario_path + '/reconstruction_scale_estim.npy')
        scales_estim_after_training = np.load(scenario_path + '/reconstruction_scale_estim_after_training.npy')

        N_ = [dataset.positions_at_time_step(step).shape[0] for step in step_range]
        k_ = params[:, 0]
        x0_ = params[:, 1]

        scales_gt = [model_4pl_scale_at_x_constant(10, A=1, B=N, k=k, x0=x0) for (N, k, x0) in zip(N_, k_, x0_)]
        # np.save(scenario_path + '/reconstruction_scale.npy', np.array(scales_gt))

        # Setup Plotting
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
    name = run_params['name']
    log_name = run_params['log_name']
    start_step = run_params['start_step']
    end_step = run_params['end_step']
    
    # step_length is extracted but not used for slicing since we want the continuous path
    step_length = run_params.get('step_length', 1) 

    scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
    config_path = os.path.join(scenario_path, "config.yaml")

    log_file_path = os.path.join(scenario_path, *["logs", log_name])
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)
    
    # Load Dataset
    config = SimulationConfig(config_path) 
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    max_steps = dataset.trajectories.shape[0]
    effective_end_step = end_step if end_step is not None and end_step <= max_steps else max_steps
    
    # Get positions and active agents mask at the exact end step
    target_idx = effective_end_step - 1
    positions, masks = dataset.positions_at_time_step_mask(target_idx)
    trajectories = dataset.trajectories[:, masks, :]

    # Setup Figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    move_figure(fig, 100, 100)
    ax.view_init(elev=33, azim=-117, roll=0)

    # --- AESTHETICS: Clean, no text ---
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Remove the axis text labels but keep the ticks/grid
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])

    # Customize the grid to look clean and subtle
    ax.grid(color='lightgray', linestyle='--', linewidth=0.5, alpha=0.5)
    fig.tight_layout(pad=0)

    N = trajectories.shape[1]
    
    # 1. Plot the full trajectories from start_step to end_step
    for i in range(N):
        x_coords = trajectories[start_step:effective_end_step, i, 0]
        y_coords = trajectories[start_step:effective_end_step, i, 1]
        z_coords = trajectories[start_step:effective_end_step, i, 2]
        
        ax.plot(
            x_coords, y_coords, z_coords, 
            color='tab:gray',   # Uniform subdued color
            alpha=0.15,         # High transparency to reveal density overlaps
            linewidth=0.6,      # Thinner lines to reduce spatial clutter
            zorder=1            # Render lines behind the agents
        )

    # 2. Plot the end positions of the swarm
    ax.scatter(
        positions[:, 0], 
        positions[:, 1], 
        positions[:, 2], 
        c='#1f2937',        # Sophisticated dark slate/navy
        s=10,                # Small point size
        alpha=0.65,         # Allow dense areas to visually compound
        edgecolors='none',  
        depthshade=True,    # Crucial for 3D depth perception
        zorder=3            
    )

    # Optional: Save with zero padding to keep the image strictly focused on the data
    # fig.savefig(f"figs/scene_traj_{name}.png", transparent=True, bbox_inches='tight', pad_inches=0)

def plot_single_scenario(run_params):
    name = run_params['name']
    log_name = run_params['log_name']
    start_step = run_params['start_step']
    end_step = run_params['end_step']
    step_length = run_params['step_length']

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
    
    step_range = np.arange(start_step, effective_end_step, step_length)
    
    positions, masks = dataset.positions_at_time_step_mask(step_range[-1])
    trajectories = dataset.trajectories[:, masks, :]

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
        # Slice to capture only the recent tail for the current agent
        x_coords = trajectories[start_idx:current_idx, i, 0]
        y_coords = trajectories[start_idx:current_idx, i, 1]
        z_coords = trajectories[start_idx:current_idx, i, 2]
        
        # 2. Apply aesthetic formatting
        # Note: Removing the 'label' argument prevents matplotlib from attempting 
        # to generate a massive, memory-intensive legend for hundreds of agents.
        ax.plot(
            x_coords, y_coords, z_coords, 
            color='tab:gray',   # Uniform subdued color (e.g., 'tab:gray' or 'steelblue')
            alpha=0.15,         # High transparency to reveal density overlaps
            linewidth=0.6,      # Thinner lines to reduce spatial clutter
            zorder=1            # Render lines behind the GMM structural components
        )

    # 3. Plot the current positions of the swarm
    # Assuming `positions` is an (N, 3) numpy array
    ax.scatter(
        positions[:, 0], 
        positions[:, 1], 
        positions[:, 2], 
        c='#1f2937',        # A sophisticated dark slate/navy instead of pure black
        s=6,                # Drastically reduce size (down from 15 to 3 or 4)
        alpha=0.65,         # Allow dense areas to visually compound
        edgecolors='none',  
        depthshade=True,    # Crucial for 3D: fades points that are further away
        zorder=3            
    )

    plt.show()

    # fig.savefig(f"figs/scene_sample_{name}.png", transparent=True, bbox_inches='tight')
    plt.show()

def overview_scaling_law():
    torch.manual_seed(123456)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # 1. Generate points in a 2D SQUARE domain [-1, 1]^2 (Z is eliminated)
    num_points = 50 # Lowered point count so individual peaks are easy to see
    points = (torch.rand((num_points, 2), device=device) * 2.0) - 1.0

    # 2. Define the 2D grid 
    grid_resolution = 200
    x = np.linspace(-1, 1, grid_resolution)
    y = np.linspace(-1, 1, grid_resolution)
    X, Y = np.meshgrid(x, y)

    # Grid coordinates are now strictly 2D
    grid_coords_np = np.vstack([X.ravel(), Y.ravel()]).T
    grid_coords = torch.tensor(grid_coords_np, dtype=torch.float32, device=device)

    # 3. Define the scales
    scales = [0.25, 0.1, 0.05]

    fig = plt.figure(figsize=(12, 10), dpi=300)
    move_figure(fig, 2800, 100)
    ax = fig.add_subplot(111, projection='3d')
    colormaps = ['viridis', 'viridis', 'viridis']
    
    stack_spacing = 2.5 
    
    for i, sigma in enumerate(scales):
        # Calculate 2D squared Euclidean distances
        sq_dists = torch.cdist(grid_coords, points, p=2.0).pow(2)
        kernel_vals = torch.exp(-sq_dists / (2 * sigma**2))
        
        # CORRECTED FOR 2D: The normalization factor for a 2D Gaussian 
        # is 1 / (2 * pi * sigma^2). Notice the power is 1, not 1.5!
        normalization_factor = 1.0 / (num_points * (2 * torch.pi * sigma**2))
        
        densities = torch.sum(kernel_vals, dim=1) * normalization_factor
        density_grid = densities.cpu().numpy().reshape(grid_resolution, grid_resolution)
        density_grid = density_grid / density_grid.max()
        
        Z_plot = (i * stack_spacing) + density_grid
        
        surf = ax.plot_surface(X, Y, Z_plot, 
                               cmap=colormaps[i % len(colormaps)], 
                               alpha=0.85, 
                               linewidth=0, 
                               antialiased=True)
        
        ax.text(-1.2, -1.2, i * stack_spacing - 0.3, f"σ = {sigma}", color='black', fontsize=12)

    # Plot Ground Truth
    top_index = len(scales)
    top_z_base = top_index * stack_spacing
    ax.plot_surface(X, Y, np.full_like(X, top_z_base), color='gray', alpha=0.15)
    
    points_cpu = points.cpu().numpy()
    
    # Points are now plotted exactly where they exist in the 2D math
    ax.scatter(points_cpu[:, 0], points_cpu[:, 1], top_z_base, 
               color='black', s=25, alpha=1.0, edgecolors='none', zorder=10)

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
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 14,       # Slightly larger for readability
        "legend.fontsize": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True, # Highly recommended for log scales
        "ytick.minor.visible": True, # Highly recommended for log scales
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--"       # Dashed grid lines are less distracting
    })

    # 2. Use plt.subplots() which is the modern standard over figure + add_subplot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # move_figure(fig, 2800, 100)  # Assuming this is a custom local function
    
    # 3. Consolidate variables and parameters (DRY Principle)
    n_points = 50
    target_scales = np.array([0.25, 0.1, 0.05])
    
    # Simplify logspace: logspace(-2, 0) is identical to logspace(log10(0.01), log10(1))
    test_scales = np.logspace(-2, 0, 40) 
    
    # Pre-calculate Y values to avoid computing the analytic solution multiple times
    y_values = analytic_solution(test_scales / 2, n_points)
    target_y_values = analytic_solution(target_scales / 2, n_points)
    zeros = np.zeros_like(target_scales)

    # 4. Plot main curve
    ax.plot(test_scales, y_values, color='#2c3e50', lw=2, label='Analytic Solution')

    # 5. Plot reference lines using keyword arguments for readability
    line_color = '#e74c3c' # A slightly softer red
    ax.hlines(y=target_y_values, xmin=zeros, xmax=target_scales, 
              colors=line_color, linestyles='--', alpha=0.8)
    
    ax.vlines(x=target_scales, ymin=zeros, ymax=target_y_values, 
              colors=line_color, linestyles='--', alpha=0.8)

    # 6. Add annotations
    # Simplify the text offset math. Instead of complex log/exp math, 
    # simply multiply by a constant (e.g., 1.15) for a visual offset on a log scale.
    offset_multiplier = 1.15 
    
    for scale, y_val in zip(target_scales, target_y_values):
        ax.text(scale * offset_multiplier, y_val, f"σ = {scale}", 
                color='black', fontsize=12, verticalalignment='center')

    # 7. Format axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Number of Modes', fontsize=12)
    ax.set_xlabel('Scale ($\sigma$)', fontsize=12)
    
    # 8. Clean layout and save
    fig.tight_layout()
    fig.savefig("figs/2d_gss_curve.png", transparent=True, bbox_inches='tight')
    # plt.show()

def visual_hull_diagram():
    run_params = {
        'name': 'swift',
        'log_name': LOG_NAME,
        'start_step': 0,
        'end_step': None,
        'step_length': 200,
    }

    name = run_params['name']
    log_name = run_params['log_name']
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

    cam_positions, cam_radius = generate_encircling_cameras(dataset, step_range, config.intrinsics_params, config.H, config.W, cam_num=4, padding=1)
    cam_poses = np.hstack((cam_positions[:2], np.tile(np.array([1, 0, 0, 0]), (2, 1)))).astype(np.float32)

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
        'lr_max_steps': 500
    }
    density_reconstructor = DensityReconstructor(max_iter=train_params['lr_max_steps'])

    time_step = step_range[0]

    positions = dataset.positions_at_time_step(time_step)
    poses, projections, _, masks = cam_system.simulate_vision(positions, renderer='projection_only')

    model, scale_spaces = \
        density_reconstructor.process_frame(cam_system, point_sets=projections, positions=positions,
                                            initGMM=None,
                                            is_adaptive_scale=True, scale=None,
                                            debug=True,
                                            train_params=train_params,
                                            reconstruction_params=reconstruction_params)

def assumption_3_error():
    import numpy as np
    import matplotlib.pyplot as plt

    def compute_exact_density(U_grid, V_grid, mu_3d, r):
        mu_x, mu_y, mu_z = mu_3d
        var = r**2
        norm = 1.0 / ((2 * np.pi * var) ** 1.5)
        
        z_min = max(0.001, mu_z - 6 * r)
        z_max = mu_z + 6 * r
        z_array = np.linspace(z_min, z_max, 200)
        
        U_exp = U_grid[..., np.newaxis]
        V_exp = V_grid[..., np.newaxis]
        z_exp = z_array[np.newaxis, np.newaxis, :]
        
        dx = U_exp * z_exp - mu_x
        dy = V_exp * z_exp - mu_y
        dz = z_exp - mu_z
        
        integrand = norm * np.exp(-(dx**2 + dy**2 + dz**2) / (2 * var)) * (z_exp**2)
        if hasattr(np, 'trapezoid'):
            return np.trapezoid(integrand, x=z_array, axis=-1)
        else:
            return np.trapz(integrand, x=z_array, axis=-1)

    def compute_affine_density(U_grid, V_grid, mu_3d, r):
        u0 = mu_3d[0] / mu_3d[2]
        v0 = mu_3d[1] / mu_3d[2]
        z0 = mu_3d[2]
        
        J = np.array([
            [1/z0,  0,    -u0/z0],
            [0,     1/z0, -v0/z0]
        ])
        
        Sigma_2d = (r**2) * (J @ J.T)
        inv_Sigma = np.linalg.inv(Sigma_2d)
        det_Sigma = np.linalg.det(Sigma_2d)
        
        dU = U_grid - u0
        dV = V_grid - v0
        
        exponent = -0.5 * (dU**2 * inv_Sigma[0,0] + 2 * dU * dV * inv_Sigma[0,1] + dV**2 * inv_Sigma[1,1])
        return (1.0 / (2 * np.pi * np.sqrt(det_Sigma))) * np.exp(exponent)

    def compute_ortho_density(U_grid, V_grid, mu_3d, r, z_bar=1.0, f=1.0):
        u0 = mu_3d[0] / mu_3d[2]
        v0 = mu_3d[1] / mu_3d[2]
        
        sigma_prime = (f / z_bar) * r
        var_2d = sigma_prime**2
        
        dU = U_grid - u0
        dV = V_grid - v0
        
        exponent = -0.5 * (dU**2 + dV**2) / var_2d
        return (1.0 / (2 * np.pi * var_2d)) * np.exp(exponent)

    def compute_hybrid_affine_density(U_grid, V_grid, mu_3d, r, z_bar=1.0):
        # 1. Find the 2D projection center (same ray as the exact/affine models)
        u0 = mu_3d[0] / mu_3d[2]
        v0 = mu_3d[1] / mu_3d[2]
        
        # 2. Treat as if sitting at depth z_bar using the affine model.
        # The virtual 3D center is [u0 * z_bar, v0 * z_bar, z_bar].
        # Therefore, we construct the Jacobian using z_bar instead of mu_3d[2].
        J = np.array([
            [1/z_bar,  0,       -u0/z_bar],
            [0,        1/z_bar, -v0/z_bar]
        ])
        
        # 3. Compute 2D Covariance
        Sigma_2d = (r**2) * (J @ J.T)
        inv_Sigma = np.linalg.inv(Sigma_2d)
        det_Sigma = np.linalg.det(Sigma_2d)
        
        # 4. Evaluate the Gaussian density over the grid
        dU = U_grid - u0
        dV = V_grid - v0
        
        exponent = -0.5 * (dU**2 * inv_Sigma[0,0] + 
                        2 * dU * dV * inv_Sigma[0,1] + 
                        dV**2 * inv_Sigma[1,1])
                        
        return (1.0 / (2 * np.pi * np.sqrt(det_Sigma))) * np.exp(exponent)

    focal_length = 1.0
    z_bar = 1.0
    
    # Define limits to iterate through
    z_vals = np.linspace(0.8, 1.2, 20)
    u_vals = np.linspace(0.0, 0.5, 20)
    # r_vals = np.linspace(0.01, 0.2, 4)
    r_vals = [0.1]
    
    u_flat = []
    z_flat = []
    r_flat = []
    err_affine_flat = []
    err_ortho_flat = []
    err_hybrid_flat = []
    
    # Populate the arrays
    for Z in z_vals:
        for U in u_vals:
            for r in r_vals:
                mu_3d = np.array([U * Z, 0.0, Z]) 
                
                sigma_prime = (focal_length / z_bar) * r
                grid_range = 4 * sigma_prime
                grid_res = sigma_prime / 8.0 
                
                u_centers = np.arange(U - grid_range, U + grid_range, grid_res)
                v_centers = np.arange(0.0 - grid_range, 0.0 + grid_range, grid_res)
                # if len(u_centers) == 0 or len(v_centers) == 0:
                #     continue
                
                U_grid, V_grid = np.meshgrid(u_centers, v_centers)
                
                D_exact = compute_exact_density(U_grid, V_grid, mu_3d, r)
                D_affine = compute_affine_density(U_grid, V_grid, mu_3d, r)
                D_ortho = compute_ortho_density(U_grid, V_grid, mu_3d, r, z_bar, focal_length)
                D_hybrid = compute_hybrid_affine_density(U_grid, V_grid, mu_3d, r, z_bar)
                
                pixel_area = grid_res**2
                e_aff = np.sum(np.abs(D_exact - D_affine)) * pixel_area
                e_ortho = np.sum(np.abs(D_exact - D_ortho)) * pixel_area
                e_hybrid = np.sum(np.abs(D_exact - D_hybrid)) * pixel_area
                
                u_flat.append(U)
                z_flat.append(Z)
                r_flat.append(r)
                err_affine_flat.append(e_aff)
                err_ortho_flat.append(e_ortho)
                err_hybrid_flat.append(e_hybrid)
            
    u_flat = np.array(u_flat)
    z_flat = np.array(z_flat)
    r_flat = np.array(r_flat)
    err_affine_flat = np.array(err_affine_flat)
    err_ortho_flat = np.array(err_ortho_flat)
    err_hybrid_flat = np.array(err_hybrid_flat)

    # Set up figure
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1
    })
    def style_3d_ax(ax):
        """Applies academic styling to a 3D axis."""
        ax.view_init(elev=25, azim=-45)
        
        # Make background panes transparent
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Subtler grid lines
        ax.xaxis._axinfo["grid"].update({"color": (0.8, 0.8, 0.8, 0.5), "linewidth": 0.5})
        ax.yaxis._axinfo["grid"].update({"color": (0.8, 0.8, 0.8, 0.5), "linewidth": 0.5})
        ax.zaxis._axinfo["grid"].update({"color": (0.8, 0.8, 0.8, 0.5), "linewidth": 0.5})
        
        # Remove axis lines to clean up the plot bounding box
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')

    # ---------------------------------------------------------
    # Figure 1: Affine Error (Scatter)
    # ---------------------------------------------------------
    def plot_3d_surface_error(err_flat, name):
        fig1 = plt.figure(figsize=(7, 6))
        move_figure(fig1, 2800, 100)
        ax1 = fig1.add_subplot(111, projection='3d')
        
        # 1. Isolate the data for the lowest r value
        min_r = np.min(r_flat)
        mask = (r_flat == min_r)
        
        # 2. Reshape the filtered 1D arrays into 2D grids for plot_surface
        grid_shape = (len(z_vals), len(u_vals))
        
        angles_filtered = (np.atan(u_flat[mask]) * 180 / np.pi).reshape(grid_shape)
        z_filtered = z_flat[mask].reshape(grid_shape)
        err_filtered = err_flat[mask].reshape(grid_shape)
        
        # 3. Create the surface plot
        surf1 = ax1.plot_surface(
            angles_filtered, 
            z_filtered, 
            err_filtered, 
            cmap='viridis', 
            edgecolor='none', 
            alpha=0.9
        )
        
        ax1.set_xlabel(r'$\theta$ (Deg)')
        ax1.set_ylabel(r'Depth $Z$')
        ax1.set_zlabel(r'Error $\mathcal{E}$')
        
        style_3d_ax(ax1)
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

        # 1. Isolate the data for the lowest r value
        min_r = np.min(r_flat)
        mask = (r_flat == min_r)
        grid_shape = (len(z_vals), len(u_vals))
        
        # 2. Reshape into 2D grids
        angles_filtered = (np.atan(u_flat[mask]) * 180 / np.pi).reshape(grid_shape)
        z_filtered = z_flat[mask].reshape(grid_shape)
        
        err_filtered = err_flat[mask].reshape(grid_shape)

        # Number of contour levels for a smooth appearance
        smooth_levels = 50

        map1 = ax1.contourf(
            angles_filtered, z_filtered - z_bar, err_filtered, 
            levels=smooth_levels, cmap='viridis'
        )

        ax1.set_xlabel(r'$\theta$ (Deg)')
        ax1.set_ylabel(r'$\delta$z')
        
        cb1 = fig1.colorbar(map1, ax=ax1, pad=0.05)
        # cb1.set_label(r'Error $\mathcal{E}$', rotation=90, labelpad=20)
        
        fig1.tight_layout()
        fig1.savefig(f'{name}_error_2d_map.png')
    
    plot_2d_map_error(err_affine_flat, 'affine')
    plot_2d_map_error(err_ortho_flat, 'ortho')
    # plot_2d_map_error(err_hybrid_flat, 'hybrid')
    plt.show()

def visual_hull_tau_vs_visual_hull_ghost():
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    # ---------------------------------------------------------
    # 1. Academic Styling Configuration
    # ---------------------------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times", "Times New Roman"],
        "mathtext.fontset": "cm", 
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
    })

    # ---------------------------------------------------------
    # 2. Setup Camera Positions and Swarm Points
    # ---------------------------------------------------------
    c1 = np.array([0.0, 1.0])   
    c2 = np.array([12.0, 1.0])  

    points = np.array([
        [3.6, 4.7], [4.2, 5.9], [4.9, 7.5], [4.6, 4.1],
        [5.2, 5.1], [5.9, 6.5], [6.6, 8.2], [5.6, 3.7],
        [6.2, 4.7], [7.0, 6.1], [7.8, 7.5], [6.9, 3.2],
        [7.5, 4.5], [8.4, 5.7]
    ])

    # ---------------------------------------------------------
    # 3. Define the High-Resolution 2D Spatial Grid
    # ---------------------------------------------------------
    res = 1200 
    x_min, x_max = -1, 13
    y_min, y_max = 0, 11.5 

    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)
    X, Y = np.meshgrid(x, y)
    grid_pts = np.c_[X.ravel(), Y.ravel()]

    # ---------------------------------------------------------
    # 4. Compute Projections and Visual Hull Sets
    # ---------------------------------------------------------
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

        in_any_cone1 = in_any_cone1 | cone1_i
        in_any_cone2 = in_any_cone2 | cone2_i
        vh_neigh = vh_neigh | (cone1_i & cone2_i)

    vh_tau = in_any_cone1 & in_any_cone2
    vh_ghost = vh_tau & (~vh_neigh)

    VH_neigh_img = vh_neigh.reshape(X.shape)
    VH_ghost_img = vh_ghost.reshape(X.shape)

    # ---------------------------------------------------------
    # 5. Professional Plotting & Aesthetics
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # Background color
    # fig.patch.set_facecolor('#F8F9FA')
    # ax.set_facecolor('#F8F9FA')

    # Add a faint blueprint dot grid for spatial context
    # gx, gy = np.meshgrid(np.arange(0, 13, 0.5), np.arange(0, 12, 0.5))
    # ax.scatter(gx, gy, color='#E5E7EB', s=2, zorder=0)

    # Color and Alpha definitions
    color_ghost = '#E78875'  
    color_neigh = '#7CB1A1'  
    color_rays = '#6B7A93'   

    fill_alpha = 0.35
    hatch_alpha = 0.85

    # --- Region 1: Ghosts (Spurious Geometries) ----
    # Layer A: Semi-transparent solid fill
    ax.contourf(X, Y, VH_ghost_img, levels=[0.5, 1.5], colors=[color_ghost], alpha=fill_alpha, zorder=2)
    # Layer B: High-opacity colored hatches (using rc_context to set hatch color dynamically)
    with plt.rc_context({'hatch.color': color_ghost, 'hatch.linewidth': 0.8}):
        ax.contourf(X, Y, VH_ghost_img, levels=[0.5, 1.5], colors=[(1, 1, 1, 0)], hatches=['////'], alpha=hatch_alpha, zorder=3)
    # Layer C: Crisp outline
    ax.contour(X, Y, VH_ghost_img, levels=[0.5], colors=[color_ghost], linewidths=1.5, alpha=1.0, zorder=4)

    # --- Region 2: Neighborhood (Actual Set) ---
    # Layer A: Semi-transparent solid fill
    ax.contourf(X, Y, VH_neigh_img, levels=[0.5, 1.5], colors=[color_neigh], alpha=fill_alpha, zorder=5)
    # Layer B: High-opacity colored hatches (using opposite diagonal pattern for visual distinction)
    with plt.rc_context({'hatch.color': color_neigh, 'hatch.linewidth': 0.8}):
        ax.contourf(X, Y, VH_neigh_img, levels=[0.5, 1.5], colors=[(1, 1, 1, 0)], hatches=['\\\\\\\\'], alpha=hatch_alpha, zorder=6)
    # Layer C: Crisp outline
    ax.contour(X, Y, VH_neigh_img, levels=[0.5], colors=[color_neigh], linewidths=1.5, alpha=1.0, zorder=7)

    # Plot actual agents with a glow effect
    # ax.scatter(points[:, 0], points[:, 1], c=color_neigh, edgecolor='None', s=250, alpha=0.3, zorder=8)
    ax.scatter(points[:, 0], points[:, 1], c='#1A1A1A', edgecolor='white', linewidth=1.2, s=70, zorder=9, label='Actual Agents ($p_i$)')

    # --------------------------------------------------------
    # 6. Draw Visual Cones, Cameras, and Individual Rays
    # ---------------------------------------------------------
    min_t1, max_t1 = np.min(theta_p1) - tau, np.max(theta_p1) + tau
    min_t2, max_t2 = np.min(theta_p2) - tau, np.max(theta_p2) + tau
    ray_length = 20 

    # Outer Frustum bounds
    ax.plot([c1[0], c1[0] + ray_length * np.cos(min_t1)], [c1[1], c1[1] + ray_length * np.sin(min_t1)], color=color_rays, lw=2, zorder=1)
    ax.plot([c1[0], c1[0] + ray_length * np.cos(max_t1)], [c1[1], c1[1] + ray_length * np.sin(max_t1)], color=color_rays, lw=2, zorder=1)
    ax.plot([c2[0], c2[0] + ray_length * np.cos(min_t2)], [c2[1], c2[1] + ray_length * np.sin(min_t2)], color=color_rays, lw=2, zorder=1)
    ax.plot([c2[0], c2[0] + ray_length * np.cos(max_t2)], [c2[1], c2[1] + ray_length * np.sin(max_t2)], color=color_rays, lw=2, zorder=1)

    # Individual connecting rays
    for p in points:
        ax.plot([c1[0], p[0]], [c1[1], p[1]], color=color_rays, alpha=0.15, lw=1.0, zorder=1)
        ax.plot([c2[0], p[0]], [c2[1], p[1]], color=color_rays, alpha=0.15, lw=1.0, zorder=1)

    # Plot Camera Centers
    ax.plot(c1[0], c1[1], marker='s', color='#1A1A1A', markersize=10, markeredgecolor='white', markeredgewidth=1.5, zorder=10)
    ax.plot(c2[0], c2[1], marker='s', color='#1A1A1A', markersize=10, markeredgecolor='white', markeredgewidth=1.5, zorder=10)

    # Camera Labels
    # ax.text(c1[0] - 0.4, c1[1] + 0.1, '$C_1$', fontsize=16, fontweight='bold', ha='right', color='#1A1A1A')
    # ax.text(c2[0] + 0.4, c2[1] + 0.1, '$C_2$', fontsize=16, fontweight='bold', ha='left', color='#1A1A1A')

    # ---------------------------------------------------------
    # 7. Annotations and Legends
    # ---------------------------------------------------------
    # Convert hex colors to precise RGBA tuples to perfectly replicate the separated alphas in the legend patches
    fc_ghost = mcolors.to_rgba(color_ghost, alpha=fill_alpha)
    ec_ghost = mcolors.to_rgba(color_ghost, alpha=hatch_alpha)

    fc_neigh = mcolors.to_rgba(color_neigh, alpha=fill_alpha)
    ec_neigh = mcolors.to_rgba(color_neigh, alpha=hatch_alpha)

    # legend_elements = [
    #     Patch(facecolor=fc_neigh, edgecolor=ec_neigh, hatch='\\\\\\\\', linewidth=1.5, label='$VH_{neigh}$ (Neighborhood Set)'),
    #     Patch(facecolor=fc_ghost, edgecolor=ec_ghost, hatch='////', linewidth=1.5, label='$VH_{ghost}$ (Spurious Geometries)'),
    #     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1A1A1A', markeredgecolor='white', markersize=10, label='Actual Agents ($p_i$)')
    # ]
    # ax.legend(handles=legend_elements, loc='upper left', fontsize=13, framealpha=0.95, edgecolor='#DDDDDD')

    # ---------------------------------------------------------
    # 8. Framing and Final Output
    # ---------------------------------------------------------
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(0.0, 11.5)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(f"figs/VH_diagram.png", transparent=True, bbox_inches='tight')
    plt.show()

def run_geometric_visual_hulls():
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import Patch

    # ==========================================
    # 1. Static Setup & Data Loading (From your snippet)
    # ==========================================
    run_params = DATASET_RUNS[0]
    name = run_params['name']
    start_step = run_params['start_step']
    end_step = run_params['end_step']
    step_length = run_params['step_length']

    scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
    config_path = os.path.join(scenario_path, "config.yaml")

    config = SimulationConfig(config_path) 
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    max_steps = dataset.trajectories.shape[0]
    effective_end_step = end_step if end_step is not None and end_step <= max_steps else max_steps
    step_range = range(start_step, effective_end_step, step_length)
    
    gt_data = np.load(scenario_path + '/reconstruction_scale.npz')
    gt_scales = gt_data['scales_gt']

    idx = 5
    time_step = step_range[idx]
    positions = dataset.positions_at_time_step(time_step)
    N = positions.shape[0]

    # Generate representative camera positions
    cam_positions, _ = generate_encircling_cameras(
        dataset, step_range, config.intrinsics_params, config.H, config.W, cam_num=4, padding=1
    )
    
    swarm_center = np.mean(positions, axis=0)

    # ==========================================
    # 2. Academic Styling Configuration
    # ==========================================
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times", "Times New Roman"],
        "mathtext.fontset": "cm", 
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
    })

    # ==========================================
    # 3. Define the High-Resolution 3D Voxel Grid
    # ==========================================
    grid_res = 60 # Resolution of the voxel grid (e.g., 60x60x60)
    padding = 2.0 # Spatial padding around the swarm
    
    min_pt = np.min(positions, axis=0) - padding
    max_pt = np.max(positions, axis=0) + padding

    x = np.linspace(min_pt[0], max_pt[0], grid_res)
    y = np.linspace(min_pt[1], max_pt[1], grid_res)
    z = np.linspace(min_pt[2], max_pt[2], grid_res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    grid_pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    V = grid_pts.shape[0]
    K = cam_positions.shape[0]

    # ==========================================
    # 4. Compute 3D Projections via Ray Angles
    # ==========================================
    tau = 0.02 # Tolerance angle in radians
    cos_tau = np.cos(tau)

    in_vh_tau = np.ones(V, dtype=bool) 
    in_vh_neigh_agents = np.ones((V, N), dtype=bool) 

    print(f"Evaluating Voxel Grid for time step {time_step}...")
    for k in range(K):
        c = cam_positions[k]
        
        # Unit vectors from camera to grid voxels
        vec_g = grid_pts - c
        norm_g = np.linalg.norm(vec_g, axis=1, keepdims=True)
        dir_g = vec_g / norm_g
        
        # Unit vectors from camera to agents
        vec_p = positions - c
        norm_p = np.linalg.norm(vec_p, axis=1, keepdims=True)
        dir_p = vec_p / norm_p
        
        # Compute angles (cos theta)
        cos_theta = np.dot(dir_g, dir_p.T)
        
        # Check tolerance
        cone_mask = cos_theta >= cos_tau
        
        # VH_tau condition: At least one agent is inside the cone for this camera
        in_vh_tau &= np.any(cone_mask, axis=1)
        
        # VH_neigh condition: Track specific agents across all cameras
        in_vh_neigh_agents &= cone_mask

    # Final Set decomposition
    vh_neigh = np.any(in_vh_neigh_agents, axis=1)
    vh_ghost = in_vh_tau & (~vh_neigh)

    VH_neigh_img = vh_neigh.reshape(X.shape)
    VH_ghost_img = vh_ghost.reshape(X.shape)
    VH_tau_img = in_vh_tau.reshape(X.shape)

    # Calculate Volume Ratio
    vol_tau = np.sum(in_vh_tau)
    vol_neigh = np.sum(vh_neigh)
    vol_ghost = np.sum(vh_ghost)
    ratio = vol_neigh / vol_tau if vol_tau > 0 else 0

    print(f"VH_tau Voxels: {vol_tau} | VH_neigh Voxels: {vol_neigh} | Ghost Voxels: {vol_ghost}")
    print(f"Ratio VH_neigh / VH_tau: {ratio:.4f}")

    # ==========================================
    # 5. Professional 3D Plotting
    # ==========================================
    color_ghost = '#E78875'  
    color_neigh = '#7CB1A1'  
    fill_alpha = 0.35

    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(r"3D Decomposition of $\mathcal{VH}_{\tau}$", pad=20)

    # Combine colors for the voxel map
    colors = np.empty(X.shape, dtype=object)
    colors[VH_neigh_img] = color_neigh
    colors[VH_ghost_img] = color_ghost

    # --- FIX: Create (N+1) Edge Grids for ax.voxels ---
    dx = (max_pt[0] - min_pt[0]) / (grid_res - 1)
    dy = (max_pt[1] - min_pt[1]) / (grid_res - 1)
    dz = (max_pt[2] - min_pt[2]) / (grid_res - 1)

    x_edge = np.linspace(min_pt[0] - dx/2, max_pt[0] + dx/2, grid_res + 1)
    y_edge = np.linspace(min_pt[1] - dy/2, max_pt[1] + dy/2, grid_res + 1)
    z_edge = np.linspace(min_pt[2] - dz/2, max_pt[2] + dz/2, grid_res + 1)
    X_edge, Y_edge, Z_edge = np.meshgrid(x_edge, y_edge, z_edge, indexing='ij')
    # --------------------------------------------------

    # Plot voxels using the new Edge grids
    ax.voxels(X_edge, Y_edge, Z_edge, VH_tau_img, facecolors=colors, edgecolor='k', linewidth=0.1, alpha=fill_alpha)

    # Plot actual agents
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
               c='#1A1A1A', edgecolor='white', linewidth=0.8, s=50, zorder=9, label='Actual Agents ($p_i$)')

    # Plot Camera Centers
    ax.scatter(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2], 
               marker='s', color='#1A1A1A', edgecolor='white', s=80, label='Cameras ($C_k$)')

    # Connect cameras to the swarm center to show viewing directions
    for c in cam_positions:
        ax.plot([c[0], swarm_center[0]], [c[1], swarm_center[1]], [c[2], swarm_center[2]], 
                color='#6B7A93', alpha=0.3, linestyle='--', linewidth=1)

    # Aesthetics
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    ax.set_xlim(min_pt[0], max_pt[0])
    ax.set_ylim(min_pt[1], max_pt[1])
    ax.set_zlim(min_pt[2], max_pt[2])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Custom Legend
    legend_elements = [
        Patch(facecolor=color_neigh, alpha=fill_alpha, edgecolor='k', label=r'$\mathcal{VH}_{neigh}$ (Neighborhood Set)'),
        Patch(facecolor=color_ghost, alpha=fill_alpha, edgecolor='k', label=r'$\mathcal{VH}_{ghost}$ (Spurious Geometries)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1A1A1A', markeredgecolor='white', markersize=8, label='Agents ($p_i$)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#1A1A1A', markeredgecolor='white', markersize=8, label='Cameras ($C_k$)')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.95, edgecolor='#DDDDDD')
    
    # Display the ratio directly on the plot
    ax.text2D(0.05, 0.95, f'Ratio ($\mathcal{{VH}}_{{neigh}} / \mathcal{{VH}}_{{\tau}}$): {ratio:.4f}', 
              transform=ax.transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()

def plot_ratio_surface(run_params, scales, cam_nums, base_tau=0.05, idx=5, grid_res=50):
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    """
    Evaluates the ratio (VH_neigh / VH_tau) across a grid of scale factors and camera counts.
    Plots a 3D surface where X=scale, Y=cam_num, Z=ratio.
    
    scales: List or array of scaling factors (e.g., for the tolerance tau)
    cam_nums: List or array of camera counts (e.g., [2, 3, 4, 5, 6])
    """
    # ==========================================
    # 1. Data Loading & Setup
    # ==========================================
    name = run_params['name']
    start_step = run_params['start_step']
    end_step = run_params['end_step']
    step_length = run_params['step_length']

    scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
    config_path = os.path.join(scenario_path, "config.yaml")

    # Assuming these are available in your global scope
    config = SimulationConfig(config_path) 
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    max_steps = dataset.trajectories.shape[0]
    effective_end_step = end_step if end_step is not None and end_step <= max_steps else max_steps
    step_range = range(start_step, effective_end_step, step_length)
    
    time_step = step_range[idx]
    positions = dataset.positions_at_time_step(time_step)
    N = positions.shape[0]

    # Pre-calculate the Voxel Grid limits based on swarm bounds
    padding = 2.0
    min_pt = np.min(positions, axis=0) - padding
    max_pt = np.max(positions, axis=0) + padding

    x = np.linspace(min_pt[0], max_pt[0], grid_res)
    y = np.linspace(min_pt[1], max_pt[1], grid_res)
    z = np.linspace(min_pt[2], max_pt[2], grid_res)
    X_grid, Y_grid, Z_grid = np.meshgrid(x, y, z, indexing='ij')

    grid_pts = np.vstack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()]).T
    V = grid_pts.shape[0]

    # ==========================================
    # 2. Grid Evaluation Loop
    # ==========================================
    # Initialize Z-axis data matrix for the surface plot
    Ratios = np.zeros((len(cam_nums), len(scales)))

    print(f"Evaluating surface grid ({len(cam_nums)}x{len(scales)} iterations)...")
    
    for i, cam_num in enumerate(cam_nums):
        # Generate cameras for the current cam_num
        cam_positions, _ = generate_encircling_cameras(
            dataset, step_range, config.intrinsics_params, config.H, config.W, cam_num=cam_num, padding=1, is_3d=True
        )
        K = cam_positions.shape[0]
        
        # Pre-compute rays from cameras to grid voxels and agents (Camera layouts are fixed for this 'i' loop)
        grid_rays = []
        agent_rays = []
        for c in cam_positions:
            # Grid rays
            vec_g = grid_pts - c
            dir_g = vec_g / np.linalg.norm(vec_g, axis=1, keepdims=True)
            grid_rays.append(dir_g)
            
            # Agent rays
            vec_p = positions - c
            dir_p = vec_p / np.linalg.norm(vec_p, axis=1, keepdims=True)
            agent_rays.append(dir_p)
            
        for j, scale in enumerate(scales):
            # Define current tau (assuming scale modifies the base tolerance)
            tau_current = base_tau * scale
            cos_tau = np.cos(tau_current)
            
            in_vh_tau = np.ones(V, dtype=bool)
            in_vh_neigh_agents = np.ones((V, N), dtype=bool)

            # Check intersections
            for k in range(K):
                # Cosine similarity between voxel rays and agent rays
                cos_theta = np.dot(grid_rays[k], agent_rays[k].T)
                cone_mask = cos_theta >= cos_tau
                
                in_vh_tau &= np.any(cone_mask, axis=1)
                in_vh_neigh_agents &= cone_mask

            vh_neigh = np.any(in_vh_neigh_agents, axis=1)
            
            # Calculate Volumes & Ratio
            vol_tau = np.sum(in_vh_tau)
            vol_neigh = np.sum(vh_neigh)
            
            ratio = vol_neigh / vol_tau if vol_tau > 0 else 0
            Ratios[i, j] = ratio
            
            print(f"Cam: {cam_num} | Scale: {scale:.2f} | Ratio: {ratio:.4f}")

    # ==========================================
    # 3. 3D Surface Plotting
    # ==========================================
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times"],
        "axes.edgecolor": "#333333",
    })

    fig = plt.figure(figsize=(12, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create the 2D meshgrid for X (Scale) and Y (Cam Num)
    X_surf, Y_surf = np.meshgrid(scales, cam_nums)

    # Plot the surface
    surf = ax.plot_surface(
        X_surf, Y_surf, Ratios, 
        cmap=cm.viridis,          # Color map
        edgecolor='k',            # Wireframe color
        linewidth=0.5, 
        alpha=0.8,
        antialiased=True
    )

    # Labels and Titles
    ax.set_title(r"Ratio $\mathcal{VH}_{neigh} / \mathcal{VH}_{\tau}$ vs Scale & Camera Count", pad=20, fontsize=16)
    ax.set_xlabel("Scale (Tolerance Multiplier)", labelpad=10, fontsize=12)
    ax.set_ylabel("Number of Cameras ($C_k$)", labelpad=10, fontsize=12)
    ax.set_zlabel("Ratio", labelpad=10, fontsize=12)

    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1, label='Ratio')

    # Adjust viewing angle for better visibility
    ax.view_init(elev=30, azim=225)

    plt.tight_layout()
    plt.show()

    return X_surf, Y_surf, Ratios

def dra_metrics():
    import numpy as np
    import matplotlib.pyplot as plt

    def generate_gaussian(x, mu, sigma, amplitude=1.0):
        """Generates a 1D Gaussian curve."""
        return amplitude * np.exp(-0.5 * ((x - mu) / sigma)**2)

    # Set up the domain
    x = np.linspace(-5, 7, 1000)

    # Define the two density fields (Ground Truth and Predicted)
    # Parameters adjusted to create clear overlapping and non-overlapping regions
    D_GT = generate_gaussian(x, mu=0.0, sigma=1.2, amplitude=1.0)  # Ground Truth
    D_P = generate_gaussian(x, mu=1.8, sigma=1.0, amplitude=0.9)   # Predicted

    # Calculate the True Positive boundary (the minimum of both curves at any point)
    TP_boundary = np.minimum(D_GT, D_P)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

    # 1. True Positive Region (Overlap)
    ax.fill_between(x, 0, TP_boundary, 
                    facecolor='#A9DFBF',   # Light Green
                    edgecolor='#1E8449', 
                    hatch='///', 
                    alpha=0.8, 
                    zorder=1)

    # 2. False Negative Region (Missed ground truth)
    # Bounded below by the TP_boundary and above by the Ground Truth curve
    ax.fill_between(x, TP_boundary, D_GT, 
                    facecolor='#AED6F1',   # Light Blue
                    edgecolor='#2874A6', 
                    hatch='\\\\\\', 
                    alpha=0.8, 
                    zorder=1)

    # 3. False Positive Region (Hallucinated mass)
    # Bounded below by the TP_boundary and above by the Predicted curve
    ax.fill_between(x, TP_boundary, D_P, 
                    facecolor='#F5B7B1',   # Light Red
                    edgecolor='#B03A2E', 
                    hatch='xxx', 
                    alpha=0.8, 
                    zorder=1)

    # Draw the solid curves on top
    ax.plot(x, D_GT, color='#2874A6', linestyle='-', linewidth=2.5, zorder=3)
    ax.plot(x, D_P, color='#B03A2E', linestyle='-', linewidth=2.5, zorder=3)

    # Clean up for academic aesthetic (no text, no legend, remove borders)
    ax.axis('off')

    # Optionally, add a simple baseline
    ax.plot([x.min(), x.max()], [0, 0], color='black', linewidth=1.5, zorder=4)

    plt.tight_layout()
    # plt.savefig("tp_fp_fn_gaussians.pdf", format='pdf', bbox_inches='tight')
    plt.show()

def one_frame_parameter_search(n_trials=50):
    import optuna
    # ==========================================
    # 1. ONE-TIME SETUP (Outside the trial loop)
    # ==========================================
    run_params = DATASET_RUNS[0]

    name = run_params['name']
    start_step = run_params['start_step']
    end_step = run_params['end_step']
    step_length = run_params['step_length']

    scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
    config_path = os.path.join(scenario_path, "config.yaml")

    CAM_NUM = 2

    # Load Dataset
    config = SimulationConfig(config_path) 
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    max_steps = dataset.trajectories.shape[0]
    effective_end_step = end_step if end_step is not None and end_step <= max_steps else max_steps
    step_range = range(start_step, effective_end_step, step_length)

    if CAM_NUM == 2:
        cam_positions, cam_radius = generate_encircling_cameras(dataset, step_range, config.intrinsics_params, config.H, config.W, cam_num=4, padding=1)
        cam_poses = np.hstack((cam_positions[:2], np.tile(np.array([1, 0, 0, 0]), (2, 1)))).astype(np.float32)
    else:
        cam_positions, cam_radius = generate_encircling_cameras(dataset, step_range, config.intrinsics_params, config.H, config.W, cam_num=CAM_NUM, padding=1)
        cam_poses = np.hstack((cam_positions, np.tile(np.array([1, 0, 0, 0]), (CAM_NUM, 1)))).astype(np.float32)
    
    # System Initialization    
    cam_system = MultiCameraSystem.create_homogeneous_system(
        state_class=CameraState,
        intrinsics=config.intrinsics_params,
        H=config.H, W=config.W, 
        poses_or_RTs=cam_poses,
        near_clip=config.near_clip, far_clip=200, 
        size=config.size,
        device='cuda')
        
    reconstruction_params = {
        'targetd_num_mode': 10,
        'voxel_scale': 0.5,
        'voxel_peak_threshold': 0.3,
        'voxel_grid_max_size': 32,
        'voxel_peaks_number': 2 * 10
    }
    
    gt_data = np.load(scenario_path + '/reconstruction_scale.npz')
    gt_scales = gt_data['scales_gt']

    idx = 5
    time_step = step_range[idx]

    positions = dataset.positions_at_time_step(time_step)
    poses, projections, _, masks = cam_system.simulate_vision(positions, renderer='projection_only')

    log_file_path = os.path.join(os.getcwd())
    output_dir = os.path.join(log_file_path, f"t_{time_step:03d}")

    # ==========================================
    # 2. OBJECTIVE FUNCTION FOR OPTUNA
    # ==========================================
    def objective(trial):
        # Define the search space for train_params. 
        # Learning rates usually optimize best on a log scale.
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
            'lr_max_steps': 100 # Keeping steps fixed so trials are directly comparable
        }

        density_reconstructor = DensityReconstructor(
            max_iter=train_params['lr_max_steps'], 
            use_decoupled=False
        )
        
        try:
            model, scale_spaces = density_reconstructor.process_frame(
                cam_system, 
                point_sets=projections, 
                positions=positions,
                initGMM=None,
                is_adaptive_scale=False, 
                scale=gt_scales[idx],
                is_store_intermediate=False, # Disabled for speed during optimization
                is_log=False,                # Disabled for speed to prevent I/O bottlenecks
                output_dir=output_dir,
                debug=False,
                train_params=train_params,
                reconstruction_params=reconstruction_params
            )
            
            return model[0].mean_loss
            
        except Exception as e:
            # If a specific parameter combination causes a crash (e.g., CUDA OOM, NaNs), prune the trial
            print(f"Trial pruned due to exception: {e}")
            raise optuna.TrialPruned()

    # ==========================================
    # 3. RUN OPTIMIZATION
    # ==========================================
    study = optuna.create_study(direction="minimize")
    print(f"Starting parameter search for {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)

    print("\n" + "="*30)
    print("BEST PARAMETERS FOUND")
    print("="*30)
    print(f"Minimum Loss: {study.best_value}")
    
    # Generate the ideal dictionary to copy-paste into your main script
    best_train_params = study.best_params
    
    for key, value in best_train_params.items():
        print(f"'{key}': {value},")

    return best_train_params

def one_frame_convergence():
    run_params = DATASET_RUNS[0]

    name = run_params['name']
    start_step = run_params['start_step']
    end_step = run_params['end_step']
    step_length = run_params['step_length']

    scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
    config_path = os.path.join(scenario_path, "config.yaml")

    CAM_NUM = 3

    # 3. Load Dataset
    config = SimulationConfig(config_path) 
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    max_steps = dataset.trajectories.shape[0]
    effective_end_step = end_step if end_step is not None and end_step <= max_steps else max_steps

    step_range = range(start_step, effective_end_step, step_length)

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
        near_clip=config.near_clip, far_clip=200, 
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
    # train_params_old = {
    #     'xyz_lr_c': 0.05,
    #     'xyz_lr_final_c': 0.2,
    #     'radius_lr_c': 0.05,
    #     'radius_lr_final_c': 0.5,
    #     'weights_lr_c': 0.10,
    #     'weights_lr_final_c': 0.5,
    #     'xyz_reg': 0,
    #     'radius_reg': 0,
    #     'radius_cutoff_inv': 0.5,
    #     'lr_max_steps': 1000
    # }
    train_params = {
        'xyz_lr_c': 0.11550156892954913,
        'xyz_lr_final_c': 0.015263086280830469,
        'radius_lr_c': 0.09585436467026787,
        'radius_lr_final_c': 0.02420618007560584,
        'weights_lr_c': 0.19814963583342243,
        'weights_lr_final_c': 0.7979132269720964,
        'xyz_reg': 0.21978381872642633,
        'radius_reg': 0.6083537781516261,
        'radius_cutoff_inv': 0.6013595613763145,
        'lr_max_steps': 100
    }
    density_reconstructor = DensityReconstructor(max_iter=train_params['lr_max_steps'], use_decoupled=False)
    
    gt_data = np.load(scenario_path + '/reconstruction_scale.npz')
    gt_scales = gt_data['scales_gt']

    # 5. Simulation Loop
    idx = 5
    time_step = step_range[idx]

    total_num = []
    positions = dataset.positions_at_time_step(time_step)
    N = positions.shape[0]
    total_num.append(positions.shape[0])
    # poses, _, images, masks = cam_system.simulate_vision(positions, renderer='gaussian')
    poses, projections, _, masks = cam_system.simulate_vision(positions, renderer='projection_only')

    log_file_path = os.path.join(os.getcwd())
    output_dir = os.path.join(log_file_path, f"t_{time_step:03d}")

    model, scale_spaces = \
    density_reconstructor.process_frame(cam_system, point_sets=projections, positions=positions,
                                        initGMM=None,
                                        is_adaptive_scale=False, scale=gt_scales[idx],
                                        is_store_intermediate=True, is_log=True,
                                        output_dir=os.path.join(log_file_path, f"t_{time_step:03d}"),
                                        debug=False,
                                        train_params=train_params,
                                        reconstruction_params=reconstruction_params)

    # training_history = GaussianModel.load_training_history(os.path.join(output_dir, f"checkpoint_level_0.pth"))
    # model = GaussianModel.load_iter(training_history, 99)

    r_means, r_weights, r_covs = GMR.runnalls_algorithm_simple_torch(
        means=torch.from_numpy(positions),
        radii=torch.full((N, 1), gt_scales[idx], device='cuda', dtype=torch.float),
        weights=torch.full((N, 1), 1.0, device='cuda', dtype=torch.float),
        L=20, DEVICE='cuda'
    )
    r_radius = torch.sqrt(r_covs[:, 0, 0]).reshape((-1, 1))

    gmr_images = []
    for camera in cam_system.cameras:
        gmr_images.append(rasterize_gaussians(
            r_means,
            r_radius,
            r_weights,
            camera.state.R,
            camera.state.T,
            camera.state.K,
            camera.state.H,
            camera.state.W,
            False
        ))
    gmr_losses = [torch.sum(scale_spaces[i][0] - gmr_images[i]).item() for i in range(CAM_NUM)]

    # gmm_visualizer = MultiGMMPlotter()
    # gmm_visualizer.add_gmm(model[0]._xyz.detach().cpu().numpy(), model[0]._radius.detach().cpu().numpy(), model[0]._weights.detach().cpu().numpy())
    # gmm_visualizer.update(real_means=positions)
    # move_figure(gmm_visualizer.fig, 100, 100)
    # gmm_visualizer.ax.view_init(elev=33, azim=-117, roll=0)
    # gmm_visualizer.fig.savefig("gmm_diagram.png", transparent=True, bbox_inches='tight')


    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(np.arange(train_params['lr_max_steps'])[::2], model[0].metrics_history['loss_history'][::2])
    # ax.plot(np.arange(train_params['lr_max_steps'])[1::2], model[0].metrics_history['loss_history'][1::2])
    # ax.set_yscale('log')

    print(f"training loss: {model[0].mean_loss}")

    min_coords = np.min(positions, axis=0)
    max_coords = np.max(positions, axis=0)

    bounds = np.vstack((min_coords - 3 * gt_scales[idx], max_coords + 3 * gt_scales[idx])).T # add three sigma padding
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
    # plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern for math rendering
    # plt.rcParams['axes.labelsize'] = 12
    # plt.rcParams['xtick.labelsize'] = 10
    # plt.rcParams['ytick.labelsize'] = 10
    # plt.rcParams['legend.fontsize'] = 10
    # plt.rcParams['figure.dpi'] = 300  # High resolution for print

    # # 1. Initialize Figure (6x4 is a standard aspect ratio that fits well in two-column formats)
    # fig, ax = plt.subplots(figsize=(6, 4))

    # # Extract variables to keep the plotting lines clean
    # steps = np.arange(train_params['lr_max_steps'])
    # loss_history = model[0].metrics_history['loss_history']

    # # 2. Plot the alternating steps
    # # Even steps (Solid line, high contrast blue)
    # ax.plot(steps[::2], loss_history[::2], 
    #         color='#1f77b4', linewidth=1.5, 
    #         label='Loss Cam 1')

    # # Odd steps (Dashed line, contrasting orange, slightly transparent so it doesn't overpower)
    # ax.plot(steps[1::2], loss_history[1::2], 
    #         color='#d62728', linewidth=1.5, linestyle='--', alpha=0.85, 
    #         label='Loss Cam 2')

    # # 3. Axis Formatting
    # ax.set_yscale('log')
    # ax.set_xlabel('Training Step')
    # ax.set_ylabel('Training Loss')

    # # 4. Clean up "Chart Junk"
    # # Remove top and right spines (standard practice in modern data visualization)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    # # Add a subtle grid to help readers track logarithmic values without distracting from the data
    # ax.grid(True, which="major", axis="y", linestyle="-", alpha=0.2)
    # ax.grid(True, which="minor", axis="y", linestyle=":", alpha=0.1)

    # # 5. Legend and Layout
    # # Remove the legend frame to keep it clean
    # ax.legend(frameon=False, loc='best')

    # # Tight layout ensures labels aren't cut off when saving to PDF/EPS
    # fig.tight_layout()
    # plt.savefig('loss_curve.pdf', format='pdf', bbox_inches='tight')

    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111)
    # ax2.plot(np.arange(train_params['lr_max_steps']), model[0].metrics_history['grad_norm_xyz_history'], label='xyz')
    # ax2.plot(np.arange(train_params['lr_max_steps']), model[0].metrics_history['grad_norm_radius_history'], label='radius')
    # ax2.plot(np.arange(train_params['lr_max_steps']), model[0].metrics_history['grad_norm_weights_history'], label='weights')
    # ax2.set_yscale('log')
    # ax2.legend()
    # plt.show()

def one_frame_dMOTA_factor_analysis(force_recalculate=False):
    import pickle

    """
    Args:
        force_recalculate (bool): Set to True to ignore the cache and force a new calculation.
    """
    cache_filename = 'dmota_cache.pkl'
    
    # Parameter spaces
    cam_nums = [2, 3, 5, 7, 9]
    comp_nums = list(range(10, 41))

    # ==========================================
    # 0. Check for Cached Data
    # ==========================================
    if not force_recalculate and os.path.exists(cache_filename):
        print(f"✅ Found cached data in '{cache_filename}'. Loading results...")
        with open(cache_filename, 'rb') as f:
            cached_data = pickle.load(f)
            gmr_dmota_results = cached_data['gmr']
            model_dmota_results = cached_data['model']
            
    else:
        print("⏳ No cache found (or recalculation forced). Starting heavy computations...")
        
        # ==========================================
        # 1. Static Setup & Data Loading
        # ==========================================
        run_params = DATASET_RUNS[0]
        name = run_params['name']
        start_step = run_params['start_step']
        end_step = run_params['end_step']
        step_length = run_params['step_length']

        scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
        config_path = os.path.join(scenario_path, "config.yaml")

        config = SimulationConfig(config_path) 
        factory = DatasetFactory()
        dataset = factory.get_dataset(config.data_file)

        max_steps = dataset.trajectories.shape[0]
        effective_end_step = end_step if end_step is not None and end_step <= max_steps else max_steps
        step_range = range(start_step, effective_end_step, step_length)
        
        gt_data = np.load(scenario_path + '/reconstruction_scale.npz')
        gt_scales = gt_data['scales_gt']

        idx = 5
        time_step = step_range[idx]
        positions = dataset.positions_at_time_step(time_step)
        N = positions.shape[0]
        
        # Pre-calculate bounds and voxel_res for metrics
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        bounds = np.vstack((min_coords - 3 * gt_scales[idx], max_coords + 3 * gt_scales[idx])).T
        voxel_res = np.max(max_coords - min_coords) * 5e-3

        train_params = {
            'xyz_lr_c': 0.11550156892954913,
            'xyz_lr_final_c': 0.015263086280830469,
            'radius_lr_c': 0.09585436467026787,
            'radius_lr_final_c': 0.02420618007560584,
            'weights_lr_c': 0.19814963583342243,
            'weights_lr_final_c': 0.7979132269720964,
            'xyz_reg': 0.21978381872642633,
            'radius_reg': 0.6083537781516261,
            'radius_cutoff_inv': 0.6013595613763145,
            'lr_max_steps': 1000
        }

        reconstruction_params_base = {
            'targetd_num_mode': 10,
            'voxel_scale': 0.5,
            'voxel_peak_threshold': 0.3,
            'voxel_grid_max_size': 32,
        }

        log_file_path = os.getcwd()

        # ==========================================
        # 2. Evaluate Baseline GMR
        # ==========================================
        print("Evaluating baseline GMR model...")
        gmr_dmota_results = []
        
        for comp_num in comp_nums:
            r_means, r_weights, r_covs = GMR.runnalls_algorithm_simple_torch(
                means=torch.from_numpy(positions),
                radii=torch.full((N, 1), gt_scales[idx], device='cuda', dtype=torch.float),
                weights=torch.full((N, 1), 1.0, device='cuda', dtype=torch.float),
                L=comp_num, # L equals component_number
                DEVICE='cuda'
            )
            r_radius = torch.sqrt(r_covs[:, 0, 0]).reshape((-1, 1))

            _, total_fp_mass, total_fn_mass = compute_metrics_batched_torch(
                means1_np=positions, sigma1=gt_scales[idx], 
                pred_means=r_means, pred_weights=r_weights, pred_sigmas=r_radius,
                bounds=bounds, voxel_res=voxel_res, batch_size=50000, device='cuda'
            )
            
            gmr_dmota = 1 - (total_fn_mass + total_fp_mass) / N
            gmr_dmota_results.append(gmr_dmota)

        # ==========================================
        # 3. Evaluate Main Model over CAM_NUMs
        # ==========================================
        model_dmota_results = {cam_num: [] for cam_num in cam_nums}

        for cam_num in cam_nums:
            print(f"Testing CAM_NUM = {cam_num}...")
            
            # Initialize camera system for this CAM_NUM
            if cam_num == 2:
                cam_positions, cam_radius = generate_encircling_cameras(dataset, step_range, config.intrinsics_params, config.H, config.W, cam_num=4, padding=1)
                cam_poses = np.hstack((cam_positions[:2], np.tile(np.array([1, 0, 0, 0]), (2, 1)))).astype(np.float32)
            else:
                cam_positions, cam_radius = generate_encircling_cameras(dataset, step_range, config.intrinsics_params, config.H, config.W, cam_num=cam_num, padding=1)
                cam_poses = np.hstack((cam_positions, np.tile(np.array([1, 0, 0, 0]), (cam_num, 1)))).astype(np.float32)

            cam_system = MultiCameraSystem.create_homogeneous_system(
                state_class=CameraState,
                intrinsics=config.intrinsics_params,
                H=config.H, W=config.W, 
                poses_or_RTs=cam_poses,
                near_clip=config.near_clip, far_clip=200, 
                size=config.size,
                device='cuda'
            )
            
            poses, projections, _, masks = cam_system.simulate_vision(positions, renderer='projection_only')
            density_reconstructor = DensityReconstructor(max_iter=train_params['lr_max_steps'], use_decoupled=False)

            for comp_num in comp_nums:
                reconstruction_params = reconstruction_params_base.copy()
                reconstruction_params['voxel_peaks_number'] = comp_num
                
                model, scale_spaces = density_reconstructor.process_frame(
                    cam_system, point_sets=projections, positions=positions,
                    initGMM=None,
                    is_adaptive_scale=False, scale=gt_scales[idx],
                    is_store_intermediate=True, is_log=True,
                    output_dir=os.path.join(log_file_path, f"t_{time_step:03d}"),
                    debug=False,
                    train_params=train_params,
                    reconstruction_params=reconstruction_params
                )

                _, total_fp_mass, total_fn_mass = compute_metrics_batched_torch(
                    means1_np=positions, sigma1=gt_scales[idx], 
                    pred_means=model[0]._xyz, pred_weights=model[0]._weights, pred_sigmas=model[0]._radius,
                    bounds=bounds, voxel_res=voxel_res, batch_size=50000, device='cuda'
                )

                model_dmota = 1 - (total_fn_mass + total_fp_mass) / N
                model_dmota_results[cam_num].append(model_dmota)
        
        # Save results to cache after calculation
        print(f"💾 Saving computed results to '{cache_filename}'...")
        with open(cache_filename, 'wb') as f:
            pickle.dump({
                'gmr': gmr_dmota_results, 
                'model': model_dmota_results
            }, f)

    # ==========================================
    # 4. Plotting
    # ==========================================
    print("🎨 Generating plot...")
    plt.figure(figsize=(6, 4))
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern for math rendering
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 300  # High resolution for print

    plt.plot(comp_nums, gmr_dmota_results, 
             label='GMR-2', 
             color='black', linewidth=2.5, linestyle='--', zorder=10)

    colormap = plt.cm.get_cmap('viridis', len(cam_nums))
    for i, cam_num in enumerate(cam_nums):
        plt.plot(comp_nums, model_dmota_results[cam_num], 
                 label=f'Ours-{cam_num}', 
                 color=colormap(i), marker='o', markersize=4, linewidth=1.5)

    # plt.title('dMOTA vs Component Number / L')
    plt.xlabel('Component Number')
    plt.ylabel('DRA')
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(True, which="major", axis="y", linestyle="-", alpha=0.3)
    
    plt.legend(loc='lower right', ncol=2, frameon=False)
    plt.tight_layout()
    
    plt.savefig('dmota_comparison.png', bbox_inches='tight')
    plt.show()


def one_frame_dMOTA_factor_analysis_2(force_recalculate=False):
    import pickle

    """
    Args:
        force_recalculate (bool): Set to True to ignore the cache and force a new calculation.
    """
    cache_filename = 'dmota_cache_2.pkl'
    
    # Parameter spaces
    cam_nums = [2, 3, 5, 7, 9]
    # We now explicitly iterate over the number of modes
    target_modes = list(range(5, 26))
    fixed_comp_nums = 20

    # ==========================================
    # 0. Check for Cached Data
    # ==========================================
    if not force_recalculate and os.path.exists(cache_filename):
        print(f"✅ Found cached data in '{cache_filename}'. Loading results...")
        with open(cache_filename, 'rb') as f:
            cached_data = pickle.load(f)
            gmr_dmota_results = cached_data['gmr']
            model_dmota_results = cached_data['model']
            computed_scales = cached_data['scales']
            
    else:
        print("⏳ No cache found (or recalculation forced). Starting heavy computations...")
        
        # ==========================================
        # 1. Static Setup & Data Loading
        # ==========================================
        run_params = DATASET_RUNS[0]
        name = run_params['name']
        start_step = run_params['start_step']
        end_step = run_params['end_step']
        step_length = run_params['step_length']

        scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
        config_path = os.path.join(scenario_path, "config.yaml")

        config = SimulationConfig(config_path) 
        factory = DatasetFactory()
        dataset = factory.get_dataset(config.data_file)

        max_steps = dataset.trajectories.shape[0]
        effective_end_step = end_step if end_step is not None and end_step <= max_steps else max_steps
        step_range = range(start_step, effective_end_step, step_length)
        
        idx = 5
        time_step = step_range[idx]
        positions = dataset.positions_at_time_step(time_step)
        N = positions.shape[0]
        
        # Pre-calculate base bounds elements 
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        voxel_res = np.max(max_coords - min_coords) * 5e-3

        train_params = {
            'xyz_lr_c': 0.11550156892954913,
            'xyz_lr_final_c': 0.015263086280830469,
            'radius_lr_c': 0.09585436467026787,
            'radius_lr_final_c': 0.02420618007560584,
            'weights_lr_c': 0.19814963583342243,
            'weights_lr_final_c': 0.7979132269720964,
            'xyz_reg': 0.21978381872642633,
            'radius_reg': 0.6083537781516261,
            'radius_cutoff_inv': 0.6013595613763145,
            'lr_max_steps': 1000
        }

        reconstruction_params_base = {
            'voxel_scale': 0.5,
            'voxel_peak_threshold': 0.3,
            'voxel_grid_max_size': 32,
        }

        log_file_path = os.getcwd()

        # ==========================================
        # 1.5 Calculate Exact Scales for Target Modes
        # ==========================================
        print("Calculating exact scales to hit target modes...")
        pos_gpu = torch.from_numpy(positions).cuda().float()
        nn_dist = torch.cdist(pos_gpu, pos_gpu) + torch.eye(pos_gpu.shape[0], device='cuda') * 1e10
        avg_nn_dist = torch.median(torch.min(nn_dist, dim=1).values).item()
        
        f = lambda s: mode_counting(pos_gpu, pos_gpu.clone(), s, max_iter=2000, tol=avg_nn_dist*5e-4)
        
        computed_scales = []
        for tm in target_modes:
            # Using your provided function to find the exact scale
            exact_scale = find_target_scale(f, tm, 0, 5)
            computed_scales.append(exact_scale)

        # ==========================================
        # 2. Evaluate Baseline GMR over Target Modes
        # ==========================================
        print("Evaluating baseline GMR model across target modes...")
        gmr_dmota_results = []
        
        for tm, current_scale in zip(target_modes, computed_scales):
            bounds = np.vstack((min_coords - 3 * current_scale, max_coords + 3 * current_scale)).T

            r_means, r_weights, r_covs = GMR.runnalls_algorithm_simple_torch(
                means=torch.from_numpy(positions),
                radii=torch.full((N, 1), current_scale, device='cuda', dtype=torch.float),
                weights=torch.full((N, 1), 1.0, device='cuda', dtype=torch.float),
                L=fixed_comp_nums, # Maintained your 2x ratio from the original snippet
                DEVICE='cuda'
            )
            r_radius = torch.sqrt(r_covs[:, 0, 0]).reshape((-1, 1))

            _, total_fp_mass, total_fn_mass = compute_metrics_batched_torch(
                means1_np=positions, sigma1=current_scale, 
                pred_means=r_means, pred_weights=r_weights, pred_sigmas=r_radius,
                bounds=bounds, voxel_res=voxel_res, batch_size=50000, device='cuda'
            )
            
            gmr_dmota = 1 - (total_fn_mass + total_fp_mass) / N
            gmr_dmota_results.append(gmr_dmota)

        # ==========================================
        # 3. Evaluate Main Model over CAM_NUMs and Target Modes
        # ==========================================
        model_dmota_results = {cam_num: [] for cam_num in cam_nums}

        for cam_num in cam_nums:
            print(f"Testing CAM_NUM = {cam_num}...")
            
            if cam_num == 2:
                cam_positions, cam_radius = generate_encircling_cameras(dataset, step_range, config.intrinsics_params, config.H, config.W, cam_num=4, padding=1)
                cam_poses = np.hstack((cam_positions[:2], np.tile(np.array([1, 0, 0, 0]), (2, 1)))).astype(np.float32)
            else:
                cam_positions, cam_radius = generate_encircling_cameras(dataset, step_range, config.intrinsics_params, config.H, config.W, cam_num=cam_num, padding=1)
                cam_poses = np.hstack((cam_positions, np.tile(np.array([1, 0, 0, 0]), (cam_num, 1)))).astype(np.float32)

            cam_system = MultiCameraSystem.create_homogeneous_system(
                state_class=CameraState,
                intrinsics=config.intrinsics_params,
                H=config.H, W=config.W, 
                poses_or_RTs=cam_poses,
                near_clip=config.near_clip, far_clip=200, 
                size=config.size,
                device='cuda'
            )
            
            poses, projections, _, masks = cam_system.simulate_vision(positions, renderer='projection_only')
            density_reconstructor = DensityReconstructor(max_iter=train_params['lr_max_steps'], use_decoupled=False)

            for tm, current_scale in zip(target_modes, computed_scales):
                bounds = np.vstack((min_coords - 3 * current_scale, max_coords + 3 * current_scale)).T
                
                # Update params dynamically for this target mode
                reconstruction_params = reconstruction_params_base.copy()
                reconstruction_params['targetd_num_mode'] = tm
                reconstruction_params['voxel_peaks_number'] = fixed_comp_nums
                
                model, scale_spaces = density_reconstructor.process_frame(
                    cam_system, point_sets=projections, positions=positions,
                    initGMM=None,
                    is_adaptive_scale=False, scale=current_scale,
                    is_store_intermediate=True, is_log=True,
                    output_dir=os.path.join(log_file_path, f"t_{time_step:03d}"),
                    debug=False,
                    train_params=train_params,
                    reconstruction_params=reconstruction_params 
                )

                _, total_fp_mass, total_fn_mass = compute_metrics_batched_torch(
                    means1_np=positions, sigma1=current_scale, 
                    pred_means=model[0]._xyz, pred_weights=model[0]._weights, pred_sigmas=model[0]._radius,
                    bounds=bounds, voxel_res=voxel_res, batch_size=50000, device='cuda'
                )

                model_dmota = 1 - (total_fn_mass + total_fp_mass) / N
                model_dmota_results[cam_num].append(model_dmota)
        
        # Save results to cache 
        print(f"💾 Saving computed results to '{cache_filename}'...")
        with open(cache_filename, 'wb') as f:
            pickle.dump({
                'gmr': gmr_dmota_results, 
                'model': model_dmota_results,
                'scales': computed_scales
            }, f)

    # ==========================================
    # 4. Plotting
    # ==========================================
    print("🎨 Generating plot...")
    plt.figure(figsize=(6, 4))
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'cm'  
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 300  

    # X-axis is now the perfectly evenly spaced target_modes
    plt.plot(target_modes, gmr_dmota_results, 
             label='GMR-2', 
             color='black', linewidth=2.5, linestyle='--', zorder=10)

    colormap = plt.cm.get_cmap('viridis', len(cam_nums))
    for i, cam_num in enumerate(cam_nums):
        plt.plot(target_modes, model_dmota_results[cam_num], 
                 label=f'Ours-{cam_num}', 
                 color=colormap(i), marker='o', markersize=4, linewidth=1.5)

    plt.xlabel('Number of Modes')
    plt.ylabel('DRA')
    
    # Force X-axis ticks to match our target modes
    plt.xticks(target_modes[::2]) # Displaying every 2nd tick to prevent crowding (5, 7, 9...)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(True, which="major", axis="y", linestyle="-", alpha=0.3)
    
    plt.legend(loc='lower right', ncol=2, frameon=False)
    plt.tight_layout()
    
    plt.savefig('dra_target_modes_comparison.png', bbox_inches='tight')
    plt.show()

def one_frame_dMOTA_noise(force_recalculate=False):
    import pickle

    """
    Args:
        force_recalculate (bool): Set to True to ignore the cache and force a new calculation.
    """
    import pickle

    cache_filename = 'dra_noise_variance_cache.pkl'
    
    # Parameter spaces
    cam_nums = [2, 3, 5, 7, 9]
    noise_levels = list(range(5, 21, 3)) # [5, 8, 11, 14, 17, 20]
    num_trials = 10
    
    # Fixed parameters for this exploration
    fixed_comp_num = 20 

    # ==========================================
    # 0. Check for Cached Data
    # ==========================================
    if not force_recalculate and os.path.exists(cache_filename):
        print(f"✅ Found cached data in '{cache_filename}'. Loading results...")
        with open(cache_filename, 'rb') as f:
            cached_data = pickle.load(f)
            gmr_dmota = cached_data['gmr']
            model_dmota_raw = cached_data['model_raw']
            
    else:
        print("⏳ No cache found (or recalculation forced). Starting heavy computations...")
        
        # ==========================================
        # 1. Static Setup & Data Loading
        # ==========================================
        run_params = DATASET_RUNS[0]
        name = run_params['name']
        start_step = run_params['start_step']
        end_step = run_params['end_step']
        step_length = run_params['step_length']

        scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
        config_path = os.path.join(scenario_path, "config.yaml")

        config = SimulationConfig(config_path) 
        factory = DatasetFactory()
        dataset = factory.get_dataset(config.data_file)

        max_steps = dataset.trajectories.shape[0]
        effective_end_step = end_step if end_step is not None and end_step <= max_steps else max_steps
        step_range = range(start_step, effective_end_step, step_length)
        
        gt_data = np.load(scenario_path + '/reconstruction_scale.npz')
        gt_scales = gt_data['scales_gt']

        idx = 5
        time_step = step_range[idx]
        positions = dataset.positions_at_time_step(time_step)
        N = positions.shape[0]
        
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        current_scale = gt_scales[idx]
        bounds = np.vstack((min_coords - 3 * current_scale, max_coords + 3 * current_scale)).T
        voxel_res = np.max(max_coords - min_coords) * 5e-3

        train_params = {
            'xyz_lr_c': 0.11550156892954913,
            'xyz_lr_final_c': 0.015263086280830469,
            'radius_lr_c': 0.09585436467026787,
            'radius_lr_final_c': 0.02420618007560584,
            'weights_lr_c': 0.19814963583342243,
            'weights_lr_final_c': 0.7979132269720964,
            'xyz_reg': 0.21978381872642633,
            'radius_reg': 0.6083537781516261,
            'radius_cutoff_inv': 0.6013595613763145,
            'lr_max_steps': 100
        }

        reconstruction_params_base = {
            'targetd_num_mode': 10,
            'voxel_scale': 0.5,
            'voxel_peak_threshold': 0.3,
            'voxel_grid_max_size': 32,
            'voxel_peaks_number': fixed_comp_num 
        }

        log_file_path = os.getcwd()

        # ==========================================
        # 2. Evaluate Baseline GMR (Calculated Once)
        # ==========================================
        print("Evaluating baseline GMR model (Invariant to 2D noise)...")
        r_means, r_weights, r_covs = GMR.runnalls_algorithm_simple_torch(
            means=torch.from_numpy(positions),
            radii=torch.full((N, 1), current_scale, device='cuda', dtype=torch.float),
            weights=torch.full((N, 1), 1.0, device='cuda', dtype=torch.float),
            L=fixed_comp_num,
            DEVICE='cuda'
        )
        r_radius = torch.sqrt(r_covs[:, 0, 0]).reshape((-1, 1))

        _, total_fp_mass, total_fn_mass = compute_metrics_batched_torch(
            means1_np=positions, sigma1=current_scale, 
            pred_means=r_means, pred_weights=r_weights, pred_sigmas=r_radius,
            bounds=bounds, voxel_res=voxel_res, batch_size=50000, device='cuda'
        )
        
        gmr_dmota = 1 - (total_fn_mass + total_fp_mass) / N

        # ==========================================
        # 3. Evaluate Main Model over Trials
        # ==========================================
        # Structure: {cam_num: {noise_std: [trial_1_score, trial_2_score, ...]}}
        model_dmota_raw = {cam_num: {n: [] for n in noise_levels} for cam_num in cam_nums}

        for cam_num in cam_nums:
            print(f"\nTesting CAM_NUM = {cam_num}...")
            
            if cam_num == 2:
                cam_positions, cam_radius = generate_encircling_cameras(dataset, step_range, config.intrinsics_params, config.H, config.W, cam_num=4, padding=1)
                cam_poses = np.hstack((cam_positions[:2], np.tile(np.array([1, 0, 0, 0]), (2, 1)))).astype(np.float32)
            else:
                cam_positions, cam_radius = generate_encircling_cameras(dataset, step_range, config.intrinsics_params, config.H, config.W, cam_num=cam_num, padding=1)
                cam_poses = np.hstack((cam_positions, np.tile(np.array([1, 0, 0, 0]), (cam_num, 1)))).astype(np.float32)

            cam_system = MultiCameraSystem.create_homogeneous_system(
                state_class=CameraState,
                intrinsics=config.intrinsics_params,
                H=config.H, W=config.W, 
                poses_or_RTs=cam_poses,
                near_clip=config.near_clip, far_clip=200, 
                size=config.size,
                device='cuda'
            )
            
            poses, base_projections, _, masks = cam_system.simulate_vision(positions, renderer='projection_only')
            density_reconstructor = DensityReconstructor(max_iter=train_params['lr_max_steps'], use_decoupled=False)

            for noise_std in noise_levels:
                print(f"  -> Noise Std: {noise_std} ({num_trials} trials)")
                
                for trial in range(num_trials):
                    # Set a unique, reproducible seed for each specific trial
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

                    # Dynamic output directory to prevent trials from overwriting each other's logs
                    trial_out_dir = os.path.join(log_file_path, f"t_{time_step:03d}_cam{cam_num}_n{noise_std}_tr{trial}")

                    model, scale_spaces = density_reconstructor.process_frame(
                        cam_system, point_sets=noisy_projections, positions=positions,
                        initGMM=None,
                        is_adaptive_scale=False, scale=current_scale,
                        is_store_intermediate=False, is_log=False, # Turned off to save disk space over 300 runs
                        output_dir=trial_out_dir,
                        debug=False,
                        train_params=train_params,
                        reconstruction_params=reconstruction_params_base
                    )

                    _, total_fp_mass, total_fn_mass = compute_metrics_batched_torch(
                        means1_np=positions, sigma1=current_scale, 
                        pred_means=model[0]._xyz, pred_weights=model[0]._weights, pred_sigmas=model[0]._radius,
                        bounds=bounds, voxel_res=voxel_res, batch_size=50000, device='cuda'
                    )

                    model_dmota = 1 - (total_fn_mass + total_fp_mass) / N
                    model_dmota_raw[cam_num][noise_std].append(model_dmota)
        
        # Save results to cache 
        print(f"\n💾 Saving computed results to '{cache_filename}'...")
        with open(cache_filename, 'wb') as f:
            pickle.dump({
                'gmr': gmr_dmota, 
                'model_raw': model_dmota_raw
            }, f)

    # ==========================================
    # 4. Plotting Mean and Std Fill
    # ==========================================
    print("🎨 Generating plot with variance...")
    plt.figure(figsize=(7, 5))
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'cm'  
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 300  

    # Plot GMR Baseline
    plt.axhline(y=gmr_dmota, label='GMR-2 Baseline', color='black', linewidth=2.5, linestyle='--', zorder=10)

    # Plot Model Results with Fill Between
    colormap = plt.cm.get_cmap('viridis', len(cam_nums))
    for i, cam_num in enumerate(cam_nums):
        means = []
        stds = []
        
        # Calculate statistics for each noise level
        for n_std in noise_levels:
            scores = model_dmota_raw[cam_num][n_std]
            means.append(np.mean(scores))
            stds.append(np.std(scores))
            
        means = np.array(means)
        stds = np.array(stds)
        color = colormap(i)
        
        # Draw the solid mean line
        plt.plot(noise_levels, means, label=f'Ours-{cam_num}', color=color, marker='o', markersize=4, linewidth=1.5)
        
        # Draw the shaded standard deviation region
        plt.fill_between(noise_levels, means - stds, means + stds, color=color, alpha=0.15, edgecolor='none')

    plt.xlabel('Noise Standard Deviation (Pixels)')
    plt.ylabel('DRA')
    plt.xticks(noise_levels) 
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
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
    import pickle
    cache_filename = 'dra_mapped_3d_noise_cache.pkl'
    
    # Target 2D Noise levels in pixels
    noise_levels_2d = list(range(5, 21, 3)) # [5, 8, 11, 14, 17, 20]
    num_trials = 10
    
    # Fixed parameters
    fixed_comp_num = 20 

    # ==========================================
    # 0. Check for Cached Data
    # ==========================================
    if not force_recalculate and os.path.exists(cache_filename):
        print(f"✅ Found cached data in '{cache_filename}'. Loading results...")
        with open(cache_filename, 'rb') as f:
            cached_data = pickle.load(f)
            noisy_full_raw = cached_data['noisy_full']
            noisy_gmr_raw = cached_data['noisy_gmr']
            noise_levels_3d = cached_data['noise_levels_3d']
            
    else:
        print("⏳ No cache found. Starting mapped 3D noise computations...")
        
        # ==========================================
        # 1. Static Setup & Data Loading
        # ==========================================
        run_params = DATASET_RUNS[0]
        name = run_params['name']
        start_step = run_params['start_step']
        end_step = run_params['end_step']
        step_length = run_params['step_length']

        scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
        config_path = os.path.join(scenario_path, "config.yaml")

        config = SimulationConfig(config_path) 
        factory = DatasetFactory()
        dataset = factory.get_dataset(config.data_file)

        max_steps = dataset.trajectories.shape[0]
        effective_end_step = end_step if end_step is not None and end_step <= max_steps else max_steps
        step_range = range(start_step, effective_end_step, step_length)
        
        gt_data = np.load(scenario_path + '/reconstruction_scale.npz')
        gt_scales = gt_data['scales_gt']

        idx = 5
        time_step = step_range[idx]
        positions = dataset.positions_at_time_step(time_step)
        N = positions.shape[0]

        # ==========================================
        # 1.5 Convert 2D Pixel Noise to 3D Spatial Noise
        # ==========================================
        # Generate representative camera positions to find distance D
        cam_positions, _ = generate_encircling_cameras(
            dataset, step_range, config.intrinsics_params, config.H, config.W, cam_num=4, padding=1
        )
        
        swarm_center = np.mean(positions, axis=0)
        # Using the first camera to estimate the distance to the swarm center
        D = np.linalg.norm(cam_positions[0] - swarm_center)
        
        # Focal length (assuming fx and fy are identical or very close, fx is at [0,0])
        focal_length = config.intrinsics_params[0, 0].item() if torch.is_tensor(config.intrinsics_params) else config.intrinsics_params[0, 0]
        
        # Conversion: n_3d = n_2d * (D / f)
        noise_levels_3d = [n_2d * (D / focal_length) for n_2d in noise_levels_2d]
        
        print(f"Calculated Swarm Distance (D): {D:.2f}")
        print(f"Focal Length (f): {focal_length:.2f}")
        for n2, n3 in zip(noise_levels_2d, noise_levels_3d):
            print(f"  Mapped 2D {n2}px -> 3D {n3:.5f} units")

        # Setup bounding box evaluation limits
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        current_scale = gt_scales[idx]
        
        max_noise_3d = np.max(noise_levels_3d)
        bounds = np.vstack((min_coords - 3 * current_scale - max_noise_3d, 
                            max_coords + 3 * current_scale + max_noise_3d)).T
        voxel_res = np.max(max_coords - min_coords) * 5e-3

        # ==========================================
        # 2. Evaluate 3D Noise over Trials
        # ==========================================
        noisy_full_raw = {n: [] for n in noise_levels_2d}
        noisy_gmr_raw = {n: [] for n in noise_levels_2d}

        # We iterate over the zipped pairs of 2D labels and 3D values
        for n_2d, n_3d in zip(noise_levels_2d, noise_levels_3d):
            print(f"Testing Equivalent 2D Noise = {n_2d}px ({num_trials} trials)...")
            
            for trial in range(num_trials):
                np.random.seed(42 + trial + int(n_2d * 100))
                
                # Apply the mapped 3D noise directly to the positions
                noise_3d_array = np.random.normal(0, n_3d, size=positions.shape)
                noisy_positions = positions + noise_3d_array
                
                # ----------------------------------------------------
                # EVAL 1: Full Perturbed Density Field (Unreduced)
                # ----------------------------------------------------
                pred_means_full = torch.from_numpy(noisy_positions).cuda().float()
                pred_weights_full = torch.ones((N, 1), device='cuda', dtype=torch.float)
                pred_sigmas_full = torch.full((N, 1), current_scale, device='cuda', dtype=torch.float)

                _, total_fp_full, total_fn_full = compute_metrics_batched_torch(
                    means1_np=positions, sigma1=current_scale, 
                    pred_means=pred_means_full, pred_weights=pred_weights_full, pred_sigmas=pred_sigmas_full,
                    bounds=bounds, voxel_res=voxel_res, batch_size=50000, device='cuda'
                )
                dra_full = 1 - (total_fn_full + total_fp_full) / N
                noisy_full_raw[n_2d].append(dra_full)

                # ----------------------------------------------------
                # EVAL 2: GMR Reduced Perturbed Density Field
                # ----------------------------------------------------
                r_means, r_weights, r_covs = GMR.runnalls_algorithm_simple_torch(
                    means=torch.from_numpy(noisy_positions),
                    radii=torch.full((N, 1), current_scale, device='cuda', dtype=torch.float),
                    weights=torch.full((N, 1), 1.0, device='cuda', dtype=torch.float),
                    L=fixed_comp_num,
                    DEVICE='cuda'
                )
                r_radius = torch.sqrt(r_covs[:, 0, 0]).reshape((-1, 1))

                _, total_fp_gmr, total_fn_gmr = compute_metrics_batched_torch(
                    means1_np=positions, sigma1=current_scale, 
                    pred_means=r_means, pred_weights=r_weights, pred_sigmas=r_radius,
                    bounds=bounds, voxel_res=voxel_res, batch_size=50000, device='cuda'
                )
                dra_gmr = 1 - (total_fn_gmr + total_fp_gmr) / N
                noisy_gmr_raw[n_2d].append(dra_gmr)
        
        # Save results to cache 
        print(f"💾 Saving computed results to '{cache_filename}'...")
        with open(cache_filename, 'wb') as f:
            pickle.dump({
                'noisy_full': noisy_full_raw, 
                'noisy_gmr': noisy_gmr_raw,
                'noise_levels_3d': noise_levels_3d
            }, f)

    # ==========================================
    # 3. Plotting Mean and Std Fill
    # ==========================================
    print("🎨 Generating plot with variance...")
    plt.figure(figsize=(7, 5))
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'cm'  
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['figure.dpi'] = 300  

    def plot_with_variance(data_dict, label, color, marker):
        means = []
        stds = []
        # We plot against the 2D pixel values for the X-axis
        for n_2d in noise_levels_2d:
            scores = data_dict[n_2d]
            means.append(np.mean(scores))
            stds.append(np.std(scores))
            
        means = np.array(means)
        stds = np.array(stds)
        
        plt.plot(noise_levels_2d, means, label=label, color=color, marker=marker, markersize=5, linewidth=2)
        plt.fill_between(noise_levels_2d, means - stds, means + stds, color=color, alpha=0.15, edgecolor='none')

    # Plot both evaluations
    plot_with_variance(noisy_full_raw, 'Perturbed Full Field', color='#1f77b4', marker='o')
    plot_with_variance(noisy_gmr_raw, f'Perturbed GMR Field ({fixed_comp_num} modes)', color='#ff7f0e', marker='s')

    # X-Axis is the equivalent 2D noise
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

def plot_dra_and_loss(run_params=None, baseline_deg=90, eval_every=1):
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
    """
    import tempfile
    from experiments.run_scenarios_angle_sweep import (
        _build_angled_cam_system,
        _precompute_gt_density,
        _build_grid,
        _compute_metrics_cached,
    )

    # ── defaults ──────────────────────────────────────────────────────
    if run_params is None:
        run_params = {
            "name": "jackdaw",
            "start_step": 350,
            "end_step": 360,
            "step_length": 10,
        }

    name = run_params["name"]
    start_step = run_params["start_step"]
    end_step = run_params["end_step"]
    step_length = run_params["step_length"]

    scenario_path = os.path.join(os.getcwd(), "scenarios", name)
    config_path = os.path.join(scenario_path, "config.yaml")

    config = SimulationConfig(config_path)
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    step_list = list(range(start_step, end_step, step_length))

    # Global bounding sphere → camera distance D
    all_positions = []
    for t in step_list:
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
    min_half_fov = min(
        np.arctan2(cx, fx),
        np.arctan2(config.W - cx, fx),
        np.arctan2(cy, fy),
        np.arctan2(config.H - cy, fy),
    )
    D = max_radius / np.sin(min_half_fov)

    gt_data = np.load(os.path.join(scenario_path, "reconstruction_scale.npz"))
    gt_scales = gt_data["scales_gt"]

    idx = 0  # first frame
    time_step = step_list[idx]
    positions = dataset.positions_at_time_step(time_step)
    gt_scale = gt_scales[idx]

    train_iters = 100
    train_params = {
        "xyz_lr_c": 0.05, "xyz_lr_final_c": 0.9,
        "radius_lr_c": 0.05, "radius_lr_final_c": 0.9,
        "weights_lr_c": 0.10, "weights_lr_final_c": 0.7,
        "xyz_reg": 1.0, "radius_reg": 0.3,
        "radius_cutoff_inv": 0.5, "lr_max_steps": train_iters,
    }
    reconstruction_params = {
        "targetd_num_mode": 10,
        "voxel_scale": 0.5,
        "voxel_peak_threshold": 0.3,
        "voxel_grid_max_size": 32,
        "voxel_peaks_number": 2 * 10,
    }

    cam_system = _build_angled_cam_system(center, D, baseline_deg, config)

    # ── Pre-compute GT density grid ───────────────────────────────────
    print(f"Pre-computing GT density grid (N={positions.shape[0]})…")
    grid = _build_grid(positions, gt_scale)
    gt_density = _precompute_gt_density(positions, gt_scale, grid)

    # ── Run training with checkpointing ───────────────────────────────
    print(f"Running training for {train_iters} iterations…")
    dr = DensityReconstructor(
        max_iter=train_params["lr_max_steps"], use_decoupled=False,
    )
    _, projections, _, _ = cam_system.simulate_vision(
        positions, renderer="projection_only",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        model, _ = dr.process_frame(
            cam_system, point_sets=projections, positions=positions,
            initGMM=None,
            is_adaptive_scale=False, scale=gt_scale,
            is_store_intermediate=True, is_log=True,
            output_dir=tmpdir, debug=False,
            train_params=train_params,
            reconstruction_params=reconstruction_params,
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
            ckpt = training_history[it + 1]  # +1 offset in history dict
            tp, fp, fn = _compute_metrics_cached(
                ckpt["_xyz"], ckpt["_weights"], ckpt["_radius"],
                gt_density, grid,
            )
            dmota_vals[e_idx] = 1.0 - (fn + fp) / N

    # ── Style ─────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 300,
    })

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Left axis: rendering loss (log scale, alternating cameras)
    color_cam1 = "#1f77b4"   # blue
    color_cam2 = "#d62728"   # red
    color_dmota = "#2ca02c"  # green

    ax1.plot(steps[::2], loss_history[::2],
             color=color_cam1, linewidth=1.2, label="Loss Cam 1")
    ax1.plot(steps[1::2], loss_history[1::2],
             color=color_cam2, linewidth=1.2, linestyle="--",
             alpha=0.85, label="Loss Cam 2")

    ax1.set_yscale("log")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss", color="#333333")
    ax1.tick_params(axis="y", labelcolor="#333333")
    ax1.set_ylim(bottom=max(1e-4, np.min(loss_history) * 0.5))

    # Right axis: dMOTA
    ax2 = ax1.twinx()
    ax2.plot(np.array(eval_iters), dmota_vals,
             color=color_dmota, linewidth=2.2, markeredgewidth=1.5,
             label="DRA", zorder=5)
    ax2.set_ylabel("DRA", color=color_dmota)
    ax2.tick_params(axis="y", labelcolor=color_dmota)
    ax2.set_ylim(max(0.0, np.nanmin(dmota_vals) - 0.05),
                 min(1.0, np.nanmax(dmota_vals) + 0.05))

    # Legend (combine both axes)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2,
        loc="lower left", frameon=False, ncol=1,
    )

    # Clean up
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
    # ── 1. Load dataset & compute global bounding sphere ────────────
    scenario_path = os.path.join(os.getcwd(), "scenarios", dataset_name)
    config_path = os.path.join(scenario_path, "config.yaml")

    config = SimulationConfig(config_path)
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    max_steps = dataset.trajectories.shape[0]
    step_range = range(0, max_steps, max(1, max_steps // 5))

    all_positions = np.vstack(
        [dataset.positions_at_time_step(t) for t in step_range]
    )
    center = (all_positions.min(axis=0) + all_positions.max(axis=0)) / 2.0
    max_radius = np.max(np.linalg.norm(all_positions - center, axis=1))

    positions = dataset.positions_at_time_step(step_range[0])

    # ── 2. Academic styling ─────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times", "Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    # ── Colour & marker scheme (Wong 2011 colourblind‑safe) ─────────
    # Each camera count gets a distinct colour + marker combination.
    CONFIG_STYLES = {
        2: {"color": "#D55E00", "marker": "o", "label": r"$K = 2$"},        # vermilion █ square
        3: {"color": "#0072B2", "marker": "o", "label": r"$K = 3$"},        # blue      ◆ diamond
        5: {"color": "#009E73", "marker": "o", "label": r"$K = 5$"},        # green     ● circle
    }

    C_SWARM     = "#9CA3AF"   # medium gray – background points
    C_ORBIT     = "#D1D5DB"   # light gray – orbit circle
    C_BOUNDING  = "#CBD5E0"   # lighter gray – swarm bounding circle
    C_DIR       = "#CBD5E0"   # lighter gray – direction tick
    C_ARC       = "#D55E00"   # match 2‑cam colour for the 90° arc
    C_ARC_TEXT  = "#A04000"   # darker variant of arc colour

    # ── 3. Pre‑compute all camera positions ─────────────────────────
    cam_nums = [2, 3, 5]
    all_cameras = {}   # cam_num → (N,3) array of positions

    # D is identical for all configurations (depends only on swarm
    # extent + intrinsics, not on cam_num), so compute once.
    D = None
    for cam_num in cam_nums:
        if cam_num == 2:
            raw_positions, D = generate_encircling_cameras(
                dataset, step_range, config.intrinsics_params,
                config.H, config.W, cam_num=4, padding=1,
            )
            all_cameras[cam_num] = raw_positions[:2]
        else:
            raw_positions, D = generate_encircling_cameras(
                dataset, step_range, config.intrinsics_params,
                config.H, config.W, cam_num=cam_num, padding=1,
            )
            all_cameras[cam_num] = raw_positions

    # ── 4. Build single‑panel figure ────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.2, 7.2))

    # -- Light shared orbit circle (thin, low contrast) --
    theta_full = np.linspace(0, 2 * np.pi, 600)
    ax.plot(center[0] + D * np.cos(theta_full),
            center[1] + D * np.sin(theta_full),
            color="#E5E7EB", linewidth=1.0, zorder=0)

    # -- Swarm points (top‑down projection, subtle) --
    ax.scatter(positions[:, 0], positions[:, 1],
               c=C_SWARM, s=1.0, alpha=0.35, edgecolors="none",
               zorder=1)

    # -- Swarm bounding circle (dashed) --
    ax.add_patch(plt.Circle(
        (center[0], center[1]), max_radius,
        fill=False, color="#CBD5E0", linewidth=0.8,
        linestyle=(0, (4, 5)), alpha=0.65, zorder=2,
    ))

    # -- Per‑configuration: dashed leader lines + markers ──────────
    for cam_num in cam_nums:
        style = CONFIG_STYLES[cam_num]
        cp = all_cameras[cam_num]               # (N_cam, 3)

        # ---- dashed leader lines: bounding circle → camera ----
        for cam in cp:
            v = cam[:2] - center[:2]
            dist = np.linalg.norm(v)
            if dist < 1e-9:
                continue
            u = v / dist
            p_start = center[:2] + u * (max_radius * 1.05)
            ax.plot([p_start[0], cam[0]], [p_start[1], cam[1]],
                    color=style["color"], linewidth=1.0,
                    linestyle=(0, (4, 5)), alpha=0.55, zorder=4)
            
            if cam_num == 2 and v[1] == 0:
                ax.plot([p_start[0], cam[0]], [p_start[1]+20, cam[1]+20],
                        color=style["color"], linewidth=1.0,
                        linestyle=(0, (4, 5)), alpha=0.55, zorder=4)

            if cam_num == 3 and v[1] == 0:
                ax.plot([p_start[0], cam[0]], [p_start[1]-20, cam[1]-20],
                        color=style["color"], linewidth=1.0,
                        linestyle=(0, (4, 5)), alpha=0.55, zorder=4)

        # ---- camera markers (large, white‑bordered) ----
        ax.scatter(
            cp[:, 0], cp[:, 1],
            marker=style["marker"], s=160,
            c=style["color"],
            edgecolors="white", linewidths=1.8,
            zorder=12, label=style["label"],
        )

    # -- Legend ──────────────────────────────────────────────────────
    legend = ax.legend(
        loc="lower right", frameon=True, fancybox=False,
        edgecolor="#CCCCCC", facecolor="white",
        framealpha=0.92, borderpad=0.6,
        handletextpad=0.5, labelspacing=0.4,
    )
    legend.set_zorder(20)

    # -- Axis limits & clean‑up ──────────────────────────────────────
    pad = D * 0.25
    ax.set_xlim(center[0] - D - pad, center[0] + D + pad)
    ax.set_ylim(center[1] - D - pad, center[1] + D + pad)
    ax.set_aspect("equal")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # ── 5. Save & return ────────────────────────────────────────────
    out_dir = os.path.join(os.getcwd(), "figs")
    os.makedirs(out_dir, exist_ok=True)

    for fmt in ("png", "pdf"):
        fig.savefig(
            os.path.join(out_dir, f"camera_configurations.{fmt}"),
            dpi=300, bbox_inches="tight", transparent=True,
        )

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

    # plot_dra_and_loss()

    plot_camera_configurations()

    plt.show()