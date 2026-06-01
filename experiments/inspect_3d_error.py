import sys
import os

sys.path.append(os.getcwd()) # To get around relative import issues, I hate Python.

import torch
import numpy as np
import matplotlib.pyplot as plt
from experiments.power_law import move_figure
from dfr.visualizer import MultiGMMPlotter
from dfr.density_field_model import GaussianModel
from dfr.simulation_config import SimulationConfig
from dfr.camera_system import MultiCameraSystem
from dfr.camera_state import CameraState
from dfr.dataset_io import DatasetFactory
from dfr.utils import calculate_gmm_dissimilarity
from dfr.gaussian_mixture_reduction import GMR

INSPECT_PARAM = {
    'name': 'jackdaw2',
    'log_name': 'base_reg_cam_2',
    'time_step': 2700,
    'start_step': 2700,
    'step_length': 20,
    'iter': 99
}

def inspect_3d_error(inspect_param):
    name = inspect_param['name']
    log_name = inspect_param['log_name']

    time_step = inspect_param['time_step']
    iter = inspect_param['iter']
    start_step = inspect_param['start_step']
    step_length = inspect_param['step_length']
    idx = int((time_step - start_step) / step_length)

    # load model
    scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
    config_path = os.path.join(scenario_path, "config.yaml")
    config = SimulationConfig(config_path)

    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    log_file_path = os.path.join(scenario_path, *["logs", log_name])
    log_data = np.load(os.path.join(log_file_path, "statistics.npz"))
    scale_history = log_data['scale']

    time_step_file_path = os.path.join(log_file_path, f"t_{time_step:03d}")

    model_path = os.path.join(time_step_file_path, "checkpoint_level_0.pth")
    training_history = GaussianModel.load_training_history(model_path)

    model = GaussianModel.load_iter(training_history, iter) 
    means = model._xyz.detach().cpu().numpy()
    radii = model._radius.detach().cpu().numpy()
    weights = model._weights.detach().cpu().numpy()

    cam_system = MultiCameraSystem.create_homogeneous_system(
        state_class=CameraState,
        intrinsics=config.intrinsics_params,
        H=config.H, W=config.W, 
        poses_or_RTs=config.cam_poses,
        near_clip=config.near_clip, far_clip=config.far_clip, 
        size=config.size,
        device='cuda')
    
    positions = dataset.positions_at_time_step(time_step)

    _, projections, _, masks = cam_system.simulate_vision(positions, is_auto_aim=True, renderer='gaussian')
    is_visible = np.ones((positions.shape[0],), dtype=np.bool)
    for i in range(len(projections)):
        is_visible = is_visible & masks[i]

    visible_positions = positions[is_visible]

    # fig = plt.figure(figsize=(15, 10)) # Wider figure for two columns
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # move_figure(fig, 2800, 100)

    # gmm_visualizer = MultiGMMPlotter(fig=fig, ax=ax)
    # gmm_visualizer.add_gmm(means, radii, weights, color='orange', label='GMM', visible=True)

    # gmm_visualizer.update(
    #         real_means=positions[is_visible], cameras=cam_system.cameras[:2]
    # )

    scale = scale_history[idx]
    current_error, removal_errors = calculate_gmm_dissimilarity(
        positions[is_visible], scale, 
        model._xyz, model._weights, model._radius, 
        use_decoupled=False, return_removal_errors=True
    )
    component_errors = current_error - removal_errors
    print(current_error)
    print(removal_errors)

    N = positions[is_visible].shape[0]
    r_means, r_weights, r_covs = GMR.runnalls_algorithm_simple_torch(
        means=torch.from_numpy(positions[is_visible]),
        radii=torch.full((N, 1), scale, device='cuda', dtype=torch.float),
        weights=torch.full((N, 1), 1.0, device='cuda', dtype=torch.float),
        L=model._xyz.shape[0], DEVICE='cuda'
    )
    r_weights = r_weights.reshape((-1, 1))
    r_radius = torch.sqrt(r_covs[:, 0, 0]).reshape((-1, 1))
    current_error, removal_errors = calculate_gmm_dissimilarity(
        positions[is_visible], scale, 
        r_means, r_weights, r_radius, 
        use_decoupled=False, return_removal_errors=True
    )
    component_errors = current_error - removal_errors
    print(current_error)

    final_means, final_unnorm_weights, final_covs = GMR.optimize_ise_isotropic(
        orig_means=torch.from_numpy(positions[is_visible]),
        orig_covs=torch.eye(3, device='cuda').unsqueeze(0).expand(N, 3, 3),
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
    current_error, removal_errors = calculate_gmm_dissimilarity(
        positions[is_visible], scale, 
        final_means, final_unnorm_weights, final_radius, 
        use_decoupled=False, return_removal_errors=True
    )
    component_errors = current_error - removal_errors
    print(current_error)




    model_path_baseline = os.path.join(time_step_file_path, "baseline_level_0.pth")
    torch.load(model_path_baseline, weights_only=False)['_xyz']

    # --- VISUALIZATION ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Plot the target point cloud (f) in faint gray
    ax.scatter(
        visible_positions[:, 0], visible_positions[:, 1], visible_positions[:, 2], 
        c='gray', alpha=0.05, s=1, label='Target Density Points'
    )

    # 2. Plot the GMM components (g) colored by their error contribution
    # Detach and move to CPU for numpy plotting
    gmm_means = model._xyz.detach().cpu().numpy()
    errors_np = component_errors.detach().cpu().numpy()
    
    # Base size on radius, adjust multiplier as needed for your coordinate scale
    sizes = model._radius.detach().cpu().numpy() * 10 + 10 

    sc = ax.scatter(
        gmm_means[:, 0], gmm_means[:, 1], gmm_means[:, 2],
        c=errors_np, cmap='coolwarm', s=sizes, alpha=0.8, label='GMM Components'
    )

    cbar = plt.colorbar(sc, shrink=0.5, pad=0.1)
    cbar.set_label('Error Contribution (Red = Hurts, Blue = Helps)')
    ax.set_title(f"GMM Component Error Analysis (NISE: {current_error:.4f})")
    ax.legend()

    plt.show()

if __name__ == "__main__":
    inspect_3d_error(INSPECT_PARAM)