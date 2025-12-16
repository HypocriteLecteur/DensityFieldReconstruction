import logging
import sys
import os
from tqdm import tqdm

sys.path.append(os.getcwd()) # To get around relative import issues. I hate Python.

import torch
import numpy as np
from density_field_reconstruction.simulation_config import SimulationConfig
from density_field_reconstruction.dataset import DatasetFactory
from density_field_reconstruction.camera_simulation import MultiCameraSystem
from density_field_reconstruction.density_reconstructor import DensityReconstructor
from density_field_reconstruction.camera_state import CameraState
from density_field_reconstruction.visualizer import SimulationVisualizer
from gaussian_rasterizer_simple_large import rasterize_gaussians

import matplotlib.pyplot as plt

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

DATASET_RUNS = [
    {
        'name': 'boids_multi',
        'log_name': 'base_init',
        'start_step': 0,
        'end_step': None,
        'step_length': 1,
    },
    # {
    #     'name': 'boids',
    #     'log_name': 'base_init',
    #     'start_step': 0,
    #     'end_step': None,
    #     'step_length': 1,
    # },
    # {
    #     'name': 'cluster',
    #     'log_name': 'base_init',
    #     'start_step': 0,
    #     'end_step': None,
    #     'step_length': 1,
    # },
    # {
    #     'name': 'clutter',
    #     'log_name': 'base_init',
    #     'start_step': 0,
    #     'end_step': None,
    #     'step_length': 1,
    # },
]

def run_single_scenario(run_params):
    # 1. Parameter extraction and Logging Setup
    name = run_params['name']
    log_name = run_params['log_name']
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
    
    # 4. System Initialization
    cam_system = MultiCameraSystem.create_system(
        intrinsics=config.intrinsics_params,
        H=config.H, W=config.W, 
        poses=config.cam_poses,
        near_clip=config.near_clip, far_clip=config.far_clip, size=config.size)
    density_reconstructor = DensityReconstructor(config.intrinsics_params, max_iter=config.iter)

    # visualizer = SimulationVisualizer(intrinsics_params=config.intrinsics_params,
    #                                   H=config.H, W=config.W, 
    #                                   cam_num=config.cam_poses.shape[0],
    #                                   mode='all',
    #                                   save_video=False, fps=30, dpi=100,
    #                                   positions_all=dataset.trajectories)
    # 5. Simulation Loop
    step_range = range(start_step, effective_end_step, step_length)

    projection_error = []
    scales = []
    dist_to_cams = []
    # for time_step in (pbar := tqdm([200], desc=f"Processing {name}")):
    for time_step in (pbar := tqdm(step_range, desc=f"Processing {name}")):
        positions = dataset.trajectories[time_step]
        poses, _, images = cam_system.simulate_vision(positions, renderer='gaussian')

        camera_states = []    
        for i, pose in enumerate(poses):
            camera_states.append(
                CameraState(i, config.intrinsics_params, pose, device='cuda')
            )

        density_reconstructor.camera_states = camera_states
        center = density_reconstructor.estimate_swarm_center_image(images)
        density_reconstructor.init_adaptive_scale_selection(center, images)
        scale_spaces_affine, per_cam_scale_factors = density_reconstructor.generate_scale_space_img(
            center, [density_reconstructor.scale], images)
        
        scale_spaces_true = []
        positions_torch = torch.from_numpy(positions).float().cuda()
        radius_torch = torch.ones((positions.shape[0], 1), dtype=torch.float, device='cuda')*density_reconstructor.scale
        weight_torch = torch.ones((positions.shape[0], 1), dtype=torch.float, device='cuda')
        
        for i in range(config.cam_poses.shape[0]):
            density = rasterize_gaussians(
                positions_torch,
                radius_torch,
                weight_torch,
                camera_states[i].R,
                camera_states[i].T,
                camera_states[i].K,
                config.H, config.W,
                False
            )
            scale_spaces_true.append(density)

        projection_error.append([torch.sum(torch.abs(scale_spaces_affine[i][0] - scale_spaces_true[i])).item()/(torch.sum(scale_spaces_true[i]).item()) for i in range(config.cam_poses.shape[0])])
        scales.append(density_reconstructor.scale)

        dist_to_cams.append([config.intrinsics_params[0, 0]/scale_factor for scale_factor in per_cam_scale_factors])
        # visualizer.update(time_step=time_step,
        #                   positions=positions,
        #                   cam_poses=poses,
        #                   imgs=images)
    projection_error = np.array(projection_error)
    dist_to_cams = np.array(dist_to_cams)

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(311)
    [ax.plot(np.arange(projection_error.shape[0]), projection_error[:, i]) for i in range(projection_error.shape[1])]
    ax2 = fig.add_subplot(312)
    ax2.plot(np.arange(projection_error.shape[0]), scales)
    ax3 = fig.add_subplot(313)
    [ax3.plot(np.arange(projection_error.shape[0]), dist_to_cams[:, i]) for i in range(dist_to_cams.shape[1])]

def run_multi_scenarios():
    for run_params in DATASET_RUNS:
        run_single_scenario(run_params)

if __name__ == "__main__":
    run_multi_scenarios()
    plt.show()