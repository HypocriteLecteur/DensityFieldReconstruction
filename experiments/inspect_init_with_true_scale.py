import logging
import sys
import os
from tqdm import tqdm

sys.path.append(os.getcwd()) # To get around relative import issues. I hate Python.

import numpy as np
from density_field_reconstruction.simulation_config import SimulationConfig
from density_field_reconstruction.dataset import DatasetFactory
from density_field_reconstruction.camera_simulation import MultiCameraSystem
from density_field_reconstruction.density_reconstructor import DensityReconstructor
from density_field_reconstruction.camera_state import CameraState
from density_field_reconstruction.utils import calculate_gmm_ise_gpu

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

DATASET_RUNS = [
    {
        'name': 'boids_multi',
        'log_name': 'base_init',
        'start_step': 0,
        'end_step': None,
        'step_length': 1,
    },
    {
        'name': 'boids',
        'log_name': 'base_init',
        'start_step': 0,
        'end_step': None,
        'step_length': 1,
    },
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

DATASET_VIS = [
    {
        'name': 'boids_multi',
        'log_name': 'base_init',
        'log_name2': 'base_init',
    },
    {
        'name': 'boids',
        'log_name': 'base_init',
        'log_name2': 'base_init',
    },
    # {
    #     'name': 'cluster',
    #     'log_name': 'base',
    #     'log_name2': 'base_init',
    # },
    # {
    #     'name': 'clutter',
    #     'log_name': 'base',
    #     'log_name2': 'base_init',
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
        'final_gmm_num': []
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
    cam_system = MultiCameraSystem.create_system(
        intrinsics=config.intrinsics_params,
        H=config.H, W=config.W, 
        poses=config.cam_poses,
        near_clip=config.near_clip, far_clip=config.far_clip, size=config.size)
    density_reconstructor = DensityReconstructor(config.intrinsics_params, max_iter=config.iter)

    # 5. Simulation Loop
    step_range = range(start_step, effective_end_step, step_length)
    model = None

    if config.cam_poses.shape[0] != 2:
        print("warning, implementation for model.sum_loss is broken.")

    for time_step in (pbar := tqdm(step_range, desc=f"Processing {name}")):
        positions = dataset.trajectories[time_step]

        import torch
        positions_torch = torch.from_numpy(positions).cuda().float()

        sigma = 10

        modes = positions_torch.clone()

        cdist = torch.cdist(positions_torch, modes)
        W = torch.exp(-0.5 / sigma**2 * cdist**2)
        D = torch.diag(torch.sum(W, dim=0))
        Q = W @ torch.inverse(D) # column-stochastic
        modes = Q.T @ positions_torch
        print()

    #     poses, _, images = cam_system.simulate_vision(positions, renderer='gaussian')

    #     camera_states = []    
    #     for i, pose in enumerate(poses):
    #         camera_states.append(
    #             CameraState(i, config.intrinsics_params, pose, device='cuda')
    #         )
        
    #     model, scale_spaces = \
    #     density_reconstructor.process_frame(camera_states, images, positions=positions,
    #                                         initGMM=model,
    #                                         is_adaptive_scale=True,
    #                                         is_store_intermediate=True, is_log=True,
    #                                         output_dir=os.path.join(log_file_path, f"t_{time_step:03d}"),
    #                                         debug=True)
    
    #     # 6. Collect Metrics
    #     for metric_name, value in density_reconstructor.time_metrics.items():
    #         time_metrics[metric_name].append(value)
        
    #     loss_metrics['final_training_loss'].append(model[0].sum_loss)
    #     loss_metrics['final_gmm_num'].append(model[0]._xyz.shape[0])

    #     _, projections, _ = cam_system.simulate_vision(positions, renderer='gaussian')

    #     is_visible = np.zeros((positions.shape[0],), dtype=np.bool)
    #     for i in range(len(poses)):
    #         projection = projections[i]
    #         is_visible_ = (projection[:, 0] > 0).squeeze() & (projection[:, 1] > 0).squeeze() & \
    #             (projection[:, 0] < config.H).squeeze() & (projection[:, 1] < config.W).squeeze()
    #         is_visible = is_visible | is_visible_
    #     loss_metrics['final_density_field_loss'].append(
    #         calculate_gmm_ise_gpu(
    #             positions[is_visible],
    #             density_reconstructor.scale, 
    #             model[0]._xyz, 
    #             model[0]._weights, 
    #             model[0]._radius))
    
    # # 7. Logging and Data Saving
    # logger.info(f"Results for {name}:")
    # if time_metrics['train_gaussian_scale_space']:
    #     mean_time = np.mean(np.array(time_metrics['train_gaussian_scale_space']))
    #     logger.info(f"Mean 'train_gaussian_scale_space' time: {mean_time:.2f} ms")
    # else:
    #     logger.info("No time steps processed.")

    # save_data = {key: np.array(val) for key, val in time_metrics.items()}
    # save_data['final_loss_history'] = np.array(loss_metrics['final_training_loss'])
    # save_data['ise_loss_history'] = np.array(loss_metrics['final_density_field_loss'])
    # save_data['final_gmm_num'] = np.array(loss_metrics['final_gmm_num'])

    # save_path = os.path.join(log_file_path, "statistics.npz")
    # np.savez(save_path, **save_data)
    # logger.info(f"Statistics saved to: {save_path}")

    # logger.info(f"Finished scenario {name}")

def run_multi_scenarios():
    for run_params in DATASET_RUNS:
        run_single_scenario(run_params)

def plot_time_multi_scenarios():
    for run_params in DATASET_VIS:
        plot_time_single_scenarios(run_params)

def plot_time_single_scenarios(run_params):
    name = run_params['name']
    log_name = run_params['log_name']
    scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
    log_file_path = os.path.join(scenario_path, *["logs", log_name])

    log_data = np.load(os.path.join(log_file_path, "statistics.npz"))

def post_processing():
    for run_params in DATASET_VIS:
        vis_simulation(run_params)

def vis_simulation(run_params):
    from density_field_reconstruction.analysis import compare_final_loss_history

    name = run_params['name']
    log_name = run_params['log_name']
    scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
    log_file_path = os.path.join(scenario_path, *["logs", log_name])

    log_name2 = run_params['log_name2']
    log_file_path2 = os.path.join(scenario_path, *["logs", log_name2])

    compare_final_loss_history(log_file_path, log_file_path2, name)

def plot_all_loss_curve():
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(211)

    for run_params in DATASET_VIS:
        name = run_params['name']
        log_name = run_params['log_name']
        scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
        log_file_path = os.path.join(scenario_path, *["logs", log_name])

        log_data = np.load(os.path.join(log_file_path, "statistics.npz"))
        ax.plot(np.arange(log_data['final_loss_history'].size), log_data['final_loss_history'], label=name)
    
    plt.title('Images Rendering Loss')
    # plt.xlabel('time step')
    plt.legend()
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # fig = plt.figure(figsize=(8, 6))
    ax2 = fig.add_subplot(212)

    for run_params in DATASET_VIS:
        name = run_params['name']
        log_name = run_params['log_name']

        scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
        log_file_path = os.path.join(scenario_path, *["logs", log_name])

        log_data = np.load(os.path.join(log_file_path, "statistics.npz"))
        ax2.plot(np.arange(log_data['ise_loss_history'].size), log_data['ise_loss_history'], label=name)
    
    plt.title('Integrated Squared Error')
    plt.xlabel('time step')
    plt.legend()
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

if __name__ == "__main__":
    run_multi_scenarios()
    # plot_time_multi_scenarios()
    post_processing()
    # plot_all_loss_curve()
    plt.show()