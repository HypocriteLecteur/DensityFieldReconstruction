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
from density_field_reconstruction.visualizer import SimulationVisualizer

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

    visualizer = SimulationVisualizer(intrinsics_params=config.intrinsics_params,
                                      H=config.H, W=config.W, 
                                      cam_num=config.cam_poses.shape[0],
                                      mode='all',
                                      save_video=False, fps=30, dpi=100,
                                      positions_all=dataset.trajectories)
    # 5. Simulation Loop
    step_range = range(start_step, effective_end_step, step_length)

    for time_step in (pbar := tqdm(step_range, desc=f"Processing {name}")):
        positions = dataset.trajectories[time_step]
        poses, _, images = cam_system.simulate_vision(positions, renderer='gaussian')

        visualizer.update(time_step=time_step,
                          positions=positions,
                          cam_poses=poses,
                          imgs=images)

def run_multi_scenarios():
    for run_params in DATASET_RUNS:
        run_single_scenario(run_params)
        try:
            run_single_scenario(run_params)
        except Exception as e:
            logger.error(f"An error occurred while processing run '{run_params['name']}': {e}")

if __name__ == "__main__":
    run_multi_scenarios()
    plt.show()