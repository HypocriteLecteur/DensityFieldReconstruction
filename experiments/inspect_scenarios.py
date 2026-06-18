import logging
import sys
import os
from tqdm import tqdm


import numpy as np
from dfr.simulation_config import SimulationConfig
from dfr.dataset_io import DatasetFactory
from dfr.camera_system import MultiCameraSystem
from dfr.camera_state import CameraState
from dfr.visualizer import SimulationVisualizer, MultiGMMPlotter
from dfr.utils import move_figure
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
        'name': 'swift',
        'log_name': 'base_init',
        'start_step': 0,
        'end_step': None,
        'step_length': 200,
    },
    # {
    #     'name': 'jackdaw',
    #     'log_name': 'base_init',
    #     'start_step': 350,
    #     'end_step': 550,
    #     'step_length': 10,
    # },
    # {
    #     'name': 'jackdaw2',
    #     'log_name': 'base_init',
    #     'start_step': 1800,
    #     'end_step': 3480,
    #     'step_length': 20,
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
    cam_system = MultiCameraSystem.create_homogeneous_system(
        state_class=CameraState,
        intrinsics=config.intrinsics_params,
        H=config.H, W=config.W, 
        poses_or_RTs=config.cam_poses,
        near_clip=config.near_clip, far_clip=config.far_clip, 
        size=config.size,
        device='cuda')

    visualizer = SimulationVisualizer(H=config.H, W=config.W, 
                                      cam_num=config.cam_poses.shape[0],
                                      start_step=start_step, end_step=effective_end_step, step_length=step_length,
                                      mode='all',
                                      save_video=False, fps=30, dpi=100,
                                      positions_all=dataset.trajectories)
    move_figure(visualizer.fig, 2800, 100)
    # 5. Simulation Loop

    # 5. Define the interactive rendering logic
    def render_step(time_step):
        """This gets called dynamically every time the slider moves."""
        positions = dataset.positions_at_time_step(time_step)
        
        # Run simulation for this specific step
        poses, _, images, _ = cam_system.simulate_vision(positions, renderer='gaussian')

        # Push updates to the UI
        visualizer.update(
            time_step=time_step,
            positions=positions,
            cameras=cam_system.cameras,
            imgs=images
        )

    # 6. Hook it up and start the GUI!
    visualizer.set_interactive_callback(render_step)
    visualizer.run(initial_step=start_step)

def run_multi_scenarios():
    for run_params in DATASET_RUNS:
        run_single_scenario(run_params)

if __name__ == "__main__":
    run_multi_scenarios()
    plt.show()