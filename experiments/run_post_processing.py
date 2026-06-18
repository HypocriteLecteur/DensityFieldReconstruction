import logging
import sys
import os
import shutil
from tqdm import tqdm


import torch
import numpy as np
from dfr.simulation_config import SimulationConfig
from dfr.dataset_io import DatasetFactory
from dfr.camera_system import MultiCameraSystem
from dfr.density_field_reconstructor import DensityReconstructor
from dfr.density_field_model import GaussianModel
from dfr.camera_state import CameraState
from dfr.utils import compute_metrics_batched_torch
from dfr.utils import move_figure

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

LOG_NAME = 'base_reg'
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
    # {
    #     'name': 'boids_multi',
    #     'log_name': LOG_NAME,
    #     'start_step': 0,
    #     'end_step': None,
    #     'step_length': 10,
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

VIS_LOG_NAME = 'base_reg'
VIS_LOG_NAME2 = 'base_reg'
DATASET_VIS = [
    {
        'name': 'starling',
        'log_name': VIS_LOG_NAME,
        'log_name2': VIS_LOG_NAME2,
    },
    {
        'name': 'jackdaw',
        'log_name': VIS_LOG_NAME,
        'log_name2': VIS_LOG_NAME2,
    },
    {
        'name': 'jackdaw2',
        'log_name': VIS_LOG_NAME,
        'log_name2': VIS_LOG_NAME2,
    },
]

def view_loss_multi_scenarios():
    for run_params in DATASET_VIS:
        view_loss_single_scenario(run_params)

def view_loss_single_scenario(run_params):
    name = run_params['name']
    log_name = run_params['log_name']
    scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
    log_file_path = os.path.join(scenario_path, *["logs", log_name])

    log_name2 = run_params['log_name2']
    log_file_path2 = os.path.join(scenario_path, *["logs", log_name2])

    log_data = np.load(os.path.join(log_file_path, "statistics.npz"))
    log_data2 = np.load(os.path.join(log_file_path2, "statistics.npz"))
    log_data2_baseline = np.load(os.path.join(log_file_path2, "statistics_baseline.npz"))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    move_figure(fig, 2800, 100)
    ax1.plot(np.arange(log_data['final_training_loss'].size), log_data['final_training_loss'], 
            color='royalblue',
            label=f'{name} + {log_name}')
    ax1.plot(np.arange(log_data2['final_training_loss'].size), log_data2['final_training_loss'], 
        color='darkorange',
        label=f'{name} + {log_name2}')
    ax1.plot(np.arange(log_data2['final_training_loss'].size), log_data2_baseline['final_training_loss'],
        color='black',
        label=f'{name} + {log_name2}_baseline')
    ax1.set_title(f'mean 2d loss: {np.mean(log_data['final_training_loss']):.3f} {np.mean(log_data2['final_training_loss']):.3f}')
    
    ax2.plot(np.arange(log_data['final_density_field_loss'].size), log_data['final_density_field_loss'], 
            color='royalblue',
            label=f'{name} + {log_name}')
    ax2.plot(np.arange(log_data2['final_density_field_loss'].size), log_data2['final_density_field_loss'], 
        color='darkorange',
        label=f'{name} + {log_name}')
    ax2.plot(np.arange(log_data2['final_density_field_loss'].size), log_data2_baseline['final_density_field_loss'],
        color='black',
        label=f'{name} + {log_name2}_baseline')
    ax2.set_title(f'mean NISE loss: {np.mean(log_data['final_density_field_loss']):.3f} {np.mean(log_data2['final_density_field_loss']):.3f}')

    ax1.legend()
    ax2.legend()

if __name__ == "__main__":
    # view_loss_multi_scenarios()

    plt.show()