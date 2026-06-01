import logging
import sys
import os
import shutil
from tqdm import tqdm

sys.path.append(os.getcwd()) # To get around relative import issues. I hate Python.

import time
import torch
import numpy as np
from dfr.simulation_config import SimulationConfig
from dfr.dataset_io import DatasetFactory
from dfr.camera_system import MultiCameraSystem
from dfr.density_field_reconstructor import DensityReconstructor
from dfr.camera_state import CameraState
from dfr.utils import calculate_gmm_dissimilarity, generate_encircling_cameras
from dfr.visualizer import MultiGMMPlotter
from dfr.gaussian_mixture_reduction import GMR
from dfr.mode_finding import find_target_scale, mode_counting, model_4pl_scale_at_x_constant, analytic_solution, analytic_solution_scale_at_x_constant
from gaussian_rasterizer_simple_large import rasterize_gaussians
from experiments.power_law import move_figure, power_2pl, power_3pl
from experiments.reconstruction_scale_determination import compute_scaling_law
from scipy.optimize import minimize_scalar

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

# 1. Define the Datasets
LOG_NAME = "your_log_name_here" # Update this to match your environment
DATASET_RUNS = [
    {'name': 'swift', 'log_name': LOG_NAME, 'start_step': 0, 'end_step': None, 'step_length': 100},
    {'name': 'starling', 'log_name': LOG_NAME, 'start_step': 0, 'end_step': None, 'step_length': 1},
    {'name': 'jackdaw', 'log_name': LOG_NAME, 'start_step': 350, 'end_step': 550, 'step_length': 10},
    {'name': 'jackdaw2', 'log_name': LOG_NAME, 'start_step': 2700, 'end_step': 3460, 'step_length': 20},
]

num_test_scale = 40

# Cache to store loaded datasets so we don't reload them multiple times
loaded_data_cache = {}

def load_scenario_data(run_index):
    """Loads the dataset and computes N array for the scatter plots."""
    if run_index in loaded_data_cache:
        return loaded_data_cache[run_index]
        
    run_params = DATASET_RUNS[run_index]
    name = run_params['name']
    print(f"Loading dataset: {name}...")
    
    scenario_path = os.path.join(os.getcwd(), "scenarios", name)
    config_path = os.path.join(scenario_path, "config.yaml")

    config = SimulationConfig(config_path) 
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    max_steps = dataset.trajectories.shape[0]
    end_step = run_params['end_step']
    effective_end_step = end_step if end_step is not None and end_step <= max_steps else max_steps
    
    step_range = list(range(run_params['start_step'], effective_end_step, run_params['step_length']))
    
    # Calculate scale_range, all_modes, AND params (which contains k and x0)
    scale_range, all_modes, params = compute_scaling_law(dataset, step_range, scenario_path)
    
    # --- ADDED: Precompute N (number of birds) for the background scatters ---
    print(f"Precomputing N array for {name}...")
    N_list = []
    for actual_step in tqdm(step_range):
        positions = dataset.positions_at_time_step(actual_step)
        N_list.append(positions.shape[0])
    N_array = np.array(N_list)
    
    # Save to cache
    loaded_data_cache[run_index] = (dataset, step_range, scale_range, all_modes, params, N_array)
    return loaded_data_cache[run_index]

# 2. Initialize Data
current_run_idx = 0
dataset, step_range, scale_range, all_modes, params, N_array = load_scenario_data(current_run_idx)

# Extract k and x0 from params (Assuming params[:, 0] is k and params[:, 1] is x0 based on power_2pl)
k_array = params[:, 2]
x0_array = params[:, 1]

# 3. Setup the Matplotlib Figure and Axes
# Widened figure further to comfortably fit 4 subplots
fig = plt.figure(figsize=(22, 6))
fig.canvas.manager.set_window_title('Bird Flock Explorer')

plt.subplots_adjust(left=0.08, right=0.98, bottom=0.25, wspace=0.3)

# Plot 1: 2D Scale Space
ax_2d = fig.add_subplot(141)
ax_2d.set_title("Scale Space Mode Count")
ax_2d.set_xlabel("Test Scales (log)")
ax_2d.set_ylabel("Mode Count")
ax_2d.set_xscale('log')
ax_2d.set_yscale('log')

# Plot 2: Fitted k vs N
ax_k = fig.add_subplot(142)
ax_k.set_title("Fitted k vs N")
ax_k.set_xlabel("Number of Birds (N)")
ax_k.set_ylabel("k parameter")

# Plot 3: Fitted x0 vs N
ax_x0 = fig.add_subplot(143)
ax_x0.set_title("Fitted x0 vs N")
ax_x0.set_xlabel("Number of Birds (N)")
ax_x0.set_ylabel("x0 parameter")

# Plot 4: 3D Positions
ax_3d = fig.add_subplot(144, projection='3d')
ax_3d.set_title("3D Bird Positions")
ax_3d.set_xlabel('X')
ax_3d.set_ylabel('Y')
ax_3d.set_zlabel('Z')

# Initialize Line Objects
line_2d, = ax_2d.plot([], [], 'r-o', markersize=4, label='Data')
line_fit, = ax_2d.plot([], [], 'b--', linewidth=2, label='4PL Fit')
ax_2d.legend()

# Scatter objects for k vs N
scat_k = ax_k.scatter(N_array, k_array, c=step_range, cmap='plasma', alpha=0.6)
cbar_k = fig.colorbar(scat_k, ax=ax_k)
cbar_k.set_label('Timestamp (Step)')
highlight_k = ax_k.scatter([], [], color='red', edgecolor='black', s=150, zorder=5, label='Current Step')
ax_k.legend(loc="upper right")

# Scatter objects for x0 vs N
scat_x0 = ax_x0.scatter(N_array, x0_array, c=step_range, cmap='plasma', alpha=0.6)
cbar_x0 = fig.colorbar(scat_x0, ax=ax_x0)
cbar_x0.set_label('Timestamp (Step)')
highlight_x0 = ax_x0.scatter([], [], color='red', edgecolor='black', s=150, zorder=5, label='Current Step')
ax_x0.legend(loc="upper right")

scat_3d = ax_3d.scatter([], [], [], c=[], cmap='viridis', s=10, alpha=0.8)

# 4. Define Widgets
ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
ax_radio = plt.axes([0.02, 0.4, 0.05, 0.2]) # Narrowed radio box slightly

step_slider = Slider(
    ax=ax_slider,
    label='Time Step Index',
    valmin=0,
    valmax=len(step_range) - 1,
    valinit=0,
    valstep=1
)

dataset_names = [d['name'] for d in DATASET_RUNS]
radio = RadioButtons(ax_radio, dataset_names, active=0)

# 5. Update Functions
def update_plot(val):
    idx = int(step_slider.val)
    actual_step = step_range[idx]

    positions = dataset.positions_at_time_step(actual_step)
    N = positions.shape[0]
    
    # Update 2D Line (Data & Fit)
    s_start, s_end = scale_range[idx]
    test_scales = np.logspace(np.log10(s_start), np.log10(s_end), num_test_scale)
    modes = all_modes[idx]
    
    line_2d.set_data(test_scales, modes)
    
    current_k = params[idx, 0]
    current_x0 = params[idx, 1]
    
    # fit_y = power_2pl(test_scales, current_k, current_x0, A=N)
    fit_y = power_3pl(test_scales, *params[idx], A=N, D=1)
    # fit_y = D + (A-D)*(x/x0​)**(-k*gamma)
    # fit_y = N * (test_scales / params[idx,1]) ** (-params[idx,0]*params[idx,2])
    line_fit.set_data(test_scales, fit_y)
    
    ax_2d.relim()
    ax_2d.autoscale_view()
    
    # Update Highlights
    highlight_k.set_offsets([[N, current_k]])
    highlight_x0.set_offsets([[N, current_x0]])
    
    # Update 3D Scatter
    scat_3d._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
    scat_3d.set_array(positions[:, 2]) 
    
    ax_3d.set_xlim(positions[:, 0].min(), positions[:, 0].max())
    ax_3d.set_ylim(positions[:, 1].min(), positions[:, 1].max())
    ax_3d.set_zlim(positions[:, 2].min(), positions[:, 2].max())
    
    ax_3d.set_title(f"3D Bird Positions (Step: {actual_step})")
    fig.canvas.draw_idle()

def update_dataset(label):
    global dataset, step_range, scale_range, all_modes, params, N_array
    global scat_k, cbar_k, highlight_k
    global scat_x0, cbar_x0, highlight_x0
    
    run_idx = dataset_names.index(label)
    dataset, step_range, scale_range, all_modes, params, N_array = load_scenario_data(run_idx)
    
    k_array = params[:, 2]
    x0_array = params[:, 1]
    
    # Reset Slider
    step_slider.eventson = False 
    step_slider.valmax = len(step_range) - 1
    step_slider.ax.set_xlim(step_slider.valmin, step_slider.valmax)
    step_slider.set_val(0)
    step_slider.eventson = True
    
    # Prevent the dreaded colorbar layout error
    if 'cbar_k' in globals() and cbar_k is not None: cbar_k.remove()
    if 'cbar_x0' in globals() and cbar_x0 is not None: cbar_x0.remove()
    
    # Redraw k vs N
    ax_k.clear()
    ax_k.set_title("Fitted k vs N")
    ax_k.set_xlabel("Number of Birds (N)")
    ax_k.set_ylabel("k parameter")
    scat_k = ax_k.scatter(N_array, k_array, c=step_range, cmap='plasma', alpha=0.6)
    cbar_k = fig.colorbar(scat_k, ax=ax_k)
    cbar_k.set_label('Timestamp (Step)')
    highlight_k = ax_k.scatter([], [], color='red', edgecolor='black', s=150, zorder=5, label='Current Step')
    ax_k.legend(loc="upper right")
    
    # Redraw x0 vs N
    ax_x0.clear()
    ax_x0.set_title("Fitted x0 vs N")
    ax_x0.set_xlabel("Number of Birds (N)")
    ax_x0.set_ylabel("x0 parameter")
    scat_x0 = ax_x0.scatter(N_array, x0_array, c=step_range, cmap='plasma', alpha=0.6)
    cbar_x0 = fig.colorbar(scat_x0, ax=ax_x0)
    cbar_x0.set_label('Timestamp (Step)')
    highlight_x0 = ax_x0.scatter([], [], color='red', edgecolor='black', s=150, zorder=5, label='Current Step')
    ax_x0.legend(loc="upper right")
    
    update_plot(0)

# 6. Connect Widgets and Show
step_slider.on_changed(update_plot)
radio.on_clicked(update_dataset)

update_plot(0)
plt.show()