import logging
import sys
import os
import shutil
import csv
import itertools
from tqdm import tqdm


import numpy as np
from dfr.simulation_config import SimulationConfig
from dfr.dataset_io import DatasetFactory
from dfr.camera_system import MultiCameraSystem
from dfr.density_field_reconstructor import DensityReconstructor
from dfr.camera_state import CameraState
from dfr.utils import calculate_gmm_dissimilarity

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

USE_DECOUPLED = False

LOG_NAME = 'base'
DATASET_RUNS = [
    {
        'name': 'starling',
        'log_name': LOG_NAME,
        'start_step': 0,
        'end_step': None,
        'step_length': 1,
    },
    {
        'name': 'boids_multi',
        'log_name': LOG_NAME,
        'start_step': 0,
        'end_step': None,
        'step_length': 20,
    },
    {
        'name': 'clutter',
        'log_name': LOG_NAME,
        'start_step': 0,
        'end_step': None,
        'step_length': 20,
    },
]

# --- Define your search grid here ---
PARAM_GRID = {
    'xyz_lr_c': [0.01, 0.05, 0.1],
    'xyz_lr_final_c': [0.7, 0.8, 0.9],
    'radius_lr_c': [0.05, 0.1, 0.2],
    'radius_lr_final_c': [0.7, 0.8, 0.9],
    'weights_lr_c': [0.1, 0.14, 0.2],
    'weights_lr_final_c': [0.7, 0.8, 0.9],
}

def get_run_signature(dataset_name, hyperparams):
    """Creates a unique tuple identifier for a run to check if it's already completed."""
    return (
        dataset_name,
        float(hyperparams['xyz_lr_c']), float(hyperparams['xyz_lr_final_c']),
        float(hyperparams['radius_lr_c']), float(hyperparams['radius_lr_final_c']),
        float(hyperparams['weights_lr_c']), float(hyperparams['weights_lr_final_c'])
    )

def run_multi_scenarios():
    keys, values = zip(*PARAM_GRID.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results_file = "grid_search_results.csv"
    completed_runs = set()
    
    # 1. Parse existing results to allow for resuming after interruption
    if os.path.exists(results_file):
        with open(results_file, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Reconstruct the signature from the CSV row
                run_sig = (
                    row['dataset_name'],
                    float(row['xyz_lr_c']), float(row['xyz_lr_final_c']),
                    float(row['radius_lr_c']), float(row['radius_lr_final_c']),
                    float(row['weights_lr_c']), float(row['weights_lr_final_c'])
                )
                completed_runs.add(run_sig)
                
        logger.info(f"Found {len(completed_runs)} already completed runs in {results_file}. Skipping those.")

    # 2. Open the results file in append mode
    file_exists = os.path.exists(results_file)
    with open(results_file, mode='a', newline='') as f:
        fieldnames = ['dataset_name'] + list(keys) + ['mean_training_loss', 'mean_density_field_loss']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()

        # 3. Iterate through datasets and experiments
        for run_params in DATASET_RUNS:
            dataset_name = run_params['name']
            
            # --- NEW: Outer tqdm progress bar ---
            exp_pbar = tqdm(enumerate(experiments), total=len(experiments), position=0, leave=True)
            
            for exp_idx, hyperparams in exp_pbar:
                # Dynamically update the description of the progress bar
                exp_pbar.set_description(f"Running {dataset_name} | Exp {exp_idx+1}/{len(experiments)}")
                
                run_sig = get_run_signature(dataset_name, hyperparams)
                
                if run_sig in completed_runs:
                    # Skip silently so we don't spam the console or break the tqdm bar visually
                    continue
                
                # Run the scenario and get the requested tuple back
                results = run_single_scenario(run_params, hyperparams, exp_idx)
                
                if results is None:
                    continue # Scenario was skipped internally (e.g., start_step >= end_step)
                    
                mean_training_loss, mean_density_loss = results
                
                # Write the results immediately
                row_data = {'dataset_name': dataset_name}
                row_data.update(hyperparams)
                row_data['mean_training_loss'] = mean_training_loss
                row_data['mean_density_field_loss'] = mean_density_loss
                
                writer.writerow(row_data)
                f.flush() # Forces Python to save to disk immediately

def run_single_scenario(run_params, hyperparams, exp_idx):
    # 1. Parameter extraction and Logging Setup
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

    # 2. Initialize Metrics (must be re-initialized for each run)
    loss_metrics = {
        'final_training_loss': [],
        'final_density_field_loss': [],
        'final_gmm_num': [],
        'scale': []
    }

    # 3. Load Dataset
    config = SimulationConfig(config_path) 
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    max_steps = dataset.trajectories.shape[0]
    effective_end_step = end_step if end_step is not None and end_step <= max_steps else max_steps
    
    if start_step >= effective_end_step:
        logger.info(f"Skipping {name}: start_step ({start_step}) >= end_step ({effective_end_step}).")
        return None

    # 4. System Initialization
    cam_system = MultiCameraSystem.create_homogeneous_system(
        state_class=CameraState,
        intrinsics=config.intrinsics_params,
        H=config.H, W=config.W, 
        poses_or_RTs=config.cam_poses,
        near_clip=config.near_clip, far_clip=config.far_clip, 
        size=config.size,
        device='cuda')
    density_reconstructor = DensityReconstructor(max_iter=config.iter, use_decoupled=USE_DECOUPLED)

    # 5. Simulation Loop
    step_range = range(start_step, effective_end_step, step_length)
    model = None

    # for time_step in (pbar := tqdm(step_range, desc=f"Processing {name}")):
    for time_step in step_range:
        positions = dataset.positions_at_time_step(time_step)
        poses, _, images, masks = cam_system.simulate_vision(positions, renderer='gaussian')
        
        # Setup Reconstructor Meta-Parameters
        xyz_reg=None
        radius_reg=None
        lr_max_steps=100
        
        model, scale_spaces = \
        density_reconstructor.process_frame(cam_system, images, positions=positions,
                                            initGMM=None,
                                            is_adaptive_scale=True,
                                            is_store_intermediate=False, is_log=False,
                                            output_dir=os.path.join(log_file_path, f"t_{time_step:03d}"),
                                            debug=True,
                                            xyz_lr_c=hyperparams['xyz_lr_c'], 
                                            xyz_lr_final_c=hyperparams['xyz_lr_final_c'], 
                                            radius_lr_c=hyperparams['radius_lr_c'], 
                                            radius_lr_final_c=hyperparams['radius_lr_final_c'], 
                                            weights_lr_c=hyperparams['weights_lr_c'], 
                                            weights_lr_final_c=hyperparams['weights_lr_final_c'], 
                                            xyz_reg=xyz_reg, radius_reg=radius_reg,
                                            lr_max_steps=lr_max_steps)
        
        # 6. Collect Metrics
        loss_metrics['final_training_loss'].append(model[0].mean_loss)

        is_visible = np.ones((positions.shape[0],), dtype=np.bool)
        for i in range(len(poses)):
            is_visible = is_visible & masks[i]
        
        loss_metrics['final_density_field_loss'].append(
            calculate_gmm_dissimilarity(
                positions[is_visible],
                density_reconstructor.scale, 
                model[0]._xyz, 
                model[0]._weights, 
                model[0]._radius, use_decoupled=USE_DECOUPLED))
    
    # 7. Logging and Data Saving
    mean_training_loss = np.mean(loss_metrics['final_training_loss'])
    mean_density_loss = np.mean(loss_metrics['final_density_field_loss'])
    
    return (mean_training_loss, mean_density_loss)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def identify_pareto_front(df, metric_x, metric_y):
    """Finds the non-dominated points (Pareto front) assuming we want to MINIMIZE both metrics."""
    # Sort by X, then Y
    sorted_df = df.sort_values(by=[metric_x, metric_y])
    pareto_front = []
    
    min_y_so_far = float('inf')
    for index, row in sorted_df.iterrows():
        if row[metric_y] < min_y_so_far:
            pareto_front.append(index)
            min_y_so_far = row[metric_y]
            
    return df.loc[pareto_front]

def visualize_normalized_tradeoffs(csv_file='grid_search_results.csv'):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        return

    raw_x = 'mean_training_loss'
    raw_y = 'mean_density_field_loss'
    norm_x = 'norm_training_loss'
    norm_y = 'norm_density_field_loss'

    # 1. Normalize the metrics PER SCENARIO (Min-Max scaling)
    # This forces all metrics into a 0.0 (Best) to 1.0 (Worst) scale based on that scenario's min/max
    df[norm_x] = df.groupby('dataset_name')[raw_x].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    # df[norm_y] = df.groupby('dataset_name')[raw_y].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    df[norm_y] = df.groupby('dataset_name')[raw_y].transform(lambda x: x)

    param_cols = [col for col in df.columns if col not in ['dataset_name', raw_x, raw_y, norm_x, norm_y]]

    # 2. Average the NORMALIZED metrics
    avg_df = df.groupby(param_cols)[[norm_x, norm_y]].mean().reset_index()

    # 3. Find the Pareto Front on the normalized data
    pareto_df = identify_pareto_front(avg_df, norm_x, norm_y)
    print(pareto_df)

    # 4. Create the Plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all combinations
    sns.scatterplot(data=avg_df, x=norm_x, y=norm_y, ax=ax, color='lightgray', alpha=0.7, label='All Parameter Sets')
    
    # Highlight Pareto front
    sns.scatterplot(data=pareto_df, x=norm_x, y=norm_y, ax=ax, color='red', s=100, label='Pareto Optimal (Best Balance)')
    sns.lineplot(data=pareto_df, x=norm_x, y=norm_y, ax=ax, color='red', alpha=0.5)

    for _, row in pareto_df.iterrows():
        label = f"idx:{row.name}" 
        ax.annotate(label, (row[norm_x], row[norm_y]), xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_title('Average NORMALIZED Performance Across All Scenarios\n(0.0 is the best possible score in every scenario)')
    ax.set_xlabel('Average Normalized Training Loss (Lower is Better)')
    ax.set_ylabel('Average Normalized Density Field Loss (Lower is Better)')
    ax.legend()
    plt.tight_layout()
    plt.show()

    print("=== Best Parameter Sets (Based on Normalized Averages) ===")
    # Print the parameters along with their normalized scores
    print(pareto_df[param_cols + [norm_x, norm_y]].to_string(index=False))

if __name__ == "__main__":
    # run_multi_scenarios()

    visualize_normalized_tradeoffs()
    plt.show()