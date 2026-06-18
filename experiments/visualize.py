import sys
import os


from dfr.utils import move_figure
from dfr.visualizer import GMMInteractivePlotter

LOG_NAME = "base_reg_cam_2"
LOG_NAME2 = "base_reg_cam_2"

DATASET_RUNS = [
    # {
    #     'name': 'starling',
    #     'start_step': 0,
    #     'end_step': None,
    #     'step_length': 1,
    # },
    # {
    #     'name': 'swift',
    #     'start_step': 0,
    #     'end_step': None,
    #     'step_length': 200,
    # },
    # {
    #     'name': 'jackdaw',
    #     'start_step': 350,
    #     'end_step': 550,
    #     'step_length': 10,
    # },
    {
        'name': 'Point3D_N68_t2.35_Xianjiahu_20231121b_data50',
        'start_step': 0,
        'end_step': None,
        'step_length': 5,
    },
]

def run_interactive_gmm_plotter(config_path: str, log_file_path: str, log_file2_path: str=None, start_step: int=0, end_step=None, step_length: int=1):
    """
    Initializes and runs the interactive GMM visualization tool.

    Args:
        config_path (str): Path to the simulation configuration file (e.g., "boids_config.yaml").
        log_file_path (str): Path to the directory containing time-stepped logs and checkpoints (e.g., "logs/boids_initGMM").
    """
    plotter = GMMInteractivePlotter(config_path, log_file_path, log_file2_path=log_file2_path, 
                                    start_step=start_step, end_step=end_step, step_length=step_length)
    move_figure(plotter.fig, 2800, 100)
    plotter.run()

if __name__ == "__main__":
    selcted_dataset = DATASET_RUNS[0]

    scenario_path = os.path.join(os.getcwd(), *["scenarios", selcted_dataset['name']])
    config_path = os.path.join(scenario_path, "config.yaml")
    log_file_path = os.path.join(scenario_path, *["logs", LOG_NAME])
    log_file2_path = os.path.join(scenario_path, *["logs", LOG_NAME2])
    
    # You would typically add logic here to ensure these files exist or handle defaults
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}. Cannot run plotter.")
    elif not os.path.exists(log_file_path):
         print(f"Error: Log directory not found at {log_file_path}. Cannot run plotter.")
    else:
        run_interactive_gmm_plotter(config_path, log_file_path, log_file2_path, 
                                    start_step=selcted_dataset['start_step'],
                                    end_step=selcted_dataset['end_step'],
                                    step_length=selcted_dataset['step_length'])