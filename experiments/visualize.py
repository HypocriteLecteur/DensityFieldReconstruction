import sys
import os

sys.path.append(os.getcwd()) # To get around relative import issues, I hate Python.

from density_field_reconstruction.visualizer import GMMInteractivePlotter

def run_interactive_gmm_plotter(config_path: str, log_file_path: str, log_file2_path: str=None):
    """
    Initializes and runs the interactive GMM visualization tool.

    Args:
        config_path (str): Path to the simulation configuration file (e.g., "boids_config.yaml").
        log_file_path (str): Path to the directory containing time-stepped logs and checkpoints (e.g., "logs/boids_initGMM").
    """
    try:
        plotter = GMMInteractivePlotter(config_path, log_file_path, log_file2_path=log_file2_path)
        plotter.run()
    except Exception as e:
        print(f"An error occurred during plotter setup or execution: {e}")
        # Re-raise for debugging if needed
        # raise

if __name__ == "__main__":
    name = "boids"
    log_name = "base_init"
    log_name2 = "base_init"

    scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
    config_path = os.path.join(scenario_path, "config.yaml")
    log_file_path = os.path.join(scenario_path, *["logs", log_name])
    log_file2_path = os.path.join(scenario_path, *["logs", log_name2])
    
    # You would typically add logic here to ensure these files exist or handle defaults
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}. Cannot run plotter.")
    elif not os.path.exists(log_file_path):
         print(f"Error: Log directory not found at {log_file_path}. Cannot run plotter.")
    else:
        run_interactive_gmm_plotter(config_path, log_file_path, log_file2_path)