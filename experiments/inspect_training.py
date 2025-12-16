import sys
import os

sys.path.append(os.getcwd()) # To get around relative import issues, I hate Python.

import torch
import numpy as np
from density_field_reconstruction.simulation_config import SimulationConfig
from density_field_reconstruction.dataset import DatasetFactory
from density_field_reconstruction.camera_simulation import MultiCameraSystem
from density_field_reconstruction.density_reconstructor import DensityReconstructor
from density_field_reconstruction.camera_state import CameraState
from density_field_reconstruction.density_field_model import GaussianModel
from density_field_reconstruction.utils import calculate_gmm_ise_gpu

import matplotlib.pyplot as plt

RUN_PARAMS = {
    'name': 'boids',
    'log_name': 'base_init',
    'time_step': 285
}

def inspect_training_at_time_step():
    time_step = RUN_PARAMS['time_step']

    name = RUN_PARAMS['name']
    log_name = RUN_PARAMS['log_name']

    scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
    config_path = os.path.join(scenario_path, "config.yaml")

    log_file_path = os.path.join(scenario_path, *["logs", log_name])
    time_step_file_path = os.path.join(log_file_path, f"t_{time_step:03d}")

    history_path = os.path.join(time_step_file_path, "history_level_0.pth")
    loaded_history = torch.load(history_path, weights_only=False)

    scene_path = os.path.join(time_step_file_path, "scene.pth")
    loaded_scene = torch.load(scene_path, weights_only=False)

    # 3. Load Dataset
    config = SimulationConfig(config_path) 
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    # 4. System Initialization
    cam_system = MultiCameraSystem.create_system(
        intrinsics=config.intrinsics_params, 
        H=config.H, W=config.W, 
        poses=config.cam_poses,
        near_clip=config.near_clip, far_clip=config.far_clip, size=config.size)
    density_reconstructor = DensityReconstructor(config.intrinsics_params, max_iter=config.iter)

    positions = dataset.trajectories[time_step]
    poses, _, images = cam_system.simulate_vision(positions, renderer='gaussian')

    camera_states = []    
    for i, pose in enumerate(poses):
        camera_states.append(
            CameraState(i, config.intrinsics_params, pose, device='cuda')
        )
    
    prev_model_path = os.path.join(log_file_path, f"t_{time_step-1:03d}", "checkpoint_level_0.pth")
    training_history = GaussianModel.load_training_history(prev_model_path)

    iter = 99
    prev_model = GaussianModel.load_iter(training_history, iter)
    
    density_reconstructor.scale = loaded_scene['scale']
    model, scale_spaces = \
        density_reconstructor.process_frame(camera_states, images, positions=positions,
                                            initGMM=[prev_model],
                                            is_adaptive_scale=False,
                                            is_log=True,
                                            output_dir="/",
                                            debug=False)
    
    # 6. 
    plt.figure()
    plt.plot(np.arange(100), model[0].metrics_history['loss_history'])
    plt.plot(np.arange(100), np.array(loaded_history['loss_history']))
    plt.show()
    # np.array(model[0].metrics_history['loss_history']) - np.array(loaded_history['loss_history'])

if __name__ == "__main__":
    inspect_training_at_time_step()