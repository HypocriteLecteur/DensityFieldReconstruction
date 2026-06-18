import logging
import shutil
import sys
import os
import matplotlib
import pandas as pd

from tqdm import tqdm


import torch
import numpy as np
from dfr.simulation_config import SimulationConfig
from dfr.dataset_io import DatasetFactory, load_camera_extrinsics
from dfr.density_field_reconstructor import DensityReconstructor
from dfr.camera_state import CameraStateUE4
from dfr.utils import calculate_gmm_dissimilarity
from dfr.visualizer import SimulationVisualizer, MultiGMMPlotter
from dfr.camera_system import MultiCameraSystem
from dfr.gaussian_mixture_reduction import GMR

import glob
import cv2
from scipy.io import loadmat
import matplotlib.pyplot as plt

from dfr.utils import move_figure

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

def thresholding(rgb_image):
    """
    Thresholds an RGB image using HSV color space, replicating the MATLAB
    colorThresholder output logic, including the wraparound Hue detection.

    Args:
        rgb_image (np.ndarray): The input image (expected to be in RGB color
                                order, e.g., loaded or converted).

    Returns:
        tuple: (BW, maskedRGBImage)
            - BW (np.ndarray): The binary mask (2D array, dtype uint8).
            - maskedRGBImage (np.ndarray): The original image with the
              background masked out (3D array, same size as input).
    """

    # 1. Convert RGB image to HSV color space
    # Assuming input is RGB. If image was loaded with cv2.imread, it will be BGR
    # and you should use cv2.COLOR_BGR2HSV instead.
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # 2. Define thresholds for 8-bit OpenCV HSV ranges (H: 0-179, S/V: 0-255)
    # Scale MATLAB's 0-1 thresholds: H * 179, S/V * 255
    channel1Min_cv = int(0.840 * 179)  # H_Min (Upper Red: ~150)
    channel1Max_cv = 179               # H_Max (Upper Red: 179)
    channel1Min2_cv = 0                # H_Min (Lower Red: 0)
    channel1Max2_cv = int(0.348 * 179) # H_Max (Lower Red: ~62)

    channel2Min_cv = int(0.000 * 255)  # S_Min (0)
    channel2Max_cv = int(1.000 * 255)  # S_Max (255)

    channel3Min_cv = int(0.000 * 255)  # V_Min (0)
    channel3Max_cv = int(1.000 * 255)  # V_Max (255)

    # 3. Create two masks for the wrap-around Hue selection (Red color)
    # Mask 1: Upper Red/Magenta range (e.g., [150, 179])
    lower_bound_1 = np.array([channel1Min_cv, channel2Min_cv, channel3Min_cv])
    upper_bound_1 = np.array([channel1Max_cv, channel2Max_cv, channel3Max_cv])
    mask_1 = cv2.inRange(hsv_image, lower_bound_1, upper_bound_1)

    # Mask 2: Lower Red range (e.g., [0, 62])
    lower_bound_2 = np.array([channel1Min2_cv, channel2Min_cv, channel3Min_cv])
    upper_bound_2 = np.array([channel1Max2_cv, channel2Max_cv, channel3Max_cv])
    mask_2 = cv2.inRange(hsv_image, lower_bound_2, upper_bound_2)

    # 4. Combine the two masks (logical OR) to get the final binary mask (BW)
    BW = cv2.bitwise_not(cv2.bitwise_or(mask_1, mask_2))

    # 5. Apply the mask to the original image
    # Note: cv2.bitwise_and is the standard way to set non-masked pixels to 0
    # The mask must be 2D (BW)
    maskedRGBImage = cv2.bitwise_and(rgb_image, rgb_image, mask=BW)

    return BW, maskedRGBImage

def find_centroids(binary_mask, min_area_threshold=10):
    """
    Finds the centroid (center of mass) of each white blob (contour) in a
    binary mask image.

    Args:
        binary_mask (np.ndarray): The 8-bit, single-channel binary mask (BW).
        min_area_threshold (int): Minimum contour area to be considered valid.

    Returns:
        list: A list of tuples, where each tuple is (x, y) centroid coordinates.
    """
    # Find contours (RETR_EXTERNAL looks for only external contours)
    # Note: The output structure of cv2.findContours changed in newer versions,
    # but this line handles both common structures.
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    
    for c in contours:
        # Filter small noise/artifacts
        if cv2.contourArea(c) < min_area_threshold:
            continue

        # Calculate moments for each contour
        M = cv2.moments(c)

        # Calculate centroid (center of mass)
        # Check if m00 (area moment) is zero to prevent division by zero
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))

    return centroids

def main():
    LOG_NAME = 'base_better_peak'
    CLEAN_LOGS = False
    USE_DECOUPLED = False

    run_params = {
        'name': 'ue4',
        'log_name': LOG_NAME,
        'start_step': 0,
        'end_step': None,
        'step_length': 1,
    }

    # 1. Parameter extraction and Logging Setup
    name = run_params['name']
    log_name = run_params['log_name']
    start_step = run_params['start_step']
    end_step = run_params['end_step']
    step_length = run_params['step_length']

    scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
    config_path = os.path.join(scenario_path, "config.yaml")

    if CLEAN_LOGS and os.path.exists(os.path.join(scenario_path, "logs")):
        shutil.rmtree(os.path.join(scenario_path, "logs"))

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
        'final_gmm_num': [],
        'scale': []
    }

    # 3. Load Dataset
    config = SimulationConfig(config_path) 
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    cam_poses_csv = pd.read_csv(config.cam_poses, header=None)
    cam_poses_csv.fillna('', inplace=True)  # Fill NaN with empty strings

    max_steps = dataset.trajectories.shape[0]
    effective_end_step = end_step if end_step is not None and end_step <= max_steps else max_steps
    
    if start_step >= effective_end_step:
        logger.info(f"Skipping {name}: start_step ({start_step}) >= end_step ({effective_end_step}).")
        return

    # 4. System Initialization
    density_reconstructor = DensityReconstructor(max_iter=config.iter, use_decoupled=USE_DECOUPLED)

    # visualizer = SimulationVisualizer(intrinsics_params=config.intrinsics_params,
    #                                   H=config.H, W=config.W, 
    #                                   cam_num=3,
    #                                   mode='all',
    #                                   save_video=False, fps=30, dpi=100,
    #                                   positions_all=dataset.trajectories)
    # mgmm_visualizer = MultiGMMVisualizer(H=config.H, W=config.W,
    #                                      near_clip=config.near_clip, far_clip=config.far_clip)

    # 5. Simulation Loop
    step_range = range(start_step, effective_end_step, step_length)

    def convert_coordinate(R1_input, T1_input):
        # --- COORDINATE SYSTEM CONVERSION ---
        # Matrix to swap X and Y axes 
        # (Maps: X_old->Y_new, Y_old->X_new, Z_old->Z_new)
        M_swap = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=R1_input.dtype)
        
        # 1. Swap the X and Y translation coordinates
        T_converted = M_swap @ T1_input
        
        # 2. Swap the X and Y basis columns in the Rotation matrix
        R_converted = M_swap @ R1_input @ M_swap

        return R_converted, T_converted

    for time_step in (pbar := tqdm(step_range, desc=f"Processing {os.name}")):
        positions = dataset.positions_at_time_step(time_step)

        # read poses from csv
        R1_input = cam_poses_csv.iloc[(time_step+1)*3:(time_step+2)*3][[2, 3, 4]].to_numpy().astype('float')
        R2_input = cam_poses_csv.iloc[(time_step+1)*3:(time_step+2)*3][[6, 7, 8]].to_numpy().astype('float')
        R3_input = cam_poses_csv.iloc[(time_step+1)*3:(time_step+2)*3][[10, 11, 12]].to_numpy().astype('float')
        T1_input = cam_poses_csv.iloc[(time_step+1)*3:(time_step+2)*3][1].to_numpy().astype('float')
        T2_input = cam_poses_csv.iloc[(time_step+1)*3:(time_step+2)*3][5].to_numpy().astype('float')
        T3_input = cam_poses_csv.iloc[(time_step+1)*3:(time_step+2)*3][9].to_numpy().astype('float')

        # correct the 90 deg rotation
        R_ = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ])
        R1_input = R_ @ R1_input
        R2_input = R_ @ R2_input
        R3_input = R_ @ R3_input

        # preprocess RT step 1 left-hand to right-hand
        RTs = [
            [*convert_coordinate(R1_input, T1_input)],
            [*convert_coordinate(R2_input, T2_input)],
            [*convert_coordinate(R3_input, T3_input)]
        ]

        # P_wrd = RP_base + T

        cam_system = MultiCameraSystem.create_homogeneous_system(
            state_class=CameraStateUE4,
            intrinsics=config.intrinsics_params,
            H=config.H, W=config.W, 
            poses_or_RTs=RTs,
            near_clip=config.near_clip, far_clip=config.far_clip, 
            size=config.size,
            device='cuda')

        # visualizer.update(time_step=time_step,
        #     # positions=positions_all[time_step],
        #     cam_poses=poses,
        #     imgs=scale_spaces)
        
        # fig = plt.figure()
        # move_figure(fig, 2800, 100)
        # ax = fig.add_subplot(111, projection='3d')

        # ax.scatter3D(positions[:, 0], positions[:, 1], positions[:, 2], color='red', label='True Positions')
        # color_='cyan'
        # for camera in cam_system.cameras:
        #     vertices = camera.state.get_world_frustum()
        #     # Define the 12 edges connecting the 8 frustum corners
        #     edges = [
        #         [0, 1], [1, 2], [2, 3], [3, 0],  # Near plane
        #         [4, 5], [5, 6], [6, 7], [7, 4],  # Far plane
        #         [0, 4], [1, 5], [2, 6], [3, 7]   # Side walls
        #     ]
            
        #     plotted_objects = []
        #     for edge in edges:
        #         line, = ax.plot(
        #             vertices[edge, 0], vertices[edge, 1], vertices[edge, 2], 
        #             color=color_, alpha=0.6, linewidth=1
        #         )
        #         plotted_objects.append(line)
                
        #     # Plot the camera optical center
        #     center = camera.state.camera_center
        #     cam_pt = ax.scatter(*center, c=color_, marker='s', s=30)

        # plt.xlabel('x')
        # plt.ylabel('y')
        # ax.set_aspect('equal', adjustable='box')
        # ax.legend()
        # plt.show()

        img_idx = time_step + 1

        img = cv2.imread(f'D:\\WindowsNoEditor\\picture1\\{img_idx}.jpg')
        img2 = cv2.imread(f'D:\\WindowsNoEditor\\picture2\\{img_idx}.jpg')
        img3 = cv2.imread(f'D:\\WindowsNoEditor\\picture3\\{img_idx}.jpg')

        # Pre-processing
        BW, _ = thresholding(img)
        centroids = find_centroids(BW, min_area_threshold=10)
        BW2, _ = thresholding(img2)
        centroids2 = find_centroids(BW2, min_area_threshold=10)
        BW3, _ = thresholding(img3)
        centroids3 = find_centroids(BW3, min_area_threshold=10)
    
        # Visualization
        # for (x, y) in centroids:
        #     cv2.circle(img, (x, y), 2, (0, 0, 255), -1) 
        # for (x, y) in centroids2:
        #     cv2.circle(img2, (x, y), 2, (0, 0, 255), -1)

        # cv2.imwrite(f"mask.jpg", BW)
        # cv2.imwrite(f"aero_{1}_detection.jpg", img)
        # cv2.imwrite(f"aero_{3}_detection.jpg", img2)
        # cv2.imshow(f"aero_{1}_detection", img)
        # cv2.imshow(f"aero_{3}_detection", img2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        centroids = np.array(centroids)
        centroids2 = np.array(centroids2)
        centroids3 = np.array(centroids3)
        
        density_reconstructor = DensityReconstructor(max_iter=100, W=config.W, H=config.H)

        reconstruction_params = {
            'targetd_num_mode': 10,
            # voxel method
            'voxel_scale': 2,
            'voxel_peak_threshold': 0.3,
            'voxel_grid_max_size': 32,
            'voxel_peaks_number': 10
        }
        train_params = {
            'xyz_lr_c': 0.05,
            'xyz_lr_final_c': 0.9,
            'radius_lr_c': 0.05,
            'radius_lr_final_c': 0.9,
            'weights_lr_c': 0.10,
            'weights_lr_final_c': 0.7,
            'xyz_reg': 1.0,
            'radius_reg': 0.3,
            'radius_cutoff_inv': 0.5,
            'lr_max_steps': 500
        }

        # poses, projections, _, masks = cam_system.simulate_vision(positions, renderer='projection_only', is_auto_aim=False)
        # plt.figure()
        # plt.imshow(img)
        # plt.scatter(projections[0][:, 0], projections[0][:, 1], label='projection')
        # plt.scatter(centroids[:, 0], centroids[:, 1], label='detection')
        # plt.legend()
        # plt.show()

        model, scale_spaces = \
        density_reconstructor.process_frame(cam_system, point_sets=[centroids, centroids2, centroids3], positions=positions,
                                            initGMM=None,
                                            is_adaptive_scale=True, scale=None,
                                            is_log=False,
                                            train_params=train_params,
                                            reconstruction_params=reconstruction_params)
        
        # for metric_name, value in density_reconstructor.time_metrics.items():
        #     time_metrics[metric_name].append(value)

        gmm_visualizer = MultiGMMPlotter()
        gmm_visualizer.add_gmm(model[0]._xyz.detach().cpu().numpy(), model[0]._radius.detach().cpu().numpy(), model[0]._weights.detach().cpu().numpy())
        # gmm_visualizer.add_gmm(r_means.detach().cpu().numpy(), r_radius.detach().cpu().numpy(), r_weights.detach().cpu().numpy())
        gmm_visualizer.update()

        gmm_visualizer.ax.scatter(
            positions[:, 0], 
            positions[:, 1], 
            positions[:, 2], 
            c='#1f2937',        # A sophisticated dark slate/navy instead of pure black
            s=8,                # Drastically reduce size (down from 15 to 3 or 4)
            alpha=0.65,         # Allow dense areas to visually compound
            edgecolors='none',  
            depthshade=True,    # Crucial for 3D: fades points that are further away
            zorder=3            
        )
        gmm_visualizer.ax.set_xlabel('X')
        gmm_visualizer.ax.set_ylabel('Y')
        gmm_visualizer.ax.set_zlabel('Z')

        plt.show()

        # loss_metrics['final_training_loss'].append(model[0].sum_loss)
        # loss_metrics['final_gmm_num'].append(model[0]._xyz.shape[0])

        # _, projections, _, _ = stereo_vision.simulate_vision(positions_all[time_step], renderer='gaussian')
        # swarm_projection, swarm_projection2 = projections
        # is_visible = (swarm_projection[:, 0] > 0).squeeze() & (swarm_projection[:, 1] > 0).squeeze() & \
        #     (swarm_projection[:, 0] < H).squeeze() & (swarm_projection[:, 1] < W).squeeze()
        # is_visible2 = (swarm_projection2[:, 0] > 0).squeeze() & (swarm_projection2[:, 1] > 0).squeeze() & \
        #     (swarm_projection2[:, 0] < H).squeeze() & (swarm_projection2[:, 1] < W).squeeze()
        # is_visible = np.logical_and(is_visible, is_visible2)
        # loss_metrics['final_density_field_loss'].append(
        #     calculate_gmm_dissimilarity(
        #         positions_all[time_step],
        #         density_reconstructor.scale, 
        #         model[0]._xyz, 
        #         model[0]._weights, 
        #         model[0]._radius))

        # visualizer.update(time_step=time_step,
        #                   positions=positions_all[time_step],
        #                   R1=camera_states[0].P_np[:, :3], T1=camera_states[0].P_np[:, 3], 
        #                   R2=camera_states[1].P_np[:, :3], T2=camera_states[1].P_np[:, 3], 
        #                   img=scale_spaces[0][0], img2=scale_spaces[1][0])
        
        # CameraState and CameraStateUE4 has differnt frames
        # visualizer.update(time_step=time_step,
        #             positions=positions_all[time_step],
        #             cam_poses=poses,
        #             imgs=scale_spaces)

        # if time_step == 1:
        #     gmm1_id = mgmm_visualizer.add_gmm(model[0]._xyz.detach().cpu().numpy(), 
        #                                       model[0]._radius.detach().cpu().numpy(), 
        #                                       model[0]._weights.detach().cpu().numpy(), color='blue', label='baseline')
        #     gmm2_id = mgmm_visualizer.add_gmm(r_means.detach().cpu().numpy(), 
        #                                       r_covs.detach().cpu().numpy(), 
        #                                       r_weights.detach().cpu().numpy(), color='orange', label='GMR')
        # else:
        #     mgmm_visualizer.update_gmm_data(gmm1_id, 
        #                                     means=model[0]._xyz.detach().cpu().numpy(), 
        #                                     covariances=model[0]._radius.detach().cpu().numpy(), 
        #                                     weights=model[0]._weights.detach().cpu().numpy(), visible=True)
        #     mgmm_visualizer.update_gmm_data(gmm2_id, 
        #                                     means=r_means.detach().cpu().numpy(), 
        #                                     covariances=r_covs.detach().cpu().numpy(), 
        #                                     weights=r_weights.detach().cpu().numpy(), visible=True)
        # mgmm_visualizer.update(
        #     real_means=positions_all[time_step],
        # )
        # plt.pause(0.001)

    # if time_metrics['train_gaussian_scale_space']:
    #     mean_time = np.mean(np.array(time_metrics['train_gaussian_scale_space'][1:]))
    #     std_time = np.std(np.array(time_metrics['train_gaussian_scale_space'][1:]))
    #     print(f"Mean 'train_gaussian_scale_space' time: {mean_time:.2f} ms +- {std_time:.2f} ms")
    # else:
    #     print("No time steps procesed.")

    # if time_metrics['adaptive_scale_selection']:
    #     mean_time = np.mean(np.array(time_metrics['adaptive_scale_selection'][1:]))
    #     std_time = np.std(np.array(time_metrics['adaptive_scale_selection'][1:]))
    #     print(f"Mean 'adaptive_scale_selection' time: {mean_time:.2f} ms +- {std_time:.2f} ms")
    # else:
    #     print("No time steps processed.")

if __name__ == "__main__":
    main()