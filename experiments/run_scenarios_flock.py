import logging
import sys
import os
import shutil
from tqdm import tqdm
import glob


import cv2
import time
import torch
import numpy as np
from dfr.simulation_config import SimulationConfig
from dfr.dataset_io import DatasetFactory
from dfr.camera_system import MultiCameraSystem
from dfr.density_field_reconstructor import DensityReconstructor
from dfr.density_field_model import GaussianModel
from dfr.camera_state import CameraState
from dfr.utils import calculate_gmm_dissimilarity, generate_encircling_cameras, compute_metrics_batched_torch
from dfr.visualizer import MultiGMMPlotter
from dfr.gaussian_mixture_reduction import GMR
from gaussian_rasterizer_simple_large import rasterize_gaussians
from dfr.utils import move_figure
import json
from scipy.spatial.transform import Rotation as R
import scipy.io
import scipy
from dfr.mode_finding import mode_counting
import pandas as pd
from dfr.visualizer import SimulationVisualizer

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

IS_LOGGING = True
CLEAN_LOGS = False

USE_DECOUPLED = False
USE_GT_SCALE = True

CAM_NUM = 2
LOG_NAME = 'base_reg_cam_2'

VIS_LOG_NAME = LOG_NAME
VIS_LOG_NAME2 = LOG_NAME
DATASET_VIS = [
    {
        'name': "Point3D_N68_t2.35_Xianjiahu_20231121b_data50",
        'log_name': VIS_LOG_NAME,
        'log_name2': VIS_LOG_NAME2,
    },
]

RUN_PARAMS = {
    'name': "Point3D_N68_t2.35_Xianjiahu_20231121b_data50",
    'log_name': LOG_NAME,
    'start_step': 0,
    'end_step': None,
    'step_length': 5
    }

def find_target_scale(func, targetd_num_mode, s_low=0, s_high=30, atol=1e-5):
    for _ in range(100):
        if func((s_low + s_high) / 2.0) == targetd_num_mode:
            break

        mid = (s_low + s_high) / 2.0
        val = func(mid)

        # Standard binary search for monotonic decreasing function
        if val > targetd_num_mode:
            s_low = mid # The target is to the right (larger s needed to reduce n)
        else:
            s_high = mid # The target is to the left
    
    return (s_low + s_high) / 2.0

def load_camera_extrinsics(extrinsics_json_path):
    """
    Loads camera parameters from a JSON file and constructs the extrinsics 
    transformation matrices for three cameras.
        
    Returns:
    -------
    transform1, transform2, transform3 : np.ndarray
        $4 \times 4$ rigid transformation matrices representing the camera extrinsics.
    """

    # Load and decode JSON data
    with open(extrinsics_json_path, 'r', encoding='utf-8') as f:
        cam_data = json.load(f)
        
    # Helper lambda to convert dot notation from MATLAB struct to Python dictionary lookups
    # Handles nested structures safely
    def get_val(keys_str):
        keys = keys_str.split('.')
        val = cam_data
        for k in keys:
            val = val[k]
        return val

    # Helper to construct 4x4 rigid transformation matrix from Euler angles (deg) and Translation
    # Note: MATLAB's rigidtform3d by default expects intrinsic ZYX convention for 3 element inputs.
    def create_rigid_transform(euler_deg, translation):
        # Match MATLAB's default behavior for rigidtform3d (Intrinsic ZYX)
        rot = R.from_euler('ZYX', [euler_deg[2], euler_deg[1], euler_deg[0]], degrees=True)
        t_matrix = np.eye(4)
        t_matrix[:3, :3] = rot.as_matrix()
        t_matrix[:3, 3] = translation
        return t_matrix

    # ==========================================
    # CAM1 Pose
    # ==========================================
    ea_cam1_x = get_val('CAM1.sensor_X_dir') - 90
    ea_cam1_y = get_val('CAM1.sensor_Y_dir')
    ea_cam1_z = 0
    euler_angles1 = [ea_cam1_x, ea_cam1_y, ea_cam1_z]
    
    abs_cam_pos1 = [0, 0, 0]
    transform1 = create_rigid_transform(euler_angles1, abs_cam_pos1)

    # ==========================================
    # CAM2 Pose
    # ==========================================
    # Angle sensor
    azimuth_cam2_in_cam1 = np.deg2rad(abs(get_val('CAM1.neg_x_axis_dir') + 180 - get_val('CAM1.CAM1_to_CAM2_dir')))
    # CAM2 orientation
    alpha2_1 = (abs(get_val('CAM1.neg_x_axis_dir') + 180 - get_val('CAM1.CAM1_to_CAM2_dir'))) % 360
    alpha2_2 = (abs(get_val('CAM2.neg_x_axis_dir') - get_val('CAM2.CAM2_to_CAM1_dir'))) % 360
    ea_cam2_z = alpha2_1 + alpha2_2

    # Option (2) configuration from the MATLAB code
    x_cam2 = get_val('CAM1_CAM2_baseline') * np.cos(azimuth_cam2_in_cam1)
    y_cam2 = get_val('CAM1_CAM2_baseline') * np.sin(azimuth_cam2_in_cam1)
    z_cam2 = 0
    abs_cam_pos2 = [x_cam2, y_cam2, z_cam2]
    
    ea_cam2_x = get_val('CAM2.sensor_X_dir') - 90
    ea_cam2_y = get_val('CAM2.sensor_Y_dir')
    euler_angles2 = [ea_cam2_x, ea_cam2_y, ea_cam2_z]
    
    transform2 = create_rigid_transform(euler_angles2, abs_cam_pos2)

    # ==========================================
    # CAM3 Pose
    # ==========================================
    # CAM3 azimuth angle sensor
    azimuth_cam3_in_cam1 = np.deg2rad(abs(get_val('CAM1.neg_x_axis_dir') + 180 - get_val('CAM1.CAM1_to_CAM3_dir')))
    # CAM3 orientation
    alpha3_1 = (abs(get_val('CAM1.neg_x_axis_dir') + 180 - get_val('CAM1.CAM1_to_CAM3_dir'))) % 360
    alpha3_2 = (abs(get_val('CAM3.neg_x_axis_dir') - get_val('CAM3.CAM3_to_CAM1_dir'))) % 360
    ea_cam3_z = alpha3_1 + alpha3_2

    # Option (2) configuration from the MATLAB code
    x_cam3 = get_val('CAM1_CAM3_baseline') * np.cos(azimuth_cam3_in_cam1)
    y_cam3 = get_val('CAM1_CAM3_baseline') * np.sin(azimuth_cam3_in_cam1)
    abs_cam_pos3 = [x_cam3, y_cam3, 0]
    
    ea_cam3_x = get_val('CAM3.sensor_X_dir') - 90
    ea_cam3_y = get_val('CAM3.sensor_Y_dir')
    euler_angles3 = [ea_cam3_x, ea_cam3_y, ea_cam3_z]
    
    transform3 = create_rigid_transform(euler_angles3, abs_cam_pos3)

    return transform1, transform2, transform3

def convert_matlab_transforms_to_poses(transforms):
    """
    Converts 4x4 MATLAB rigid transformation matrices into 7-element poses 
    compatible with the CameraState class.
    
    Parameters:
    ----------
    transforms : list of np.ndarray
        A list containing the 4x4 homogeneous matrices extracted from MATLAB.
        
    Returns:
    -------
    poses : list of np.ndarray
        A list of [x, y, z, qx, qy, qz, qw] arrays ready for CameraState.
    """
    base2cam = np.array([
        [0, -1,  0],
        [0,  0, -1],
        [1,  0,  0]
    ])
    
    poses = []
    
    for T_mat in transforms:
        # 1. Extract R_pose (3x3) and t_pose (3,) from the 4x4 rigid matrix
        R_pose = T_mat[:3, :3]
        t_pose = T_mat[:3, 3]
        
        # 2. Derive R_ext according to your specification: R_ext = R_pose.T
        R_ext = R_pose.T
        
        # 3. Calculate target R_world relative to your custom base2cam frame
        # Math: R_world = R_ext.T @ base2cam
        R_world = R_ext.T @ base2cam
        
        # 4. Convert the rotation matrix to a quaternion [qx, qy, qz, qw]
        quat = R.from_matrix(R_world).as_quat()
        
        # 5. Position matches the original translation vector (t_world = t_pose)
        t_world = t_pose
        
        # 6. Combine into the final 7-element format: [x, y, z, qx, qy, qz, qw]
        pose_7d = np.concatenate([t_world, quat])
        poses.append(pose_7d)
        
    return poses

def run_flock_scenario():
    # 1. Parameter extraction and Logging Setup
    path = r"E:\科研相关\博士相关\博士课题\项目\观鸟\鸟群数据传承\2023-2024鸟群-长沙-数据\ChangshaObservation2023\synchronized\Xianjiahu_20231121b_data50"
    image1_name = "XJH_1_50_N68_HW1_CollectiveTurn1_VID_20231121_163029_(2399_2818)"
    image2_name = "XJH_2_50_N68_HW2_CollectiveTurn1_VID_20231121_163104_(927_1346)"
    csv1_name = "N68_HW1_CollectiveTurn1_VID_20231121_163029_(2399_2818)_labels_2024-01-09-21-55.csv"
    csv2_name = "N68_HW2_CollectiveTurn1_VID_20231121_163104_(927_1346)_labels_2024-01-09-21-55.csv"
    name = RUN_PARAMS['name']
    extrinsics_json_path = r"E:\科研相关\博士相关\博士课题\项目\观鸟\鸟群数据传承\MATLAB_3D_reconstruction\params\20231121b_pair_Xianjiahu.json"
    intrinsics1_path = r"E:\科研相关\博士相关\博士课题\项目\观鸟\鸟群数据传承\MATLAB_3D_reconstruction\intrinsics\HW1\cameraParams.mat"
    intrinsics2_path = r"E:\科研相关\博士相关\博士课题\项目\观鸟\鸟群数据传承\MATLAB_3D_reconstruction\intrinsics\HW2\cameraParams.mat"
    intrinsics3_path = r"E:\科研相关\博士相关\博士课题\项目\观鸟\鸟群数据传承\MATLAB_3D_reconstruction\intrinsics\HW3\cameraParams.mat"

    img_array = np.fromfile(os.path.join(path, image1_name, '230.jpg'), dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR) 
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    csv1 = pd.read_csv(os.path.join(path, csv1_name))
    unique_images_csv1 = csv1['Img Name'].unique()
    csv2 = pd.read_csv(os.path.join(path, csv2_name))
    unique_images_csv2 = csv2['Img Name'].unique()

    def get_detections(time_step, unique_images, csv):
        target_img = unique_images[time_step]
        # Filter the dataframe for that specific image
        detections = csv[csv['Img Name'] == target_img][['cx', 'cy']].to_numpy()
        return detections
    
    def undistort_detections(points, camera_matrix, dist_coeffs):
        """
        Undistorts an N*2 array of [cx, cy] coordinates.
        
        Parameters:
        - points: np.array of shape (N, 2)
        - camera_matrix: 3x3 intrinsic matrix
        - dist_coeffs: np.array of radial/tangential coefficients
        
        Returns:
        - undistorted_points: np.array of shape (N, 2) in image coordinates
        """
        # 1. Reshape points to (N, 1, 2) as required by OpenCV
        points_reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        
        # 2. Undistort points
        # P=camera_matrix ensures the output stays in pixel coordinates 
        # instead of normalized coordinates (-1 to 1)
        undistorted = cv2.undistortPoints(
            src=points_reshaped, 
            cameraMatrix=camera_matrix, 
            distCoeffs=dist_coeffs, 
            P=camera_matrix
        )
        
        # 3. Reshape back to (N, 2)
        return undistorted.reshape(-1, 2)

    log_name = RUN_PARAMS['log_name']
    start_step = RUN_PARAMS['start_step']
    end_step = RUN_PARAMS['end_step']
    step_length = RUN_PARAMS['step_length']

    scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
    config_path = os.path.join(scenario_path, "config.yaml")

    if CLEAN_LOGS:
        if os.path.exists(os.path.join(scenario_path, "logs")):
            shutil.rmtree(os.path.join(scenario_path, "logs"))
        
        files_to_delete = glob.glob(os.path.join(scenario_path, 'metrics_*.npz'))
        for file_path in files_to_delete:
            try:
                os.remove(file_path)  # Deletes the file
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")
        return

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

    # Load the .mat file
    mat_data = scipy.io.loadmat(os.path.join(path, name + ".mat"))
    trajectories = mat_data['xyzTensorValid']

    cache_file_path = os.path.join(scenario_path, 'reconstruction_scale.npz')
    # if False:
    if os.path.exists(cache_file_path):
        print(f"Loading cached ground truth scales from: {cache_file_path}")
        # Load the compressed npz file
        loaded_gt = np.load(cache_file_path)
        
        # Extract arrays back into your target variables
        num_gmm_gt = loaded_gt['num_gmm_gt'].tolist() # Convert back to a list to match your original type
        scales_gt = loaded_gt['scales_gt'].tolist()
    else:
        print("Cached scale file not found. Starting computation...")
        
        num_gmm_gt = [10] * trajectories.shape[0]
        scales_gt = []
        
        for idx, time_step in enumerate(tqdm(range(trajectories.shape[0]))):
            pos_gpu = torch.from_numpy(trajectories[time_step]).cuda().float()
            nn_dist = torch.cdist(pos_gpu, pos_gpu) + torch.eye(pos_gpu.shape[0], device='cuda') * 1e10
            avg_nn_dist = torch.median(torch.min(nn_dist, dim=1).values).item()
            
            f = lambda s: mode_counting(pos_gpu, pos_gpu.clone(), s, max_iter=2000, tol=avg_nn_dist * 5e-4)

            scale_gt = find_target_scale(f, 10, 0, 15)
            scales_gt.append(scale_gt)
            
            if f(scale_gt) != 10:
                raise ValueError("find scale fails.")

        # Package the newly computed results
        gt = {
            'num_gmm_gt': np.array(num_gmm_gt),
            'scales_gt': np.array(scales_gt)
        }
        
        # Ensure the target directory exists before saving
        os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
        np.savez(cache_file_path, **gt)
        print(f"Computation complete. Results saved to: {cache_file_path}")


    max_steps = trajectories.shape[0]
    effective_end_step = end_step if end_step is not None and end_step <= max_steps else max_steps
    
    if start_step >= effective_end_step:
        logger.info(f"Skipping {name}: start_step ({start_step}) >= end_step ({effective_end_step}).")
        return
    
    step_range = range(start_step, effective_end_step, step_length)

    # Camera Configurations
    transform1, transform2, transform3 = load_camera_extrinsics(extrinsics_json_path)
    cam_poses = convert_matlab_transforms_to_poses([transform1, transform2])

    cameraParameters1 = np.array([
        [3328.2389, 0, 1858.5952],
        [0, 3362.5043, 1037.2734],
        [0, 0, 1]
    ])
    radial1 = np.array([0.1266, 0.0674, 0, 0])

    cameraParameters2 = np.array([
        [3392.0252, 0, 2364.9331],
        [0, 3402.5383, 1021.1155],
        [0, 0, 1]
    ])
    radial2 = np.array([-0.3591, 0.8730, 0, 0])

    W = 3840
    H = 2160

    # 4. System Initialization    
    cam_system = MultiCameraSystem.create_homogeneous_system(
        state_class=CameraState,
        intrinsics=cameraParameters1,
        H=H, W=W, 
        poses_or_RTs=cam_poses,
        near_clip=1, far_clip=100, 
        size=1,
        device='cuda')
    cam_system.cameras[1].state.intrinsics_params = cameraParameters2
    cam_system.cameras[1].state.K = torch.tensor(cameraParameters2, dtype=torch.float, device='cuda').contiguous()

    reconstruction_params = {
        'targetd_num_mode': 10,
        # voxel method
        'voxel_scale': 0.5,
        'voxel_peak_threshold': 0.3,
        'voxel_grid_max_size': 32,
        'voxel_peaks_number': 2 * 10
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
        'lr_max_steps': 100
    }
    density_reconstructor = DensityReconstructor(
        H=H, W=W,
        max_iter=train_params['lr_max_steps'], use_decoupled=USE_DECOUPLED)

    # 5. Simulation Loop
    total_num = []
    for idx, time_step in enumerate(tqdm(step_range, desc=f"Processing {name}")):
        positions = trajectories[time_step]
        total_num.append(positions.shape[0])

        detections1 = get_detections(time_step, unique_images_csv1, csv1)
        detections2 = get_detections(time_step, unique_images_csv2, csv2)
        detections1_undistorted = undistort_detections(detections1, cameraParameters1, radial1)
        detections2_undistorted = undistort_detections(detections2, cameraParameters2, radial2)

        poses, projections, _, masks = cam_system.simulate_vision(positions, renderer='projection_only', is_auto_aim=False)
        projections = [detections1_undistorted, detections2_undistorted]

        # gmm_visualizer = MultiGMMPlotter()
        # # gmm_visualizer.add_gmm(model[0]._xyz.detach().cpu().numpy(), model[0]._radius.detach().cpu().numpy(), model[0]._weights.detach().cpu().numpy())
        # gmm_visualizer.update(real_means=positions, cameras=cam_system.cameras)
        # move_figure(gmm_visualizer.fig, 100, 100)
        # gmm_visualizer.ax.view_init(elev=33, azim=-117, roll=0)
        # # gmm_visualizer.fig.savefig("gmm_diagram.png", transparent=True, bbox_inches='tight')
        # plt.show()
        
        
        model, scale_spaces = \
        density_reconstructor.process_frame(cam_system, point_sets=projections, positions=positions,
                                            initGMM=None,
                                            is_adaptive_scale=False, scale=scales_gt[idx],
                                            is_store_intermediate=IS_LOGGING, is_log=IS_LOGGING,
                                            output_dir=os.path.join(log_file_path, f"t_{time_step:03d}"),
                                            debug=False,
                                            train_params=train_params,
                                            reconstruction_params=reconstruction_params)

        # gmm_visualizer = MultiGMMPlotter()
        # gmm_visualizer.add_gmm(model[0]._xyz.detach().cpu().numpy(), model[0]._radius.detach().cpu().numpy(), model[0]._weights.detach().cpu().numpy())
        # gmm_visualizer.update(real_means=positions, cameras=cam_system.cameras)
        # move_figure(gmm_visualizer.fig, 100, 100)
        # gmm_visualizer.ax.view_init(elev=33, azim=-117, roll=0)
        # # gmm_visualizer.fig.savefig("gmm_diagram.png", transparent=True, bbox_inches='tight')
        # plt.show()

        # 6. Collect Metrics
        for metric_name, value in density_reconstructor.time_metrics.items():
            time_metrics[metric_name].append(value)
        
        loss_metrics['final_training_loss'].append(model[0].mean_loss)
        loss_metrics['final_gmm_num'].append(model[0]._xyz.shape[0])
        loss_metrics['scale'].append(density_reconstructor.scale)

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
    logger.info(f"Results for {name}:")
    if time_metrics['train_gaussian_scale_space']:
        mean_time = np.mean(np.array(time_metrics['train_gaussian_scale_space']))
        logger.info(f"Mean 'train_gaussian_scale_space' time: {mean_time:.2f} ms")
    else:
        logger.info("No time steps processed.")
    
    save_data = {**{k: np.array(v) for k, v in time_metrics.items()}, 
             **{k: np.array(v) for k, v in loss_metrics.items()}}

    save_path = os.path.join(log_file_path, "statistics.npz")
    if IS_LOGGING:
        np.savez(save_path, **save_data)
        logger.info(f"Statistics saved to: {save_path}")
    logger.info(f"Finished scenario {name}")

from matplotlib.widgets import Slider
def visualize_trained_model_interactive():
    # 1. Parameter extraction and Logging Setup
    path = r"E:\科研相关\博士相关\博士课题\项目\观鸟\鸟群数据传承\2023-2024鸟群-长沙-数据\ChangshaObservation2023\synchronized\Xianjiahu_20231121b_data50"
    name = RUN_PARAMS['name']

    log_name = RUN_PARAMS['log_name']
    start_step = RUN_PARAMS['start_step']
    end_step = RUN_PARAMS['end_step']
    
    # Load the .mat file
    mat_data = scipy.io.loadmat(os.path.join(path, name + ".mat"))
    trajectories = mat_data['xyzTensorValid']

    # Safely extract start and end steps, providing integer fallbacks if they are None
    start_step = RUN_PARAMS.get('start_step')
    if start_step is None:
        start_step = 0  # Default to 0 if missing or None
        
    end_step = RUN_PARAMS.get('end_step')
    if end_step is None:
        end_step = len(trajectories) - 1  # Default to the last frame of data

    scenario_path = os.path.join(os.getcwd(), *["scenarios", name])

    if CLEAN_LOGS:
        if os.path.exists(os.path.join(scenario_path, "logs")):
            shutil.rmtree(os.path.join(scenario_path, "logs"))
        
        files_to_delete = glob.glob(os.path.join(scenario_path, 'metrics_*.npz'))
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")
        return

    log_file_path = os.path.join(scenario_path, *["logs", log_name])
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)

    # 2. Figure Setup (Adjusted for UI)
    fig = plt.figure(figsize=(8, 9))
    # Leave space at the bottom for the sliders
    plt.subplots_adjust(bottom=0.25) 
    ax = fig.add_subplot(111, projection='3d')
    gmm_visualizer = MultiGMMPlotter(fig=fig, ax=ax)

    # Variable to track the GMM artist ID
    gmm1_id = None

    # 3. Define Slider Axes [left, bottom, width, height]
    ax_time = plt.axes([0.2, 0.1, 0.65, 0.03])
    ax_iter = plt.axes([0.2, 0.05, 0.65, 0.03])

    # 4. Create Sliders
    time_slider = Slider(
        ax=ax_time, label='Time Step', 
        valmin=start_step, valmax=end_step, 
        valinit=start_step, valstep=1
    )
    
    iter_slider = Slider(
        ax=ax_iter, label='Iteration', 
        valmin=0, valmax=99, 
        valinit=0, valstep=1
    )

    # 5. The Update Function
    def update(val):
        nonlocal gmm1_id # Allows us to modify the outer scope variable
        
        t = int(time_slider.val)
        it = int(iter_slider.val)

        checkpoint_path = os.path.join(log_file_path, f"t_{t:03d}", "checkpoint_level_0.pth")
        
        try:
            # Load new model based on slider values
            training_history = GaussianModel.load_training_history(checkpoint_path)
            GM_1 = GaussianModel.load_iter(training_history, it)

            means1 = GM_1._xyz.detach().cpu().numpy()
            radii1 = GM_1._radius.detach().cpu().numpy()
            weights1 = GM_1._weights.detach().cpu().numpy()

            # Update or create the plot
            if gmm1_id is None:
                gmm1_id = gmm_visualizer.add_gmm(means1, radii1, weights1, color='blue', label='GMM', visible=True)
            else:
                gmm_visualizer.update_gmm_data(gmm1_id, means=means1, covariances=radii1, weights=weights1, visible=True)

            gmm_visualizer.update(real_means=trajectories[t])
            ax.set_title(f'Timestep {t} Iteration {it}')
            
            # Redraw the canvas
            fig.canvas.draw_idle()
            
        except Exception as e:
            print(f"Warning: Could not load data for Timestep {t}, Iteration {it}. Error: {e}")

    # 6. Bind the update function to the sliders
    time_slider.on_changed(update)
    iter_slider.on_changed(update)

    # Run the update function once to initialize the plot at start values
    update(None)

    plt.show()

    # CRITICAL: Return the sliders so they aren't garbage collected by Python
    return time_slider, iter_slider

def run_single_scenario_baseline():
    # 1. Parameter extraction and Logging Setup
    name = RUN_PARAMS['name']
    log_name = RUN_PARAMS['log_name']
    start_step = RUN_PARAMS['start_step']
    end_step = RUN_PARAMS['end_step']
    step_length = RUN_PARAMS['step_length']

    logger.info(f"Running scenario {name}")

    scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
    config_path = os.path.join(scenario_path, "config.yaml")

    log_file_path = os.path.join(scenario_path, *["logs", log_name])
    if not os.path.exists(log_file_path):
        ValueError(f'Log for {log_name} does not exist')
    
    log_file_path = os.path.join(scenario_path, *["logs", log_name])
    log_data = np.load(os.path.join(log_file_path, "statistics.npz"))
    scale_history = log_data['scale']
    gmm_num_history = log_data['final_gmm_num']

    # 2. Initialize Metrics (must be re-initialized for each run)
    loss_metrics = {
        'final_training_loss': [],
        'final_density_field_loss': [],
    }

    # 3. Load Dataset
    # Load the .mat file
    path = r"E:\科研相关\博士相关\博士课题\项目\观鸟\鸟群数据传承\2023-2024鸟群-长沙-数据\ChangshaObservation2023\synchronized\Xianjiahu_20231121b_data50"
    mat_data = scipy.io.loadmat(os.path.join(path, name + ".mat"))
    trajectories = mat_data['xyzTensorValid']

    max_steps = trajectories.shape[0]
    effective_end_step = end_step if end_step is not None and end_step <= max_steps else max_steps
    
    if start_step >= effective_end_step:
        logger.info(f"Skipping {name}: start_step ({start_step}) >= end_step ({effective_end_step}).")
        return

    # 5. Simulation Loop
    step_range = range(start_step, effective_end_step, step_length)

    for idx, time_step in enumerate(tqdm(step_range, desc=f"Processing {name}")):
        save_path = os.path.join(log_file_path, f"t_{time_step:03d}", f"baseline_level_{0}.pth")

        positions = trajectories[time_step]
        is_visible = np.ones((positions.shape[0],), dtype=np.bool)
        
        scale = scale_history[idx]
        gmm_num = gmm_num_history[idx]

        N = positions[is_visible].shape[0]

        r_means, r_weights, r_covs = GMR.runnalls_algorithm_simple_torch(
            means=torch.from_numpy(positions[is_visible]),
            radii=torch.full((N, 1), scale, device='cuda', dtype=torch.float),
            weights=torch.full((N, 1), 1.0, device='cuda', dtype=torch.float),
            L=gmm_num, DEVICE='cuda'
        )
        r_weights = r_weights.reshape((-1, 1))
        r_radius = torch.sqrt(r_covs[:, 0, 0]).reshape((-1, 1))

        final_means, final_unnorm_weights, final_covs = GMR.optimize_ise_isotropic(
            orig_means=torch.from_numpy(positions[is_visible]),
            orig_covs=(torch.eye(3, device='cuda') * (scale ** 2)).unsqueeze(0).expand(N, 3, 3),
            orig_weights=torch.full((N, 1), 1.0, device='cuda', dtype=torch.float),
            reduced_means=r_means,
            reduced_covs=r_covs,
            reduced_weights=r_weights,
            num_iterations=200,
            lr_mu_pct = 0.05,  # Percentage of data scale (e.g., 5%)
            lr_var = 0.05,     # Fixed LR for log-variances
            lr_weight = 0.05,  # Fixed LR for softmax logits
            DEVICE='cuda'
        )
        final_unnorm_weights = final_unnorm_weights.reshape((-1, 1))
        final_radius = torch.sqrt(final_covs[:, 0, 0]).reshape((-1, 1))

        checkpoint = {
            '_xyz': final_means.detach().clone(),
            '_radius': final_radius.detach().clone(),
            '_weights': final_unnorm_weights.detach().clone(),
            }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)
        
        # 6. Collect Metrics
        images_baseline = [
            rasterize_gaussians(
            final_means,
            final_radius,
            final_unnorm_weights,
            cam.state.R,
            cam.state.T,
            cam.state.K,
            cam.state.H,
            cam.state.W,
            False)
            for cam in cam_system.cameras
        ]

        # blur the images
        scale_spaces = []

        for i, (cam, img) in enumerate(zip(cam_system.cameras, images)):
            dist_cam = np.linalg.norm(cam.state.camera_center - np.mean(positions[is_visible], axis=0))

            pixel_scales = torch.tensor(
                [scale] / dist_cam * cam.state.intrinsics_params[0, 0].item(),
                device='cuda', 
                dtype=torch.float32
            )

            scale_space = DensityReconstructor.generate_scale_space_img_rfft(img.to(dtype=torch.float32), pixel_scales)
            scale_spaces.append(scale_space)

        losses = [torch.sum(torch.abs(scale_spaces[i] - images_baseline[i])).item() for i in range(len(scale_spaces))]

        loss_metrics['final_training_loss'].append(losses[-1] + losses[0])

        loss_metrics['final_density_field_loss'].append(
            calculate_gmm_dissimilarity(
                positions[is_visible],
                scale, 
                final_means, 
                final_unnorm_weights, 
                final_radius, use_decoupled=USE_DECOUPLED))
    
    # 7. Logging and Data Saving
    logger.info(f"Results for {name}:")
    
    save_data = {**{k: np.array(v) for k, v in loss_metrics.items()}}

    save_path = os.path.join(log_file_path, "statistics_baseline.npz")
    if IS_LOGGING:
        np.savez(save_path, **save_data)
        logger.info(f"Statistics saved to: {save_path}")
    logger.info(f"Finished scenario {name}")

def plot_time_multi_scenarios():
    for run_params in DATASET_VIS:
        plot_time_single_scenarios(run_params)

def plot_time_single_scenarios(run_params):
    name = run_params['name']
    log_name = run_params['log_name']
    scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
    log_file_path = os.path.join(scenario_path, *["logs", log_name])

    log_data = np.load(os.path.join(log_file_path, "statistics.npz"))

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    time_stamp = np.arange(log_data['estimate_swarm_center'].shape[0])
    ax.plot(time_stamp, log_data['estimate_swarm_center'], label='estimate_swarm_center')
    ax.plot(time_stamp, log_data['adaptive_scale_selection'], label='adaptive_scale_selection')
    ax.plot(time_stamp, log_data['generate_scale_space'], label='generate_scale_space')
    ax.plot(time_stamp, log_data['estimate_scale_space_peaks'], label='estimate_scale_space_peaks')
    ax.plot(time_stamp, log_data['setup_gaussian_scale_space'], label='setup_gaussian_scale_space')
    ax.plot(time_stamp, log_data['train_gaussian_scale_space'], label='train_gaussian_scale_space')

    plt.title(f'time for scenario {name}')
    plt.xlabel('time step')
    plt.yscale('log')
    plt.legend()

def print_global_metrics(label, metric_data):
    # 1. Sum up the raw counts/masses across all frames
    sum_tp = np.sum(metric_data['tp'])
    sum_fp = np.sum(metric_data['fp'])
    sum_fn = np.sum(metric_data['fn'])
    total_N = np.sum(metric_data['N'])
    total_weights = np.sum(metric_data['w'])

    # 3. Compute final global metrics
    global_recall = sum_tp / total_N if total_N > 0 else 0.0
    global_miss = sum_fn / total_N if total_N > 0 else 0.0
    global_hallucination = sum_fp / total_weights if total_weights > 0 else 0.0
    global_dmota = 1.0 - ((sum_fn + sum_fp) / total_N) if total_N > 0 else 0.0
    global_weight_err = np.mean(np.abs(metric_data['w'] - metric_data['N']) / metric_data['N'])

    # 4. Print results (comparing global vs. simple mean for reference)
    error_str = f"{global_recall:.3f} & {global_hallucination:.3f} & {global_dmota:.3f} &"
    # error_str = f"{global_recall:.3f} & {global_hallucination:.3f} & {global_dmota:.3f} & {metric_data['train_time']:.0f} & {global_weight_err*100:.1f} &"
    # error_str = f"{global_dmota:.3f} & {metric_data['train_time']:.0f} &"
    # error_str = f"{global_recall:.3f} & {global_dmota:.3f} &"
    return error_str

def compute_metrics_multi_scenarios():
    gt_error = []
    estim_error = []
    
    for run_params in [RUN_PARAMS]:
        scenario_name = run_params.get('name', 'Unknown Scenario')
        try:
            results = compute_metrics_single_scenario(run_params)
            
            if results is not None:
                gt_res, estim_res = results
                
                # Only append gt_res if a baseline actually existed for this scenario
                if gt_res is not None:
                    gt_error.append(gt_res)
                    
                estim_error.append(estim_res)
            else:
                logging.warning(f"Skipping results for {scenario_name} due to prior errors.")
                
        except Exception as e:
            logging.error(f"Critical failure in multi-scenario loop for {scenario_name}: {e}")

    if gt_error:
        print('--------- GT ----------')
        print(*gt_error)
        
    print('--------- ESTIM ----------')
    print(*estim_error)

def compute_metrics_single_scenario(run_params):
    try:
        force_update = True

        name = run_params['name']
        log_name = run_params['log_name']
        start_step = run_params['start_step']
        end_step = run_params['end_step']
        step_length = run_params['step_length']

        scenario_path = os.path.join(os.getcwd(), "scenarios", name)
        log_file_path = os.path.join(scenario_path, "logs", log_name)
        config_path = os.path.join(scenario_path, "config.yaml")
        
        if not os.path.exists(scenario_path):
            raise FileNotFoundError(f"Scenario path not found: {scenario_path}")

        # Load the .mat file
        path = r"E:\科研相关\博士相关\博士课题\项目\观鸟\鸟群数据传承\2023-2024鸟群-长沙-数据\ChangshaObservation2023\synchronized\Xianjiahu_20231121b_data50"
        mat_data = scipy.io.loadmat(os.path.join(path, name + ".mat"))
        trajectories = mat_data['xyzTensorValid']

        max_steps = trajectories.shape[0]
        effective_end_step = end_step if end_step is not None and end_step <= max_steps else max_steps
        step_range = range(start_step, effective_end_step, step_length)

        if not step_range:
            logging.warning(f"Step range is empty for {name}. Skipping.")
            return None

        # CHECK 1: Check baseline existence ONLY ONCE using the first time step
        first_time_step_path = os.path.join(log_file_path, f"t_{step_range[0]:03d}")
        has_baseline = os.path.exists(os.path.join(first_time_step_path, "baseline_level_0.pth"))
        
        if not has_baseline:
            logging.info(f"No GT baseline found for {name}. Computing estim metrics only.")

        gt_data_path = os.path.join(scenario_path, 'reconstruction_scale.npz')
        if not os.path.exists(gt_data_path):
            raise FileNotFoundError(f"GT data not found: {gt_data_path}")
            
        gt_data = np.load(gt_data_path)
        scale_history = gt_data['scales_gt']

        try:
            stats_path = os.path.join(log_file_path, "statistics.npz")
            estim_train_time = np.mean(np.load(stats_path)['train_gaussian_scale_space']).item()
        except (FileNotFoundError, KeyError):
            estim_train_time = 0

        # Initialize metrics
        metrics_estim = {
            'tp': [], 'fp': [], 'fn': [], 'N': [], 'w': [],
            'coverage_recall': [], 'miss': [], 'hallucination': [], 'dMOTA': [],
            'train_time': estim_train_time
        }

        metrics = {
            'tp': [], 'fp': [], 'fn': [], 'N': [], 'w': [],
            'coverage_recall': [], 'miss': [], 'hallucination': [], 'dMOTA': [],
            'train_time': 0
        } if has_baseline else None

        metrics_estim_file = os.path.join(scenario_path, f"metrics_estim_{log_name}.npz")
        metrics_gt_file = os.path.join(scenario_path, f"metrics_{log_name}.npz")

        if force_update or not os.path.exists(metrics_estim_file):
            for idx, time_step in enumerate(tqdm(step_range, desc=f"Processing {name}")):
                time_step_path = os.path.join(log_file_path, f"t_{time_step:03d}")

                try:
                    training_history = GaussianModel.load_training_history(os.path.join(time_step_path, "checkpoint_level_0.pth"))
                    model_estim = GaussianModel.load_iter(training_history, iter=99)
                    
                    if has_baseline:
                        model_gt = torch.load(os.path.join(time_step_path, "baseline_level_0.pth"))
                        
                except Exception as e:
                    # logging.error(f"Failed to load models for {name} at step {time_step}: {e}")
                    continue

                positions = trajectories[time_step]
                N = positions.shape[0]
                
                if N == 0:
                    continue

                min_coords = np.min(positions, axis=0)
                max_coords = np.max(positions, axis=0)
                bounds = np.vstack((min_coords - 3 * scale_history[idx], max_coords + 3 * scale_history[idx])).T
                voxel_res = np.max(max_coords - min_coords) * 5e-3

                # --- ESTIM METRICS ---
                total_tp_mass, total_fp_mass, total_fn_mass = compute_metrics_batched_torch(
                    means1_np=positions, sigma1=scale_history[idx], 
                    pred_means=model_estim._xyz, pred_weights=model_estim._weights, pred_sigmas=model_estim._radius,
                    bounds=bounds, voxel_res=voxel_res, batch_size=50000, device='cuda'
                )
                
                estim_w_sum = model_estim._weights.sum().item()

                metrics_estim['tp'].append(total_tp_mass)
                metrics_estim['fp'].append(total_fp_mass)
                metrics_estim['fn'].append(total_fn_mass)
                metrics_estim['N'].append(N)
                metrics_estim['w'].append(estim_w_sum)
                metrics_estim['coverage_recall'].append(total_tp_mass / N)
                metrics_estim['miss'].append(total_fn_mass / N)
                metrics_estim['hallucination'].append(total_fp_mass / estim_w_sum if estim_w_sum > 0 else 0.0)
                metrics_estim['dMOTA'].append(1 - (total_fn_mass + total_fp_mass) / N)

                # --- GT METRICS (Only if baseline exists) ---
                if has_baseline:
                    total_tp_mass_gt, total_fp_mass_gt, total_fn_mass_gt = compute_metrics_batched_torch(
                        means1_np=positions, sigma1=scale_history[idx], 
                        pred_means=model_gt['_xyz'], pred_weights=model_gt['_weights'], pred_sigmas=model_gt['_radius'],
                        bounds=bounds, voxel_res=voxel_res, batch_size=50000, device='cuda'
                    )
                    
                    gt_w_sum = model_gt['_weights'].sum().item()

                    metrics['tp'].append(total_tp_mass_gt)
                    metrics['fp'].append(total_fp_mass_gt)
                    metrics['fn'].append(total_fn_mass_gt)
                    metrics['N'].append(N)
                    metrics['w'].append(gt_w_sum)
                    metrics['coverage_recall'].append(total_tp_mass_gt / N)
                    metrics['miss'].append(total_fn_mass_gt / N)
                    metrics['hallucination'].append(total_fp_mass_gt / gt_w_sum if gt_w_sum > 0 else 0.0)
                    metrics['dMOTA'].append(1 - (total_fn_mass_gt + total_fp_mass_gt) / N)

            # if not metrics_estim['N']:
            #     raise ValueError(f"No valid metric data was generated for {name}.")

            # Save arrays
            metrics_estim_arrays = {k: np.array(v) for k, v in metrics_estim.items()}
            np.savez(metrics_estim_file, **metrics_estim_arrays)
            metrics_estim = metrics_estim_arrays
            
            if has_baseline:
                metrics_arrays = {k: np.array(v) for k, v in metrics.items()}
                np.savez(metrics_gt_file, **metrics_arrays)
                metrics = metrics_arrays
            
        else:
            # Load existing metrics
            try:
                metrics_estim_loaded = np.load(metrics_estim_file)
                metrics_estim = {k: np.array(v) for k, v in metrics_estim_loaded.items()}
                metrics_estim['train_time'] = estim_train_time
                
                # Check if cached GT metrics exist before loading
                if os.path.exists(metrics_gt_file):
                    metrics_loaded = np.load(metrics_gt_file)
                    metrics = {k: np.array(v) for k, v in metrics_loaded.items()}
                    metrics['train_time'] = 0
                    has_baseline = True
                else:
                    metrics = None
                    has_baseline = False
                    
            except Exception as e:
                raise IOError(f"Failed to load cached metrics for {name}: {e}")

        estim_result = print_global_metrics('estim', metrics_estim)
        gt_result = print_global_metrics('gt', metrics) if has_baseline else None
        
        return gt_result, estim_result

    except Exception as e:
        logging.error(f"Error processing scenario {run_params.get('name', 'Unknown')}: {str(e)}")
        return None

if __name__ == "__main__":
    # run_flock_scenario()

    visualize_trained_model_interactive()

    # run_single_scenario_baseline()

    # compute_metrics_multi_scenarios()
    # plot_time_multi_scenarios()

    plt.show()