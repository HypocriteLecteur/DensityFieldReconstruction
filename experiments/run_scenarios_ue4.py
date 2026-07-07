import argparse
from pathlib import Path
import pandas as pd

from tqdm import tqdm


import numpy as np
from dfr.simulation_config import SimulationConfig
from dfr.dataset_io import DatasetFactory
from dfr.camera_state import CameraStateUE4
from dfr.camera_system import MultiCameraSystem

import cv2
from scipy.spatial.transform import Rotation as R

from experiments.common import setup_logger
from dfr.artifacts import OutputConfig
from dfr.config import ReconstructionParams, TrainingParams
from dfr.reconstruction.observations import (
    ExternalObservationFrame,
    reconstruct_observations,
)

# Setup logger
logger = setup_logger(__name__)

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

def _projection_array(points):
    """Return a float32 ``(N, 2)`` centroid array, preserving empty detections."""
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def _pose_from_ue4_rt(rotation, translation):
    """Convert a UE4-style world pose into the saved 7D pose convention."""
    quaternion = R.from_matrix(rotation).as_quat()
    return np.concatenate([np.asarray(translation, dtype=np.float32), quaternion])


def run_ue4(*, project_root, image_roots, output=None, seed=12345, save_output=True):
    LOG_NAME = 'base_better_peak'
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

    root = Path(project_root).expanduser().resolve()
    scenario_path = root / "scenarios" / name
    config_path = scenario_path / "config.yaml"

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

    # 4. External observation assembly. The primary UE4 path now feeds
    # thresholded image detections through the shared package workflow rather
    # than hand-running DensityReconstructor inside this experiment script.
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

    observations = []
    for time_step in tqdm(step_range, desc=f"Processing {name}"):
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
        camera_poses = np.asarray(
            [_pose_from_ue4_rt(rotation, translation) for rotation, translation in RTs],
            dtype=np.float32,
        )

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

        img = cv2.imread(str(image_roots[0] / f"{img_idx}.jpg"))
        img2 = cv2.imread(str(image_roots[1] / f"{img_idx}.jpg"))
        img3 = cv2.imread(str(image_roots[2] / f"{img_idx}.jpg"))
        if any(value is None for value in (img, img2, img3)):
            raise FileNotFoundError(
                f"Missing UE4 camera image for frame {img_idx} under {image_roots}."
            )

        # Pre-processing
        BW, _ = thresholding(img)
        centroids = find_centroids(BW, min_area_threshold=10)
        BW2, _ = thresholding(img2)
        centroids2 = find_centroids(BW2, min_area_threshold=10)
        BW3, _ = thresholding(img3)
        centroids3 = find_centroids(BW3, min_area_threshold=10)
        _, _, _, masks = cam_system.simulate_vision(
            positions,
            renderer="projection_only",
            is_auto_aim=False,
        )
        observations.append(
            ExternalObservationFrame(
                dataset_name=name,
                frame=time_step,
                positions=positions,
                projections=(
                    _projection_array(centroids),
                    _projection_array(centroids2),
                    _projection_array(centroids3),
                ),
                camera_system=cam_system,
                camera_poses=camera_poses,
                visible_mask=np.logical_and.reduce(masks),
                metadata={
                    "source": "ue4_thresholded_images",
                    "image_index": img_idx,
                    "image_roots": [str(path) for path in image_roots],
                },
            )
        )

    if output is None and save_output:
        output = OutputConfig(
            workflow="reconstruction",
            name=f"ue4-{log_name}",
            project_root=root,
        )
    run = reconstruct_observations(
        observations,
        training=TrainingParams(
            xyz_lr_c=0.05,
            xyz_lr_final_c=0.9,
            radius_lr_c=0.05,
            radius_lr_final_c=0.9,
            weights_lr_c=0.10,
            weights_lr_final_c=0.7,
            xyz_reg=1.0,
            radius_reg=0.3,
            radius_cutoff_inv=0.5,
            lr_max_steps=500,
        ),
        reconstruction=ReconstructionParams(
            targetd_num_mode=10,
            voxel_scale=2,
            voxel_peak_threshold=0.3,
            voxel_grid_max_size=32,
            voxel_peaks_number=10,
        ),
        use_decoupled=USE_DECOUPLED,
        seed=seed,
        output=output,
    )
    if run.run_dir is not None:
        logger.info("Managed UE4 reconstruction artifacts saved to: %s", run.run_dir)
    return run

def create_parser():
    parser = argparse.ArgumentParser(
        description="UE4 reconstruction from three image-detection streams."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--image-roots",
        type=Path,
        nargs=3,
        required=True,
        metavar=("CAMERA_1", "CAMERA_2", "CAMERA_3"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Return reconstructed arrays without creating a managed run directory.",
    )
    return parser


def main(argv=None):
    args = create_parser().parse_args(argv)
    roots = tuple(path.expanduser().resolve() for path in args.image_roots)
    for path in roots:
        if not path.is_dir():
            raise FileNotFoundError(f"UE4 image directory does not exist: {path}")
    output = None
    if not args.no_output:
        output = OutputConfig(
            workflow="reconstruction",
            name="ue4-base_better_peak",
            root=args.output_root,
            run_id=args.run_id,
            project_root=args.project_root,
            resume=args.resume,
            overwrite=args.overwrite,
        )
    run_ue4(
        project_root=args.project_root,
        image_roots=roots,
        output=output,
        seed=args.seed,
        save_output=not args.no_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
