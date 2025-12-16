import sys
import os

sys.path.append(os.getcwd()) # To get around relative import issues, I hate Python.

import numpy as np
from density_field_reconstruction.density_reconstructor import DensityReconstructor
from density_field_reconstruction.camera_state import CameraStateUE4
from density_field_reconstruction.utils import calculate_gmm_ise_gpu
from density_field_reconstruction.visualizer import SimulationVisualizer, MultiGMMVisualizer
from density_field_reconstruction.gaussian_mixture_reduction import GMR

import glob
import cv2
from scipy.io import loadmat
import matplotlib.pyplot as plt

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
    data = loadmat('./camera-aero/matlab.mat')

    positions_all = data['drone_positions']

    search_directory = 'camera-aero/camera_picture_aero1'
    jpg_files = glob.glob(os.path.join(search_directory, '*.jpg'))

    number_of_images = len(jpg_files)

    intrinsics_params = np.array([
            [7329, 0, 320],
            [0, 7329, 240],
            [0, 0, 1]
        ])

    W = 640
    H = 480

    visualizer = SimulationVisualizer(intrinsics_params=intrinsics_params,
                                      H=H, W=W, 
                                      cam_num=2,
                                      mode='3d_only',
                                      save_video=False, fps=30, dpi=100)
    mgmm_visualizer = MultiGMMVisualizer(H=H, W=W)

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
        'final_gmm_num': []
    }

    model = None
    for time_step in range(1, number_of_images):
        img_idx = time_step + 1

        img = cv2.imread(f'camera-aero/camera_picture_aero1/{img_idx}.jpg')
        img2 = cv2.imread(f'camera-aero/camera_picture_aero3/{img_idx}.jpg')

        # Pre-processing
        BW, _ = thresholding(img)
        centroids = find_centroids(BW, min_area_threshold=10)
        BW2, _ = thresholding(img2)
        centroids2 = find_centroids(BW2, min_area_threshold=10)
    
        # Visualization
        for (x, y) in centroids:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1) 
        for (x, y) in centroids2:
            cv2.circle(img2, (x, y), 2, (0, 0, 255), -1)

        # cv2.imwrite(f"mask.jpg", BW)
        # cv2.imwrite(f"aero_{1}_detection.jpg", img)
        # cv2.imwrite(f"aero_{3}_detection.jpg", img2)
        # cv2.imshow(f"aero_{1}_detection", img)
        # cv2.imshow(f"aero_{3}_detection", img2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        centroids = np.array(centroids)
        centroids2 = np.array(centroids2)

        R1_input = np.array([
            [-0.7071, 0.7071, 0],
            [-0.7071, -0.7071, 0],
            [0, 0, 1]
        ])
        T1_input = np.array([-1000, 0, -4.88])

        R3_input = np.array([
            [0.7071, 0.7071, 0],
            [-0.7071, 0.7071, 0],
            [0, 0, 1]
        ])
        T3_input = np.array([1000, 0, -4.88])

        camera_states = []    
        camera_states.append(
            CameraStateUE4(1, W, H, intrinsics_params, R1_input, T1_input, device='cuda'))
        camera_states.append(
            CameraStateUE4(3, W, H, intrinsics_params, R3_input, T3_input, device='cuda'))
        
        density_reconstructor = DensityReconstructor(intrinsics_params, max_iter=100, W=W, H=H)

        model, scale_spaces = \
        density_reconstructor.process_frame(camera_states, point_sets=[centroids, centroids2],
                                                is_adaptive_scale=True,
                                                is_store_intermediate=False, is_log=False)
        
        for metric_name, value in density_reconstructor.time_metrics.items():
            time_metrics[metric_name].append(value)
        
        loss_metrics['final_training_loss'].append(model[0].sum_loss)
        loss_metrics['final_gmm_num'].append(model[0]._xyz.shape[0])

        # _, projections, _ = stereo_vision.simulate_vision(positions_all[time_step], renderer='gaussian')
        # swarm_projection, swarm_projection2 = projections
        # is_visible = (swarm_projection[:, 0] > 0).squeeze() & (swarm_projection[:, 1] > 0).squeeze() & \
        #     (swarm_projection[:, 0] < H).squeeze() & (swarm_projection[:, 1] < W).squeeze()
        # is_visible2 = (swarm_projection2[:, 0] > 0).squeeze() & (swarm_projection2[:, 1] > 0).squeeze() & \
        #     (swarm_projection2[:, 0] < H).squeeze() & (swarm_projection2[:, 1] < W).squeeze()
        # is_visible = np.logical_and(is_visible, is_visible2)
        loss_metrics['final_density_field_loss'].append(
            calculate_gmm_ise_gpu(
                positions_all[time_step],
                density_reconstructor.scale, 
                model[0]._xyz, 
                model[0]._weights, 
                model[0]._radius))

        # visualizer.update(time_step=time_step,
        #                   positions=positions_all[time_step],
        #                   R1=camera_states[0].P_np[:, :3], T1=camera_states[0].P_np[:, 3], 
        #                   R2=camera_states[1].P_np[:, :3], T2=camera_states[1].P_np[:, 3], 
        #                   img=scale_spaces[0][0], img2=scale_spaces[1][0])
        
        # CameraState and CameraStateUE4 has differnt frames
        visualizer.update(time_step=time_step,
                    positions=positions_all[time_step],
                    cam_poses=poses,
                    imgs=scale_spaces)
        
        r_means, r_weights, r_covs = GMR.kmeans_numpy(
            positions_all[time_step], 
            density_reconstructor.scale, 
            model[0].num_gaussians
        )

        if time_step == 1:
            gmm1_id = mgmm_visualizer.add_gmm(model[0]._xyz.detach().cpu().numpy(), 
                                              model[0]._radius.detach().cpu().numpy(), 
                                              model[0]._weights.detach().cpu().numpy(), color='blue', label='baseline')
            gmm2_id = mgmm_visualizer.add_gmm(r_means.detach().cpu().numpy(), 
                                              r_covs.detach().cpu().numpy(), 
                                              r_weights.detach().cpu().numpy(), color='orange', label='GMR')
        else:
            mgmm_visualizer.update_gmm_data(gmm1_id, 
                                            means=model[0]._xyz.detach().cpu().numpy(), 
                                            covariances=model[0]._radius.detach().cpu().numpy(), 
                                            weights=model[0]._weights.detach().cpu().numpy(), visible=True)
            mgmm_visualizer.update_gmm_data(gmm2_id, 
                                            means=r_means.detach().cpu().numpy(), 
                                            covariances=r_covs.detach().cpu().numpy(), 
                                            weights=r_weights.detach().cpu().numpy(), visible=True)
        mgmm_visualizer.update(
            real_means=positions_all[time_step],
        )
        plt.pause(0.001)

    if time_metrics['train_gaussian_scale_space']:
        mean_time = np.mean(np.array(time_metrics['train_gaussian_scale_space'][1:]))
        std_time = np.std(np.array(time_metrics['train_gaussian_scale_space'][1:]))
        print(f"Mean 'train_gaussian_scale_space' time: {mean_time:.2f} ms +- {std_time:.2f} ms")
    else:
        print("No time steps procesed.")

    if time_metrics['adaptive_scale_selection']:
        mean_time = np.mean(np.array(time_metrics['adaptive_scale_selection'][1:]))
        std_time = np.std(np.array(time_metrics['adaptive_scale_selection'][1:]))
        print(f"Mean 'adaptive_scale_selection' time: {mean_time:.2f} ms +- {std_time:.2f} ms")
    else:
        print("No time steps processed.")

    # 转台1相机:
    # t=[-1000;0;-4.88]
    # R=[-0.7071 0.7071 0;-0.7071 -0.7071 0;0 0 1]
    # 转台2相机:
    # t=[0;0;-4.88]
    # R=[0 1 0;-1 0 0;0 0 1]
    # 转台3相机:
    # t=[1000;0;-4.88]
    # R=[0.7071 0.7071 0;-0.7071 0.7071 0;0 0 1]

if __name__ == "__main__":
    main()