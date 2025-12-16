import torch
import numpy as np
import cv2
import os
import torch.nn.functional as F
from scipy.optimize import curve_fit
from density_field_reconstruction.density_field_model import GaussianModel
from gaussian_rasterizer_simple_small import rasterize_gaussians
from density_field_reconstruction.density_peak import match_points, find_local_peaks_simple
from density_field_reconstruction.scale_estimation import critical_scale_detection
from density_field_reconstruction.utils import calculate_fundamental_matrix_pytorch, get_outlier_neighbors
from density_field_reconstruction.gaussian_mixture_reduction import GMR
from density_field_reconstruction.camera_state import CameraState
import time

class DensityReconstructor:
    def __init__(self, intrinsics_params, max_iter=100, W=1000, H=1000, far_clip=2000):
        self.device = 'cuda'

        self.intrinsics_params = intrinsics_params
        self.intrinsics_params_cuda = torch.tensor(self.intrinsics_params, dtype=torch.float, device='cuda').contiguous()

        self.camera_states: list[CameraState] = None

        self.max_iter = max_iter

        self.num_scales = 1
        self.scale = None
        self.scaling_law = None

        self.GSP = [GaussianModel(H=H, W=W) for _ in range(self.num_scales)]

        self.time_metrics = {
            'estimate_swarm_center': 0.0,
            'adaptive_scale_selection': 0.0,
            'generate_scale_space': 0.0,
            'estimate_scale_space_peaks': 0.0,
            'setup_gaussian_scale_space': 0.0,
            'train_gaussian_scale_space': 0.0,
        }
    
    def projection(self, camera_states, cam_num, level):
        if cam_num == 1:
            R = camera_states[0].R
            T = camera_states[0].T
        elif cam_num == 2:
            R = camera_states[1].R
            T = camera_states[1].T
        else:
            raise ValueError("cam_num must be 1 or 2")
        
        return rasterize_gaussians(
            self.GSP[level]._xyz,
            self.GSP[level]._radius,
            self.GSP[level]._weights,
            R,
            T,
            self.intrinsics_params_cuda,
            False
        )
    
    def estimate_swarm_center(self, centroids: list[torch.Tensor]):
        cam1 = self.camera_states[0]
        cam2 = self.camera_states[1]
        
        # P's are (3, 4) [R|t] matrices
        P1_proj = cam1.intrinsics_params @ cam1.P_np
        P2_proj = cam2.intrinsics_params @ cam2.P_np
        
        pnts4D = cv2.triangulatePoints(P1_proj, P2_proj, 
                                       np.mean(centroids[0], axis=0), np.mean(centroids[1], axis=0))
        
        # Convert homogeneous coordinates to 3D
        center = (pnts4D[:3, :] / pnts4D[3].T).reshape((3,))
        return center

    def estimate_swarm_center_image(self, images: list[torch.Tensor]):
        """
        Estimates the 3D center of the swarm by triangulating the image-intensity centroids.
        Requires at least two images.
        """
        if len(images) < 2:
            raise ValueError("Need at least two images for triangulation.")
    
        centroids_np = []
        for img in images:
            H, W = img.shape[-2:]

            # Compute centroid (weighted mean of pixel coordinates)
            total_intensity = img.sum()
            if total_intensity.item() == 0:
                 raise ValueError(f"Image from camera {len(centroids_np)} is empty (sum=0).")
            
            x_coords = torch.arange(W, dtype=torch.float32, device=img.device)
            y_coords = torch.arange(H, dtype=torch.float32, device=img.device)
            
            # Sum over H for x-weighted sum, sum over W for y-weighted sum
            x_weighted = (x_coords * img.sum(dim=-2)).sum() / total_intensity
            y_weighted = (y_coords * img.sum(dim=-1)).sum() / total_intensity

            centroids_np.append(np.array([x_weighted.cpu().item() + 0.5, y_weighted.cpu().item() + 0.5])) # +0.5 for pixel center

        # Triangulate using the first two camera states
        cam1 = self.camera_states[0]
        cam2 = self.camera_states[1]
        
        # P's are (3, 4) [R|t] matrices
        P1_proj = cam1.K.cpu().numpy() @ cam1.P_np
        P2_proj = cam2.K.cpu().numpy() @ cam2.P_np
        
        pnts4D = cv2.triangulatePoints(P1_proj, P2_proj, centroids_np[0], centroids_np[1])
        
        # Convert homogeneous coordinates to 3D
        center = (pnts4D[:3, :] / pnts4D[3]).T.reshape((3,))
        return center
    
    def generate_scale_space_img_large_scale(self, image, scales, device='cuda'):
        # Assume image is (1, 1, H, W) float32 on device
        h, w = image.shape[-2:]
        freqs_y = torch.fft.fftfreq(h, device=image.device)
        freqs_x = torch.fft.fftfreq(w, device=image.device)
        uy, ux = torch.meshgrid(freqs_y, freqs_x, indexing='ij')  # (H, W)
        freq_sq = ux**2 + uy**2  # (H, W)

        # Compute FFT of image once
        fft_image = torch.fft.fft2(image)  # (1, 1, H, W) complex

        # Vectorized mask computation for all scales
        scales = scales.unsqueeze(-1).unsqueeze(-1)  # (num_scales, 1, 1)
        exponents = -2 * torch.pi**2 * scales**2 * freq_sq  # (num_scales, H, W)
        gmasks = torch.exp(exponents)  # (num_scales, H, W)

        # Broadcast: multiply and IFFT
        blurred_ffts = fft_image.unsqueeze(0) * gmasks.unsqueeze(1).unsqueeze(1)  # (num_scales, 1, 1, H, W) complex
        scale_space = torch.fft.ifft2(blurred_ffts).real  # (num_scales, 1, 1, H, W) -> take real part (imag near zero)

        return scale_space.squeeze(1).squeeze(1)  # (num_scales, H, W) to match your specified shape
    
    def generate_scale_space_img(self, center, world_scales, images: list[torch.Tensor]):
        """
        Computes Gaussian scale space for all images with camera distance scaling.
        
        Returns: 
            list[torch.Tensor]: list of scale spaces (num_scales, H, W)
            np.ndarray: scales in world space
            list[float]: list of per-camera scale factors (pixel_scale_sigma / world_scale_sigma)
        """
        scale_spaces = []
        per_cam_scale_factors = [] # A list of scale factors (pixel_sigma / world_sigma)

        # 1. Calculate base distance and scaling factor (using cam 0 as reference)
        cam0 = self.camera_states[0]
        dist_cam0 = np.linalg.norm(cam0.pose_np[:3] - center)
        focal_length_pix = cam0.K[0, 0].item() # Assuming square pixels and K[0,0] = K[1,1]

        # 3. Generate scale space for each camera
        for i, (cam, img) in enumerate(zip(self.camera_states, images)):
            dist_cam = np.linalg.norm(cam.pose_np[:3] - center)
            cam_scale_factor = dist_cam / dist_cam0 # relative to cam 0

            pixel_scales = torch.tensor(
                world_scales / dist_cam * focal_length_pix, 
                device=self.device, 
                dtype=torch.float32
            )

            scale_space = self.generate_scale_space_img_large_scale(img.to(dtype=torch.float32), pixel_scales)
            scale_spaces.append(scale_space)

            per_cam_scale_factors.append((focal_length_pix / dist_cam).item())

        return scale_spaces, per_cam_scale_factors
    
    @staticmethod
    def density_from_homoscedastic_gmm_2d_torch(exponent_diff, sigma, weights, W=1000, H=1000):
        """
        Efficiently rasterizes a homoscedastic 2D GMM onto a grid using vectorized PyTorch operations.
        Args:
            gmm_means_2d: (N, 2) tensor of 2D means.
            sigma: scalar standard deviation.
            weights: (N, 1) tensor of weights.
            W, H: output grid size.
        Returns:
            (H, W) tensor of density.
        """
        exponent = -0.5 * exponent_diff / (sigma ** 2)  # (N, W*H)
        norm_factor = 1 / (2 * torch.pi * sigma ** 2)
        weighted = weights * norm_factor * torch.exp(exponent)  # (N, W*H)
        return weighted.sum(dim=0).reshape(H, W)
    
    def generate_scale_space(self, center, world_scales, point_sets: list[torch.Tensor]):
        scale_spaces = []
        per_cam_scale_factors = [] # A list of scale factors (pixel_sigma / world_sigma)

        # 1. Calculate base distance and scaling factor (using cam 0 as reference)
        cam0 = self.camera_states[0]
        dist_cam0 = np.linalg.norm(cam0.T_world - center)
        focal_length_pix = cam0.K[0, 0].item() # Assuming square pixels and K[0,0] = K[1,1]

        # 3. Generate scale space for each camera
        for i, (cam, point_set) in enumerate(zip(self.camera_states, point_sets)):
            dist_cam = np.linalg.norm(cam.T_world - center)
            cam_scale_factor = dist_cam / dist_cam0 # relative to cam 0

            pixel_scales = torch.tensor(
                world_scales / dist_cam * focal_length_pix, 
                device=self.device, 
                dtype=torch.float32
            )

            swarm_projection_cuda = torch.tensor(point_set, dtype=torch.float32, device='cuda')
            projection_weights = torch.ones((swarm_projection_cuda.shape[0], 1), dtype=torch.float32, device='cuda') * 255

            W, H = cam0.W, cam0.H
            x = torch.arange(W, device='cuda', dtype=torch.float32) + 0.5
            y = torch.arange(H, device='cuda', dtype=torch.float32) + 0.5
            X, Y = torch.meshgrid(x, y, indexing='xy')  # (W, H)
            grid = torch.stack([X, Y], dim=-1)  # (W, H, 2)
            grid = grid.view(-1, 2)  # (W*H, 2)
            grid_exp = grid.unsqueeze(0)  # (1, W*H, 2)
            means = swarm_projection_cuda.unsqueeze(1)  # (N, 1, 2)
            diff = grid_exp - means  # (N, W*H, 2)
            exponent_diff = (diff ** 2).sum(dim=2)

            density_scale_space = torch.zeros(len(world_scales), H, W, device='cuda')
            for i in range(len(world_scales)):
                density_scale_space[i] = self.density_from_homoscedastic_gmm_2d_torch(exponent_diff, pixel_scales[i], projection_weights, W=W, H=H)

            scale_spaces.append(density_scale_space)

            per_cam_scale_factors.append(focal_length_pix / dist_cam)

        return scale_spaces, per_cam_scale_factors

    def estimate_scale_space_peaks_2d(self, scale_spaces: list[torch.Tensor], world_scales, per_cam_scale_factors: list[float]):
        """
        Estimates and matches 3D peaks across the scale space of the first two cameras.
        Returns: (peaks3D_SP, peaks_value, peaks2_value)
        """
        # 1. Compute maxima using 3x3 max-pooling
        peaks_SP, peaks2_SP = [], []

        for i, density_space in enumerate(scale_spaces[:2]): # Only use first two for matching
            # Pad with -inf for border handling during max pooling
            padded = F.pad(density_space, pad=(1, 1, 1, 1), mode='constant', value=float('-inf'))
            # Max pool over spatial dimensions (H, W)
            max_pool = F.max_pool2d(padded.unsqueeze(0), kernel_size=3, stride=1, padding=0, return_indices=False).squeeze(0)
            center_slice = padded[:, 1:-1, 1:-1]
            maxima = (center_slice == max_pool) & (center_slice > 1e-3)

            # For each scale, find local peaks
            current_cam_scale_factor = per_cam_scale_factors[i]

            for level in range(world_scales.shape[0]):
                # The scale passed to find_local_peaks_simple is the *pixel* sigma for that level
                pixel_scale_sigma = world_scales[level] * current_cam_scale_factor 
                peaks_torch, _ = find_local_peaks_simple(density_space[level, ...], maxima[level], pixel_scale_sigma)

                if i == 0:
                    peaks_SP.append(peaks_torch.detach().cpu().numpy())
                else:
                    peaks2_SP.append(peaks_torch.detach().cpu().numpy())
        return peaks_SP, peaks2_SP
    
    def init_adaptive_scale_selection(self, center, images=None, point_sets=None):
        test_scales = np.linspace(1, 30, 10)

        if images is not None:
            scale_spaces, per_cam_scale_factors = self.generate_scale_space_img(
                center, test_scales, images)
        else:
            scale_spaces, per_cam_scale_factors = self.generate_scale_space(
                center, test_scales, point_sets)
        
        peaks_SP, peaks2_SP = self.estimate_scale_space_peaks_2d(
            scale_spaces,
            test_scales,
            per_cam_scale_factors
        )

        peaks_num = [peaks_SP[i].shape[0] for i in range(len(peaks_SP))]
        
        adaptive_scale, popt = critical_scale_detection(test_scales, peaks_num)

        # scale lower bound
        if images is not None:
            depth = np.linalg.norm(self.camera_states[0].pose_np[:3] - center)
        else:
            depth = np.linalg.norm(self.camera_states[0].T_world - center)
        f = self.camera_states[0].intrinsics_params[0, 0]
        adaptive_scale_lower_bound = 16 * depth / f # minimum 2d scale is 16 for each object
        adaptive_scale = max(adaptive_scale, adaptive_scale_lower_bound)

        self.scale = adaptive_scale
        self.scaling_law = popt
    
    def adaptive_scale_selection(self, center, images=None, point_sets=None):
        margin = 0.1

        test_scales = np.linspace(1, (1 + margin) * self.scale, 5)

        if images is not None:
            scale_spaces, per_cam_scale_factors = self.generate_scale_space_img(
                center, test_scales, images)
        else:
            scale_spaces, per_cam_scale_factors = self.generate_scale_space(
                center, test_scales, point_sets)
        
        peaks_SP, peaks2_SP = self.estimate_scale_space_peaks_2d(
            scale_spaces,
            test_scales,
            per_cam_scale_factors
        )

        peaks_num = [peaks_SP[i].shape[0] for i in range(len(peaks_SP))]

        adaptive_scale, popt = critical_scale_detection(test_scales, peaks_num, p0=self.scaling_law)
        A, B, k, alpha = popt
        if alpha > 4: # if no power law decay is shown
            self.scaling_law = (A, B, k, alpha) # update scaling law but keep scale unchanged
        else:
            if adaptive_scale is None:
                if adaptive_scale == test_scales[-1]:
                    adaptive_scale = (1 + margin) * self.scale
                else:
                    adaptive_scale = self.scale
            else:
                # prevent sudden large change in adaptive_scale, allow maximum 10% change
                if abs(self.scale - adaptive_scale) / self.scale > margin:
                    adaptive_scale = (1 + margin*np.sign(adaptive_scale - self.scale)) * self.scale
        
        self.scale = adaptive_scale
        self.scaling_law = (A, B, k, alpha)

    def estimate_scale_space_peaks_3d(self, scale_spaces: list[torch.Tensor], peaks_SP, peaks2_SP):
        """
        Estimates and matches 3D peaks across the scale space of the first two cameras.
        Returns: (peaks3D_SP, peaks_value, peaks2_value)
        """
        if len(scale_spaces) < 2:
            raise ValueError("Need at least two scale spaces for peak estimation and matching.")
        
        if not peaks_SP or not peaks2_SP:
            raise ValueError("At least one of the cameras have no peaks.")

        cam1 = self.camera_states[0]
        cam2 = self.camera_states[1]
        density_scale_space = scale_spaces[0]
        density2_scale_space = scale_spaces[1]

        # 2. Match and Triangulate Peaks
        peaks3D_SP, peaks_value, peaks2_value = [], [], []
        
        # Compute Fundamental Matrix
        F_matrix = calculate_fundamental_matrix_pytorch(
            cam1.R, cam1.T, cam1.K, cam2.R, cam2.T, cam2.K).detach().cpu().numpy()

        P1_proj_np = cam1.K.cpu().numpy() @ cam1.P_np
        P2_proj_np = cam2.K.cpu().numpy() @ cam2.P_np
        
        for level in range(self.num_scales):
            # Match points using the Fundamental Matrix
            matches = np.array(match_points(peaks_SP[level], peaks2_SP[level], F_matrix, threshold=None))
            
            if matches.shape[0] == 0:
                peaks3D_SP.append(np.empty((0, 3)))
                peaks_value.append(torch.empty(0, dtype=torch.float, device=self.device))
                peaks2_value.append(torch.empty(0, dtype=torch.float, device=self.device))
                continue
                
            # x, y are (col, row)
            pnts_left = peaks_SP[level][matches[:, 0]].T.astype(np.float32)
            pnts_right = peaks2_SP[level][matches[:, 1]].T.astype(np.float32)
            
            # Triangulate
            pnts4D = cv2.triangulatePoints(P1_proj_np, P2_proj_np, pnts_left, pnts_right)
            peaks_3d = (pnts4D[:3, :] / pnts4D[3]).T

            # Get density values at the matched pixel locations (pnts_left[1] is row/v, pnts_left[0] is col/u)
            v1, u1 = pnts_left[1].astype(int), pnts_left[0].astype(int)
            v2, u2 = pnts_right[1].astype(int), pnts_right[0].astype(int)
            
            values1 = density_scale_space[level][v1, u1]
            values2 = density2_scale_space[level][v2, u2]
            
            peaks3D_SP.append(peaks_3d)
            peaks_value.append(values1)
            peaks2_value.append(values2)
        return peaks3D_SP, peaks_value, peaks2_value

    def filter_peaks(self, peaks3D_SP, peaks_value, peaks2_value, center):
        """
        Filter the estimated peaks based on their distance from the swarm center.

        Input:
        - peaks3D_SP: List of 3D peaks in the scale space.
        - center: Estimated 3D coordinates of the swarm center.
        
        Output:
        - Filtered list of peaks that are within a certain distance from the center.
        """
        filtered_peaks3D = []
        filtered_peaks = []
        filtered_peaks2 = []
        for level in range(self.num_scales):
            if peaks3D_SP[level].shape[0] < 5:
                filtered_peaks3D.append(peaks3D_SP[level])
                filtered_peaks.append(peaks_value[level])
                filtered_peaks2.append(peaks2_value[level])
                continue
            dist_to_center = torch.norm(torch.tensor(peaks3D_SP[level]) - torch.tensor(center), dim=1)
            valid_idx = dist_to_center <= torch.median(dist_to_center) * 3
            filtered_peaks3D.append(peaks3D_SP[level][valid_idx])
            filtered_peaks.append(peaks_value[level][valid_idx])
            filtered_peaks2.append(peaks2_value[level][valid_idx])
        return filtered_peaks3D, filtered_peaks, filtered_peaks2
    
    def setup_gaussian_scale_space(self, peaks3D_SP: list[np.ndarray], peaks_value, peaks2_value, scale_samples, 
                                   images: list[torch.Tensor], point_sets: list[np.ndarray],
                                   initGMM=None, xyz_lr=None, radius_lr=None, weights_lr=None, xyz_reg=None, radius_reg=None):
        if images is not None:
            num_estim = sum(img.sum().item() for img in images) / len(images) / 255.0
            num_cams = len(images)
        else:
            num_estim = sum(point_set.shape[0] for point_set in point_sets)/ len(point_sets)
            num_cams = len(point_sets)

        for level in range(self.num_scales):
            peaks3D = peaks3D_SP[level]
            N = peaks3D.shape[0]
            GM = self.GSP[level]

            if N == 0:
                raise ValueError("No 3D peaks detected.")
            
            # 1. Mean (XYZ)
            gmm_mean = torch.tensor(peaks3D, dtype=torch.float, device=self.device)

            # 2. Weight (weights)
            # Distribute total density mass across all found peaks for this level
            gmm_weights = torch.ones((N, 1), dtype=torch.float, device=self.device) * num_estim / N

            # 3. Radius (Scale)
            # Estimate radii based on density value and distance
            cam1 = self.camera_states[0]
            mean3d_cam = (cam1.P_np[:, :3] @ peaks3D.T).T + cam1.P_np[:, 3]

            gmm_radius = np.sqrt(255 * (num_estim / N) / (2*torch.pi)) / self.intrinsics_params[0, 0] * torch.from_numpy(mean3d_cam[:, 2]).float()
            # gmm_radius = np.sqrt(255 * (num_estim / N) / (2*torch.pi)) / self.intrinsics_params[0, 0] * torch.from_numpy(mean3d_cam[:, 2]).float() / torch.sqrt(peaks_value[level]).cpu()
            gmm_radius = gmm_radius.reshape((-1, 1)).float().cuda()

            GM.create_from_guess(gmm_mean, gmm_radius, gmm_weights, num_cams)
            GM.training_setup(xyz_lr_c=xyz_lr, radius_lr_c=radius_lr, weights_lr_c=weights_lr, xyz_reg=xyz_reg, radius_reg=radius_reg)

    def setup_gaussian_scale_space_initGMM(self, initGMM, images: list[torch.Tensor], point_sets: list[np.ndarray], 
                                           xyz_lr=None, radius_lr=None, weights_lr=None, xyz_reg=None, radius_reg=None):
        if images is not None:
            num_cams = len(images)
        else:
            num_cams = len(point_sets)

        for level in range(self.num_scales):
            GM = self.GSP[level]

            GM.create_from_guess(initGMM[level]._xyz, initGMM[level]._radius, initGMM[level]._weights, num_cams)
            GM.training_setup(xyz_lr_c=xyz_lr, radius_lr_c=radius_lr, weights_lr_c=weights_lr, xyz_reg=xyz_reg, radius_reg=radius_reg)

    def train_gaussian_scale_space(self, scale_spaces, density=None, is_store_intermediate=False, is_log=False, output_dir="",
                                   debug=False):
        for level in range(self.num_scales):
            GM = self.GSP[level]
            if GM.num_gaussians >= 4:
                outlier_indices, outlier_neighbors = get_outlier_neighbors(GM._xyz, K=3, outlier_percentage=10.0)

                unique_pairs = []
                for i, outlier_idx in enumerate(outlier_indices):
                    neighbors = outlier_neighbors[i]
                    for neighbor_idx in neighbors:
                        # Normalize the pair by sorting the indices
                        pair = list(sorted((outlier_idx.item(), neighbor_idx.item())))
                        unique_pairs.append(pair)
                unique_pairs = np.unique(np.array(unique_pairs), axis=0)
                GM.unpool_gaussians_init(unique_pairs)
            else:
                split_mask = GM._weights > 0.
                GM.split_from_source(split_mask)

                prune_mask = torch.zeros((GM._radius.shape[0],), dtype=torch.bool)
                prune_mask[:split_mask.shape[0]] = True
                GM.prune(prune_mask)

        if is_store_intermediate:
            for level in range(self.num_scales):
                GM = self.GSP[level]
                GM.clear_history()
                GM.save_checkpoint()

            for level in range(self.num_scales):
                GM = self.GSP[level]
                for iter in range(self.max_iter):
                    # print(f'level {level} iter {iter}')
                    scale_space_reconstructed, train_time, loss = \
                        GM.train_iter(iter, level, self.camera_states, scale_spaces, is_log=is_log, debug=debug)

                    GM.save_checkpoint()

                    if iter == self.max_iter - 2:
                        GM.sum_loss = loss
                    elif iter == self.max_iter - 1:
                        GM.sum_loss += loss
                save_path = os.path.join(output_dir, f"checkpoint_level_{level}.pth")
                GM.write_checkpoints(save_path)
        
        else:
            for level in range(self.num_scales):
                GM = self.GSP[level]
                GM.clear_history()
                for iter in range(self.max_iter):
                    # print(f'level {level} iter {iter}')
                    scale_space_reconstructed, train_time, loss = \
                        GM.train_iter(iter, level, self.camera_states, scale_spaces, is_log=is_log, debug=debug)

                    if iter == self.max_iter - 2:
                        GM.sum_loss = loss
                    elif iter == self.max_iter - 1:
                        GM.sum_loss += loss

        if GM.num_gaussians > 30:
            new_means, new_weights, new_cov = GMR.runnalls_algorithm_simple_torch(GM._xyz.detach().clone(), 
                                                                                GM._radius.detach().clone(), 
                                                                                GM._weights.detach().clone(), 23)
            
            GM._xyz = new_means
            GM._weights = new_weights.reshape((-1, 1))
            GM._radius = torch.sqrt(new_cov[:, 0, 0].reshape((-1, 1)))

            if torch.isnan(GM._xyz).any() or torch.isnan(GM._weights).any() or torch.isnan(GM._radius).any(): pass

        if is_log:
            for level in range(self.num_scales):
                save_path = os.path.join(output_dir, f"history_level_{level}.pth")
                self.GSP[level].save_history(save_path)
    
    def save_scene(self, scale_spaces, output_dir=""):
        torch.save({
            'R1': self.camera_states[0].R,
            'R2': self.camera_states[1].R,
            'T1': self.camera_states[0].T,
            'T2': self.camera_states[1].T,
            'pose': self.camera_states[0].pose_np,
            'pose2': self.camera_states[1].pose_np,
            'intrinsics_params': self.intrinsics_params_cuda,
            'scale': self.scale
        }, os.path.join(output_dir, f"scene.pth"))
    
    def process_frame(self, camera_states: list[CameraState], 
                      images: list[torch.Tensor]=None, point_sets: list[np.ndarray]=None, 
                      initGMM=None, is_adaptive_scale=True, 
                      positions=None, density=None, is_store_intermediate=False, is_log=False, output_dir=None,
                      xyz_lr=None, radius_lr=None, weights_lr=None, xyz_reg=None, radius_reg=None,
                      debug=False):
        """
        Processes a single frame given poses and images from multiple cameras.
        
        Args:
            poses (list[np.ndarray]): List of camera poses [x, y, z, qx, qy, qz, qw].
            images (list[torch.Tensor]): List of camera grayscale images (H, W).
            **kwargs: Configuration for training (lr, reg, log, etc.)
            
        Returns:
            tuple: (final_gmm_list, scale_spaces)
        """
        if ((is_store_intermediate == True) or (is_log == True)) and output_dir == None:
            raise ValueError("Must provide output_dir if saving")
        if len(camera_states) < 2:
            raise ValueError("Must provide at least two poses and corresponding images.")
        
        self.camera_states = camera_states

        start = time.perf_counter()
        if images is not None:
            center = self.estimate_swarm_center_image(images)
        else:
            center = self.estimate_swarm_center(point_sets)
        end = time.perf_counter()
        self.time_metrics['estimate_swarm_center'] = (end - start)*1000

        start = time.perf_counter()
        if is_adaptive_scale:
            if images is not None:
                if self.scaling_law is None or np.isnan(self.scale):
                    self.init_adaptive_scale_selection(center, images)
                else:
                    self.adaptive_scale_selection(center, images)
            else:
                if self.scaling_law is None or np.isnan(self.scale):
                    self.init_adaptive_scale_selection(center, point_sets=point_sets)
                else:
                    self.adaptive_scale_selection(center, point_sets=point_sets)
        end = time.perf_counter()
        self.time_metrics['adaptive_scale_selection'] = (end - start)*1000

        start = time.perf_counter()
        if images is not None:
            scale_spaces, per_cam_scale_factors = self.generate_scale_space_img(
                center, [self.scale], images)
        else:
            scale_spaces, per_cam_scale_factors = self.generate_scale_space(
                center, [self.scale], point_sets)
        end = time.perf_counter()
        self.time_metrics['generate_scale_space'] = (end - start)*1000

        start = time.perf_counter()
        if initGMM is None:
            peaks_SP, peaks2_SP = self.estimate_scale_space_peaks_2d(
                scale_spaces,
                np.array([self.scale]), 
                per_cam_scale_factors
            )

            peaks3D_SP, peaks_value, peaks2_value = self.estimate_scale_space_peaks_3d(
                scale_spaces,
                peaks_SP, peaks2_SP,
            )
            peaks3D_SP, peaks_value, peaks2_value = self.filter_peaks(peaks3D_SP, peaks_value, peaks2_value, center)
        end = time.perf_counter()
        self.time_metrics['estimate_scale_space_peaks'] = (end - start)*1000

        start = time.perf_counter()
        if initGMM is None:
            self.setup_gaussian_scale_space(peaks3D_SP, peaks_value, peaks2_value, [self.scale], images, point_sets, initGMM=initGMM,
                                            xyz_lr=xyz_lr, radius_lr=radius_lr, weights_lr=weights_lr, xyz_reg=xyz_reg, radius_reg=radius_reg)
        else:
            self.setup_gaussian_scale_space_initGMM(initGMM, images, point_sets, 
                                                    xyz_lr=xyz_lr, radius_lr=radius_lr, weights_lr=weights_lr, 
                                                    xyz_reg=xyz_reg, radius_reg=radius_reg)
        end = time.perf_counter()
        self.time_metrics['setup_gaussian_scale_space'] = (end - start)*1000

        start = time.perf_counter()
        self.train_gaussian_scale_space(scale_spaces, density=density, 
                                        is_store_intermediate=is_store_intermediate, is_log=is_log, output_dir=output_dir,
                                        debug=debug)
        end = time.perf_counter()
        self.time_metrics['train_gaussian_scale_space'] = (end - start)*1000
        
        if is_log:
            self.save_scene(scale_spaces, output_dir=output_dir)

        return self.GSP, scale_spaces