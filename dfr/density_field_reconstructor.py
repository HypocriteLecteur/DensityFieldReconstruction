import time
import torch
import numpy as np
import cv2
import os
import torch.nn.functional as F
from dfr.density_field_model import GaussianModel
from dfr.camera_system import MultiCameraSystem, convolution_cupy_wrapper
from dfr.reconstruction_scale_determination import reconstruct_visual_hull, visualize_voxel_ellipsoid_mpl
from dfr.mode_finding import analytic_solution_scale_at_x_constant
from dfr.visualizer import MultiGMMPlotter
from dfr.config import TrainingParams, ReconstructionParams
import warnings
import matplotlib.pyplot as plt
from gaussian_rasterizer_simple_large import rasterize_gaussians

class DensityReconstructor:
    def __init__(self, max_iter=100, W=1000, H=1000, far_clip=2000, use_decoupled=False):
        self.device = 'cuda'

        self.camera_system = None
        self.cameras = None

        self.max_iter = max_iter

        self.num_scales = 1
        self.scale = None

        self.use_decoupled = use_decoupled

        if self.use_decoupled:
            self.GSP = [GaussianModelDecoupled(H=H, W=W, far_clip=far_clip) for _ in range(self.num_scales)]
        else:
            self.GSP = [GaussianModel(H=H, W=W, far_clip=far_clip) for _ in range(self.num_scales)]

        self.time_metrics = {
            'estimate_swarm_center': 0.0,
            'adaptive_scale_selection': 0.0,
            'generate_scale_space': 0.0,
            'estimate_scale_space_peaks': 0.0,
            'setup_gaussian_scale_space': 0.0,
            'train_gaussian_scale_space': 0.0,
        }
    
    def estimate_swarm_center(self, point_sets: list[torch.Tensor]):
        if len(self.cameras) > 2:
            warnings.warn("Only the first two cameras will be used for swarm center estimation.", UserWarning)

        cam1 = self.cameras[0].state
        cam2 = self.cameras[1].state

        # P's are (3, 4) [R|t] matrices
        P1_proj = cam1.intrinsics_params @ cam1.P_np
        P2_proj = cam2.intrinsics_params @ cam2.P_np
        
        pnts4D = cv2.triangulatePoints(P1_proj, P2_proj, 
                                       np.mean(point_sets[0], axis=0), np.mean(point_sets[1], axis=0))
        
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
        
        if len(images) < 2:
            warnings.warn("Only the first two cameras will be used for swarm center estimation.", UserWarning)

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
        cam1 = self.cameras[0].state
        cam2 = self.cameras[1].state
        
        # P's are (3, 4) [R|t] matrices
        P1_proj = cam1.intrinsics_params @ cam1.P_np
        P2_proj = cam2.intrinsics_params @ cam2.P_np
        
        pnts4D = cv2.triangulatePoints(P1_proj, P2_proj, centroids_np[0], centroids_np[1])
        
        # Convert homogeneous coordinates to 3D
        center = (pnts4D[:3, :] / pnts4D[3]).T.reshape((3,))
        return center
    
    def reconstruction_scale_determination(self, images: list[torch.Tensor], 
                                           scale=0.5, peak_threshold=0.3, grid_max_size=32, M=30, positions=None, debug=False, point_sets=None):
        aabb, grid_size, peaks_pos, f_peaks_pos = \
            reconstruct_visual_hull(self.cameras, images, scale=scale, grid_max_size=grid_max_size, M=M, positions=positions)
        if debug:
            peaks_pos_cpu = peaks_pos.detach().cpu().numpy()
            fig = visualize_voxel_ellipsoid_mpl(peaks_pos_cpu.T, grid_size, aabb)
            ax = plt.gca()
            ax.scatter3D(positions[:, 0], positions[:, 1], positions[:, 2], s=2)
            # move_figure(fig, 2800, 100)
            plt.show()
        if peaks_pos.shape[1] == 0:
            aabb, grid_size, peaks_pos, f_peaks_pos = \
                reconstruct_visual_hull(self.cameras, images, scale=scale, grid_max_size=grid_max_size, M=M, positions=positions)
        cov = torch.cov(peaks_pos)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        voxel_sizes_np = (aabb[:, 1] - aabb[:, 0]) / grid_size
        voxel_sizes = torch.tensor(voxel_sizes_np, device=cov.device, dtype=cov.dtype)
        eigenvalues = torch.clamp(eigenvalues, min=0.0)
        radii = 2 * torch.sqrt(eigenvalues)

        # 2. Project voxel dimensions onto the eigenvectors
        # eigenvectors shape is [3, 3] where columns are the eigenvectors
        projected_voxel_thickness = torch.abs(eigenvectors).t() @ voxel_sizes

        # 3. Apply the fallback
        # A radius is half the full width, so we ensure the minimum radius is half the projected voxel thickness
        min_radii = projected_voxel_thickness / 2.0

        A, h = None, None
        if radii[2] / radii[0] > 2.5:
            A = np.pi * torch.prod(radii[1:]).item()
            h = 2 * torch.max(min_radii[0], radii[0])

        radii = torch.maximum(radii, min_radii)
        volume = ((4/3) * np.pi * torch.prod(radii)).item()
        peaks3D_SP = [f_peaks_pos]

        return A, h, volume, peaks3D_SP, radii

    @staticmethod
    def generate_scale_space_img_rfft(image, scales):
        # image: (1, 1, H, W)
        h, w = image.shape[-2:]
        
        # 1. Determine padding size. 
        # A safe margin is 4 * max_sigma to ensure the Gaussian tail drops to ~0.
        max_sigma = scales.max().item()
        pad_h = int(4 * max_sigma)
        pad_w = int(4 * max_sigma)
        
        # Apply zero padding: (left, right, top, bottom)
        padded_image = F.pad(image, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        ph, pw = padded_image.shape[-2:]

        # 2. RFFT on padded image
        fft_image = torch.fft.rfft2(padded_image)
        
        freqs_y = torch.fft.fftfreq(ph, device=image.device)
        freqs_x = torch.fft.rfftfreq(pw, device=image.device)
        uy, ux = torch.meshgrid(freqs_y, freqs_x, indexing='ij')
        freq_sq = ux**2 + uy**2

        # 3. Apply Gaussian masks
        scales_grid = scales.view(-1, 1, 1)
        exponents = -2 * (torch.pi**2) * (scales_grid**2) * freq_sq
        gmasks = torch.exp(exponents)
        
        blurred_ffts = fft_image * gmasks.unsqueeze(1)
        
        # 4. IFFT back to spatial domain
        scale_space_padded = torch.fft.irfft2(blurred_ffts, s=(ph, pw))
        
        # 5. Crop back to original dimensions
        # scale_space_padded is (num_scales, 1, ph, pw)
        # We remove the pad_h and pad_w from each side
        output = scale_space_padded[:, 0, pad_h:pad_h+h, pad_w:pad_w+w]
        
        return output
    
    def generate_scale_space_img(self, center, world_scales, images: list[torch.Tensor]):
        """
        Computes Gaussian scale space for all images with camera distance scaling.
        
        Returns: 
            list[torch.Tensor]: list of scale spaces (num_scales, H, W)
            np.ndarray: scales in world space
            list[float]: list of per-camera scale factors (pixel_scale_sigma / world_scale_sigma)
        """
        scale_spaces = []

        for i, (cam, img) in enumerate(zip(self.cameras, images)):
            dist_cam = np.linalg.norm(cam.state.camera_center - center)

            pixel_scales = torch.tensor(
                world_scales / dist_cam * cam.state.intrinsics_params[0, 0].item(),
                device=self.device, 
                dtype=torch.float32
            )

            scale_space = self.generate_scale_space_img_rfft(img.to(dtype=torch.float32), pixel_scales)
            scale_spaces.append(scale_space)
        return scale_spaces

    def setup_gaussian_scale_space(self, peaks3D_SP: list[np.ndarray | torch.Tensor], scale_samples,  
                                   images: list[torch.Tensor], point_sets: list[np.ndarray],
                                   positions=None,
                                   volume=None):
        if images is not None:
            num_estim = sum(img.sum().item() for img in images) / len(images) / 255.0
            num_cams = len(images)
        else:
            num_estim = sum(point_set.shape[0] for point_set in point_sets)/ len(point_sets)
            num_cams = len(point_sets)
        
        if positions is not None:
            num_estim = positions.shape[0]

        for level in range(self.num_scales):
            peaks3D = peaks3D_SP[level]
            N = peaks3D.shape[0]
            GM = self.GSP[level]

            if N == 0:
                raise ValueError("No 3D peaks detected.")
            
            # 1. Mean (XYZ)
            if type(peaks3D) != torch.Tensor:
                gmm_mean = torch.tensor(peaks3D, dtype=torch.float, device=self.device)
            else:
                gmm_mean = peaks3D.to(self.device)

            # 2. Weight (weights)
            # Distribute total density mass across all found peaks for this level
            gmm_weights = torch.ones((N, 1), dtype=torch.float, device=self.device) * num_estim / N

            # 3. Radius (Scale)
            gmm_radius = torch.ones((N, 1), dtype=torch.float, device=self.device) * (volume/N *3/4/torch.pi)**(1/3)
            gmm_radius = gmm_radius.reshape((-1, 1)).float().cuda()

            if self.use_decoupled: # apply scaling factor to account for gaussian normalization
                d = torch.linalg.norm(gmm_mean - self.cameras[0].state.T, dim=1)
                gmm_weights = gmm_weights * ((d / self.cameras[0].state.intrinsics_params[0, 0] / gmm_radius[:, 0])**2).reshape((-1, 1))

            if torch.sum(gmm_radius <= 0).item() > 0:
                raise ValueError("Invalid radius.")

            if torch.sum(gmm_weights <= 0).item() > 0:
                raise ValueError("Invalid weights.")
            
            # gmm_visualizer = MultiGMMPlotter()
            # gmm_visualizer.add_gmm(gmm_mean.detach().cpu().numpy(), gmm_radius.detach().cpu().numpy(), gmm_weights.detach().cpu().numpy())
            # gmm_visualizer.update()
            # move_figure(gmm_visualizer.fig, 2800, 100)
            # gmm_visualizer.ax.view_init(elev=33, azim=-117, roll=0)
            # # Save as a vector PDF or high-res PNG
            # # fig.savefig("gmm_diagram.pdf", transparent=True, bbox_inches='tight')
            # plt.show()

            GM.create_from_guess(gmm_mean, gmm_radius, gmm_weights, num_cams)
            GM.training_setup(xyz_lr_c=self.train_params['xyz_lr_c'], xyz_lr_final_c=self.train_params['xyz_lr_final_c'], 
                              radius_lr_c=self.train_params['radius_lr_c'], radius_lr_final_c=self.train_params['radius_lr_final_c'], 
                              weights_lr_c=self.train_params['weights_lr_c'], weights_lr_final_c=self.train_params['weights_lr_final_c'], 
                              xyz_reg=self.train_params['xyz_reg'], radius_reg=self.train_params['radius_reg'], radius_cutoff_inv=self.train_params['radius_cutoff_inv'],
                              lr_max_steps=self.train_params['lr_max_steps'])

    def setup_gaussian_scale_space_initGMM(self, initGMM, images: list[torch.Tensor], point_sets: list[np.ndarray], 
                                           positions=None):
        if images is not None:
            num_cams = len(images)
        else:
            num_cams = len(point_sets)

        for level in range(self.num_scales):
            GM = self.GSP[level]

            GM.create_from_guess(initGMM[level]._xyz, initGMM[level]._radius, initGMM[level]._weights, num_cams)
            GM.training_setup(xyz_lr_c=self.train_params['xyz_lr_c'], xyz_lr_final_c=self.train_params['xyz_lr_final_c'], 
                              radius_lr_c=self.train_params['radius_lr_c'], radius_lr_final_c=self.train_params['radius_lr_final_c'], 
                              weights_lr_c=self.train_params['weights_lr_c'], weights_lr_final_c=self.train_params['weights_lr_final_c'], 
                              xyz_reg=self.train_params['xyz_reg'], radius_reg=self.train_params['radius_reg'], radius_cutoff_inv=self.train_params['radius_cutoff_inv'],
                              lr_max_steps=self.train_params['lr_max_steps'])

    def train_gaussian_scale_space(self, scale_spaces, is_store_intermediate=False, is_log=False, output_dir="",
                                   debug=False):
        # for level in range(self.num_scales):
        #     GM = self.GSP[level]
        #     if GM.num_gaussians >= 4:
        #         outlier_indices, outlier_neighbors = get_outlier_neighbors(GM._xyz, K=3, outlier_percentage=10.0)

        #         unique_pairs = []
        #         for i, outlier_idx in enumerate(outlier_indices):
        #             neighbors = outlier_neighbors[i]
        #             for neighbor_idx in neighbors:
        #                 # Normalize the pair by sorting the indices
        #                 pair = list(sorted((outlier_idx.item(), neighbor_idx.item())))
        #                 unique_pairs.append(pair)
        #         unique_pairs = np.unique(np.array(unique_pairs), axis=0)
        #         GM.unpool_gaussians_init(unique_pairs)
        #     else:
        #         split_mask = GM._weights > 0.
        #         GM.split_from_source(split_mask)

        #         prune_mask = torch.zeros((GM._radius.shape[0],), dtype=torch.bool)
        #         prune_mask[:split_mask.shape[0]] = True
        #         GM.prune(prune_mask)

        if is_store_intermediate:
            for level in range(self.num_scales):
                GM = self.GSP[level]
                GM.clear_history()
                GM.save_checkpoint()

            for level in range(self.num_scales):
                GM = self.GSP[level]
                for iter in range(self.max_iter):
                    # print(f'level {level} iter {iter}')
                    GM.update_learning_rate(iter)
                    scale_space_reconstructed, train_time, loss = \
                        GM.train_iter(iter, level, [c.state for c in self.cameras], scale_spaces, is_log=is_log, debug=debug)

                    GM.save_checkpoint()

                _, mean_loss = GM.forward_cost(level, [c.state for c in self.cameras], scale_spaces)
                GM.mean_loss = mean_loss

                save_path = os.path.join(output_dir, f"checkpoint_level_{level}.pth")
                GM.write_checkpoints(save_path)
        
        else:
            for level in range(self.num_scales):
                GM = self.GSP[level]
                GM.clear_history()

                for iter in range(self.max_iter):
                    # print(f'level {level} iter {iter}')
                    GM.update_learning_rate(iter)
                    scale_space_reconstructed, train_time, loss = \
                        GM.train_iter(iter, level, [c.state for c in self.cameras], scale_spaces, is_log=is_log, debug=debug)

                _, mean_loss = GM.forward_cost(level, [c.state for c in self.cameras], scale_spaces)
                GM.mean_loss = mean_loss

        # if GM.num_gaussians > 30:
        #     new_means, new_weights, new_cov = GMR.runnalls_algorithm_simple_torch(GM._xyz.detach().clone(), 
        #                                                                         GM._radius.detach().clone(), 
        #                                                                         GM._weights.detach().clone(), 23)
            
        #     GM._xyz = new_means
        #     GM._weights = new_weights.reshape((-1, 1))
        #     GM._radius = torch.sqrt(new_cov[:, 0, 0].reshape((-1, 1)))

        #     if torch.isnan(GM._xyz).any() or torch.isnan(GM._weights).any() or torch.isnan(GM._radius).any(): pass

        if is_log:
            for level in range(self.num_scales):
                save_path = os.path.join(output_dir, f"history_level_{level}.pth")
                self.GSP[level].save_history(save_path)

    def process_frame(self, camera_system: MultiCameraSystem, 
                      images: list[torch.Tensor]=None, point_sets: list[np.ndarray]=None, 
                      initGMM=None, is_adaptive_scale=True, scale=None, estimate_scale_only=False,
                      positions=None, is_store_intermediate=False, is_log=False, output_dir=None,
                      train_params=None,
                      reconstruction_params=None,
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
        # Normalize configs: accept either dict or typed dataclass
        if isinstance(train_params, TrainingParams):
            train_params = train_params.to_dict()
        if isinstance(reconstruction_params, ReconstructionParams):
            reconstruction_params = reconstruction_params.to_dict()

        targetd_num_mode = reconstruction_params['targetd_num_mode']
        voxel_scale = reconstruction_params['voxel_scale']
        voxel_peak_threshold = reconstruction_params['voxel_peak_threshold']
        voxel_grid_max_size = reconstruction_params['voxel_grid_max_size']
        voxel_peaks_number = reconstruction_params['voxel_peaks_number']

        # Setup Reconstructor Meta-Parameters
        self.train_params = train_params

        if ((is_store_intermediate == True) or (is_log == True)) and output_dir == None:
            raise ValueError("Must provide output_dir if saving")
        
        self.camera_system = camera_system
        self.cameras = camera_system.cameras

        if len(self.cameras) < 2:
            raise ValueError("Must provide at least two poses and corresponding images.")
        
        # 1. Estimate swarm center
        start = time.perf_counter()
        if images is not None:
            center = self.estimate_swarm_center_image(images)
        else:
            center = self.estimate_swarm_center(point_sets)
        end = time.perf_counter()
        self.time_metrics['estimate_swarm_center'] = (end - start)*1000

        # 2. Reconstruction Scale Determination
        start = time.perf_counter()
        if images is not None:
            A, h, volume, peaks3D_SP, radii = self.reconstruction_scale_determination(images, voxel_scale, voxel_peak_threshold, voxel_grid_max_size, voxel_peaks_number, positions=positions, debug=debug)
        else:
            images_pnt_set = []
            N = point_sets[0].shape[0]
            for i, cam in enumerate(self.cameras):
                points_2d_torch = torch.tensor(point_sets[i], dtype=torch.float32).cuda()
                dist_cam = np.linalg.norm(cam.state.camera_center - center)
                radius = (voxel_scale / dist_cam * cam.state.intrinsics_params[0, 0].item()).item()
                height, width = cam.state.H, cam.state.W
                img = convolution_cupy_wrapper(points_2d_torch, radius, height, width, sigma_multiple=4.0)
                if torch.sum(img).item() == torch.nan:
                    raise ValueError("ASLDKJASLDKJASLDK")
                images_pnt_set.append(img)

            A, h, volume, peaks3D_SP, radii = self.reconstruction_scale_determination(images_pnt_set, voxel_scale, voxel_peak_threshold, voxel_grid_max_size, voxel_peaks_number, positions=positions, debug=debug, point_sets=point_sets)
        self.A = A
        self.h = h
        self.volume = volume
        self.radii = radii
        if not is_adaptive_scale:
            if scale is None:
                raise ValueError("Scale cannot be None")
            self.scale = scale
        else:
            if self.A is None:
                self.scale = analytic_solution_scale_at_x_constant(targetd_num_mode, N=positions.shape[0], d=3, V=self.volume)
            else:
                self.scale = analytic_solution_scale_at_x_constant(targetd_num_mode, N=positions.shape[0], d=2, V=self.A)
        end = time.perf_counter()
        self.time_metrics['adaptive_scale_selection'] = (end - start)*1000

        if estimate_scale_only:
            return None, None

        # 3. Generate Gaussian Scale Space
        # scale_spaces = []
        # for camera in camera_system.cameras:
        #     scale_spaces.append(rasterize_gaussians(
        #         torch.tensor(positions, dtype=torch.float32).cuda(),
        #         torch.ones((N, 1), dtype=torch.float, device=camera.state.device) * self.scale,
        #         torch.ones((N, 1), dtype=torch.float, device=camera.state.device),
        #         camera.state.R,
        #         camera.state.T,
        #         camera.state.K,
        #         camera.state.H,
        #         camera.state.W,
        #         False
        #     ).reshape((1, camera.state.H, camera.state.W)))
        start = time.perf_counter()
        if images is not None:
            scale_spaces = self.generate_scale_space_img(center, [self.scale], images)
        else:
            if self.scale > voxel_scale:
                scale_spaces = self.generate_scale_space_img(center, [np.sqrt(self.scale**2 - voxel_scale**2).item()], images_pnt_set)
            else:
                scale_spaces = []
                N = point_sets[0].shape[0]
                for i, cam in enumerate(self.cameras):
                    points_2d_torch = torch.tensor(point_sets[i], dtype=torch.float32).cuda()
                    dist_cam = np.linalg.norm(cam.state.camera_center - center)
                    radius = (self.scale / dist_cam * cam.state.intrinsics_params[0, 0].item()).item()
                    height, width = cam.state.H, cam.state.W
                    img = convolution_cupy_wrapper(points_2d_torch, radius, height, width, sigma_multiple=4.0)
                    if torch.sum(img).item() == torch.nan:
                        raise ValueError("ASLDKJASLDKJASLDK")
                    scale_spaces.append(img.reshape(-1, height, width))
            # scale_spaces = self.generate_scale_space_img(center, [self.scale], images_pnt_set)
        end = time.perf_counter()
        self.time_metrics['generate_scale_space'] = (end - start)*1000

        start = time.perf_counter()
        if initGMM is None:
            self.setup_gaussian_scale_space(peaks3D_SP, [self.scale], images, point_sets, positions=positions,
                                            volume=volume)
        else:
            self.setup_gaussian_scale_space_initGMM(initGMM, images, point_sets, positions=positions)
        end = time.perf_counter()
        self.time_metrics['setup_gaussian_scale_space'] = (end - start)*1000

        start = time.perf_counter()
        self.train_gaussian_scale_space(scale_spaces,
                                        is_store_intermediate=is_store_intermediate, is_log=is_log, output_dir=output_dir,
                                        debug=debug)
        end = time.perf_counter()
        self.time_metrics['train_gaussian_scale_space'] = (end - start)*1000

        return self.GSP, scale_spaces