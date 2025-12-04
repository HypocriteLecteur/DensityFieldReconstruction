import torch
import numpy as np
from gaussian_rasterizer_simple_small import GaussianRasterizerSimpleSmall
import cv2
from density_field_reconstruction.density_peak import match_points, find_local_peaks_simple
from density_field_reconstruction.utils import calculate_fundamental_matrix_pytorch
import os
from typing import Tuple
import torch.nn.functional as F

def calculate_size_aware_repulsion_loss_gradient(
    mu: torch.Tensor, 
    sigma: torch.Tensor, 
    R_cutoff_inv: float, 
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Calculates the parallel size-aware inverse distance repulsion loss.

    This loss penalizes Gaussians whose effective separation is too small,
    where effective separation is normalized by their combined size (sigma_i + sigma_j).

    NOTE: This implementation calculates all N^2 interactions and is O(N^2) in complexity.
    For large numbers of Gaussians (N > 10,000), replace the distance matrix calculation 
    with a CUDA-accelerated radius search (e.g., using torch-cluster or PyTorch3D).

    Args:
        mu (torch.Tensor): Gaussian means, shape (N, 3). Must be on GPU for efficiency.
        sigma (torch.Tensor): Isotropic standard deviations (sigma_i), shape (N, 1).
        R_cutoff_inv (float): The inverse of the target size-normalized separation (1 / R_cutoff_size).
                              Pairs where the Overlap Score exceeds this value are penalized.
        epsilon (float): Small value to prevent division by zero.

    Returns:
        torch.Tensor: The total repulsion loss (scalar).
    """
    N = mu.size(0)
    
    # 1. Calculate N x N Euclidean Distance Matrix (Denominator)
    # torch.cdist is highly optimized for the GPU. Output shape: (N, N)
    distance_matrix = torch.cdist(mu, mu)
    
    # Add epsilon to prevent division by zero for self-interaction or near-collapse
    # Note: Self-interaction will be masked out later, but this guards against mu_i == mu_j
    distance_matrix = distance_matrix + epsilon

    # 2. Calculate N x N Combined Scale Matrix (Numerator)
    # The scale is sqrt(sigma_i^2 + sigma_j^2)
    
    # Expand sigmas to (N, 1) and (1, N) for broadcasted addition
    sigma_sq_i = sigma.pow(2)  # shape (N, 1)
    sigma_sq_j = sigma_sq_i.transpose(0, 1) # shape (1, N)
    
    # Combined variance matrix (sigma_i^2 + sigma_j^2) -> shape (N, N)
    combined_variance = sigma_sq_i + sigma_sq_j
    
    # Take the square root to get the combined scale (Numerator)
    combined_scale = torch.sqrt(combined_variance) # shape (N, N)

    # 3. Calculate the Overlap Score Matrix (Repulsion Strength)
    # Score_ij = combined_scale / distance_matrix
    # overlap_score_matrix = combined_scale / distance_matrix # shape (N, N)

    # # 4. Apply the Repulsion Loss Function
    # # Loss_ij = max(0, Score_ij - R_cutoff_inv)
    # raw_loss_matrix = torch.relu(overlap_score_matrix - R_cutoff_inv) # shape (N, N)

    # # 5. Mask Self-Interaction and Sum
    
    # # Create a mask to set the diagonal (i=j) elements to zero
    # mask = torch.ones(N, N, device=mu.device).fill_diagonal_(0.0)
    
    # # Apply the mask
    # masked_loss = raw_loss_matrix * mask
    
    # # Sum all pairs and divide by 2 since the matrix is symmetric (L_ij = L_ji)
    # total_loss = masked_loss.sum() * 0.5

    mask = (distance_matrix / combined_scale <= R_cutoff_inv).fill_diagonal_(0.0)
    diff = mu[:, None, :] - mu[None, :, :]
    # diff will have shape (N, N, 3)
    # where diff[i, j, :] = X[i, :] - X[j, :]
    partial = -combined_scale / (distance_matrix ** 3) 

    tensor_3d = mask.triu_(diagonal=1).unsqueeze(dim=2) * partial.unsqueeze(dim=2) * diff
    return torch.sum(tensor_3d, axis=1)

class GaussianModel:
    def __init__(self, optimizer_type="default", H=1000, W=1000, far_clip=2000):
        # Parameter attributes
        self._xyz = torch.empty(0)
        self._radius = torch.empty(0)
        self._weights = torch.empty(0)

        # Learning rate
        self.xyz_lr_c = None
        self.radius_lr_c = None
        self.weights_lr_c = None
        self.xyz_reg = None
        self.radius_reg = None

        # Optimizer-related state
        self.optimizer_type = optimizer_type
        self.optimizer = None

        # Configuration
        self.rasterizer_h = H
        self.rasterizer_w = W
        self.rasterizer_p_max = 512
        self.far_clip = 2000

        # Rasterizer instance
        self.GS = None # Will be initialized in create_from_guess or load
        self.GS2 = None

        self.sum_loss = None

        # Logging history
        self.metrics_history = {
            'loss_history': [],
            'grad_norm_history': [],
            'grad_norm_xyz_history': [],
            'grad_norm_radius_history': [],
            'grad_norm_weights_history': [],
        }
    
    @property
    def num_gaussians(self) -> int:
        return self._xyz.shape[0]
    
    def create_from_guess(self, gmm_mean, gmm_radius, gmm_weights):
        self._xyz = torch.nn.Parameter(gmm_mean.reshape((-1, 3)).requires_grad_(True).contiguous())
        self._radius = torch.nn.Parameter(gmm_radius.reshape((-1, 1)).requires_grad_(True).contiguous())
        self._weights = torch.nn.Parameter(gmm_weights.reshape((-1, 1)).requires_grad_(True).contiguous())
        
        self.GS = GaussianRasterizerSimpleSmall(
            H=self.rasterizer_h, W=self.rasterizer_w, P_max=self.rasterizer_p_max
        )
        self.GS2 = GaussianRasterizerSimpleSmall(
            H=self.rasterizer_h, W=self.rasterizer_w, P_max=self.rasterizer_p_max
        )
    
    def save_checkpoint(self, path):
        """Saves model from a certain iter."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            # --- Model Parameters (detached from graph) ---
            '_xyz': self._xyz.detach(),
            '_radius': self._radius.detach(),
            '_weights': self._weights.detach(),

            '_xyz_grad': self._xyz.grad.detach() if self._xyz.grad is not None else None,
            '_radius_grad': self._radius.grad.detach() if self._radius.grad is not None else None,
            '_weights_grad': self._weights.grad.detach() if self._weights.grad is not None else None,

            'xyz_reg': self.xyz_reg,
            'radius_reg': self.radius_reg,
            'xyz_lr_c': self.xyz_lr_c,
            'radius_lr_c': self.radius_lr_c,
            'weights_lr_c': self.weights_lr_c,
            
            # --- Optimizer State ---
            'optimizer_state_dict': self.optimizer.state_dict(),

            # --- Configuration ---
            'optimizer_type': self.optimizer_type,
            'rasterizer_h': self.rasterizer_h,
            'rasterizer_w': self.rasterizer_w,
            'rasterizer_p_max': self.rasterizer_p_max,
        }
        
        torch.save(checkpoint, path)
    
    def save_history(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.metrics_history, path)
    
    @classmethod
    def load_model(cls, path, device='cuda'):
        checkpoint = torch.load(path, map_location=device)

        # Create a new, empty model instance with the saved configuration
        model = cls(optimizer_type=checkpoint['optimizer_type'])
        model.rasterizer_h = checkpoint.get('rasterizer_h', 1000)
        model.rasterizer_w = checkpoint.get('rasterizer_w', 1000)
        model.rasterizer_p_max = checkpoint.get('rasterizer_p_max', 512)
        
        # Load parameters and wrap them as nn.Parameter
        model._xyz = torch.nn.Parameter(checkpoint['_xyz'].to(device).requires_grad_(True))
        model._radius = torch.nn.Parameter(checkpoint['_radius'].to(device).requires_grad_(True))
        model._weights = torch.nn.Parameter(checkpoint['_weights'].to(device).requires_grad_(True))

        model._xyz.grad = checkpoint['_xyz_grad'].to(device) if checkpoint['_xyz_grad'] is not None else None
        model._radius.grad = checkpoint['_radius_grad'].to(device) if checkpoint['_radius_grad'] is not None else None
        model._weights.grad = checkpoint['_weights_grad'].to(device) if checkpoint['_weights_grad'] is not None else None
        
        # Re-initialize the non-serializable rasterizer
        model.GS = GaussianRasterizerSimpleSmall(
            H=model.rasterizer_h, W=model.rasterizer_w, P_max=model.rasterizer_p_max
        )
        model.GS2 = GaussianRasterizerSimpleSmall(
            H=model.rasterizer_h, W=model.rasterizer_w, P_max=model.rasterizer_p_max
        )
 
        # Run training_setup to create the optimizer
        model.training_setup(
            xyz_reg=checkpoint['xyz_reg'], 
            radius_reg=checkpoint['radius_reg'],
            xyz_lr_c=checkpoint['xyz_lr_c'],
            radius_lr_c=checkpoint['radius_lr_c'],
            weights_lr_c=checkpoint['weights_lr_c'],
        )
        
        # Load the optimizer's state *after* it has been created
        if 'optimizer_state_dict' in checkpoint and model.optimizer is not None:
            try:
                model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except ValueError as e:
                print(f"Warning: Optimizer state could not be loaded: {e}. Starting with a fresh state.")

        return model

    def training_setup(self, xyz_lr_c=None, radius_lr_c=None, weights_lr_c=None, xyz_reg=None, radius_reg=None):
        # Set regularization parameters
        self.xyz_reg = xyz_reg
        self.radius_reg = radius_reg

        if xyz_lr_c is None:
            xyz_lr_c = 0.05
        self.xyz_lr_c = xyz_lr_c
        position_lr_init = torch.mean(torch.norm(self._xyz - torch.mean(self._xyz, dim=0), dim=1)) * xyz_lr_c if self.num_gaussians > 1 else 1
        # position_lr_final = torch.mean(torch.norm(self._xyz - torch.mean(self._xyz, dim=0), dim=1)) * xyz_lr_c if self.num_gaussians > 1 else 1
        # position_lr_delay_mult = 0
        # position_lr_max_steps = 30
        
        if radius_lr_c is None:
            radius_lr_c = 0.1
        self.radius_lr_c = radius_lr_c
        radius_lr = torch.mean(self._radius).item()*radius_lr_c
        
        if weights_lr_c is None:
            weights_lr_c = 0.14
        self.weights_lr_c = weights_lr_c
        weights_lr = torch.mean(self._weights).item()*weights_lr_c

        param_groups = [
            {'params': [self._xyz], 'lr': position_lr_init, "name": "xyz"},
            {'params': [self._radius], 'lr': radius_lr, "name": "radius"},
            {'params': [self._weights], 'lr': weights_lr, "name": "weight"}
        ]

        if self.optimizer_type == "default":
            # Adam with fused=True for better CUDA performance
            self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-8, weight_decay=0, fused=True)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(param_groups, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15, weight_decay=0)
        
        # self.xyz_scheduler_args = self.get_expon_lr_func(lr_init=position_lr_init,
        #                                     lr_final=position_lr_final,
        #                                     lr_delay_mult=position_lr_delay_mult,
        #                                     max_steps=position_lr_max_steps)
    
    # def update_learning_rate(self, iteration):
    #     ''' Learning rate scheduling per step '''
    #     for param_group in self.optimizer.param_groups:
    #         if param_group["name"] == "xyz":
    #             lr = self.xyz_scheduler_args(iteration)
    #             param_group['lr'] = lr
    #             return lr

    # @staticmethod
    # def get_expon_lr_func(
    #     lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000
    # ):
    #     """
    #     Copied from Plenoxels

    #     Continuous learning rate decay function. Adapted from JaxNeRF
    #     The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    #     is log-linearly interpolated elsewhere (equivalent to exponential decay).
    #     If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    #     function of lr_delay_mult, such that the initial learning rate is
    #     lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    #     to the normal learning rate when steps>lr_delay_steps.
    #     :param conf: config subtree 'lr' or similar
    #     :param max_steps: int, the number of steps during optimization.
    #     :return HoF which takes step as input
    #     """

    #     def helper(step):
    #         if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
    #             # Disable this parameter
    #             return 0.0
    #         if lr_delay_steps > 0:
    #             # A kind of reverse cosine decay.
    #             delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
    #                 0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
    #             )
    #         else:
    #             delay_rate = 1.0
    #         t = np.clip(step / max_steps, 0, 1)
    #         log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    #         return delay_rate * log_lerp

    #     return helper

    def forward_cost(self, R1, T1, R2, T2, K, density1, density2, mask=None, low_pass=None):
        '''
        Keep in mind that all gradient tensors except sum_loss are slices that get updated once another rasterize_foward_backward is called.
        '''
        radius = self._radius.detach()
        if low_pass is not None:
            radius = torch.sqrt(self._radius**2 + low_pass**2)

        if mask is None:
            grad_gmm_mean, grad_gmm_radius, grad_gmm_weights, density_estim, sum_loss = self.GS.rasterize_forward_backward(
                self._xyz, radius, self._weights,
                R1, T1, K, density1, profile=False
            )
            grad_gmm_mean2, grad_gmm_radius2, grad_gmm_weights2, density_estim2, sum_loss2 = self.GS2.rasterize_forward_backward(
                self._xyz, radius, self._weights,
                R2, T2, K, density2, profile=False
            )
        else:
            mask_ = mask.squeeze()
            grad_gmm_mean, grad_gmm_radius, grad_gmm_weights, density_estim, sum_loss = self.GS.rasterize_forward_backward(
                self._xyz[mask_], radius[mask_], self._weights[mask_],
                R1, T1, K, density1, profile=False
            )
            grad_gmm_mean2, grad_gmm_radius2, grad_gmm_weights2, density_estim2, sum_loss2 = self.GS2.rasterize_forward_backward(
                self._xyz[mask_], radius[mask_], self._weights[mask_],
                R2, T2, K, density2, profile=False
            )
        return density_estim, density_estim2, (sum_loss + sum_loss2).item()

    def forward(self, iter, R1, T1, R2, T2, K, density1, density2, low_pass=None, train_both=False):
        '''
        Keep in mind that all gradient tensors except sum_loss are slices that get updated once another rasterize_foward_backward is called.
        '''
        radius = self._radius.detach()
        if low_pass is not None:
            radius = torch.sqrt(self._radius**2 + low_pass**2)
        
        if train_both:
            grad_gmm_mean, grad_gmm_radius, grad_gmm_weights, density_estim, sum_loss = self.GS.rasterize_forward_backward(
                self._xyz, radius, self._weights,
                R1, T1, K, density1, profile=False
            )

            grad_gmm_mean2, grad_gmm_radius2, grad_gmm_weights2, density_estim2, sum_loss2 = self.GS2.rasterize_forward_backward(
                self._xyz, radius, self._weights,
                R2, T2, K, density2, profile=False
            )
            grad_gmm_mean = (grad_gmm_mean + grad_gmm_mean2) / 2
            grad_gmm_radius = (grad_gmm_radius + grad_gmm_radius2) / 2
            grad_gmm_weights = (grad_gmm_weights + grad_gmm_weights2) / 2

            # Regularization
            self._apply_gradients_and_regularize(
                grad_gmm_mean, grad_gmm_radius, grad_gmm_weights)
            return density_estim, density_estim2, ((sum_loss + sum_loss2).item()) / 2
        else:
            if iter % 2 == 0:
                grad_gmm_mean, grad_gmm_radius, grad_gmm_weights, density_estim, sum_loss = self.GS.rasterize_forward_backward(
                    self._xyz, radius, self._weights,
                    R1, T1, K, density1, profile=False
                )
            else:
                grad_gmm_mean, grad_gmm_radius, grad_gmm_weights, density_estim, sum_loss = self.GS2.rasterize_forward_backward(
                    self._xyz, radius, self._weights,
                    R2, T2, K, density2, profile=False
                )
            self._apply_gradients_and_regularize(
                grad_gmm_mean, grad_gmm_radius, grad_gmm_weights)
        return density_estim, None, sum_loss.item()
    
    def _apply_gradients_and_regularize(
            self, 
            grad_mean: torch.Tensor, 
            grad_radius: torch.Tensor, 
            grad_weights: torch.Tensor
        ):
            """Applies regularization and sets the .grad attributes for the optimizer."""
            
            radius = self._radius.detach()
            
            # --- Means (XYZ) Regularization (Repulsion Loss Gradient) ---
            if self.xyz_reg is not None and self.num_gaussians > 1:
                grad_repulsion = calculate_size_aware_repulsion_loss_gradient(
                    mu=self._xyz,
                    sigma=radius, 
                    R_cutoff_inv=1.0 # Assuming R_cutoff_inv=1 is the default
                )
                
                if grad_repulsion.any():
                    # Scale the repulsion gradient to match the rendering gradient magnitude
                    norm_render_grad = torch.norm(grad_mean)
                    norm_reg_grad = torch.norm(grad_repulsion)
                    
                    # Gradient accumulation: render_grad + scaled_reg_grad
                    scaled_grad_repulsion = (norm_render_grad / norm_reg_grad) * self.xyz_reg * grad_repulsion
                    self._xyz.grad = grad_mean.clone() + scaled_grad_repulsion
                else:
                    self._xyz.grad = grad_mean.clone()
            else:
                self._xyz.grad = grad_mean.clone()

            # --- Radius Regularization (Homogeneity Loss Gradient) ---
            if self.radius_reg is not None and self.num_gaussians > 1:
                # Homogeneity regularization: penalizes large deviations from mean log-radius
                log_sigma = torch.log(radius)
                log_sigma_rel = log_sigma - torch.mean(log_sigma)
                
                # Gradient of L_homo = sum_i (log_sigma_rel_i^2) / N 
                # d L_homo / d sigma_i = (2 * log_sigma_rel_i) / (N * sigma_i) * (1 - 1/N)
                # The original constant (24/N) is a manual tuning factor.
                grad_radius_homo = (24.0 / self.num_gaussians) * (1.0 / radius) * log_sigma_rel
                
                if torch.mean(torch.abs(grad_radius_homo)) > 0:
                    # Scale the homogeneity gradient
                    mean_abs_render_grad = torch.mean(torch.abs(grad_radius))
                    mean_abs_reg_grad = torch.mean(torch.abs(grad_radius_homo))
                    
                    # Gradient accumulation: render_grad + scaled_reg_grad
                    scaled_grad_radius_homo = (mean_abs_render_grad / mean_abs_reg_grad) * self.radius_reg * grad_radius_homo
                    self._radius.grad = grad_radius.clone() + scaled_grad_radius_homo
                else:
                    self._radius.grad = grad_radius.clone()
            else:
                self._radius.grad = grad_radius.clone()
            
            # --- Opacity/Weights Gradient ---
            self._weights.grad = grad_weights.clone()
    
    def clear_history(self):
        self.metrics_history = {
            'loss_history': [],
            'grad_norm_history': [],
            'grad_norm_xyz_history': [],
            'grad_norm_radius_history': [],
            'grad_norm_weights_history': [],
        }

    def train_iter(self, iter, R1, T1, R2, T2, K, density1, density2, low_pass=None, is_log=False, debug=False):
        if debug:
            xyz_old = self._xyz.detach().clone()
            radius_old = self._radius.detach().clone()
            weight_old = self._weights.detach().clone()

        if self.num_gaussians == 0:
            current_loss = torch.sum(torch.abs(density1 if iter % 2 == 0 else density2)).item()
            return None, None, None, current_loss
        
        # density_estim, density_estim2, loss = self.forward(iter, R1, T1, R2, T2, K, density1, density2, low_pass=low_pass, train_both=False)
        if iter >= 20 and iter % 30 == 0:
            density_estim, density_estim2, loss = self.forward(iter, R1, T1, R2, T2, K, density1, density2, low_pass=low_pass, train_both=True)
        else:
            density_estim, density_estim2, loss = self.forward(iter, R1, T1, R2, T2, K, density1, density2, low_pass=low_pass, train_both=False)
        self.optimizer.step()

        if is_log:
            self.metrics_history['loss_history'].append(loss)
            self.metrics_history['grad_norm_xyz_history'].append(torch.mean(torch.norm(self._xyz.grad, dim=1)).item())
            self.metrics_history['grad_norm_radius_history'].append(torch.mean(torch.abs(self._radius.grad)).item())
            self.metrics_history['grad_norm_weights_history'].append(torch.mean(torch.abs(self._weights.grad)).item())
            self.metrics_history['grad_norm_history'].append(np.sqrt(
                self.metrics_history['grad_norm_xyz_history'][-1]**2 + \
                self.metrics_history['grad_norm_radius_history'][-1]**2 + \
                self.metrics_history['grad_norm_weights_history'][-1]**2
                ).item())
        
        if iter % 9 == 0:
            prune_mask = torch.logical_or(self._weights < 0., self._radius < 0.).squeeze()
            if prune_mask.any():
                self.prune(prune_mask)
        
        if iter % 40 == 0 or iter == 99:
            mean3d_cam = (R1 @ self._xyz.T).T + T1
            u = K[0, 0] * mean3d_cam[:, 0] / mean3d_cam[:, 2] + K[0, 2]
            v = K[1, 1] * mean3d_cam[:, 1] / mean3d_cam[:, 2] + K[1, 2]
            prune_u = torch.logical_or(u < 0, u > self.rasterizer_w)
            prune_v = torch.logical_or(v < 0, v > self.rasterizer_h)
            prune_depth = mean3d_cam[:, 2] > self.far_clip

            prune_mask = torch.logical_or(prune_u, prune_v)
            prune_mask = torch.logical_or(prune_mask, prune_depth)

            mean3d_cam = (R2 @ self._xyz.T).T + T2
            u = K[0, 0] * mean3d_cam[:, 0] / mean3d_cam[:, 2] + K[0, 2]
            v = K[1, 1] * mean3d_cam[:, 1] / mean3d_cam[:, 2] + K[1, 2]
            prune_u = torch.logical_or(u < 0, u > self.rasterizer_w)
            prune_v = torch.logical_or(v < 0, v > self.rasterizer_h)
            prune_depth = mean3d_cam[:, 2] > self.far_clip
            prune_mask2 = torch.logical_or(prune_u, prune_v)

            prune_mask2 = torch.logical_or(prune_u, prune_v)
            prune_mask2 = torch.logical_or(prune_mask2, prune_depth)

            prune_mask_all = torch.logical_or(prune_mask, prune_mask2)
            if prune_mask_all.any():
                self.prune(prune_mask_all)
        
        # if iter >= 20 and iter % 30 == 0:
        #     # if self.num_gaussians < 4:
        #     #     split_mask = self._weights > 0.
        #     #     self.split_from_source(split_mask)

        #     #     prune_mask = torch.zeros((self._radius.shape[0],), dtype=torch.bool)
        #     #     prune_mask[:split_mask.shape[0]] = True
        #     #     self.prune(prune_mask)
        #     # else:
        #     #     outlier_indices, outlier_neighbors = get_outlier_neighbors(self._xyz, K=3, outlier_percentage=10.0)
        #     #     unique_pairs = []
        #     #     for i, outlier_idx in enumerate(outlier_indices):
        #     #         neighbors = outlier_neighbors[i]
        #     #         for neighbor_idx in neighbors:
        #     #             pair = list(sorted((outlier_idx.item(), neighbor_idx.item())))
        #     #             unique_pairs.append(pair)
        #     #     unique_pairs = np.unique(np.array(unique_pairs), axis=0)
        #     #     # start = time.perf_counter()
        #     #     if iter % 2 == 0:
        #     #         self.unpool_gaussians(unique_pairs, R1, T1, K)
        #     #     else:
        #     #         self.unpool_gaussians(unique_pairs, R2, T2, K)
        #     # end = time.perf_counter()
        #     # print(f'unpool time {((time.perf_counter() - start)*1e3):.2f}')

        #     residual1 = density1 - density_estim
        #     maxima1 = self._find_maxima_in_residual(residual1)
        #     peaks_loc, peaks_value = find_local_peaks_simple(residual1, maxima1)

        #     if peaks_value.shape[0] == 0:
        #         return density_estim, density_estim2, None, loss

        #     peaks_loc_sorted_selected = self._residual_peak_selection(peaks_loc, peaks_value)

        #     residual2 = density2 - density_estim2
        #     maxima2 = self._find_maxima_in_residual(residual2)
        #     peaks_loc, peaks_value = find_local_peaks_simple(residual2, maxima2)

        #     if peaks_value.shape[0] == 0:
        #         return density_estim, density_estim2, None, loss

        #     peaks_loc_sorted2_selected = self._residual_peak_selection(peaks_loc, peaks_value)

        #     F_matrix = calculate_fundamental_matrix_pytorch(
        #         R1, T1, K, R2, T2, K).detach().cpu().numpy()
            
        #     matches = np.array(match_points(peaks_loc_sorted_selected, 
        #                                     peaks_loc_sorted2_selected, F_matrix, threshold=None))

        #     # x, y are (col, row)
        #     pnts_left = peaks_loc_sorted_selected[matches[:, 0]].T.astype(np.float32)
        #     pnts_right = peaks_loc_sorted2_selected[matches[:, 1]].T.astype(np.float32)
            
        #     # Triangulate
        #     P1_proj_np = (K @ torch.hstack((R1, T1.reshape((-1, 1))))).detach().cpu().numpy()
        #     P2_proj_np = (K @ torch.hstack((R2, T2.reshape((-1, 1))))).detach().cpu().numpy()
        #     pnts4D = cv2.triangulatePoints(P1_proj_np, P2_proj_np, pnts_left, pnts_right)
        #     peaks_3d = (pnts4D[:3, :] / pnts4D[3]).T

        #     # import matplotlib.pyplot as plt
        #     # plt.ion()
        #     # fig = plt.figure(figsize=(10, 6))
        #     # ax = fig.add_subplot(231)
        #     # ax.imshow(density1.cpu())
        #     # ax2 = fig.add_subplot(232)
        #     # ax2.imshow(density_estim.cpu())
        #     # ax3 = fig.add_subplot(233)
        #     # ax3.imshow((density1 - density_estim).cpu())
        #     # ax3.scatter(pnts_left[0, :], pnts_left[1, :])
        #     # ax4 = fig.add_subplot(234)
        #     # ax4.imshow(density2.cpu())
        #     # ax5 = fig.add_subplot(235)
        #     # ax5.imshow(density_estim2.cpu())
        #     # ax6 = fig.add_subplot(236)
        #     # ax6.imshow((density2 - density_estim2).cpu())
        #     # ax6.scatter(pnts_right[0, :], pnts_right[1, :])
        #     # plt.show()

        #     mean3d_cam = (R1 @ torch.from_numpy(peaks_3d).cuda().T).T + T1
        #     D = mean3d_cam[:, 2]

        #     new_radius = torch.median(self._radius).item() * torch.ones((peaks_3d.shape[0], 1)).cuda()
        #     new_weights = 2*torch.pi * (K[0, 0] / D)**2 * torch.median(self._radius).item()**2 * (residual1)[pnts_left[1, :], pnts_left[0, :]] / 255
        #     new_weights = new_weights.reshape((-1, 1))

        #     self.density_preserving_spawn(
        #         torch.from_numpy(peaks_3d).cuda(), 
        #         new_radius, 
        #         new_weights,
        #         R1, T1, K
        #     )
        if torch.isnan(torch.sum(self._xyz)):
            pass
        return density_estim, density_estim2, None, loss
    
    def _find_maxima_in_residual(self, residual: torch.Tensor) -> torch.Tensor:
            """Finds local maxima in the residual image above a threshold."""
            # Pad with negative infinity to ensure border pixels are not local maxima
            padded = F.pad(residual, pad=(1, 1, 1, 1), mode='constant', value=float('-inf'))
            # 3x3 max pooling (max value in a 3x3 window)
            max_pool = F.max_pool2d(
                padded.unsqueeze(0), kernel_size=3, stride=1, padding=0, return_indices=False
            ).squeeze(0)
            
            # Slice back to the original size
            center_slice = padded[1:-1, 1:-1]
            
            # A pixel is a local maximum if its value equals the max value in the 3x3 window 
            # centered at it, AND its value is above a threshold.
            maxima = (center_slice == max_pool) & (center_slice > 0.01) # 0.01 is the original threshold

            return maxima.squeeze()

    def _residual_peak_selection(self, peaks_loc, peaks_value, percentage=10.0):
        _, peaks_value_idx = torch.sort(peaks_value, dim=0, descending=True)
        peaks_loc_sorted = peaks_loc[peaks_value_idx]

        N1 = max(int(peaks_loc.shape[0] * percentage / 100), 1)
        peaks_loc_sorted_selected = peaks_loc_sorted[:N1]
        peaks_loc_sorted_selected = peaks_loc_sorted_selected.detach().cpu().numpy()
        return peaks_loc_sorted_selected

    def split_from_source(self, split_mask):
        indices_to_split = torch.where(split_mask)[0]

        self._radius.data[indices_to_split] *= 0.4
        self._weights.data[indices_to_split] *= 0.2

        new_xyz, new_radius, _, source_indices_1D = self._split_to_tetrahedron(
            indices_to_split,
            radius_scale_factor=3.5 # Use the same factor for the new ones
        )

        # Calculate the new weights for the 4 split ones: 80% of original weights / 4
        # (M, 1) * 0.8 / 4.0. We use the *original* weights for this calculation.
        original_weights = self._weights.data[indices_to_split] / 0.2 # Reconstruct the original weight
        new_weights_base = original_weights * 0.7 / 4.0
        new_weights = new_weights_base.repeat_interleave(4, dim=0)
        
        # 3. Append the new Gaussians
        # source_indices is passed as a (M*4, 1) tensor containing the index of the single parent.
        self.densification_postfix(
            new_xyz, 
            new_radius, 
            new_weights
        )

    def _split_to_tetrahedron(
        self, indices_to_split: torch.Tensor, radius_scale_factor: float = 5.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the parameters for four new Gaussians forming a tetrahedron
        around the center of the original Gaussians.

        Args:
            indices_to_split (torch.Tensor): 1D tensor of indices of the Gaussians to split.
            radius_scale_factor (float): Factor by which the radius of the new Gaussians is scaled.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - new_xyz (torch.Tensor): Positions of the new GMM components, shape (4*M, 3).
                - new_radius (torch.Tensor): Radii of the new GMM components, shape (4*M, 1).
                - new_weight (torch.Tensor): Weights of the new GMM components, shape (4*M, 1).
                - new_indices_1D (torch.Tensor): 1D tensor of parent indices, shape (4*M).
        """
        # Select the Gaussians to be split
        split_xyz = self._xyz.data[indices_to_split]  # (M, 3)
        split_radius = self._radius.data[indices_to_split]  # (M, 1)
        split_weight = self._weights.data[indices_to_split]  # (M, 1)
        M = split_xyz.shape[0]

        # --- 1. Define Tetrahedron Vertices ---
        # Normalized coordinates for the 4 vertices of a regular tetrahedron
        # Scale factor is applied to the standard deviation (radius)
        s = split_radius * radius_scale_factor
        
        # Normalized coordinates (using a standard, regular tetrahedron basis)
        a = 1.0 / torch.sqrt(torch.tensor(3.0, device=self._xyz.device))
        b = torch.sqrt(torch.tensor(2.0/3.0, device=self._xyz.device))

        tetrahedron_offsets_normalized = torch.tensor([
            [0.0, 0.0, 1.0],
            [b * np.cos(torch.pi * 2 / 3), b * np.sin(torch.pi * 2 / 3), -a],
            [b * np.cos(torch.pi * 4 / 3), b * np.sin(torch.pi * 4 / 3), -a],
            [b, 0.0, -a]
        ], dtype=split_xyz.dtype, device=self._xyz.device) # (4, 3)

        # --- 2. Calculate Offsets and New Centers ---
        # offsets shape: (M, 4, 3)
        offsets = s.unsqueeze(1) * tetrahedron_offsets_normalized.unsqueeze(0)
        
        # New positions: (M, 4, 3). Flattened to (4*M, 3)
        new_xyz_split = split_xyz.unsqueeze(1) + offsets
        new_xyz = new_xyz_split.reshape(-1, 3)

        # --- 3. Calculate New Radius and Weight ---
        # New radius: (M, 1) scaled. Repeated 4 times -> (4*M, 1)
        new_radius_base = split_radius * np.sqrt(radius_scale_factor)
        new_radius = new_radius_base.repeat_interleave(4, dim=0)

        # New weight: (M, 1) quartered. Repeated 4 times -> (4*M, 1)
        # Note: A common strategy is to keep 80% of weight for the original, and distribute 20% to new.
        # Here, we distribute 100% of the weight to the 4 new ones.
        new_weight_base = split_weight / 4.0
        new_weight = new_weight_base.repeat_interleave(4, dim=0)
        
        # --- 4. Source Indices for Optimizer State Transfer ---
        # Each original index is the parent for 4 new Gaussians.
        new_indices_1D = indices_to_split.repeat_interleave(4)

        return new_xyz, new_radius, new_weight, new_indices_1D

    def prune(self, prune_mask):
        valid_points_mask = ~prune_mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._radius = optimizable_tensors["radius"]
        self._weights = optimizable_tensors["weight"]

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    # def unpool_gaussians(self, unique_pairs, R, T, K):
    #     '''
    #     unique_pairs: Numpy list of [outlier_index, neighbor_index]

    #     self._xyz (N, 3)
    #     self._radius (N, 1)
    #     '''
    #     new_xyz = (self._xyz[unique_pairs[:, 0], :] + self._xyz[unique_pairs[:, 1], :]) / 2
    #     new_radius = self._radius[unique_pairs[:, 1], :]

    #     new_weights = (self._weights[unique_pairs[:, 0], :] + self._weights[unique_pairs[:, 1], :]) * 0.5
    #     scaled_old_weights, new_weights = self.scale_gmm_weights_for_density_preservation_optimized(
    #         self._xyz,
    #         self._radius,
    #         self._weights,
    #         new_xyz,
    #         new_radius,
    #         new_weights,
    #         R, T, K,
    #         H, W)
    #     self._weights = scaled_old_weights

    #     self.densification_postfix(new_xyz, new_radius, new_weights, unique_pairs)

    def unpool_gaussians_init(self, unique_pairs):
        '''
        unique_pairs: Numpy list of [outlier_index, neighbor_index]
        '''
        new_xyz = (self._xyz[unique_pairs[:, 0], :] + self._xyz[unique_pairs[:, 1], :]) / 2
        new_radius = self._radius[unique_pairs[:, 1], :]
        new_weights = (self._weights[unique_pairs[:, 1], :])
        self.densification_postfix(new_xyz, new_radius, new_weights)
    
    def density_preserving_spawn(self, new_xyz, new_radius, new_weights, R, T, K):
        scaled_old_weights, new_weights = self.scale_gmm_weights_for_density_preservation_optimized(
            self._xyz,
            self._radius,
            self._weights,
            new_xyz,
            new_radius,
            new_weights,
            R, T, K,
            self.rasterizer_h, self.rasterizer_w)
        self._weights = scaled_old_weights

    def densification_postfix(self, new_xyz, new_radius, new_weights, source_indices=None):
        d = {
            'xyz': new_xyz,
            'radius': new_radius,
            'weight': new_weights
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d, source_indices)
        self._xyz = optimizable_tensors["xyz"]
        self._weights = optimizable_tensors["weight"]
        self._radius = optimizable_tensors["radius"]

    def cat_tensors_to_optimizer(self, tensors_dict, source_indices=None):
        optimizable_tensors = {}
        if source_indices is None:
            for group in self.optimizer.param_groups:
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        else:
            for group in self.optimizer.param_groups:
                param_name = group["name"]
                if param_name not in tensors_dict:
                    continue

                extension_tensor = tensors_dict[param_name]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                
                # This is the original tensor before concatenation
                original_param = group["params"][0]

                if stored_state is not None:
                    # *** MODIFIED LOGIC: Average parent states instead of using zeros ***
                    # Get the states of the two parent Gaussians for each new Gaussian
                    # source_state_1 = original_param.data[source_indices[:, 0]]
                    # source_state_2 = original_param.data[source_indices[:, 1]]
                    
                    # Average them to initialize the new state
                    avg_exp_avg = (stored_state["exp_avg"][source_indices[:, 0]] + stored_state["exp_avg"][source_indices[:, 1]]) * 0.5
                    avg_exp_avg_sq = (stored_state["exp_avg_sq"][source_indices[:, 0]] + stored_state["exp_avg_sq"][source_indices[:, 1]]) * 0.5

                    # Concatenate the new averaged states
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], avg_exp_avg), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], avg_exp_avg_sq), dim=0)

                    # Update the optimizer state
                    del self.optimizer.state[original_param]
                    group["params"][0] = torch.nn.Parameter(torch.cat((original_param.data, extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[param_name] = group["params"][0]
                else:
                    # If no state exists (e.g., first optimization step), just concatenate
                    group["params"][0] = torch.nn.Parameter(torch.cat((original_param.data, extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[param_name] = group["params"][0]

        return optimizable_tensors

    @staticmethod
    def scale_gmm_weights_for_density_preservation_optimized(
        old_xyz: torch.Tensor,
        old_radius: torch.Tensor,
        old_weights: torch.Tensor,
        new_xyz: torch.Tensor,
        new_radius: torch.Tensor,
        new_weights: torch.Tensor,
        camera_R: torch.Tensor,
        camera_T: torch.Tensor,
        camera_K: torch.Tensor,
        image_height: int,
        image_width: int,
        tile_size: int = 32,
    ) -> torch.Tensor:
        """
        Scales the weights of an old Gaussian Mixture Model (GMM) to preserve density
        after inserting new components, based on 2D projection. (Vectorized)

        Args:
            old_xyz (torch.Tensor): Positions of old GMM components, shape (N, 3).
            old_radius (torch.Tensor): Radii (std) of old GMM components, shape (N, 1).
            old_weights (torch.Tensor): Weights of old GMM components, shape (N, 1).
            new_xyz (torch.Tensor): Positions of new GMM components, shape (M, 3).
            new_radius (torch.Tensor): Radii of new GMM components, shape (M, 1).
            new_weights (torch.Tensor): Weights of new GMM components, shape (M, 1).
            camera_R (torch.Tensor): Camera rotation matrix (world to camera), shape (3, 3).
            camera_T (torch.Tensor): Camera translation vector (world to camera), shape (3,).
            camera_K (torch.Tensor): Camera intrinsics matrix, shape (3, 3).
            image_height (int): The height of the image plane.
            image_width (int): The width of the image plane.
            tile_size (int): The size of the tiles for approximation. Defaults to 32.

        Returns:
            torch.Tensor: The scaled weights for the old GMM components, shape (N, 1).
        """
        device = old_xyz.device
        scaled_old_weights = old_weights.clone()
        scaled_new_weights = new_weights.clone()

        # Step 1: Calculate camera-center positions
        old_xyz_cam = old_xyz @ camera_R.T + camera_T.unsqueeze(0)
        new_xyz_cam = new_xyz @ camera_R.T + camera_T.unsqueeze(0)

        old_dist_to_cam = old_xyz_cam[:, 2:3]
        new_dist_to_cam = new_xyz_cam[:, 2:3]
        
        # Step 2: Calculate 2D positions (uv)
        def project(xyz_cam, K):
            uvw = K @ xyz_cam.T
            # Add epsilon to prevent division by zero for points at the camera center
            uv = uvw[:2, :] / (uvw[2:3, :] + 1e-8)
            return uv.T

        old_uv = project(old_xyz_cam, camera_K)
        new_uv = project(new_xyz_cam, camera_K)

        # Step 3: Calculate 2D radius
        focal_length = (camera_K[0, 0] + camera_K[1, 1]) * 0.5
        old_radius_2d = old_radius * focal_length / old_dist_to_cam
        new_radius_2d = new_radius * focal_length / new_dist_to_cam

        # Helper function for 2D Gaussian evaluation
        def eval_gaussian_2d(uv_points, mu, radius_2d):
            dist_sq = torch.sum((uv_points.unsqueeze(1) - mu.unsqueeze(0))**2, dim=-1)
            # Add epsilon to avoid division by zero
            return torch.exp(-0.5 * dist_sq / (radius_2d.pow(2).T + 1e-8))

        # =================== Vectorized Implementation ===================

        # Step 4: Filter out Gaussians that are behind the camera or outside the frame
        valid_old_mask = (old_dist_to_cam.squeeze() > 0.1)
        valid_new_mask = (new_dist_to_cam.squeeze() > 0.1) & \
                        (new_uv[:, 0] >= 0) & (new_uv[:, 0] < image_width) & \
                        (new_uv[:, 1] >= 0) & (new_uv[:, 1] < image_height)
        
        # If there are no valid new Gaussians to process, return original weights
        if not torch.any(valid_new_mask):
            return scaled_old_weights, scaled_new_weights

        # Work only with valid Gaussians to reduce computation
        valid_old_indices = torch.where(valid_old_mask)[0]
        valid_new_indices = torch.where(valid_new_mask)[0]

        valid_old_uv = old_uv[valid_old_indices]
        valid_old_r2d = old_radius_2d[valid_old_indices]
        valid_old_weights = scaled_old_weights[valid_old_indices]

        valid_new_uv = new_uv[valid_new_indices]
        valid_new_r2d = new_radius_2d[valid_new_indices]
        valid_new_weights = scaled_new_weights[valid_new_indices]
        
        # Step 5: Find the tile center for each new Gaussian
        tile_indices_new = torch.floor(valid_new_uv / tile_size)
        tile_centers_new = (tile_indices_new + 0.5) * tile_size

        # Step 6: Identify which old Gaussians contribute to each new Gaussian's tile
        # This creates a boolean matrix of shape (num_valid_old, num_valid_new)
        # torch.cdist is highly optimized for this pairwise distance calculation
        dists_to_tile_centers = torch.cdist(valid_old_uv, tile_centers_new)
        contribution_mask = (dists_to_tile_centers < valid_old_r2d)

        # Step 7: Calculate contributions C_old and C_new in parallel
        
        # Evaluate all valid old Gaussians at all valid new tile centers
        # This results in a matrix of shape (num_valid_old, num_valid_new)
        gaussian_vals_old_at_tiles = eval_gaussian_2d(tile_centers_new, valid_old_uv, valid_old_r2d).T
        
        # Calculate C_old for each new tile by summing contributions from relevant old Gaussians
        # We use the mask to zero out non-contributing values before summing
        C_old_per_new = torch.sum(valid_old_weights * gaussian_vals_old_at_tiles * contribution_mask, dim=0)
        
        # Calculate C_new for each new Gaussian at its own tile center
        dist_sq_new = torch.sum((valid_new_uv - tile_centers_new)**2, dim=-1)
        gaussian_vals_new = torch.exp(-0.5 * dist_sq_new / (valid_new_r2d.squeeze().pow(2) + 1e-8))
        C_new_per_new = valid_new_weights.squeeze() * gaussian_vals_new
        
        # Step 8: Calculate the scaling factor for each new Gaussian's tile
        # Add epsilon to denominator to prevent division by zero
        denominator = C_old_per_new + C_new_per_new + 1e-8
        scale_factors = 0.9 * C_old_per_new / denominator
        
        # Ensure scale factors are at most 1 to avoid increasing density
        scale_factors.clamp_(max=1.0)
        
        # Step 9: Apply the scaling factors
        
        # For old weights, an old Gaussian can be affected by multiple new ones.
        # The original loop implies a multiplicative effect (w = w * s1 * s2 * ...).
        # This can be done in parallel using log-exp trick: exp(sum(log(s)))
        log_scale_factors = torch.log(scale_factors)
        # Use the mask to assign the log_scale_factor to each contributing old Gaussian
        log_updates = torch.where(contribution_mask, log_scale_factors.unsqueeze(0), 0.0)
        # Sum the log factors for each old Gaussian
        total_log_scaling_per_old = torch.sum(log_updates, dim=1)
        # Convert back and apply to the valid old weights
        final_scaling_per_old = torch.exp(total_log_scaling_per_old)
        scaled_old_weights[valid_old_indices] *= final_scaling_per_old.unsqueeze(1)
        
        # For new weights, each is scaled by the factor calculated for its tile
        scaled_new_weights[valid_new_indices] *= scale_factors.unsqueeze(1)
        
        return scaled_old_weights, scaled_new_weights