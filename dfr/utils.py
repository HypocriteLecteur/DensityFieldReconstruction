import torch
import numpy as np

def _compute_integral_term(
    means_a: torch.Tensor, weights_a: torch.Tensor, sigmas_a: torch.Tensor,
    means_b: torch.Tensor, weights_b: torch.Tensor, sigmas_b: torch.Tensor,
    use_decoupled=False, return_component_wise=False
) -> torch.Tensor:
    """Computes the integral of the product of two GMMs: ∫a(x)b(x) dx

    Args:
        means_a (torch.Tensor): Shape (N, D), means of the first GMM.
        weights_a (torch.Tensor): Shape (N, 1), weights of the first GMM.
        sigmas_a (torch.Tensor): Shape (N, 1) for scalar variances or (N, D, D) for covariance matrices.
        means_b (torch.Tensor): Shape (M, D), means of the second GMM.
        weights_b (torch.Tensor): Shape (M, 1), weights of the second GMM.
        sigmas_b (torch.Tensor): Shape (M, 1) for scalar variances or (M, D, D) for covariance matrices.

    Returns:
        torch.Tensor: Scalar value of the integral.
    """
    N, D = means_a.shape
    M = means_b.shape[0]
    
    # Check if sigmas are scalar variances (N/M, 1) or covariance matrices (N/M, D, D)
    is_scalar_a = sigmas_a.dim() == 2 and sigmas_a.shape[1] == 1
    is_scalar_b = sigmas_b.dim() == 2 and sigmas_b.shape[1] == 1

    # Pairwise mean differences: (N, 1, D) - (1, M, D) -> (N, M, D)
    mu_diff = means_a.unsqueeze(1) - means_b.unsqueeze(0)

    if is_scalar_a and is_scalar_b:
        # Case 1: Both GMMs have scalar variances (isotropic Gaussians)
        var_a = sigmas_a**2  # (N, 1)
        var_b = sigmas_b**2  # (M, 1)
        var_sum = var_a.unsqueeze(1) + var_b.unsqueeze(0)  # (N, M, 1)
        var_sum = var_sum[..., 0]  # (N, M)

        # Pairwise squared Mahalanobis distance
        dist_sq = torch.sum(mu_diff**2, dim=2)  # (N, M)

        # Prefactor: (2π)^(-D/2) / sqrt(det(Σ_a + Σ_b)) = (2π)^(-D/2) / (σ_a^2 + σ_b^2)^(D/2)
        if use_decoupled:
            prefactor = (2 * torch.pi) ** (-D / 2.0)
        else:
            prefactor = (2 * torch.pi * var_sum) ** (-D / 2.0)
        exp_term = torch.exp(-dist_sq / (2 * var_sum))
    else:
        # Case 2: At least one GMM has full covariance matrices
        # Ensure sigmas are (N/M, D, D)
        if is_scalar_a:
            sigmas_a = (sigmas_a**2).reshape((-1,1,1)) * torch.eye(D, device=means_a.device).unsqueeze(0).expand(N, D, D)
        if is_scalar_b:
            sigmas_b = (sigmas_b**2).reshape((-1,1,1)) * torch.eye(D, device=means_b.device).unsqueeze(0).expand(M, D, D)

        # Pairwise covariance sum: (N, 1, D, D) + (1, M, D, D) -> (N, M, D, D)
        cov_sum = sigmas_a.unsqueeze(1) + sigmas_b.unsqueeze(0)

        # Compute determinants of covariance sums
        det_cov_sum = torch.det(cov_sum)  # (N, M)

        # Prefactor: (2π)^(-D/2) / sqrt(det(Σ_a + Σ_b))
        if use_decoupled:
            prefactor = (2 * torch.pi) ** (-D / 2.0)
        else:
            prefactor = (2 * torch.pi) ** (-D / 2.0) / torch.sqrt(det_cov_sum)

        # Compute inverse of covariance sum for Mahalanobis distance
        cov_sum_inv = torch.inverse(cov_sum)  # (N, M, D, D)

        # Mahalanobis distance: μ_diff^T @ Σ_inv @ μ_diff
        mu_diff = mu_diff.unsqueeze(-1)  # (N, M, D, 1)
        mahalanobis = mu_diff.transpose(-2, -1) @ cov_sum_inv @ mu_diff  # (N, M, 1, 1)
        mahalanobis = mahalanobis.squeeze(-1).squeeze(-1)  # (N, M)

        exp_term = torch.exp(-0.5 * mahalanobis)

    integral_matrix = prefactor * exp_term  # Shape: (N, M)

    # Total integral scalar
    total_integral = (weights_a.T @ integral_matrix @ weights_b).item()

    if return_component_wise:
        # Collapse the 'A' dimension to see how much each 'B' component contributes
        # Shape: (1, N) @ (N, M) * (1, M) -> (1, M)
        component_wise = (weights_a.T @ integral_matrix) * weights_b.T
        return total_integral, component_wise.squeeze(0)
    
    return total_integral

# NISE
# def calculate_gmm_dissimilarity(
#     means1_np: np.ndarray,
#     sigma1: float,
#     means2_torch: torch.Tensor,
#     weights2_torch: torch.Tensor,
#     sigmas2_torch: torch.Tensor,
#     weights1=None,
#     use_decoupled=False,
#     return_removal_errors=False
# ):
#     if means2_torch.shape[0] == 0:
#         return 1.0, None if return_removal_errors else 1.0

#     # --- 1. Data Preparation and GPU Transfer ---
#     device = means2_torch.device
#     N, D = means1_np.shape
#     M = means2_torch.shape[0]

#     means1_torch = torch.from_numpy(means1_np).float().to(device)
    
#     if weights1 is not None:
#         weights1_torch = torch.full((N, 1), weights1, device=device, dtype=torch.float)
#     else:
#         weights1_torch = torch.full((N, 1), 1.0, device=device, dtype=torch.float)

#     sigmas1_torch = torch.full((N, 1), sigma1, device=device, dtype=torch.float)

#     # --- 2. Calculate the Integrals ---
#     # ∫f(x)² dx
#     int_ff = _compute_integral_term(
#         means1_torch, weights1_torch, sigmas1_torch,
#         means1_torch, weights1_torch, sigmas1_torch,
#         use_decoupled=use_decoupled
#     )

#     if return_removal_errors:
#         # ∫g(x)² dx and marginals
#         int_gg, gg_comps = _compute_integral_term(
#             means2_torch, weights2_torch, sigmas2_torch,
#             means2_torch, weights2_torch, sigmas2_torch,
#             use_decoupled=use_decoupled, return_component_wise=True
#         )

#         # ∫f(x)g(x) dx and marginals
#         int_fg, fg_comps = _compute_integral_term(
#             means1_torch, weights1_torch, sigmas1_torch,
#             means2_torch, weights2_torch, sigmas2_torch,
#             use_decoupled=use_decoupled, return_component_wise=True
#         )
        
#         # Base Error Calculation
#         denominator_old = int_ff + int_gg
#         numerator_old = 2 * int_fg
#         current_nise = 1 - (numerator_old / denominator_old)
        
#         # --- 3. Exact Removal Calculation ---
#         # Calculate the self-integral of each component: ∫g_i(x)² dx
#         # This assumes sigmas are scalar variances (isotropic Gaussians)
#         variances = sigmas2_torch ** 2
        
#         if use_decoupled:
#             # If standard NISE is decoupled from volume
#             self_integral_prefactors = (2 * torch.pi) ** (-D / 2.0)
#         else:
#             # Standard integral of squared Gaussian: (4 * pi * sigma^2)^(-D/2)
#             self_integral_prefactors = (4 * torch.pi * variances) ** (-D / 2.0)
        
#         # weight^2 * prefactor
#         self_gg_comps = (weights2_torch ** 2) * self_integral_prefactors
#         self_gg_comps = self_gg_comps.squeeze() # Shape: (M,)
        
#         # Adjust Numerator and Denominator for each component
#         N_new = numerator_old - 2 * fg_comps
#         D_new = denominator_old - 2 * gg_comps + self_gg_comps
        
#         # Calculate exact new errors array if component i is removed
#         exact_new_errors = 1 - (N_new / D_new)

#         return current_nise, exact_new_errors

#     else:
#         # Standard fast path (no component-wise calculations)
#         int_gg = _compute_integral_term(
#             means2_torch, weights2_torch, sigmas2_torch,
#             means2_torch, weights2_torch, sigmas2_torch,
#             use_decoupled=use_decoupled
#         )
#         int_fg = _compute_integral_term(
#             means1_torch, weights1_torch, sigmas1_torch,
#             means2_torch, weights2_torch, sigmas2_torch,
#             use_decoupled=use_decoupled
#         )
#         return 1 - 2 * int_fg / (int_ff + int_gg)

def calculate_gmm_dissimilarity(
    means1_np: np.ndarray,
    sigma1: float,
    means2_torch: torch.Tensor,
    weights2_torch: torch.Tensor,
    sigmas2_torch: torch.Tensor,
    weights1=None,  # Kept for signature compatibility, but we will override it
    use_decoupled=False,
    return_removal_errors=False
):
    # --- 1. Data Preparation and GPU Transfer ---
    device = means2_torch.device
    N, D = means1_np.shape
    M = means2_torch.shape[0]

    means1_torch = torch.from_numpy(means1_np).float().to(device)
    
    # Apply the new 1/N weighting scheme
    weights1_torch = torch.full((N, 1), 1.0, device=device, dtype=torch.float)
    # Scale GMM weights by 1/N
    # weights2_torch = weights2_torch
    
    sigmas1_torch = torch.full((N, 1), sigma1, device=device, dtype=torch.float)

    # If the GMM is empty, the error is simply ∫f(x)² dx
    if M == 0:
        int_ff = _compute_integral_term(
            means1_torch, weights1_torch, sigmas1_torch,
            means1_torch, weights1_torch, sigmas1_torch,
            use_decoupled=use_decoupled
        )
        return int_ff, None if return_removal_errors else int_ff

    # --- 2. Calculate the Integrals ---
    # ∫f(x)² dx
    int_ff = _compute_integral_term(
        means1_torch, weights1_torch, sigmas1_torch,
        means1_torch, weights1_torch, sigmas1_torch,
        use_decoupled=use_decoupled
    )

    if return_removal_errors:
        # ∫g(x)² dx and marginals
        int_gg, gg_comps = _compute_integral_term(
            means2_torch, weights2_torch, sigmas2_torch,
            means2_torch, weights2_torch, sigmas2_torch,
            use_decoupled=use_decoupled, return_component_wise=True
        )

        # ∫f(x)g(x) dx and marginals
        int_fg, fg_comps = _compute_integral_term(
            means1_torch, weights1_torch, sigmas1_torch,
            means2_torch, weights2_torch, sigmas2_torch,
            use_decoupled=use_decoupled, return_component_wise=True
        )
        
        # Base ISE Calculation: ∫f² - 2∫fg + ∫g²
        current_ise = int_ff - 2 * int_fg + int_gg
        
        # --- 3. Exact Removal Calculation ---
        # Calculate the self-integral of each component: ∫g_i(x)² dx
        variances = sigmas2_torch ** 2
        
        if use_decoupled:
            self_integral_prefactors = (2 * torch.pi) ** (-D / 2.0)
        else:
            self_integral_prefactors = (4 * torch.pi * variances) ** (-D / 2.0)
        
        # weight^2 * prefactor (using the new 1/N scaled weights)
        self_gg_comps = (weights2_torch ** 2) * self_integral_prefactors
        self_gg_comps = self_gg_comps.squeeze() # Shape: (M,)
        
        # Calculate exact new errors array if component i is removed
        exact_new_errors = current_ise + 2 * fg_comps - 2 * gg_comps + self_gg_comps

        current_rise = current_ise / int_ff
        exact_new_errors_rise = exact_new_errors / int_ff

        return current_rise, exact_new_errors_rise

    else:
        # Standard fast path (no component-wise calculations)
        int_gg = _compute_integral_term(
            means2_torch, weights2_torch, sigmas2_torch,
            means2_torch, weights2_torch, sigmas2_torch,
            use_decoupled=use_decoupled
        )
        int_fg = _compute_integral_term(
            means1_torch, weights1_torch, sigmas1_torch,
            means2_torch, weights2_torch, sigmas2_torch,
            use_decoupled=use_decoupled
        )
        # Base ISE Calculation

        return (int_ff - 2 * int_fg + int_gg) / int_ff

def eval_isotropic_gmm_torch(coords, means, weights, sigmas):
    """
    Evaluates GMM density using PyTorch. All inputs must be tensors on the same device.
    coords: (B, 3) tensor of query points
    means: (K, 3) tensor of Gaussian means
    weights: (K,) tensor of Gaussian weights
    sigmas: (K,) tensor of Gaussian standard deviations
    """
    # torch.cdist computes the pairwise Euclidean distance highly efficiently.
    # We square it to get the squared distances. Shape becomes (B, K)
    sq_dists = torch.cdist(coords, means).pow(2)
    
    variances = sigmas.pow(2)
    
    # 3D Gaussian normalization factor: 1 / (2 * pi * sigma^2)^(3/2)
    normalization = (2 * torch.pi * variances).pow(1.5)
    
    # Calculate exponential component: exp(-d^2 / 2*sigma^2)
    exponents = -sq_dists / (2 * variances)
    
    # Matrix of densities for each component at each point: shape (B, K)
    component_densities = (weights / normalization) * torch.exp(exponents)
    
    # Sum across all K components to get total density at each point: shape (B,)
    return torch.sum(component_densities, dim=1)

def compute_metrics_batched_torch(means1_np: np.ndarray,
                                  sigma1: float,
                                  pred_means: torch.Tensor,
                                  pred_weights: torch.Tensor,
                                  pred_sigmas: torch.Tensor,
                                  bounds, voxel_res, batch_size=500000, device='cuda'):
    """
    Computes TP, FP, and FN mass over a bounded 3D space using GPU acceleration.
    """
    # 1. Move GMM parameters to the target device (GPU)
    N = means1_np.shape[0]
    gt_means = torch.from_numpy(means1_np).cuda().float()
    gt_weights = torch.full((N, 1), 1.0, device=device, dtype=torch.float)
    gt_sigmas = torch.full((N, 1), sigma1, device=device, dtype=torch.float)

    # 2. Create 1D ticks for each axis directly on the GPU
    x_ticks = torch.arange(bounds[0][0], bounds[0][1], voxel_res, device=device)
    y_ticks = torch.arange(bounds[1][0], bounds[1][1], voxel_res, device=device)
    z_ticks = torch.arange(bounds[2][0], bounds[2][1], voxel_res, device=device)
    
    nx, ny, nz = len(x_ticks), len(y_ticks), len(z_ticks)
    total_voxels = nx * ny * nz
    voxel_volume = voxel_res**3
    
    total_tp_mass = 0.0
    total_fp_mass = 0.0
    total_fn_mass = 0.0
    
    # 3. Process in batches
    for start_idx in range(0, total_voxels, batch_size):
        end_idx = min(start_idx + batch_size, total_voxels)
        
        # 1D indices for the current batch
        batch_indices = torch.arange(start_idx, end_idx, device=device)
        
        # 4. Map 1D batch indices back to 3D grid indices using tensor arithmetic
        # This acts as a highly efficient GPU-based np.unravel_index
        ix = batch_indices // (ny * nz)
        iy = (batch_indices // nz) % ny
        iz = batch_indices % nz
        
        # 5. Construct the 3D coordinates for this specific batch
        batch_coords = torch.stack([x_ticks[ix], y_ticks[iy], z_ticks[iz]], dim=-1)
        
        # 6. Evaluate densities
        density_gt = eval_isotropic_gmm_torch(batch_coords, gt_means, gt_weights.reshape(-1,), gt_sigmas.reshape(-1,))
        density_pred = eval_isotropic_gmm_torch(batch_coords, pred_means, pred_weights.reshape(-1,), pred_sigmas.reshape(-1,))
        
        # 7. Accumulate Mass
        # Use .item() to pull the scalar sum off the GPU back to CPU float. 
        # This prevents the computation graph from holding onto memory across loop iterations.
        total_tp_mass += torch.sum(torch.minimum(density_gt, density_pred)).item() * voxel_volume
        total_fp_mass += torch.sum(torch.clamp(density_pred - density_gt, min=0)).item() * voxel_volume
        total_fn_mass += torch.sum(torch.clamp(density_gt - density_pred, min=0)).item() * voxel_volume

    return total_tp_mass.item(), total_fp_mass.item(), total_fn_mass.item()

def generate_encircling_cameras(dataset, step_range, intrinsic_params, H, W, cam_num, padding=1, is_3d=False):
    """
    Generates positions and extrinsics for cameras encircling a dynamic point dataset.
    If is_3d=False, cameras are placed on a 2D great circle in the XY plane.
    If is_3d=True, cameras are evenly distributed on a 3D sphere (S2) using a Fibonacci lattice.
    """
    # 1. Aggregate points across all time steps to find the global bounds
    all_positions = []
    for t in step_range:
        pos = dataset.positions_at_time_step(t)
        all_positions.append(pos)
    
    # Shape: (Total_N, 3)
    all_positions = np.vstack(all_positions) 

    # 2. Compute the center and the bounding sphere radius
    min_bounds = all_positions.min(axis=0)
    max_bounds = all_positions.max(axis=0)
    center = (min_bounds + max_bounds) / 2.0

    # Maximum distance from the center to any point across all time steps
    max_radius = np.max(np.linalg.norm(all_positions - center, axis=1))
    
    # Apply padding so points aren't exactly on the pixel edge
    safe_radius = max_radius * padding

    # 3. Calculate the minimum safe half-FOV from intrinsics
    fx = intrinsic_params[0, 0]
    fy = intrinsic_params[1, 1]
    cx = intrinsic_params[0, 2]
    cy = intrinsic_params[1, 2]

    # Calculate angles to the left, right, top, and bottom edges
    theta_x_left = np.arctan2(cx, fx)
    theta_x_right = np.arctan2(W - cx, fx)
    theta_y_top = np.arctan2(cy, fy)
    theta_y_bottom = np.arctan2(H - cy, fy)

    # The tightest angle dictates our required distance
    min_half_fov = min(theta_x_left, theta_x_right, theta_y_top, theta_y_bottom)

    # 4. Calculate required distance from center
    D = safe_radius / np.sin(min_half_fov)

    # 5. Generate Camera Extrinsics (World-to-Camera)
    camera_positions = []

    if not is_3d:
        # === 2D Great Circle Mode (XY Plane) ===
        for i in range(cam_num):
            angle = 2 * np.pi * i / cam_num
            
            # Position the camera on the circle (parallel to XY plane, Z is constant)
            pos_x = center[0] + D * np.cos(angle)
            pos_y = center[1] + D * np.sin(angle)
            pos_z = center[2]
            cam_pos = np.array([pos_x, pos_y, pos_z])
            camera_positions.append(cam_pos)
            
    else:
        # === 3D Sphere Distribution Mode (Fibonacci Sphere) ===
        # The Golden Angle
        phi = np.pi * (3.0 - np.sqrt(5.0)) 
        
        for i in range(cam_num):
            if cam_num == 1:
                # Handle edge case where there is only 1 camera
                z = 1.0 
            else:
                # z goes from 1 to -1
                z = 1 - (i / float(cam_num - 1)) * 2  
                
            # Radius at the current z height
            radius = np.sqrt(1 - z * z) 
            
            # Angle around the Z axis
            theta = phi * i 
            
            pos_x = center[0] + D * np.cos(theta) * radius
            pos_y = center[1] + D * np.sin(theta) * radius
            pos_z = center[2] + D * z
            
            cam_pos = np.array([pos_x, pos_y, pos_z])
            camera_positions.append(cam_pos)

    return np.array(camera_positions), D.item()