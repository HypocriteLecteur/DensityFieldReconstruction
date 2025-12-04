import torch
import numpy as np

from scipy.special import lambertw
from scipy.optimize import root_scalar

from typing import Tuple

def power_law_exponential_decay(x, A, B, k, alpha):
    return A * np.exp(-k * x) + B * x ** -alpha

def get_outlier_neighbors(points: torch.Tensor, K: int, outlier_percentage: float) -> tuple[torch.Tensor, torch.Tensor]:
    # 1. Calculate the pairwise Euclidean distance matrix
    distance_matrix = torch.cdist(points, points)

    # 2. Find the K+1 smallest distances and their indices for all points
    # The first neighbor is always the point itself with distance 0.
    distances, indices = torch.topk(distance_matrix, k=K + 1, dim=1, largest=False)
    
    # 3. Calculate the mean distances for all points
    mean_distances = torch.mean(distances[:, 1:], dim=1)

    # 4. Find the indices of the top 'outlier_percentage' points
    N = points.shape[0]
    num_outliers = int(N * outlier_percentage / 100.0)
    if num_outliers == 0 and N > 0:
        num_outliers = 1
    elif N == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    
    _, outlier_indices = torch.topk(mean_distances, k=num_outliers, largest=True)

    # 5. Retrieve the neighbor indices for the identified outliers
    # We use the outlier_indices to index into the full indices tensor
    outlier_neighbor_indices = indices[outlier_indices, 1:]

    return outlier_indices, outlier_neighbor_indices

def _compute_integral_term(
    means_a: torch.Tensor,
    weights_a: torch.Tensor,
    sigmas_a: torch.Tensor,
    means_b: torch.Tensor,
    weights_b: torch.Tensor,
    sigmas_b: torch.Tensor
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
        prefactor = (2 * torch.pi) ** (-D / 2.0) / torch.sqrt(det_cov_sum)

        # Compute inverse of covariance sum for Mahalanobis distance
        cov_sum_inv = torch.inverse(cov_sum)  # (N, M, D, D)

        # Mahalanobis distance: μ_diff^T @ Σ_inv @ μ_diff
        mu_diff = mu_diff.unsqueeze(-1)  # (N, M, D, 1)
        mahalanobis = mu_diff.transpose(-2, -1) @ cov_sum_inv @ mu_diff  # (N, M, 1, 1)
        mahalanobis = mahalanobis.squeeze(-1).squeeze(-1)  # (N, M)

        exp_term = torch.exp(-0.5 * mahalanobis)

    # Compute the integral: sum over weights * prefactor * exp_term
    result = (weights_a.T @ (prefactor * exp_term) @ weights_b).item()
    return result

def calculate_gmm_ise_gpu(
    means1_np: np.ndarray,
    sigma1: float,
    means2_torch: torch.Tensor,
    weights2_torch: torch.Tensor,
    sigmas2_torch: torch.Tensor,
    weights1=None
) -> float:
    """
    Calculates the Integrated Squared Error (ISE) between two Gaussian Mixture Models (GMMs)
    using a closed-form expression, accelerated on the GPU.

    The ISE is defined as: ISE(f, g) = ∫(f(x) - g(x))² dx
    This expands to: ∫f(x)² dx + ∫g(x)² dx - 2∫f(x)g(x) dx

    Args:
        means1_np (np.ndarray): A NumPy array of shape (N, 3) for the means of the first GMM.
        sigma1 (float): A shared standard deviation for all components of the first GMM.
        means2_torch (torch.Tensor): A Torch tensor of shape (M, 3) for the means of the second GMM,
                                     residing on a CUDA device.
        weights2_torch (torch.Tensor): A Torch tensor of shape (M, 1) for the weights of the second GMM,
                                       on the same CUDA device.
        sigmas2_torch (torch.Tensor): A Torch tensor of shape (M, 1) for standard deviations
                                      or (M, 3, 3) for covariance matrices of the second GMM,
                                      on the same CUDA device.
        weights1 (float, optional): Weight for the first GMM components. If None, uniform weights are used.

    Returns:
        float: The calculated Integrated Squared Error, a single scalar value.
    """
    # --- 1. Data Preparation and GPU Transfer ---
    # Ensure all tensors are on the same GPU device
    device = means2_torch.device
    if not (weights2_torch.device == device and sigmas2_torch.device == device):
        raise ValueError("All input torch tensors for the second GMM must be on the same CUDA device.")

    # Get dimensions
    N, D = means1_np.shape
    M = means2_torch.shape[0]

    # Convert the first GMM's parameters to torch tensors and move to GPU
    means1_torch = torch.from_numpy(means1_np).float().to(device)
    # GMM1 has uniform weights
    if weights1 is not None:
        weights1_torch = torch.full((N, 1), weights1, device=device, dtype=torch.float)
    else:
        weights1_torch = torch.full((N, 1), 1.0, device=device, dtype=torch.float)
    sigmas1_torch = torch.full((N, 1), sigma1, device=device, dtype=torch.float)

    # --- 2. Calculate the Three ISE Components ---
    # ∫f(x)² dx
    integral_f_f = _compute_integral_term(
        means1_torch, weights1_torch, sigmas1_torch,
        means1_torch, weights1_torch, sigmas1_torch
    )

    # ∫g(x)² dx
    integral_g_g = _compute_integral_term(
        means2_torch, weights2_torch, sigmas2_torch,
        means2_torch, weights2_torch, sigmas2_torch
    )

    # ∫f(x)g(x) dx
    integral_f_g = _compute_integral_term(
        means1_torch, weights1_torch, sigmas1_torch,
        means2_torch, weights2_torch, sigmas2_torch
    )
    
    # --- 3. Combine and Return Final ISE Value ---
    ise = integral_f_f + integral_g_g - 2 * integral_f_g

    return ise

# def _compute_integral_term(
#     means_a, weights_a, sigmas_a,
#     means_b, weights_b, sigmas_b
# ) -> torch.Tensor:
#     """Computes the integral of the product of two GMMs: ∫a(x)b(x) dx"""
#     # Use broadcasting to compute pairwise interactions between components
#     # (N, 1, D) and (1, M, D) -> (N, M, D)
#     mu_diff = means_a.unsqueeze(1) - means_b.unsqueeze(0)
#     dist_sq = torch.sum(mu_diff**2, dim=2) # Pairwise squared Euclidean distances

#     # (N, 1) and (1, M) -> (N, M)
#     var_a = sigmas_a**2
#     var_b = sigmas_b**2
#     var_sum = var_a.unsqueeze(1) + var_b.unsqueeze(0) # Pairwise variance sums
#     var_sum = var_sum[..., 0]

#     # Closed-form for the integral of the product of two Gaussians
#     # N(x|μ_i, σ_i^2*I) * N(x|μ_j, σ_j^2*I) is N(μ_i | μ_j, (σ_i^2+σ_j^2)*I)
#     prefactor = (2 * torch.pi * var_sum)**(-3 / 2.0)
#     exp_term = torch.exp(-dist_sq / (2 * var_sum))
    
#     return (weights_a.T @ (prefactor * exp_term) @ weights_b).item()

# def calculate_gmm_ise_gpu(
#     means1_np: np.ndarray,
#     sigma1: float,
#     means2_torch: torch.Tensor,
#     weights2_torch: torch.Tensor,
#     sigmas2_torch: torch.Tensor,
#     weights1 = None
# ) -> float:
#     """
#     Calculates the Integrated Squared Error (ISE) between two Gaussian Mixture Models (GMMs)
#     using a closed-form expression, accelerated on the GPU.

#     The ISE is defined as: ISE(f, g) = ∫(f(x) - g(x))² dx
#     This expands to: ∫f(x)² dx + ∫g(x)² dx - 2∫f(x)g(x) dx

#     Args:
#         means1_np (np.ndarray): A NumPy array of shape (N, 3) for the means of the first GMM.
#         sigma1 (float): A shared standard deviation for all components of the first GMM.
#         means2_torch (torch.Tensor): A Torch tensor of shape (M, 3) for the means of the second GMM,
#                                      residing on a CUDA device.
#         weights2_torch (torch.Tensor): A Torch tensor of shape (M, 1) for the weights of the second GMM,
#                                        on the same CUDA device.
#         sigmas2_torch (torch.Tensor): A Torch tensor of shape (M, 1) for the standard deviations
#                                       of the second GMM, on the same CUDA device.

#     Returns:
#         float: The calculated Integrated Squared Error, a single scalar value.
#     """
#     # --- 1. Data Preparation and GPU Transfer ---
#     # Ensure all tensors are on the same GPU device
#     device = means2_torch.device
#     if not (weights2_torch.device == device and sigmas2_torch.device == device):
#         raise ValueError("All input torch tensors for the second GMM must be on the same CUDA device.")

#     # Get dimensions
#     N, D = means1_np.shape
#     M = means2_torch.shape[0]

#     # Convert the first GMM's parameters to torch tensors and move to GPU
#     means1_torch = torch.from_numpy(means1_np).float().to(device)
#     # GMM1 has uniform weights
#     if weights1 is not None:
#         weights1_torch = torch.full((N, 1), weights1, device=device, dtype=torch.float)
#     else:
#         weights1_torch = torch.full((N, 1), 1.0, device=device, dtype=torch.float)
#     sigmas1_torch = torch.full((N, 1), sigma1, device=device, dtype=torch.float)

#     # --- 3. Calculate the Three ISE Components ---
#     # ∫f(x)² dx
#     integral_f_f = _compute_integral_term(
#         means1_torch, weights1_torch, sigmas1_torch,
#         means1_torch, weights1_torch, sigmas1_torch
#     )

#     # ∫g(x)² dx
#     integral_g_g = _compute_integral_term(
#         means2_torch, weights2_torch, sigmas2_torch,
#         means2_torch, weights2_torch, sigmas2_torch
#     )

#     # ∫f(x)g(x) dx
#     integral_f_g = _compute_integral_term(
#         means1_torch, weights1_torch, sigmas1_torch,
#         means2_torch, weights2_torch, sigmas2_torch
#     )
    
#     # --- 4. Combine and Return Final ISE Value ---
#     ise = integral_f_f + integral_g_g - 2 * integral_f_g

#     return ise

def find_rightmost_positive_index(arr: np.ndarray):
    """
    Finds the largest (right-most) index in a 1D array whose value is strictly
    greater than zero. This index estimates the position just before the
    final zero-crossing.

    Args:
        arr: A 1D NumPy array of numerical values.

    Returns:
        The largest index i such that arr[i] > 0.
        Returns -1 if no positive values are found in the array.
    """
    if arr.ndim != 1:
        print("Error: Input array must be 1-dimensional.")
        return -1

    # 1. Create a boolean mask where elements are positive
    mask = arr > 0

    # 2. Get the indices where the mask is True
    # np.where returns a tuple; we need the first element which is the index array
    positive_indices = np.where(mask)[0]

    # 3. Check if any positive indices were found
    if positive_indices.size == 0:
        # No positive values in the array
        return -1
    else:
        # Return the maximum (largest/right-most) index
        return np.max(positive_indices)

def solve_transcendental_equation(x0: float, A: float, B: float, k: float, alpha: float):
    """
    Numerically solves the transcendental equation A * exp(-k*x) = B * x**(-alpha) for x.
    The equation is rewritten into the root-finding problem: f(x) = A * exp(-k*x) - B * x**(-alpha) = 0.

    Args:
        x0 (float): An initial guess for the solution x.
        A (float): The constant A in the equation.
        B (float): The constant B in the equation.
        k (float): The constant k in the exponent.
        alpha (float): The constant alpha in the exponent of x.

    Returns:
        float | None: The numerical solution x if converged, otherwise None.
    """

    def f(x, A, k, B, alpha):
        # Defend against non-positive x for the x**(-alpha) term.
        if x <= 1e-9: # Use a small epsilon instead of 0 for numerical stability
            return np.inf

        # Use np.float64 for high precision in root finding
        left_side = A * np.exp(-k * x)
        right_side = B * (x ** (-alpha))

        return left_side - right_side

    # --- Root Finding ---
    try:
        # Bracket should be generous but ensure x > 0
        result = root_scalar(
            f,
            args=(A, k, B, alpha),
            x0=x0,
            # Using 'brentq' or 'brenth' is often more robust if a bracket is known
            method='brentq', 
            bracket=[1e-9, 1000.0] # Wider, non-zero positive bracket
        )

        if result.converged:
            return float(result.root)
        else:
            print(f"Warning: Solver failed to converge. Status: {result.flag}. Solution attempt: {result.root}")
            return None

    except Exception as e:
        print(f"Error during numerical solving: {e}")
        return None

def solve_with_lambertw(A, B, k, alpha):
    """
    Solves A * exp(-k*x) = B * x**(-alpha) for x using the Lambert W function.
    Returns the real solutions on the -1 branch (k=-1).
    """
    # 1. Calculate the argument 'z' for W(z)
    # The term (B/A)**(1/alpha) must be real for real solutions, so handle signs.
    ratio = B / A
    if ratio < 0 and alpha % 2 == 0:
        # If B/A is negative and alpha is an even integer, (B/A)**(1/alpha) is complex.
        # This case requires complex Lambert W, but we'll focus on real solutions here.
        raise ValueError("No real solutions exist for these parameters (ratio < 0 and alpha is an even integer).")
        
    term_pwr = ratio**(1/alpha)
    z = -(k / alpha) * term_pwr

    # 2. Check for real solution existence (z >= -1/e)
    if np.real(z) < -np.exp(-1):
        # We check the real part just in case of slight imaginary residue, but 
        # for real A, B, k, alpha, the problem is with real z < -1/e
        print(f"No real solutions exist (z = {z:.4f} is < -1/e).")
        return []
    
    solutions = []
    
    w0 = lambertw(z, k=-1)
    x0 = -(alpha / k) * w0
    
    if np.isreal(x0):
        solutions.append(np.real(x0))
    
    return solutions

def vector_to_skew_symmetric_pytorch(vector):
    batch_dims = vector.shape[:-1]
    skew = torch.zeros((*batch_dims, 3, 3), dtype=vector.dtype, device=vector.device)
    vx, vy, vz = vector[..., 0], vector[..., 1], vector[..., 2]
    skew[..., 0, 1] = -vz
    skew[..., 0, 2] = vy
    skew[..., 1, 0] = vz
    skew[..., 1, 2] = -vx
    skew[..., 2, 0] = -vy
    skew[..., 2, 1] = vx
    return skew

def calculate_fundamental_matrix_pytorch(R1, T1, K1, R2, T2, K2):
    # p2 = R_relative @ p1 + T_relative transform 3d points from cam 1 to cam 2
    R_relative = R2 @ R1.T
    T_relative = T2 - R_relative @ T1 
    return torch.inverse(K2).T @ vector_to_skew_symmetric_pytorch(T_relative) @ R_relative @ torch.inverse(K1)

