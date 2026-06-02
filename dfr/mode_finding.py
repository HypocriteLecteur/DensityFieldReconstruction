import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import DBSCAN
import time
import itertools

def analytic_solution(x, N=1000, d=2, A=1, pbc=False):
    B = N
    if d == 2:
        xi = 1/(4*np.sqrt(3)*np.pi)
        if pbc:
            k = 2 + 4.950595955353048 * N**-0.3515463947893045 # pbc
            x0 = 0.35597367503024324 * N**-0.5334539230881985
        else:
            k = 2 + 4.703183622216619 * N**-0.337789657437471 # no pbc
            x0 = 0.3991016137653485 * N**-0.5472965521349495
    else:
        if d == 3:
            xi = (29*np.sqrt(6)/288 - 1/8)/np.pi**2
            if pbc:
                k = 3 + 6.985280925198838 * N**-0.3269168879106131 # pbc
                x0 = 0.3360391539543154 * N**-0.3551798732086752
            else:
                k = 3 + 6.939639899500436 * N**-0.2627997835686439 # no pbc
                x0 = 0.4028242833700554 * N**-0.3702108300180316 # no pbc
    # x0 = (2*xi / (N+1)) ** (1 / d)
    return A + (B - A)/(1 + (x/x0)**k)

def analytic_solution_scale_at_x_percentile(x, N=1000, d=2, V=1):
    return 0
    A = 1
    B = N
    if d == 2:
        xi = 1/(4*np.sqrt(3)*np.pi)
        k = -0.04363698*np.log(N) + 2.67270022 # no pbc
        # k = -0.10265236*np.log(N) + 3.08746808 # pbc
        xd = 1.3210
    else:
        if d == 3:
            xi = (29*np.sqrt(6)/288 - 1/8)/np.pi**2
            k = -0.10553786*np.log(N) + 4.65197154 # no pbc
            # k = -0.25133963*np.log(N) + 5.50219737 # pbc
            xd = 1.2860
    return ( V * ( (B-A)/(x/100*B-A) - 1 )) ** (1/k) * xd * (xi/(B-A)) ** (1/d)

def analytic_solution_scale_at_x_constant(x, N=1000, d=2, V=1):
    A = 1
    B = N
    if d == 2:
        xi = 1/(4*np.sqrt(3)*np.pi)
        k = -0.04363698*np.log(N) + 2.67270022 # no pbc
        # k = -0.10265236*np.log(N) + 3.08746808 # pbc
        xd = 1.3210
    else:
        if d == 3:
            xi = (29*np.sqrt(6)/288 - 1/8)/np.pi**2
            k = -0.10553786*np.log(N) + 4.65197154 # no pbc
            # k = -0.25133963*np.log(N) + 5.50219737 # pbc
            xd = 1.2860
    return (( (B-A)/(x-A) - 1 ) ** (1/k) * V**(1/d) * xd * (xi/(B-A)) ** (1/d)).item()

def model_4pl_scale_at_x_constant(x, A, B, k, x0):
    return (( ( (B-A)/(x-A) - 1 )) ** (1/k) * x0).item()

def find_scale_interval(func, N, s_initial_guess=30, atol=1e-5):
    """
    Identifies the 'interesting' interval [s_start, s_end] for a monotonically 
    decreasing function f(data, s) -> n.

    Args:
        data: The dataset passed to func.
        func: The expensive function f(data, s) -> int.
        N (int): The known constant value of n when s -> 0.
        s_initial_guess (float): A starting guess for the upper bound (default 50).
        tolerance (float): How close to the target value is acceptable.

    Returns:
        tuple: (s_start, s_end) defining the region of transition.
    """
    target_high = int(N * 0.99)
    target_low = max(int(min(N * 0.01, 5)), 1) 

    # 2. Bracket the Search Space (Find an upper bound where f(s) ~ 1)
    # We start at 0 (known N) and expand outwards until we hit the floor (1).
    low_bound = 0.0
    high_bound = float(s_initial_guess)
    
    # Exponentially expand high_bound if we haven't hit the floor yet
    current_val = func(high_bound)
    while current_val > target_low:
        low_bound = high_bound # We know the transition is after the old high
        high_bound *= 2
        current_val = func(high_bound)

    # 3. Generic Binary Search Helper
    def binary_search_s(target_n, s_min, s_max):
        """
        Narrow down s until the interval size is within relative tolerance.
        Invariant: func(s_min) > target >= func(s_max)
        """
        # We limit max_iter to 100 to prevent infinite loops,
        # though tolerance usually triggers first.
        for _ in range(20):
            # Dynamic Tolerance Check:
            # We stop if the gap (s_max - s_min) is smaller than X% of the current value
            # This automatically gives coarse precision for large s, fine for small s.
            threshold = atol

            if (s_max - s_min) < threshold:
                break

            mid = max((s_min + s_max) / 2.0, 1e-12)
            val = func(mid)

            # Standard binary search for monotonic decreasing function
            if val > target_n:
                s_min = mid # The target is to the right (larger s needed to reduce n)
            else:
                s_max = mid # The target is to the left

        return (s_min + s_max) / 2.0

    # 4. Execute Search
    # Find start: Scan [0, high_bound] for target_high
    s_start = binary_search_s(target_high, 0.0, high_bound)
    
    # Find end: Scan [s_start, high_bound] for target_low
    # Optimization: We start searching from s_start, not 0
    search_min = max(s_start, low_bound)
    s_end = binary_search_s(target_low, search_min, high_bound)

    return s_start, s_end

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





# Enable TF32 tensor cores for free ~2x matmul speedup
torch.set_float32_matmul_precision('high')


def mean_shift_step_tiled(positions, modes, sigma, batch_size=1024):
    """GPU-accelerated mean-shift step with TF32 tensor cores."""
    N, d = positions.shape
    M, _ = modes.shape
    new_modes = torch.zeros_like(modes)

    sigma_safe = max(sigma, 1e-12)

    for i in range(0, M, batch_size):
        batch_end = min(i + batch_size, M)
        batch_modes = modes[i : batch_end]
        B = batch_end - i
        # Compute pairwise distances via direct differences to avoid
        # catastrophic cancellation in ||x||^2+||y||^2-2x·y for float32.
        diff = positions.unsqueeze(1) - batch_modes.unsqueeze(0)  # [N, B, d]
        dist_sq = (diff * diff).sum(dim=2)                         # [N, B]
        cdist = torch.sqrt(dist_sq)

        W = torch.exp(-0.5 * (cdist / sigma_safe) ** 2)
        weight_sum = W.sum(dim=0, keepdim=True)
        weight_sum = weight_sum.clamp(min=1e-12)
        new_modes[i : batch_end] = (W.T @ positions) / weight_sum.T

    return new_modes

def mean_shift_step(
    positions: torch.Tensor,
    modes: torch.Tensor,
    sigma: float
) -> torch.Tensor:
    # Delegate to tiled version with direct-diff (avoids torch.cdist bugs)
    return mean_shift_step_tiled(positions, modes, sigma)

def mean_shift_mask_accelerated(positions_torch: torch.Tensor, modes: torch.Tensor, sigma: float, max_iter: int=1000, tol: float=1e-2) -> torch.Tensor:
    tol = max(tol, 1e-8)  # prevent never-converging when tol=0
    old_modes = modes.clone()
    active_modes_mask = torch.ones(modes.shape[0], dtype=torch.bool, device=modes.device)

    for iter in range(max_iter):
        # Only process active modes — saves O(N * (M - M_active)) per iteration
        active_indices = torch.where(active_modes_mask)[0]
        old_active = old_modes[active_indices]
        new_active = mean_shift_step_tiled(positions_torch, old_active, sigma)

        shift_dist = torch.norm(new_active - old_active, dim=1)
        still_active = shift_dist >= tol

        # Update: mark converged modes as inactive
        active_modes_mask[active_indices[~still_active]] = False

        # Update positions of still-active modes
        active_modes_mask_copy = active_modes_mask.clone()
        old_modes[active_indices[still_active]] = new_active[still_active]

        if not active_modes_mask.any():
            break

    return old_modes, iter

def mean_shift_mask(positions_torch: torch.Tensor, modes: torch.Tensor, sigma: float, max_iter: int=1000, tol: float=1e-2) -> torch.Tensor:
    tol = max(tol, 1e-8)
    old_modes = modes.clone()
    active_modes_mask = torch.ones(modes.shape[0], dtype=torch.bool, device=modes.device)

    for iter in range(max_iter):
        active_indices = torch.where(active_modes_mask)[0]
        old_active = old_modes[active_indices]
        new_active = mean_shift_step_tiled(positions_torch, old_active, sigma)

        shift_dist = torch.norm(new_active - old_active, dim=1)
        still_active = shift_dist >= tol

        active_modes_mask[active_indices[~still_active]] = False
        old_modes[active_indices[still_active]] = new_active[still_active]

        if not active_modes_mask.any():
            break

    return old_modes, iter

def mean_shift(positions_torch: torch.Tensor, modes: torch.Tensor, sigma: float, max_iter: int=1000, tol: float=1e-2) -> torch.Tensor:
    old_modes = modes.clone()

    for iter in range(max_iter):
        new_modes = mean_shift_step(positions_torch, old_modes, sigma)

        mean_shift_dist = torch.max(torch.norm(new_modes - old_modes, dim=1).reshape((-1,)))
        if mean_shift_dist < tol:
            break

        old_modes = new_modes
    return new_modes, iter

def modes_clustering(modes: np.ndarray, distance: float):
    modes_valid = modes[~np.isnan(modes[:, 0])]
    eps = max(distance, 1e-12)  # DBSCAN requires eps > 0
    clustering = DBSCAN(eps=eps, min_samples=1).fit(modes_valid)
    number_of_cluster = np.max(clustering.labels_)+1
    cluster_center = np.zeros((number_of_cluster, modes_valid.shape[1]))
    for i in range(number_of_cluster):
        cluster_center[i, :] = np.mean(modes_valid[clustering.labels_ == i], axis=0)
    return cluster_center

def mode_counting(positions_torch: torch.Tensor, modes: torch.Tensor, scale: float, max_iter: int=1000, tol: float=1e-2):
    new_modes, iter = mean_shift_mask_accelerated(positions_torch, modes, scale, max_iter=max_iter, tol=tol)
    new_modes_np = new_modes.detach().cpu().numpy()
    clusters = modes_clustering(new_modes_np, distance=scale)
    return clusters.shape[0]

def mode_counting_modified(positions_torch: torch.Tensor, modes: torch.Tensor, scale: float, max_iter: int=1000, tol: float=1e-2):
    new_modes, iter = mean_shift_mask_accelerated(positions_torch, modes, scale, max_iter=max_iter, tol=tol)
    new_modes_np = new_modes.detach().cpu().numpy()
    clusters = modes_clustering(new_modes_np, distance=scale)
    return clusters.shape[0], clusters


def get_pbc_diff(x, y, domain_size):
    """
    Computes the difference vector (x - y) using the Minimum Image Convention.
    Result ranges from -domain_size/2 to +domain_size/2.
    """
    diff = x - y
    diff = diff - domain_size * torch.round(diff / domain_size)
    return diff

def mode_counting_pbc(positions_torch: torch.Tensor, modes: torch.Tensor, scale: float, domain_size: float=1.0, max_iter: int=1000, tol: float=1e-2):
    """
    Wrapper for counting modes with Periodic Boundary Conditions.
    """
    # 1. Run PBC Mean Shift
    new_modes, iter = mean_shift_mask_accelerated_pbc(positions_torch, modes, scale, domain_size, max_iter=max_iter, tol=tol)
    
    # 2. Prepare for Clustering (move to CPU/Numpy)
    new_modes_np = new_modes.detach().cpu().numpy()
    
    # 3. Cluster with PBC-aware distance
    # We remove NaNs (diverged modes) before clustering
    valid_mask = ~np.isnan(new_modes_np[:, 0])
    valid_modes = new_modes_np[valid_mask]
    
    if len(valid_modes) == 0:
        return 0
        
    clusters = modes_clustering_pbc(valid_modes, distance=scale, domain_size=domain_size)
    return clusters.shape[0]

def mode_counting_modified_pbc(positions_torch: torch.Tensor, modes: torch.Tensor, scale: float, domain_size: float=1.0, max_iter: int=1000, tol: float=1e-2):
    """
    Wrapper for counting modes with Periodic Boundary Conditions.
    """
    # 1. Run PBC Mean Shift
    new_modes, iter = mean_shift_mask_accelerated_pbc(positions_torch, modes, scale, domain_size, max_iter=max_iter, tol=tol)
    
    # 2. Prepare for Clustering (move to CPU/Numpy)
    new_modes_np = new_modes.detach().cpu().numpy()
    
    # 3. Cluster with PBC-aware distance
    # We remove NaNs (diverged modes) before clustering
    valid_mask = ~np.isnan(new_modes_np[:, 0])
    valid_modes = new_modes_np[valid_mask]
    
    if len(valid_modes) == 0:
        return 0
        
    clusters = modes_clustering_pbc(valid_modes, distance=scale, domain_size=domain_size)
    return clusters.shape[0], clusters

def mean_shift_mask_accelerated_pbc(positions_torch: torch.Tensor, modes: torch.Tensor, sigma: float, domain_size: float, max_iter: int=1000, tol: float=1e-2) -> torch.Tensor:
    old_modes = modes.clone()
    active_modes_mask = torch.ones(modes.shape[0], dtype=torch.bool, device=modes.device)

    for iter in range(max_iter):
        # Calculate the SHIFT vector, not the new position directly
        shift_vector = mean_shift_step_tiled_pbc(positions_torch, old_modes[active_modes_mask], sigma, domain_size)

        # Standard mean-shift step (no over-relaxation, to avoid oscillation)
        total_shift = 1.0 * shift_vector
        
        # Update modes and wrap around the domain (Modulus operator)
        # Note: We update strictly based on the shift to preserve toroidal topology
        new_active_modes = (old_modes[active_modes_mask] + total_shift) % domain_size
        
        # Check convergence based on the magnitude of the shift
        shift_dist = torch.norm(total_shift, dim=1)
        
        # Identify which modes are still moving
        still_active = shift_dist >= tol
        
        # Update the global tensor
        # We need to be careful with indexing here since we are slicing a slice
        current_active_indices = torch.nonzero(active_modes_mask).squeeze()
        
        # If only one mode is active, squeeze returns 0-d tensor, handle shape
        if current_active_indices.dim() == 0:
            current_active_indices = current_active_indices.unsqueeze(0)
            
        indices_to_update = current_active_indices  # All currently active
        indices_to_keep_active = current_active_indices[still_active]
        
        # Update positions of all currently active modes
        old_modes[indices_to_update] = new_active_modes
        
        # Update the mask for the next iteration
        active_modes_mask[:] = False
        active_modes_mask[indices_to_keep_active] = True

        if not active_modes_mask.any():
            break

    return old_modes, iter

def mean_shift_step_tiled_pbc(positions, modes, sigma, domain_size, batch_size=1024):
    """
    Computes the Mean Shift vector using PBC.
    Returns: The shift vector (delta), NOT the new absolute positions.
    """
    N, d = positions.shape
    M, _ = modes.shape
    shifts = torch.zeros_like(modes)

    for i in range(0, M, batch_size):
        batch_modes = modes[i : i + batch_size] # [batch, d]
        
        # 1. Compute Difference with Minimum Image Convention
        # We use broadcasting: (N, 1, d) - (1, batch, d) -> (N, batch, d)
        # WARNING: This uses O(N*batch*d) memory. If N is huge, reduce batch_size.
        diff = positions.unsqueeze(1) - batch_modes.unsqueeze(0)
        
        # Apply PBC wrapping to the difference
        diff = diff - domain_size * torch.round(diff / domain_size)
        
        # 2. Compute Distances and Weights
        dist_sq = diff.pow(2).sum(dim=-1) # [N, batch]
        W = torch.exp(-0.5 * (dist_sq / sigma**2)) # [N, batch]
        
        # 3. Compute Weighted Average of the *Differences* (Shifts)
        weight_sum = W.sum(dim=0, keepdim=True).T # [batch, 1]
        
        # Avoid division by zero
        weight_sum[weight_sum == 0] = 1e-10
        
        # (batch, N) @ (N, batch, d) -> This matmul is tricky with 3D tensors.
        # We can use einsum or simpler broadcasting:
        # W is (N, batch), diff is (N, batch, d)
        # Weighted sum of diffs:
        weighted_diff_sum = (diff * W.unsqueeze(-1)).sum(dim=0) # [batch, d]
        
        shifts[i : i + batch_size] = weighted_diff_sum / weight_sum
        
    return shifts

def modes_clustering_pbc(modes: np.ndarray, distance: float, domain_size: float):
    """
    DBSCAN clustering using a precomputed distance matrix with PBC.
    """
    M = modes.shape[0]
    if M == 0:
        return np.array([])

    # 1. Compute pairwise distance matrix with PBC
    # (M, 1, d) - (1, M, d)
    diff = modes[:, np.newaxis, :] - modes[np.newaxis, :, :]
    diff = diff - domain_size * np.round(diff / domain_size)
    dist_matrix = np.linalg.norm(diff, axis=2)

    # 2. Run DBSCAN with precomputed metric
    clustering = DBSCAN(eps=distance, min_samples=1, metric='precomputed').fit(dist_matrix)
    
    number_of_cluster = np.max(clustering.labels_) + 1
    cluster_center = np.zeros((number_of_cluster, modes.shape[1]))
    
    for i in range(number_of_cluster):
        members = modes[clustering.labels_ == i]
        
        # Careful! We cannot just mean(members) because of PBC wrapping.
        # e.g. mean(0.1, 9.9) = 5.0 (WRONG). Correct is 0.0 (or 10.0).
        # Fix: Pick the first point as reference, unwrap others relative to it, mean, then wrap back.
        ref_point = members[0]
        diffs = members - ref_point
        # Minimal image of diffs
        diffs = diffs - domain_size * np.round(diffs / domain_size)
        
        mean_diff = np.mean(diffs, axis=0)
        center = (ref_point + mean_diff) % domain_size
        cluster_center[i, :] = center
        
    return cluster_center