import numpy as np
import torch
from sklearn.cluster import KMeans
from typing import Tuple

def get_indexes_by_value(arr):
    # Flatten the array to a 1D vector for easier processing
    arr_flat = arr.flatten()

    # Get the unique values from the array
    unique_values = np.unique(arr_flat)

    # Use a dictionary comprehension to build the desired output
    # For each unique value, we find all the indexes where it appears
    indexes_by_value = {
        value: np.where(arr_flat == value)[0].tolist()
        for value in unique_values
    }
    return indexes_by_value

class GMR:
    @staticmethod
    def runnalls_algorithm_simple_torch(
        means: torch.Tensor, 
        radii: torch.Tensor, 
        weights: torch.Tensor,
        L: int, DEVICE='cuda'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        PyTorch/GPU implementation of the simple, greedy Gaussian Mixture Reduction 
        algorithm, which forces merged components to be isotropic (spherical).

        The initial covariance is derived from the 'radii' tensor: P_i = radii_i^2 * I.

        Args:
            means (torch.Tensor): Initial component means (N, D).
            radii (torch.Tensor): Initial component standard deviations (N, 1).
            weights (torch.Tensor): Initial component weights (N, 1). Assumed unnormalized.
            L (int): Target number of components.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                Final means (L, D), unnormalized weights (L), and covariances (L, D, D).
        """
        
        # 0. Setup and Initial Checks
        N, D = means.shape  # N: initial components, D: dimension (assumed 3)
        
        # Move all inputs to the target device
        means = means.to(DEVICE)
        radii = radii.to(DEVICE)
        weights = weights.to(DEVICE)

        if N <= L:
            # Short-circuit: no merging needed
            radii_sq = radii.squeeze(-1) ** 2
            initial_covs = torch.diag_embed(radii_sq.unsqueeze(-1).repeat(1, D))
            return means, weights.squeeze(-1), initial_covs

        current_num = N
        
        # 1. Initialization of Parameters
        
        # Normalize weights to sum to 1 (used for mixing/merging)
        norm_weights = weights.squeeze(-1) / weights.sum() # (N,)
        
        # Calculate initial isotropic covariances P_i = r_i^2 * I
        radii_sq = radii.squeeze(-1) ** 2 # (N,)
        
        # Create a batch of diagonal DxD matrices
        covs = torch.diag_embed(radii_sq.unsqueeze(-1).repeat(1, D)) # (N, D, D)

        # Initial log-determinants: log(det(P_i)) = log((r_i^2)^D) = D * log(r_i^2)
        log_dets = D * torch.log(radii_sq) # (N,)

        # 2. Compute initial cost matrix (Fully Vectorized)
        
        # Reshape for broadcasting
        w_i = norm_weights[:, None]               # (N, 1)
        w_j = norm_weights[None, :]               # (1, N)
        mu_i = means[:, None, :]                  # (N, 1, D)
        mu_j = means[None, :, :]                  # (1, N, D)
        P_i = covs[:, None, :, :]                 # (N, 1, D, D)
        P_j = covs[None, :, :, :]                 # (1, N, D, D)

        # Merged parameters (w_m, mu_m)
        w_m = w_i + w_j                           # (N, N)
        mu_m = (w_i[..., None] * mu_i + w_j[..., None] * mu_j) / w_m[..., None] # (N, N, D)

        # Mean differences and outer products (needed to compute P_m before isotropic approx)
        dmu_i = mu_i - mu_m
        dmu_j = mu_j - mu_m
        outer_i = torch.einsum('ijk,ijl->ijkl', dmu_i, dmu_i) # (N, N, D, D)
        outer_j = torch.einsum('ijk,ijl->ijkl', dmu_j, dmu_j) # (N, N, D, D)
        
        # Merged Covariance P_m (Moment Matching - BEFORE isotropic simplification)
        P_m = (w_i[..., None, None] * (P_i + outer_i) + \
                w_j[..., None, None] * (P_j + outer_j)) / w_m[..., None, None] # (N, N, D, D)
        
        # --- Apply Isotropic Approximation (Crucial for Simple GMR) ---
        # trace(P_m) / D gives the average variance, which is put on the diagonal.
        P_m_trace = torch.einsum('...ii->...', P_m) # Trace of each matrix (N, N)
        P_m_iso_variance = P_m_trace / D
        
        # Build isotropic covariance batch: var * Identity
        I_batch = torch.eye(D, device=DEVICE).repeat(N, N, 1, 1) # (N, N, D, D)
        P_m_iso = P_m_iso_variance.view(N, N, 1, 1) * I_batch
        
        # Log-determinant of P_m_iso (using the isotropic result)
        log_det_pm = torch.logdet(P_m_iso)      # (N, N)
        
        # Log-determinants of original P_i and P_j
        log_det_i = log_dets[:, None]      # (N, 1)
        log_det_j = log_dets[None, :]      # (1, N)
        
        # Cost Matrix (Full KL-divergence form using P_m_iso)
        cost_matrix = 0.5 * (w_m * log_det_pm - w_i * log_det_i - w_j * log_det_j) # (N, N)
        
        # Fill diagonal with inf
        cost_matrix.fill_diagonal_(torch.inf)
        
        # --- Sequential Merging Loop ---
        
        # Use Python lists to manage parameters during the sequential reduction for efficiency
        norm_weights_list = norm_weights.tolist()
        means_list = means.tolist()
        covs_list = covs.tolist()
        log_dets_list = log_dets.tolist()
        
        # We also need to track the unnormalized weights separately for the final output
        unnorm_weights_list = weights.squeeze(-1).tolist()
        
        while current_num > L:
            # 1. Find the pair with the minimum cost
            old_current_num = current_num
            min_idx = torch.argmin(cost_matrix).item()
            i, j = min_idx // old_current_num, min_idx % old_current_num
            i, j = sorted([i, j])
            
            # 2. Merge components i and j (Scalar Operations for merging, Tensor for math)
            w_i_unnorm = unnorm_weights_list[i]
            w_j_unnorm = unnorm_weights_list[j]
            w_i_norm = norm_weights_list[i]
            w_j_norm = norm_weights_list[j]
            
            mu_i_val = torch.tensor(means_list[i], device=DEVICE)
            mu_j_val = torch.tensor(means_list[j], device=DEVICE)
            P_i_val = torch.tensor(covs_list[i], device=DEVICE)
            P_j_val = torch.tensor(covs_list[j], device=DEVICE)
            
            # Calculate Merged Parameters (Moment Matching)
            w_m_unnorm = w_i_unnorm + w_j_unnorm
            w_m_norm = w_i_norm + w_j_norm
            
            mu_m = (w_i_norm * mu_i_val + w_j_norm * mu_j_val) / w_m_norm
            dmu_i = mu_i_val - mu_m
            dmu_j = mu_j_val - mu_m
            outer_i = torch.ger(dmu_i, dmu_i)
            outer_j = torch.ger(dmu_j, dmu_j)
            
            P_m_full = (w_i_norm * (P_i_val + outer_i) + w_j_norm * (P_j_val + outer_j)) / w_m_norm
            
            # --- APPLY ISOTROPIC APPROXIMATION ---
            P_m = torch.eye(D, device=DEVICE) * torch.trace(P_m_full) / D
            log_det_m = torch.logdet(P_m)
            
            # 3. Component removal and list update
            
            # Remove j first, then i (to handle indices correctly)
            for lst in [unnorm_weights_list, norm_weights_list, means_list, covs_list, log_dets_list]:
                lst.pop(j)
                lst.pop(i)
            
            # Add the new merged component m to the end
            unnorm_weights_list.append(w_m_unnorm)
            norm_weights_list.append(w_m_norm)
            means_list.append(mu_m.tolist())
            covs_list.append(P_m.tolist())
            log_dets_list.append(log_det_m.item())
            
            current_num -= 1
            
            # 4. Update Cost Matrix (Vectorized Update)
            
            # Convert lists back to tensors for vectorized update
            current_means = torch.tensor(means_list, device=DEVICE)       # (C, D)
            current_norm_weights = torch.tensor(norm_weights_list, device=DEVICE)   # (C,)
            current_covs = torch.tensor(covs_list, device=DEVICE)         # (C, D, D)
            current_log_dets = torch.tensor(log_dets_list, device=DEVICE) # (C,)

            if current_num > 1:
                # The new component 'm' is always the last element (index: current_num - 1)
                
                # Extract kept components (k)
                k_indices = torch.arange(current_num - 1, device=DEVICE)
                w_k = current_norm_weights[k_indices]                        # (C-1,)
                mu_k = current_means[k_indices]                              # (C-1, D)
                P_k = current_covs[k_indices]                                # (C-1, D, D)
                log_det_k = current_log_dets[k_indices]                      # (C-1,)
                
                # The merged component m is the last one
                w_m_vec = current_norm_weights[-1]                           # Scalar
                mu_m_vec = current_means[-1]                                 # (D,)
                P_m_vec = current_covs[-1]                                   # (D, D)
                log_det_m_vec = current_log_dets[-1]                         # Scalar
                
                # Reshape for broadcast against k components
                mu_m_rs = mu_m_vec[None, :]                                  # (1, D)
                P_m_rs = P_m_vec[None, :, :]                                 # (1, D, D)
                
                # Calculate costs C_km (Fully Vectorized)
                w_mk = w_k + w_m_vec
                mu_mk = (w_k[:, None] * mu_k + w_m_vec * mu_m_rs) / w_mk[:, None]
                
                dmu_k = mu_k - mu_mk
                dmu_m = mu_m_rs - mu_mk
                
                outer_k = torch.einsum('ij,ik->ijk', dmu_k, dmu_k)   
                outer_m = torch.einsum('ij,ik->ijk', dmu_m, dmu_m)
                
                # P_mk (Moment Matching - BEFORE isotropic simplification)
                P_mk_full = (w_k[:, None, None] * (P_k + outer_k) + \
                                w_m_vec * (P_m_rs + outer_m)) / w_mk[:, None, None]
                
                # --- APPLY ISOTROPIC APPROXIMATION TO P_mk ---
                P_mk_trace = torch.einsum('...ii->...', P_mk_full) 
                P_mk_iso_variance = P_mk_trace / D
                
                I_batch_k = torch.eye(D, device=DEVICE).repeat(current_num - 1, 1, 1)
                P_mk_iso = P_mk_iso_variance.view(current_num - 1, 1, 1) * I_batch_k
                
                log_det_pmk = torch.logdet(P_mk_iso)                       # (C-1,)
                
                # Costs C_km
                costs_k_m = 0.5 * (w_mk * log_det_pmk - w_k * log_det_k - w_m_vec * log_det_m_vec) # (C-1,)

                # 1. Create the new (C x C) matrix
                new_cost_matrix = torch.full((current_num, current_num), torch.inf, device=DEVICE)
                
                # 2. Create a mask of the *kept* indices from the old matrix
                # old_current_num is the size of the old cost_matrix (e.g., C+1)
                kept_indices_mask = torch.ones(old_current_num, dtype=torch.bool, device=DEVICE)
                kept_indices_mask[i] = False
                kept_indices_mask[j] = False
                
                # 3. Select the submatrix C(k, k') using the mask
                # This selects all rows *except* i,j, and then all columns *except* i,j
                # The result is shape (C-1, C-1)
                sub_cost = cost_matrix[kept_indices_mask][:, kept_indices_mask]

                # 4. Assign the C(k, k') submatrix (which is (C-1)x(C-1))
                # new_cost_matrix[:-1, :-1] is the (C-1)x(C-1) top-left block
                new_cost_matrix[:-1, :-1] = sub_cost
                
                # 5. Fill in the costs for the new component 'm' (last row/col)
                new_cost_matrix[-1, :-1] = costs_k_m
                new_cost_matrix[:-1, -1] = costs_k_m
                
                cost_matrix = new_cost_matrix
            
        # 5. Finalize and return
        final_means = current_means
        final_weights = torch.tensor(unnorm_weights_list, device=DEVICE) # Return unnormalized weights
        final_covs = current_covs
        
        return final_means, final_weights, final_covs

    @staticmethod
    def kmeans_numpy(means: np.ndarray, sigma: float, cluster_size: int):
        N = means.shape[0]
        Pi = np.eye(3) * sigma**2

        kmeans = KMeans(n_clusters=cluster_size, random_state=42, n_init='auto')
        kmeans.fit(means)

        knn_means = np.zeros((cluster_size, 3))
        knn_weights = np.zeros((cluster_size, 1))
        knn_covs = np.zeros((cluster_size, 3, 3))
        indexes_by_value = get_indexes_by_value(kmeans.labels_)
        for i in range(knn_means.shape[0]):
            knn_weights[i] = len(indexes_by_value[i])/means.shape[0]
            knn_means[i] = np.mean(means[indexes_by_value[i]], axis=0)
            for j in indexes_by_value[i]:
                knn_covs[i] += np.outer(knn_means[i] - means[j], knn_means[i] - means[j])
            knn_covs[i] += len(indexes_by_value[i]) * Pi
            knn_covs[i] /= len(indexes_by_value[i])
        return torch.from_numpy(knn_means).float().cuda(), torch.from_numpy(knn_weights*N).float().cuda(), torch.from_numpy(knn_covs).float().cuda()