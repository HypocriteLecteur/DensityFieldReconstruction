import torch
import numpy as np

def find_local_peaks_simple(tensor, maxima, threshold=1):
    device = maxima.device
    
    # Get indices and values of all maxima
    all_maxima_indices = torch.nonzero(maxima, as_tuple=False) # (N, 2) (y, x)
    all_maxima_values = tensor[all_maxima_indices[:, 0], all_maxima_indices[:, 1]] # (N,)

    n = all_maxima_indices.shape[0]
    if n == 0:
        return torch.empty((0, 2), device=device, dtype=torch.long), torch.empty((0, 1), device=device, dtype=torch.float32)
    return all_maxima_indices[:, [1, 0]], all_maxima_values # Switch to (x, y)

def compute_distances(points_h, lines, norms):
    """Compute perpendicular distances from points to epipolar lines."""
    D = np.abs(points_h @ lines)  # Matrix multiplication: (m, n)
    return D / norms  # Normalize distances

# @jit(nopython=True)
def find_mutual_matches(j_stars, i_stars, D_l2r, D_r2l, threshold):
    """Find mutual matches with optional thresholding."""
    matches = []
    n = j_stars.shape[0]
    for i in range(n):
        j = j_stars[i]
        if i_stars[j] == i:  # Check if mutual nearest neighbors
            if threshold is None or (D_l2r[j, i] < threshold and D_r2l[i, j] < threshold):
                matches.append((i, j))
    return matches

def match_points(points_left, points_right, F, threshold=None):
    """
    Match points between two images using the fundamental matrix.
    
    Args:
        points_left (np.ndarray): (n, 2) points in the left image.
        points_right (np.ndarray): (m, 2) points in the right image.
        F (np.ndarray): (3, 3) fundamental matrix.
        threshold (float, optional): Max distance for valid matches.
    
    Returns:
        list of tuples: Matched point indices (i, j).
    """
    # Convert to homogeneous coordinates
    points_left_h = np.hstack((points_left, np.ones((points_left.shape[0], 1))))
    points_right_h = np.hstack((points_right, np.ones((points_right.shape[0], 1))))
    
    # Compute epipolar lines
    l_primes = F @ points_left_h.T  # Lines in right image: (3, n)
    norms_left = np.sqrt(l_primes[0, :]**2 + l_primes[1, :]**2)
    l_s = F.T @ points_right_h.T    # Lines in left image: (3, m)
    norms_right = np.sqrt(l_s[0, :]**2 + l_s[1, :]**2)
    
    # Compute distance matrices
    D_l2r = compute_distances(points_right_h, l_primes, norms_left)  # (m, n)
    D_r2l = compute_distances(points_left_h, l_s, norms_right)      # (n, m)
    
    # Find nearest neighbors
    j_stars = np.argmin(D_l2r, axis=0)  # Closest right points for left points
    i_stars = np.argmin(D_r2l, axis=0)  # Closest left points for right points
    
    # Find mutual matches
    matches = find_mutual_matches(j_stars, i_stars, D_l2r, D_r2l, threshold)
    
    return matches