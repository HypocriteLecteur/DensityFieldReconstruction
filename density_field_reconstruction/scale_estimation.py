import numpy as np
from scipy.optimize import curve_fit, root_scalar
from scipy.special import lambertw

def power_law_exponential_decay(x, A, B, k, alpha):
    return A * np.exp(-k * x) + B * x ** -alpha

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

def critical_scale_detection(scales: np.ndarray, peaks_num: np.ndarray, p0=[5e2, 1e2, 0.2, 1.0]) -> tuple[float, tuple]:
    popt, pcov = curve_fit(power_law_exponential_decay, scales, peaks_num, 
                    p0=p0, sigma=peaks_num, absolute_sigma=True,
                    bounds=([0, 0, 0, 0], [np.inf, np.inf, 10, 5]))
    
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(test_scales, peaks_num, 'o', label='Raw Data', alpha=0.6)
    # y_fit = power_law_exponential_decay(test_scales, *popt)
    # plt.plot(test_scales, y_fit, '-')

    A, B, k, alpha = popt
    diff = A * np.exp(-k * scales) - B * scales ** -alpha

    smallest_positive_index = find_rightmost_positive_index(diff)

    if smallest_positive_index == -1:
        critical_scale = scales[-1]
    else:
        initial_guess = scales[smallest_positive_index]
        critical_scale = solve_transcendental_equation(initial_guess, *popt)
    
    return critical_scale, popt