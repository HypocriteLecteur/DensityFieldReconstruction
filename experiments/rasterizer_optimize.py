import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from gaussian_rasterizer_simple_small import GaussianRasterizerSimpleSmall
from gaussian_rasterizer_simple_large import GaussianRasterizerSimpleLarge

def benchmark_2d_sweep(RasterizerClass, P_values, radius_values, num_trials=100, H=1000, W=1000, P_max=100):
    """
    Benchmarks the rasterizer across a grid of P_values and radius_values.
    Returns a dictionary mapping radius -> (means_array, stds_array).
    """
    results = {}
    rasterizer = RasterizerClass(H=H, W=W, P_max=P_max)
    
    for radius in radius_values:
        means = []
        stds = []
        
        for P in P_values:
            # 1. Setup Data for current P and radius
            torch.manual_seed(12345)
            gmm_mean = (torch.rand((P, 3), dtype=torch.float, device='cuda') - 0.5) * 20
            gmm_radius = torch.full((P, 1), float(radius), dtype=torch.float, device='cuda')
            gmm_weights = torch.rand((P, 1), dtype=torch.float, device='cuda') + 0.5
            R = torch.eye(3, dtype=torch.float, device='cuda')
            T = torch.zeros((3, 1), dtype=torch.float, device='cuda')
            T[2] = 50
            intrinsics = torch.eye(3, dtype=torch.float, device='cuda') * 1000
            intrinsics[0, 2] = 500
            intrinsics[1, 2] = 500
            density = torch.zeros((H, W), dtype=torch.float, device='cuda')
            
            # Ensure contiguous memory layout
            gmm_mean = gmm_mean.contiguous()
            gmm_radius = gmm_radius.contiguous()
            gmm_weights = gmm_weights.contiguous()
            R = R.contiguous()
            T = T.contiguous()
            intrinsics = intrinsics.contiguous()
            
            # 2. Warm-up (Critical for changing tensor sizes)
            for _ in range(1):
                rasterizer.rasterize_forward_backward(
                    gmm_mean, gmm_radius, gmm_weights,
                    R, T, intrinsics, density, profile=False
                )
            torch.cuda.synchronize()
            
            # 3. Benchmark Loop
            trial_times = []
            for _ in range(num_trials):
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                rasterizer.rasterize_forward_backward(
                    gmm_mean, gmm_radius, gmm_weights,
                    R, T, intrinsics, density, profile=False
                )
                
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                trial_times.append(end_time - start_time)
                
            means.append(np.mean(trial_times) * 1e3)
            stds.append(np.std(trial_times) * 1e3)
            
        results[radius] = (np.array(means), np.array(stds))
        
    return results

def main():
    # Setup test grid
    P_values = list(range(10, 101, 10))  # 10 to 100 with interval 10
    radius_values = [1.0, 2.0]  # Subset of radii to keep the graph readable
    
    num_trials = 100
    H, W = 1000, 1000
    P_max = 500
    
    print("Benchmarking GaussianRasterizerSimpleSmall...")
    results_small = benchmark_2d_sweep(
        GaussianRasterizerSimpleSmall, P_values, radius_values, num_trials, H, W, P_max
    )
    
    print("Benchmarking GaussianRasterizerSimpleLarge...")
    results_large = benchmark_2d_sweep(
        GaussianRasterizerSimpleLarge, P_values, radius_values, num_trials, H, W, P_max
    )
    
    # Plotting: Create a 2x2 grid of subplots for the 4 radius values
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()  # Flatten to iterate easily
    
    for idx, radius in enumerate(radius_values):
        ax = axes[idx]
        
        # Get data for current radius
        mean_small, std_small = results_small[radius]
        mean_large, std_large = results_large[radius]
        
        # Plot Small
        ax.plot(P_values, mean_small, label='Simple Small', color='blue', marker='o')
        ax.fill_between(P_values, mean_small - std_small, mean_small + std_small, color='blue', alpha=0.2)
        
        # Plot Large
        ax.plot(P_values, mean_large, label='Simple Large', color='red', marker='s')
        ax.fill_between(P_values, mean_large - std_large, mean_large + std_large, color='red', alpha=0.2)
        
        # Subplot formatting
        ax.set_title(f'Gaussian Radius = {radius}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Gaussians (P)', fontsize=10)
        ax.set_ylabel('Processing Time (ms)', fontsize=10)
        ax.set_xticks(P_values)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Global formatting
    plt.suptitle('Performance Comparison: Small vs. Large Rasterizer by Radius Size', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save or show the plot
    plt.savefig('rasterizer_comparison_by_radius.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()