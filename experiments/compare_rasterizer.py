import torch
from gaussian_rasterizer_simple_small import GaussianRasterizerSimpleSmall
from gaussian_rasterizer_simple_large import GaussianRasterizerSimpleLarge, rasterize_gaussians
import time

def main():
    H = 1000
    W = 1000
    P = 50

    gs = GaussianRasterizerSimpleSmall(H=H, W=W, P_max=200)
    GS = GaussianRasterizerSimpleLarge(H=H, W=W, P_max=200)

    torch.manual_seed(12345)
    gmm_mean_contig = ((torch.rand((P, 3), dtype=torch.float, device='cuda') - 0.5) * 20).contiguous()
    gmm_radius_contig = ((torch.rand((P, 1), dtype=torch.float, device='cuda') + 0.5) * 2).contiguous()
    gmm_weights_contig = (torch.rand((P, 1), dtype=torch.float, device='cuda') + 0.5).contiguous()
    R_contig = torch.eye(3, dtype=torch.float, device='cuda').contiguous()
    T_contig = torch.zeros((3, 1), dtype=torch.float, device='cuda')
    T_contig[2] = 30
    T_contig = T_contig.contiguous()
    intrinsics_contig = torch.eye(3, dtype=torch.float, device='cuda') * 1000
    intrinsics_contig[0, 2] = 500
    intrinsics_contig[1, 2] = 500
    intrinsics_contig = intrinsics_contig.contiguous()
    density = torch.zeros((H, W), dtype=torch.float, device='cuda')
    profile = False

    total_run = 50

    # warm-up run
    grad_gmm_mean, grad_gmm_radius, grad_gmm_weights, density_estim, sum_loss = \
            gs.rasterize_forward_backward(
            gmm_mean_contig,
            gmm_radius_contig,
            gmm_weights_contig,
            R_contig,
            T_contig,
            intrinsics_contig,
            density,
            profile
        )
    start = time.perf_counter()
    for i in range(total_run):
        grad_gmm_mean, grad_gmm_radius, grad_gmm_weights, density_estim, sum_loss = \
            gs.rasterize_forward_backward(
            gmm_mean_contig,
            gmm_radius_contig,
            gmm_weights_contig,
            R_contig,
            T_contig,
            intrinsics_contig,
            density,
            profile
        )
    end = time.perf_counter()

    # warm-up run
    grad_gmm_mean, grad_gmm_radius, grad_gmm_weights, density_estim, sum_loss = GS.rasterize_forward_backward(
        gmm_mean_contig, 
        gmm_radius_contig, 
        gmm_weights_contig,
        R_contig, 
        T_contig, 
        intrinsics_contig, 
        density, 
        profile
    )
    start2 = time.perf_counter()
    for i in range(total_run):
        grad_gmm_mean, grad_gmm_radius, grad_gmm_weights, density_estim, sum_loss = GS.rasterize_forward_backward(
            gmm_mean_contig, 
            gmm_radius_contig, 
            gmm_weights_contig,
            R_contig, 
            T_contig, 
            intrinsics_contig, 
            density, 
            profile
        )
    end2 = time.perf_counter()

    print(f"small avg {((end-start)*1e6) / total_run} us")
    print(f"large avg {((end2-start2)*1e6) / total_run} us")

if __name__ == "__main__":
    main()