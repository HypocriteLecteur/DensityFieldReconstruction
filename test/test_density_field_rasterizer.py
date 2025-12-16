import torch
from gaussian_rasterizer_simple_small import GaussianRasterizerSimpleSmall
from gaussian_rasterizer_simple_large import GaussianRasterizerSimpleLarge, rasterize_gaussians
import matplotlib.pyplot as plt

def test_gaussian_rasterizer_simple_small():
    H = 1000
    W = 1000
    P = 30

    gs = GaussianRasterizerSimpleSmall(H=H, W=W, P_max=100)

    torch.manual_seed(12345)
    gmm_mean_contig = (torch.rand((P, 3), dtype=torch.float, device='cuda') - 5) * 20
    gmm_radius_contig = (torch.rand((P, 1), dtype=torch.float, device='cuda') + 0.5) * 2
    gmm_weights_contig = torch.rand((P, 1), dtype=torch.float, device='cuda') + 0.5
    R_contig = torch.eye(3, dtype=torch.float, device='cuda')
    T_contig = torch.zeros((3, 1), dtype=torch.float, device='cuda')
    T_contig[2] = 30
    intrinsics_contig = torch.eye(3, dtype=torch.float, device='cuda') * 1000
    intrinsics_contig[0, 2] = 500
    intrinsics_contig[1, 2] = 500
    density = torch.zeros((H, W), dtype=torch.float, device='cuda')
    profile = False

    for i in range(1):
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

def test_rasterize():
    H = 1000
    W = 1000
    P = 30

    torch.manual_seed(12345)
    gmm_mean_contig = (torch.rand((P, 3), dtype=torch.float, device='cuda') - 0.5) * 20
    gmm_radius_contig = (torch.rand((P, 1), dtype=torch.float, device='cuda') + 0.5) * 2
    gmm_weights_contig = torch.rand((P, 1), dtype=torch.float, device='cuda') + 0.5
    R_contig = torch.eye(3, dtype=torch.float, device='cuda')
    T_contig = torch.zeros((3, 1), dtype=torch.float, device='cuda')
    T_contig[2] = 30
    intrinsics_contig = torch.eye(3, dtype=torch.float, device='cuda') * 1000
    intrinsics_contig[0, 2] = 500
    intrinsics_contig[1, 2] = 500
    profile = False

    density = rasterize_gaussians(
        gmm_mean_contig,
        gmm_radius_contig,
        gmm_weights_contig,
        R_contig,
        T_contig,
        intrinsics_contig,
        H, W, 
        profile
    )

    # plt.imshow(density.detach().cpu())
    # plt.show()

def test_GaussianRasterizerSimpleLarge():
    H = 1000
    W = 1000
    P = 30

    GS = GaussianRasterizerSimpleLarge(
            H=H, W=W, P_max=100
        )
    
    torch.manual_seed(12345)
    gmm_mean_contig = (torch.rand((P, 3), dtype=torch.float, device='cuda') - 0.5) * 20
    gmm_radius_contig = (torch.rand((P, 1), dtype=torch.float, device='cuda') + 0.5) * 2
    gmm_weights_contig = torch.rand((P, 1), dtype=torch.float, device='cuda') + 0.5
    R_contig = torch.eye(3, dtype=torch.float, device='cuda')
    T_contig = torch.zeros((3, 1), dtype=torch.float, device='cuda')
    T_contig[2] = 30
    intrinsics_contig = torch.eye(3, dtype=torch.float, device='cuda') * 1000
    intrinsics_contig[0, 2] = 500
    intrinsics_contig[1, 2] = 500
    density = torch.zeros((H, W), dtype=torch.float, device='cuda')

    for i in range(5):
        grad_gmm_mean, grad_gmm_radius, grad_gmm_weights, density_estim, sum_loss = GS.rasterize_forward_backward(
            gmm_mean_contig.contiguous(), 
            gmm_radius_contig.contiguous(), 
            gmm_weights_contig.contiguous(),
            R_contig.contiguous(), 
            T_contig.contiguous(), 
            intrinsics_contig.contiguous(), 
            density, 
            profile=False
        )
    
    # plt.imshow(density_estim.detach().cpu())
    # plt.show()