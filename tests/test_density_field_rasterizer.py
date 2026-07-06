"""CUDA smoke tests for the custom rasterizer extensions."""

import pytest
import torch


pytestmark = pytest.mark.cuda

try:
    from gaussian_rasterizer_simple_large import (
        GaussianRasterizerSimpleLarge,
        rasterize_gaussians,
    )
    from gaussian_rasterizer_simple_small import GaussianRasterizerSimpleSmall
except ImportError:
    pytest.skip("Compiled rasterizer extensions are unavailable", allow_module_level=True)

if not torch.cuda.is_available():
    pytest.skip("CUDA is unavailable", allow_module_level=True)


@pytest.fixture
def gaussian_inputs():
    height = width = 64
    count = 4
    torch.manual_seed(12345)
    means = (torch.rand((count, 3), device="cuda") - 0.5) * 2
    radii = torch.rand((count, 1), device="cuda") + 0.5
    weights = torch.rand((count, 1), device="cuda") + 0.5
    rotation = torch.eye(3, device="cuda")
    translation = torch.tensor([0.0, 0.0, 10.0], device="cuda")
    intrinsics = torch.tensor(
        [[64.0, 0.0, 32.0], [0.0, 64.0, 32.0], [0.0, 0.0, 1.0]],
        device="cuda",
    )
    target = torch.zeros((height, width), device="cuda")
    return height, width, means, radii, weights, rotation, translation, intrinsics, target


@pytest.mark.parametrize(
    "rasterizer_class", [GaussianRasterizerSimpleSmall, GaussianRasterizerSimpleLarge]
)
def test_forward_backward_smoke(rasterizer_class, gaussian_inputs):
    height, width, means, radii, weights, rotation, translation, intrinsics, target = (
        gaussian_inputs
    )
    rasterizer = rasterizer_class(H=height, W=width, P_max=8)

    *_, density, loss = rasterizer.rasterize_forward_backward(
        means.contiguous(),
        radii.contiguous(),
        weights.contiguous(),
        rotation.contiguous(),
        translation.contiguous(),
        intrinsics.contiguous(),
        target,
        profile=False,
    )

    assert density.shape == (height, width)
    assert torch.isfinite(density).all()
    assert torch.isfinite(loss)


def test_forward_smoke(gaussian_inputs):
    height, width, means, radii, weights, rotation, translation, intrinsics, _ = (
        gaussian_inputs
    )

    density = rasterize_gaussians(
        means,
        radii,
        weights,
        rotation,
        translation,
        intrinsics,
        height,
        width,
        profile=False,
    )

    assert density.shape == (height, width)
    assert torch.isfinite(density).all()
