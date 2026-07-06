"""CUDA smoke tests for the custom rasterizer extensions."""

import pytest
import torch


pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA is unavailable", allow_module_level=True)

try:
    from gaussian_rasterizer_simple_large import (
        GaussianRasterizerSimpleLarge,
        rasterize_gaussians,
    )
except ImportError:
    GaussianRasterizerSimpleLarge = None
    rasterize_gaussians = None

try:
    from gaussian_rasterizer_simple_small import GaussianRasterizerSimpleSmall
except ImportError:
    GaussianRasterizerSimpleSmall = None


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


def run_forward_backward_smoke(rasterizer_class, gaussian_inputs):
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


@pytest.mark.skipif(
    GaussianRasterizerSimpleSmall is None,
    reason="The small rasterizer extension is unavailable",
)
def test_small_forward_backward_smoke(gaussian_inputs):
    run_forward_backward_smoke(GaussianRasterizerSimpleSmall, gaussian_inputs)


@pytest.mark.skipif(
    GaussianRasterizerSimpleLarge is None,
    reason="The large rasterizer extension is unavailable",
)
def test_large_forward_backward_smoke(gaussian_inputs):
    run_forward_backward_smoke(GaussianRasterizerSimpleLarge, gaussian_inputs)


@pytest.mark.skipif(
    rasterize_gaussians is None,
    reason="The large rasterizer extension is unavailable",
)
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
        False,
    )

    assert density.shape == (height, width)
    assert torch.isfinite(density).all()
