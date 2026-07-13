import numpy as np
import torch

from dfr.evaluation import (
    build_isotropic_density_grid,
    sample_isotropic_density_grid,
)


def test_isotropic_density_grid_samples_on_cpu():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    grid = build_isotropic_density_grid(
        positions,
        0.5,
        voxel_res_fraction=0.5,
        device="cpu",
    )

    density = sample_isotropic_density_grid(
        positions, 0.5, grid, batch_size=17, device="cpu"
    )

    assert density.device.type == "cpu"
    assert density.dtype == torch.float32
    assert density.numel() == grid["total_voxels"]
    assert torch.all(density >= 0)
    assert torch.any(density > 0)


def test_isotropic_density_grid_rejects_degenerate_positions():
    positions = np.zeros((2, 3), dtype=np.float32)

    try:
        build_isotropic_density_grid(positions, 0.5, device="cpu")
    except ValueError as error:
        assert "non-zero spatial extent" in str(error)
    else:
        raise AssertionError("degenerate positions must be rejected")
