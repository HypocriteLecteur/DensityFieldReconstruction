import numpy as np
import torch

from dfr.utils import compute_metrics_batched_torch, eval_isotropic_gmm_torch


def test_isotropic_gmm_value_at_its_mean():
    density = eval_isotropic_gmm_torch(
        coords=torch.zeros((1, 3)),
        means=torch.zeros((1, 3)),
        weights=torch.ones(1),
        sigmas=torch.ones(1),
    )

    expected = 1.0 / (2.0 * np.pi) ** 1.5
    assert torch.isclose(density[0], torch.tensor(expected), atol=1e-7)


def test_identical_density_fields_have_no_false_mass_on_cpu():
    means = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    pred_means = torch.from_numpy(means.copy())
    pred_weights = torch.ones((1, 1))
    pred_sigmas = torch.ones((1, 1))

    tp, fp, fn = compute_metrics_batched_torch(
        means1_np=means,
        sigma1=1.0,
        pred_means=pred_means,
        pred_weights=pred_weights,
        pred_sigmas=pred_sigmas,
        bounds=((-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)),
        voxel_res=0.5,
        batch_size=128,
        device="cpu",
    )

    assert tp > 0.9
    assert fp == 0.0
    assert fn == 0.0
