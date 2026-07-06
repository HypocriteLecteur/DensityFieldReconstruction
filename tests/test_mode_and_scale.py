import numpy as np
import torch

from dfr.mode_finding import find_scale_interval, find_target_scale, mode_counting


def test_mode_counting_separates_two_compact_clusters_on_cpu():
    positions = torch.tensor(
        [[0.0, 0.0], [0.02, 0.0], [5.0, 5.0], [5.02, 5.0]],
        dtype=torch.float32,
    )

    count = mode_counting(
        positions,
        positions.clone(),
        scale=0.2,
        max_iter=100,
        tol=1e-5,
    )

    assert count == 2


def test_scale_search_characterizes_monotonic_step_curve():
    def mode_curve(scale):
        if scale < 1.0:
            return 100
        if scale < 2.0:
            return 50
        return 1

    target = find_target_scale(mode_curve, targetd_num_mode=50, s_low=0, s_high=4)
    start, end = find_scale_interval(mode_curve, N=100, s_initial_guess=4)

    assert 1.0 <= target < 2.0
    assert np.isclose(start, 1.0, atol=1e-3)
    assert np.isclose(end, 2.0, atol=1e-3)
