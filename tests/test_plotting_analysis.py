import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dfr.analysis import select_adaptive_density_scales, validate_nnd_bounds
from dfr.plotting import plot_mode_count_curve


def test_validate_nnd_bounds_accepts_positive_interval():
    assert validate_nnd_bounds((0.5, 1.5)) == (0.5, 1.5)


def test_select_adaptive_density_scales_uses_interior_samples():
    scales = np.geomspace(0.5, 1.5, 8)
    counts = np.array([80, 70, 55, 40, 20, 10, 5, 3])

    indices, selected = select_adaptive_density_scales(
        scales,
        counts,
        n_selected=3,
    )

    assert np.all(indices > 0)
    assert np.all(indices < len(scales) - 1)
    assert np.allclose(selected, scales[indices])
    assert len(indices) == 3


def test_plot_mode_count_curve_returns_figure_and_marks_slices():
    scales = np.geomspace(0.5, 1.5, 9)
    counts = np.array([90, 80, 65, 45, 32, 20, 12, 8, 5])

    fig, ax = plot_mode_count_curve(
        scales,
        counts,
        dataset_name="jackdaw2",
        frame=2800,
        number_of_agents=90,
        n_slices=4,
    )

    assert fig is ax.figure
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    assert ax.get_xlabel() == r"Normalized scale ($\sigma / \mathrm{NND}$)"
    # Main curve plus four selected slice markers and the NND marker.
    assert len(ax.lines) == 6
    plt.close(fig)


def test_plot_mode_count_curve_validates_inputs():
    with pytest.raises(ValueError, match="strictly increasing"):
        plot_mode_count_curve([1.0, 1.0, 2.0], [3, 2, 1], n_slices=1)
    with pytest.raises(ValueError, match="at least one"):
        plot_mode_count_curve([1.0, 2.0, 3.0], [3, 0, 1], n_slices=1)
    with pytest.raises(ValueError, match="nnd_bounds"):
        plot_mode_count_curve([1.0, 2.0, 3.0], [3, 2, 1], nnd_bounds=(2, 1))
