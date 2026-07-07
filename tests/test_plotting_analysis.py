import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dfr.analysis import select_adaptive_density_scales, validate_nnd_bounds
from dfr.plotting import (
    plot_dra_scale_model_order_surface,
    plot_dra_surface_grid,
    plot_mode_count_curve,
)


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


def test_plot_dra_scale_model_order_surface_draws_surface_and_fit():
    scales = np.array([0.5, 1.0, 1.5])
    components = np.array([4, 8])
    dra = np.array([[0.9, 0.8], [0.7, 0.5], [0.6, 0.4]])
    prediction = dra * 0.95

    fig, ax, surface = plot_dra_scale_model_order_surface(
        scales,
        components,
        dra,
        number_of_animals=16,
        fitted_dra=prediction,
        wireframe_label="Fitted surface",
        max_model_order_ticks=2,
    )

    assert fig is ax.figure
    assert surface in ax.collections
    assert ax.get_xlabel() == r"Normalized scale ($\sigma / \mathrm{NND}$)"
    assert ax.get_ylabel() == "Model order / N (%)"
    assert ax.get_zlabel() == "DRA"
    assert len(ax.collections) >= 2
    plt.close(fig)


def test_plot_dra_surface_grid_returns_one_axis_per_result():
    result = (
        np.array([0.5, 1.0, 1.5]),
        np.array([0.5, 1.0, 1.5]),
        np.array([4, 8]),
        np.array([[0.9, 0.8], [0.7, 0.5], [0.6, 0.4]]),
        0.25,
        16,
    )
    fit = {
        "best_name": "linear",
        "candidates": {
            "linear": {
                "prediction": result[3] * 0.95,
                "r_squared": 0.987,
            }
        },
    }

    fig, axes = plot_dra_surface_grid({"jackdaw2": result}, {"jackdaw2": fit})

    assert len(axes) == 1
    assert axes[0].get_title() == "Jackdaw2 (linear, $R^2$=0.987)"
    assert len(fig.axes) == 2  # one 3D axis plus the colorbar axis
    plt.close(fig)


def test_plot_dra_surface_validates_inputs():
    with pytest.raises(ValueError, match="dra must have shape"):
        plot_dra_scale_model_order_surface(
            [0.5, 1.0],
            [4, 8],
            np.ones((2, 3)),
            number_of_animals=16,
        )
    with pytest.raises(ValueError, match="fitted_dra"):
        plot_dra_scale_model_order_surface(
            [0.5, 1.0],
            [4, 8],
            np.ones((2, 2)),
            number_of_animals=16,
            fitted_dra=np.ones((2, 1)),
        )
