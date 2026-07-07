import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dfr.plotting import plot_density_field_3d, plot_multiscale_density_fields


def _density_case():
    density = np.zeros((3, 3, 3), dtype=float)
    density[1, 1, 1] = 1.0
    density[1, 1, 2] = 0.2
    ticks = np.array([-1.0, 0.0, 1.0])
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    return density, ticks, positions


def test_plot_density_field_3d_draws_shells_and_agents():
    density, ticks, positions = _density_case()

    fig, ax = plot_density_field_3d(
        density,
        ticks,
        ticks,
        ticks,
        positions,
        normalized_scale=0.75,
        mode_count=12,
    )

    assert fig is ax.figure
    assert len(ax.collections) >= 2
    assert len(ax.texts) == 1
    assert "0.750 x NND (12 modes)" in ax.texts[0].get_text()
    plt.close(fig)


def test_plot_multiscale_density_fields_returns_one_figure_per_scale():
    density, ticks, positions = _density_case()
    data = [
        {"density": density, "x_ticks": ticks, "y_ticks": ticks, "z_ticks": ticks},
        {"density": density * 0.5, "x_ticks": ticks, "y_ticks": ticks, "z_ticks": ticks},
    ]

    figures = plot_multiscale_density_fields(
        data,
        positions,
        normalized_scales=[0.5, 1.0],
        mode_counts=[20, 5],
    )

    assert len(figures) == 2
    assert [len(ax.texts) for _, ax in figures] == [1, 1]
    for fig, _ in figures:
        plt.close(fig)


def test_plot_density_field_3d_validates_inputs():
    density, ticks, positions = _density_case()
    with pytest.raises(ValueError, match="density_3d"):
        plot_density_field_3d(np.ones((3, 3)), ticks, ticks, ticks, positions)
    with pytest.raises(ValueError, match="tick arrays"):
        plot_density_field_3d(density, ticks[:2], ticks, ticks, positions)
    with pytest.raises(ValueError, match="positions"):
        plot_density_field_3d(density, ticks, ticks, ticks, np.ones((2, 2)))
