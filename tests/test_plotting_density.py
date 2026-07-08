import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dfr.plotting import (
    plot_density_field_3d,
    plot_multiscale_density_fields,
    render_gmm_means,
    render_gmm_wireframes,
    render_reconstructed_gmm_3d,
)


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


def test_gmm_renderers_draw_wireframes_and_means():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    means = np.array([[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]])
    sigmas = np.array([0.25, 0.5])
    weights = np.array([1.0, 0.5])

    wireframes = render_gmm_wireframes(
        ax,
        means,
        sigmas,
        weights,
        sphere_res=6,
    )
    means_collection = render_gmm_means(ax, means)

    assert len(wireframes) == 2
    assert means_collection in ax.collections
    plt.close(fig)


def test_render_reconstructed_gmm_3d_combines_density_and_gmm_layers():
    density, ticks, positions = _density_case()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    render_reconstructed_gmm_3d(
        ax,
        density,
        ticks,
        ticks,
        ticks,
        positions,
        means=np.array([[0.0, 0.0, 0.0]]),
        sigmas=np.array([0.4]),
        weights=np.array([1.0]),
    )

    assert len(ax.collections) >= 3
    plt.close(fig)


def test_plot_density_field_3d_validates_inputs():
    density, ticks, positions = _density_case()
    with pytest.raises(ValueError, match="density_3d"):
        plot_density_field_3d(np.ones((3, 3)), ticks, ticks, ticks, positions)
    with pytest.raises(ValueError, match="tick arrays"):
        plot_density_field_3d(density, ticks[:2], ticks, ticks, positions)
    with pytest.raises(ValueError, match="positions"):
        plot_density_field_3d(density, ticks, ticks, ticks, np.ones((2, 2)))
    with pytest.raises(ValueError, match="sigmas"):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        try:
            render_gmm_wireframes(
                ax,
                np.zeros((2, 3)),
                np.ones(1),
                np.ones(2),
            )
        finally:
            plt.close(fig)
