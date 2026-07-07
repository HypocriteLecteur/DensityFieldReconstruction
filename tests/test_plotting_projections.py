import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dfr.plotting import (
    plot_density_image,
    plot_projected_gmm_density,
    plot_projection_points,
    transparent_colormap,
)


def test_plot_projection_points_returns_image_axes():
    points = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    fig, ax = plot_projection_points(points, image_shape=(10, 20))

    assert fig is ax.figure
    assert len(ax.collections) == 1
    assert ax.get_xlim() == (0.0, 20.0)
    assert ax.get_ylim() == (10.0, 0.0)
    plt.close(fig)


def test_plot_density_image_adds_contours():
    yy, xx = np.mgrid[:16, :20]
    density = np.exp(-((xx - 10) ** 2 + (yy - 8) ** 2) / 12.0)

    fig, ax = plot_density_image(
        density,
        density_cutoff=1e-3,
        num_levels=5,
        cmap=transparent_colormap(),
    )

    assert fig is ax.figure
    assert len(ax.collections) >= 1
    assert ax.get_xlim() == (0.0, 19.0)
    assert ax.get_ylim() == (15.0, 0.0)
    plt.close(fig)


def test_plot_projected_gmm_density_draws_ellipses():
    density = np.ones((12, 14), dtype=np.float32)
    means = np.array([[5.0, 6.0], [8.0, 3.0]], dtype=np.float32)
    covariances = np.array(
        [
            [[4.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.2], [0.2, 2.0]],
        ],
        dtype=np.float32,
    )

    fig, ax = plot_projected_gmm_density(
        density,
        means,
        covariances,
        weights=np.array([1.0, 0.5], dtype=np.float32),
    )

    assert len(ax.patches) == 2
    plt.close(fig)


def test_projection_plotters_validate_inputs():
    with pytest.raises(ValueError, match="2D points"):
        plot_projection_points(np.zeros((2, 3)), image_shape=(10, 10))
    with pytest.raises(ValueError, match="density"):
        plot_density_image(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="covariances"):
        plot_projected_gmm_density(
            np.ones((4, 4)),
            np.zeros((2, 2)),
            np.zeros((1, 2, 2)),
            np.ones(2),
        )
