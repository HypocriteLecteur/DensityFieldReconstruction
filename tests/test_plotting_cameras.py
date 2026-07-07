import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dfr.plotting import plot_camera_configurations


def test_plot_camera_configurations_returns_figure_without_saving():
    positions = np.array(
        [[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [0.5, -0.5, 0.25]],
        dtype=np.float32,
    )
    cameras = {
        2: np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]], dtype=np.float32),
        3: np.array(
            [[5.0, 0.0, 0.0], [-2.5, 4.3, 0.0], [-2.5, -4.3, 0.0]],
            dtype=np.float32,
        ),
    }

    fig, ax = plot_camera_configurations(
        positions,
        cameras,
        center=np.array([0.0, 0.0, 0.0]),
        swarm_radius=1.5,
        orbit_radius=5.0,
    )

    assert fig is ax.figure
    assert ax.get_aspect() == 1.0
    assert len(ax.collections) >= 3
    assert ax.get_legend() is not None
    plt.close(fig)


def test_plot_camera_configurations_accepts_existing_axes():
    fig, ax = plt.subplots()

    returned_fig, returned_ax = plot_camera_configurations(
        np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        {2: np.array([[3.0, 0.0], [-3.0, 0.0]], dtype=np.float32)},
        ax=ax,
        apply_style=False,
    )

    assert returned_fig is fig
    assert returned_ax is ax
    plt.close(fig)


def test_plot_camera_configurations_validates_inputs():
    with pytest.raises(ValueError, match="positions"):
        plot_camera_configurations(np.zeros((2, 4)), {2: np.zeros((2, 2))})
    with pytest.raises(ValueError, match="camera_positions"):
        plot_camera_configurations(np.zeros((2, 2)), {})
    with pytest.raises(ValueError, match="orbit_radius"):
        plot_camera_configurations(
            np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            {2: np.ones((2, 2))},
            orbit_radius=0,
        )
