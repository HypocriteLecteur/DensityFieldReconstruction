import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dfr.plotting import plot_trajectory_snapshot


def test_plot_trajectory_snapshot_returns_figure_without_saving():
    trajectories = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.5, 0.5, 0.0], [1.5, 0.5, 0.0]],
            [[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    positions = trajectories[-1]

    fig, ax = plot_trajectory_snapshot(trajectories, positions)

    assert fig is ax.figure
    assert len(ax.lines) == 2
    assert len(ax.collections) == 1
    assert not ax.axison
    plt.close(fig)


def test_plot_trajectory_snapshot_accepts_existing_axes():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    returned_fig, returned_ax = plot_trajectory_snapshot(
        np.zeros((2, 1, 3), dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
        ax=ax,
        view=None,
        axis_off=False,
    )

    assert returned_fig is fig
    assert returned_ax is ax
    assert len(ax.lines) == 1
    assert len(ax.collections) == 1
    plt.close(fig)


def test_plot_trajectory_snapshot_validates_shapes():
    with pytest.raises(ValueError, match="trajectories"):
        plot_trajectory_snapshot(np.zeros((2, 3)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="positions"):
        plot_trajectory_snapshot(np.zeros((2, 3, 3)), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="one point per trajectory"):
        plot_trajectory_snapshot(np.zeros((2, 3, 3)), np.zeros((2, 3)))
