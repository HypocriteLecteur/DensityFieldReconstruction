import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from dfr.plotting import (
    apply_academic_style,
    apply_figure_layout,
    prepare_3d_axis,
    save_figure,
    set_3d_view,
    style_3d_axis,
)


def test_apply_academic_style_accepts_overrides():
    original_size = plt.rcParams["font.size"]
    try:
        apply_academic_style({"font.size": 17})

        assert plt.rcParams["font.family"][0] == "serif"
        assert plt.rcParams["font.size"] == 17
    finally:
        plt.rcParams["font.size"] = original_size


def test_style_3d_axis_sets_transparent_panes():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    style_3d_axis(ax)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        assert not axis.pane.fill
        assert axis._axinfo["grid"]["linewidth"] == 0.5
    plt.close(fig)


def test_prepare_3d_axis_sets_view_and_axis_visibility():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    prepare_3d_axis(ax, view=(22, -45, 0), axis_off=True)

    assert not ax.axison
    assert ax.elev == 22
    assert ax.azim == -45
    plt.close(fig)


def test_set_3d_view_accepts_two_value_views():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    set_3d_view(ax, (12, 34))

    assert ax.elev == 12
    assert ax.azim == 34
    plt.close(fig)


def test_apply_figure_layout_accepts_tight_and_adjust_modes():
    fig, ax = plt.subplots()

    apply_figure_layout(fig, pad=0.5)
    apply_figure_layout(fig, adjust={"left": 0.2, "right": 0.8})

    assert fig.subplotpars.left == 0.2
    assert fig.subplotpars.right == 0.8
    plt.close(fig)


def test_save_figure_creates_parent_directory(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    target = tmp_path / "nested" / "figure.png"

    returned = save_figure(fig, target, dpi=72, transparent=True)

    assert returned == target
    assert target.is_file()
    plt.close(fig)
