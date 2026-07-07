import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from dfr.plotting import apply_academic_style, save_figure, style_3d_axis


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


def test_save_figure_creates_parent_directory(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    target = tmp_path / "nested" / "figure.png"

    returned = save_figure(fig, target, dpi=72, transparent=True)

    assert returned == target
    assert target.is_file()
    plt.close(fig)
