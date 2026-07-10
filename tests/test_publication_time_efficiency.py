import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt

from experiments import plot_publication_time_efficiency as time_efficiency


def test_time_efficiency_plot_returns_figure_axes_without_saving():
    fig, ax = time_efficiency.plot_time_efficiency()

    assert fig is ax.figure
    assert ax.get_xlabel() == "Training Iterations"
    assert ax.get_ylabel() == "Training Time (msec)"
    assert len(ax.collections) == len(time_efficiency.DATASETS) * len(time_efficiency.METHODS)
    assert len(ax.lines) == 1
    assert ax.get_legend() is not None
    plt.close(fig)


def test_time_efficiency_save_helper_writes_requested_formats(tmp_path):
    saved = time_efficiency.save_time_efficiency_figure(tmp_path, formats=("png",))

    assert saved == [tmp_path / "table_dra_vs_iters.png"]
    assert saved[0].is_file()


def test_time_efficiency_named_script_has_no_legacy_plot_archive_dependency():
    source = Path(time_efficiency.__file__).read_text(encoding="utf-8")

    assert "dfr_plot" not in source
