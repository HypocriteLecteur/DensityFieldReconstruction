import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt

from experiments import plot_publication_table2 as table2


def test_table2_capacity_plot_returns_figure_axes_without_saving():
    fig, ax = table2.plot_capacity_scaling()

    assert fig is ax.figure
    assert ax.get_xlabel() == "Number of Cameras"
    assert ax.get_ylabel() == "DEA"
    assert len(ax.lines) == len(table2.DATASETS) * 2
    assert ax.get_legend() is not None
    plt.close(fig)


def test_table2_tradeoff_plot_returns_figure_axes_without_saving():
    fig, ax = table2.plot_recall_hallucination_tradeoff()

    assert fig is ax.figure
    assert ax.get_xlabel() == "Hallucination"
    assert ax.get_ylabel() == "Recall"
    assert len(ax.collections) == len(table2.DATASETS) * len(table2.ALL_METHODS)
    assert len(ax.lines) == 4
    assert ax.get_legend() is not None
    plt.close(fig)


def test_table2_save_helper_writes_requested_formats(tmp_path):
    saved = table2.save_table2_figures(tmp_path, formats=("png",))

    assert saved == [
        tmp_path / "table2_dea_capacity_scaling.png",
        tmp_path / "table2_recall_hallu_tradeoff.png",
    ]
    assert all(path.is_file() for path in saved)


def test_table2_named_script_has_no_legacy_plot_archive_dependency():
    source = Path(table2.__file__).read_text(encoding="utf-8")

    assert "dfr_plot" not in source
