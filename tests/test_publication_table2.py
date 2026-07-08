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


def test_table2_legacy_wrapper_delegates_to_named_script():
    root = Path(__file__).resolve().parents[1]
    source = (root / "experiments" / "dfr_plot.py").read_text(encoding="utf-8")
    wrapper = source.split("def plot_table_2_results", 1)[1].split(
        "def plot_table_time_efficiency", 1
    )[0]
    active_wrapper = wrapper.split("return fig1, fig2", 1)[0]

    assert "plot_publication_table2" in active_wrapper
    assert "plot_capacity_scaling()" in active_wrapper
    assert "plot_recall_hallucination_tradeoff()" in active_wrapper
    assert "save_dir is not None" in active_wrapper
    assert "save_figure(" in active_wrapper
    assert "if show:" in active_wrapper
    assert 'os.path.join(os.getcwd(), "figs")' not in active_wrapper
