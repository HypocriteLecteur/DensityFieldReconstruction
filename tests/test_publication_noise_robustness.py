import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt

from experiments import plot_publication_noise_robustness as noise


def test_noise_robustness_plot_returns_figure_axes_without_saving():
    fig, ax = noise.plot_noise_robustness()

    assert fig is ax.figure
    assert "Normalized Noise" in ax.get_xlabel()
    assert ax.get_ylabel() == r"DEA degradation $\%$"
    assert len(ax.collections) == len(noise.DATASETS) * len(noise.CAMERA_LABELS)
    assert len(ax.lines) == len(noise.DATASETS) * len(noise.CAMERA_LABELS) + 2
    assert ax.get_legend() is not None
    plt.close(fig)


def test_noise_robustness_save_helper_writes_requested_formats(tmp_path):
    saved = noise.save_noise_robustness_figure(tmp_path, formats=("png",))

    assert saved == [tmp_path / "table_noise_robustness.png"]
    assert saved[0].is_file()


def test_noise_robustness_legacy_wrapper_delegates_to_named_script():
    root = Path(__file__).resolve().parents[1]
    source = (root / "experiments" / "dfr_plot.py").read_text(encoding="utf-8")
    wrapper = source.split("def plot_table_noise_robustness", 1)[1].split(
        'if __name__ == "__main__"', 1
    )[0]

    assert "plot_publication_noise_robustness" in wrapper
    assert "plot_noise_robustness()" in wrapper
    assert "save_figure(" in wrapper
