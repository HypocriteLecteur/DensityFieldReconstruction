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


def test_noise_named_script_has_no_legacy_plot_archive_dependency():
    source = Path(noise.__file__).read_text(encoding="utf-8")

    assert "dfr_plot" not in source
