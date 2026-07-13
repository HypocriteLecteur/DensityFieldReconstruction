import argparse
from pathlib import Path

import numpy as np

from dfr.analysis import add_managed_output_arguments
from experiments.fit_dra_multiframe import (
    MULTIFRAME_NORMALIZED_SCALES,
    PREFERRED_FRAMES,
    seed_existing_cache,
)


SUPPORTED_ANALYSIS_SCRIPTS = (
    "plot_dra_scale_model_order.py",
    "fit_dra_multiframe.py",
    "parameter_manifold.py",
    "parameter_manifold_2pl.py",
    "mechanistic_derivation.py",
    "synthetic_benchmark.py",
    "validate_mode_counting.py",
)


def test_supported_analysis_scripts_avoid_legacy_output_roots():
    experiments = Path(__file__).resolve().parents[1] / "experiments"
    for filename in SUPPORTED_ANALYSIS_SCRIPTS:
        source = (experiments / filename).read_text(encoding="utf-8")
        assert '"figs/' not in source, filename
        assert '"results/' not in source, filename
        assert "os.getcwd()" not in source, filename


def test_dra_scale_model_order_cli_uses_package_plotting():
    experiments = Path(__file__).resolve().parents[1] / "experiments"
    source = (experiments / "plot_dra_scale_model_order.py").read_text(
        encoding="utf-8"
    )
    plotting_region = source.split("def plot_surfaces", 1)[1].split(
        "def parse_args", 1
    )[0]

    assert "from dfr.plotting import plot_dra_surface_grid" in source
    assert "save_figure(figure, output_path" in plotting_region
    assert "plot_dra_surface_grid(results, fits)" in plotting_region
    assert "plot_surface(" not in plotting_region
    assert "plot_wireframe(" not in plotting_region


def test_supported_analysis_figures_use_package_save_helper():
    experiments = Path(__file__).resolve().parents[1] / "experiments"
    for filename in SUPPORTED_ANALYSIS_SCRIPTS:
        source = (experiments / filename).read_text(encoding="utf-8")
        assert ".savefig(" not in source, filename
        assert "plt.savefig(" not in source, filename

    for filename in (
        "plot_dra_scale_model_order.py",
        "fit_dra_multiframe.py",
        "parameter_manifold.py",
        "parameter_manifold_2pl.py",
        "mechanistic_derivation.py",
        "validate_mode_counting.py",
    ):
        source = (experiments / filename).read_text(encoding="utf-8")
        assert "save_figure" in source, filename


def test_supported_analysis_figures_use_package_layout_helpers():
    experiments = Path(__file__).resolve().parents[1] / "experiments"
    for filename in (
        "fit_dra_multiframe.py",
        "parameter_manifold.py",
        "parameter_manifold_2pl.py",
        "mechanistic_derivation.py",
        "validate_mode_counting.py",
    ):
        source = (experiments / filename).read_text(encoding="utf-8")
        assert ".tight_layout(" not in source, filename
        assert ".subplots_adjust(" not in source, filename
        assert "apply_figure_layout" in source, filename


def test_supported_analysis_styles_use_package_helper():
    experiments = Path(__file__).resolve().parents[1] / "experiments"
    for filename in ("parameter_manifold.py", "parameter_manifold_2pl.py"):
        source = (experiments / filename).read_text(encoding="utf-8")
        assert "apply_academic_style(" in source, filename
        assert "plt.rcParams.update" not in source, filename


def test_analysis_catalog_documents_every_supported_entrypoint():
    experiments = Path(__file__).resolve().parents[1] / "experiments"
    catalog = (experiments / "README.md").read_text(encoding="utf-8")
    for filename in SUPPORTED_ANALYSIS_SCRIPTS:
        assert filename.removesuffix(".py") in catalog


def test_parameter_manifold_scripts_do_not_fall_back_to_legacy_figures():
    experiments = Path(__file__).resolve().parents[1] / "experiments"
    for filename in ("parameter_manifold.py", "parameter_manifold_2pl.py"):
        source = (experiments / filename).read_text(encoding="utf-8")

        assert '_FIGURE_DIR: Path | None = None' in source
        assert "def _figure_path(" in source
        assert "_FIGURE_DIR = artifacts.figures_dir" in source
        assert 'Path("figs")' not in source


def test_multiframe_legacy_cache_requires_an_explicit_root(tmp_path):
    legacy_root = tmp_path / "historical-cache"
    source = legacy_root / "dra_scale_model_order" / "jackdaw2_dra_scale_model_order.npz"
    source.parent.mkdir(parents=True)
    np.savez(source, normalized_scales=MULTIFRAME_NORMALIZED_SCALES)
    frame_dir = tmp_path / "managed" / "frame"
    frame_dir.mkdir(parents=True)
    destination = frame_dir / source.name

    seed_existing_cache(
        "jackdaw2",
        PREFERRED_FRAMES["jackdaw2"],
        frame_dir,
        False,
        MULTIFRAME_NORMALIZED_SCALES,
    )
    assert not destination.exists()

    seed_existing_cache(
        "jackdaw2",
        PREFERRED_FRAMES["jackdaw2"],
        frame_dir,
        False,
        MULTIFRAME_NORMALIZED_SCALES,
        legacy_root,
    )
    assert destination.is_file()


def test_shared_analysis_cli_arguments_enforce_collision_policy():
    parser = argparse.ArgumentParser()
    add_managed_output_arguments(parser)

    args = parser.parse_args(["--run-id", "demo", "--resume"])

    assert args.run_id == "demo" and args.resume and not args.overwrite_run
