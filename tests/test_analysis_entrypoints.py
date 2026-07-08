import argparse
from pathlib import Path

from dfr.analysis import add_managed_output_arguments


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


def test_supported_analysis_styles_use_package_helper():
    experiments = Path(__file__).resolve().parents[1] / "experiments"
    for filename in ("parameter_manifold.py", "parameter_manifold_2pl.py"):
        source = (experiments / filename).read_text(encoding="utf-8")
        assert "apply_academic_style(" in source, filename
        assert "plt.rcParams.update" not in source, filename


def test_legacy_plotting_utils_style_helpers_delegate_to_package():
    experiments = Path(__file__).resolve().parents[1] / "experiments"
    source = (experiments / "plotting_utils.py").read_text(encoding="utf-8")
    style_region = source.split("def _set_academic_style", 1)[1].split(
        "def build_voxel_grid", 1
    )[0]

    assert "apply_academic_style" in source
    assert "style_3d_axis" in source
    assert "apply_academic_style(" in style_region
    assert "style_3d_axis(ax)" in style_region
    assert "plt.rcParams.update" not in style_region
    assert "axis._axinfo" not in style_region


def test_legacy_plotting_utils_density_helpers_delegate_to_package():
    experiments = Path(__file__).resolve().parents[1] / "experiments"
    source = (experiments / "plotting_utils.py").read_text(encoding="utf-8")
    rendering_region = source.split("def render_density_shells", 1)[1]

    assert "render_density_shells as _render_density_shells" in source
    assert "render_gmm_wireframes as _render_gmm_wireframes" in source
    assert "render_reconstructed_gmm_3d as _render_reconstructed_gmm_3d" in source
    assert "_render_density_shells(" in rendering_region
    assert "_render_gmm_wireframes(" in rendering_region
    assert "_render_reconstructed_gmm_3d(" in rendering_region
    assert "mcolors.PowerNorm" not in rendering_region
    assert "np.outer(np.cos" not in rendering_region


def test_analysis_catalog_documents_every_supported_entrypoint():
    experiments = Path(__file__).resolve().parents[1] / "experiments"
    catalog = (experiments / "README.md").read_text(encoding="utf-8")
    for filename in SUPPORTED_ANALYSIS_SCRIPTS:
        assert filename.removesuffix(".py") in catalog


def test_legacy_analysis_modules_require_explicit_dispatch():
    experiments = Path(__file__).resolve().parents[1] / "experiments"
    power_law = (experiments / "power_law.py").read_text(encoding="utf-8")
    scale = (experiments / "reconstruction_scale_determination.py").read_text(
        encoding="utf-8"
    )
    dfr_plot = (experiments / "dfr_plot.py").read_text(encoding="utf-8")

    assert 'parser.add_argument(\n        "experiment"' in power_law
    assert 'parser.add_argument(\n        "experiment"' in scale
    assert "No implicit figure is selected" in dfr_plot


def test_shared_analysis_cli_arguments_enforce_collision_policy():
    parser = argparse.ArgumentParser()
    add_managed_output_arguments(parser)

    args = parser.parse_args(["--run-id", "demo", "--resume"])

    assert args.run_id == "demo" and args.resume and not args.overwrite_run
