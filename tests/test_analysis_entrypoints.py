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
