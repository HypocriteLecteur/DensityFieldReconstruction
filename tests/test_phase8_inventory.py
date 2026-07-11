import ast
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _python_files():
    for folder in ("dfr", "experiments", "tests", "examples"):
        yield from (ROOT / folder).rglob("*.py")


def _imports(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            yield module


def test_no_active_python_imports_dfr_plot_archive():
    offenders = []
    for path in _python_files():
        relative = path.relative_to(ROOT).as_posix()
        for module in _imports(path):
            if module in {"dfr_plot", "experiments.dfr_plot"}:
                offenders.append(f"{relative}: {module}")

    assert offenders == []


def test_no_active_python_imports_plotting_utils_archive():
    offenders = []
    for path in _python_files():
        relative = path.relative_to(ROOT).as_posix()
        for module in _imports(path):
            if module in {"plotting_utils", "experiments.plotting_utils"}:
                offenders.append(f"{relative}: {module}")

    assert offenders == []


def test_retired_plot_archive_files_are_absent():
    for name in ("dfr_plot.py", "plotting_utils.py"):
        assert not (ROOT / "experiments" / name).exists()


def test_isolated_scenario_log_diagnostics_are_absent():
    for name in ("visualize.py", "inspect_3d_error.py", "run_post_processing.py"):
        assert not (ROOT / "experiments" / name).exists()


def test_isolated_unmanaged_analysis_scripts_are_absent():
    for name in (
        "reconstruction_scale_determination.py",
        "search_learning_parameters.py",
        "search_regularization_parameters.py",
        "compute_metrics_from_pretrained.py",
    ):
        assert not (ROOT / "experiments" / name).exists()


def test_isolated_interactive_diagnostics_are_absent():
    for name in (
        "dataset_viewer_test.py",
        "inspect_scenarios.py",
        "investigate_initialization.py",
        "rasterizer_optimize.py",
    ):
        assert not (ROOT / "experiments" / name).exists()


def test_phase8_inventory_documents_cleanup_boundaries():
    text = (ROOT / "docs" / "PHASE8_COMPATIBILITY_INVENTORY.md").read_text(
        encoding="utf-8"
    )

    for snippet in (
        "`experiments/dfr_plot.py` and `experiments/plotting_utils.py` were removed",
        "No active Python module imports `experiments.dfr_plot`",
        "No active Python module imports `experiments.plotting_utils`",
        "python -m experiments.plot_catalog --list-functions",
        "density_field_reconstruction_copy/",
        "experiments_legacy/",
        "`figs/`",
        "`results/`",
        "`scenarios/*/logs/`",
        "Do not add new scenario-log producers",
        "docs/PHASE8_ARCHIVE_POLICY.md",
    ):
        assert snippet in text


def test_phase8_archive_policy_documents_deletion_rules():
    text = (ROOT / "docs" / "PHASE8_ARCHIVE_POLICY.md").read_text(
        encoding="utf-8"
    )

    for snippet in (
        "local Git history",
        "`v0.1.0`",
        "Do not keep duplicate source trees",
        "Retired legacy plot archive",
        "`density_field_reconstruction_copy/`",
        "`experiments_legacy/`",
        "`outputs/`",
        "one surface per commit",
        "were never stored in Git",
    ):
        assert snippet in text


def test_phase8_inventory_documents_remaining_legacy_output_surfaces():
    text = (ROOT / "docs" / "PHASE8_COMPATIBILITY_INVENTORY.md").read_text(
        encoding="utf-8"
    )

    for snippet in (
        "experiments.power_law",
        "experiments.parameter_manifold",
        "experiments.parameter_manifold_2pl",
        "experiments.fit_dra_multiframe.seed_existing_cache",
        "experiments.run_scenarios_angle_sweep",
        "experiments.run_scenarios_flock",
        "reconstruction_scale_determination",
        "search_learning_parameters",
        "search_regularization_parameters",
        "compute_metrics_from_pretrained",
        "were removed on 2026-07-11",
    ):
        assert snippet in text


def test_no_active_code_references_copied_backup_directories():
    names = ("density_field_reconstruction_copy", "experiments_legacy")
    offenders = []
    for folder in ("dfr", "experiments", "examples"):
        for path in (ROOT / folder).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if any(name in text for name in names):
                offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []


def test_copied_backup_directories_are_absent_after_cleanup():
    for name in ("density_field_reconstruction_copy", "experiments_legacy"):
        assert not (ROOT / name).exists()
