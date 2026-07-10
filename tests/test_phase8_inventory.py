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
        if relative == "experiments/dfr_plot.py":
            continue
        for module in _imports(path):
            if module in {"dfr_plot", "experiments.dfr_plot"}:
                offenders.append(f"{relative}: {module}")

    assert offenders == []


def test_plotting_utils_is_only_used_by_dfr_plot_archive():
    offenders = []
    for path in _python_files():
        relative = path.relative_to(ROOT).as_posix()
        for module in _imports(path):
            if module in {"plotting_utils", "experiments.plotting_utils"}:
                offenders.append(f"{relative}: {module}")

    assert offenders == ["experiments/dfr_plot.py: experiments.plotting_utils"]


def test_phase8_inventory_documents_cleanup_boundaries():
    text = (ROOT / "docs" / "PHASE8_COMPATIBILITY_INVENTORY.md").read_text(
        encoding="utf-8"
    )

    for snippet in (
        "No active Python module imports `experiments.dfr_plot`",
        "`experiments.plotting_utils` is imported only by `experiments.dfr_plot`",
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
        "`experiments.dfr_plot`",
        "`experiments.plotting_utils`",
        "`density_field_reconstruction_copy/`",
        "`experiments_legacy/`",
        "`outputs/`",
        "one surface per commit",
    ):
        assert snippet in text
