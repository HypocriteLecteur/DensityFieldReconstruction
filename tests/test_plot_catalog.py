import ast
from pathlib import Path


def test_dfr_plot_catalog_lists_every_top_level_function():
    root = Path(__file__).resolve().parents[1]
    source = root / "experiments" / "dfr_plot.py"
    catalog = root / "experiments" / "DFR_PLOT_CATALOG.md"

    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = [
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    ]
    document = catalog.read_text(encoding="utf-8")

    assert len(functions) == 36
    for name in functions:
        assert f"`{name}`" in document


def test_camera_configuration_legacy_wrapper_uses_package_plotting():
    root = Path(__file__).resolve().parents[1]
    source = (root / "experiments" / "dfr_plot.py").read_text(encoding="utf-8")
    wrapper = source.split("def plot_camera_configurations", 1)[1].split(
        "def plot_table_2_results", 1
    )[0]

    assert "from dfr.plotting import plot_camera_configurations" in wrapper
    assert "_plot_camera_configurations(" in wrapper
    assert 'os.path.join(os.getcwd(), "figs")' in wrapper
