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
