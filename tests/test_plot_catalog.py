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


def test_trajectory_legacy_wrapper_uses_package_plotting():
    root = Path(__file__).resolve().parents[1]
    source = (root / "experiments" / "dfr_plot.py").read_text(encoding="utf-8")
    wrapper = source.split("def plot_single_scenario_new", 1)[1].split(
        "def plot_jackdaw2_density_field", 1
    )[0]

    assert "from dfr.plotting import plot_trajectory_snapshot" in wrapper
    assert "plot_trajectory_snapshot(" in wrapper
    assert 'fig.savefig(f"figs/scene_traj_{name}.png"' in wrapper
    assert "logs" not in wrapper


def test_2d_projection_legacy_wrappers_use_package_plotting():
    root = Path(__file__).resolve().parents[1]
    source = (root / "experiments" / "dfr_plot.py").read_text(encoding="utf-8")
    gmm_wrapper = source.split("def plot_jackdaw2_2d_gmm", 1)[1].split(
        "def plot_jackdaw2_2d_observations", 1
    )[0]
    observation_wrapper = source.split("def plot_jackdaw2_2d_observations", 1)[
        1
    ].split("def plot_single_scenario", 1)[0]

    assert "plot_projected_gmm_density" in gmm_wrapper
    assert "transparent_colormap" in gmm_wrapper
    assert "Ellipse(" not in gmm_wrapper
    assert "plot_projection_points" in observation_wrapper
    assert "plot_density_image" in observation_wrapper
    assert "PowerNorm(" not in observation_wrapper


def test_mode_count_curve_legacy_wrapper_uses_package_plotting():
    root = Path(__file__).resolve().parents[1]
    source = (root / "experiments" / "dfr_plot.py").read_text(encoding="utf-8")
    helper_region = source.split("def _validate_nnd_bounds", 1)[1].split(
        "def plot_jackdaw2_mode_count_curve", 1
    )[0]
    wrapper = source.split("def plot_jackdaw2_mode_count_curve", 1)[1].split(
        "def plot_jackdaw2_multiscale_density", 1
    )[0]

    assert "validate_nnd_bounds" in helper_region
    assert "select_adaptive_density_scales" in helper_region
    assert "plot_mode_count_curve" in wrapper
    assert "plt.subplots" not in wrapper
    assert 'os.path.join(os.getcwd(), "figs")' in wrapper
