from pathlib import Path

import dfr.plotting as plotting


SMOKE_MATRIX = {
    "camera": {
        "test_file": "test_plotting_cameras.py",
        "functions": ("plot_camera_configurations",),
    },
    "2d": {
        "test_file": "test_plotting_projections.py",
        "functions": (
            "plot_projection_points",
            "plot_density_image",
            "plot_projected_gmm_density",
        ),
    },
    "3d": {
        "test_file": "test_plotting_density.py",
        "functions": (
            "plot_density_field_3d",
            "plot_multiscale_density_fields",
            "plot_frame_reconstruction_gmm_3d",
        ),
    },
    "trajectory": {
        "test_file": "test_plotting_trajectories.py",
        "functions": ("plot_trajectory_snapshot",),
    },
    "scale": {
        "test_file": "test_plotting_analysis.py",
        "functions": (
            "plot_mode_count_curve",
            "plot_dra_scale_model_order_surface",
            "plot_dra_surface_grid",
        ),
    },
    "evaluation": {
        "test_file": "test_plotting_evaluation.py",
        "functions": (
            "plot_evaluation_summary",
            "plot_evaluation_metric_series",
        ),
    },
}


def test_representative_plotting_smoke_matrix_stays_covered():
    tests_dir = Path(__file__).resolve().parent

    for family, spec in SMOKE_MATRIX.items():
        source = (tests_dir / spec["test_file"]).read_text(encoding="utf-8")
        assert 'matplotlib.use("Agg")' in source, family
        for function_name in spec["functions"]:
            assert function_name in plotting.__all__, function_name
            assert hasattr(plotting, function_name), function_name
            assert function_name in source, function_name
