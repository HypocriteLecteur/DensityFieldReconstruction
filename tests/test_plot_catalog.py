import re
from pathlib import Path

from experiments.plot_catalog import legacy_function_names, main


SUPPORTED_FUNCTIONS = {
    "plot_single_scenario_new",
    "plot_jackdaw2_2d_gmm",
    "plot_jackdaw2_2d_observations",
    "plot_jackdaw2_mode_count_curve",
    "plot_jackdaw2_multiscale_density",
    "plot_jackdaw2_dra_scale_model_order_surface",
    "plot_camera_configurations",
    "plot_table_2_results",
    "plot_table_time_efficiency",
    "plot_table_noise_robustness",
}

ARCHIVED_FUNCTIONS = {
    "scale_estimation",
    "plot_multiple_scenarios",
    "plot_jackdaw2_density_field",
    "plot_all_ground_truth_density_fields",
    "plot_single_scenario",
    "overview_scaling_law",
    "plot_scale_space_curve",
    "visual_hull_diagram",
    "assumption_3_error",
    "visual_hull_tau_vs_visual_hull_ghost",
    "run_geometric_visual_hulls",
    "plot_ratio_surface",
    "dra_metrics",
    "one_frame_parameter_search",
    "one_frame_convergence",
    "one_frame_dMOTA_factor_analysis",
    "one_frame_dMOTA_factor_analysis_2",
    "one_frame_dMOTA_noise",
    "one_frame_dMOTA_3d_noise",
    "plot_dra_and_loss",
}


def _catalog_policy_sets():
    path = Path(__file__).resolve().parents[1] / "experiments" / "DFR_PLOT_CATALOG.md"
    text = path.read_text(encoding="utf-8")
    support_section = text.split("### Supported compatibility wrappers", 1)[1].split(
        "### Archive-only public functions", 1
    )[0]
    archive_section = text.split("### Archive-only public functions", 1)[1].split(
        "## Open questions for the owner", 1
    )[0]
    pattern = r"^- `([A-Za-z0-9_]+)`"
    return (
        set(re.findall(pattern, support_section, re.MULTILINE)),
        set(re.findall(pattern, archive_section, re.MULTILINE)),
    )


def test_frozen_catalog_preserves_every_legacy_public_function():
    names = set(legacy_function_names())

    assert len(names) == 30
    assert names == SUPPORTED_FUNCTIONS | ARCHIVED_FUNCTIONS


def test_catalog_support_policy_remains_complete_and_disjoint():
    supported, archived = _catalog_policy_sets()

    assert supported == SUPPORTED_FUNCTIONS
    assert archived == ARCHIVED_FUNCTIONS
    assert supported.isdisjoint(archived)
    assert supported | archived == set(legacy_function_names())


def test_standalone_catalog_lists_frozen_legacy_public_functions(capsys):
    expected = sorted(SUPPORTED_FUNCTIONS | ARCHIVED_FUNCTIONS)

    assert main(["--list-functions"]) == 0
    assert capsys.readouterr().out.splitlines() == expected
