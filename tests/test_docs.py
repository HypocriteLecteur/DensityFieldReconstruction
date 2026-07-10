from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_links_phase_7_docs():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "docs/WORKFLOW.md" in readme
    assert "docs/MODULE_OWNERSHIP.md" in readme
    assert "docs/COMMAND_VERIFICATION.md" in readme


def test_workflow_docs_cover_public_path_and_output_policy():
    guide = (ROOT / "docs" / "WORKFLOW.md").read_text(encoding="utf-8")

    for snippet in (
        "load_dataset -> analyze -> reconstruct -> evaluate -> plot",
        "dfr.load_dataset",
        "dfr.analyze",
        "dfr.reconstruct",
        "dfr.evaluate",
        "plot_mode_count_curve",
        "plot_frame_reconstruction_gmm_3d",
        "python examples/toy_workflow.py",
        "writes nothing",
        "outputs/reconstruction/<run-id>/",
    ):
        assert snippet in guide


def test_module_ownership_docs_cover_package_and_experiment_boundaries():
    guide = (ROOT / "docs" / "MODULE_OWNERSHIP.md").read_text(encoding="utf-8")

    for snippet in (
        "dfr.analysis",
        "dfr.reconstruction",
        "dfr.evaluation",
        "dfr.plotting",
        "experiments.plot_catalog",
        "outputs/<workflow>/<run-id>/",
        "Do not add new producers",
    ):
        assert snippet in guide


def test_command_verification_docs_classify_runnable_and_heavy_commands():
    guide = (ROOT / "docs" / "COMMAND_VERIFICATION.md").read_text(encoding="utf-8")

    for snippet in (
        "python examples/toy_workflow.py",
        "python -m experiments.reconstruct_one_frame --help",
        "python -m experiments.run_scenarios_ue4 --help",
        "Intentionally not executed",
        "CUDA reconstruction",
        "large ignored datasets",
        "compiled rasterizers",
    ):
        assert snippet in guide
