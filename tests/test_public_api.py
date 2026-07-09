import dfr


EXPECTED_PUBLIC_API = {
    "AnalysisConfig",
    "CameraConfig",
    "Dataset",
    "DatasetSpec",
    "EvaluationConfig",
    "EvaluationRun",
    "EvaluationSummary",
    "ExternalObservationFrame",
    "FrameEvaluation",
    "OutputConfig",
    "FrameReconstruction",
    "ReconstructionRequest",
    "ReconstructionRun",
    "RunArtifacts",
    "RunConfig",
    "ScenarioRegistry",
    "ScenarioRunSpec",
    "analyze",
    "evaluate",
    "load_dataset",
    "resolve_dataset",
    "reconstruct",
    "reconstruct_observations",
    "run_scenario",
    "run_scenarios",
    "select_frame_indices",
}


def test_top_level_public_api_is_intentional():
    assert set(dfr.__all__) == EXPECTED_PUBLIC_API
    for name in EXPECTED_PUBLIC_API:
        assert hasattr(dfr, name), name


def test_top_level_package_docstring_documents_core_contract():
    doc = dfr.__doc__ or ""

    assert "load_dataset" in doc
    assert "analyze" in doc
    assert "reconstruct" in doc
    assert "evaluate" in doc
    assert "world-coordinate" in doc
    assert "do not write artifacts" in doc
