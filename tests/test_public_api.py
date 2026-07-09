import dfr
from dfr.data import load_dataset, resolve_dataset
from dfr.evaluation import evaluate
from dfr.plotting import plot_density_field_3d, plot_projected_gmm_density
from dfr.reconstruction import reconstruct


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


def test_high_traffic_api_docstrings_document_contracts():
    expectations = {
        load_dataset: ("world-coordinate", "shape", "does not create output"),
        resolve_dataset: ("scenario", "paths", "load_dataset"),
        reconstruct: ("world-coordinate", "CUDA", "writes nothing"),
        evaluate: ("world-coordinate", "writes nothing", "EvaluationRun"),
        plot_density_field_3d: ("world-coordinate", "returns", "does not save"),
        plot_projected_gmm_density: ("shape", "returns", "leaves saving"),
    }

    for function, snippets in expectations.items():
        doc = function.__doc__ or ""
        for snippet in snippets:
            assert snippet in doc, f"{function.__name__} docstring is missing {snippet!r}"
