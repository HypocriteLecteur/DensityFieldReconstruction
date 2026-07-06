import json

import numpy as np
import pytest
import torch

import dfr
from dfr import CameraConfig, EvaluationConfig, OutputConfig, load_dataset
from dfr.config import ReconstructionParams, TrainingParams
from dfr.evaluation import EvaluationRun, EvaluationSummary
from dfr.evaluation import compute_density_overlap_masses
from dfr.reconstruction import (
    FrameReconstruction,
    ReconstructionRequest,
    ReconstructionRun,
)


def make_reconstruction(tmp_path):
    source = tmp_path / "points.npy"
    positions = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
    np.save(source, positions)
    dataset = load_dataset(source, project_root=tmp_path)
    request = ReconstructionRequest(
        dataset=dataset,
        frames=(0,),
        cameras=CameraConfig.explicit(
            [[-10, 0, 0, 0, 0, 0, 1], [0, -10, 0, 0, 0, 0, 1]],
            device="cuda",
        ),
        training=TrainingParams(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 1, 1),
        reconstruction=ReconstructionParams(1, 0.5, 0.1, 8, 2),
        scale=1.0,
    )
    frame = FrameReconstruction(
        dataset_name="points",
        frame=0,
        positions=positions[0],
        means=positions[0],
        radii=[[1.0]],
        weights=[[1.0]],
        camera_poses=request.cameras.poses,
        projections=(np.zeros((1, 2)), np.zeros((1, 2))),
        visible_mask=[True],
        scale=1.0,
        mean_training_loss=0.0,
        density_dissimilarity=0.0,
        time_ms={},
        scale_space_shapes=((1, 8, 8), (1, 8, 8)),
    )
    return ReconstructionRun(request, (frame,)), dataset


def cpu_config():
    return EvaluationConfig(
        voxel_resolution=0.5,
        batch_size=128,
        bounds=((-3, 3), (-3, 3), (-3, 3)),
        device="cpu",
    )


def test_evaluation_summary_uses_historic_metric_equations():
    summary = EvaluationSummary(8.0, 1.0, 2.0, 10.0, 9.0)

    assert summary.recall == 0.8
    assert summary.miss == 0.2
    assert summary.hallucination == pytest.approx(1 / 9)
    assert summary.dmota == 0.7


def test_public_evaluate_returns_typed_identical_density_result(tmp_path):
    reconstruction, dataset = make_reconstruction(tmp_path)

    result = dfr.evaluate(reconstruction, ground_truth=dataset, config=cpu_config())

    assert isinstance(result, EvaluationRun)
    assert result.frames[0].summary.false_positive_mass == 0
    assert result.frames[0].summary.false_negative_mass == 0
    assert result.summary.recall > 0.9
    assert result.summary.dmota == 1.0
    assert result.artifacts is None


def test_evaluate_can_write_managed_metrics_explicitly(tmp_path):
    reconstruction, _ = make_reconstruction(tmp_path)
    output = OutputConfig(
        workflow="evaluation",
        name="tiny evaluation",
        root="outputs",
        run_id="tiny-eval",
        project_root=tmp_path,
    )

    result = dfr.evaluate(reconstruction, config=cpu_config(), output=output)

    assert result.run_dir == (tmp_path / "outputs" / "evaluation" / "tiny-eval")
    assert (result.artifacts.metrics_dir / "frame_000000.json").is_file()
    summary = json.loads(
        (result.artifacts.metrics_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["dmota"] == 1.0


def test_evaluate_loads_saved_single_frame_reconstruction(tmp_path):
    run_dir = tmp_path / "saved-run"
    (run_dir / "data").mkdir(parents=True)
    (run_dir / "metrics").mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"workflow": "reconstruction"}), encoding="utf-8"
    )
    np.savez(
        run_dir / "data" / "reconstruction.npz",
        positions=np.zeros((1, 3), dtype=np.float32),
        means=np.zeros((1, 3), dtype=np.float32),
        radii=np.ones((1, 1), dtype=np.float32),
        weights=np.ones((1, 1), dtype=np.float32),
        scale=np.asarray(1.0),
    )
    (run_dir / "metrics" / "summary.json").write_text(
        json.dumps({"dataset": "saved", "frame": 12}), encoding="utf-8"
    )

    result = dfr.evaluate(run_dir, config=cpu_config())

    assert result.frames[0].dataset_name == "saved"
    assert result.frames[0].frame == 12
    assert result.summary.dmota == 1.0


def test_evaluate_rejects_wrong_output_workflow(tmp_path):
    reconstruction, _ = make_reconstruction(tmp_path)
    with pytest.raises(ValueError, match="output workflow"):
        dfr.evaluate(
            reconstruction,
            config=cpu_config(),
            output=OutputConfig(workflow="analysis", name="wrong"),
        )


@pytest.mark.cuda
def test_density_overlap_runs_on_cuda_with_identical_semantics():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    masses = compute_density_overlap_masses(
        np.zeros((1, 3), dtype=np.float32),
        1.0,
        np.zeros((1, 3), dtype=np.float32),
        np.ones((1, 1), dtype=np.float32),
        np.ones((1, 1), dtype=np.float32),
        bounds=((-2, 2), (-2, 2), (-2, 2)),
        voxel_resolution=1.0,
        batch_size=16,
        device="cuda",
    )

    assert masses[0] > 0
    assert masses[1:] == (0.0, 0.0)
