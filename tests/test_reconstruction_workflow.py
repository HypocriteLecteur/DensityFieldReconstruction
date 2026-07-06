from pathlib import Path

import numpy as np
import pytest

from dfr import CameraConfig, OutputConfig, load_dataset
from dfr.config import ReconstructionParams
from dfr.reconstruction import build_camera_system
from dfr.reconstruction.pipeline import (
    default_reconstruction_params,
    default_training_params,
)
import dfr.reconstruction.pipeline as reconstruction_pipeline
from dfr.reconstruction.results import (
    FrameReconstruction,
    ReconstructionRequest,
    ReconstructionRun,
)
from dfr.simulation_config import SimulationConfig


def make_scenario(tmp_path):
    project = tmp_path / "project"
    data_dir = project / "dataset"
    scenario_dir = project / "scenarios" / "tiny"
    data_dir.mkdir(parents=True)
    scenario_dir.mkdir(parents=True)
    positions = np.array(
        [[[-0.5, -0.5, 0.0], [0.5, 0.5, 0.0]]], dtype=np.float32
    )
    np.save(data_dir / "tiny.npy", positions)
    config_path = scenario_dir / "config.yaml"
    config_path.write_text(
        "data_file: dataset/tiny.npy\n"
        "cam_poses:\n"
        "  - [-10, 0, 0, 0, 0, 0, 1]\n"
        "  - [0, -10, 0, 0, 0, 0, 1]\n"
        "intrinsics_params: [[60, 0, 31.5], [0, 60, 31.5], [0, 0, 1]]\n"
        "H: 64\nW: 64\nnear_clip: 1\nfar_clip: 30\niter: 1\n"
        "size: 0.2\nsave_video: false\nfps: 30\ndpi: 100\n",
        encoding="utf-8",
    )
    return load_dataset("tiny", project_root=project), config_path


def make_frame(frame=0):
    return FrameReconstruction(
        dataset_name="tiny",
        frame=frame,
        positions=[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
        means=[[0.0, 0.0, 0.0]],
        radii=[[0.5]],
        weights=[[2.0]],
        camera_poses=[[-10, 0, 0, 0, 0, 0, 1], [0, -10, 0, 0, 0, 0, 1]],
        projections=(np.zeros((2, 2)), np.ones((2, 2))),
        visible_mask=[True, False],
        scale=0.5,
        mean_training_loss=0.1,
        density_dissimilarity=0.2,
        time_ms={"train": 3.5},
        scale_space_shapes=((1, 64, 64), (1, 64, 64)),
    )


def test_frame_reconstruction_exposes_data_and_summary():
    result = make_frame()

    assert result.gaussian_count == 1
    assert result.summary()["visible_agent_count"] == 1
    assert result.summary()["time_ms"] == {"train": 3.5}


def test_request_and_run_validate_resolved_contract(tmp_path):
    dataset, config_path = make_scenario(tmp_path)
    request = ReconstructionRequest(
        dataset=dataset,
        frames=(0,),
        cameras=CameraConfig.encircling(device="cuda"),
        training=default_training_params(1),
        reconstruction=default_reconstruction_params(),
        scenario_config=config_path,
    )
    run = ReconstructionRun(request=request, frames=(make_frame(),))

    assert run.run_dir is None
    assert request.to_dict()["dataset"]["scenario_config"] == str(
        config_path.resolve()
    )
    with pytest.raises(ValueError, match="align"):
        ReconstructionRun(request=request, frames=(make_frame(1),))


def test_request_rejects_invalid_backend_and_voxel_controls(tmp_path):
    dataset, config_path = make_scenario(tmp_path)
    with pytest.raises(ValueError, match="device='cuda'"):
        ReconstructionRequest(
            dataset=dataset,
            frames=(0,),
            cameras=CameraConfig.encircling(device="cpu"),
            training=default_training_params(1),
            reconstruction=default_reconstruction_params(),
            scenario_config=config_path,
        )
    with pytest.raises(ValueError, match="threshold"):
        ReconstructionRequest(
            dataset=dataset,
            frames=(0,),
            cameras=CameraConfig.encircling(device="cuda"),
            training=default_training_params(1),
            reconstruction=ReconstructionParams(2, 0.5, 1.5, 12, 8),
            scenario_config=config_path,
        )
    with pytest.raises(ValueError, match="output workflow"):
        ReconstructionRequest(
            dataset=dataset,
            frames=(0,),
            cameras=CameraConfig.encircling(device="cuda"),
            training=default_training_params(1),
            reconstruction=default_reconstruction_params(),
            output=OutputConfig(workflow="analysis", name="wrong"),
            scenario_config=config_path,
        )


def test_camera_builder_supports_explicit_and_encircling_layouts_on_cpu(tmp_path):
    dataset, config_path = make_scenario(tmp_path)
    simulation = SimulationConfig(str(config_path))
    explicit = CameraConfig.explicit(
        [[-10, 0, 0, 0, 0, 0, 1], [0, -10, 0, 0, 0, 0, 1]],
        device="cpu",
    )

    explicit_system = build_camera_system(dataset, (0,), simulation, explicit)
    ring_system = build_camera_system(
        dataset, (0,), simulation, CameraConfig.encircling(device="cpu")
    )

    assert len(explicit_system.cameras) == len(ring_system.cameras) == 2
    np.testing.assert_allclose(explicit_system.cameras[0].state.pose_np, explicit.poses[0])


def test_public_pipeline_returns_typed_run_without_writing(monkeypatch, tmp_path):
    dataset, _ = make_scenario(tmp_path)
    expected = make_frame()
    monkeypatch.setattr(reconstruction_pipeline.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        reconstruction_pipeline, "build_camera_system", lambda *args: object()
    )
    monkeypatch.setattr(
        reconstruction_pipeline,
        "_reconstruct_frame",
        lambda request, frame, simulation, cameras: (expected, object()),
    )

    run = reconstruction_pipeline.reconstruct(
        dataset,
        frames=0,
        cameras=CameraConfig.encircling(device="cuda"),
        training=default_training_params(1),
    )

    assert isinstance(run, ReconstructionRun)
    assert run.frames == (expected,)
    assert run.artifacts is None


@pytest.mark.parametrize("iterations", [0, -1])
def test_default_training_params_rejects_invalid_iterations(iterations):
    with pytest.raises(ValueError, match="positive"):
        default_training_params(iterations)
