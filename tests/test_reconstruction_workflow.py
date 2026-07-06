from pathlib import Path

import numpy as np
import pytest

from dfr import CameraConfig, OutputConfig, load_dataset
from dfr.config import ReconstructionParams
from dfr.reconstruction import add_bounded_projection_noise, build_camera_system
from dfr.camera_state import CameraState
from dfr.camera_system import MultiCameraSystem
from dfr.utils import generate_encircling_cameras
from experiments.common import load_scenario, setup_camera_system
from dfr.reconstruction.pipeline import (
    default_reconstruction_params,
    default_training_params,
)
import dfr.reconstruction.pipeline as reconstruction_pipeline
import dfr.reconstruction.scenarios as scenario_runner
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


def test_common_camera_adapter_matches_legacy_auto_aim_projections(tmp_path):
    dataset, config_path = make_scenario(tmp_path)
    simulation = SimulationConfig(str(config_path))
    positions, _ = generate_encircling_cameras(
        dataset,
        (0,),
        simulation.intrinsics_params,
        simulation.H,
        simulation.W,
        cam_num=4,
    )
    legacy_poses = np.hstack(
        (positions[:2], np.tile(np.array([1, 0, 0, 0]), (2, 1)))
    ).astype(np.float32)
    legacy = MultiCameraSystem.create_homogeneous_system(
        CameraState,
        simulation.intrinsics_params,
        simulation.H,
        simulation.W,
        legacy_poses,
        simulation.near_clip,
        simulation.far_clip,
        simulation.size,
        "cpu",
    )
    adapted = setup_camera_system(dataset, (0,), simulation, 2, device="cpu")

    legacy_output = legacy.simulate_vision(
        dataset.positions_at_time_step(0), renderer="projection_only"
    )
    adapted_output = adapted.simulate_vision(
        dataset.positions_at_time_step(0), renderer="projection_only"
    )

    np.testing.assert_allclose(legacy_output[0], adapted_output[0], atol=1e-6)
    for old, new in zip(legacy_output[1], adapted_output[1]):
        np.testing.assert_allclose(old, new, atol=1e-5)


def test_common_scenario_loader_delegates_to_canonical_registry(tmp_path):
    expected, config_path = make_scenario(tmp_path)

    config, loaded = load_scenario("tiny", str(config_path.parent))

    assert config.W == 64
    assert loaded.metadata["dataset_name"] == "tiny"
    np.testing.assert_array_equal(loaded.trajectories, expected.trajectories)


def test_request_supports_explicit_per_frame_scales_and_noise(tmp_path):
    dataset, config_path = make_scenario(tmp_path)
    request = ReconstructionRequest(
        dataset=dataset,
        frames=(0, 0),
        cameras=CameraConfig.encircling(device="cuda"),
        training=default_training_params(1),
        reconstruction=default_reconstruction_params(),
        frame_scales=(0.5, 0.75),
        projection_noise_std=1.0,
        scenario_config=config_path,
    )

    assert request.scale_for_index(0) == 0.5
    assert request.scale_for_index(1) == 0.75
    with pytest.raises(ValueError, match="mutually exclusive"):
        ReconstructionRequest(
            dataset=dataset,
            frames=(0,),
            cameras=CameraConfig.encircling(device="cuda"),
            training=default_training_params(1),
            reconstruction=default_reconstruction_params(),
            scale=0.5,
            frame_scales=(0.5,),
            scenario_config=config_path,
        )


def test_projection_noise_is_seeded_and_bounded(tmp_path):
    dataset, config_path = make_scenario(tmp_path)
    simulation = SimulationConfig(str(config_path))
    cameras = setup_camera_system(dataset, (0,), simulation, 2, device="cpu")
    projections = [np.array([[31.5, 31.5]]) for _ in range(2)]

    first = add_bounded_projection_noise(
        projections, cameras, 3.0, np.random.default_rng(7)
    )
    second = add_bounded_projection_noise(
        projections, cameras, 3.0, np.random.default_rng(7)
    )

    for actual, repeated in zip(first, second):
        np.testing.assert_allclose(actual, repeated)
        assert np.all((actual >= 0) & (actual <= 64))


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
        lambda request, frame_index, frame, simulation, cameras, rng: (
            expected,
            object(),
        ),
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


def test_representative_scenario_runner_dispatches_to_public_workflow():
    source = (
        Path(__file__).resolve().parents[1] / "experiments" / "run_scenarios.py"
    ).read_text(encoding="utf-8")
    managed = source.split("def run_single_scenario", 1)[1].split(
        "def _run_single_scenario_legacy", 1
    )[0]

    assert "run = run_scenario(" in managed
    assert "ScenarioRunSpec(" in managed
    assert "projection_noise_std=" in managed


def test_scenario_runner_resolves_frames_scales_and_public_dispatch(
    monkeypatch, tmp_path
):
    dataset, config_path = make_scenario(tmp_path)
    np.savez(config_path.parent / "reconstruction_scale.npz", scales_gt=[0.75])
    captured = {}
    expected = object()

    monkeypatch.setattr(scenario_runner, "load_dataset", lambda *args, **kwargs: dataset)

    def fake_reconstruct(loaded, **kwargs):
        captured.update(kwargs)
        assert loaded is dataset
        return expected

    monkeypatch.setattr(scenario_runner, "reconstruct", fake_reconstruct)
    monkeypatch.setattr(scenario_runner, "_save_statistics", lambda run: None)
    spec = scenario_runner.ScenarioRunSpec(
        dataset="tiny",
        cameras=CameraConfig.encircling(device="cuda"),
        training=default_training_params(7),
    )

    actual = scenario_runner.run_scenario(spec, project_root=tmp_path)

    assert actual is expected
    assert captured["frames"] == (0,)
    assert captured["frame_scales"] == (0.75,)
    assert captured["training"].lr_max_steps == 7


def test_scenario_spec_validation_and_serialization():
    spec = scenario_runner.ScenarioRunSpec(
        dataset="starling",
        start=2,
        stop=8,
        step=2,
        use_ground_truth_scales=False,
        projection_noise_std=1.5,
    )

    assert spec.to_dict()["cameras"]["count"] == 2
    assert spec.to_dict()["projection_noise_std"] == 1.5
    with pytest.raises(ValueError, match="step"):
        scenario_runner.ScenarioRunSpec(dataset="starling", step=0)
