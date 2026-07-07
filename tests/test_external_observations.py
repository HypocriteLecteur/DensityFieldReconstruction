import numpy as np
import pytest
import torch

from dfr import OutputConfig
from dfr.reconstruction.observations import (
    ExternalObservationFrame,
    reconstruct_observations,
)
from dfr.reconstruction.pipeline import default_training_params
from dfr.reconstruction.results import ReconstructionRun
import dfr.density_field_reconstructor as reconstructor_module
import dfr.reconstruction.observations as observation_pipeline


class _State:
    W = 64
    H = 48
    far_clip = 30

    def __init__(self, pose):
        self.pose_np = np.asarray(pose, dtype=np.float32)


class _Camera:
    def __init__(self, pose):
        self.state = _State(pose)


class _CameraSystem:
    def __init__(self):
        self.cameras = [
            _Camera([-10, 0, 0, 0, 0, 0, 1]),
            _Camera([0, -10, 0, 0, 0, 0, 1]),
        ]


class _Model:
    def __init__(self):
        self._xyz = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        self._radius = torch.tensor([[0.5]], dtype=torch.float32)
        self._weights = torch.tensor([[1.0]], dtype=torch.float32)
        self.mean_loss = 0.125


def _observation(**overrides):
    values = {
        "dataset_name": "measured-flock",
        "frame": 7,
        "positions": np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
        ),
        "projections": (
            np.array([[10.0, 12.0], [20.0, 22.0]], dtype=np.float32),
            np.array([[11.0, 13.0], [21.0, 23.0]], dtype=np.float32),
        ),
        "camera_system": _CameraSystem(),
        "visible_mask": np.array([False, False]),
    }
    values.update(overrides)
    return ExternalObservationFrame(**values)


def test_external_observation_validates_camera_projection_contract():
    frame = _observation()

    assert frame.camera_poses.shape == (2, 7)
    assert frame.to_dict()["projection_counts"] == [2, 2]
    with pytest.raises(ValueError, match="one array per camera"):
        _observation(projections=(np.zeros((2, 2), dtype=np.float32),))
    with pytest.raises(ValueError, match="positions"):
        _observation(positions=np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="visible_mask"):
        _observation(visible_mask=np.array([True]))


def test_reconstruct_observations_dispatches_external_inputs(monkeypatch):
    captured = {}

    class FakeReconstructor:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            self.scale = 0.75
            self.time_metrics = {"train": 4.0}

        def process_frame(self, camera_system, **kwargs):
            captured["camera_system"] = camera_system
            captured["process"] = kwargs
            return [_Model()], [np.zeros((1, 8, 8), dtype=np.float32)]

    monkeypatch.setattr(observation_pipeline.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        reconstructor_module, "DensityReconstructor", FakeReconstructor
    )
    frame = _observation()

    run = reconstruct_observations(
        [frame],
        frame_scales=[0.75],
        training=default_training_params(3),
        seed=9,
    )

    assert isinstance(run, ReconstructionRun)
    assert run.request.dataset.metadata["observation_source"] == "external"
    assert run.frames[0].dataset_name == "measured-flock"
    assert run.frames[0].scale == 0.75
    assert run.frames[0].density_dissimilarity is None
    assert captured["init"]["W"] == 64
    assert captured["init"]["H"] == 48
    assert captured["init"]["far_clip"] == 30
    assert captured["process"]["point_sets"] == list(frame.projections)
    assert captured["process"]["positions"] is frame.positions
    assert captured["process"]["scale"] == 0.75
    assert captured["process"]["is_adaptive_scale"] is False


def test_reconstruct_observations_can_create_managed_artifacts(
    monkeypatch, tmp_path
):
    class FakeReconstructor:
        def __init__(self, **kwargs):
            self.scale = 1.25
            self.time_metrics = {"train": 2.0}

        def process_frame(self, *args, **kwargs):
            return [_Model()], [np.zeros((1, 4, 4), dtype=np.float32)]

    monkeypatch.setattr(observation_pipeline.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        reconstructor_module, "DensityReconstructor", FakeReconstructor
    )
    monkeypatch.setattr(observation_pipeline, "_save_frame", lambda *args, **kwargs: None)

    run = reconstruct_observations(
        [_observation()],
        output=OutputConfig(
            workflow="reconstruction",
            name="external-test",
            root=tmp_path / "outputs",
            run_id="external-test",
        ),
    )

    assert run.run_dir == tmp_path / "outputs" / "reconstruction" / "external-test"
    assert (run.run_dir / "config.yaml").is_file()
    assert (run.run_dir / "data" / "statistics.npz").is_file()
