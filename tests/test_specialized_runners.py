from pathlib import Path

import pytest

import experiments.run_scenarios_angle_sweep as angle_runner
import experiments.run_scenarios_flock as flock_runner
import experiments.run_scenarios_ue4 as ue4_runner


def test_angle_ordinary_path_uses_shared_scenario_runner(monkeypatch, tmp_path):
    captured = {}
    expected = object()

    def fake_run(spec, *, project_root):
        captured["spec"] = spec
        captured["project_root"] = project_root
        return expected

    monkeypatch.setattr(angle_runner, "run_scenario", fake_run)
    params = {
        "name": "starling",
        "log_name": "test",
        "start_step": 1,
        "end_step": 5,
        "step_length": 2,
        "noise_std": 0.5,
    }

    actual = angle_runner.run_single_scenario(
        params, project_root=tmp_path, seed=9
    )

    assert actual is expected
    assert captured["project_root"] == tmp_path.resolve()
    assert captured["spec"].dataset == "starling"
    assert captured["spec"].projection_noise_std == 0.5
    assert captured["spec"].training.lr_max_steps == 100


def test_specialized_runners_require_explicit_dispatch():
    assert angle_runner.create_parser().parse_args(["profile"]).study == "profile"
    assert flock_runner.create_parser().parse_args(["visualize"]).study == "visualize"
    with pytest.raises(SystemExit):
        angle_runner.create_parser().parse_args([])
    with pytest.raises(SystemExit):
        flock_runner.create_parser().parse_args([])
    with pytest.raises(SystemExit):
        ue4_runner.create_parser().parse_args([])


def test_flock_inputs_are_explicit_and_validated(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    inputs = []
    for name in ("extrinsics.json", "camera1.csv", "camera2.csv"):
        path = tmp_path / name
        path.write_text("fixture", encoding="utf-8")
        inputs.append(path)

    config = flock_runner.FlockInputConfig(
        data_root=data_root,
        extrinsics_json=inputs[0],
        detections_camera_1=inputs[1],
        detections_camera_2=inputs[2],
        project_root=tmp_path,
    )

    assert config.data_root == data_root.resolve()
    with pytest.raises(FileNotFoundError, match="input file"):
        flock_runner.FlockInputConfig(
            data_root=data_root,
            extrinsics_json=tmp_path / "missing.json",
            detections_camera_1=inputs[1],
            detections_camera_2=inputs[2],
        )


def test_flock_primary_run_uses_external_observation_workflow():
    source = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "run_scenarios_flock.py"
    ).read_text(encoding="utf-8")
    primary = source.split("def run_flock_scenario", 1)[1].split(
        "from matplotlib.widgets import Slider", 1
    )[0]

    assert "ExternalObservationFrame(" in primary
    assert "reconstruct_observations(" in primary
    assert "DensityReconstructor(" not in primary
    assert "output_dir=" not in primary


def test_flock_scale_selection_supports_frame_and_ordered_caches():
    assert flock_runner._select_frame_scales([0.1, 0.2, 0.3, 0.4], [0, 2]) == [
        0.1,
        0.3,
    ]
    assert flock_runner._select_frame_scales([1.0, 2.0], [5, 10]) == [1.0, 2.0]
    with pytest.raises(ValueError, match="Scale cache"):
        flock_runner._select_frame_scales([1.0], [5, 10])


def test_specialized_modules_do_not_install_root_file_loggers():
    root = Path(__file__).resolve().parents[1] / "experiments"
    for filename in ("run_scenarios_flock.py", "run_scenarios_ue4.py"):
        source = (root / filename).read_text(encoding="utf-8")
        assert "logging.FileHandler" not in source
        assert "if __name__ == \"__main__\":" in source
