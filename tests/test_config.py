import yaml
import pytest

from dfr import OutputConfig
from dfr.config import (
    AnalysisConfig,
    CameraConfig,
    EvaluationConfig,
    ReconstructionParams,
    RunConfig,
    TrainingParams,
)
from dfr.data import DatasetSpec


def test_typed_configs_round_trip_dicts():
    training_values = {
        "xyz_lr_c": 0.1,
        "xyz_lr_final_c": 0.01,
        "radius_lr_c": 0.2,
        "radius_lr_final_c": 0.02,
        "weights_lr_c": 0.3,
        "weights_lr_final_c": 0.03,
        "xyz_reg": 1.0,
        "radius_reg": 2.0,
        "radius_cutoff_inv": 10.0,
        "lr_max_steps": 100,
    }
    reconstruction_values = {
        "targetd_num_mode": 10,
        "voxel_scale": 0.5,
        "voxel_peak_threshold": 0.2,
        "voxel_grid_max_size": 128,
        "voxel_peaks_number": 20,
    }

    assert TrainingParams.from_dict(training_values).to_dict() == training_values
    assert ReconstructionParams.from_dict(reconstruction_values).to_dict() == reconstruction_values
    assert ReconstructionParams.from_dict(
        {**reconstruction_values, "target_mode_count": 10}
    ).target_mode_count == 10


def test_nested_run_config_round_trips_through_yaml(tmp_path):
    dataset = DatasetSpec(
        name="demo",
        data_path=tmp_path / "dataset" / "demo.npy",
        config_path=tmp_path / "scenarios" / "demo" / "config.yaml",
        project_root=tmp_path,
    )
    output = OutputConfig(
        workflow="reconstruction",
        name="demo frame",
        root=tmp_path / "outputs",
        run_id="demo-frame-0",
    )
    camera = CameraConfig.encircling(count=4, padding=1.25, is_3d=True)
    analysis = AnalysisConfig(frames=(0, 5, -1), scales=(0.5, 1.0, 2.0))
    training = TrainingParams(
        xyz_lr_c=0.1,
        xyz_lr_final_c=0.01,
        radius_lr_c=0.2,
        radius_lr_final_c=0.02,
        weights_lr_c=0.3,
        weights_lr_final_c=0.03,
        xyz_reg=1.0,
        radius_reg=2.0,
        radius_cutoff_inv=10.0,
        lr_max_steps=100,
    )
    reconstruction = ReconstructionParams(
        targetd_num_mode=10,
        voxel_scale=0.5,
        voxel_peak_threshold=0.2,
        voxel_grid_max_size=64,
        voxel_peaks_number=20,
    )
    evaluation = EvaluationConfig(
        voxel_resolution=0.25,
        batch_size=2048,
        bounds=((-2, 2), (-3, 3), (-1, 1)),
    )
    original = RunConfig(
        dataset=dataset,
        output=output,
        camera=camera,
        analysis=analysis,
        training=training,
        reconstruction=reconstruction,
        evaluation=evaluation,
        seed=7,
    )

    yaml_text = yaml.safe_dump(original.serializable(), sort_keys=False)
    restored = RunConfig.from_dict(yaml.safe_load(yaml_text))

    assert restored.dataset == dataset
    assert restored.output.to_dict() == output.to_dict()
    assert restored.camera == camera
    assert restored.analysis == analysis
    assert restored.training == training
    assert restored.reconstruction == reconstruction
    assert restored.evaluation == evaluation
    assert restored.seed == 7


def test_explicit_camera_config_normalizes_pose_values():
    camera = CameraConfig.explicit(
        [
            [0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
        ],
        device="cpu",
    )

    assert camera.count == 2
    assert camera.layout == "explicit"
    assert camera.poses[1][0] == 1.0
    assert CameraConfig.from_dict(camera.to_dict()) == camera


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: CameraConfig(count=1), "at least 2"),
        (
            lambda: CameraConfig(count=2, layout="explicit", poses=((0,) * 7,)),
            "one pose per camera",
        ),
        (lambda: AnalysisConfig(frames=()), "must not be empty"),
        (lambda: AnalysisConfig(scales=(1.0, 0.5)), "strictly increasing"),
        (lambda: EvaluationConfig(voxel_resolution=0), "must be positive"),
        (
            lambda: EvaluationConfig(bounds=((0, 1), (0, 1))),
            "three.*pairs",
        ),
        (
            lambda: RunConfig(
                dataset="", output=OutputConfig(workflow="x", name="x")
            ),
            "must not be empty",
        ),
    ],
)
def test_shared_config_validation(factory, message):
    with pytest.raises((ValueError, TypeError), match=message):
        factory()


def test_run_config_rejects_unknown_schema(tmp_path):
    values = RunConfig(
        dataset="demo",
        output=OutputConfig(
            workflow="analysis", name="demo", root=tmp_path, run_id="demo"
        ),
    ).to_dict()
    values["schema_version"] = 99

    with pytest.raises(ValueError, match="Unsupported RunConfig schema"):
        RunConfig.from_dict(values)
