from pathlib import Path

import numpy as np
import pytest

import dfr
from dfr.data import Dataset, DatasetSpec, ScenarioRegistry, select_frame_indices
from dfr.dataset_io import InvalidFileFormatError


@pytest.fixture
def project(tmp_path):
    data_dir = tmp_path / "dataset"
    scenario_dir = tmp_path / "scenarios" / "demo"
    data_dir.mkdir()
    scenario_dir.mkdir(parents=True)
    trajectories = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            [[0.5, 0.0, 0.0], [1.5, 1.0, 1.0]],
            [[1.0, 0.0, 0.0], [2.0, 1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    np.save(data_dir / "demo.npy", trajectories)
    (scenario_dir / "config.yaml").write_text(
        "data_file: dataset/demo.npy\n", encoding="utf-8"
    )
    return tmp_path, trajectories


def test_named_scenario_loads_through_public_api(project):
    root, trajectories = project

    dataset = dfr.load_dataset("demo", project_root=root)

    assert isinstance(dataset, Dataset)
    assert len(dataset) == dataset.frame_count == 3
    assert dataset.source_path == (root / "dataset" / "demo.npy").resolve()
    assert dataset.metadata["dataset_name"] == "demo"
    assert dataset.metadata["loader"] == "NpyLoader"
    assert dataset.coordinate_system is None
    assert dataset.timestamps is None
    assert not dataset.has_timestamps
    assert not dataset.has_velocities
    np.testing.assert_allclose(dataset.ground_truth_positions, trajectories)


def test_registry_resolves_named_config_and_explicit_data(project):
    root, _ = project
    registry = ScenarioRegistry(root)

    named = registry.resolve("demo")
    configured = registry.resolve(root / "scenarios" / "demo" / "config.yaml")
    explicit = registry.resolve(Path("dataset") / "demo.npy")

    assert registry.available_scenarios() == ("demo",)
    assert named == configured
    assert named.name == "demo"
    assert named.project_root == root.resolve()
    assert named.config_path == (root / "scenarios" / "demo" / "config.yaml").resolve()
    assert explicit == DatasetSpec(
        name="demo",
        data_path=root / "dataset" / "demo.npy",
        project_root=root,
    )


def test_standalone_config_resolves_data_relative_to_config(tmp_path):
    bundle = tmp_path / "portable"
    bundle.mkdir()
    np.save(bundle / "points.npy", np.zeros((1, 2, 3), dtype=np.float32))
    config = bundle / "experiment.yaml"
    config.write_text("data_file: points.npy\n", encoding="utf-8")

    spec = dfr.resolve_dataset(config, project_root=tmp_path)
    dataset = dfr.load_dataset(config, project_root=tmp_path)

    assert spec.data_path == (bundle / "points.npy").resolve()
    assert spec.name == "portable"
    assert dataset.frame_count == 1


def test_frame_selection_normalizes_and_validates(project):
    root, _ = project
    dataset = dfr.load_dataset("demo", project_root=root)

    assert select_frame_indices(dataset, None) == (0, 1, 2)
    assert select_frame_indices(dataset, -1) == (2,)
    assert select_frame_indices(dataset, slice(0, None, 2)) == (0, 2)
    assert select_frame_indices(dataset, [2, 0, 2]) == (2, 0, 2)
    np.testing.assert_allclose(
        dataset.positions_at_time_step(-1), dataset.trajectories[2]
    )

    with pytest.raises(IndexError, match="valid range"):
        select_frame_indices(dataset, 3)
    with pytest.raises(ValueError, match="empty"):
        select_frame_indices(dataset, [])
    with pytest.raises(ValueError, match="step cannot be zero"):
        select_frame_indices(dataset, slice(None, None, 0))
    with pytest.raises(TypeError, match="must not be a string"):
        select_frame_indices(dataset, "0")


def test_errors_identify_scenario_path_format_and_optional_data(project, tmp_path):
    root, _ = project
    registry = ScenarioRegistry(root)

    with pytest.raises(FileNotFoundError, match="Available scenarios: demo"):
        registry.resolve("missing")

    missing_config = root / "scenarios" / "missing-data"
    missing_config.mkdir()
    (missing_config / "config.yaml").write_text(
        "data_file: dataset/not-there.npy\n", encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="not-there.npy"):
        registry.resolve("missing-data")

    unsupported = tmp_path / "points.xyz"
    unsupported.write_text("0 0 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported dataset extension '.xyz'"):
        dfr.load_dataset(unsupported, project_root=root)

    malformed = tmp_path / "malformed.npy"
    np.save(malformed, np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(InvalidFileFormatError, match="could not be parsed"):
        dfr.load_dataset(malformed, project_root=root)

    dataset = dfr.load_dataset("demo", project_root=root)
    with pytest.raises(ValueError, match="calculate_velocities"):
        _ = dataset.velocities
