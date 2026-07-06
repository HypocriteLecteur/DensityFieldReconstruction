import json
from pathlib import Path

import numpy as np
import pytest
import h5py
import scipy.io

from dfr.dataset_io import (
    CentralDifference,
    DatasetFactory,
    ForwardDifference,
    InvalidFileFormatError,
    NpyLoader,
)


FIXTURE = Path(__file__).parent / "fixtures" / "tiny_trajectory.json"


@pytest.fixture
def trajectory_fixture():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data, np.asarray(data["trajectories"], dtype=np.float32)


def test_factory_loads_npy_and_filters_missing_agents(tmp_path, trajectory_fixture):
    expected, trajectories = trajectory_fixture
    path = tmp_path / "trajectory.npy"
    np.save(path, trajectories)

    dataset = DatasetFactory().get_dataset(path)

    assert dataset.trajectories.shape == (3, 3, 3)
    assert [len(dataset.positions_at_time_step(i)) for i in range(3)] == expected[
        "valid_counts"
    ]
    positions, mask = dataset.positions_at_time_step_mask(0)
    np.testing.assert_array_equal(mask, [True, True, False])
    np.testing.assert_allclose(positions, trajectories[0, :2])


def test_factory_loads_standard_npz_velocities(tmp_path, trajectory_fixture):
    _, trajectories = trajectory_fixture
    velocities = CentralDifference().calculate(trajectories)
    path = tmp_path / "trajectory.npz"
    np.savez(path, trajectories=trajectories, velocities=velocities)

    dataset = DatasetFactory().get_dataset(path)

    np.testing.assert_allclose(dataset.trajectories, trajectories, equal_nan=True)
    np.testing.assert_allclose(dataset.velocities, velocities, equal_nan=True)


def test_factory_loads_positions_npz_layout(tmp_path, trajectory_fixture):
    _, trajectories = trajectory_fixture
    path = tmp_path / "positions.npz"
    np.savez(path, positions=trajectories)

    dataset = DatasetFactory().get_dataset(path)

    assert dataset.metadata["loader"] == "NpzPositionsLoader"
    np.testing.assert_allclose(dataset.trajectories, trajectories, equal_nan=True)


def test_factory_loads_matlab_swarm_layout(tmp_path):
    source = np.arange(3 * 2 * 4, dtype=np.float64).reshape(3, 2, 4)
    path = tmp_path / "swarm.mat"
    scipy.io.savemat(path, {"swarm_data": {"positions": source}})

    dataset = DatasetFactory().get_dataset(path)

    assert dataset.trajectories.shape == (4, 2, 3)
    np.testing.assert_allclose(dataset.trajectories, np.transpose(source, (2, 1, 0)))


def test_factory_loads_two_frame_rtf_layout(tmp_path):
    path = tmp_path / "flock.rtf"
    path.write_text(
        "header\n"
        "#  x(t1)    y(t1)    z(t1)      x(t2)  y(t2)    z(t2)\n"
        "0 0 0 1 0 0\n"
        "5 5 5 6 5 5\n",
        encoding="utf-8",
    )

    dataset = DatasetFactory().get_dataset(path)

    assert dataset.trajectories.shape == (2, 2, 3)
    np.testing.assert_allclose(dataset.trajectories[1], [[1, 0, 0], [6, 5, 5]])


def test_factory_loads_hdf5_positions_velocities_and_timestamps(tmp_path):
    path = tmp_path / "tracked.hdf5"
    with h5py.File(path, "w") as target:
        for timestamp, offset in ((10, 0.0), (20, 1.0)):
            group = target.create_group(str(timestamp))
            group["tid"] = np.array([7, 9], dtype=np.int64)
            group["x"] = np.array([offset, offset + 2.0])
            group["y"] = np.array([0.0, 3.0])
            group["z"] = np.array([1.0, 4.0])
            group["vx"] = np.array([1.0, 1.0])
            group["vy"] = np.array([0.0, 0.0])
            group["vz"] = np.array([0.0, 0.0])

    dataset = DatasetFactory().get_dataset(path)

    assert dataset.trajectories.shape == dataset.velocities.shape == (2, 2, 3)
    np.testing.assert_array_equal(dataset.timestamps, [10, 20])
    np.testing.assert_allclose(dataset.positions_at_time_step(1)[0], [1.0, 0.0, 1.0])


def test_factory_loads_drone_csv_and_records_coordinate_conversion(tmp_path):
    path = tmp_path / "drones.csv"
    path.write_text(
        "Time,Drone00_X,Drone00_Y,Drone00_Z\n"
        "0.0,1.0,2.0,3.0\n"
        "0.5,4.0,5.0,6.0\n",
        encoding="utf-8",
    )

    dataset = DatasetFactory().get_dataset(path)

    np.testing.assert_array_equal(dataset.timestamps, [0.0, 0.5])
    np.testing.assert_allclose(dataset.trajectories[:, 0], [[2, 1, 3], [5, 4, 6]])
    assert "X/Y axes swapped" in dataset.coordinate_system


def test_velocity_strategies_preserve_shape(trajectory_fixture):
    expected, trajectories = trajectory_fixture

    forward = ForwardDifference().calculate(trajectories)
    central = CentralDifference().calculate(trajectories)

    assert forward.shape == central.shape == trajectories.shape
    np.testing.assert_allclose(
        forward[0, 0], expected["first_agent_forward_velocity"]
    )
    np.testing.assert_allclose(forward[-1], forward[-2], equal_nan=True)


def test_npy_loader_rejects_non_trajectory_shape(tmp_path):
    path = tmp_path / "invalid.npy"
    np.save(path, np.zeros((3, 3), dtype=np.float32))

    with pytest.raises(InvalidFileFormatError, match="frames, agents, 3"):
        NpyLoader().load(path)
