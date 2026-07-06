import json
from pathlib import Path

import numpy as np
import pytest

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
