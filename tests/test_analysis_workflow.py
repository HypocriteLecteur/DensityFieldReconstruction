import numpy as np
import pytest

import dfr
from dfr import AnalysisConfig, load_dataset
from dfr.analysis import ScaleAnalysisResult
import dfr.workflows as workflows


def make_dataset(tmp_path):
    offsets = np.arange(6, dtype=np.float32)[:, None] * np.array(
        [[0.01, 0.0, 0.0]]
    )
    points = np.concatenate((offsets, offsets + np.array([[5.0, 5.0, 0.0]])))[None]
    path = tmp_path / "facade.npy"
    np.save(path, points)
    return load_dataset(path)


def test_analyze_facade_runs_explicit_cpu_mode_curve(tmp_path):
    dataset = make_dataset(tmp_path)

    result = dfr.analyze(
        dataset,
        kind="modes",
        config=AnalysisConfig(frames=(0,), scales=(0.2, 10.0), device="cpu"),
        tolerance=1e-5,
    )

    np.testing.assert_array_equal(result.mode_counts, [2, 1])


def test_analyze_facade_constructs_and_dispatches_dra(monkeypatch, tmp_path):
    dataset = make_dataset(tmp_path)
    received = {}

    def fake_compute(positions, result, *, batch_size):
        received["shape"] = positions.shape
        received["batch_size"] = batch_size
        result.dra[:] = 0.75
        return result

    monkeypatch.setattr(workflows, "compute_scale_model_order_surface", fake_compute)
    result = dfr.analyze(
        dataset,
        kind="dra",
        frames=0,
        scales=(0.5, 1.0),
        model_order_steps=2,
        batch_size=123,
    )

    assert isinstance(result, ScaleAnalysisResult)
    assert received == {"shape": (12, 3), "batch_size": 123}
    np.testing.assert_array_equal(result.dra, np.full((2, 2), 0.75))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "unknown", "frames": 0, "scales": (1.0,)}, "kind must"),
        ({"kind": "modes", "frames": 0}, "scales are required"),
        ({"kind": "modes", "frames": (0, 0), "scales": (1.0,)}, "exactly one"),
    ],
)
def test_analyze_facade_rejects_ambiguous_requests(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        dfr.analyze(make_dataset(tmp_path), **kwargs)
