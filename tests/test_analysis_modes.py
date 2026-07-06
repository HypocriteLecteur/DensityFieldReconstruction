import numpy as np
import pytest

from dfr import load_dataset
from dfr.analysis import analyze_dataset_modes, compute_mode_curve, count_modes


POINTS = np.array(
    [[0.0, 0.0], [0.02, 0.0], [5.0, 5.0], [5.02, 5.0]], dtype=np.float32
)


def test_count_modes_and_curve_on_cpu():
    assert count_modes(POINTS, 0.2, device="cpu", tolerance=1e-5) == 2
    curve = compute_mode_curve(
        POINTS,
        [0.2, 10.0],
        dataset_name="synthetic",
        frame=3,
        device="cpu",
        tolerance=1e-5,
    )

    np.testing.assert_array_equal(curve.mode_counts, [2, 1])
    assert curve.dataset_name == "synthetic" and curve.frame == 3


def test_dataset_mode_analysis_uses_dataset_metadata(tmp_path):
    project = tmp_path / "project"
    data_dir = project / "dataset"
    scenario_dir = project / "scenarios" / "tiny"
    data_dir.mkdir(parents=True)
    scenario_dir.mkdir(parents=True)
    points_3d = np.column_stack((POINTS, np.zeros(len(POINTS), dtype=np.float32)))
    np.save(data_dir / "tiny.npy", points_3d[None, ...])
    (scenario_dir / "config.yaml").write_text(
        "data_file: dataset/tiny.npy\n", encoding="utf-8"
    )
    dataset = load_dataset("tiny", project_root=project)

    curve = analyze_dataset_modes(
        dataset, 0, [0.2, 10.0], device="cpu", tolerance=1e-5
    )

    assert curve.dataset_name == "tiny" and curve.frame == 0
    np.testing.assert_array_equal(curve.mode_counts, [2, 1])


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: count_modes(POINTS, 0, device="cpu"), "scale must be positive"),
        (
            lambda: count_modes(np.zeros((2, 1)), 1, device="cpu"),
            "positions must be",
        ),
        (
            lambda: compute_mode_curve(POINTS, [1.0, 0.5], device="cpu"),
            "strictly increasing",
        ),
    ],
)
def test_mode_analysis_validation(call, message):
    with pytest.raises(ValueError, match=message):
        call()
