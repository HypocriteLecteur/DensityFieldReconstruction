from pathlib import Path

from examples.toy_workflow import main


def test_toy_workflow_example_runs(tmp_path):
    target = main(tmp_path / "toy_workflow")

    assert target == tmp_path / "toy_workflow" / "mode_curve.png"
    assert target.is_file()
    assert (tmp_path / "toy_workflow" / "dataset" / "toy.npy").is_file()
    assert (
        tmp_path / "toy_workflow" / "scenarios" / "toy" / "config.yaml"
    ).is_file()
