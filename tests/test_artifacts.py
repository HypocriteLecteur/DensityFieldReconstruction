import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dfr import OutputConfig, RunArtifacts
from dfr.artifacts import ARTIFACT_CATEGORIES, to_serializable, warn_legacy_output
from dfr.config import TrainingParams


NOW = datetime(2026, 7, 6, 8, 9, 10, 123456, tzinfo=timezone.utc)


def output_config(tmp_path, **overrides):
    values = {
        "workflow": "analysis",
        "name": "mode sweep",
        "root": tmp_path / "outputs",
        "run_id": "demo-run",
    }
    values.update(overrides)
    return OutputConfig(**values)


def test_create_writes_layout_manifest_and_resolved_config(tmp_path):
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
        lr_max_steps=20,
    )
    artifacts = RunArtifacts.create(
        output_config(tmp_path),
        resolved_config={
            "dataset": Path("dataset/demo.npy"),
            "scales": np.array([0.5, 1.0]),
            "training": training,
        },
        device=torch.device("cuda:0"),
        metadata={"seed": np.int64(123)},
        git_commit="abc123",
        now=NOW,
    )

    assert artifacts.run_dir == (
        tmp_path / "outputs" / "analysis" / "demo-run"
    ).resolve()
    assert all(artifacts.directory(name).is_dir() for name in ARTIFACT_CATEGORIES)
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(artifacts.config_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["created_at_utc"] == "2026-07-06T08:09:10.123456Z"
    assert manifest["git_commit"] == "abc123"
    assert manifest["device"] == "cuda:0"
    assert manifest["metadata"] == {"seed": 123}
    assert config["schema_version"] == 1
    assert config["output"]["root"] == str((tmp_path / "outputs").resolve())
    assert config["experiment"]["scales"] == [0.5, 1.0]
    assert config["experiment"]["training"]["lr_max_steps"] == 20


def test_artifact_writers_use_categories_and_explicit_overwrite(tmp_path):
    artifacts = RunArtifacts.create(
        output_config(tmp_path), resolved_config={"dataset": "demo"}, now=NOW
    )

    json_path = artifacts.save_json(
        "summary.json", {"value": np.float32(1.25)}, category="metrics"
    )
    npz_path = artifacts.save_npz(
        "surface.npz", scale=np.array([1.0, 2.0]), score=np.array([0.8, 0.9])
    )
    checkpoint_path = artifacts.save_checkpoint(
        "model.pth", {"weights": torch.tensor([1.0, 2.0])}
    )
    figure, axis = plt.subplots()
    axis.plot([0, 1], [1, 0])
    figure_path = artifacts.save_figure("curve.png", figure, dpi=40)
    plt.close(figure)

    assert json.loads(json_path.read_text(encoding="utf-8")) == {"value": 1.25}
    with np.load(npz_path) as values:
        np.testing.assert_allclose(values["score"], [0.8, 0.9])
    restored = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    torch.testing.assert_close(restored["weights"], torch.tensor([1.0, 2.0]))
    assert figure_path.is_file() and figure_path.stat().st_size > 0

    with pytest.raises(FileExistsError, match="overwrite=True"):
        artifacts.save_json("summary.json", {"value": 2}, category="metrics")
    artifacts.save_json(
        "summary.json", {"value": 2}, category="metrics", overwrite=True
    )
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"value": 2}


def test_existing_run_requires_resume_or_overwrite(tmp_path):
    config = {"dataset": "demo", "frames": [0]}
    first = RunArtifacts.create(output_config(tmp_path), resolved_config=config, now=NOW)
    sentinel = first.data_dir / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="resume=True"):
        RunArtifacts.create(output_config(tmp_path), resolved_config=config, now=NOW)

    resumed = RunArtifacts.create(
        output_config(tmp_path, resume=True),
        resolved_config=config,
        now=datetime(2026, 7, 7, tzinfo=timezone.utc),
    )
    assert sentinel.exists()
    assert resumed.manifest["resume_count"] == 1
    assert resumed.manifest["last_resumed_at_utc"] == "2026-07-07T00:00:00Z"

    with pytest.raises(ValueError, match="configuration differs"):
        RunArtifacts.create(
            output_config(tmp_path, resume=True),
            resolved_config={"dataset": "different"},
            now=NOW,
        )

    overwritten = RunArtifacts.create(
        output_config(tmp_path, overwrite=True),
        resolved_config={"dataset": "different"},
        now=NOW,
    )
    assert not sentinel.exists()
    assert overwritten.manifest["resume_count"] == 0


def test_generated_run_id_and_config_validation(tmp_path):
    generated = OutputConfig(
        workflow="analysis",
        name="Scale Sweep / Demo",
        root=tmp_path,
    ).resolved_run_id(NOW)
    assert generated == "20260706T080910123456Z-Scale-Sweep-Demo"

    with pytest.raises(ValueError, match="one path-safe segment"):
        OutputConfig(workflow="../analysis", name="bad", root=tmp_path)
    with pytest.raises(ValueError, match="mutually exclusive"):
        output_config(tmp_path, resume=True, overwrite=True)
    with pytest.raises(ValueError, match="requires an explicit run_id"):
        OutputConfig(
            workflow="analysis", name="bad resume", root=tmp_path, resume=True
        )
    with pytest.raises(TypeError, match="cannot be serialized"):
        to_serializable(object())


def test_artifact_paths_cannot_escape_managed_directory(tmp_path):
    artifacts = RunArtifacts.create(output_config(tmp_path), now=NOW)

    with pytest.raises(ValueError, match="escapes managed root"):
        artifacts.path("data", "../manifest.json")
    with pytest.raises(ValueError, match="must be relative"):
        artifacts.path("data", tmp_path / "outside.json")
    with pytest.raises(ValueError, match="Unknown artifact category"):
        artifacts.path("unknown", "value.json")


def test_legacy_output_helper_warns_only_for_legacy_locations():
    with pytest.warns(FutureWarning, match="Legacy output path"):
        warn_legacy_output(Path("results") / "analysis.json")
    with pytest.warns(FutureWarning, match="Legacy output path"):
        warn_legacy_output(Path("scenarios") / "demo" / "logs" / "run")
    with warnings_suppressed():
        warn_legacy_output(Path("outputs") / "analysis" / "run")


class warnings_suppressed:
    def __enter__(self):
        import warnings

        self.catch = warnings.catch_warnings()
        self.records = self.catch.__enter__()
        warnings.simplefilter("error")
        return self.records

    def __exit__(self, *args):
        return self.catch.__exit__(*args)
