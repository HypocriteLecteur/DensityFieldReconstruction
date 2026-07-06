"""High-level evaluation workflow for in-memory or saved reconstructions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from dfr.artifacts import OutputConfig, RunArtifacts
from dfr.config import EvaluationConfig
from dfr.data.base import Dataset
from dfr.evaluation.metrics import (
    automatic_evaluation_bounds,
    compute_density_overlap_masses,
)
from dfr.evaluation.results import EvaluationRun, EvaluationSummary, FrameEvaluation
from dfr.reconstruction.results import ReconstructionRun


@dataclass(frozen=True, slots=True)
class _EvaluationInput:
    dataset_name: str
    frame: int
    positions: np.ndarray
    means: np.ndarray
    radii: np.ndarray
    weights: np.ndarray
    scale: float


def evaluate(
    reconstruction: ReconstructionRun | str | Path,
    *,
    ground_truth: Optional[Dataset] = None,
    config: Optional[EvaluationConfig] = None,
    output: Optional[OutputConfig] = None,
) -> EvaluationRun:
    """Evaluate reconstructed densities against point-density ground truth.

    ``reconstruction`` may be an in-memory :class:`ReconstructionRun` or its
    managed run directory. Saved reconstruction arrays contain the source
    positions, so ``ground_truth`` is optional; pass a dataset to explicitly
    reload the requested frames. No files are written unless ``output`` is set.
    """
    selected_config = config or EvaluationConfig()
    if output is not None and output.workflow != "evaluation":
        raise ValueError("Evaluation output workflow must be 'evaluation'.")
    inputs, source_description = _inputs(reconstruction)
    artifacts = (
        RunArtifacts.create(
            output,
            resolved_config={
                "evaluation": selected_config,
                "source": source_description,
            },
            device=selected_config.device,
            metadata={"entrypoint": "dfr.evaluate"},
        )
        if output is not None
        else None
    )
    results = []
    for item in inputs:
        truth = (
            ground_truth.positions_at_time_step(item.frame)
            if ground_truth is not None
            else item.positions
        )
        bounds_array = (
            np.asarray(selected_config.bounds, dtype=np.float64)
            if selected_config.bounds is not None
            else automatic_evaluation_bounds(truth, item.scale)
        )
        true_positive, false_positive, false_negative = (
            compute_density_overlap_masses(
                truth,
                item.scale,
                item.means,
                item.weights,
                item.radii,
                bounds=bounds_array,
                voxel_resolution=selected_config.voxel_resolution,
                batch_size=selected_config.batch_size,
                device=selected_config.device,
            )
        )
        summary = EvaluationSummary(
            true_positive_mass=true_positive,
            false_positive_mass=false_positive,
            false_negative_mass=false_negative,
            ground_truth_mass=float(len(truth)),
            predicted_mass=float(np.sum(item.weights)),
        )
        result = FrameEvaluation(
            dataset_name=item.dataset_name,
            frame=item.frame,
            summary=summary,
            bounds=tuple(tuple(float(value) for value in axis) for axis in bounds_array),
            voxel_resolution=selected_config.voxel_resolution,
        )
        results.append(result)
        if artifacts is not None:
            artifacts.save_json(
                f"frame_{item.frame:06d}.json",
                result.to_dict(),
                category="metrics",
                overwrite=output.resume,
            )
    run = EvaluationRun(tuple(results), selected_config, artifacts)
    if artifacts is not None:
        artifacts.save_json(
            "summary.json",
            run.summary.to_dict(),
            category="metrics",
            overwrite=output.resume,
        )
    return run


def _inputs(
    reconstruction: ReconstructionRun | str | Path,
) -> tuple[tuple[_EvaluationInput, ...], dict]:
    if isinstance(reconstruction, ReconstructionRun):
        return (
            tuple(
                _EvaluationInput(
                    dataset_name=frame.dataset_name,
                    frame=frame.frame,
                    positions=frame.positions,
                    means=frame.means,
                    radii=frame.radii,
                    weights=frame.weights,
                    scale=frame.scale,
                )
                for frame in reconstruction.frames
            ),
            {
                "kind": "in_memory",
                "reconstruction_run": (
                    str(reconstruction.run_dir) if reconstruction.run_dir else None
                ),
            },
        )
    run_dir = Path(reconstruction).expanduser().resolve()
    loaded = _load_managed_inputs(run_dir)
    return loaded, {"kind": "managed_run", "reconstruction_run": str(run_dir)}


def _load_managed_inputs(run_dir: Path) -> tuple[_EvaluationInput, ...]:
    data_dir = run_dir / "data"
    metrics_dir = run_dir / "metrics"
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file() or not data_dir.is_dir():
        raise ValueError(f"Not a managed reconstruction run directory: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("workflow") != "reconstruction":
        raise ValueError(f"Managed run is not a reconstruction workflow: {run_dir}")
    single = data_dir / "reconstruction.npz"
    paths = (
        [single]
        if single.is_file()
        else sorted(data_dir.glob("frame_*/reconstruction.npz"))
    )
    if not paths:
        raise FileNotFoundError(f"No reconstruction arrays found under: {data_dir}")
    loaded = []
    for path in paths:
        prefix = "" if path == single else f"{path.parent.name}/"
        summary_path = metrics_dir / prefix / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Reconstruction summary is missing: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        with np.load(path, allow_pickle=False) as arrays:
            loaded.append(
                _EvaluationInput(
                    dataset_name=str(summary["dataset"]),
                    frame=int(summary["frame"]),
                    positions=arrays["positions"],
                    means=arrays["means"],
                    radii=arrays["radii"],
                    weights=arrays["weights"],
                    scale=float(arrays["scale"]),
                )
            )
    return tuple(loaded)
