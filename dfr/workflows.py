"""Small high-level workflow facades for common DFR tasks."""

from __future__ import annotations

from typing import Literal, Optional

from dfr.analysis.dra import create_scale_analysis, compute_scale_model_order_surface
from dfr.analysis.modes import analyze_dataset_modes
from dfr.analysis.results import ModeCurveResult, ScaleAnalysisResult
from dfr.config import AnalysisConfig
from dfr.data.base import Dataset
from dfr.data.frames import select_frame_indices


def analyze(
    dataset: Dataset,
    *,
    kind: Literal["modes", "dra"] = "modes",
    config: Optional[AnalysisConfig] = None,
    frames=None,
    scales=None,
    max_iter: int = 1000,
    tolerance: float = 1e-2,
    voxel_res_fraction: float = 0.01,
    model_order_steps: int = 10,
    batch_size: int = 200_000,
) -> ModeCurveResult | ScaleAnalysisResult:
    """Analyze one dataset frame with an explicit mode-curve or DRA workflow.

    For ``kind="modes"``, scales are world-space density scales. For
    ``kind="dra"``, scales are multiples of mean nearest-neighbour distance
    and CUDA is required. This function does not save results implicitly.
    """
    if kind not in ("modes", "dra"):
        raise ValueError("kind must be either 'modes' or 'dra'.")
    selected_frames = frames
    selected_scales = scales
    device = None
    if config is not None:
        if frames is not None or scales is not None:
            raise ValueError(
                "Pass frame/scale values either in config or explicitly, not both."
            )
        selected_frames = config.frames
        selected_scales = config.scales
        device = config.device
    frame_ids = select_frame_indices(dataset, selected_frames)
    if len(frame_ids) != 1:
        raise ValueError("analyze currently requires exactly one selected frame.")
    if selected_scales is None:
        raise ValueError("Analysis scales are required.")
    frame = frame_ids[0]
    if kind == "modes":
        return analyze_dataset_modes(
            dataset,
            frame,
            selected_scales,
            device=device,
            max_iter=max_iter,
            tolerance=tolerance,
        )

    positions = dataset.positions_at_time_step(frame)
    dataset_name = str(dataset.metadata.get("dataset_name") or "dataset")
    result = create_scale_analysis(
        dataset_name,
        frame,
        positions,
        selected_scales,
        voxel_res_fraction,
        model_order_steps,
    )
    return compute_scale_model_order_surface(
        positions,
        result,
        batch_size=batch_size,
    )
