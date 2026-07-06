"""
Shared utilities for experiment runner scripts.

Extract duplicated logging, metrics, and camera setup to reduce
code duplication across run_scenarios*.py scripts.
"""

import logging
import sys
from pathlib import Path
import numpy as np

from dfr.simulation_config import SimulationConfig
from dfr import CameraConfig, load_dataset, resolve_dataset
from dfr.evaluation import EvaluationSummary
from dfr.reconstruction import build_camera_system


def setup_logger(name: str, log_file: str = 'run_experiments.log') -> logging.Logger:
    """Configure and return a logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def load_scenario(scenario_name: str, scenario_path: str):
    """Load a scenario's config and dataset."""
    project_root = Path(scenario_path).resolve().parents[1]
    spec = resolve_dataset(scenario_name, project_root=project_root)
    config = SimulationConfig(str(spec.config_path))
    dataset = load_dataset(spec)
    return config, dataset


def setup_camera_system(dataset, step_range, config, cam_num: int, device='cuda'):
    """Compatibility adapter for the package encircling-camera builder.

    Older runners initialized quaternions as ``[1, 0, 0, 0]`` before calling
    ``simulate_vision`` with auto-aim. Auto-aim replaces those orientations, so
    the package's valid identity quaternion produces identical observations.
    """
    return build_camera_system(
        dataset,
        tuple(step_range),
        config,
        CameraConfig.encircling(count=cam_num, device=device),
    )


def print_global_metrics(label: str, metric_data: dict) -> str:
    """Compute and format global TP/FP/FN/dMOTA metrics as a LaTeX table row."""
    sum_tp = np.sum(metric_data['tp'])
    sum_fp = np.sum(metric_data['fp'])
    sum_fn = np.sum(metric_data['fn'])
    total_N = np.sum(metric_data['N'])
    total_weights = np.sum(metric_data['w'])

    if total_N <= 0:
        global_recall = global_hallucination = global_dmota = 0.0
    else:
        summary = EvaluationSummary(
            float(sum_tp),
            float(sum_fp),
            float(sum_fn),
            float(total_N),
            float(total_weights),
        )
        global_recall = summary.recall
        global_hallucination = summary.hallucination
        global_dmota = summary.dmota

    return f"{global_recall:.3f} & {global_hallucination:.3f} & {global_dmota:.3f} &"


def _apply_projection_noise(projections, cam_system, noise_std: float):
    """Add bounded Gaussian noise to 2D projections in-place."""
    for cam_idx in range(len(projections)):
        max_w = cam_system.cameras[cam_idx].state.W
        max_h = cam_system.cameras[cam_idx].state.H
        new_projections = projections[cam_idx].copy()
        needs_noise = np.ones(new_projections.shape[0], dtype=bool)
        while np.any(needs_noise):
            num_needs = np.sum(needs_noise)
            noise = np.random.normal(0, noise_std, size=(num_needs, 2))
            candidate_proj = projections[cam_idx][needs_noise] + noise
            in_bounds = (
                (candidate_proj[:, 0] >= 0) & (candidate_proj[:, 0] <= max_w) &
                (candidate_proj[:, 1] >= 0) & (candidate_proj[:, 1] <= max_h)
            )
            valid_indices = np.where(needs_noise)[0][in_bounds]
            new_projections[valid_indices] = candidate_proj[in_bounds]
            needs_noise[valid_indices] = False
        projections[cam_idx] = new_projections
