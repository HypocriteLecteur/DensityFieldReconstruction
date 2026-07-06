"""
Shared utilities for experiment runner scripts.

Extract duplicated logging, metrics, and camera setup to reduce
code duplication across run_scenarios*.py scripts.
"""

import logging
import sys
import os
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from typing import Optional

from dfr.simulation_config import SimulationConfig
from dfr import load_dataset
from dfr.camera_system import MultiCameraSystem
from dfr.density_field_reconstructor import DensityReconstructor
from dfr.density_field_model import GaussianModel
from dfr.camera_state import CameraState
from dfr.utils import calculate_gmm_dissimilarity, generate_encircling_cameras, compute_metrics_batched_torch


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
    config_path = os.path.join(scenario_path, "config.yaml")
    config = SimulationConfig(config_path)
    project_root = Path(scenario_path).resolve().parents[1]
    dataset = load_dataset(config.data_file, project_root=project_root)
    return config, dataset


def setup_camera_system(dataset, step_range, config, cam_num: int, device='cuda'):
    """Generate encircling cameras and return a MultiCameraSystem."""
    if cam_num == 2:
        cam_positions, cam_radius = generate_encircling_cameras(
            dataset, step_range, config.intrinsics_params, config.H, config.W,
            cam_num=4, padding=1
        )
        cam_poses = np.hstack((
            cam_positions[:2],
            np.tile(np.array([1, 0, 0, 0]), (2, 1))
        )).astype(np.float32)
    else:
        cam_positions, cam_radius = generate_encircling_cameras(
            dataset, step_range, config.intrinsics_params, config.H, config.W,
            cam_num=cam_num, padding=1
        )
        cam_poses = np.hstack((
            cam_positions,
            np.tile(np.array([1, 0, 0, 0]), (cam_num, 1))
        )).astype(np.float32)

    return MultiCameraSystem.create_homogeneous_system(
        state_class=CameraState,
        intrinsics=config.intrinsics_params,
        H=config.H, W=config.W,
        poses_or_RTs=cam_poses,
        near_clip=config.near_clip, far_clip=config.far_clip,
        size=config.size,
        device=device
    )


def print_global_metrics(label: str, metric_data: dict) -> str:
    """Compute and format global TP/FP/FN/dMOTA metrics as a LaTeX table row."""
    sum_tp = np.sum(metric_data['tp'])
    sum_fp = np.sum(metric_data['fp'])
    sum_fn = np.sum(metric_data['fn'])
    total_N = np.sum(metric_data['N'])
    total_weights = np.sum(metric_data['w'])

    global_recall = sum_tp / total_N if total_N > 0 else 0.0
    global_hallucination = sum_fp / total_weights if total_weights > 0 else 0.0
    global_dmota = 1.0 - ((sum_fn + sum_fp) / total_N) if total_N > 0 else 0.0

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
