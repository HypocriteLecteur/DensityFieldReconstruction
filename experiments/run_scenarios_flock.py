"""Managed reconstruction for measured two-camera flock detections.

This is intentionally a specialized external-observation workflow: supplied
2D detections and calibrated, asymmetric camera poses cannot be substituted
with the simulated-projection scenario runner.  Historical visualization,
baseline, timing, and scenario-log tools live in local Git history.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
import scipy.io
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from dfr.artifacts import OutputConfig
from dfr.camera_state import CameraState
from dfr.camera_system import MultiCameraSystem
from dfr.config import ReconstructionParams, TrainingParams
from dfr.data.registry import default_project_root
from dfr.mode_finding import mode_counting
from dfr.reconstruction.observations import ExternalObservationFrame, reconstruct_observations


DEFAULT_DATASET_NAME = "Point3D_N68_t2.35_Xianjiahu_20231121b_data50"
DEFAULT_START_STEP = 0
DEFAULT_STEP_LENGTH = 5
TARGET_MODE_COUNT = 10


@dataclass(frozen=True, slots=True)
class FlockInputConfig:
    """External data, camera calibration, and optional scale cache inputs."""

    data_root: Path
    extrinsics_json: Path
    detections_camera_1: Path
    detections_camera_2: Path
    project_root: Path = Path.cwd()
    scale_cache: Path | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "data_root",
            "extrinsics_json",
            "detections_camera_1",
            "detections_camera_2",
            "project_root",
        ):
            value = Path(getattr(self, field_name)).expanduser().resolve()
            object.__setattr__(self, field_name, value)
        if self.scale_cache is not None:
            object.__setattr__(
                self, "scale_cache", Path(self.scale_cache).expanduser().resolve()
            )
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"Flock data directory does not exist: {self.data_root}")
        for path in (
            self.extrinsics_json,
            self.detections_camera_1,
            self.detections_camera_2,
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Flock input file does not exist: {path}")
        if self.scale_cache is not None and not self.scale_cache.is_file():
            raise FileNotFoundError(f"Flock scale cache does not exist: {self.scale_cache}")


def find_target_scale(func, target_mode_count, s_low=0.0, s_high=30.0):
    """Binary-search a scale that produces the requested mode count."""
    for _ in range(100):
        mid = (s_low + s_high) / 2.0
        value = func(mid)
        if value == target_mode_count:
            return mid
        if value > target_mode_count:
            s_low = mid
        else:
            s_high = mid
    return (s_low + s_high) / 2.0


def load_camera_extrinsics(extrinsics_json_path: Path):
    """Convert the three-camera calibration JSON into MATLAB-style transforms."""
    with Path(extrinsics_json_path).open(encoding="utf-8") as stream:
        cam_data = json.load(stream)

    def value(key_path: str):
        item = cam_data
        for key in key_path.split("."):
            item = item[key]
        return item

    def transform(euler_deg, translation):
        rotation = R.from_euler(
            "ZYX", [euler_deg[2], euler_deg[1], euler_deg[0]], degrees=True
        )
        result = np.eye(4)
        result[:3, :3] = rotation.as_matrix()
        result[:3, 3] = translation
        return result

    cam1 = transform([value("CAM1.sensor_X_dir") - 90, value("CAM1.sensor_Y_dir"), 0], [0, 0, 0])

    angle2 = abs(value("CAM1.neg_x_axis_dir") + 180 - value("CAM1.CAM1_to_CAM2_dir"))
    azimuth2 = np.deg2rad(angle2)
    heading2 = angle2 % 360 + abs(value("CAM2.neg_x_axis_dir") - value("CAM2.CAM2_to_CAM1_dir")) % 360
    baseline2 = value("CAM1_CAM2_baseline")
    cam2 = transform(
        [value("CAM2.sensor_X_dir") - 90, value("CAM2.sensor_Y_dir"), heading2],
        [baseline2 * np.cos(azimuth2), baseline2 * np.sin(azimuth2), 0],
    )

    angle3 = abs(value("CAM1.neg_x_axis_dir") + 180 - value("CAM1.CAM1_to_CAM3_dir"))
    azimuth3 = np.deg2rad(angle3)
    heading3 = angle3 % 360 + abs(value("CAM3.neg_x_axis_dir") - value("CAM3.CAM3_to_CAM1_dir")) % 360
    baseline3 = value("CAM1_CAM3_baseline")
    cam3 = transform(
        [value("CAM3.sensor_X_dir") - 90, value("CAM3.sensor_Y_dir"), heading3],
        [baseline3 * np.cos(azimuth3), baseline3 * np.sin(azimuth3), 0],
    )
    return cam1, cam2, cam3


def convert_matlab_transforms_to_poses(transforms):
    """Convert MATLAB transforms to DFR ``[x, y, z, qx, qy, qz, qw]`` poses."""
    base_to_camera = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    poses = []
    for transform in transforms:
        rotation_world = transform[:3, :3] @ base_to_camera
        poses.append(np.concatenate([transform[:3, 3], R.from_matrix(rotation_world).as_quat()]))
    return poses


def _select_frame_scales(scales, frames):
    """Select scales from either per-frame or selected-order cache arrays."""
    values = list(scales)
    selected_frames = [int(frame) for frame in frames]
    if not selected_frames:
        return []
    if len(values) > max(selected_frames):
        return [float(values[frame]) for frame in selected_frames]
    if len(values) >= len(selected_frames):
        return [float(values[index]) for index, _ in enumerate(selected_frames)]
    raise ValueError(f"Scale cache has {len(values)} entries for {len(selected_frames)} frames.")


def _compute_scales(trajectories: np.ndarray) -> list[float]:
    """Compute unsaved ground-truth scales when no explicit cache is supplied."""
    scales = []
    for positions in tqdm(trajectories, desc="Computing flock ground-truth scales"):
        points = torch.from_numpy(positions).cuda().float()
        distances = torch.cdist(points, points) + torch.eye(points.shape[0], device="cuda") * 1e10
        nearest = torch.median(torch.min(distances, dim=1).values).item()
        count = lambda scale: mode_counting(
            points, points.clone(), scale, max_iter=2000, tol=nearest * 5e-4
        )
        scale = find_target_scale(count, TARGET_MODE_COUNT, 0, 15)
        if count(scale) != TARGET_MODE_COUNT:
            raise ValueError("Unable to find a flock scale with the requested mode count.")
        scales.append(scale)
    return scales


def _load_or_compute_scales(inputs: FlockInputConfig, trajectories: np.ndarray):
    if inputs.scale_cache is None:
        return _compute_scales(trajectories), True
    cached = np.load(inputs.scale_cache)
    return cached["scales_gt"].tolist(), False


def run_flock_scenario(
    inputs: FlockInputConfig,
    *,
    output: OutputConfig | None = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
    start_step: int = DEFAULT_START_STEP,
    stop_step: int | None = None,
    step_length: int = DEFAULT_STEP_LENGTH,
):
    """Reconstruct calibrated flock detections with optional managed artifacts."""
    csv1 = pd.read_csv(inputs.detections_camera_1)
    csv2 = pd.read_csv(inputs.detections_camera_2)
    images1, images2 = csv1["Img Name"].unique(), csv2["Img Name"].unique()
    trajectories = scipy.io.loadmat(inputs.data_root / f"{dataset_name}.mat")["xyzTensorValid"]
    scales, computed_scales = _load_or_compute_scales(inputs, trajectories)

    end = min(stop_step or trajectories.shape[0], trajectories.shape[0])
    if start_step >= end or step_length < 1:
        raise ValueError("Flock start/stop/step select no valid frames.")
    selected_steps = range(start_step, end, step_length)
    transform1, transform2, _ = load_camera_extrinsics(inputs.extrinsics_json)
    camera_poses = convert_matlab_transforms_to_poses([transform1, transform2])
    intrinsics1 = np.array([[3328.2389, 0, 1858.5952], [0, 3362.5043, 1037.2734], [0, 0, 1]])
    intrinsics2 = np.array([[3392.0252, 0, 2364.9331], [0, 3402.5383, 1021.1155], [0, 0, 1]])
    distortion1, distortion2 = np.array([0.1266, 0.0674, 0, 0]), np.array([-0.3591, 0.8730, 0, 0])
    cameras = MultiCameraSystem.create_homogeneous_system(
        state_class=CameraState, intrinsics=intrinsics1, H=2160, W=3840,
        poses_or_RTs=camera_poses, near_clip=1, far_clip=100, size=1, device="cuda"
    )
    cameras.cameras[1].state.intrinsics_params = intrinsics2
    cameras.cameras[1].state.K = torch.tensor(intrinsics2, dtype=torch.float, device="cuda").contiguous()

    observations = []
    for frame in tqdm(selected_steps, desc=f"Processing {dataset_name}"):
        def detections(table, images, intrinsics, distortion):
            selected = table[table["Img Name"] == images[frame]][["cx", "cy"]].to_numpy()
            return cv2.undistortPoints(selected.reshape(-1, 1, 2).astype(np.float32), intrinsics, distortion, P=intrinsics).reshape(-1, 2)

        positions = trajectories[frame]
        poses, _, _, masks = cameras.simulate_vision(positions, renderer="projection_only", is_auto_aim=False)
        observations.append(ExternalObservationFrame(
            dataset_name=dataset_name, frame=frame, positions=positions,
            projections=[detections(csv1, images1, intrinsics1, distortion1), detections(csv2, images2, intrinsics2, distortion2)],
            camera_system=cameras, camera_poses=poses, visible_mask=np.logical_and.reduce(masks),
            metadata={"source": "measured_flock", "camera_1_image": str(images1[frame]), "camera_2_image": str(images2[frame])},
        ))

    run = reconstruct_observations(
        observations,
        frame_scales=_select_frame_scales(scales, [item.frame for item in observations]),
        training=TrainingParams(0.05, 0.9, 0.05, 0.9, 0.10, 0.7, 1.0, 0.3, 0.5, 100),
        reconstruction=ReconstructionParams(TARGET_MODE_COUNT, 0.5, 0.3, 32, 2 * TARGET_MODE_COUNT),
        use_decoupled=False,
        output=output,
    )
    if computed_scales and run.artifacts is not None:
        run.artifacts.save_npz("computed_ground_truth_scales.npz", category="cache", scales_gt=np.asarray(scales))
    return run


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run managed external-detection flock reconstruction.")
    parser.add_argument("study", choices=("run",))
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--extrinsics-json", type=Path)
    parser.add_argument("--detections-camera-1", type=Path)
    parser.add_argument("--detections-camera-2", type=Path)
    parser.add_argument("--scale-cache", type=Path)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--start-step", type=int, default=DEFAULT_START_STEP)
    parser.add_argument("--stop-step", type=int)
    parser.add_argument("--step-length", type=int, default=DEFAULT_STEP_LENGTH)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-id")
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--resume", action="store_true")
    policy.add_argument("--overwrite-run", action="store_true")
    parser.add_argument("--no-output", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    required = {
        "--data-root": args.data_root,
        "--extrinsics-json": args.extrinsics_json,
        "--detections-camera-1": args.detections_camera_1,
        "--detections-camera-2": args.detections_camera_2,
    }
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        raise ValueError(f"The flock run requires: {', '.join(missing)}")
    inputs = FlockInputConfig(
        data_root=args.data_root, extrinsics_json=args.extrinsics_json,
        detections_camera_1=args.detections_camera_1, detections_camera_2=args.detections_camera_2,
        project_root=args.project_root, scale_cache=args.scale_cache,
    )
    output = None if args.no_output else OutputConfig(
        workflow="reconstruction", name=f"flock-{args.dataset_name}", root=args.output_root,
        run_id=args.run_id, project_root=args.project_root, resume=args.resume, overwrite=args.overwrite_run,
    )
    run_flock_scenario(inputs, output=output, dataset_name=args.dataset_name,
                       start_step=args.start_step, stop_step=args.stop_step, step_length=args.step_length)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
