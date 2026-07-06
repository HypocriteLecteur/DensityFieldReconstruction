"""Small end-to-end CUDA reconstruction smoke test."""

import numpy as np
import pytest
import torch


pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA is unavailable", allow_module_level=True)

pytest.importorskip(
    "gaussian_rasterizer_simple_large",
    reason="The large rasterizer extension is unavailable",
)

from dfr.camera_state import CameraState
from dfr.camera_system import MultiCameraSystem
from dfr.config import ReconstructionParams, TrainingParams
from dfr.density_field_reconstructor import DensityReconstructor
from experiments.reconstruct_one_frame import build_parser, run


def test_tiny_two_camera_reconstruction():
    torch.manual_seed(12345)
    height = width = 64
    intrinsics = np.array(
        [[60.0, 0.0, 31.5], [0.0, 60.0, 31.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    poses = np.array(
        [
            [-10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, -10.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    positions = np.array(
        [
            [-0.5, -0.5, -0.25],
            [0.5, -0.5, 0.25],
            [-0.5, 0.5, 0.25],
            [0.5, 0.5, -0.25],
        ],
        dtype=np.float32,
    )
    cameras = MultiCameraSystem.create_homogeneous_system(
        state_class=CameraState,
        intrinsics=intrinsics,
        H=height,
        W=width,
        poses_or_RTs=poses,
        near_clip=1.0,
        far_clip=30.0,
        size=0.2,
        device="cuda",
    )
    _, projections, images, _ = cameras.simulate_vision(
        positions,
        renderer="gaussian",
        is_auto_aim=True,
        scale=0.35,
    )
    train_params = TrainingParams(
        xyz_lr_c=0.01,
        xyz_lr_final_c=0.001,
        radius_lr_c=0.01,
        radius_lr_final_c=0.001,
        weights_lr_c=0.01,
        weights_lr_final_c=0.001,
        xyz_reg=0.0,
        radius_reg=0.0,
        radius_cutoff_inv=10.0,
        lr_max_steps=1,
    )
    reconstruction_params = ReconstructionParams(
        targetd_num_mode=2,
        voxel_scale=0.5,
        voxel_peak_threshold=0.05,
        voxel_grid_max_size=12,
        voxel_peaks_number=8,
    )

    reconstructor = DensityReconstructor(
        max_iter=1,
        W=width,
        H=height,
        far_clip=30.0,
    )
    models, scale_spaces = reconstructor.process_frame(
        cameras,
        images=images,
        point_sets=projections,
        is_adaptive_scale=False,
        scale=0.5,
        positions=positions,
        train_params=train_params,
        reconstruction_params=reconstruction_params,
    )

    assert len(models) == 1
    assert 0 < models[0].num_gaussians <= reconstruction_params.voxel_peaks_number
    assert len(scale_spaces) == 2
    assert all(space.shape == (1, height, width) for space in scale_spaces)
    assert torch.isfinite(models[0]._xyz).all()
    assert torch.isfinite(models[0]._radius).all()
    assert torch.isfinite(models[0]._weights).all()


def test_one_frame_cli_writes_managed_artifacts(tmp_path):
    project = tmp_path / "project"
    data_dir = project / "dataset"
    scenario_dir = project / "scenarios" / "tiny"
    data_dir.mkdir(parents=True)
    scenario_dir.mkdir(parents=True)
    positions = np.array(
        [
            [-0.5, -0.5, -0.25],
            [0.5, -0.5, 0.25],
            [-0.5, 0.5, 0.25],
            [0.5, 0.5, -0.25],
        ],
        dtype=np.float32,
    )
    np.save(data_dir / "tiny.npy", positions[None, ...])
    (scenario_dir / "config.yaml").write_text(
        "data_file: dataset/tiny.npy\n"
        "cam_poses:\n"
        "  - [-10, 0, 0, 0, 0, 0, 1]\n"
        "  - [0, -10, 0, 0, 0, 0, 1]\n"
        "intrinsics_params: [[60, 0, 31.5], [0, 60, 31.5], [0, 0, 1]]\n"
        "H: 64\nW: 64\nnear_clip: 1\nfar_clip: 30\niter: 1\n"
        "size: 0.2\nsave_video: false\nfps: 30\ndpi: 100\n",
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--dataset",
            "tiny",
            "--frame",
            "0",
            "--iterations",
            "1",
            "--scale",
            "0.5",
            "--voxel-grid-max-size",
            "12",
            "--voxel-peaks-number",
            "8",
            "--voxel-peak-threshold",
            "0.05",
            "--project-root",
            str(project),
            "--output-root",
            "outputs",
            "--run-id",
            "cli-smoke",
        ]
    )

    artifacts = run(args)

    assert artifacts.run_dir == (
        project / "outputs" / "reconstruction" / "cli-smoke"
    ).resolve()
    assert artifacts.manifest_path.is_file()
    assert artifacts.config_path.is_file()
    assert (artifacts.data_dir / "reconstruction.npz").is_file()
    assert (artifacts.checkpoints_dir / "final_model.pth").is_file()
    assert (artifacts.metrics_dir / "summary.json").is_file()
