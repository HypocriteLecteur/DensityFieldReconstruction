import numpy as np
import torch

from dfr.camera_state import CameraState
from dfr.camera_system import Camera


def make_camera():
    intrinsics = np.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    state = CameraState(
        cam_id=0,
        W=100,
        H=100,
        near_clip=1.0,
        far_clip=20.0,
        intrinsics_params=intrinsics,
        pose=pose,
        device=torch.device("cpu"),
    )
    return Camera(state=state, size=1.0)


def test_projection_and_culling_on_cpu():
    camera = make_camera()
    world_positions = np.array(
        [
            [10.0, 0.0, 0.0],
            [10.0, -1.0, 0.0],
            [10.0, 0.0, -1.0],
            [0.5, 0.0, 0.0],
            [10.0, -10.0, 0.0],
        ],
        dtype=np.float32,
    )

    points, depth, mask = camera.project_world_to_image(world_positions)

    np.testing.assert_allclose(points, [[50.0, 50.0], [60.0, 50.0], [50.0, 60.0]])
    np.testing.assert_allclose(depth, [10.0, 10.0, 10.0])
    np.testing.assert_array_equal(mask, [True, True, True, False, False])


def test_projection_only_renderer_has_no_cuda_dependency():
    camera = make_camera()
    points, image, mask = camera.simulate_view(
        np.array([[10.0, 0.0, 0.0]], dtype=np.float32),
        renderer_type="projection_only",
    )

    np.testing.assert_allclose(points, [[50.0, 50.0]])
    assert image is None
    np.testing.assert_array_equal(mask, [True])
    assert camera.state.get_local_frustum().shape == (8, 3)
