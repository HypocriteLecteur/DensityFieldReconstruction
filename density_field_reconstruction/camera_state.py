import numpy as np
import torch
from scipy.spatial.transform import Rotation

class CameraState:
    def __init__(self, cam_id: int, intrinsics_params: np.ndarray, pose: np.ndarray, device: torch.device):
        self.cam_id = cam_id
        self.intrinsics_params = intrinsics_params
        self.pose_np = pose # [x, y, z, qx, qy, qz, qw]
        self.P_np = CameraState.wrd_to_cam(pose) # (3, 4) extrinsic matrix
        
        # Torch tensors on GPU
        P_torch = torch.tensor(self.P_np, dtype=torch.float, device=device)
        self.R = P_torch[:, :3].contiguous() # (3, 3) Rotation
        self.T = P_torch[:, 3].contiguous() # (3) Translation
        self.K = torch.tensor(intrinsics_params, dtype=torch.float, device=device).contiguous() # (3, 3) Intrinsics
    
    @staticmethod
    def wrd_to_cam(pose):
        """
        Convert a world pose to a camera pose in the base frame.
        Input:
        - pose: A numpy array of shape (7,) representing the pose in the format [x, y, z, qx, qy, qz, qw].
        Output:
        - A numpy array of shape (3, 4) representing the camera pose in the base frame.
        """
        rot = Rotation.from_quat(pose[3:]).as_matrix().T
        t = -rot @ pose[:3]

        base2cam = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])

        return base2cam @ np.hstack((rot, t.reshape((3, 1))))

class CameraStateUE4:
    def __init__(self, cam_id: int, W: int, H: int, intrinsics_params: np.ndarray, R: np.ndarray, T: np.ndarray, device: torch.device):
        self.cam_id = cam_id
        self.intrinsics_params = intrinsics_params
        self.T_world = T
        self.P_np = CameraStateUE4.wrd_to_cam(R, T)

        # Torch tensors on GPU
        P_torch = torch.tensor(self.P_np, dtype=torch.float, device=device)
        self.R = P_torch[:, :3].contiguous() # (3, 3) Rotation
        self.T = P_torch[:, 3].contiguous() # (3) Translation
        self.K = torch.tensor(intrinsics_params, dtype=torch.float, device=device).contiguous() # (3, 3) Intrinsics

        self.W = W
        self.H = H

    @staticmethod
    def wrd_to_cam(R, T):
        """ Convert a world pose to a camera pose in the base frame."""

        # P_camera = R⋅(P_world + t)
        T_prime = R @ T

        # UE4 left-handed frame
        base2cam = np.array([
            [0, 1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])

        return base2cam @ np.hstack((R, T_prime.reshape((3, 1))))