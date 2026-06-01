import numpy as np
import torch
from scipy.spatial.transform import Rotation

class CameraState:
    def __init__(self, cam_id: int, W: int, H: int, near_clip: float, far_clip: float, intrinsics_params: np.ndarray, pose: np.ndarray, device: torch.device):
        self.cam_id = cam_id
        self.W = W
        self.H = H
        self.near_clip = near_clip
        self.far_clip = far_clip
        self.intrinsics_params = intrinsics_params
        self.device = device

        # Cache intrinsics tensor once
        self.K = torch.tensor(intrinsics_params, dtype=torch.float, device=device).contiguous()
        
        # Initialize pose and build cached extrinsics
        self.update_pose(pose)
    
    @property
    def camera_center(self):
        """Returns the true world location of the camera."""
        return self.T_world

    def update_pose(self, new_pose: np.ndarray):
        """Updates the internal pose and synchronizes all downstream matrices and GPU tensors."""
        self.pose_np = new_pose.astype(np.float32) # [x, y, z, qx, qy, qz, qw]
        self.T_world = self.pose_np[:3]
        self.P_np = self.wrd_to_cam(self.pose_np)
        
        # Rebuild GPU tensors
        P_torch = torch.tensor(self.P_np, dtype=torch.float, device=self.device)
        self.R = P_torch[:, :3].contiguous()
        self.T = P_torch[:, 3].contiguous()
        self.P = P_torch.contiguous()

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
    
    def aim_at_location(self, target_position):
        """Updates orientation to look at target_position and syncs GPU tensors."""
        cam_pos = self.T_world
        xb = target_position - cam_pos
        norm = np.linalg.norm(xb)
        if norm < 1e-6:
            return 
            
        xb = xb / norm
        yb = np.cross([0, 0, 1], xb)
        zb = np.cross(xb, yb)
        
        mat = np.array([xb, yb, zb]).T
        quat = Rotation.from_matrix(mat).as_quat()
        
        # Update state using the parent method to keep CPU/GPU synced
        new_pose = self.pose_np.copy()
        new_pose[3:] = quat
        self.update_pose(new_pose)
    
    def get_local_frustum(self):
        """Calculates the 8 corners of the frustum in local camera coordinates."""
        intrinsics = self.intrinsics_params
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # 1. Image corners in pixel coordinates
        corners_pixel = np.array([
            [0, 0], [self.W, 0], [self.W, self.H], [0, self.H]
        ])
        
        # 2. Normalized image coordinates
        corners_norm = (corners_pixel - np.array([cx, cy])) / np.array([fx, fy])

        # 3. Vectorized calculation for near and far planes
        z_vals = np.array([self.near_clip, self.far_clip]).reshape(2, 1) # (2, 1)
        
        # Multiply (2, 1) by (4, 2) utilizing broadcasting to get (2, 4, 2) XY coordinates
        xy_vals = z_vals[..., np.newaxis] * corners_norm[np.newaxis, ...] 
        
        # Broadcast Z values to match XY shape, then concatenate
        z_broadcast = np.broadcast_to(z_vals[..., np.newaxis], (2, 4, 1))
        local_vertices = np.concatenate((xy_vals, z_broadcast), axis=-1)
        
        return local_vertices.reshape(8, 3)

    def get_world_frustum(self):
        """Transforms the local frustum vertices to global world coordinates."""
        local_vertices = self.get_local_frustum()
        
        # Extract Rotation and Translation from the state's Extrinsic matrix
        R_cam = self.P_np[:, :3]
        T_cam = self.P_np[:, 3]

        # Math: X_world = R^T * (X_cam - T)
        # We transpose the result to return an (8, 3) array
        world_vertices = (R_cam.T @ (local_vertices - T_cam).T).T
        
        return world_vertices

class CameraStateUE4(CameraState):
    def __init__(self, cam_id: int, W: int, H: int, near_clip: float, far_clip: float, intrinsics_params: np.ndarray, 
                 R: np.ndarray, T: np.ndarray, device: torch.device):
            self.cam_id = cam_id
            self.W = W
            self.H = H
            self.near_clip = near_clip
            self.far_clip = far_clip
            self.intrinsics_params = intrinsics_params
            self.device = device
            
            self.K = torch.tensor(intrinsics_params, dtype=torch.float, device=device).contiguous()

            self.update_pose(R, T)

    @property
    def camera_center(self):
        """Returns the true world location of the camera."""
        return self.T_world

    def update_pose(self, R: np.ndarray, T: np.ndarray):
        """Updates internal numpy arrays and regenerates GPU tensors."""
        # input R and T: P_w = RP_b + T
        # convert to P_b = R^T P_w - R^T T
        self.R_world = R
        self.T_world = T
        self.P_np = self.wrd_to_cam(R, T)

        # Update Torch tensors on GPU
        P_torch = torch.tensor(self.P_np, dtype=torch.float, device=self.device)
        self.R = P_torch[:, :3].contiguous() 
        self.T = P_torch[:, 3].contiguous()
        self.P = P_torch.contiguous()

    @staticmethod
    def wrd_to_cam(R, T):
        """ Convert a world pose to a camera pose in the base frame."""

        base2cam = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])

        return base2cam @ np.hstack((R.T, -R.T @ T.reshape((3, 1))))
        # return np.hstack((R, T_prime.reshape((3, 1))))

    def aim_at_location(self, target_position):
        """Updates orientation to look at target_position and syncs GPU tensors for UE4."""
        
        # 1. Extract true camera position 
        # (Because wrd_to_cam does T_prime = R @ T, self.T_world is mathematically -C)
        cam_pos = -self.T_world
        
        xb = target_position - cam_pos
        norm = np.linalg.norm(xb)
        if norm < 1e-6:
            return 
            
        xb = xb / norm
        yb = np.cross([0, 0, 1], xb)
        
        # Safety check: if looking straight up or down, the cross product is zero
        norm_yb = np.linalg.norm(yb)
        if norm_yb < 1e-6:
            yb = np.array([0, 1, 0]) # Fallback right vector
        else:
            yb = yb / norm_yb
            
        zb = np.cross(xb, yb)
        
        # 2. Build World-to-Camera rotation matrix directly
        # Stacking as rows inherently creates the inverse/transpose of the Camera-to-World matrix
        new_R = np.array([xb, yb, zb])
        
        # 3. Update state (Translation doesn't change since the camera didn't move)
        self.update_pose(new_R, self.T_world)