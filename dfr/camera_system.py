import numpy as np
import torch
from scipy.spatial.transform import Rotation
from dfr.camera_state import CameraState
from dfr.rendering import RenderStrategy, convolution_cupy_wrapper, select_rasterizer
import time
    
class Camera:
    def __init__(self, state: CameraState, size: float, name="cam"):
        """
        Takes a CameraState object as long as it exposes:
        P_np, K, R, T, H, W, near_clip, far_clip, and aim_at_location()
        """
        self.state = state
        self.size = size
        self.name = name

    def aim_at_location(self, target_position):
        self.state.aim_at_location(target_position)

    def project_world_to_image(self, world_positions):
        """Agnostic projection utilizing the state's P_np matrix."""
        # 1. World -> Camera Frame using P_np (3x4)
        N = world_positions.shape[0]
        world_homo = np.hstack((world_positions, np.ones((N, 1))))
        positions_cam = (self.state.P_np @ world_homo.T).T  # Shape: (N, 3)

        # 2. Projection (Camera -> Image Plane)
        projected = (self.state.intrinsics_params @ positions_cam.T).T
        points_2d = projected[:, :2] / projected[:, 2].reshape((-1, 1))

        # 3. Culling
        if self.state.far_clip is not None and self.state.near_clip is not None:
            mask_dist = (positions_cam[:, 2] >= self.state.near_clip) & (positions_cam[:, 2] <= self.state.far_clip)
            mask_hw = (points_2d[:, 0] >= 0) & (points_2d[:, 0] <= self.state.W) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] <= self.state.H)
            mask = mask_dist & mask_hw
        else:
            mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] <= self.state.W) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] <= self.state.H)

        return points_2d[mask], positions_cam[mask, 2], mask

    def depth_to_radii(self, depth):
        return self.size / depth * self.state.intrinsics_params[0, 0]

    def simulate_view(self, swarm_positions, renderer_type='gaussian', scale=None):
        if renderer_type == 'gaussian':
            proj_2d, image, mask = RenderStrategy.gaussian_rasterizer(self, swarm_positions, scale)
        elif renderer_type == 'cuda_circles':
            proj_2d, image, mask = RenderStrategy.cuda_circles(self, swarm_positions, scale)
        elif renderer_type == 'projection_only':
            proj_2d, image, mask = RenderStrategy.projection_only(self, swarm_positions, scale)
        else:
            raise ValueError(f"Unknown renderer type: {renderer_type}")
        
        return proj_2d, image, mask

class MultiCameraSystem:
    def __init__(self, cameras_list):
        """
        Args:
            cameras_list: A list of Camera objects.
        """
        self.cameras = cameras_list

    # @classmethod
    # def create_system(cls, intrinsics, H, W, poses, near_clip, far_clip, size):
    #     cams = [Camera(intrinsics, poses[i], near_clip, far_clip, size, H, W, name=f"cam_{i}") for i in range(poses.shape[0])]
    #     return cls(cams)

    @classmethod
    def create_homogeneous_system(cls, state_class, intrinsics, H, W, poses_or_RTs, near_clip, far_clip, size, device):
        """
        Factory method to create a system where all cameras use the SAME state type.
        - state_class: Either CameraState or CameraStateUE4
        - poses_or_RTs: A list of poses (for standard) or tuples of (R, T) (for UE4)
        """
        cams = []
        for i, pose_data in enumerate(poses_or_RTs):
            # 1. Initialize the specific mathematical state
            if state_class.__name__ == "CameraStateUE4":
                R, T = pose_data
                state = state_class(i, W, H, near_clip, far_clip, intrinsics, R, T, device)
            else:
                state = state_class(i, W, H, near_clip, far_clip, intrinsics, pose_data, device)
            
            # 2. Inject the state into the generic Camera wrapper
            cam = Camera(state=state, size=size, name=f"{state_class.__name__}_{i}")
            cams.append(cam)
            
        return cls(cams)
    
    def aim_all_at_swarm(self, swarm_positions):
        """Aims all cameras at the center of the swarm, regardless of their state type."""
        center = np.mean(swarm_positions, axis=0)
        for cam in self.cameras:
            cam.aim_at_location(center)

    def simulate_vision(self, swarm_positions, renderer='gaussian', is_auto_aim=True, scale=None):
        """Simulates vision for ALL cameras in the system."""
        if is_auto_aim:
            self.aim_all_at_swarm(swarm_positions)

        poses, projections, images, masks = [], [], [], []

        for cam in self.cameras:
            proj, img, mask = cam.simulate_view(swarm_positions, renderer_type=renderer, scale=scale)
            
            # --- MODIFICATION HERE ---
            # Safely extract the pose based on the state's available attributes
            if hasattr(cam.state, 'pose_np'):
                poses.append(cam.state.pose_np)
            else:
                # CameraStateUE4 fallback: use the 3x4 Extrinsics matrix
                poses.append(cam.state.P_np) 
                
            projections.append(proj)
            images.append(img)
            masks.append(mask)

        return poses, projections, images, masks