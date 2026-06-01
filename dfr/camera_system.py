import numpy as np
import torch
from scipy.spatial.transform import Rotation
from gaussian_rasterizer_simple_large import rasterize_gaussians
from dfr.camera_state import CameraState
import time

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("Warning: CuPy not found. CUDA circle rendering will not work.")

CIRCLE_RENDER_KERNEL_CODE = r'''
extern "C" __global__
void render_kernel(const float2* points, const float* sigmas, float* image, 
                   int num_points, int height, int width, float sigma_multiple) {
    
    // 1 thread processes exactly 1 point
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= num_points) return;
    
    // Get point data using float2
    float u = points[point_idx].x;
    float v = points[point_idx].y;
    float sigma = fmaxf(sigmas[point_idx], 1e-4f);
    
    // Compute bounding box
    int box_size = (int)(sigma_multiple * sigma);
    int u_min = (int)fmaxf(0.0f, u - box_size);
    int u_max = (int)fminf((float)width, u + box_size + 1.0f);
    int v_min = (int)fmaxf(0.0f, v - box_size);
    int v_max = (int)fminf((float)height, v + box_size + 1.0f);
    
    // Pre-compute constant values for the loop
    float inv_sigma_sq = 1.0f / (sigma * sigma);
    float norm_factor = 0.5f * 0.318309886184f * 255.0f * inv_sigma_sq;
    
    // Loop only over the valid pixels for THIS specific point
    for (int pixel_y = v_min; pixel_y < v_max; ++pixel_y) {
        for (int pixel_x = u_min; pixel_x < u_max; ++pixel_x) {
            
            float dx = pixel_x - u;
            float dy = pixel_y - v;
            float dist_sq = dx * dx + dy * dy;
            
            // Calculate intensity using float-optimized math (expf)
            float intensity = expf(-dist_sq * 0.5f * inv_sigma_sq) * norm_factor;
            
            // Atomically add to image
            atomicAdd(&image[pixel_y * width + pixel_x], intensity);
        }
    }
}
'''
RENDER_KERNEL = cp.RawKernel(CIRCLE_RENDER_KERNEL_CODE, 'render_kernel')

CIRCLE_RENDER_KERNEL_CODE2 = r'''
extern "C" __global__
void render_kernel(const float2* points, float sigma, float* image, 
                   int num_points, int height, int width, float sigma_multiple) {
    
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= num_points) return;
    
    float u = points[point_idx].x;
    float v = points[point_idx].y;
    
    // Safety clamp to absolutely prevent divide-by-zero or NaN math
    sigma = fmaxf(sigma, 1e-4f);
    
    int box_size = (int)(sigma_multiple * sigma);
    int u_min = (int)fmaxf(0.0f, u - box_size);
    int u_max = (int)fminf((float)width, u + box_size + 1.0f);
    int v_min = (int)fmaxf(0.0f, v - box_size);
    int v_max = (int)fminf((float)height, v + box_size + 1.0f);
    
    float inv_sigma_sq = 1.0f / (sigma * sigma);
    float norm_factor = 0.5f * 0.318309886184f * 255.0f * inv_sigma_sq;
    
    for (int pixel_y = v_min; pixel_y < v_max; ++pixel_y) {
        for (int pixel_x = u_min; pixel_x < u_max; ++pixel_x) {
            
            float dx = pixel_x - u;
            float dy = pixel_y - v;
            float dist_sq = dx * dx + dy * dy;
            
            float intensity = expf(-dist_sq * 0.5f * inv_sigma_sq) * norm_factor;
            
            atomicAdd(&image[pixel_y * width + pixel_x], intensity);
        }
    }
}
'''
RENDER_KERNEL2 = cp.RawKernel(CIRCLE_RENDER_KERNEL_CODE2, 'render_kernel')

def convolution_cupy_wrapper(points_2d_torch, radius_val, height, width, sigma_multiple=4.0):
    # 1. Guarantee point data is ready and contiguous
    points_2d_torch = points_2d_torch.contiguous()
    
    num_points = points_2d_torch.shape[0]
    
    # 2. Allocate output image safely in PyTorch
    image_torch = torch.zeros((height, width), dtype=torch.float32, device='cuda')
    
    # 3. Share memory via zero-copy DLPack
    points_cp = cp.from_dlpack(points_2d_torch)
    image_cp = cp.from_dlpack(image_torch)
    
    block_size = 256
    grid_size = (num_points + block_size - 1) // block_size
    
    # 4. Launch kernel with scalar float32 for radius
    # PyTorch/CuPy array memory race condition with using radius array
    RENDER_KERNEL2(
        (grid_size,), (block_size,), 
        (points_cp, cp.float32(radius_val), image_cp, num_points, height, width, cp.float32(sigma_multiple))
    )
    
    # 5. Guarantee kernel finishes before returning the tensor
    cp.cuda.Stream.null.synchronize()
    
    return image_torch

class RenderStrategy:
    """Namespace for different rendering implementations."""
    @staticmethod
    def projection_only(camera, swarm_positions, scale=None):
        points_2d, depth, mask = camera.project_world_to_image(swarm_positions)

        return points_2d, None, mask

    @staticmethod
    def cuda_circles(camera, swarm_positions, scale=None, sigma_multiple=4.0):
        """
        Renders using the custom CUDA kernel with circles.
        """
        if not HAS_CUPY:
            raise RuntimeError("CuPy is required for 'cuda_circles' renderer.")

        # 1. Manually project 3D -> 2D using Camera math
        points_2d, depth, mask = camera.project_world_to_image(swarm_positions)
        
        # 2. Calculate radii based on depth
        radii = camera.depth_to_radii(depth)
        
        # 3. Prepare Tensors
        # Convert to torch if they are numpy (project_world_to_image returns numpy)
        points_2d_torch = torch.tensor(points_2d, dtype=torch.float32).cuda()
        radii_torch = torch.tensor(radii, dtype=torch.float32).cuda()
        
        height, width = camera.state.H, camera.state.W
        
        # 4. Run Kernel
        image = convolution_cupy_wrapper(points_2d_torch, radii_torch, height, width, sigma_multiple)
        
        return points_2d, image, mask
    
    @staticmethod
    def gaussian_rasterizer(camera, swarm_positions, scale=None):
        """
        Renders using the imported rasterize_gaussians function.
        """
        points_2d, depth, mask = camera.project_world_to_image(swarm_positions)
        valid_swarm_positions = swarm_positions[mask]
        N = valid_swarm_positions.shape[0]
        
        positions_torch = torch.tensor(valid_swarm_positions, dtype=torch.float32).cuda()
        
        if scale is not None:
            simulated_scale = np.sqrt(camera.size**2 + scale**2)
        else:
            simulated_scale = camera.size
        
        image = rasterize_gaussians(
            positions_torch,
            torch.ones((N, 1), dtype=torch.float, device=camera.state.device) * simulated_scale,
            torch.ones((N, 1), dtype=torch.float, device=camera.state.device),
            camera.state.R,
            camera.state.T,
            camera.state.K,
            camera.state.H,
            camera.state.W,
            False
        )
        
        return points_2d, image, mask
    
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