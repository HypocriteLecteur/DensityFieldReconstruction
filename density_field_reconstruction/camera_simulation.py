import numpy as np
import torch
from scipy.spatial.transform import Rotation
from gaussian_rasterizer_simple_large import rasterize_gaussians
import time

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("Warning: CuPy not found. CUDA circle rendering will not work.")

CIRCLE_RENDER_KERNEL = r'''
extern "C" __global__
void render_kernel(const float* points_u, const float* points_v, const float* sigmas, float* image, 
                   int num_points, int height, int width, float sigma_multiple) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute point index and pixel coordinates
    int max_pixels_per_point = (int)(2 * sigma_multiple * 30 + 1); // Assume max sigma=30 for bounding box
    int point_idx = idx / (max_pixels_per_point * max_pixels_per_point);
    if (point_idx >= num_points) return;
    
    int local_idx = idx % (max_pixels_per_point * max_pixels_per_point);
    int local_x = local_idx % max_pixels_per_point;
    int local_y = local_idx / max_pixels_per_point;
    
    // Get point data
    float u = points_u[point_idx];
    float v = points_v[point_idx];
    float sigma = sigmas[point_idx];
    
    // Compute bounding box
    int box_size = (int)(sigma_multiple * sigma);
    int u_min = max(0, (int)(u - box_size));
    int u_max = min(width, (int)(u + box_size + 1));
    int v_min = max(0, (int)(v - box_size));
    int v_max = min(height, (int)(v + box_size + 1));
    
    // Map local coordinates to global pixel coordinates
    int pixel_x = u_min + local_x;
    int pixel_y = v_min + local_y;
    
    // Check if pixel is within bounds
    if (pixel_x >= u_min && pixel_x < u_max && pixel_y >= v_min && pixel_y < v_max) {
        float dx = pixel_x - u;
        float dy = pixel_y - v;
        float dist_sq = dx * dx + dy * dy;
        float intensity = exp(-dist_sq / (2 * sigma * sigma));
       
        atomicAdd(&image[pixel_y * width + pixel_x], intensity);
    }
}
'''

class RenderStrategy:
    """Namespace for different rendering implementations."""
    
    @staticmethod
    def gaussian_rasterizer(camera, swarm_positions, scale=None):
        """
        Renders using the imported rasterize_gaussians function.
        """
        N = swarm_positions.shape[0]
        
        # Get Camera Extrinsics (R, T)
        P1 = camera.get_view_matrix_numpy()
        P1_torch = torch.tensor(P1, dtype=torch.float, device='cuda')
        R1 = P1_torch[:, :3].contiguous()
        T1 = P1_torch[:, 3].contiguous()

        # Prepare Swarm Data
        positions_torch = torch.tensor(swarm_positions, dtype=torch.float32).cuda()
        
        if scale is not None:
            simulated_scale = np.sqrt(camera.size**2 + scale**2)
        else:
            simulated_scale = camera.size

        # Call External Rasterizer
        image = rasterize_gaussians(
            positions_torch,
            torch.ones((N, 1), dtype=torch.float).cuda() * simulated_scale,
            torch.ones((N, 1), dtype=torch.float).cuda(),
            R1,
            T1,
            torch.tensor(camera.intrinsics_params, dtype=torch.float).cuda(),
            camera.H,
            camera.W,
            False
        )

        points_2d, depth = camera.project_world_to_image(swarm_positions)
        
        # Return standard format: (2D Projections, Image)
        # Note: Gaussian rasterizer does projection internally, so we return None for 2D points 
        # unless we want to calculate them redundantly.
        return points_2d, image

    @staticmethod
    def cuda_circles(camera, swarm_positions, scale=None, sigma_multiple=3.0):
        """
        Renders using the custom CUDA kernel with circles.
        """
        if not HAS_CUPY:
            raise RuntimeError("CuPy is required for 'cuda_circles' renderer.")

        # 1. Manually project 3D -> 2D using Camera math
        points_2d, depth = camera.project_world_to_image(swarm_positions)
        
        # 2. Calculate radii based on depth
        radii = camera.depth_to_radii(depth)
        
        # 3. Prepare Tensors
        # Convert to torch if they are numpy (project_world_to_image returns numpy)
        points_2d_torch = torch.tensor(points_2d, dtype=torch.float32).cuda()
        radii_torch = torch.tensor(radii, dtype=torch.float32).cuda()
        
        height, width = camera.H, camera.W
        
        # 4. Run Kernel
        # Ensure inputs are on CUDA device and contiguous
        points_u_cp = cp.asarray(points_2d_torch[:, 0].contiguous(), dtype=cp.float32)
        points_v_cp = cp.asarray(points_2d_torch[:, 1].contiguous(), dtype=cp.float32)
        sigmas_cp = cp.asarray(radii_torch.contiguous(), dtype=cp.float32)
        image_cp = cp.zeros((height, width), dtype=cp.float32)
        
        render_kernel = cp.RawKernel(CIRCLE_RENDER_KERNEL, 'render_kernel')
        
        # Estimate max bbox
        max_pixels_per_point = int(2 * sigma_multiple * 30 + 1)
        num_points = points_2d_torch.shape[0]
        total_tasks = num_points * max_pixels_per_point * max_pixels_per_point
        
        block_size = 256
        grid_size = (total_tasks + block_size - 1) // block_size
        
        render_kernel((grid_size,), (block_size,), 
                      (points_u_cp, points_v_cp, sigmas_cp, image_cp, 
                       num_points, height, width, cp.float32(sigma_multiple)))
        
        # Transfer back to Torch
        image = torch.as_tensor(image_cp, device='cuda')
        image = (image * 255).clamp(0, 255).byte()
        
        return points_2d, image


class Camera:
    def __init__(self, intrinsics_params, initpose, near_clip, far_clip, size, H=1000, W=1000, name="cam"):
        self.name = name
        self.intrinsics_params = intrinsics_params
        self.pose = initpose.astype(np.float32) # [x, y, z, qx, qy, qz, qw]
        self.near_clip = near_clip
        self.far_clip = far_clip
        self.size = size
        self.H = H
        self.W = W
        self.last_render_time = 0

    def get_view_matrix_numpy(self):
        """Calculates the 3x4 World-to-Camera matrix."""
        rot = Rotation.from_quat(self.pose[3:]).as_matrix().T
        t = -rot @ self.pose[:3]
        
        # Basis conversion (Blender/ROS style to standard CV style if needed, 
        # or just standardizing orientation)
        base2cam = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        return base2cam @ np.hstack((rot, t.reshape((3, 1))))

    def aim_at_location(self, target_position):
        """
        Updates self.pose orientation to look at target_position.
        """
        cam_pos = self.pose[:3]
        xb = target_position - cam_pos
        norm = np.linalg.norm(xb)
        if norm < 1e-6:
            return # Avoid division by zero
            
        xb = xb / norm
        yb = np.cross([0, 0, 1], xb)
        zb = np.cross(xb, yb)
        
        # Create rotation matrix [xb, yb, zb]
        mat = np.array([xb, yb, zb]).T
        quat = Rotation.from_matrix(mat).as_quat()
        
        self.pose[3:] = quat

    def project_world_to_image(self, world_positions):
        """
        Projects 3D world coordinates to 2D image coordinates.
        Returns: (points_2d, depths)
        """
        # 1. Transform World -> Base Frame
        rot = Rotation.from_quat(self.pose[3:]).as_matrix().T
        positions_base = (rot @ (world_positions.T - self.pose[:3].reshape((3, 1)))).T

        # 2. Transform Base -> Camera Frame (Coordinate shuffle)
        positions_cam = np.zeros_like(positions_base)
        positions_cam[:, 0] = -positions_base[:, 1]
        positions_cam[:, 1] = -positions_base[:, 2]
        positions_cam[:, 2] = positions_base[:, 0]

        # 3. Culling
        if self.far_clip is not None and self.near_clip is not None:
            mask = (positions_cam[:, 2] >= self.near_clip) & (positions_cam[:, 2] <= self.far_clip)
            # Filter points
            positions_valid = positions_cam[mask]
        else:
            positions_valid = positions_cam
            mask = np.ones(positions_cam.shape[0], dtype=bool)

        depth = positions_valid[:, 2]

        # 4. Projection (Camera -> Image Plane)
        # Apply intrinsics: [u, v, w] = K @ [X, Y, Z]
        projected = (self.intrinsics_params @ positions_valid.T).T
        
        # Perspective divide
        points_2d = projected[:, :2] / projected[:, 2].reshape((-1, 1))

        return points_2d, depth

    def depth_to_radii(self, depth):
        """Calculates screen-space radius based on depth."""
        # radius = world_size / depth * focal_length
        return self.size / depth * self.intrinsics_params[0, 0]

    def simulate_view(self, swarm_positions, renderer_type='gaussian', scale=None):
        """
        Generic simulation method for a single camera.
        
        Args:
            renderer_type (str): 'gaussian' or 'cuda_circles'
        """
        start = time.perf_counter()
        
        if renderer_type == 'gaussian':
            proj_2d, image = RenderStrategy.gaussian_rasterizer(self, swarm_positions, scale)
        elif renderer_type == 'cuda_circles':
            proj_2d, image = RenderStrategy.cuda_circles(self, swarm_positions, scale)
        else:
            raise ValueError(f"Unknown renderer type: {renderer_type}")
            
        end = time.perf_counter()
        self.last_render_time = (end - start) * 1000
        
        return self.pose, proj_2d, image

class MultiCameraSystem:
    def __init__(self, cameras_list):
        """
        Args:
            cameras_list: A list of Camera objects.
        """
        self.cameras = cameras_list

    @classmethod
    def create_stereo(cls, intrinsics, H, W, pose1, pose2, near_clip, far_clip, size):
        """Factory method to create a standard Stereo setup."""
        cam1 = Camera(intrinsics, pose1, near_clip, far_clip, size, H, W, name="left")
        cam2 = Camera(intrinsics, pose2, near_clip, far_clip, size, H, W, name="right")
        return cls([cam1, cam2])

    def aim_all_at_swarm(self, swarm_positions):
        """Aims all cameras at the center of the swarm."""
        center = np.mean(swarm_positions, axis=0)
        for cam in self.cameras:
            cam.aim_at_location(center)

    def simulate_vision(self, swarm_positions, renderer='gaussian', is_auto_aim=True, scale=None):
        """
        Simulates vision for ALL cameras in the system.
        
        Returns:
            poses: List of camera poses.
            projections: List of 2D point arrays (one per camera).
            images: List of rendered images (one per camera).
        """
        if is_auto_aim:
            self.aim_all_at_swarm(swarm_positions)

        poses = []
        projections = []
        images = []

        for cam in self.cameras:
            pose, proj, img = cam.simulate_view(swarm_positions, renderer_type=renderer, scale=scale)
            poses.append(pose)
            projections.append(proj)
            images.append(img)

        return poses, projections, images