"""
Rendering implementations for projecting 3D points to 2D density images.

Rendering is kept separate from camera geometry (camera_system.py).
Only projection-only rendering works without CUDA; other renderers
fail with clear error messages when dependencies are unavailable.
"""

import numpy as np
import torch

# --- CuPy / CUDA guards ---

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

# --- CUDA kernel code (for CuPy circle rendering) ---

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

    float inv_sigma_sq = 1.0f / (sigma * sigma);
    float norm_factor = 0.5f * 0.318309886184f * 255.0f * inv_sigma_sq;

    for (int pixel_y = v_min; pixel_y < v_max; ++pixel_y) {
        for (int pixel_x = u_min; pixel_x < u_max; ++pixel_x) {
            float dx = pixel_x - u;
            float dy = pixel_y - v;
            float dist_sq = dx * dx + dy * dy;
            float intensity = expf(-dist_sq * 0.5f * inv_sigma_sq) * norm_factor;

            // Atomically add to image
            atomicAdd(&image[pixel_y * width + pixel_x], intensity);
        }
    }
}
'''

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

RENDER_KERNEL = None
RENDER_KERNEL2 = None
if HAS_CUPY:
    RENDER_KERNEL = cp.RawKernel(CIRCLE_RENDER_KERNEL_CODE, 'render_kernel')
    RENDER_KERNEL2 = cp.RawKernel(CIRCLE_RENDER_KERNEL_CODE2, 'render_kernel')


def convolution_cupy_wrapper(points_2d_torch, radius_val, height, width, sigma_multiple=4.0):
    """Render Gaussian circles at 2D points using CuPy CUDA kernel."""
    if not HAS_CUPY:
        raise RuntimeError("CuPy is required for convolution_cupy_wrapper but is not installed.")

    points_2d_torch = points_2d_torch.contiguous()
    num_points = points_2d_torch.shape[0]

    image_torch = torch.zeros((height, width), dtype=torch.float32, device='cuda')
    points_cp = cp.from_dlpack(points_2d_torch)
    image_cp = cp.from_dlpack(image_torch)

    block_size = 256
    grid_size = (num_points + block_size - 1) // block_size

    RENDER_KERNEL2(
        (grid_size,), (block_size,),
        (points_cp, cp.float32(radius_val), image_cp, num_points, height, width, cp.float32(sigma_multiple))
    )

    cp.cuda.Stream.null.synchronize()
    return image_torch


# --- Canonical rasterizer selection ---

def select_rasterizer(name='large'):
    """Return the rasterize_gaussians function for the chosen variant.

    Args:
        name: One of 'small', 'large', 'decoupled'.

    Returns:
        The rasterize_gaussians function from the selected variant.

    Raises:
        RuntimeError: If the rasterizer module cannot be imported.
    """
    module_map = {
        'small': 'gaussian_rasterizer_simple_small',
        'large': 'gaussian_rasterizer_simple_large',
        'decoupled': 'gaussian_rasterizer_simple_small_decoupled',
    }
    if name not in module_map:
        raise ValueError(f"Unknown rasterizer '{name}'. Choose from: {list(module_map.keys())}")

    try:
        mod = __import__(module_map[name], fromlist=['rasterize_gaussians'])
        return mod.rasterize_gaussians
    except ImportError as e:
        raise RuntimeError(
            f"Rasterizer '{name}' ({module_map[name]}) is not available. "
            f"Build it with: cd density_field_rasterizer/{module_map[name]} && python setup.py install"
        ) from e


# --- RenderStrategy (kept for backward compatibility) ---

class RenderStrategy:
    """Namespace for different rendering implementations.

    Each static method takes a camera object, swarm_positions, and optional scale.
    """

    @staticmethod
    def projection_only(camera, swarm_positions, scale=None):
        points_2d, depth, mask = camera.project_world_to_image(swarm_positions)
        return points_2d, None, mask

    @staticmethod
    def cuda_circles(camera, swarm_positions, scale=None, sigma_multiple=4.0):
        if not HAS_CUPY:
            raise RuntimeError("CuPy is required for 'cuda_circles' renderer but is not installed.")

        points_2d, depth, mask = camera.project_world_to_image(swarm_positions)
        radii = camera.depth_to_radii(depth)

        points_2d_torch = torch.tensor(points_2d, dtype=torch.float32).cuda()
        radii_torch = torch.tensor(radii, dtype=torch.float32).cuda()

        height, width = camera.state.H, camera.state.W
        image = convolution_cupy_wrapper(points_2d_torch, radii_torch, height, width, sigma_multiple)

        return points_2d, image, mask

    @staticmethod
    def gaussian_rasterizer(camera, swarm_positions, scale=None):
        rasterize_gaussians = select_rasterizer('large')
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
