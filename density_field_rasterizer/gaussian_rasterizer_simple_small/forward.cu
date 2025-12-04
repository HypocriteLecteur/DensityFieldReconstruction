#include "config.h"
#include <cooperative_groups.h>
#include <glm/vec3.hpp>               // vec3, bvec3, dvec3, ivec3 and uvec3
#include <glm/vec4.hpp>               // vec4, bvec4, dvec4, ivec4 and uvec4
#include <glm/mat2x2.hpp>             // mat2, dmat2
#include <glm/mat2x3.hpp>             // mat2x3, dmat2x3
#include <glm/mat3x3.hpp>             // mat3, dmat3
#include <glm/matrix.hpp>             // all the GLSL matrix functions: transpose, inverse, etc.
#include <stdio.h>
#include <inttypes.h>
namespace cg = cooperative_groups;

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ float3 transformPoint(const float3& p, const float* R, const float* T)
{
	float3 transformed = {
		R[0] * p.x + R[1] * p.y + R[2] * p.z + T[0],
		R[3] * p.x + R[4] * p.y + R[5] * p.z + T[1],
		R[6] * p.x + R[7] * p.y + R[8] * p.z + T[2],
	};
	return transformed;
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, const float* intrinsics, const float radius)
{
    const float fx = intrinsics[0];
    const float fy = intrinsics[4];

	glm::mat2x3 J = glm::mat2x3(
		fx / mean.z, 0.0f, -(fx * mean.x) / (mean.z * mean.z),
		0.0f, fy / mean.z, -(fy * mean.y) / (mean.z * mean.z));

	glm::mat2 cov = radius * radius * glm::transpose(J) * J;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

__global__ void preprocessCUDA(int P,
    const float3* __restrict__ gmm_mean_pt,
    const float* __restrict__ gmm_radius_pt,
    const float* __restrict__ gmm_weights_pt,
    const float* __restrict__ R_pt,
    const float* __restrict__ T_pt,
    const float* __restrict__ intrinsics_params_pt,
    int H,
    int W,
    float2* __restrict__ means2D,
    float4* __restrict__ invcov2D,
    int* __restrict__ radii,
    const dim3 grid)
{
    // Declare shared memory for constant data
    __shared__ float R_shared[9];
    __shared__ float T_shared[3];
    __shared__ float intrinsics_shared[9]; // Assuming a full 3x3 intrinsic matrix

    // Load data into shared memory using threads in the block
    if (threadIdx.x < 9) {
        R_shared[threadIdx.x] = R_pt[threadIdx.x];
        if (threadIdx.x < 3) {
            T_shared[threadIdx.x] = T_pt[threadIdx.x];
        }
        intrinsics_shared[threadIdx.x] = intrinsics_params_pt[threadIdx.x];
    }
    __syncthreads(); // Ensure all threads wait until loading is complete

    auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

    // Initialize touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
    radii[idx] = 0;

    float3 mean3d = gmm_mean_pt[idx];
    // printf("Thread %d: mean3d = (%f, %f, %f)\n", idx, mean3d.x, mean3d.y, mean3d.z);

    // transform to camera frame
    float3 mean3d_cam = transformPoint(mean3d, R_shared, T_shared);

    // Compute 2D covariance matrix
    float3 cov = computeCov2D(mean3d_cam, intrinsics_shared, gmm_radius_pt[idx]);

    // invert 2D covariance
    float det = cov.x * cov.z - cov.y * cov.y;
    float det_inv = 1.f / det;
    float4 cov_inv = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv, sqrtf(det_inv)};

    const float fx = intrinsics_shared[0];
    const float fy = intrinsics_shared[4];
    const float cx = intrinsics_shared[2];
    const float cy = intrinsics_shared[5];
    means2D[idx].x = fx * mean3d_cam.x / mean3d_cam.z + cx;
    means2D[idx].y = fy * mean3d_cam.y / mean3d_cam.z + cy;
    invcov2D[idx] = cov_inv;

    // Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrtf(fmaxf(0.1f, mid * mid - det));
	float lambda2 = mid - sqrtf(fmaxf(0.1f, mid * mid - det));
	int my_radius = int(ceil(3.f * sqrtf(fmaxf(lambda1, lambda2))));
	uint2 rect_min, rect_max;
	getRect(means2D[idx], my_radius, rect_min, rect_max, grid);

	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

    radii[idx] = my_radius;
    return;
}

// __global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
// renderCUDA(const dim3 grid, int P,
//            float* __restrict__ density_estim_pt,
//            int H, int W,
//            const float2* __restrict__ means2D,
//            const float4* __restrict__ invcov2D,
//            const int* __restrict__ radii,
//            const float* __restrict__ gmm_weights_pt)
// {
//     // Identify current tile and associated min/max pixel range.
//     auto block = cg::this_thread_block();
//     const uint32_t thread_rank = block.thread_rank();
//     const uint32_t lane_id = thread_rank % 32;

//     const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
//     const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
//     const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
//     const uint32_t pix_id = W * pix.y + pix.x;
//     const float2 pixf = { (float)pix.x, (float)pix.y };
//     // Check if this thread is associated with a valid pixel or outside.
//     const bool inside = pix.x < W && pix.y < H;

//     uint2 rect_min, rect_max;

//     // --- Shared Memory Declarations ---
//     __shared__ int collected_radii[BLOCK_SIZE];
//     __shared__ float collected_weights[BLOCK_SIZE];
//     __shared__ float2 collected_xy[BLOCK_SIZE];
//     __shared__ float4 collected_conic_weights[BLOCK_SIZE];

//     const int rounds = ((P + BLOCK_SIZE - 1) / BLOCK_SIZE);
//     int toDo = P;

//     float density = 0.0f;
//     if (inside) {
//         for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
//         {
//             block.sync();
//             const int progress = i * BLOCK_SIZE + block.thread_rank();
//             if (progress < P)
//             {
//                 const int coll_id = progress;
//                 collected_radii[block.thread_rank()] = radii[coll_id];
//                 collected_weights[block.thread_rank()] = gmm_weights_pt[coll_id];
//                 collected_xy[block.thread_rank()] = means2D[coll_id];
//                 collected_conic_weights[block.thread_rank()] = invcov2D[coll_id];
//             }
//             block.sync();
//             const int num_gaussians_in_batch = min(BLOCK_SIZE, toDo);
//             for (int j = 0; j < num_gaussians_in_batch; j++)
//             {
//                 const float2 xy = collected_xy[j];
//                 const float4 con_o = collected_conic_weights[j];
//                 const int radius = collected_radii[j];
//                 const float gmm_weight = collected_weights[j];
//                 const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
//                 getRect(xy, radius, rect_min, rect_max, grid);
//                 if (rect_max.x < blockIdx.x || 
//                     rect_min.x > blockIdx.x || 
//                     rect_max.y < blockIdx.y || 
//                     rect_min.y > blockIdx.y)
//                 {
//                     continue;
//                 }
//                 float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
//                 if (power < -9.0f) { // -9.0f or -12.0f are common cutoffs
//                     continue;
//                 }
//                 density += con_o.w * __expf(power) * gmm_weight;
//             }
//         }
//         density *= 0.5f * 0.318309886184f * 255.0f;
//         density_estim_pt[pix_id] = density;
//     }
// }

__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(const dim3 grid, int P,
           float* __restrict__ density_estim_pt,
           int H, int W,
           const float2* __restrict__ means2D,
           const float4* __restrict__ invcov2D,
           const int* __restrict__ radii,
           const float* __restrict__ gmm_weights_pt)
{
    // Identify current tile and associated min/max pixel range.
    auto block = cg::this_thread_block();
    const uint32_t thread_rank = block.thread_rank();
    const uint32_t lane_id = thread_rank % 32;

    const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    const uint32_t pix_id = W * pix.y + pix.x;
    const float2 pixf = { (float)pix.x, (float)pix.y };
    // Check if this thread is associated with a valid pixel or outside.
    const bool inside = pix.x < W && pix.y < H;

    // Render pixels (sum contributions, no filtering or sorting)
    float density = 0.0f;
    if (inside) {
        for (int i = 0; i < P; i++) {
            const float2 xy = means2D[i];
            // const int radius = radii[i];
            const float4 con_o = invcov2D[i];
            const float gmm_weight = gmm_weights_pt[i];
            const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
            // if (fabs(d.x) > radius || fabs(d.y) > radius) continue;
            const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
            density += con_o.w * __expf(power) * gmm_weight;
        }
        density *= 0.5f * 0.318309886184f * 255.0f;
        density_estim_pt[pix_id] = density;
    }
}

void preprocessCUDAWrapper(
    const int P,
    const float3* gmm_mean_pt,
    const float* gmm_radius_pt,
    const float* gmm_weights_pt,
    const float* R_pt,
    const float* T_pt,
    const float* intrinsics_params_pt,
    const int H,
    const int W,
    float2* means2D,
    float4* invcov2D,
    int* radii,
    const dim3 grid)
{
	preprocessCUDA<<<(P + 255) / 256, 256>>> (
        P,
        gmm_mean_pt,
        gmm_radius_pt,
        gmm_weights_pt,
        R_pt,
        T_pt,
        intrinsics_params_pt,
        H,
        W,
        means2D,
        invcov2D,
        radii,
        grid
    );
}

void renderCUDAWrapper(
    const dim3 tile_grid,
    const dim3 block,
    int P,
    float* density_estim_pt,
	int H, int W,
	const float2* means2D,
	const float4* invcov2D,
    const int* radii,
    const float* gmm_weights_pt)
{
	renderCUDA<<<tile_grid, block>>> (
        tile_grid,
        P,
        density_estim_pt,
		H,
        W,
        means2D,
        invcov2D,
        radii,
        gmm_weights_pt);
}