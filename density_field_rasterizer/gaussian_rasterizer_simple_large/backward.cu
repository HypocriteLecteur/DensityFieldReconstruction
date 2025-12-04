#include "config.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Helper function for warp-level reduction
__device__ __forceinline__ float WarpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float2 WarpReduceSum(float2 val) {
    // This can be further optimized by treating float2 as a single 64-bit value (ull)
    // but this is clearer and often optimized well by the compiler.
    val.x = WarpReduceSum(val.x);
    val.y = WarpReduceSum(val.y);
    return val;
}

__device__ __forceinline__ float4 WarpReduceSum(float4 val) {
    val.x = WarpReduceSum(val.x);
    val.y = WarpReduceSum(val.y);
    val.z = WarpReduceSum(val.z);
    val.w = WarpReduceSum(val.w);
    return val;
}

__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
renderbackwardCUDA(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int P,
    int H, int W,
    const float* __restrict__ grad_output,
    const float* __restrict__ gmm_weights,
    const float2* __restrict__ means2D,
    const float4* __restrict__ invcov2D,
    float2* __restrict__ dL_dmean2D,
    float4* __restrict__ dL_dconic2D)
{
    // --- Block and Pixel Info Setup ---
    auto block = cg::this_thread_block();
    const uint32_t block_size = block.size();
    const uint32_t warp_id = block.thread_rank() / 32;
    const uint32_t lane_id = block.thread_rank() % 32;
    // This is the runtime value used for loops and logic
    const uint32_t warps_per_block = block_size / 32;

    const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    const bool inside = pix.x < W && pix.y < H;
    const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

    // Replace the early return with an activity flag
    const bool active = inside && range.x < range.y;

    const uint32_t pix_id = W * pix.y + pix.x;
    const float2 pixf = { (float)pix.x, (float)pix.y };
    const float grad_out_pix = grad_output[pix_id];

    // --- Shared Memory Declarations ---
    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_weights[BLOCK_SIZE];

    __shared__ float2 partial_means[MAX_WARPS_PER_BLOCK];
    __shared__ float4 partial_conics[MAX_WARPS_PER_BLOCK];

    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;

    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
    {
        block.sync();
        const int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y)
        {
            const int coll_id = point_list[range.x + progress];
            collected_id[block.thread_rank()] = coll_id;
            collected_xy[block.thread_rank()] = means2D[coll_id];
            collected_conic_weights[block.thread_rank()] = invcov2D[coll_id];
        }
        block.sync();

        const int num_gaussians_in_batch = min(BLOCK_SIZE, toDo);
        for (int j = 0; j < num_gaussians_in_batch; j++)
        {
            const float2 xy = collected_xy[j];
            const float4 con_o = collected_conic_weights[j];

            float2 grad_mean = {0.f, 0.f};
            float4 grad_conic = {0.f, 0.f, 0.f, 0.f};

            if (active) {
                const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
                const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
                if (power <= 0.0f) {
                    const float g = con_o.w * __expf(power);
                    const float g_grad_out = grad_out_pix * g;
                    if (g_grad_out != 0.f) {
                        const float gdx = g_grad_out * d.x;
                        const float gdy = g_grad_out * d.y;
                        grad_mean = {-gdx * con_o.x - gdy * con_o.y, -gdy * con_o.z - gdx * con_o.y};
                        grad_conic = {gdx * d.x, gdx * d.y, gdy * d.y, g_grad_out};
                    }
                }
            }

            grad_mean = WarpReduceSum(grad_mean);
            grad_conic = WarpReduceSum(grad_conic);

            if (lane_id == 0) {
                partial_means[warp_id] = grad_mean;
                partial_conics[warp_id] = grad_conic;
            }
            block.sync();

            if (block.thread_rank() == 0) {
                float2 final_grad_mean = {0.f, 0.f};
                float4 final_grad_conic = {0.f, 0.f, 0.f, 0.f};
                for(int k = 0; k < warps_per_block; k++) {
                    final_grad_mean.x += partial_means[k].x;
                    final_grad_mean.y += partial_means[k].y;
                    final_grad_conic.x += partial_conics[k].x;
                    final_grad_conic.y += partial_conics[k].y;
                    final_grad_conic.z += partial_conics[k].z;
                    final_grad_conic.w += partial_conics[k].w;
                }

                const int coll_id = collected_id[j];
                atomicAdd(&dL_dmean2D[coll_id].x, final_grad_mean.x);
                atomicAdd(&dL_dmean2D[coll_id].y, final_grad_mean.y);
                atomicAdd(&dL_dconic2D[coll_id].x, final_grad_conic.x);
                atomicAdd(&dL_dconic2D[coll_id].y, final_grad_conic.y);
                atomicAdd(&dL_dconic2D[coll_id].z, final_grad_conic.z);
                atomicAdd(&dL_dconic2D[coll_id].w, final_grad_conic.w);
            }
            block.sync();
        }
    }
}

__global__ void computeGradientCUDA(
    int P,
	const float3* __restrict__ means,
    const float* __restrict__ gmm_radius,
    const float* __restrict__ gmm_weights_pt,
	const float4* __restrict__ invcov2D,
    const float* __restrict__ R,
    const float* __restrict__ T,
    const float* __restrict__ intrinsics_params,
    float2* __restrict__ dL_dmean2D,
    float4* __restrict__ dL_dconics,
    float3* __restrict__ dL_dmean3D,
    float* __restrict__ dL_dradius,
    float* __restrict__ dL_dw)
{
    // Declare shared memory for constant data
    __shared__ float R_shared[9];
    __shared__ float T_shared[3];
    __shared__ float intrinsics_shared[9]; // Assuming a full 3x3 intrinsic matrix

    // Load data into shared memory using threads in the block
    if (threadIdx.x < 9) {
        R_shared[threadIdx.x] = R[threadIdx.x];
        if (threadIdx.x < 3) {
            T_shared[threadIdx.x] = T[threadIdx.x];
        }
        intrinsics_shared[threadIdx.x] = intrinsics_params[threadIdx.x];
    }
    __syncthreads(); // Ensure all threads wait until loading is complete

	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

    // Apply factor to gradients
    float factor = 0.5f * 0.318309886184f * gmm_weights_pt[idx];
    dL_dmean2D[idx].x *= factor;
    dL_dmean2D[idx].y *= factor;
    dL_dconics[idx].x *= 0.5f * factor;
    dL_dconics[idx].y *= 0.5f * factor;
    dL_dconics[idx].z *= 0.5f * factor;
    dL_dw[idx] = 0.5f * 0.318309886184f * dL_dconics[idx].w;
    dL_dconics[idx].w *= 0.5f * factor;

    // Transform mean
    float3 mean = means[idx];
    float3 mean3d_cam = {
        R_shared[0] * mean.x + R_shared[1] * mean.y + R_shared[2] * mean.z + T_shared[0],
        R_shared[3] * mean.x + R_shared[4] * mean.y + R_shared[5] * mean.z + T_shared[1],
        R_shared[6] * mean.x + R_shared[7] * mean.y + R_shared[8] * mean.z + T_shared[2]
    };

    // Compute Jacobian
    float fx = intrinsics_shared[0];
    float fy = intrinsics_shared[4];
    float J[2][3] = {
        {fx / mean3d_cam.z, 0.0f, -(fx * mean3d_cam.x) / (mean3d_cam.z * mean3d_cam.z)},
        {0.0f, fy / mean3d_cam.z, -(fy * mean3d_cam.y) / (mean3d_cam.z * mean3d_cam.z)}
    };

    // Compute dmean2D_dmean3D
    float dmean2D_dmean3D[2][3];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            dmean2D_dmean3D[i][j] = J[i][0] * R[j] + J[i][1] * R[j+3] + J[i][2] * R[j + 6];
        }
    } 

    // Compute dL_dmean3D_left
    float3 dL_dmean3D_left = {
        dL_dmean2D[idx].x * dmean2D_dmean3D[0][0] + dL_dmean2D[idx].y * dmean2D_dmean3D[1][0],
        dL_dmean2D[idx].x * dmean2D_dmean3D[0][1] + dL_dmean2D[idx].y * dmean2D_dmean3D[1][1],
        dL_dmean2D[idx].x * dmean2D_dmean3D[0][2] + dL_dmean2D[idx].y * dmean2D_dmean3D[1][2]
    };

    // Compute matRMN
    float r2 = gmm_radius[idx] * gmm_radius[idx];
    float matRMN[3][2]{
        {r2 * fx / mean3d_cam.z, 0.0f},
        {0.0f, r2 * fy / mean3d_cam.z},
        {-r2*(fx * mean3d_cam.x) / (mean3d_cam.z * mean3d_cam.z), -r2*(fy * mean3d_cam.y) / (mean3d_cam.z * mean3d_cam.z)}
    };

    // Compute dcov2d_dmean terms
    float dcov2d_dmeancamx[2][2] = {{0}};
    dcov2d_dmeancamx[0][0] = -2*r2*J[0][0]*J[0][2] / mean3d_cam.z;
    dcov2d_dmeancamx[0][1] = -r2*J[0][0]*J[1][2] / mean3d_cam.z;
    dcov2d_dmeancamx[1][0] = dcov2d_dmeancamx[0][1];

    float dcov2d_dmeancamy[2][2] = {{0}};
    dcov2d_dmeancamy[0][1] = -r2*J[1][1]*J[0][2] / mean3d_cam.z ;
    dcov2d_dmeancamy[1][0] = dcov2d_dmeancamy[0][1];
    dcov2d_dmeancamy[1][1] = -2*r2*J[1][1]*J[1][2] / mean3d_cam.z;

    float dcov2d_dmeancamz[2][2] = {{0}};
    float dJ_dmeanz[2][3] = {{0}};
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            dJ_dmeanz[i][j] = J[i][j] / -mean3d_cam.z;
        }
    }
    dJ_dmeanz[0][2] *= 2;
    dJ_dmeanz[1][2] *= 2;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 3; ++k) {
                dcov2d_dmeancamz[i][j] += matRMN[k][i] * dJ_dmeanz[j][k] + dJ_dmeanz[i][k] * matRMN[k][j];
            }
        }
    }

    float dcov2d_dmeanx[2][2] = {{0}};
    float dcov2d_dmeany[2][2] = {{0}};
    float dcov2d_dmeanz[2][2] = {{0}};
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            dcov2d_dmeanx[i][j] = R_shared[0] * dcov2d_dmeancamx[i][j] + R_shared[3] * dcov2d_dmeancamy[i][j] + R_shared[6] * dcov2d_dmeancamz[i][j];
            dcov2d_dmeany[i][j] = R_shared[1] * dcov2d_dmeancamx[i][j] + R_shared[4] * dcov2d_dmeancamy[i][j] + R_shared[7] * dcov2d_dmeancamz[i][j];
            dcov2d_dmeanz[i][j] = R_shared[2] * dcov2d_dmeancamx[i][j] + R_shared[5] * dcov2d_dmeancamy[i][j] + R_shared[8] * dcov2d_dmeancamz[i][j];
        }
    }

    // Compute dg_dcov2d
    float mat_ee[2][2] = {
        {dL_dconics[idx].x, dL_dconics[idx].y},
        {dL_dconics[idx].y, dL_dconics[idx].z}
    };
    float matinvcov2D[2][2] = {
        {invcov2D[idx].x, invcov2D[idx].y},
        {invcov2D[idx].y, invcov2D[idx].z}
    };
    float tmp[2][2] = {{0}};
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                tmp[i][j] += mat_ee[i][k] * matinvcov2D[k][j];
            }
        }
    }
    float dg_dcov2d[2][2] = {{0}};
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                dg_dcov2d[i][j] += matinvcov2D[i][k] * tmp[k][j];
            }
            dg_dcov2d[i][j] -= dL_dconics[idx].w * matinvcov2D[i][j];
        }
    }

    // Update dL_dmean3D
    float toSum[2][2];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            toSum[i][j] = dcov2d_dmeanx[i][j] * dg_dcov2d[i][j];
        }
    }
    dL_dmean3D[idx].x = toSum[0][0] + toSum[0][1] + toSum[1][0] + toSum[1][1] + dL_dmean3D_left.x;

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            toSum[i][j] = dcov2d_dmeany[i][j] * dg_dcov2d[i][j];
        }
    }
    dL_dmean3D[idx].y = toSum[0][0] + toSum[0][1] + toSum[1][0] + toSum[1][1] + dL_dmean3D_left.y;

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            toSum[i][j] = dcov2d_dmeanz[i][j] * dg_dcov2d[i][j];
        }
    }
    dL_dmean3D[idx].z = toSum[0][0] + toSum[0][1] + toSum[1][0] + toSum[1][1] + dL_dmean3D_left.z;

    // Compute scaling gradients
    float dcov2d_dr[2][2] = {{0}};
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 3; ++k) {
                for (int m = 0; m < 3; ++m) {
                    dcov2d_dr[i][j] += 2 * gmm_radius[idx] * J[i][k] * J[j][m];
                }
            }
        }
    }
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            toSum[i][j] = dcov2d_dr[i][j] * dg_dcov2d[i][j];
        }
    }
    dL_dradius[idx] = toSum[0][0] + toSum[0][1] + toSum[1][0] + toSum[1][1];
}

void renderbackwardCUDAWrapper(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int P,
	int H, int W,
    const float* __restrict__ grad_output,
    const float* __restrict__ gmm_weights_pt,
	const float2* __restrict__ means2D,
	const float4* __restrict__ invcov2D,
	float2* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D)
{
    dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    renderbackwardCUDA<<<tile_grid, block>>>(
        ranges,
		point_list,
		P,
		H,
        W,
        grad_output,
        gmm_weights_pt,
        means2D,
        invcov2D,
		dL_dmean2D,
		dL_dconic2D
    );
}

void computeGradientCUDAWrapper(int P,
	const float3* means,
    const float* gmm_radius,
    const float* gmm_weights_pt,
	const float4* invcov2D,
    const float* R,
    const float* T,
    const float* intrinsics_params,
    float2* dL_dmean2D,
    float4* dL_dconics,
    float3* dL_dmean3D,
    float* dL_dradius,
    float* dL_dw)
{
    computeGradientCUDA <<<(P + 256) / 256, 256>>> (
        P,
        means,
        gmm_radius,
        gmm_weights_pt,
        invcov2D,
        R,
        T,
        intrinsics_params,
        dL_dmean2D,
		dL_dconics,
        dL_dmean3D,
        dL_dradius,
        dL_dw
    );
}