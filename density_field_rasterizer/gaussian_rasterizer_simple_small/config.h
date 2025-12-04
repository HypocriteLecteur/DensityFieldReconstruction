#pragma once

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define MAX_WARPS_PER_BLOCK 8 // Max threads per block is 1024, so max warps is 1024/32=32
#define MAX_TILES_PER_GAUSSIAN 64

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
    const dim3 grid);

void renderCUDAWrapper(
    const dim3 tile_grid,
    const dim3 block,
    int P,
    float* density_estim_pt,
	int H, int W,
	const float2* means2D,
	const float4* invcov2D,
    const int* radii,
    const float* gmm_weights_pt);

void renderbackwardCUDAWrapper(
    const dim3 tile_grid,
    const dim3 block,
    int P,
	int H, int W,
    const float* __restrict__ grad_output,
    const float* __restrict__ gmm_weights_pt,
	const float2* __restrict__ means2D,
	const float4* __restrict__ invcov2D,
    const int* __restrict__ radii,
	float2* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D);

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
    float* dL_dw);