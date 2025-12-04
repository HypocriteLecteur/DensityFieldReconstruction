#pragma once

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define MAX_WARPS_PER_BLOCK 8 // Max threads per block is 1024, so max warps is 1024/32=32
#define MAX_TILES_PER_GAUSSIAN 256

#include <functional>
#include <cuda_runtime.h>

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
    uint32_t* tiles_touched
);

void renderCUDAWrapper(
    int P,
    const uint2* ranges,
	const uint32_t* point_list,
    float* density_estim_pt,
	int H, int W,
	const float2* means2D,
	const float4* invcov2D,
    const float* gmm_weights_pt
);

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
	float4* __restrict__ dL_dconic2D
);

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
    float* dL_dw
);

template <typename T>
static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
{
    std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
    ptr = reinterpret_cast<T*>(offset);
    char* next_chunk = reinterpret_cast<char*>(ptr + count);

    // Check if next_chunk exceeds the allocated buffer size
    if (next_chunk > (char*)chunk + 1024*1024*1024) // 1GB limit, adjust as needed
    {
        printf("ERROR: obtain() exceeds buffer limit\n");
    }
    chunk = next_chunk;
};

struct GeometryState
{
    float2* means2D;
    float4* invcov2D;
    int* internal_radii;
    size_t scan_size;
    char* scanning_space;
    uint32_t* point_offsets;
    uint32_t* tiles_touched;
    static GeometryState fromChunk(char*& chunk, size_t P);
    static size_t required(size_t P);
};

struct BinningState
{
    size_t sorting_size;
    uint32_t* point_list_keys_unsorted;
    uint32_t* point_list_keys;
    uint32_t* point_list_unsorted;
    uint32_t* point_list;
    char* list_sorting_space;

    static BinningState fromChunk(char*& chunk, size_t P);
};

struct ImageState
{
    uint2* ranges;
    static ImageState fromChunk(char*& chunk, size_t N);
};

template<typename T> 
size_t required(size_t P)
{
    // Use a dummy buffer for pointer arithmetic to avoid undefined behavior
    char dummy[4096];
    char* size = dummy;
    T::fromChunk(size, P);
    return (size_t)(size - dummy) + 128;
}

BinningState prepareRender(GeometryState& geomState, ImageState& imgState, std::function<char*(size_t)> binningbuffer, int P, int H, int W);
void prepareRender(int& num_rendered, GeometryState& geomState, ImageState& imgState, BinningState& binningState, int P, int H, int W, int P_max);