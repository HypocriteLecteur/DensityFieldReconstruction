#include "config.h"
#include <glm/vec3.hpp>               // vec3, bvec3, dvec3, ivec3 and uvec3
#include <glm/vec4.hpp>               // vec4, bvec4, dvec4, ivec4 and uvec4
#include <glm/mat2x2.hpp>             // mat2, dmat2
#include <glm/mat2x3.hpp>             // mat2x3, dmat2x3
#include <glm/mat3x3.hpp>             // mat3, dmat3
#include <glm/matrix.hpp>             // all the GLSL matrix functions: transpose, inverse, etc.
#include <cub/cub.cuh>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

GeometryState GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
    geom.scan_size = 0; // Initialize to 0 for safety, especially in `required` context.
    obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.invcov2D, P, 128);
    obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

size_t GeometryState::required(size_t P)
{
    size_t offset = 0;
    auto align = [](size_t off, size_t a) { return (off + a - 1) & ~(a - 1); };

    // Simulate GeometryState::fromChunk
    offset = align(offset, 128); offset += P * sizeof(int);        // internal_radii
    offset = align(offset, 128); offset += P * sizeof(float2);    // means2D
    offset = align(offset, 128); offset += P * sizeof(float4);    // invcov2D
    offset = align(offset, 128); offset += P * sizeof(uint32_t);  // tiles_touched

    size_t scan_size = 0;
    cub::DeviceScan::InclusiveSum(nullptr, scan_size, (uint32_t*)nullptr, (uint32_t*)nullptr, P);
    // printf("scan_size is %zu bytes\n", scan_size);

    offset = align(offset, 128); offset += scan_size;             // scanning_space
    offset = align(offset, 128); offset += P * sizeof(uint32_t);  // point_offsets

    return offset + 128; // Extra padding
}

BinningState BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

ImageState ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.ranges, N, 128);
	return img;
}

uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

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

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const uint32_t* offsets,
	uint32_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
    int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

    // Find this Gaussian's offset in buffer for writing keys/values.
    uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
    uint2 rect_min, rect_max;

    getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);
    // For each tile that the bounding rect overlaps, emit a 
    // key/value pair. The key is tile ID ,
    // and the value is the ID of the Gaussian. Sorting the values 
    // with this key yields Gaussian IDs in a list, such that they
    // are first sorted by tile and then by depth. 
    for (int y = rect_min.y; y < rect_max.y; y++)
    {
        for (int x = rect_min.x; x < rect_max.x; x++)
        {
            uint32_t key = y * grid.x + x;
            gaussian_keys_unsorted[off] = key;
            gaussian_values_unsorted[off] = idx;
            off++;
        }
    }
}

__global__ void identifyTileRanges(int L, uint32_t* point_list_keys, uint2* ranges, int num_tiles)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= L)
        return;

    uint32_t currtile = point_list_keys[idx];
    if (idx == 0)
        ranges[currtile].x = 0;
    else {
        uint32_t prevtile = point_list_keys[idx - 1];
        if (currtile != prevtile) {
            ranges[prevtile].y = idx;
            ranges[currtile].x = idx;
        }
    }
    if (idx == L - 1)
        ranges[currtile].y = L;
}

BinningState prepareRender(GeometryState& geomState, ImageState& imgState, std::function<char*(size_t)> binningbuffer, int P, int H, int W)
{
    dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);

    cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P);

    // Retrieve total number of Gaussian instances to launch and resize aux buffers
    int num_rendered;
	cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost);

    size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningbuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

    cudaMemset(binningState.point_list_keys_unsorted, 0, num_rendered * sizeof(uint32_t));
    cudaMemset(binningState.point_list_keys, 0, num_rendered * sizeof(uint32_t));
    cudaMemset(binningState.point_list_unsorted, 0, num_rendered * sizeof(uint32_t));
    cudaMemset(binningState.point_list, 0, num_rendered * sizeof(uint32_t));

    duplicateWithKeys <<< (P + 255) / 256, 256 >>> (
		P,
		geomState.means2D,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
        geomState.internal_radii,
		tile_grid
    );
    
    int bit = getHigherMsb(tile_grid.x * tile_grid.y);
    cub::DeviceRadixSort::SortPairs(
        binningState.list_sorting_space,
        binningState.sorting_size,
        binningState.point_list_keys_unsorted, binningState.point_list_keys,
        binningState.point_list_unsorted, binningState.point_list,
        num_rendered, 0, bit
    );

    cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2));
    
    identifyTileRanges <<<(num_rendered + 255) / 256, 256 >>> (
        num_rendered,
        binningState.point_list_keys,
        imgState.ranges,
        tile_grid.x * tile_grid.y
    );

    return binningState;
}

void prepareRender(int& num_rendered, GeometryState& geomState, ImageState& imgState, BinningState& binningState, int P, int H, int W, int P_max)
{
    dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);

    cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P);

	cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost);

    if (num_rendered > MAX_TILES_PER_GAUSSIAN * P_max) return;

    cudaMemset(binningState.point_list_keys_unsorted, 0, num_rendered * sizeof(uint32_t));
    cudaMemset(binningState.point_list_unsorted, 0, num_rendered * sizeof(uint32_t));

    duplicateWithKeys <<< (P + 255) / 256, 256 >>> (
		P,
		geomState.means2D,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
        geomState.internal_radii,
		tile_grid
    );

    int bit = getHigherMsb(tile_grid.x * tile_grid.y);
    cub::DeviceRadixSort::SortPairs(
        binningState.list_sorting_space,
        binningState.sorting_size,
        binningState.point_list_keys_unsorted, binningState.point_list_keys,
        binningState.point_list_unsorted, binningState.point_list,
        num_rendered, 0, bit
    );
    
    cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2));

    identifyTileRanges <<<(num_rendered + 255) / 256, 256 >>> (
        num_rendered,
        binningState.point_list_keys,
        imgState.ranges,
        tile_grid.x * tile_grid.y
    );

    return;
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
    float2* __restrict__ means2D_pt,
    float4* __restrict__ invcov2D_pt,
    int* __restrict__ radii_pt,
    const dim3 grid,
    uint32_t* __restrict__ tiles_touched_pt)
{
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
    
    radii_pt[idx] = 0;
	tiles_touched_pt[idx] = 0;

    float3 mean3d = gmm_mean_pt[idx];
    // printf("Thread %d: mean3d = (%f, %f, %f)\n", idx, mean3d.x, mean3d.y, mean3d.z);

    float3 mean3d_cam = transformPoint(mean3d, R_shared, T_shared);
    float3 cov = computeCov2D(mean3d_cam, intrinsics_shared, gmm_radius_pt[idx]);

    float det = cov.x * cov.z - cov.y * cov.y;
    float det_inv = 1.f / det;
    float4 cov_inv = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv, sqrtf(det_inv)};
    invcov2D_pt[idx] = cov_inv;

    const float fx = intrinsics_shared[0];
    const float fy = intrinsics_shared[4];
    const float cx = intrinsics_shared[2];
    const float cy = intrinsics_shared[5];
    float2 mean2D = {fx * mean3d_cam.x / mean3d_cam.z + cx, fy * mean3d_cam.y / mean3d_cam.z + cy};
    means2D_pt[idx] = mean2D;

    // Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrtf(fmaxf(0.1f, mid * mid - det));
	float lambda2 = mid - sqrtf(fmaxf(0.1f, mid * mid - det));
	int my_radius = int(ceil(3.f * sqrtf(fmaxf(lambda1, lambda2))));
    radii_pt[idx] = my_radius;

	uint2 rect_min, rect_max;
	getRect(mean2D, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

    tiles_touched_pt[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
    return;
}

__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(int P,
           const uint2* __restrict__ ranges,
           const uint32_t* __restrict__ point_list,
           float* __restrict__ density_estim_pt,
           int H, int W,
           const float2* __restrict__ means2D,
           const float4* __restrict__ invcov2D,
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
    const bool done_thread = !inside;

    // Load start/end range of IDs to process in bit sorted list.
    const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;

    // Allocate storage for batches of collectively fetched data.
    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_weights[BLOCK_SIZE];

    // Iterate over batches until all done or range is complete
    float density = 0.0f; // Accumulator for this pixel's density

    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
    {
        // End if entire block votes that it is done rasterizing
        if (__syncthreads_count(done_thread) == BLOCK_SIZE)
            break;

        // Collectively fetch per-Gaussian data from global to shared
        const int progress = i * BLOCK_SIZE + thread_rank;
        if (range.x + progress < range.y)
        {
            const int coll_id = point_list[range.x + progress];
            collected_id[thread_rank] = coll_id;
            collected_xy[thread_rank] = means2D[coll_id];
            collected_conic_weights[thread_rank] = invcov2D[coll_id];
        }
        block.sync(); // Ensure all threads have loaded their batch data into shared memory
                                                                                                          
		// Only proceed if this thread corresponds to an 'inside' pixel.
		if (inside)
        {
			// Iterate over current batch
			const int current_batch_size = min(BLOCK_SIZE, toDo);
			for (int j = 0; j < current_batch_size; j++)
            {
                const float2 xy_sh = collected_xy[j];
                const float4 con_o_sh = collected_conic_weights[j];
                const int coll_id_sh = collected_id[j];
                const float gmm_weight_sh = gmm_weights_pt[coll_id_sh];

                const float2 d = { xy_sh.x - pixf.x, xy_sh.y - pixf.y };
                const float power = fma(-0.5f, fma(con_o_sh.x, d.x * d.x, con_o_sh.z * d.y * d.y), -con_o_sh.y * d.x * d.y);
                
                density += con_o_sh.w * __expf(power) * gmm_weight_sh;
                // density += __expf(power) * gmm_weight_sh;
            }
        }
    }

    if (inside) {
        density *= 0.5f * 0.318309886184f * 255.0f;
        density_estim_pt[pix_id] = density;
    };
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
    float2* means2D_pt,
    float4* invcov2D_pt,
    int* radii_pt,
    uint32_t* tiles_touched_pt)
{
    dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
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
        means2D_pt,
        invcov2D_pt,
        radii_pt,
        tile_grid,
        tiles_touched_pt
    );
}

void renderCUDAWrapper(
    int P,
    const uint2* ranges,
	const uint32_t* point_list,
    float* density_estim_pt,
	int H, int W,
	const float2* means2D,
	const float4* invcov2D,
    const float* gmm_weights_pt)
{
    dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

    renderCUDA<<<tile_grid, block>>> (
        P,
        ranges,
        point_list,
        density_estim_pt,
		H,
        W,
        means2D,
        invcov2D,
        gmm_weights_pt);
}