#include <torch/extension.h>
#include <cuda_runtime.h>
#include "config.h"

// Efficient buffer resizing: only reallocates if needed, avoids unnecessary .contiguous()
std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        // Only reallocate if needed
        if (!t.defined() || t.numel() < N) {
            // Use torch::empty for efficiency, always float dtype for alignment
            t = torch::empty({(long long)N}, t.options().dtype(torch::kUInt8));
        }
        return reinterpret_cast<char*>(t.data_ptr());
    };
    return lambda;
}

torch::Tensor
RasterizeGaussians(
    const torch::Tensor& gmm_mean,
    const torch::Tensor& gmm_radius,
    const torch::Tensor& gmm_weights,
    const torch::Tensor& R,
    const torch::Tensor& T,
    const torch::Tensor& intrinsics_params,
    const int H,
    const int W,
    bool profile)
{
    cudaEvent_t start_event, stop_event;
    float total_forward_ms = 0.0f, total_backward_ms = 0.0f;
    if (profile) {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }
    // Helper lambda to start the timer
    auto record_start = [&]() {
        if (profile) cudaEventRecord(start_event);
    };
    // Helper lambda to stop the timer and print the duration
    auto record_stop_and_print = [&](const char* name, float& total_ms) {
        if (profile) {
            float ms;
            cudaEventRecord(stop_event);
            cudaEventSynchronize(stop_event);
            cudaEventElapsedTime(&ms, start_event, stop_event);
            printf("Module '%s': %.4f us\n", name, ms*1000);
            total_ms += ms;
        }
    };

    // --------------------------------------------------------------------
    // Prepare
    record_start();

    // create tensors
    torch::Device device(torch::kCUDA);
    auto options = torch::TensorOptions().dtype(torch::kFloat).device(device);
    auto options_int = torch::TensorOptions().dtype(torch::kInt).device(device);
    auto options_uint32 = torch::TensorOptions().dtype(torch::kInt32).device(device);;

    torch::Tensor density_estim = torch::zeros({H, W}, options);
    const int P = gmm_mean.size(0);

    torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
    torch::Tensor binnBuffer = torch::empty({0}, options.device(device));
    torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> geometryBuffer = resizeFunctional(geomBuffer);
    std::function<char*(size_t)> binningBuffer = resizeFunctional(binnBuffer);
    std::function<char*(size_t)> imageBuffer = resizeFunctional(imgBuffer);

    // get pointers
    torch::Tensor gmm_mean_contig = gmm_mean.is_contiguous() ? gmm_mean : gmm_mean.contiguous();
    torch::Tensor gmm_radius_contig = gmm_radius.is_contiguous() ? gmm_radius : gmm_radius.contiguous();
    torch::Tensor gmm_weights_contig = gmm_weights.is_contiguous() ? gmm_weights : gmm_weights.contiguous();
    torch::Tensor R_contig = R.is_contiguous() ? R : R.contiguous();
    torch::Tensor T_contig = T.is_contiguous() ? T : T.contiguous();
    torch::Tensor intrinsics_contig = intrinsics_params.is_contiguous() ? intrinsics_params : intrinsics_params.contiguous();

    float* density_estim_pt = density_estim.data_ptr<float>();
    const float* gmm_mean_pt = gmm_mean_contig.data_ptr<float>();
    const float* gmm_radius_pt = gmm_radius_contig.data_ptr<float>();
    const float* gmm_weights_pt = gmm_weights_contig.data_ptr<float>();
    const float* R_pt = R_contig.data_ptr<float>();
    const float* T_pt = T_contig.data_ptr<float>();
    const float* intrinsics_params_pt = intrinsics_contig.data_ptr<float>();

    size_t chunk_size = required<GeometryState>(P);
    char* chunkptr = geometryBuffer(chunk_size);
    GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

    int tile_grid_x = (W + BLOCK_X - 1) / BLOCK_X;
    int tile_grid_y = (H + BLOCK_Y - 1) / BLOCK_Y;

    size_t img_chunk_size = required<ImageState>(tile_grid_x * tile_grid_y);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, tile_grid_x * tile_grid_y);

    record_stop_and_print("prepare", total_forward_ms);
    // --------------------------------------------------------------------
    // Preprocess
    record_start();

    preprocessCUDAWrapper(
        P,
        (float3*)gmm_mean_pt,
        gmm_radius_pt,
        gmm_weights_pt,
        R_pt,
        T_pt,
        intrinsics_params_pt,
        H,
        W,
        geomState.means2D,
        geomState.invcov2D,
        geomState.internal_radii,
        geomState.tiles_touched
    );

    record_stop_and_print("preprocess", total_forward_ms);
    // --------------------------------------------------------------------
    // Prepare render
    record_start();

    BinningState binningState = prepareRender(geomState, imgState, binningBuffer, P, H, W);

    record_stop_and_print("prepare render", total_forward_ms);
    // --------------------------------------------------------------------
    // Render
    record_start();

	renderCUDAWrapper(
        P,
        imgState.ranges,
        binningState.point_list,
        density_estim_pt,
		H,
        W,
        geomState.means2D,
        geomState.invcov2D,
        gmm_weights_pt
    );

    record_stop_and_print("render", total_forward_ms);
    return density_estim;
}

class GaussianRasterizerSimpleLarge {
public:
    // Constructor to initialize the rasterizer with image dimensions and max Gaussians
    GaussianRasterizerSimpleLarge(int H, int W, int P_max);

    // The main method to perform rasterization and backward pass
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    rasterize_forward_backward(
        const torch::Tensor& gmm_mean,
        const torch::Tensor& gmm_radius,
        const torch::Tensor& gmm_weights,
        const torch::Tensor& R,
        const torch::Tensor& T,
        const torch::Tensor& intrinsics,
        const torch::Tensor& density,
        bool profile);

private:
    int H;
    int W;
    int P_max_;

    // Pre-allocated buffers
    torch::Tensor imgBuffer_;
    torch::Tensor geomBuffer_;
    torch::Tensor binnBuffer_;
    GeometryState geomState;
    ImageState imgState;
    BinningState binningState;

    torch::Tensor density_estim;
    torch::Tensor grad_gmm_mean;
    torch::Tensor grad_gmm_radius;
    torch::Tensor grad_gmm_weights;
    torch::Tensor dL_dmean2D_;
    torch::Tensor dL_dconic_;
};

GaussianRasterizerSimpleLarge::GaussianRasterizerSimpleLarge(int H, int W, int P_max) : H(H), W(W), P_max_(P_max) {
    torch::Device device(torch::kCUDA);
    auto options = torch::TensorOptions().dtype(torch::kByte).device(device);
    auto options_float = torch::TensorOptions().dtype(torch::kFloat).device(device);

    int tile_grid_x = (W + BLOCK_X - 1) / BLOCK_X;
    int tile_grid_y = (H + BLOCK_Y - 1) / BLOCK_Y;

    // Pre-allocate image-based buffer based on image dimensions
    size_t img_chunk_size = required<ImageState>(tile_grid_x * tile_grid_y);
    imgBuffer_ = torch::empty({0}, options);
    std::function<char*(size_t)> imageBuffer = resizeFunctional(imgBuffer_);
    char* img_chunkptr = imageBuffer(img_chunk_size);
    imgState = ImageState::fromChunk(img_chunkptr, tile_grid_x * tile_grid_y);

    geomBuffer_ = torch::empty({0}, options);
    std::function<char*(size_t)> geometryBuffer = resizeFunctional(geomBuffer_);
    size_t chunk_size = GeometryState::required(P_max_);
    char* chunkptr = geometryBuffer(chunk_size);
    geomState = GeometryState::fromChunk(chunkptr, P_max_);

    binnBuffer_ = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> binningBuffer = resizeFunctional(binnBuffer_);
    size_t binning_chunk_size = required<BinningState>(MAX_TILES_PER_GAUSSIAN * P_max_);
    char* binning_chunkptr = binningBuffer(binning_chunk_size);
    binningState = BinningState::fromChunk(binning_chunkptr, MAX_TILES_PER_GAUSSIAN * P_max_);

    density_estim = torch::zeros({H, W}, options_float);

    grad_gmm_mean = torch::zeros({P_max_, 3}, options_float);
    grad_gmm_radius = torch::zeros({P_max_, 1}, options_float);
    grad_gmm_weights = torch::zeros({P_max_, 1}, options_float);
    dL_dmean2D_ = torch::zeros({P_max_, 2}, options_float);
    dL_dconic_ = torch::zeros({P_max_, 2, 2}, options_float);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GaussianRasterizerSimpleLarge::rasterize_forward_backward(
    const torch::Tensor& gmm_mean_contig,
    const torch::Tensor& gmm_radius_contig,
    const torch::Tensor& gmm_weights_contig,
    const torch::Tensor& R_contig,
    const torch::Tensor& T_contig,
    const torch::Tensor& intrinsics_contig,
    const torch::Tensor& density,
    bool profile = false)
{
    cudaEvent_t start_event, stop_event;
    float total_forward_ms = 0.0f, total_backward_ms = 0.0f;
    if (profile) {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }
    // Helper lambda to start the timer
    auto record_start = [&]() {
        if (profile) cudaEventRecord(start_event);
    };
    // Helper lambda to stop the timer and print the duration
    auto record_stop_and_print = [&](const char* name, float& total_ms) {
        if (profile) {
            float ms;
            cudaEventRecord(stop_event);
            cudaEventSynchronize(stop_event);
            cudaEventElapsedTime(&ms, start_event, stop_event);
            printf("Module '%s': %.4f us\n", name, ms*1000);
            total_ms += ms;
        }
    };
    
    // --------------------------------------------------------------------
    // Preprocess
    record_start();

    float* density_estim_pt = density_estim.data_ptr<float>();
    const float* gmm_mean_pt = gmm_mean_contig.data_ptr<float>();
    const float* gmm_radius_pt = gmm_radius_contig.data_ptr<float>();
    const float* gmm_weights_pt = gmm_weights_contig.data_ptr<float>();
    const float* R_pt = R_contig.data_ptr<float>();
    const float* T_pt = T_contig.data_ptr<float>();
    const float* intrinsics_params_pt = intrinsics_contig.data_ptr<float>();

    const int P = gmm_mean_contig.size(0);

    preprocessCUDAWrapper(
        P,
        (float3*)gmm_mean_pt,
        gmm_radius_pt,
        gmm_weights_pt,
        R_pt,
        T_pt,
        intrinsics_params_pt,
        H,
        W,
        geomState.means2D,
        geomState.invcov2D,
        geomState.internal_radii,
        geomState.tiles_touched
    );

    record_stop_and_print("preprocess", total_forward_ms);
    // --------------------------------------------------------------------
    // Prepare render
    record_start();

    int num_rendered;
    prepareRender(num_rendered, geomState, imgState, binningState, P, H, W, P_max_);
    if (num_rendered > MAX_TILES_PER_GAUSSIAN * P_max_)
    {
        throw std::runtime_error("num_rendered " + std::to_string(num_rendered) + " exceeds buffer size, which is " + std::to_string(MAX_TILES_PER_GAUSSIAN * P_max_));
    }

    record_stop_and_print("prepare render", total_forward_ms);
    // --------------------------------------------------------------------
    // Render
    record_start();

	renderCUDAWrapper(
        P,
        imgState.ranges,
        binningState.point_list,
        density_estim_pt,
		H,
        W,
        geomState.means2D,
        geomState.invcov2D,
        gmm_weights_pt
    );

    record_stop_and_print("render", total_forward_ms);
    // --------------------------------------------------------------------
    // Prepare backward
    record_start();

    torch::Tensor loss = density_estim - density;
    torch::Tensor sum_loss = torch::sum(torch::abs(loss));
    torch::Tensor grad_out = torch::sign(loss).contiguous();

    cudaMemset(dL_dmean2D_.data_ptr<float>(), 0, P * 2 * sizeof(float));
    cudaMemset(dL_dconic_.data_ptr<float>(), 0, P * 4 * sizeof(float));

    float* grad_output_pt = grad_out.data_ptr<float>();
    float* dL_dmean3D_pt = grad_gmm_mean.data_ptr<float>();
    float* dL_dradius_pt = grad_gmm_radius.data_ptr<float>();
    float* dL_dmean2D_pt = dL_dmean2D_.data_ptr<float>();
    float* dL_dconic_pt = dL_dconic_.data_ptr<float>();
    float* dL_dw_pt = grad_gmm_weights.data_ptr<float>();

    record_stop_and_print("backward prepare", total_forward_ms);
    // --------------------------------------------------------------------
    // Render backward
    record_start();

    renderbackwardCUDAWrapper(
        imgState.ranges,
		binningState.point_list,
		P,
		H,
        W,
        grad_output_pt,
        gmm_weights_pt,
        geomState.means2D,
        geomState.invcov2D,
		(float2*)dL_dmean2D_pt,
		(float4*)dL_dconic_pt
    );

    record_stop_and_print("render backward", total_forward_ms);
    // --------------------------------------------------------------------
    // Compute gradient
    record_start();

    computeGradientCUDAWrapper(
        P,
        (float3*)gmm_mean_pt,
        gmm_radius_pt,
        gmm_weights_pt,
        geomState.invcov2D,
        R_pt,
        T_pt,
        intrinsics_params_pt,
        (float2*)dL_dmean2D_pt,
		(float4*)dL_dconic_pt,
        (float3*)dL_dmean3D_pt,
        dL_dradius_pt,
        dL_dw_pt
    );

    record_stop_and_print("compute gradient", total_backward_ms);

    return std::make_tuple(
        grad_gmm_mean.slice(0, 0, P), 
        grad_gmm_radius.slice(0, 0, P), 
        grad_gmm_weights.slice(0, 0, P), 
        density_estim, 
        sum_loss);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_gaussians", &RasterizeGaussians);
    // m.def("rasterize_gaussians_and_backward", &RasterizeGaussiansAndBackwardCUDA);
    py::class_<GaussianRasterizerSimpleLarge>(m, "GaussianRasterizerSimpleLarge")
    .def(py::init<int, int, int>(),
            py::arg("H"), 
            py::arg("W"), 
            py::arg("P_max"),
            "Initializes the Gaussian Rasterizer with image dimensions and maximum number of Gaussians.")
    
    .def("rasterize_forward_backward", &GaussianRasterizerSimpleLarge::rasterize_forward_backward,
            "Rasterizes a set of Gaussians and computes gradients for backpropagation.",
            py::arg("gmm_mean"),
            py::arg("gmm_radius"),
            py::arg("gmm_weights"),
            py::arg("R"),
            py::arg("T"),
            py::arg("intrinsics"),
            py::arg("density"),
            py::arg("profile"));
}