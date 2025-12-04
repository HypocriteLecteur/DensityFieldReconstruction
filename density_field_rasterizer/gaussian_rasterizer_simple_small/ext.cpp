#include <torch/extension.h>
#include <cuda_runtime.h>
#include "config.h"

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

    torch::Device device(torch::kCUDA);
    auto options_int = torch::TensorOptions().dtype(torch::kInt).device(device);
    auto options_float = torch::TensorOptions().dtype(torch::kFloat).device(device);

    torch::Tensor gmm_mean_contig = gmm_mean.is_contiguous() ? gmm_mean : gmm_mean.contiguous();
    torch::Tensor gmm_radius_contig = gmm_radius.is_contiguous() ? gmm_radius : gmm_radius.contiguous();
    torch::Tensor gmm_weights_contig = gmm_weights.is_contiguous() ? gmm_weights : gmm_weights.contiguous();
    torch::Tensor R_contig = R.is_contiguous() ? R : R.contiguous();
    torch::Tensor T_contig = T.is_contiguous() ? T : T.contiguous();
    torch::Tensor intrinsics_contig = intrinsics_params.is_contiguous() ? intrinsics_params : intrinsics_params.contiguous();

    const int P = gmm_mean_contig.size(0);

    torch::Tensor density_estim = torch::zeros({H, W}, options_float);

    dim3 tile_grid_ = dim3((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block = dim3(BLOCK_X, BLOCK_Y, 1);

    torch::Tensor means2D_ = torch::zeros({P, 2}, options_float);
    torch::Tensor invcov2D_ = torch::zeros({P, 4}, options_float);
    torch::Tensor internal_radii_ = torch::zeros({P, 1}, options_int);

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
    float* means2D_pt = means2D_.data_ptr<float>();
    float* invcov2D_pt = invcov2D_.data_ptr<float>();
    int* internal_radii_pt = internal_radii_.data_ptr<int>();

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
        (float2*)means2D_pt,
        (float4*)invcov2D_pt,
        internal_radii_pt,
        tile_grid_);

    record_stop_and_print("Preprocess", total_forward_ms);
    // --------------------------------------------------------------------
    // Render
    record_start();

	renderCUDAWrapper(
        tile_grid_,
        block,
        P,
        density_estim_pt,
		H,
        W,
        (float2*)means2D_pt,
        (float4*)invcov2D_pt,
        internal_radii_pt,
        gmm_weights_pt);
    
    record_stop_and_print("renderCUDA", total_forward_ms);
    // --------------------------------------------------------------------
    return density_estim;
}

class GaussianRasterizerSimpleSmall {
public:
    GaussianRasterizerSimpleSmall(int H, int W, int P_max);

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
    // Config
    int H;
    int W;
    int P_max_;

    // Output
    torch::Tensor density_estim;
    torch::Tensor grad_gmm_mean;
    torch::Tensor grad_gmm_radius;
    torch::Tensor grad_gmm_weights;

    dim3 tile_grid_;
    dim3 block;
    
    // Intermediate result
    torch::Tensor means2D_;
    torch::Tensor invcov2D_;
    torch::Tensor internal_radii_;

    torch::Tensor dL_dmean2D_;
    torch::Tensor dL_dconic_;
};

GaussianRasterizerSimpleSmall::GaussianRasterizerSimpleSmall(int H, int W, int P_max) : H(H), W(W), P_max_(P_max) {
    torch::Device device(torch::kCUDA);
    auto options_int = torch::TensorOptions().dtype(torch::kInt).device(device);
    auto options_float = torch::TensorOptions().dtype(torch::kFloat).device(device);

    density_estim = torch::zeros({H, W}, options_float);
    grad_gmm_mean = torch::zeros({P_max_, 3}, options_float);
    grad_gmm_radius = torch::zeros({P_max_, 1}, options_float);
    grad_gmm_weights = torch::zeros({P_max_, 1}, options_float);

    tile_grid_ = dim3((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
    block = dim3(BLOCK_X, BLOCK_Y, 1);

    means2D_ = torch::zeros({P_max_, 2}, options_float);
    invcov2D_ = torch::zeros({P_max_, 4}, options_float);
    internal_radii_ = torch::zeros({P_max_, 1}, options_int);

    dL_dmean2D_ = torch::zeros({P_max_, 2}, options_float);
    dL_dconic_ = torch::zeros({P_max_, 2, 2}, options_float);
};

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GaussianRasterizerSimpleSmall::rasterize_forward_backward(
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
    float* means2D_pt = means2D_.data_ptr<float>();
    float* invcov2D_pt = invcov2D_.data_ptr<float>();
    int* internal_radii_pt = internal_radii_.data_ptr<int>();

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
        (float2*)means2D_pt,
        (float4*)invcov2D_pt,
        internal_radii_pt,
        tile_grid_);

    record_stop_and_print("Preprocess", total_forward_ms);
    // --------------------------------------------------------------------
    // Render
    record_start();

	renderCUDAWrapper(
        tile_grid_,
        block,
        P,
        density_estim_pt,
		H,
        W,
        (float2*)means2D_pt,
        (float4*)invcov2D_pt,
        internal_radii_pt,
        gmm_weights_pt);
    
    record_stop_and_print("renderCUDA", total_forward_ms);
    // --------------------------------------------------------------------
    // Backward prepare
    record_start();

    torch::Tensor loss = density_estim - density;
    torch::Tensor sum_loss = torch::sum(torch::abs(loss));
    torch::Tensor grad_out = torch::sign(loss).contiguous();

    // Skip the reset for the output gradients
    // cudaMemset(grad_gmm_mean.data_ptr<float>(), 0, P * 3 * sizeof(float));
    // cudaMemset(grad_gmm_radius.data_ptr<float>(), 0, P * 1 * sizeof(float));
    // cudaMemset(grad_gmm_weights.data_ptr<float>(), 0, P * 1 * sizeof(float));
    cudaMemset(dL_dmean2D_.data_ptr<float>(), 0, P * 2 * sizeof(float));
    cudaMemset(dL_dconic_.data_ptr<float>(), 0, P * 4 * sizeof(float));
    
    float* grad_output_pt = grad_out.data_ptr<float>();
    float* dL_dmean3D_pt = grad_gmm_mean.data_ptr<float>();
    float* dL_dradius_pt = grad_gmm_radius.data_ptr<float>();
    float* dL_dmean2D_pt = dL_dmean2D_.data_ptr<float>();
    float* dL_dconic_pt = dL_dconic_.data_ptr<float>();
    float* dL_dw_pt = grad_gmm_weights.data_ptr<float>();

    record_stop_and_print("backward_prepare", total_forward_ms);
    // --------------------------------------------------------------------
    // Compute loss gradients w.r.t. 2D mean position, inverse covariance matrix,
	// weights from per-pixel loss gradients.
    record_start();
    renderbackwardCUDAWrapper(
        tile_grid_,
        block,
		P,
		H,
        W,
        grad_output_pt,
        gmm_weights_pt,
        (float2*)means2D_pt,
        (float4*)invcov2D_pt,
        internal_radii_pt,
		(float2*)dL_dmean2D_pt,
		(float4*)dL_dconic_pt
    );
    record_stop_and_print("renderBackwardCUDA", total_backward_ms);

    record_start();
    computeGradientCUDAWrapper(
        P,
        (float3*)gmm_mean_pt,
        gmm_radius_pt,
        gmm_weights_pt,
        (float4*)invcov2D_pt,
        R_pt,
        T_pt,
        intrinsics_params_pt,
        (float2*)dL_dmean2D_pt,
		(float4*)dL_dconic_pt,
        (float3*)dL_dmean3D_pt,
        dL_dradius_pt,
        dL_dw_pt
    );
    record_stop_and_print("computeGradientCUDA", total_backward_ms);
    // --------------------------------------------------------------------
    
    return std::make_tuple(grad_gmm_mean.slice(0, 0, P), grad_gmm_radius.slice(0, 0, P), grad_gmm_weights.slice(0, 0, P), density_estim, sum_loss);
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_gaussians", &RasterizeGaussians);
    py::class_<GaussianRasterizerSimpleSmall>(m, "GaussianRasterizerSimpleSmall")
    .def(py::init<int, int, int>(),
            py::arg("H"), 
            py::arg("W"), 
            py::arg("P_max"),
            "Initializes the Gaussian Rasterizer with image dimensions and maximum number of Gaussians.")
    
    .def("rasterize_forward_backward", &GaussianRasterizerSimpleSmall::rasterize_forward_backward,
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