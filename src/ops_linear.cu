#include "ops.h"

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudasep {

bool g_quantize_fp16 = false;

namespace {

constexpr int kBlockSize = 256;

int64_t ceildiv(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

Tensor ensure_f32(const Tensor& x) {
    if (x.dtype() == DType::Float32) return x.contiguous();
    return x.to_f32().contiguous();
}

Tensor maybe_cast_back(const Tensor& result, DType orig) {
    if (orig == DType::Float16) return result.to_f16();
    return result;
}

void check_cublas(cublasStatus_t status, const char* op) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(op) + " failed with cuBLAS status " +
                                 std::to_string((int)status));
    }
}

__global__ void add_bias_kernel(float* __restrict__ out,
                                const float* __restrict__ bias,
                                int64_t rows,
                                int64_t cols) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        out[idx] += bias[idx % cols];
    }
}

Tensor linear_gemm_f32_output(const Tensor& x,
                              const Tensor& weight,
                              int64_t& total_batch,
                              int64_t& out_features,
                              std::vector<int64_t>& out_shape) {
    int64_t in_features = weight.size(1);
    out_features = weight.size(0);

    if (weight.dtype() == DType::Float16) {
        Tensor xh = (x.dtype() == DType::Float16) ? x.contiguous() : x.to_f16().contiguous();
        Tensor wh = weight.contiguous();
        total_batch = xh.numel() / in_features;
        out_shape = xh.shape();
        out_shape.back() = out_features;

        Tensor out = Tensor::empty({total_batch, out_features}, DType::Float32);
        float alpha = 1.0f;
        float beta = 0.0f;
        check_cublas(cublasGemmEx(CudaContext::instance().cublas(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            (int)out_features, (int)total_batch, (int)in_features,
            &alpha,
            wh.data_ptr(), CUDA_R_16F, (int)in_features,
            xh.data_ptr(), CUDA_R_16F, (int)in_features,
            &beta,
            out.data_f32(), CUDA_R_32F, (int)out_features,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP), "linear FP16 GEMM");
        return out;
    }

    Tensor xf = ensure_f32(x);
    Tensor wf = ensure_f32(weight);
    assert(wf.ndim() == 2);
    assert(xf.size(xf.ndim() - 1) == in_features);

    total_batch = xf.numel() / in_features;
    out_shape = xf.shape();
    out_shape.back() = out_features;

    Tensor out = Tensor::empty({total_batch, out_features}, DType::Float32);
    float alpha = 1.0f;
    float beta = 0.0f;
    check_cublas(cublasGemmEx(CudaContext::instance().cublas(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        (int)out_features, (int)total_batch, (int)in_features,
        &alpha,
        wf.data_f32(), CUDA_R_32F, (int)in_features,
        xf.data_f32(), CUDA_R_32F, (int)in_features,
        &beta,
        out.data_f32(), CUDA_R_32F, (int)out_features,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP), "linear FP32 GEMM");
    return out;
}

}  // namespace

CudaContext::CudaContext() {
    cublasStatus_t status = cublasCreate(&cublas_);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasCreate failed with status " + std::to_string((int)status));
    }
    CUDA_CHECK(cudaStreamCreate(&stream_));
    cublasSetStream(cublas_, stream_);
    cublasSetMathMode(cublas_, CUBLAS_TF32_TENSOR_OP_MATH);
}

CudaContext::~CudaContext() {
    cublasDestroy(cublas_);
    cudaStreamDestroy(stream_);
}

CudaContext& CudaContext::instance() {
    static CudaContext ctx;
    return ctx;
}

namespace ops {

Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    DType orig = x.dtype();
    Tensor bf = ensure_f32(bias);

    int64_t total_batch = 0;
    int64_t out_features = 0;
    std::vector<int64_t> out_shape;
    Tensor out = linear_gemm_f32_output(x, weight, total_batch, out_features, out_shape);

    assert(bf.ndim() == 1 && bf.size(0) == out_features);
    int64_t total = total_batch * out_features;
    int blocks = (int)ceildiv(total, (int64_t)kBlockSize);
    add_bias_kernel<<<blocks, kBlockSize>>>(out.data_f32(), bf.data_f32(), total_batch, out_features);
    CUDA_CHECK(cudaGetLastError());

    Tensor result = out.reshape(out_shape);
    return (weight.dtype() == DType::Float16) ? result : maybe_cast_back(result, orig);
}

Tensor linear_no_bias(const Tensor& x, const Tensor& weight) {
    DType orig = x.dtype();

    int64_t total_batch = 0;
    int64_t out_features = 0;
    std::vector<int64_t> out_shape;
    Tensor out = linear_gemm_f32_output(x, weight, total_batch, out_features, out_shape);
    Tensor result = out.reshape(out_shape);
    return (weight.dtype() == DType::Float16) ? result : maybe_cast_back(result, orig);
}

}  // namespace ops
}  // namespace cudasep
