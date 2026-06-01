#include "ops.h"

#include <cmath>

namespace cudasep {

namespace {

constexpr int kBlockSize = 256;

int64_t ceildiv(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__ float gelu_fast(float x) {
    constexpr float kInvSqrt2 = 0.7071067811865475f;
    return 0.5f * x * (1.0f + erff(x * kInvSqrt2));
}

__device__ __forceinline__ float sigmoid_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void fused_bias_gelu_kernel(float* __restrict__ out,
                                       const float* __restrict__ bias,
                                       int64_t total,
                                       int64_t out_features) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        out[idx] = gelu_fast(out[idx] + bias[idx % out_features]);
    }
}

__global__ void fused_bias_sigmoid_kernel(float* __restrict__ out,
                                          const float* __restrict__ bias,
                                          int64_t total,
                                          int64_t out_features) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        out[idx] = sigmoid_fast(out[idx] + bias[idx % out_features]);
    }
}

__global__ void overlap_add_kernel(float* __restrict__ dest,
                                   const float* __restrict__ src,
                                   const float* __restrict__ window,
                                   int64_t dest_offset,
                                   int64_t chunk_len,
                                   int64_t num_channels,
                                   int64_t dest_total_len) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = num_channels * chunk_len;
    if (idx >= total) return;

    int64_t c = idx / chunk_len;
    int64_t i = idx % chunk_len;
    int64_t dest_idx = c * dest_total_len + dest_offset + i;
    atomicAdd(&dest[dest_idx], src[c * chunk_len + i] * window[i]);
}

__global__ void weight_accumulate_kernel(float* __restrict__ weight_sum,
                                         const float* __restrict__ window,
                                         int64_t offset,
                                         int64_t chunk_len) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < chunk_len) {
        atomicAdd(&weight_sum[offset + idx], window[idx]);
    }
}

__global__ void normalize_by_weights_kernel(float* __restrict__ data,
                                            const float* __restrict__ weight_sum,
                                            int64_t total_len,
                                            int64_t num_channels) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = num_channels * total_len;
    if (idx >= total) return;

    float w = weight_sum[idx % total_len];
    if (w > 1e-8f) {
        data[idx] /= w;
    }
}

Tensor do_gemm(const Tensor& x, const Tensor& weight,
               int64_t& total_batch, int64_t& out_features,
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
        cublasGemmEx(CudaContext::instance().cublas(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            (int)out_features, (int)total_batch, (int)in_features,
            &alpha,
            wh.data_ptr(), CUDA_R_16F, (int)in_features,
            xh.data_ptr(), CUDA_R_16F, (int)in_features,
            &beta,
            out.data_f32(), CUDA_R_32F, (int)out_features,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        return out;
    }

    Tensor xf = (x.dtype() == DType::Float32) ? x.contiguous() : x.to_f32().contiguous();
    Tensor wf = (weight.dtype() == DType::Float32) ? weight.contiguous() : weight.to_f32().contiguous();
    total_batch = xf.numel() / in_features;
    out_shape = xf.shape();
    out_shape.back() = out_features;

    Tensor out = Tensor::empty({total_batch, out_features}, DType::Float32);
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasGemmEx(CudaContext::instance().cublas(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        (int)out_features, (int)total_batch, (int)in_features,
        &alpha,
        wf.data_f32(), CUDA_R_32F, (int)in_features,
        xf.data_f32(), CUDA_R_32F, (int)in_features,
        &beta,
        out.data_f32(), CUDA_R_32F, (int)out_features,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return out;
}

Tensor linear_activation(const Tensor& x, const Tensor& weight, const Tensor& bias, bool gelu) {
    DType orig = x.dtype();
    Tensor bf = (bias.dtype() == DType::Float32) ? bias.contiguous() : bias.to_f32().contiguous();

    int64_t total_batch = 0;
    int64_t out_features = 0;
    std::vector<int64_t> out_shape;
    Tensor out = do_gemm(x, weight, total_batch, out_features, out_shape);

    int64_t total = total_batch * out_features;
    int grid = (int)ceildiv(total, (int64_t)kBlockSize);
    if (gelu) {
        fused_bias_gelu_kernel<<<grid, kBlockSize>>>(out.data_f32(), bf.data_f32(), total, out_features);
    } else {
        fused_bias_sigmoid_kernel<<<grid, kBlockSize>>>(out.data_f32(), bf.data_f32(), total, out_features);
    }
    CUDA_CHECK(cudaGetLastError());

    Tensor result = out.reshape(out_shape);
    return (orig == DType::Float16) ? result.to_f16() : result;
}

}  // namespace

namespace ops {

Tensor linear_gelu(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    return linear_activation(x, weight, bias, true);
}

Tensor linear_sigmoid(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    return linear_activation(x, weight, bias, false);
}

void overlap_add(Tensor& dest, const Tensor& src, const Tensor& window, int64_t offset) {
    int64_t chunk_len = src.size(-1);
    int64_t num_channels = src.numel() / chunk_len;
    int64_t total = num_channels * chunk_len;
    int grid = (int)ceildiv(total, (int64_t)kBlockSize);
    overlap_add_kernel<<<grid, kBlockSize>>>(
        dest.data_f32(), src.data_f32(), window.data_f32(),
        offset, chunk_len, num_channels, dest.size(-1));
    CUDA_CHECK(cudaGetLastError());
}

void weight_accumulate(Tensor& weight_sum, const Tensor& window, int64_t offset) {
    int64_t chunk_len = window.numel();
    int grid = (int)ceildiv(chunk_len, (int64_t)kBlockSize);
    weight_accumulate_kernel<<<grid, kBlockSize>>>(
        weight_sum.data_f32(), window.data_f32(), offset, chunk_len);
    CUDA_CHECK(cudaGetLastError());
}

void normalize_by_weights(Tensor& data, const Tensor& weight_sum) {
    int64_t total_len = data.size(-1);
    int64_t num_channels = data.numel() / total_len;
    int64_t total = num_channels * total_len;
    int grid = (int)ceildiv(total, (int64_t)kBlockSize);
    normalize_by_weights_kernel<<<grid, kBlockSize>>>(
        data.data_f32(), weight_sum.data_f32(), total_len, num_channels);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace ops
}  // namespace cudasep
