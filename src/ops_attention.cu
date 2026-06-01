#include "ops.h"

#include <cassert>
#include <cfloat>
#include <cmath>
#include <stdexcept>

namespace cudasep {

namespace {

constexpr int kBlockSize = 256;

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

__global__ void fused_scale_softmax_kernel(float* __restrict__ data,
                                           float scale,
                                           int rows,
                                           int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float smem[];
    float* row_data = data + (int64_t)row * cols;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    constexpr int kMaxElems = 32;
    float reg[kMaxElems];
    int elems_per_thread = (cols + num_threads - 1) / num_threads;

    float thread_max = -FLT_MAX;
    for (int i = 0; i < elems_per_thread; ++i) {
        int j = tid + i * num_threads;
        float value = (j < cols) ? row_data[j] * scale : -FLT_MAX;
        reg[i] = value;
        if (value > thread_max) thread_max = value;
    }

    smem[tid] = thread_max;
    __syncthreads();
    for (int s = num_threads >> 1; s > 0; s >>= 1) {
        if (tid < s && smem[tid + s] > smem[tid]) smem[tid] = smem[tid + s];
        __syncthreads();
    }
    float max_val = smem[0];
    __syncthreads();

    float thread_sum = 0.0f;
    for (int i = 0; i < elems_per_thread; ++i) {
        int j = tid + i * num_threads;
        if (j < cols) {
            float e = __expf(reg[i] - max_val);
            reg[i] = e;
            thread_sum += e;
        }
    }

    smem[tid] = thread_sum;
    __syncthreads();
    for (int s = num_threads >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / smem[0];
    __syncthreads();

    for (int i = 0; i < elems_per_thread; ++i) {
        int j = tid + i * num_threads;
        if (j < cols) {
            row_data[j] = reg[i] * inv_sum;
        }
    }
}

__global__ void fused_scale_softmax_to_half_kernel(const float* __restrict__ data,
                                                   __half* __restrict__ out,
                                                   float scale,
                                                   int rows,
                                                   int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float smem[];
    const float* row_data = data + (int64_t)row * cols;
    __half* row_out = out + (int64_t)row * cols;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    constexpr int kMaxElems = 32;
    float reg[kMaxElems];
    int elems_per_thread = (cols + num_threads - 1) / num_threads;

    float thread_max = -FLT_MAX;
    for (int i = 0; i < elems_per_thread; ++i) {
        int j = tid + i * num_threads;
        float value = (j < cols) ? row_data[j] * scale : -FLT_MAX;
        reg[i] = value;
        if (value > thread_max) thread_max = value;
    }

    smem[tid] = thread_max;
    __syncthreads();
    for (int s = num_threads >> 1; s > 0; s >>= 1) {
        if (tid < s && smem[tid + s] > smem[tid]) smem[tid] = smem[tid + s];
        __syncthreads();
    }
    float max_val = smem[0];
    __syncthreads();

    float thread_sum = 0.0f;
    for (int i = 0; i < elems_per_thread; ++i) {
        int j = tid + i * num_threads;
        if (j < cols) {
            float e = __expf(reg[i] - max_val);
            reg[i] = e;
            thread_sum += e;
        }
    }

    smem[tid] = thread_sum;
    __syncthreads();
    for (int s = num_threads >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / smem[0];
    __syncthreads();

    for (int i = 0; i < elems_per_thread; ++i) {
        int j = tid + i * num_threads;
        if (j < cols) {
            row_out[j] = __float2half(reg[i] * inv_sum);
        }
    }
}

void launch_materialized_attention(cublasHandle_t handle,
                                   const Tensor& qf,
                                   const Tensor& kf,
                                   const Tensor& vf,
                                   const Tensor* vh,
                                   bool use_fp16_value_gemm,
                                   float scale,
                                   Tensor& out,
                                   Tensor& scores,
                                   Tensor* scores_half) {
    const int64_t B = qf.size(0);
    const int64_t H = qf.size(1);
    const int64_t N = qf.size(2);
    const int64_t D = qf.size(3);
    const int64_t N_k = kf.size(2);
    const int64_t BH = B * H;

    float alpha = 1.0f;
    float beta = 0.0f;
    check_cublas(cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        (int)N_k, (int)N, (int)D,
        &alpha,
        kf.data_f32(), CUDA_R_32F, (int)D, (long long)(N_k * D),
        qf.data_f32(), CUDA_R_32F, (int)D, (long long)(N * D),
        &beta,
        scores.data_f32(), CUDA_R_32F, (int)N_k, (long long)(N * N_k),
        (int)BH,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP), "attention QK GEMM");

    int total_rows = (int)(BH * N);
    int cols = (int)N_k;
    int threads = 1;
    while (threads < cols && threads < kBlockSize) threads <<= 1;
    size_t smem_bytes = threads * sizeof(float);
    if (use_fp16_value_gemm) {
        fused_scale_softmax_to_half_kernel<<<total_rows, threads, smem_bytes>>>(
            scores.data_f32(), scores_half->data_f16(), scale, total_rows, cols);
    } else {
        fused_scale_softmax_kernel<<<total_rows, threads, smem_bytes>>>(
            scores.data_f32(), scale, total_rows, cols);
    }
    CUDA_CHECK(cudaGetLastError());

    if (use_fp16_value_gemm) {
        check_cublas(cublasGemmStridedBatchedEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            (int)D, (int)N, (int)N_k,
            &alpha,
            vh->data_ptr(), CUDA_R_16F, (int)D, (long long)(N_k * D),
            scores_half->data_ptr(), CUDA_R_16F, (int)N_k, (long long)(N * N_k),
            &beta,
            out.data_f32(), CUDA_R_32F, (int)D, (long long)(N * D),
            (int)BH,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP), "attention AV FP16 GEMM");
    } else {
        check_cublas(cublasGemmStridedBatchedEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            (int)D, (int)N, (int)N_k,
            &alpha,
            vf.data_f32(), CUDA_R_32F, (int)D, (long long)(N_k * D),
            scores.data_f32(), CUDA_R_32F, (int)N_k, (long long)(N * N_k),
            &beta,
            out.data_f32(), CUDA_R_32F, (int)D, (long long)(N * D),
            (int)BH,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP), "attention AV FP32 GEMM");
    }
}

}  // namespace

namespace ops {

Tensor scaled_dot_product_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                    float scale, float dropout) {
    (void)dropout;
    assert(q.ndim() == 4 && k.ndim() == 4 && v.ndim() == 4);

    const int64_t B = q.size(0);
    const int64_t H = q.size(1);
    const int64_t N = q.size(2);
    const int64_t D = q.size(3);
    const int64_t N_k = k.size(2);
    assert(k.size(0) == B && k.size(1) == H && k.size(3) == D);
    assert(v.size(0) == B && v.size(1) == H && v.size(2) == N_k && v.size(3) == D);

    DType orig_dtype = q.dtype();
    bool use_fp16_value_gemm = g_quantize_fp16 && D <= 128;
    if (scale == 0.0f) {
        scale = 1.0f / sqrtf(static_cast<float>(D));
    }

    Tensor qf = ensure_f32(q);
    Tensor kf = ensure_f32(k);
    Tensor vf = ensure_f32(v);
    Tensor vh;
    if (use_fp16_value_gemm) {
        vh = vf.to_f16().contiguous();
    }

    Tensor out = Tensor::empty({B * H, N, D}, DType::Float32);
    Tensor scores = Tensor::empty({B * H, N, N_k}, DType::Float32);
    Tensor scores_half;
    if (use_fp16_value_gemm) {
        scores_half = Tensor::empty({B * H, N, N_k}, DType::Float16);
    }

    launch_materialized_attention(CudaContext::instance().cublas(),
                                  qf,
                                  kf,
                                  vf,
                                  use_fp16_value_gemm ? &vh : nullptr,
                                  use_fp16_value_gemm,
                                  scale,
                                  out,
                                  scores,
                                  use_fp16_value_gemm ? &scores_half : nullptr);

    return maybe_cast_back(out.reshape({B, H, N, D}), orig_dtype);
}

}  // namespace ops
}  // namespace cudasep
