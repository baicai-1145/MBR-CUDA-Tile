#include "ops.h"

#include <cassert>

namespace cudasep {

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

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < num_warps) ? smem[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

__global__ void rms_norm_warp_kernel(const float* __restrict__ x,
                                     const float* __restrict__ gamma,
                                     float* __restrict__ out,
                                     int D,
                                     float scale,
                                     int64_t num_rows) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (row >= num_rows) return;

    const float* row_in = x + (int64_t)row * D;
    float* row_out = out + (int64_t)row * D;

    float sum_sq = 0.0f;
    for (int j = lane; j < D; j += 32) {
        float v = row_in[j];
        sum_sq += v * v;
    }
    sum_sq = warp_reduce_sum(sum_sq);
    float inv_norm = rsqrtf(sum_sq + 1e-12f);

    for (int j = lane; j < D; j += 32) {
        row_out[j] = row_in[j] * inv_norm * gamma[j] * scale;
    }
}

__global__ void rms_norm_block_kernel(const float* __restrict__ x,
                                      const float* __restrict__ gamma,
                                      float* __restrict__ out,
                                      int D,
                                      float scale,
                                      int64_t num_rows) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    extern __shared__ float smem[];
    const float* row_in = x + (int64_t)row * D;
    float* row_out = out + (int64_t)row * D;

    float sum_sq = 0.0f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float v = row_in[j];
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq, smem);
    float inv_norm = rsqrtf(sum_sq + 1e-12f);

    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        row_out[j] = row_in[j] * inv_norm * gamma[j] * scale;
    }
}

}  // namespace

namespace ops {

Tensor rms_norm(const Tensor& x, const Tensor& gamma, float scale) {
    DType orig = x.dtype();
    Tensor xf = ensure_f32(x);
    Tensor gf = ensure_f32(gamma);

    int ndim = xf.ndim();
    assert(ndim >= 1);
    int D = (int)xf.size(ndim - 1);
    int64_t num_rows = xf.numel() / D;
    assert(gf.numel() == D);

    Tensor out = Tensor::empty(xf.shape(), DType::Float32);
    if (D <= 1024) {
        int warps_per_block = kBlockSize / 32;
        int threads = warps_per_block * 32;
        int blocks = (int)ceildiv(num_rows, (int64_t)warps_per_block);
        rms_norm_warp_kernel<<<blocks, threads>>>(
            xf.data_f32(), gf.data_f32(), out.data_f32(), D, scale, num_rows);
    } else {
        int threads = kBlockSize;
        int num_warps = (threads + 31) / 32;
        rms_norm_block_kernel<<<(int)num_rows, threads, num_warps * sizeof(float)>>>(
            xf.data_f32(), gf.data_f32(), out.data_f32(), D, scale, num_rows);
    }
    CUDA_CHECK(cudaGetLastError());
    return maybe_cast_back(out, orig);
}

}  // namespace ops
}  // namespace cudasep
