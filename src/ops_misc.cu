#include "ops.h"

#include <stdexcept>

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

__global__ void rotary_emb_kernel(float* q, float* k,
                                  const float* cos_f, const float* sin_f,
                                  int B, int H, int N, int D) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int half_D = D / 2;
    int64_t total = (int64_t)B * H * N * half_D;
    if (idx >= total) return;

    int i = idx % half_D;
    int n = (idx / half_D) % N;
    int h = (idx / ((int64_t)half_D * N)) % H;
    int b = (int)(idx / ((int64_t)half_D * N * H));

    float c = cos_f[n * half_D + i];
    float s = sin_f[n * half_D + i];
    int64_t base = ((int64_t)b * H + h) * N * D + (int64_t)n * D;

    float q0 = q[base + 2 * i];
    float q1 = q[base + 2 * i + 1];
    q[base + 2 * i] = q0 * c - q1 * s;
    q[base + 2 * i + 1] = q0 * s + q1 * c;

    float k0 = k[base + 2 * i];
    float k1 = k[base + 2 * i + 1];
    k[base + 2 * i] = k0 * c - k1 * s;
    k[base + 2 * i + 1] = k0 * s + k1 * c;
}

__global__ void scatter_add_kernel(float* dest,
                                   const int64_t* indices,
                                   const float* src,
                                   int64_t total,
                                   int64_t dim_size_src,
                                   int64_t inner_size,
                                   int64_t dim_size_dest) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int64_t inner = idx % inner_size;
    int64_t outer = idx / (dim_size_src * inner_size);
    int64_t target_dim = indices[idx];
    int64_t dest_idx = outer * (dim_size_dest * inner_size) + target_dim * inner_size + inner;
    atomicAdd(&dest[dest_idx], src[idx]);
}

}  // namespace

namespace ops {

void apply_rotary_emb(Tensor& q, Tensor& k,
                      const Tensor& cos_freqs, const Tensor& sin_freqs) {
    if (q.ndim() != 4 || k.ndim() != 4) {
        throw std::runtime_error("apply_rotary_emb: q and k must be 4-D [B,H,N,D]");
    }

    bool need_cast = (q.dtype() == DType::Float16);
    Tensor q_work = need_cast ? q.to_f32().contiguous() : q.contiguous();
    Tensor k_work = need_cast ? k.to_f32().contiguous() : k.contiguous();
    Tensor cos_work = ensure_f32(cos_freqs);
    Tensor sin_work = ensure_f32(sin_freqs);

    int B = (int)q_work.size(0);
    int H = (int)q_work.size(1);
    int N = (int)q_work.size(2);
    int D = (int)q_work.size(3);
    int64_t total = (int64_t)B * H * N * (D / 2);
    int grid = (int)ceildiv(total, (int64_t)kBlockSize);

    rotary_emb_kernel<<<grid, kBlockSize>>>(
        q_work.data_f32(), k_work.data_f32(),
        cos_work.data_f32(), sin_work.data_f32(),
        B, H, N, D);
    CUDA_CHECK(cudaGetLastError());

    q = need_cast ? q_work.to_f16() : q_work;
    k = need_cast ? k_work.to_f16() : k_work;
}

void scatter_add(Tensor& dest, int dim, const Tensor& indices, const Tensor& src) {
    if (indices.dtype() != DType::Int64) {
        throw std::runtime_error("scatter_add: indices must be Int64");
    }
    if (src.numel() != indices.numel()) {
        throw std::runtime_error("scatter_add: src and indices must have same number of elements");
    }

    int ndim = src.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("scatter_add: dim out of range");
    }

    bool dest_was_f16 = (dest.dtype() == DType::Float16);
    Tensor dest_f32 = ensure_f32(dest);
    Tensor src_f32 = ensure_f32(src);
    Tensor idx_c = indices.contiguous();

    int64_t inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) {
        inner_size *= src.size(i);
    }

    int64_t dim_size_src = src.size(dim);
    int64_t dim_size_dest = dest.size(dim);
    int64_t total = src_f32.numel();
    int grid = (int)ceildiv(total, (int64_t)kBlockSize);

    scatter_add_kernel<<<grid, kBlockSize>>>(
        dest_f32.data_f32(), idx_c.data_i64(), src_f32.data_f32(),
        total, dim_size_src, inner_size, dim_size_dest);
    CUDA_CHECK(cudaGetLastError());

    dest = dest_was_f16 ? dest_f32.to_f16() : dest_f32;
}

}  // namespace ops
}  // namespace cudasep
