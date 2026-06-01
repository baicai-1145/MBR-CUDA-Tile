#include "ops.h"

#include <cmath>

namespace cudasep {

static constexpr int kBlockSize = 256;

static inline int64_t ceildiv(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

static Tensor ensure_f32(const Tensor& x) {
    if (x.dtype() == DType::Float32) return x.contiguous();
    return x.to_f32().contiguous();
}

static Tensor maybe_cast_back(const Tensor& result, DType orig) {
    if (orig == DType::Float16) return result.to_f16();
    return result;
}

struct TanhOp {
    __device__ float operator()(float x) const {
        return tanhf(x);
    }
};

template <typename OpFunc>
__global__ void unary_kernel(const float* __restrict__ in,
                             float* __restrict__ out,
                             int64_t n,
                             OpFunc op) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = op(in[i]);
}

template <typename OpFunc>
static Tensor apply_unary(const Tensor& x, OpFunc op) {
    DType orig = x.dtype();
    Tensor xf = ensure_f32(x);
    Tensor out = Tensor::empty(xf.shape(), DType::Float32);

    int grid = (int)ceildiv(xf.numel(), (int64_t)kBlockSize);
    unary_kernel<<<grid, kBlockSize>>>(xf.data_f32(), out.data_f32(), xf.numel(), op);
    CUDA_CHECK(cudaGetLastError());
    return maybe_cast_back(out, orig);
}

__global__ void complex_mul_kernel(const float* __restrict__ a,
                                   const float* __restrict__ b,
                                   float* __restrict__ out,
                                   int64_t num_complex) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_complex) return;

    float ar = a[2 * i];
    float ai = a[2 * i + 1];
    float br = b[2 * i];
    float bi = b[2 * i + 1];
    out[2 * i] = ar * br - ai * bi;
    out[2 * i + 1] = ar * bi + ai * br;
}

__global__ void glu_fused_kernel(const float* __restrict__ x,
                                 float* __restrict__ out,
                                 int64_t half_dim,
                                 int64_t inner_size,
                                 int64_t full_dim_stride,
                                 int64_t total) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    int64_t inner_idx = i % inner_size;
    int64_t tmp = i / inner_size;
    int64_t dim_idx = tmp % half_dim;
    int64_t outer_idx = tmp / half_dim;

    int64_t base = outer_idx * full_dim_stride + inner_idx;
    float first = x[base + dim_idx * inner_size];
    float second = x[base + (dim_idx + half_dim) * inner_size];
    float gate = 1.0f / (1.0f + expf(-second));
    out[i] = first * gate;
}

__global__ void index_fill_kernel(float* __restrict__ data,
                                  int64_t dim_size,
                                  int64_t inner_size,
                                  int64_t target_idx,
                                  float value,
                                  int64_t total_outer) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_outer * inner_size) return;

    int64_t outer_idx = i / inner_size;
    int64_t in_idx = i % inner_size;
    int64_t linear = outer_idx * dim_size * inner_size + target_idx * inner_size + in_idx;
    data[linear] = value;
}

namespace ops {

Tensor tanh_act(const Tensor& x) {
    return apply_unary(x, TanhOp{});
}

Tensor complex_mul(const Tensor& a, const Tensor& b) {
    DType orig = a.dtype();
    Tensor af = ensure_f32(a);
    Tensor bf = ensure_f32(b);

    if (af.ndim() < 1 || af.size(af.ndim() - 1) != 2 || af.shape() != bf.shape()) {
        throw std::runtime_error("complex_mul expects matching tensors with last dim == 2");
    }

    int64_t num_complex = af.numel() / 2;
    Tensor out = Tensor::empty(af.shape(), DType::Float32);
    int grid = (int)ceildiv(num_complex, (int64_t)kBlockSize);
    complex_mul_kernel<<<grid, kBlockSize>>>(af.data_f32(), bf.data_f32(), out.data_f32(), num_complex);
    CUDA_CHECK(cudaGetLastError());
    return maybe_cast_back(out, orig);
}

Tensor glu(const Tensor& x, int dim) {
    DType orig = x.dtype();
    Tensor xf = ensure_f32(x);

    int ndim = xf.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim || (xf.size(dim) % 2) != 0) {
        throw std::runtime_error("glu: invalid split dimension");
    }

    int64_t half = xf.size(dim) / 2;
    int64_t full_dim = xf.size(dim);

    std::vector<int64_t> out_shape = xf.shape();
    out_shape[dim] = half;

    int64_t inner_size = 1;
    for (int d = dim + 1; d < ndim; d++) inner_size *= xf.size(d);

    int64_t total = 1;
    for (auto s : out_shape) total *= s;

    Tensor out = Tensor::empty(out_shape, DType::Float32);
    int grid = (int)ceildiv(total, (int64_t)kBlockSize);
    glu_fused_kernel<<<grid, kBlockSize>>>(
        xf.data_f32(), out.data_f32(), half, inner_size, full_dim * inner_size, total);
    CUDA_CHECK(cudaGetLastError());
    return maybe_cast_back(out, orig);
}

Tensor index_fill(const Tensor& x, int dim, int64_t index, float value) {
    DType orig = x.dtype();
    Tensor out = ensure_f32(x).clone();

    int ndim = out.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim || index < 0 || index >= out.size(dim)) {
        throw std::runtime_error("index_fill: index out of range");
    }

    int64_t outer = 1;
    for (int i = 0; i < dim; ++i) outer *= out.size(i);
    int64_t inner = 1;
    for (int i = dim + 1; i < ndim; ++i) inner *= out.size(i);

    int64_t total = outer * inner;
    int grid = (int)ceildiv(total, (int64_t)kBlockSize);
    index_fill_kernel<<<grid, kBlockSize>>>(
        out.data_f32(), out.size(dim), inner, index, value, outer);
    CUDA_CHECK(cudaGetLastError());
    return maybe_cast_back(out, orig);
}

}  // namespace ops
}  // namespace cudasep
