#include "tensor_cuda_tile.h"

#include "cuda_tile.h"

#include <cuda_fp16.h>
#include <stdexcept>

namespace cudasep::tensor_tile {
namespace {

namespace ct = cuda::tiles;
using namespace ct::literals;

constexpr int kTile = 256;
using I64Tile = ct::tile<long long, ct::shape<kTile>>;
using F32Tile = ct::tile<float, ct::shape<kTile>>;
using F16Tile = ct::tile<__half, ct::shape<kTile>>;
using ByteTile = ct::tile<unsigned char, ct::shape<kTile>>;

static inline int64_t ceildiv(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

static inline int grid_size(int64_t n) {
    return (int)ceildiv(n, kTile);
}

template <typename Meta>
Meta* upload_single_meta(const Meta& meta) {
    static Meta* d_meta = nullptr;
    if (d_meta == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_meta, sizeof(Meta)));
    }
    CUDA_CHECK(cudaMemcpy(d_meta, &meta, sizeof(Meta), cudaMemcpyHostToDevice));
    return d_meta;
}

static __tile__ I64Tile broadcast_offset(I64Tile linear_idx,
                                         const int64_t* out_shape,
                                         const int64_t* src_strides,
                                         int ndim) {
    I64Tile offset = linear_idx * 0LL;
    for (int d = ndim - 1; d >= 0; --d) {
        auto coord = linear_idx % out_shape[d];
        linear_idx = linear_idx / out_shape[d];
        offset = offset + coord * src_strides[d];
    }
    return offset;
}

template <typename T>
static __tile__ T apply_binary(T a, T b, BinaryOp op) {
    auto value = a + b;
    value = ct::select(op == BinaryOp::Sub, a - b, value);
    value = ct::select(op == BinaryOp::Mul, a * b, value);
    value = ct::select(op == BinaryOp::Div, a / b, value);
    return value;
}

template <typename T>
__tile_global__ void binary_kernel(const T* __restrict__ a,
                                   const T* __restrict__ b,
                                   T* __restrict__ out,
                                   const BroadcastMeta* __restrict__ meta,
                                   BinaryOp op,
                                   long long total) {
    a = ct::assume_aligned(a, 16_ic);
    b = ct::assume_aligned(b, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto a_off = broadcast_offset(idx, meta->out_shape, meta->a_strides, meta->ndim);
    auto b_off = broadcast_offset(idx, meta->out_shape, meta->b_strides, meta->ndim);
    auto av = ct::load_masked(a + a_off, in_bounds);
    auto bv = ct::load_masked(b + b_off, in_bounds);
    ct::store_masked(out + idx, apply_binary(av, bv, op), in_bounds);
}

template <typename T>
__tile_global__ void binary_inplace_kernel(T* __restrict__ a,
                                           const T* __restrict__ b,
                                           const BroadcastMeta* __restrict__ meta,
                                           BinaryOp op,
                                           long long total) {
    a = ct::assume_aligned(a, 16_ic);
    b = ct::assume_aligned(b, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto b_off = broadcast_offset(idx, meta->out_shape, meta->b_strides, meta->ndim);
    auto av = ct::load_masked(a + idx, in_bounds);
    auto bv = ct::load_masked(b + b_off, in_bounds);
    ct::store_masked(a + idx, apply_binary(av, bv, op), in_bounds);
}

__tile_global__ void index_select_kernel(const unsigned char* __restrict__ src,
                                         unsigned char* __restrict__ dst,
                                         const int64_t* __restrict__ indices,
                                         const StridedCopyMeta* __restrict__ meta,
                                         int select_dim,
                                         int elem_size,
                                         long long total) {
    src = ct::assume_aligned(src, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);
    indices = ct::assume_aligned(indices, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto tmp = idx;
    I64Tile src_offset = idx * 0LL;
    for (int d = meta->ndim - 1; d >= 0; --d) {
        auto coord = tmp % meta->dst_shape[d];
        tmp = tmp / meta->dst_shape[d];
        auto src_coord = coord;
        if (d == select_dim) {
            src_coord = ct::load_masked(indices + coord, in_bounds);
        }
        src_offset = src_offset + src_coord * meta->src_strides[d];
    }
    for (int i = 0; i < elem_size; ++i) {
        auto values = ct::load_masked(src + src_offset * elem_size + i, in_bounds);
        ct::store_masked(dst + idx * elem_size + i, values, in_bounds);
    }
}

template <typename T>
__tile_global__ void strided_copy_value_kernel(const T* __restrict__ src,
                                               T* __restrict__ dst,
                                               const StridedCopyMeta* __restrict__ meta,
                                               long long total) {
    src = ct::assume_aligned(src, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto tmp = idx;
    I64Tile src_offset = idx * 0LL;
    for (int d = meta->ndim - 1; d >= 0; --d) {
        auto coord = tmp % meta->dst_shape[d];
        tmp = tmp / meta->dst_shape[d];
        src_offset = src_offset + coord * meta->src_strides[d];
    }
    auto values = ct::load_masked(src + src_offset, in_bounds);
    ct::store_masked(dst + idx, values, in_bounds);
}

__tile_global__ void strided_copy_bytes_kernel(const unsigned char* __restrict__ src,
                                               unsigned char* __restrict__ dst,
                                               const StridedCopyMeta* __restrict__ meta,
                                               int elem_size,
                                               long long total) {
    src = ct::assume_aligned(src, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto tmp = idx;
    I64Tile src_offset = idx * 0LL;
    for (int d = meta->ndim - 1; d >= 0; --d) {
        auto coord = tmp % meta->dst_shape[d];
        tmp = tmp / meta->dst_shape[d];
        src_offset = src_offset + coord * meta->src_strides[d];
    }
    for (int i = 0; i < elem_size; ++i) {
        auto values = ct::load_masked(src + src_offset * elem_size + i, in_bounds);
        ct::store_masked(dst + idx * elem_size + i, values, in_bounds);
    }
}

__tile_global__ void pad_const_kernel(const unsigned char* __restrict__ src,
                                      unsigned char* __restrict__ dst,
                                      const PadMeta* __restrict__ meta,
                                      float pad_value,
                                      int dtype_code,
                                      long long total) {
    src = ct::assume_aligned(src, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto tmp = idx;
    I64Tile src_linear = idx * 0LL;
    long long src_stride = 1;
    auto inside = in_bounds;
    for (int d = meta->ndim - 1; d >= 0; --d) {
        auto coord = tmp % meta->dst_shape[d];
        tmp = tmp / meta->dst_shape[d];
        auto src_coord = coord - meta->pad_before[d];
        inside = inside && (src_coord >= 0) && (src_coord < meta->src_shape[d]);
        src_linear = src_linear + src_coord * src_stride;
        src_stride *= meta->src_shape[d];
    }

    if (dtype_code == 0) {
        auto pad = ct::element_cast<float>(idx * 0LL) + pad_value;
        auto values = ct::load_masked(reinterpret_cast<const float*>(src) + src_linear, inside);
        ct::store_masked(reinterpret_cast<float*>(dst) + idx, pad, in_bounds);
        ct::store_masked(reinterpret_cast<float*>(dst) + idx, values, inside);
    } else if (dtype_code == 1) {
        F16Tile pad = ct::element_cast<__half>(ct::element_cast<float>(idx * 0LL) + pad_value);
        auto values = ct::load_masked(reinterpret_cast<const __half*>(src) + src_linear, inside);
        ct::store_masked(reinterpret_cast<__half*>(dst) + idx, pad, in_bounds);
        ct::store_masked(reinterpret_cast<__half*>(dst) + idx, values, inside);
    } else {
        auto pad = idx * 0LL + (long long)pad_value;
        auto values = ct::load_masked(reinterpret_cast<const long long*>(src) + src_linear, inside);
        ct::store_masked(reinterpret_cast<long long*>(dst) + idx, pad, in_bounds);
        ct::store_masked(reinterpret_cast<long long*>(dst) + idx, values, inside);
    }
}

__tile_global__ void pad_reflect_kernel(const unsigned char* __restrict__ src,
                                        unsigned char* __restrict__ dst,
                                        const PadMeta* __restrict__ meta,
                                        int elem_size,
                                        long long total) {
    src = ct::assume_aligned(src, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto tmp = idx;
    I64Tile src_linear = idx * 0LL;
    long long src_stride = 1;
    for (int d = meta->ndim - 1; d >= 0; --d) {
        auto src_coord = (tmp % meta->dst_shape[d]) - meta->pad_before[d];
        tmp = tmp / meta->dst_shape[d];
        src_coord = ct::select(src_coord < 0, -src_coord, src_coord);
        src_coord = ct::select(src_coord >= meta->src_shape[d],
                               2LL * (meta->src_shape[d] - 1) - src_coord,
                               src_coord);
        src_coord = ct::min(ct::max(src_coord, 0LL), meta->src_shape[d] - 1);
        src_linear = src_linear + src_coord * src_stride;
        src_stride *= meta->src_shape[d];
    }

    for (int i = 0; i < elem_size; ++i) {
        auto values = ct::load_masked(src + src_linear * elem_size + i, in_bounds);
        ct::store_masked(dst + idx * elem_size + i, values, in_bounds);
    }
}

__tile_global__ void cat_kernel(unsigned char* __restrict__ out,
                                const CatSrcInfo* __restrict__ infos,
                                long long outer_size,
                                long long total_cat_dim,
                                long long inner_bytes) {
    out = ct::assume_aligned(out, 16_ic);

    int tensor_idx = (int)ct::bid().y;
    const CatSrcInfo* info = infos + tensor_idx;
    const auto* src = static_cast<const unsigned char*>(info->ptr);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    long long src_bytes = outer_size * info->dim_size * inner_bytes;
    auto in_bounds = idx < src_bytes;

    long long dim_bytes = info->dim_size * inner_bytes;
    auto outer_idx = idx / dim_bytes;
    auto rem = idx % dim_bytes;
    auto dst_idx = outer_idx * total_cat_dim * inner_bytes +
                   info->dim_offset * inner_bytes + rem;

    ByteTile values = ct::load_masked(src + idx, in_bounds);
    ct::store_masked(out + dst_idx, values, in_bounds);
}

__tile_global__ void sum_reduce_general_kernel(const float* __restrict__ src,
                                               float* __restrict__ dst,
                                               long long reduce,
                                               long long inner,
                                               long long total) {
    src = ct::assume_aligned(src, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto outer_idx = idx / inner;
    auto inner_idx = idx % inner;
    auto value = ct::zeros<F32Tile>();
    for (long long r = 0; r < reduce; ++r) {
        value = value + ct::load_masked(src + outer_idx * reduce * inner + r * inner + inner_idx,
                                     in_bounds);
    }
    ct::store_masked(dst + idx, value, in_bounds);
}

__tile_global__ void max_reduce_general_kernel(const float* __restrict__ src,
                                               float* __restrict__ dst,
                                               long long reduce,
                                               long long inner,
                                               long long total) {
    src = ct::assume_aligned(src, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto outer_idx = idx / inner;
    auto inner_idx = idx % inner;
    auto value = ct::element_cast<float>(idx * 0LL) - INFINITY;
    for (long long r = 0; r < reduce; ++r) {
        auto next = ct::load_masked(src + outer_idx * reduce * inner + r * inner + inner_idx,
                                    in_bounds);
        value = ct::max(value, next);
    }
    ct::store_masked(dst + idx, value, in_bounds);
}

}  // namespace

void binary_glue(const Tensor& a, const Tensor& b, Tensor& out,
                 const BroadcastMeta& meta, BinaryOp op) {
    int64_t total = out.numel();
    if (total == 0) return;
    int grid = grid_size(total);
    const BroadcastMeta* d_meta = upload_single_meta(meta);
    if (out.dtype() == DType::Float32) {
        binary_kernel<<<grid, 1>>>(a.data_f32(), b.data_f32(), out.data_f32(),
                                   d_meta, op, total);
    } else if (out.dtype() == DType::Float16) {
        binary_kernel<<<grid, 1>>>((const __half*)a.data_ptr(), (const __half*)b.data_ptr(),
                                   out.data_f16(), d_meta, op, total);
    } else {
        throw std::runtime_error("binary op: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}

void binary_inplace_glue(Tensor& a, const Tensor& b,
                         const BroadcastMeta& meta, BinaryOp op) {
    int64_t total = a.numel();
    if (total == 0) return;
    int grid = grid_size(total);
    const BroadcastMeta* d_meta = upload_single_meta(meta);
    if (a.dtype() == DType::Float32) {
        binary_inplace_kernel<<<grid, 1>>>(a.data_f32(), b.data_f32(), d_meta, op, total);
    } else if (a.dtype() == DType::Float16) {
        binary_inplace_kernel<<<grid, 1>>>(a.data_f16(), (const __half*)b.data_ptr(),
                                           d_meta, op, total);
    } else {
        throw std::runtime_error("inplace binary op: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}

void index_select_glue(const Tensor& src, Tensor& dst, const Tensor& indices,
                       const StridedCopyMeta& meta, int select_dim) {
    int64_t total = dst.numel();
    if (total == 0) return;
    const StridedCopyMeta* d_meta = upload_single_meta(meta);
    index_select_kernel<<<grid_size(total), 1>>>(
        (const unsigned char*)src.data_ptr(), (unsigned char*)dst.data_ptr(), indices.data_i64(), d_meta,
        select_dim, (int)dtype_size(src.dtype()), total);
    CUDA_CHECK(cudaGetLastError());
}

void strided_copy_glue(const Tensor& src, Tensor& dst, const StridedCopyMeta& meta) {
    int64_t total = dst.numel();
    if (total == 0) return;
    int grid = grid_size(total);
    const StridedCopyMeta* d_meta = upload_single_meta(meta);
    if (src.dtype() == DType::Float32) {
        strided_copy_value_kernel<<<grid, 1>>>(src.data_f32(), dst.data_f32(), d_meta, total);
    } else if (src.dtype() == DType::Float16) {
        strided_copy_value_kernel<<<grid, 1>>>((const __half*)src.data_ptr(), dst.data_f16(),
                                               d_meta, total);
    } else {
        strided_copy_bytes_kernel<<<grid, 1>>>(
            (const unsigned char*)src.data_ptr(), (unsigned char*)dst.data_ptr(), d_meta,
            (int)dtype_size(src.dtype()), total);
    }
    CUDA_CHECK(cudaGetLastError());
}

void pad_const_glue(const Tensor& src, Tensor& dst, const PadMeta& meta, float value) {
    int64_t total = dst.numel();
    if (total == 0) return;
    const PadMeta* d_meta = upload_single_meta(meta);
    pad_const_kernel<<<grid_size(total), 1>>>(
        (const unsigned char*)src.data_ptr(), (unsigned char*)dst.data_ptr(),
        d_meta, value, (int)src.dtype(), total);
    CUDA_CHECK(cudaGetLastError());
}

void pad_reflect_glue(const Tensor& src, Tensor& dst, const PadMeta& meta) {
    int64_t total = dst.numel();
    if (total == 0) return;
    const PadMeta* d_meta = upload_single_meta(meta);
    pad_reflect_kernel<<<grid_size(total), 1>>>(
        (const unsigned char*)src.data_ptr(), (unsigned char*)dst.data_ptr(), d_meta,
        (int)dtype_size(src.dtype()), total);
    CUDA_CHECK(cudaGetLastError());
}

void cat_glue(char* out, const CatSrcInfo* infos, int n_infos,
              int64_t outer_size, int64_t total_cat_dim,
              int64_t inner_bytes, int64_t max_src_bytes) {
    static CatSrcInfo* d_infos = nullptr;
    static size_t d_infos_cap = 0;
    size_t infos_bytes = (size_t)n_infos * sizeof(CatSrcInfo);
    if (infos_bytes > d_infos_cap) {
        if (d_infos) cudaFree(d_infos);
        CUDA_CHECK(cudaMalloc(&d_infos, infos_bytes));
        d_infos_cap = infos_bytes;
    }
    CUDA_CHECK(cudaMemcpy(d_infos, infos, infos_bytes, cudaMemcpyHostToDevice));

    dim3 grid((unsigned)ceildiv(max_src_bytes, kTile), (unsigned)n_infos);
    cat_kernel<<<grid, 1>>>((unsigned char*)out, d_infos, outer_size, total_cat_dim, inner_bytes);
    CUDA_CHECK(cudaGetLastError());
}

void sum_reduce_general_glue(const Tensor& src, Tensor& dst,
                             int64_t outer, int64_t reduce, int64_t inner) {
    int64_t total = outer * inner;
    if (total == 0) return;
    sum_reduce_general_kernel<<<grid_size(total), 1>>>(
        src.data_f32(), dst.data_f32(), reduce, inner, total);
    CUDA_CHECK(cudaGetLastError());
}

void max_reduce_general_glue(const Tensor& src, Tensor& dst,
                             int64_t outer, int64_t reduce, int64_t inner) {
    int64_t total = outer * inner;
    if (total == 0) return;
    max_reduce_general_kernel<<<grid_size(total), 1>>>(
        src.data_f32(), dst.data_f32(), reduce, inner, total);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace cudasep::tensor_tile
