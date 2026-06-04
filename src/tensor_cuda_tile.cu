#include "tensor_cuda_tile.h"

#include "cuda_tile.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>

namespace cudasep::tensor_tile {
namespace {

namespace ct = cuda::tiles;
using namespace ct::literals;

constexpr int kTile = 256;
constexpr int kReduceTile = 2048;
using I64Tile = ct::tile<long long, ct::shape<kTile>>;
using F32Tile = ct::tile<float, ct::shape<kTile>>;
using F16Tile = ct::tile<__half, ct::shape<kTile>>;
using BF16Tile = ct::tile<__nv_bfloat16, ct::shape<kTile>>;
using ReduceI64Tile = ct::tile<long long, ct::shape<kReduceTile>>;
using ReduceF32Tile = ct::tile<float, ct::shape<kReduceTile>>;

static inline int64_t ceildiv(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

static inline int grid_size(int64_t n) {
    return (int)ceildiv(n, kTile);
}

template <int TileSize>
static inline int grid_size_for(int64_t n) {
    return (int)ceildiv(n, TileSize);
}

static int binary_inplace_bf16_tile_size() {
    static int tile = []() {
        const char* raw = std::getenv("CUDASEP_BINARY_INPLACE_BF16_TILE");
        if (!raw || !raw[0]) return 256;
        int parsed = std::atoi(raw);
        if (parsed == 128 || parsed == 512 || parsed == 1024) return parsed;
        return 256;
    }();
    return tile;
}

template <typename T>
__tile_global__ void fill_kernel(T* __restrict__ dst, T value, long long total) {
    dst = ct::assume_aligned(dst, 16_ic);
    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto zeros = ct::load_masked(dst + idx, in_bounds) * T(0);
    ct::store_masked(dst + idx, zeros + value, in_bounds);
}

__tile_global__ void arange_f32_kernel(float* __restrict__ dst,
                                       long long start,
                                       long long total) {
    dst = ct::assume_aligned(dst, 16_ic);
    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    ct::store_masked(dst + idx, ct::element_cast<float>(idx + start), in_bounds);
}

__tile_global__ void arange_f16_kernel(__half* __restrict__ dst,
                                       long long start,
                                       long long total) {
    dst = ct::assume_aligned(dst, 16_ic);
    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    F32Tile values = ct::element_cast<float>(idx + start);
    ct::store_masked(dst + idx, ct::element_cast<__half>(values), in_bounds);
}

__tile_global__ void arange_i64_kernel(int64_t* __restrict__ dst,
                                       long long start,
                                       long long total) {
    dst = ct::assume_aligned(dst, 16_ic);
    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    ct::store_masked(dst + idx, idx + start, in_bounds);
}

__tile_global__ void f32_to_f16_kernel(const float* __restrict__ src,
                                       __half* __restrict__ dst,
                                       long long total) {
    src = ct::assume_aligned(src, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);
    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto values = ct::load_masked(src + idx, in_bounds);
    ct::store_masked(dst + idx, ct::element_cast<__half>(values), in_bounds);
}

__tile_global__ void f16_to_f32_kernel(const __half* __restrict__ src,
                                       float* __restrict__ dst,
                                       long long total) {
    src = ct::assume_aligned(src, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);
    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto values = ct::load_masked(src + idx, in_bounds);
    ct::store_masked(dst + idx, ct::element_cast<float>(values), in_bounds);
}

__tile_global__ void f32_to_bf16_kernel(const float* __restrict__ src,
                                        __nv_bfloat16* __restrict__ dst,
                                        long long total) {
    src = ct::assume_aligned(src, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);
    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto values = ct::load_masked(src + idx, in_bounds);
    ct::store_masked(dst + idx, ct::element_cast<__nv_bfloat16>(values), in_bounds);
}

__tile_global__ void bf16_to_f32_kernel(const __nv_bfloat16* __restrict__ src,
                                        float* __restrict__ dst,
                                        long long total) {
    src = ct::assume_aligned(src, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);
    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto values = ct::load_masked(src + idx, in_bounds);
    ct::store_masked(dst + idx, ct::element_cast<float>(values), in_bounds);
}

__tile_global__ void i64_to_f32_kernel(const int64_t* __restrict__ src,
                                       float* __restrict__ dst,
                                       long long total) {
    src = ct::assume_aligned(src, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);
    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto values = ct::load_masked(src + idx, in_bounds);
    ct::store_masked(dst + idx, ct::element_cast<float>(values), in_bounds);
}

__tile_global__ void f32_to_i64_kernel(const float* __restrict__ src,
                                       int64_t* __restrict__ dst,
                                       long long total) {
    src = ct::assume_aligned(src, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);
    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto values = ct::load_masked(src + idx, in_bounds);
    ct::store_masked(dst + idx, ct::element_cast<long long>(values), in_bounds);
}

__tile_global__ void i64_to_f16_kernel(const int64_t* __restrict__ src,
                                       __half* __restrict__ dst,
                                       long long total) {
    src = ct::assume_aligned(src, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);
    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    F32Tile values = ct::element_cast<float>(ct::load_masked(src + idx, in_bounds));
    ct::store_masked(dst + idx, ct::element_cast<__half>(values), in_bounds);
}

__tile_global__ void f16_to_i64_kernel(const __half* __restrict__ src,
                                       int64_t* __restrict__ dst,
                                       long long total) {
    src = ct::assume_aligned(src, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);
    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    F32Tile values = ct::element_cast<float>(ct::load_masked(src + idx, in_bounds));
    ct::store_masked(dst + idx, ct::element_cast<long long>(values), in_bounds);
}

__tile_global__ void i64_to_bf16_kernel(const int64_t* __restrict__ src,
                                        __nv_bfloat16* __restrict__ dst,
                                        long long total) {
    src = ct::assume_aligned(src, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);
    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    F32Tile values = ct::element_cast<float>(ct::load_masked(src + idx, in_bounds));
    ct::store_masked(dst + idx, ct::element_cast<__nv_bfloat16>(values), in_bounds);
}

__tile_global__ void bf16_to_i64_kernel(const __nv_bfloat16* __restrict__ src,
                                        int64_t* __restrict__ dst,
                                        long long total) {
    src = ct::assume_aligned(src, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);
    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    F32Tile values = ct::element_cast<float>(ct::load_masked(src + idx, in_bounds));
    ct::store_masked(dst + idx, ct::element_cast<long long>(values), in_bounds);
}

template <typename T>
__tile_global__ void add_scalar_kernel(T* __restrict__ data, T value, long long total) {
    data = ct::assume_aligned(data, 16_ic);
    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto values = ct::load_masked(data + idx, in_bounds) + value;
    ct::store_masked(data + idx, values, in_bounds);
}

template <typename T>
__tile_global__ void mul_scalar_kernel(T* __restrict__ data, T value, long long total) {
    data = ct::assume_aligned(data, 16_ic);
    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto values = ct::load_masked(data + idx, in_bounds) * value;
    ct::store_masked(data + idx, values, in_bounds);
}

__tile_global__ void clamp_f32_kernel(float* __restrict__ data,
                                      float min_value,
                                      float max_value,
                                      long long total) {
    data = ct::assume_aligned(data, 16_ic);
    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto values = ct::load_masked(data + idx, in_bounds);
    values = ct::min(ct::max(values, min_value), max_value);
    ct::store_masked(data + idx, values, in_bounds);
}

__tile_global__ void clamp_f16_kernel(__half* __restrict__ data,
                                      __half min_value,
                                      __half max_value,
                                      long long total) {
    data = ct::assume_aligned(data, 16_ic);
    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto values = ct::load_masked(data + idx, in_bounds);
    values = ct::min(ct::max(values, min_value), max_value);
    ct::store_masked(data + idx, values, in_bounds);
}

template <typename T>
__tile_global__ void negate_kernel(const T* __restrict__ src,
                                   T* __restrict__ dst,
                                   long long total) {
    src = ct::assume_aligned(src, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);
    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    ct::store_masked(dst + idx, -ct::load_masked(src + idx, in_bounds), in_bounds);
}

template <typename T>
__tile_global__ void greater_than_kernel(const T* __restrict__ src,
                                         T* __restrict__ dst,
                                         T value,
                                         long long total) {
    src = ct::assume_aligned(src, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);
    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto values = ct::load_masked(src + idx, in_bounds);
    auto zero = values * T(0);
    ct::store_masked(dst + idx, ct::select(values > value, zero + T(1), zero), in_bounds);
}

__tile_global__ void sum_reduce_last_dim_kernel(const float* __restrict__ src,
                                                float* __restrict__ dst,
                                                long long inner_size) {
    src = ct::assume_aligned(src, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);

    long long row = (long long)ct::bid().x;
    ReduceI64Tile i = ct::iota<ReduceI64Tile>();
    auto in_bounds = i < inner_size;
    auto values = ct::load_masked(src + row * inner_size + i, in_bounds);
    auto sum = ct::sum<0>(ct::select(in_bounds, values, values * 0.0f));
    ct::store_masked(dst + row + ct::iota<ct::tile<long long, ct::shape<1>>>(), sum, true);
}

__tile_global__ void max_reduce_last_dim_kernel(const float* __restrict__ src,
                                                float* __restrict__ dst,
                                                long long inner_size) {
    src = ct::assume_aligned(src, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);

    long long row = (long long)ct::bid().x;
    ReduceI64Tile i = ct::iota<ReduceI64Tile>();
    auto in_bounds = i < inner_size;
    auto values = ct::load_masked(src + row * inner_size + i, in_bounds);
    auto floor = values * 0.0f - 3.402823466e38f;
    auto m = ct::reduce_max<0>(ct::select(in_bounds, values, floor));
    ct::store_masked(dst + row + ct::iota<ct::tile<long long, ct::shape<1>>>(), m, true);
}

template <BinaryOp Op, int TileSize, typename T>
__tile_global__ void binary_same_shape_kernel(const T* __restrict__ a,
                                              const T* __restrict__ b,
                                              T* __restrict__ out,
                                              long long total) {
    a = ct::assume_aligned(a, 16_ic);
    b = ct::assume_aligned(b, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    using BinaryI64Tile = ct::tile<long long, ct::shape<TileSize>>;
    BinaryI64Tile idx = (long long)ct::bid().x * TileSize + ct::iota<BinaryI64Tile>();
    auto in_bounds = idx < total;
    auto av = ct::load_masked(a + idx, in_bounds);
    auto bv = ct::load_masked(b + idx, in_bounds);
    auto value = av + bv;
    if constexpr (Op == BinaryOp::Sub) {
        value = av - bv;
    } else if constexpr (Op == BinaryOp::Mul) {
        value = av * bv;
    } else if constexpr (Op == BinaryOp::Div) {
        value = av / bv;
    }
    ct::store_masked(out + idx, value, in_bounds);
}

template <BinaryOp Op, int TileSize, typename T>
__tile_global__ void binary_inplace_same_shape_kernel(T* __restrict__ a,
                                                      const T* __restrict__ b,
                                                      long long total) {
    a = ct::assume_aligned(a, 16_ic);
    b = ct::assume_aligned(b, 16_ic);

    using BinaryI64Tile = ct::tile<long long, ct::shape<TileSize>>;
    BinaryI64Tile idx = (long long)ct::bid().x * TileSize + ct::iota<BinaryI64Tile>();
    auto in_bounds = idx < total;
    auto av = ct::load_masked(a + idx, in_bounds);
    auto bv = ct::load_masked(b + idx, in_bounds);
    auto value = av + bv;
    if constexpr (Op == BinaryOp::Sub) {
        value = av - bv;
    } else if constexpr (Op == BinaryOp::Mul) {
        value = av * bv;
    } else if constexpr (Op == BinaryOp::Div) {
        value = av / bv;
    }
    ct::store_masked(a + idx, value, in_bounds);
}

template <BinaryOp Op, int TileSize = kTile>
static void launch_binary_same_shape(const Tensor& a, const Tensor& b, Tensor& out) {
    int grid = grid_size_for<TileSize>(out.numel());
    if (out.dtype() == DType::Float32) {
        binary_same_shape_kernel<Op, TileSize><<<grid, 1>>>(a.data_f32(), b.data_f32(),
                                                            out.data_f32(), out.numel());
    } else if (out.dtype() == DType::Float16) {
        binary_same_shape_kernel<Op, TileSize><<<grid, 1>>>((const __half*)a.data_ptr(),
                                                            (const __half*)b.data_ptr(),
                                                            out.data_f16(), out.numel());
    } else if (out.dtype() == DType::BFloat16) {
        binary_same_shape_kernel<Op, TileSize><<<grid, 1>>>((const __nv_bfloat16*)a.data_ptr(),
                                                            (const __nv_bfloat16*)b.data_ptr(),
                                                            out.data_bf16(), out.numel());
    } else {
        throw std::runtime_error("binary op: unsupported dtype");
    }
}

template <BinaryOp Op, int TileSize>
static void launch_binary_inplace_same_shape_tiled(Tensor& a, const Tensor& b) {
    int grid = grid_size_for<TileSize>(a.numel());
    if (a.dtype() == DType::Float32) {
        binary_inplace_same_shape_kernel<Op, TileSize><<<grid, 1>>>(a.data_f32(),
                                                                    b.data_f32(),
                                                                    a.numel());
    } else if (a.dtype() == DType::Float16) {
        binary_inplace_same_shape_kernel<Op, TileSize><<<grid, 1>>>(a.data_f16(),
                                                                    (const __half*)b.data_ptr(),
                                                                    a.numel());
    } else if (a.dtype() == DType::BFloat16) {
        binary_inplace_same_shape_kernel<Op, TileSize><<<grid, 1>>>(
            a.data_bf16(), (const __nv_bfloat16*)b.data_ptr(), a.numel());
    } else {
        throw std::runtime_error("inplace binary op: unsupported dtype");
    }
}

template <BinaryOp Op>
static void launch_binary_inplace_same_shape(Tensor& a, const Tensor& b) {
    if (a.dtype() == DType::BFloat16) {
        switch (binary_inplace_bf16_tile_size()) {
            case 128:
                launch_binary_inplace_same_shape_tiled<Op, 128>(a, b);
                break;
            case 512:
                launch_binary_inplace_same_shape_tiled<Op, 512>(a, b);
                break;
            case 1024:
                launch_binary_inplace_same_shape_tiled<Op, 1024>(a, b);
                break;
            default:
                launch_binary_inplace_same_shape_tiled<Op, 256>(a, b);
                break;
        }
        return;
    }
    launch_binary_inplace_same_shape_tiled<Op, 256>(a, b);
}

}  // namespace

void fill(Tensor& dst, float value) {
    if (dst.numel() == 0) return;
    int grid = grid_size(dst.numel());
    switch (dst.dtype()) {
        case DType::Float32:
            fill_kernel<<<grid, 1>>>(dst.data_f32(), value, dst.numel());
            break;
        case DType::Float16:
            fill_kernel<<<grid, 1>>>(dst.data_f16(), __float2half(value), dst.numel());
            break;
        case DType::BFloat16:
            fill_kernel<<<grid, 1>>>(dst.data_bf16(), __float2bfloat16(value), dst.numel());
            break;
        case DType::Int64:
            fill_kernel<<<grid, 1>>>(dst.data_i64(), (int64_t)value, dst.numel());
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}

void arange(Tensor& dst, int64_t start) {
    if (dst.numel() == 0) return;
    int grid = grid_size(dst.numel());
    switch (dst.dtype()) {
        case DType::Float32:
            arange_f32_kernel<<<grid, 1>>>(dst.data_f32(), start, dst.numel());
            break;
        case DType::Float16:
            arange_f16_kernel<<<grid, 1>>>(dst.data_f16(), start, dst.numel());
            break;
        case DType::BFloat16:
            throw std::runtime_error("arange: unsupported BF16 dtype");
        case DType::Int64:
            arange_i64_kernel<<<grid, 1>>>(dst.data_i64(), start, dst.numel());
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}

void convert_dtype(const Tensor& src, Tensor& dst) {
    if (src.numel() == 0) return;
    int grid = grid_size(src.numel());
    DType in = src.dtype();
    DType out = dst.dtype();

    if (in == DType::Float32 && out == DType::Float16) {
        f32_to_f16_kernel<<<grid, 1>>>(src.data_f32(), dst.data_f16(), src.numel());
    } else if (in == DType::Float16 && out == DType::Float32) {
        f16_to_f32_kernel<<<grid, 1>>>((const __half*)src.data_ptr(), dst.data_f32(), src.numel());
    } else if (in == DType::Float32 && out == DType::BFloat16) {
        f32_to_bf16_kernel<<<grid, 1>>>(src.data_f32(), dst.data_bf16(), src.numel());
    } else if (in == DType::BFloat16 && out == DType::Float32) {
        bf16_to_f32_kernel<<<grid, 1>>>((const __nv_bfloat16*)src.data_ptr(), dst.data_f32(), src.numel());
    } else if (in == DType::Int64 && out == DType::Float32) {
        i64_to_f32_kernel<<<grid, 1>>>(src.data_i64(), dst.data_f32(), src.numel());
    } else if (in == DType::Float32 && out == DType::Int64) {
        f32_to_i64_kernel<<<grid, 1>>>(src.data_f32(), dst.data_i64(), src.numel());
    } else if (in == DType::Int64 && out == DType::Float16) {
        i64_to_f16_kernel<<<grid, 1>>>(src.data_i64(), dst.data_f16(), src.numel());
    } else if (in == DType::Float16 && out == DType::Int64) {
        f16_to_i64_kernel<<<grid, 1>>>((const __half*)src.data_ptr(), dst.data_i64(), src.numel());
    } else if (in == DType::Int64 && out == DType::BFloat16) {
        i64_to_bf16_kernel<<<grid, 1>>>(src.data_i64(), dst.data_bf16(), src.numel());
    } else if (in == DType::BFloat16 && out == DType::Int64) {
        bf16_to_i64_kernel<<<grid, 1>>>((const __nv_bfloat16*)src.data_ptr(), dst.data_i64(), src.numel());
    } else {
        throw std::runtime_error("to_dtype: unsupported conversion");
    }
    CUDA_CHECK(cudaGetLastError());
}

void add_scalar(Tensor& data, float value) {
    if (data.numel() == 0) return;
    int grid = grid_size(data.numel());
    if (data.dtype() == DType::Float32) {
        add_scalar_kernel<<<grid, 1>>>(data.data_f32(), value, data.numel());
    } else if (data.dtype() == DType::Float16) {
        add_scalar_kernel<<<grid, 1>>>(data.data_f16(), __float2half(value), data.numel());
    } else {
        throw std::runtime_error("add_scalar_: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}

void mul_scalar(Tensor& data, float value) {
    if (data.numel() == 0) return;
    int grid = grid_size(data.numel());
    if (data.dtype() == DType::Float32) {
        mul_scalar_kernel<<<grid, 1>>>(data.data_f32(), value, data.numel());
    } else if (data.dtype() == DType::Float16) {
        mul_scalar_kernel<<<grid, 1>>>(data.data_f16(), __float2half(value), data.numel());
    } else {
        throw std::runtime_error("mul_scalar_: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}

void clamp(Tensor& data, float min_value, float max_value) {
    if (data.numel() == 0) return;
    int grid = grid_size(data.numel());
    if (data.dtype() == DType::Float32) {
        clamp_f32_kernel<<<grid, 1>>>(data.data_f32(), min_value, max_value, data.numel());
    } else if (data.dtype() == DType::Float16) {
        clamp_f16_kernel<<<grid, 1>>>(data.data_f16(), __float2half(min_value),
                                      __float2half(max_value), data.numel());
    } else {
        throw std::runtime_error("clamp_: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}

void negate(const Tensor& src, Tensor& dst) {
    if (src.numel() == 0) return;
    int grid = grid_size(src.numel());
    if (src.dtype() == DType::Float32) {
        negate_kernel<<<grid, 1>>>(src.data_f32(), dst.data_f32(), src.numel());
    } else if (src.dtype() == DType::Float16) {
        negate_kernel<<<grid, 1>>>((const __half*)src.data_ptr(), dst.data_f16(), src.numel());
    } else {
        throw std::runtime_error("negate: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}

void greater_than(const Tensor& src, Tensor& dst, float value) {
    if (src.numel() == 0) return;
    int grid = grid_size(src.numel());
    if (src.dtype() == DType::Float32) {
        greater_than_kernel<<<grid, 1>>>(src.data_f32(), dst.data_f32(), value, src.numel());
    } else if (src.dtype() == DType::Float16) {
        greater_than_kernel<<<grid, 1>>>((const __half*)src.data_ptr(), dst.data_f16(),
                                         __float2half(value), src.numel());
    } else {
        throw std::runtime_error("operator>: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}

void binary_same_shape(const Tensor& a, const Tensor& b, Tensor& out, BinaryOp op) {
    if (out.numel() == 0) return;
    switch (op) {
        case BinaryOp::Add:
            launch_binary_same_shape<BinaryOp::Add>(a, b, out);
            break;
        case BinaryOp::Sub:
            launch_binary_same_shape<BinaryOp::Sub>(a, b, out);
            break;
        case BinaryOp::Mul:
            launch_binary_same_shape<BinaryOp::Mul>(a, b, out);
            break;
        case BinaryOp::Div:
            launch_binary_same_shape<BinaryOp::Div>(a, b, out);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}

void binary_inplace_same_shape(Tensor& a, const Tensor& b, BinaryOp op) {
    if (a.numel() == 0) return;
    switch (op) {
        case BinaryOp::Add:
            launch_binary_inplace_same_shape<BinaryOp::Add>(a, b);
            break;
        case BinaryOp::Sub:
            launch_binary_inplace_same_shape<BinaryOp::Sub>(a, b);
            break;
        case BinaryOp::Mul:
            launch_binary_inplace_same_shape<BinaryOp::Mul>(a, b);
            break;
        case BinaryOp::Div:
            launch_binary_inplace_same_shape<BinaryOp::Div>(a, b);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}

void sum_reduce_last_dim(const Tensor& src, Tensor& dst,
                         int64_t inner_size, int64_t outer_size) {
    if (inner_size > kReduceTile) {
        throw std::runtime_error("sum: last dimension exceeds tile size");
    }
    sum_reduce_last_dim_kernel<<<(int)outer_size, 1>>>(src.data_f32(), dst.data_f32(), inner_size);
    CUDA_CHECK(cudaGetLastError());
}

void max_reduce_last_dim(const Tensor& src, Tensor& dst,
                         int64_t inner_size, int64_t outer_size) {
    if (inner_size > kReduceTile) {
        throw std::runtime_error("max: last dimension exceeds tile size");
    }
    max_reduce_last_dim_kernel<<<(int)outer_size, 1>>>(src.data_f32(), dst.data_f32(), inner_size);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace cudasep::tensor_tile
