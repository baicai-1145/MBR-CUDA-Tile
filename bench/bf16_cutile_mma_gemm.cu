#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

#define CUDA_CHECK(call)                                                             \
    do {                                                                            \
        cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            throw std::runtime_error(std::string(#call) + " failed: " +             \
                                     cudaGetErrorString(err__));                    \
        }                                                                           \
    } while (0)

namespace ct = cuda::tiles;
using namespace ct::literals;

constexpr int kInitTile = 256;
constexpr double kA10gDenseBf16Tflops = 70.0;
constexpr int kQkvHeads = 8;
constexpr int kQkvDim = 64;
constexpr int kTimeSeq = 1301;
constexpr int kTimeBatches = 60;

using I64Tile = ct::tile<long long, ct::shape<kInitTile>>;
using F32Tile = ct::tile<float, ct::shape<kInitTile>>;
using BF16Tile = ct::tile<__nv_bfloat16, ct::shape<kInitTile>>;

struct Shape {
    const char* name;
    int m;
    int n;
    int k;
    int iters;
};

struct Options {
    std::string preset = "infer_linear";
    std::string variant = "all";
    std::string shape = "all";
    int warmup = 2;
    int iters_override = 0;
};

int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}

int parse_int_arg(const char* name, const char* value) {
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (!end || *end != '\0' || parsed <= 0 || parsed > 1000000L) {
        throw std::runtime_error(std::string("invalid value for ") + name + ": " + value);
    }
    return static_cast<int>(parsed);
}

Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (std::strcmp(argv[i], "--preset") == 0) {
            opts.preset = need_value(argv[i]);
        } else if (std::strcmp(argv[i], "--variant") == 0) {
            opts.variant = need_value(argv[i]);
        } else if (std::strcmp(argv[i], "--shape") == 0) {
            opts.shape = need_value(argv[i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0) {
            opts.warmup = parse_int_arg(argv[i], need_value(argv[i]));
        } else if (std::strcmp(argv[i], "--iters") == 0) {
            opts.iters_override = parse_int_arg(argv[i], need_value(argv[i]));
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::printf(
                "Usage: bench_bf16_cutile_mma_gemm [options]\n"
                "  --preset NAME   infer_linear, infer_small_linear; default infer_linear\n"
                "  --shape NAME    all, infer_qkv, infer_ffn1, infer_ffn2, infer_attn_out,\n"
                "                  gate_n8_k256,\n"
                "                  band_n256_k16, band_n256_k32, band_n256_k64,\n"
                "                  mask_hid_k256, mask_hid_k1024, mask_out48,\n"
                "                  mask_out128, mask_out400, mask_out1040\n"
                "  --variant NAME  all, t32x32x16, t32x64x16, t32x64x16u,\n"
                "                  t32x64x16m, t32x64x16s, t32x128x16s,\n"
                "                  t32x256x16s, t64x64x16s, t32x64x32s,\n"
                "                  t32x128x32s, t32x256x32s, t32x512x32s,\n"
                "                  qkv_t16x256x32s, qkv_t64x256x32s,\n"
                "                  qkv_t32x256x64s, qkv_t32x256x32mat,\n"
                "                  qkv_t32x256x32splitn128,\n"
                "                  qkv_t32x256x32manual, qkv_t32x256x32bkn,\n"
                "                  qkv_t32x32x32bkn, qkv_t32x64x32bkn,\n"
                "                  qkv_t32x128x32bkn,\n"
                "                  qkv_t32x256x32bkn_tiled,\n"
                "                  qkv_t32x256x32bkn_pairk,\n"
                "                  qkv_t32x256x32bkn_unroll8,\n"
                "                  qkv_t16x256x32bkn, qkv_t64x256x32bkn,\n"
                "                  qkv_t32x512x32bkn,\n"
                "                  qkv_t16x256x16bkn, qkv_t32x128x16bkn,\n"
                "                  qkv_t64x128x16bkn,\n"
                "                  qkv_t32x256x8bkn, qkv_t32x256x16bkn,\n"
                "                  qkv_t32x256x16bkn_pairk,\n"
                "                  qkv_t32x256x16bkn_loadtmp,\n"
                "                  qkv_t32x256x16bkn_matmul,\n"
                "                  qkv_t32x256x16bkn_k128,\n"
                "                  qkv_t32x256x16bkn_occ8,\n"
                "                  qkv_t32x256x16bkn_occ12,\n"
                "                  qkv_t32x256x16bkn_occ16,\n"
                "                  qkv_t32x256x16bkn_occ24,\n"
                "                  qkv_t32x256x16bkn_lat1,\n"
                "                  qkv_t32x256x16bkn_lat2,\n"
                "                  qkv_t64x128x16bkn_lat2,\n"
                "                  qkv_t32x256x16bkn_lat4,\n"
                "                  qkv_t32x256x16bkn_lat7,\n"
                "                  qkv_t32x256x16bkn_lat10,\n"
                "                  qkv_t32x256x16bkn_a2_b2_s0,\n"
                "                  qkv_t32x256x16bkn_a2_b1_s0,\n"
                "                  qkv_t32x256p128x16bkn,\n"
                "                  qkv_t64x256x16bkn, qkv_t32x512x16bkn,\n"
                "                  qkv_t32x256x16bkn_scatter_time,\n"
                "                  qkv_t32x256x16bkn_split_contig,\n"
                "                  qkv_t32x256x16bkn_split_contig_direct_store,\n"
                "                  qkv_t32x256x16bkn_split_contig_latecast,\n"
                "                  qkv_t32x256x16bkn_split_contig_lateview,\n"
                "                  qkv_t16x256x16bkn_split_contig_lat2,\n"
                "                  qkv_t32x32x16bkn_split_contig_lat2,\n"
                "                  qkv_t32x64x16bkn_split_contig_lat2,\n"
                "                  qkv_t32x128x16bkn_split_contig_lat2,\n"
                "                  qkv_t32x512x16bkn_split_contig,\n"
                "                  qkv_t32x256x16bkn_split_contig_lat2,\n"
                "                  qkv_t64x256x16bkn_split_contig_lat2,\n"
                "                  qkv_t32x256x16bkn_split_contig_direct_store_lat2,\n"
                "                  qkv_t32x256x16bkn_split_contig_latecast_lat2,\n"
                "                  qkv_t32x256x16bkn_split_contig_lateview_lat2,\n"
                "                  qkv_t32x512x16bkn_split_contig_lat2,\n"
                "                  qkv_t32x256x16bkn_split_contig_a2_b0_s0,\n"
                "                  qkv_t32x256x16bkn_split_contig_a0_b2_s0,\n"
                "                  qkv_t32x256x16bkn_split_contig_a0_b0_s2,\n"
                "                  qkv_t32x256x16bkn_split_contig_a2_b1_s2,\n"
                "                  qkv_t32x256x16bkn_split_contig_a1_b2_s2,\n"
                "                  qkv_t32x256x16bkn_split_contig_a2_b2_s0,\n"
                "                  qkv_t32x256x64bkn, qkv_t32x256x128bkn,\n"
                "                  t32x256x32bkn, t32x64x64s, t32x64x32,\n"
                "                  t32x64x32m, t32x64x64, t32x128x16, t64x64x16,\n"
                "                  t32x256x16, t64x64x16sm, t32x64x64sm,\n"
                "                  t64x64x16sp, t64x64x32sp, t32x64x64sp,\n"
                "                  t32x128x32sp, t32x128x64sp,\n"
                "                  t64x128x32sp, t32x256x32sp,\n"
                "                  t32x8x32smn, t32x16x32smn, t32x64x32smn,\n"
                "                  t32x64x64smn, default all\n"
                "                  attnres_t16x128x32, attnres_t32x64x32,\n"
                "                  attnres_t32x128x32, attnres_t32x256x32,\n"
                "                  attnres_t64x128x32\n"
                "  --warmup N      warmup launches per shape, default 2\n"
                "  --iters N       measured launches per shape, overrides shape default\n");
            std::exit(0);
        } else {
            throw std::runtime_error(std::string("unknown argument: ") + argv[i]);
        }
    }
    return opts;
}

std::vector<Shape> infer_linear_shapes() {
    return {
        {"infer_qkv", 78060, 1536, 256, 4},
        {"infer_ffn1", 78060, 1024, 256, 4},
        {"infer_ffn2", 78060, 256, 1024, 4},
        {"infer_attn_out", 78060, 256, 512, 4},
        {"gate_n8_k256", 78060, 8, 256, 8},
    };
}

std::vector<Shape> infer_small_linear_shapes() {
    return {
        {"band_n256_k16", 1301, 256, 16, 16},
        {"band_n256_k32", 1301, 256, 32, 16},
        {"band_n256_k64", 1301, 256, 64, 16},
        {"mask_hid_k256", 1301, 1024, 256, 8},
        {"mask_hid_k1024", 1301, 1024, 1024, 8},
        {"mask_out48", 1301, 48, 1024, 16},
        {"mask_out128", 1301, 128, 1024, 16},
        {"mask_out400", 1301, 400, 1024, 8},
        {"mask_out1040", 1301, 1040, 1024, 8},
    };
}

float percentile(std::vector<float> values, float q) {
    std::sort(values.begin(), values.end());
    float pos = q * static_cast<float>(values.size() - 1);
    int lo = static_cast<int>(pos);
    int hi = std::min(lo + 1, static_cast<int>(values.size() - 1));
    float t = pos - static_cast<float>(lo);
    return values[lo] * (1.0f - t) + values[hi] * t;
}

__tile_global__ void fill_bf16_kernel(__nv_bfloat16* __restrict__ dst, long long total) {
    dst = ct::assume_aligned(dst, 16_ic);
    I64Tile idx = (long long)ct::bid().x * kInitTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    F32Tile values = 0.25f + ct::element_cast<float>((idx * 13LL) & 1023LL) * 0.0009765625f;
    ct::store_masked(dst + idx, ct::element_cast<__nv_bfloat16>(values), in_bounds);
}

template <int TM, int TN, int TK, bool Masked, bool UseMatmul>
__tile_global__ void cutile_mma_gemm_nt_bf16_kernel(const __nv_bfloat16* __restrict__ a,
                                                    const __nv_bfloat16* __restrict__ b_nt,
                                                    __nv_bfloat16* __restrict__ c,
                                                    std::size_t m,
                                                    std::size_t n,
                                                    std::size_t k) {
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_nt = ct::assume_aligned(b_nt, 16_ic);
    c = ct::assume_aligned(c, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::extents{m, k}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_nt, ct::extents{k, n}, ct::layout_left{}},
        ct::shape<TK, TN>{}
    };
    auto c_view = ct::partition_view{
        ct::tensor_span{c, ct::extents{m, n}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    std::size_t k_tiles = (k + static_cast<std::size_t>(TK) - 1) / static_cast<std::size_t>(TK);
    for (auto kk : ct::irange(std::size_t{0}, k_tiles)) {
        if constexpr (Masked) {
            auto a_tile = a_view.load_masked(tile_m, kk);
            auto b_tile = b_view.load_masked(kk, tile_n);
            if constexpr (UseMatmul) {
                acc = acc + ct::matmul(a_tile, b_tile);
            } else {
                acc = ct::mma(a_tile, b_tile, acc);
            }
        } else {
            auto a_tile = a_view.load(tile_m, kk);
            auto b_tile = b_view.load(kk, tile_n);
            if constexpr (UseMatmul) {
                acc = acc + ct::matmul(a_tile, b_tile);
            } else {
                acc = ct::mma(a_tile, b_tile, acc);
            }
        }
    }
    if constexpr (Masked) {
        c_view.store_masked(ct::element_cast<__nv_bfloat16>(acc), tile_m, tile_n);
    } else {
        c_view.store(ct::element_cast<__nv_bfloat16>(acc), tile_m, tile_n);
    }
}

template <int TM, int TN, int TK, int M, int N, int K, bool UseMatmul = false>
__tile_global__ void cutile_mma_gemm_nt_bf16_static_kernel(const __nv_bfloat16* __restrict__ a,
                                                           const __nv_bfloat16* __restrict__ b_nt,
                                                           __nv_bfloat16* __restrict__ c) {
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_nt = ct::assume_aligned(b_nt, 16_ic);
    c = ct::assume_aligned(c, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_nt, ct::shape<K, N>{}, ct::layout_left{}},
        ct::shape<TK, TN>{}
    };
    auto c_view = ct::partition_view{
        ct::tensor_span{c, ct::shape<M, N>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        auto a_tile = a_view.load(tile_m, kk);
        auto b_tile = b_view.load(kk, tile_n);
        if constexpr (UseMatmul) {
            acc = acc + ct::matmul(a_tile, b_tile);
        } else {
            acc = ct::mma(a_tile, b_tile, acc);
        }
    }
    c_view.store(ct::element_cast<__nv_bfloat16>(acc), tile_m, tile_n);
}

template <int TM, int TN, int TK, int M, int N, int K>
__tile_global__ void attn_out_residual_static_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_nt,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ c) {
    static_assert(M % TM == 0);
    static_assert(N % TN == 0);
    static_assert(K % TK == 0);
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_nt = ct::assume_aligned(b_nt, 16_ic);
    residual = ct::assume_aligned(residual, 16_ic);
    c = ct::assume_aligned(c, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_nt, ct::shape<K, N>{}, ct::layout_left{}},
        ct::shape<TK, TN>{}
    };
    auto residual_view = ct::partition_view{
        ct::tensor_span{residual, ct::shape<M, N>{}},
        ct::shape<TM, TN>{}
    };
    auto c_view = ct::partition_view{
        ct::tensor_span{c, ct::shape<M, N>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        acc = ct::mma(a_view.load(tile_m, kk), b_view.load(kk, tile_n), acc);
    }
    auto value = ct::element_cast<__nv_bfloat16>(acc);
    value = value + residual_view.load(tile_m, tile_n);
    c_view.store(value, tile_m, tile_n);
}

template <int M, int N, int K>
__tile_global__ void qkv_split_n128_static_kernel(const __nv_bfloat16* __restrict__ a,
                                                  const __nv_bfloat16* __restrict__ b_nt,
                                                  __nv_bfloat16* __restrict__ c) {
    constexpr int TM = 32;
    constexpr int MacroTN = 256;
    constexpr int SubTN = 128;
    constexpr int TK = 32;
    using AccTile = ct::tile<float, ct::shape<TM, SubTN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_nt = ct::assume_aligned(b_nt, 16_ic);
    c = ct::assume_aligned(c, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_nt, ct::shape<K, N>{}, ct::layout_left{}},
        ct::shape<TK, SubTN>{}
    };
    auto c_view = ct::partition_view{
        ct::tensor_span{c, ct::shape<M, N>{}},
        ct::shape<TM, SubTN>{}
    };

    auto [tile_m, macro_tile_n, tile_z] = ct::bid();
    (void)tile_z;
    for (auto sub : ct::irange(std::size_t{0}, std::size_t{MacroTN / SubTN})) {
        auto tile_n = macro_tile_n * (MacroTN / SubTN) + sub;
        auto acc = ct::full<AccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
            acc = ct::mma(a_view.load(tile_m, kk), b_view.load(kk, tile_n), acc);
        }
        c_view.store(ct::element_cast<__nv_bfloat16>(acc), tile_m, tile_n);
    }
}

template <int M, int N, int K>
__tile_global__ void qkv_manual_static_kernel(const __nv_bfloat16* __restrict__ a,
                                              const __nv_bfloat16* __restrict__ b_nt,
                                              __nv_bfloat16* __restrict__ c) {
    constexpr int TM = 32;
    constexpr int TN = 256;
    constexpr int TK = 32;
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using ATileIndex = ct::tile<long long, ct::shape<TM, TK>>;
    using BTileIndex = ct::tile<long long, ct::shape<TK, TN>>;
    using CTileIndex = ct::tile<long long, ct::shape<TM, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_nt = ct::assume_aligned(b_nt, 16_ic);
    c = ct::assume_aligned(c, 16_ic);

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;

    auto a_index = ct::iota<ATileIndex>();
    auto a_row = a_index / TK;
    auto a_col = a_index - a_row * TK;
    auto b_index = ct::iota<BTileIndex>();
    auto b_row = b_index / TN;
    auto b_col = b_index - b_row * TN;
    auto c_index = ct::iota<CTileIndex>();
    auto c_row = c_index / TN;
    auto c_col = c_index - c_row * TN;

    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        auto a_offsets = ((long long)tile_m * TM + a_row) * K +
                         (long long)kk * TK + a_col;
        auto b_offsets = ((long long)tile_n * TN + b_col) * K +
                         (long long)kk * TK + b_row;
        auto a_tile = ct::load(a + a_offsets);
        auto b_tile = ct::load(b_nt + b_offsets);
        acc = ct::mma(a_tile, b_tile, acc);
    }

    auto c_offsets = ((long long)tile_m * TM + c_row) * N +
                     (long long)tile_n * TN + c_col;
    ct::store(c + c_offsets, ct::element_cast<__nv_bfloat16>(acc));
}

template <int M, int N, int K>
__tile_global__ void qkv_bkn_static_kernel(const __nv_bfloat16* __restrict__ a,
                                           const __nv_bfloat16* __restrict__ b_kn,
                                           __nv_bfloat16* __restrict__ c) {
    constexpr int TM = 32;
    constexpr int TN = 256;
    constexpr int TK = 32;
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_kn = ct::assume_aligned(b_kn, 16_ic);
    c = ct::assume_aligned(c, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_kn, ct::shape<K, N>{}},
        ct::shape<TK, TN>{}
    };
    auto c_view = ct::partition_view{
        ct::tensor_span{c, ct::shape<M, N>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        acc = ct::mma(a_view.load(tile_m, kk), b_view.load(kk, tile_n), acc);
    }
    c_view.store(ct::element_cast<__nv_bfloat16>(acc), tile_m, tile_n);
}

template <int M, int N, int K>
__tile_global__ void qkv_bkn_static_unroll8_kernel(const __nv_bfloat16* __restrict__ a,
                                                   const __nv_bfloat16* __restrict__ b_kn,
                                                   __nv_bfloat16* __restrict__ c) {
    static_assert(K == 256);
    constexpr int TM = 32;
    constexpr int TN = 256;
    constexpr int TK = 32;
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_kn = ct::assume_aligned(b_kn, 16_ic);
    c = ct::assume_aligned(c, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_kn, ct::shape<K, N>{}},
        ct::shape<TK, TN>{}
    };
    auto c_view = ct::partition_view{
        ct::tensor_span{c, ct::shape<M, N>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    acc = ct::mma(a_view.load(tile_m, 0), b_view.load(0, tile_n), acc);
    acc = ct::mma(a_view.load(tile_m, 1), b_view.load(1, tile_n), acc);
    acc = ct::mma(a_view.load(tile_m, 2), b_view.load(2, tile_n), acc);
    acc = ct::mma(a_view.load(tile_m, 3), b_view.load(3, tile_n), acc);
    acc = ct::mma(a_view.load(tile_m, 4), b_view.load(4, tile_n), acc);
    acc = ct::mma(a_view.load(tile_m, 5), b_view.load(5, tile_n), acc);
    acc = ct::mma(a_view.load(tile_m, 6), b_view.load(6, tile_n), acc);
    acc = ct::mma(a_view.load(tile_m, 7), b_view.load(7, tile_n), acc);
    c_view.store(ct::element_cast<__nv_bfloat16>(acc), tile_m, tile_n);
}

template <int M, int N, int K>
__tile_global__ void qkv_bkn_static_pairk_kernel(const __nv_bfloat16* __restrict__ a,
                                                 const __nv_bfloat16* __restrict__ b_kn,
                                                 __nv_bfloat16* __restrict__ c) {
    static_assert(K == 256);
    constexpr int TM = 32;
    constexpr int TN = 256;
    constexpr int TK = 32;
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_kn = ct::assume_aligned(b_kn, 16_ic);
    c = ct::assume_aligned(c, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_kn, ct::shape<K, N>{}},
        ct::shape<TK, TN>{}
    };
    auto c_view = ct::partition_view{
        ct::tensor_span{c, ct::shape<M, N>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kpair : ct::irange(std::size_t{0}, std::size_t{K / (2 * TK)})) {
        auto kk0 = kpair * 2;
        auto kk1 = kk0 + 1;
        acc = ct::mma(a_view.load(tile_m, kk0), b_view.load(kk0, tile_n), acc);
        acc = ct::mma(a_view.load(tile_m, kk1), b_view.load(kk1, tile_n), acc);
    }
    c_view.store(ct::element_cast<__nv_bfloat16>(acc), tile_m, tile_n);
}

template <int TM, int TN, int TK, int M, int N, int K>
__tile_global__ void qkv_bkn_static_tiled_kernel(const __nv_bfloat16* __restrict__ a,
                                                 const __nv_bfloat16* __restrict__ b_kn,
                                                 __nv_bfloat16* __restrict__ c) {
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_kn = ct::assume_aligned(b_kn, 16_ic);
    c = ct::assume_aligned(c, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_kn, ct::shape<K, N>{}},
        ct::shape<TK, TN>{}
    };
    auto c_view = ct::partition_view{
        ct::tensor_span{c, ct::shape<M, N>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        acc = ct::mma(a_view.load(tile_m, kk), b_view.load(kk, tile_n), acc);
    }
    c_view.store(ct::element_cast<__nv_bfloat16>(acc), tile_m, tile_n);
}

template <int TM, int TN, int TK, int M, int N, int K>
__tile_global__ void qkv_bkn_static_tiled_pairk_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_kn,
    __nv_bfloat16* __restrict__ c) {
    static_assert(K % (2 * TK) == 0);
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_kn = ct::assume_aligned(b_kn, 16_ic);
    c = ct::assume_aligned(c, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_kn, ct::shape<K, N>{}},
        ct::shape<TK, TN>{}
    };
    auto c_view = ct::partition_view{
        ct::tensor_span{c, ct::shape<M, N>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kpair : ct::irange(std::size_t{0}, std::size_t{K / (2 * TK)})) {
        auto kk0 = kpair * 2;
        auto kk1 = kk0 + 1;
        acc = ct::mma(a_view.load(tile_m, kk0), b_view.load(kk0, tile_n), acc);
        acc = ct::mma(a_view.load(tile_m, kk1), b_view.load(kk1, tile_n), acc);
    }
    c_view.store(ct::element_cast<__nv_bfloat16>(acc), tile_m, tile_n);
}

template <int TM, int TN, int TK, int M, int N, int K>
__tile_global__ void qkv_bkn_static_tiled_loadtmp_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_kn,
    __nv_bfloat16* __restrict__ c) {
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using BTile = ct::tile<__nv_bfloat16, ct::shape<TK, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_kn = ct::assume_aligned(b_kn, 16_ic);
    c = ct::assume_aligned(c, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_kn, ct::shape<K, N>{}},
        ct::shape<TK, TN>{}
    };
    auto c_view = ct::partition_view{
        ct::tensor_span{c, ct::shape<M, N>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        ATile a_tile = a_view.load(tile_m, kk);
        BTile b_tile = b_view.load(kk, tile_n);
        acc = ct::mma(a_tile, b_tile, acc);
    }
    c_view.store(ct::element_cast<__nv_bfloat16>(acc), tile_m, tile_n);
}

template <int TM, int TN, int TK, int M, int N, int K>
__tile_global__ void qkv_bkn_static_tiled_matmul_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_kn,
    __nv_bfloat16* __restrict__ c) {
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_kn = ct::assume_aligned(b_kn, 16_ic);
    c = ct::assume_aligned(c, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_kn, ct::shape<K, N>{}},
        ct::shape<TK, TN>{}
    };
    auto c_view = ct::partition_view{
        ct::tensor_span{c, ct::shape<M, N>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        acc = acc + ct::matmul(a_view.load(tile_m, kk), b_view.load(kk, tile_n));
    }
    c_view.store(ct::element_cast<__nv_bfloat16>(acc), tile_m, tile_n);
}

template <int TM, int TN, int TK, int M, int N, int KFull, int KRun>
__tile_global__ void qkv_bkn_static_tiled_kprefix_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_kn,
    __nv_bfloat16* __restrict__ c) {
    static_assert(KRun % TK == 0);
    static_assert(KRun <= KFull);
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using AShape = ct::shape<M, KRun>;
    using AStrides = ct::shape<KFull, 1>;
    using ALayout = ct::layout_strided<AStrides>;
    using AMapping = typename ALayout::template mapping<AShape>;

    a = ct::assume_aligned(a, 16_ic);
    b_kn = ct::assume_aligned(b_kn, 16_ic);
    c = ct::assume_aligned(c, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, AMapping{AShape{}, AStrides{}}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_kn, ct::shape<KRun, N>{}},
        ct::shape<TK, TN>{}
    };
    auto c_view = ct::partition_view{
        ct::tensor_span{c, ct::shape<M, N>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{KRun / TK})) {
        acc = ct::mma(a_view.load(tile_m, kk), b_view.load(kk, tile_n), acc);
    }
    c_view.store(ct::element_cast<__nv_bfloat16>(acc), tile_m, tile_n);
}

template <int TM, int TN, int TK, int M, int N, int K, int Occupancy>
[[ cutile::hint(0, occupancy=Occupancy) ]]
__tile_global__ void qkv_bkn_static_tiled_occ_kernel(const __nv_bfloat16* __restrict__ a,
                                                     const __nv_bfloat16* __restrict__ b_kn,
                                                     __nv_bfloat16* __restrict__ c) {
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_kn = ct::assume_aligned(b_kn, 16_ic);
    c = ct::assume_aligned(c, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_kn, ct::shape<K, N>{}},
        ct::shape<TK, TN>{}
    };
    auto c_view = ct::partition_view{
        ct::tensor_span{c, ct::shape<M, N>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        acc = ct::mma(a_view.load(tile_m, kk), b_view.load(kk, tile_n), acc);
    }
    c_view.store(ct::element_cast<__nv_bfloat16>(acc), tile_m, tile_n);
}

template <int TM,
          int TN,
          int TK,
          int M,
          int N,
          int K,
          int LoadLatency,
          int BLoadLatency = LoadLatency,
          int StoreLatency = LoadLatency>
__tile_global__ void qkv_bkn_static_tiled_latency_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_kn,
    __nv_bfloat16* __restrict__ c) {
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using BTile = ct::tile<__nv_bfloat16, ct::shape<TK, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_kn = ct::assume_aligned(b_kn, 16_ic);
    c = ct::assume_aligned(c, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_kn, ct::shape<K, N>{}},
        ct::shape<TK, TN>{}
    };
    auto c_view = ct::partition_view{
        ct::tensor_span{c, ct::shape<M, N>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        ATile a_tile;
        BTile b_tile;
        if constexpr (LoadLatency > 0) {
            [[ cutile::hint(0, latency=LoadLatency) ]]
            a_tile = a_view.load(tile_m, kk);
        } else {
            a_tile = a_view.load(tile_m, kk);
        }
        if constexpr (BLoadLatency > 0) {
            [[ cutile::hint(0, latency=BLoadLatency) ]]
            b_tile = b_view.load(kk, tile_n);
        } else {
            b_tile = b_view.load(kk, tile_n);
        }
        acc = ct::mma(a_tile, b_tile, acc);
    }
    if constexpr (StoreLatency > 0) {
        [[ cutile::hint(0, latency=StoreLatency) ]]
        c_view.store(ct::element_cast<__nv_bfloat16>(acc), tile_m, tile_n);
    } else {
        c_view.store(ct::element_cast<__nv_bfloat16>(acc), tile_m, tile_n);
    }
}

template <int TM, int TK, int M, int N, int K>
__tile_global__ void qkv_bkn_static_compound_256p128_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_kn,
    __nv_bfloat16* __restrict__ c) {
    static_assert(N % 384 == 0);
    using Acc0Tile = ct::tile<float, ct::shape<TM, 256>>;
    using Acc1Tile = ct::tile<float, ct::shape<TM, 128>>;
    using B0Shape = ct::shape<K, 256>;
    using B1Shape = ct::shape<K, 128>;
    using C0Shape = ct::shape<M, 256>;
    using C1Shape = ct::shape<M, 128>;
    using Strides = ct::shape<N, 1>;
    using StridedLayout = ct::layout_strided<Strides>;
    using B0Mapping = typename StridedLayout::template mapping<B0Shape>;
    using B1Mapping = typename StridedLayout::template mapping<B1Shape>;
    using C0Mapping = typename StridedLayout::template mapping<C0Shape>;
    using C1Mapping = typename StridedLayout::template mapping<C1Shape>;

    a = ct::assume_aligned(a, 16_ic);
    b_kn = ct::assume_aligned(b_kn, 16_ic);
    c = ct::assume_aligned(c, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };

    auto [tile_m, tile_ng, tile_z] = ct::bid();
    (void)tile_z;
    std::size_t n_base = static_cast<std::size_t>(tile_ng) * 384;

    auto b0_view = ct::partition_view{
        ct::tensor_span{b_kn + n_base, B0Mapping{B0Shape{}, Strides{}}},
        ct::shape<TK, 256>{}
    };
    auto b1_view = ct::partition_view{
        ct::tensor_span{b_kn + n_base + 256, B1Mapping{B1Shape{}, Strides{}}},
        ct::shape<TK, 128>{}
    };
    auto c0_view = ct::partition_view{
        ct::tensor_span{c + n_base, C0Mapping{C0Shape{}, Strides{}}},
        ct::shape<TM, 256>{}
    };
    auto c1_view = ct::partition_view{
        ct::tensor_span{c + n_base + 256, C1Mapping{C1Shape{}, Strides{}}},
        ct::shape<TM, 128>{}
    };

    auto acc0 = ct::full<Acc0Tile>(0.0f);
    auto acc1 = ct::full<Acc1Tile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        auto a_tile = a_view.load(tile_m, kk);
        acc0 = ct::mma(a_tile, b0_view.load(kk, 0), acc0);
        acc1 = ct::mma(a_tile, b1_view.load(kk, 0), acc1);
    }
    c0_view.store(ct::element_cast<__nv_bfloat16>(acc0), tile_m, 0);
    c1_view.store(ct::element_cast<__nv_bfloat16>(acc1), tile_m, 0);
}

template <int TM, int TN, int TK, int M, int N, int K>
__tile_global__ void qkv_bkn_static_tiled_scatter_time_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_kn,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v) {
    static_assert(N == 3 * kQkvHeads * kQkvDim);
    static_assert(K % TK == 0);
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_kn = ct::assume_aligned(b_kn, 16_ic);
    q = ct::assume_aligned(q, 16_ic);
    k_out = ct::assume_aligned(k_out, 16_ic);
    v = ct::assume_aligned(v, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_kn, ct::shape<K, N>{}},
        ct::shape<TK, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        acc = ct::mma(a_view.load(tile_m, kk), b_view.load(kk, tile_n), acc);
    }

    I64OutTile local = ct::iota<I64OutTile>();
    auto rows = static_cast<long long>(tile_m) * TM + local / TN;
    auto cols = static_cast<long long>(tile_n) * TN + local % TN;
    auto qkv_col = cols % (kQkvHeads * kQkvDim);
    auto head = qkv_col / kQkvDim;
    auto dim = qkv_col % kQkvDim;
    auto batch = rows / kTimeSeq;
    auto token = rows - batch * kTimeSeq;
    auto offsets =
        ((batch * kQkvHeads + head) * kTimeSeq + token) * kQkvDim + dim;
    auto out = ct::element_cast<__nv_bfloat16>(acc);

    if (tile_n < 2) {
        ct::store(q + offsets, out);
    } else if (tile_n < 4) {
        ct::store(k_out + offsets, out);
    } else {
        ct::store(v + offsets, out);
    }
}

template <int TM,
          int TN,
          int TK,
          int M,
          int N,
          int K,
          int LoadLatency = 0,
          int BLoadLatency = LoadLatency,
          int StoreLatency = LoadLatency,
          bool BranchLocalCast = false,
          bool BranchLocalView = false>
__tile_global__ void qkv_bkn_static_tiled_split_contig_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_kn,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v) {
    static_assert(N == 3 * kQkvHeads * kQkvDim);
    static_assert((N / 3) % TN == 0);
    static_assert(K % TK == 0);
    constexpr int kComponentTiles = (N / 3) / TN;
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using BTile = ct::tile<__nv_bfloat16, ct::shape<TK, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_kn = ct::assume_aligned(b_kn, 16_ic);
    q = ct::assume_aligned(q, 16_ic);
    k_out = ct::assume_aligned(k_out, 16_ic);
    v = ct::assume_aligned(v, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_kn, ct::shape<K, N>{}},
        ct::shape<TK, TN>{}
    };
    if constexpr (BranchLocalView) {
        auto [tile_m, tile_n, tile_z] = ct::bid();
        (void)tile_z;
        auto acc = ct::full<AccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
            if constexpr (LoadLatency > 0 || BLoadLatency > 0) {
                ATile a_tile;
                BTile b_tile;
                if constexpr (LoadLatency > 0) {
                    [[ cutile::hint(0, latency=LoadLatency) ]]
                    a_tile = a_view.load(tile_m, kk);
                } else {
                    a_tile = a_view.load(tile_m, kk);
                }
                if constexpr (BLoadLatency > 0) {
                    [[ cutile::hint(0, latency=BLoadLatency) ]]
                    b_tile = b_view.load(kk, tile_n);
                } else {
                    b_tile = b_view.load(kk, tile_n);
                }
                acc = ct::mma(a_tile, b_tile, acc);
            } else {
                acc = ct::mma(a_view.load(tile_m, kk), b_view.load(kk, tile_n), acc);
            }
        }

        if constexpr (BranchLocalCast) {
            if (tile_n < kComponentTiles) {
                auto q_view = ct::partition_view{
                    ct::tensor_span{q, ct::shape<M, N / 3>{}},
                    ct::shape<TM, TN>{}
                };
                if constexpr (StoreLatency > 0) {
                    [[ cutile::hint(0, latency=StoreLatency) ]]
                    q_view.store(ct::element_cast<__nv_bfloat16>(acc), tile_m, tile_n);
                } else {
                    q_view.store(ct::element_cast<__nv_bfloat16>(acc), tile_m, tile_n);
                }
            } else if (tile_n < 2 * kComponentTiles) {
                auto k_view = ct::partition_view{
                    ct::tensor_span{k_out, ct::shape<M, N / 3>{}},
                    ct::shape<TM, TN>{}
                };
                if constexpr (StoreLatency > 0) {
                    [[ cutile::hint(0, latency=StoreLatency) ]]
                    k_view.store(ct::element_cast<__nv_bfloat16>(acc),
                                 tile_m,
                                 tile_n - kComponentTiles);
                } else {
                    k_view.store(ct::element_cast<__nv_bfloat16>(acc),
                                 tile_m,
                                 tile_n - kComponentTiles);
                }
            } else {
                auto v_view = ct::partition_view{
                    ct::tensor_span{v, ct::shape<M, N / 3>{}},
                    ct::shape<TM, TN>{}
                };
                if constexpr (StoreLatency > 0) {
                    [[ cutile::hint(0, latency=StoreLatency) ]]
                    v_view.store(ct::element_cast<__nv_bfloat16>(acc),
                                 tile_m,
                                 tile_n - 2 * kComponentTiles);
                } else {
                    v_view.store(ct::element_cast<__nv_bfloat16>(acc),
                                 tile_m,
                                 tile_n - 2 * kComponentTiles);
                }
            }
        } else {
            auto out = ct::element_cast<__nv_bfloat16>(acc);
            if (tile_n < kComponentTiles) {
                auto q_view = ct::partition_view{
                    ct::tensor_span{q, ct::shape<M, N / 3>{}},
                    ct::shape<TM, TN>{}
                };
                if constexpr (StoreLatency > 0) {
                    [[ cutile::hint(0, latency=StoreLatency) ]]
                    q_view.store(out, tile_m, tile_n);
                } else {
                    q_view.store(out, tile_m, tile_n);
                }
            } else if (tile_n < 2 * kComponentTiles) {
                auto k_view = ct::partition_view{
                    ct::tensor_span{k_out, ct::shape<M, N / 3>{}},
                    ct::shape<TM, TN>{}
                };
                if constexpr (StoreLatency > 0) {
                    [[ cutile::hint(0, latency=StoreLatency) ]]
                    k_view.store(out, tile_m, tile_n - kComponentTiles);
                } else {
                    k_view.store(out, tile_m, tile_n - kComponentTiles);
                }
            } else {
                auto v_view = ct::partition_view{
                    ct::tensor_span{v, ct::shape<M, N / 3>{}},
                    ct::shape<TM, TN>{}
                };
                if constexpr (StoreLatency > 0) {
                    [[ cutile::hint(0, latency=StoreLatency) ]]
                    v_view.store(out, tile_m, tile_n - 2 * kComponentTiles);
                } else {
                    v_view.store(out, tile_m, tile_n - 2 * kComponentTiles);
                }
            }
        }
    } else {
        auto q_view = ct::partition_view{
            ct::tensor_span{q, ct::shape<M, N / 3>{}},
            ct::shape<TM, TN>{}
        };
        auto k_view = ct::partition_view{
            ct::tensor_span{k_out, ct::shape<M, N / 3>{}},
            ct::shape<TM, TN>{}
        };
        auto v_view = ct::partition_view{
            ct::tensor_span{v, ct::shape<M, N / 3>{}},
            ct::shape<TM, TN>{}
        };

        auto [tile_m, tile_n, tile_z] = ct::bid();
        (void)tile_z;
        auto acc = ct::full<AccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
            if constexpr (LoadLatency > 0 || BLoadLatency > 0) {
                ATile a_tile;
                BTile b_tile;
                if constexpr (LoadLatency > 0) {
                    [[ cutile::hint(0, latency=LoadLatency) ]]
                    a_tile = a_view.load(tile_m, kk);
                } else {
                    a_tile = a_view.load(tile_m, kk);
                }
                if constexpr (BLoadLatency > 0) {
                    [[ cutile::hint(0, latency=BLoadLatency) ]]
                    b_tile = b_view.load(kk, tile_n);
                } else {
                    b_tile = b_view.load(kk, tile_n);
                }
                acc = ct::mma(a_tile, b_tile, acc);
            } else {
                acc = ct::mma(a_view.load(tile_m, kk), b_view.load(kk, tile_n), acc);
            }
        }

        if constexpr (BranchLocalCast) {
            if (tile_n < kComponentTiles) {
                if constexpr (StoreLatency > 0) {
                    [[ cutile::hint(0, latency=StoreLatency) ]]
                    q_view.store(ct::element_cast<__nv_bfloat16>(acc), tile_m, tile_n);
                } else {
                    q_view.store(ct::element_cast<__nv_bfloat16>(acc), tile_m, tile_n);
                }
            } else if (tile_n < 2 * kComponentTiles) {
                if constexpr (StoreLatency > 0) {
                    [[ cutile::hint(0, latency=StoreLatency) ]]
                    k_view.store(ct::element_cast<__nv_bfloat16>(acc),
                                 tile_m,
                                 tile_n - kComponentTiles);
                } else {
                    k_view.store(ct::element_cast<__nv_bfloat16>(acc),
                                 tile_m,
                                 tile_n - kComponentTiles);
                }
            } else {
                if constexpr (StoreLatency > 0) {
                    [[ cutile::hint(0, latency=StoreLatency) ]]
                    v_view.store(ct::element_cast<__nv_bfloat16>(acc),
                                 tile_m,
                                 tile_n - 2 * kComponentTiles);
                } else {
                    v_view.store(ct::element_cast<__nv_bfloat16>(acc),
                                 tile_m,
                                 tile_n - 2 * kComponentTiles);
                }
            }
        } else {
            auto out = ct::element_cast<__nv_bfloat16>(acc);
            if (tile_n < kComponentTiles) {
                if constexpr (StoreLatency > 0) {
                    [[ cutile::hint(0, latency=StoreLatency) ]]
                    q_view.store(out, tile_m, tile_n);
                } else {
                    q_view.store(out, tile_m, tile_n);
                }
            } else if (tile_n < 2 * kComponentTiles) {
                if constexpr (StoreLatency > 0) {
                    [[ cutile::hint(0, latency=StoreLatency) ]]
                    k_view.store(out, tile_m, tile_n - kComponentTiles);
                } else {
                    k_view.store(out, tile_m, tile_n - kComponentTiles);
                }
            } else {
                if constexpr (StoreLatency > 0) {
                    [[ cutile::hint(0, latency=StoreLatency) ]]
                    v_view.store(out, tile_m, tile_n - 2 * kComponentTiles);
                } else {
                    v_view.store(out, tile_m, tile_n - 2 * kComponentTiles);
                }
            }
        }
    }
}

template <int TM,
          int TN,
          int TK,
          int M,
          int N,
          int K,
          int LoadLatency = 0,
          int BLoadLatency = LoadLatency,
          int StoreLatency = LoadLatency>
__tile_global__ void qkv_bkn_static_tiled_split_contig_direct_store_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_kn,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v) {
    static_assert(N == 3 * kQkvHeads * kQkvDim);
    static_assert((N / 3) % TN == 0);
    static_assert(K % TK == 0);
    constexpr int kComponentTiles = (N / 3) / TN;
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using BTile = ct::tile<__nv_bfloat16, ct::shape<TK, TN>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_kn = ct::assume_aligned(b_kn, 16_ic);
    q = ct::assume_aligned(q, 16_ic);
    k_out = ct::assume_aligned(k_out, 16_ic);
    v = ct::assume_aligned(v, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_kn, ct::shape<K, N>{}},
        ct::shape<TK, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        if constexpr (LoadLatency > 0 || BLoadLatency > 0) {
            ATile a_tile;
            BTile b_tile;
            if constexpr (LoadLatency > 0) {
                [[ cutile::hint(0, latency=LoadLatency) ]]
                a_tile = a_view.load(tile_m, kk);
            } else {
                a_tile = a_view.load(tile_m, kk);
            }
            if constexpr (BLoadLatency > 0) {
                [[ cutile::hint(0, latency=BLoadLatency) ]]
                b_tile = b_view.load(kk, tile_n);
            } else {
                b_tile = b_view.load(kk, tile_n);
            }
            acc = ct::mma(a_tile, b_tile, acc);
        } else {
            acc = ct::mma(a_view.load(tile_m, kk), b_view.load(kk, tile_n), acc);
        }
    }

    I64OutTile local = ct::iota<I64OutTile>();
    auto row = static_cast<long long>(tile_m) * TM + local / TN;
    auto col = local % TN;
    auto out = ct::element_cast<__nv_bfloat16>(acc);
    if (tile_n < kComponentTiles) {
        auto offset = row * (N / 3) + tile_n * TN + col;
        if constexpr (StoreLatency > 0) {
            [[ cutile::hint(0, latency=StoreLatency) ]]
            ct::store(q + offset, out);
        } else {
            ct::store(q + offset, out);
        }
    } else if (tile_n < 2 * kComponentTiles) {
        auto component_tile = tile_n - kComponentTiles;
        auto offset = row * (N / 3) + component_tile * TN + col;
        if constexpr (StoreLatency > 0) {
            [[ cutile::hint(0, latency=StoreLatency) ]]
            ct::store(k_out + offset, out);
        } else {
            ct::store(k_out + offset, out);
        }
    } else {
        auto component_tile = tile_n - 2 * kComponentTiles;
        auto offset = row * (N / 3) + component_tile * TN + col;
        if constexpr (StoreLatency > 0) {
            [[ cutile::hint(0, latency=StoreLatency) ]]
            ct::store(v + offset, out);
        } else {
            ct::store(v + offset, out);
        }
    }
}

template <int TM,
          int TN,
          int TK,
          int M,
          int N,
          int K,
          int Component,
          int LoadLatency = 0,
          int BLoadLatency = LoadLatency,
          int StoreLatency = LoadLatency>
__tile_global__ void qkv_bkn_static_tiled_split_contig_component_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_kn,
    __nv_bfloat16* __restrict__ out) {
    static_assert(N == 3 * kQkvHeads * kQkvDim);
    static_assert((N / 3) % TN == 0);
    static_assert(K % TK == 0);
    static_assert(Component >= 0 && Component < 3);
    constexpr int kComponentN = N / 3;
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using BTile = ct::tile<__nv_bfloat16, ct::shape<TK, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_kn = ct::assume_aligned(b_kn, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_kn + Component * kComponentN, ct::shape<K, kComponentN>{}},
        ct::shape<TK, TN>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out, ct::shape<M, kComponentN>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        if constexpr (LoadLatency > 0 || BLoadLatency > 0) {
            ATile a_tile;
            BTile b_tile;
            if constexpr (LoadLatency > 0) {
                [[ cutile::hint(0, latency=LoadLatency) ]]
                a_tile = a_view.load(tile_m, kk);
            } else {
                a_tile = a_view.load(tile_m, kk);
            }
            if constexpr (BLoadLatency > 0) {
                [[ cutile::hint(0, latency=BLoadLatency) ]]
                b_tile = b_view.load(kk, tile_n);
            } else {
                b_tile = b_view.load(kk, tile_n);
            }
            acc = ct::mma(a_tile, b_tile, acc);
        } else {
            acc = ct::mma(a_view.load(tile_m, kk), b_view.load(kk, tile_n), acc);
        }
    }

    auto out_tile = ct::element_cast<__nv_bfloat16>(acc);
    if constexpr (StoreLatency > 0) {
        [[ cutile::hint(0, latency=StoreLatency) ]]
        out_view.store(out_tile, tile_m, tile_n);
    } else {
        out_view.store(out_tile, tile_m, tile_n);
    }
}

template <int TM,
          int TN,
          int TK,
          int M,
          int N,
          int K,
          int LoadLatency = 0,
          int BLoadLatency = LoadLatency,
          int StoreLatency = LoadLatency>
__tile_global__ void qkv_bkn_static_tiled_split_contig_zcomponent_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_kn,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v) {
    static_assert(N == 3 * kQkvHeads * kQkvDim);
    static_assert((N / 3) % TN == 0);
    static_assert(K % TK == 0);
    constexpr int kComponentN = N / 3;
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using BTile = ct::tile<__nv_bfloat16, ct::shape<TK, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_kn = ct::assume_aligned(b_kn, 16_ic);
    q = ct::assume_aligned(q, 16_ic);
    k_out = ct::assume_aligned(k_out, 16_ic);
    v = ct::assume_aligned(v, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };

    auto [tile_m, tile_n, component_raw] = ct::bid();
    int component = static_cast<int>(component_raw);
    auto b_view = ct::partition_view{
        ct::tensor_span{b_kn + component * kComponentN, ct::shape<K, kComponentN>{}},
        ct::shape<TK, TN>{}
    };

    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        if constexpr (LoadLatency > 0 || BLoadLatency > 0) {
            ATile a_tile;
            BTile b_tile;
            if constexpr (LoadLatency > 0) {
                [[ cutile::hint(0, latency=LoadLatency) ]]
                a_tile = a_view.load(tile_m, kk);
            } else {
                a_tile = a_view.load(tile_m, kk);
            }
            if constexpr (BLoadLatency > 0) {
                [[ cutile::hint(0, latency=BLoadLatency) ]]
                b_tile = b_view.load(kk, tile_n);
            } else {
                b_tile = b_view.load(kk, tile_n);
            }
            acc = ct::mma(a_tile, b_tile, acc);
        } else {
            acc = ct::mma(a_view.load(tile_m, kk), b_view.load(kk, tile_n), acc);
        }
    }

    auto out_tile = ct::element_cast<__nv_bfloat16>(acc);
    if (component == 0) {
        auto q_view = ct::partition_view{
            ct::tensor_span{q, ct::shape<M, kComponentN>{}},
            ct::shape<TM, TN>{}
        };
        if constexpr (StoreLatency > 0) {
            [[ cutile::hint(0, latency=StoreLatency) ]]
            q_view.store(out_tile, tile_m, tile_n);
        } else {
            q_view.store(out_tile, tile_m, tile_n);
        }
    } else if (component == 1) {
        auto k_view = ct::partition_view{
            ct::tensor_span{k_out, ct::shape<M, kComponentN>{}},
            ct::shape<TM, TN>{}
        };
        if constexpr (StoreLatency > 0) {
            [[ cutile::hint(0, latency=StoreLatency) ]]
            k_view.store(out_tile, tile_m, tile_n);
        } else {
            k_view.store(out_tile, tile_m, tile_n);
        }
    } else {
        auto v_view = ct::partition_view{
            ct::tensor_span{v, ct::shape<M, kComponentN>{}},
            ct::shape<TM, TN>{}
        };
        if constexpr (StoreLatency > 0) {
            [[ cutile::hint(0, latency=StoreLatency) ]]
            v_view.store(out_tile, tile_m, tile_n);
        } else {
            v_view.store(out_tile, tile_m, tile_n);
        }
    }
}

template <int TM, int TN, int TK, int M, int N, int K>
__tile_global__ void cutile_mma_gemm_nt_bf16_static_masked_m_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_nt,
    __nv_bfloat16* __restrict__ c) {
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_nt = ct::assume_aligned(b_nt, 16_ic);
    c = ct::assume_aligned(c, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_nt, ct::shape<K, N>{}, ct::layout_left{}},
        ct::shape<TK, TN>{}
    };
    auto c_view = ct::partition_view{
        ct::tensor_span{c, ct::shape<M, N>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    bool full_m_tile = tile_m < M / TM;
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        auto b_tile = b_view.load(kk, tile_n);
        if (full_m_tile) {
            acc = ct::mma(a_view.load(tile_m, kk), b_tile, acc);
        } else {
            acc = ct::mma(a_view.load_masked(tile_m, kk), b_tile, acc);
        }
    }

    if (full_m_tile) {
        c_view.store(ct::element_cast<__nv_bfloat16>(acc), tile_m, tile_n);
    } else {
        c_view.store_masked(ct::element_cast<__nv_bfloat16>(acc), tile_m, tile_n);
    }
}

template <int TM, int TN, int TK, int MPad, int MActual, int N, int K>
__tile_global__ void cutile_mma_gemm_nt_bf16_static_padded_m_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_nt,
    __nv_bfloat16* __restrict__ c) {
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using I64ATile = ct::tile<long long, ct::shape<TM, TK>>;
    using I64CTile = ct::tile<long long, ct::shape<TM, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_nt = ct::assume_aligned(b_nt, 16_ic);
    c = ct::assume_aligned(c, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<MPad, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_nt, ct::shape<K, N>{}, ct::layout_left{}},
        ct::shape<TK, TN>{}
    };
    auto c_view = ct::partition_view{
        ct::tensor_span{c, ct::shape<MPad, N>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    bool full_m_tile = tile_m < MActual / TM;
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        auto b_tile = b_view.load(kk, tile_n);
        if (full_m_tile) {
            acc = ct::mma(a_view.load(tile_m, kk), b_tile, acc);
        } else {
            I64ATile local = ct::iota<I64ATile>();
            auto rows = static_cast<long long>(tile_m) * TM + local / TK;
            auto k_cols = static_cast<long long>(kk) * TK + local % TK;
            auto valid = rows < MActual;
            auto a_tile = ct::load_masked(a + rows * K + k_cols, valid);
            acc = ct::mma(a_tile, b_tile, acc);
        }
    }

    auto out = ct::element_cast<__nv_bfloat16>(acc);
    if (full_m_tile) {
        c_view.store(out, tile_m, tile_n);
    } else {
        I64CTile local = ct::iota<I64CTile>();
        auto rows = static_cast<long long>(tile_m) * TM + local / TN;
        auto cols = static_cast<long long>(tile_n) * TN + local % TN;
        auto valid = rows < MActual;
        ct::store_masked(c + rows * N + cols, out, valid);
    }
}

template <int TM, int TN, int TK, int M, int N, int K>
__tile_global__ void cutile_mma_gemm_nt_bf16_static_masked_mn_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_nt,
    __nv_bfloat16* __restrict__ c) {
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_nt = ct::assume_aligned(b_nt, 16_ic);
    c = ct::assume_aligned(c, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_nt, ct::shape<K, N>{}, ct::layout_left{}},
        ct::shape<TK, TN>{}
    };
    auto c_view = ct::partition_view{
        ct::tensor_span{c, ct::shape<M, N>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    bool full_m_tile = tile_m < M / TM;
    bool full_n_tile = false;
    if constexpr (N >= TN) {
        full_n_tile = tile_n < N / TN;
    }
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        auto a_tile = full_m_tile ? a_view.load(tile_m, kk)
                                  : a_view.load_masked(tile_m, kk);
        auto b_tile = full_n_tile ? b_view.load(kk, tile_n)
                                  : b_view.load_masked(kk, tile_n);
        acc = ct::mma(a_tile, b_tile, acc);
    }

    auto out = ct::element_cast<__nv_bfloat16>(acc);
    if (full_m_tile && full_n_tile) {
        c_view.store(out, tile_m, tile_n);
    } else {
        c_view.store_masked(out, tile_m, tile_n);
    }
}

void init_bf16(__nv_bfloat16* ptr, size_t elems) {
    fill_bf16_kernel<<<ceildiv(static_cast<int>(elems), kInitTile), 1>>>(
        ptr, static_cast<long long>(elems));
    CUDA_CHECK(cudaGetLastError());
}

template <int TM, int TN, int TK, bool Masked, bool UseMatmul>
void launch_cutile(const Shape& shape,
                   const __nv_bfloat16* d_a,
                   const __nv_bfloat16* d_b,
                   __nv_bfloat16* d_c,
                   int m_run) {
    dim3 grid(ceildiv(m_run, TM), ceildiv(shape.n, TN), 1);
    cutile_mma_gemm_nt_bf16_kernel<TM, TN, TK, Masked, UseMatmul><<<grid, 1>>>(
        d_a, d_b, d_c,
        static_cast<std::size_t>(m_run),
        static_cast<std::size_t>(shape.n),
        static_cast<std::size_t>(shape.k));
}

template <int TM, int TN, int TK, int M, int N, int K, bool UseMatmul = false>
void launch_static_cutile(const __nv_bfloat16* d_a,
                          const __nv_bfloat16* d_b,
                          __nv_bfloat16* d_c) {
    dim3 grid(M / TM, N / TN, 1);
    cutile_mma_gemm_nt_bf16_static_kernel<TM, TN, TK, M, N, K, UseMatmul><<<grid, 1>>>(
        d_a, d_b, d_c);
}

template <int M, int N, int K>
void launch_static_qkv_split_n128_cutile(const __nv_bfloat16* d_a,
                                         const __nv_bfloat16* d_b,
                                         __nv_bfloat16* d_c) {
    dim3 grid(M / 32, N / 256, 1);
    qkv_split_n128_static_kernel<M, N, K><<<grid, 1>>>(d_a, d_b, d_c);
}

template <int M, int N, int K>
void launch_static_qkv_manual_cutile(const __nv_bfloat16* d_a,
                                     const __nv_bfloat16* d_b,
                                     __nv_bfloat16* d_c) {
    dim3 grid(M / 32, N / 256, 1);
    qkv_manual_static_kernel<M, N, K><<<grid, 1>>>(d_a, d_b, d_c);
}

template <int M, int N, int K>
void launch_static_qkv_bkn_cutile(const __nv_bfloat16* d_a,
                                  const __nv_bfloat16* d_b,
                                  __nv_bfloat16* d_c) {
    dim3 grid(M / 32, N / 256, 1);
    qkv_bkn_static_kernel<M, N, K><<<grid, 1>>>(d_a, d_b, d_c);
}

template <int M, int N, int K>
void launch_static_qkv_bkn_unroll8_cutile(const __nv_bfloat16* d_a,
                                          const __nv_bfloat16* d_b,
                                          __nv_bfloat16* d_c) {
    dim3 grid(M / 32, N / 256, 1);
    qkv_bkn_static_unroll8_kernel<M, N, K><<<grid, 1>>>(d_a, d_b, d_c);
}

template <int M, int N, int K>
void launch_static_qkv_bkn_pairk_cutile(const __nv_bfloat16* d_a,
                                        const __nv_bfloat16* d_b,
                                        __nv_bfloat16* d_c) {
    dim3 grid(M / 32, N / 256, 1);
    qkv_bkn_static_pairk_kernel<M, N, K><<<grid, 1>>>(d_a, d_b, d_c);
}

template <int TM, int TN, int TK, int M, int N, int K>
void launch_static_qkv_bkn_tiled_cutile(const __nv_bfloat16* d_a,
                                        const __nv_bfloat16* d_b,
                                        __nv_bfloat16* d_c) {
    dim3 grid(M / TM, N / TN, 1);
    qkv_bkn_static_tiled_kernel<TM, TN, TK, M, N, K><<<grid, 1>>>(d_a, d_b, d_c);
}

template <int TM, int TN, int TK, int M, int N, int K>
void launch_static_qkv_bkn_tiled_pairk_cutile(const __nv_bfloat16* d_a,
                                              const __nv_bfloat16* d_b,
                                              __nv_bfloat16* d_c) {
    dim3 grid(M / TM, N / TN, 1);
    qkv_bkn_static_tiled_pairk_kernel<TM, TN, TK, M, N, K><<<grid, 1>>>(
        d_a, d_b, d_c);
}

template <int TM, int TN, int TK, int M, int N, int K>
void launch_static_qkv_bkn_tiled_loadtmp_cutile(const __nv_bfloat16* d_a,
                                                const __nv_bfloat16* d_b,
                                                __nv_bfloat16* d_c) {
    dim3 grid(M / TM, N / TN, 1);
    qkv_bkn_static_tiled_loadtmp_kernel<TM, TN, TK, M, N, K><<<grid, 1>>>(
        d_a, d_b, d_c);
}

template <int TM, int TN, int TK, int M, int N, int K>
void launch_static_qkv_bkn_tiled_matmul_cutile(const __nv_bfloat16* d_a,
                                               const __nv_bfloat16* d_b,
                                               __nv_bfloat16* d_c) {
    dim3 grid(M / TM, N / TN, 1);
    qkv_bkn_static_tiled_matmul_kernel<TM, TN, TK, M, N, K><<<grid, 1>>>(
        d_a, d_b, d_c);
}

template <int TM, int TN, int TK, int M, int N, int KFull, int KRun>
void launch_static_qkv_bkn_tiled_kprefix_cutile(const __nv_bfloat16* d_a,
                                                const __nv_bfloat16* d_b,
                                                __nv_bfloat16* d_c) {
    dim3 grid(M / TM, N / TN, 1);
    qkv_bkn_static_tiled_kprefix_kernel<TM, TN, TK, M, N, KFull, KRun><<<grid, 1>>>(
        d_a, d_b, d_c);
}

template <int TM, int TN, int TK, int M, int N, int K, int Occupancy>
void launch_static_qkv_bkn_tiled_occ_cutile(const __nv_bfloat16* d_a,
                                            const __nv_bfloat16* d_b,
                                            __nv_bfloat16* d_c) {
    dim3 grid(M / TM, N / TN, 1);
    qkv_bkn_static_tiled_occ_kernel<TM, TN, TK, M, N, K, Occupancy><<<grid, 1>>>(
        d_a, d_b, d_c);
}

template <int TM,
          int TN,
          int TK,
          int M,
          int N,
          int K,
          int LoadLatency,
          int BLoadLatency = LoadLatency,
          int StoreLatency = LoadLatency>
void launch_static_qkv_bkn_tiled_latency_cutile(const __nv_bfloat16* d_a,
                                                 const __nv_bfloat16* d_b,
                                                 __nv_bfloat16* d_c) {
    dim3 grid(M / TM, N / TN, 1);
    qkv_bkn_static_tiled_latency_kernel<TM,
                                        TN,
                                        TK,
                                        M,
                                        N,
                                        K,
                                        LoadLatency,
                                        BLoadLatency,
                                        StoreLatency><<<grid, 1>>>(
        d_a, d_b, d_c);
}

template <int TM, int TK, int M, int N, int K>
void launch_static_qkv_bkn_compound_256p128_cutile(const __nv_bfloat16* d_a,
                                                   const __nv_bfloat16* d_b,
                                                   __nv_bfloat16* d_c) {
    dim3 grid(M / TM, N / 384, 1);
    qkv_bkn_static_compound_256p128_kernel<TM, TK, M, N, K><<<grid, 1>>>(
        d_a, d_b, d_c);
}

template <int TM, int TN, int TK, int M, int N, int K>
void launch_static_qkv_bkn_tiled_scatter_time_cutile(const __nv_bfloat16* d_a,
                                                     const __nv_bfloat16* d_b,
                                                     __nv_bfloat16* d_q,
                                                     __nv_bfloat16* d_k,
                                                     __nv_bfloat16* d_v) {
    dim3 grid(M / TM, N / TN, 1);
    qkv_bkn_static_tiled_scatter_time_kernel<TM, TN, TK, M, N, K><<<grid, 1>>>(
        d_a, d_b, d_q, d_k, d_v);
}

template <int TM,
          int TN,
          int TK,
          int M,
          int N,
          int K,
          int LoadLatency = 0,
          int BLoadLatency = LoadLatency,
          int StoreLatency = LoadLatency,
          bool BranchLocalCast = false,
          bool BranchLocalView = false>
void launch_static_qkv_bkn_tiled_split_contig_cutile(const __nv_bfloat16* d_a,
                                                     const __nv_bfloat16* d_b,
                                                     __nv_bfloat16* d_q,
                                                     __nv_bfloat16* d_k,
                                                     __nv_bfloat16* d_v) {
    dim3 grid(M / TM, N / TN, 1);
    qkv_bkn_static_tiled_split_contig_kernel<TM,
                                             TN,
                                             TK,
                                             M,
                                             N,
                                             K,
                                             LoadLatency,
                                             BLoadLatency,
                                             StoreLatency,
                                             BranchLocalCast,
                                             BranchLocalView>
        <<<grid, 1>>>(d_a, d_b, d_q, d_k, d_v);
}

template <int TM,
          int TN,
          int TK,
          int M,
          int N,
          int K,
          int LoadLatency = 0,
          int BLoadLatency = LoadLatency,
          int StoreLatency = LoadLatency>
void launch_static_qkv_bkn_tiled_split_contig_direct_store_cutile(
    const __nv_bfloat16* d_a,
    const __nv_bfloat16* d_b,
    __nv_bfloat16* d_q,
    __nv_bfloat16* d_k,
    __nv_bfloat16* d_v) {
    dim3 grid(M / TM, N / TN, 1);
    qkv_bkn_static_tiled_split_contig_direct_store_kernel<TM,
                                                          TN,
                                                          TK,
                                                          M,
                                                          N,
                                                          K,
                                                          LoadLatency,
                                                          BLoadLatency,
                                                          StoreLatency>
        <<<grid, 1>>>(d_a, d_b, d_q, d_k, d_v);
}

template <int TM,
          int TN,
          int TK,
          int M,
          int N,
          int K,
          int LoadLatency = 0,
          int BLoadLatency = LoadLatency,
          int StoreLatency = LoadLatency>
void launch_static_qkv_bkn_tiled_split_contig_component_cutile(
    const __nv_bfloat16* d_a,
    const __nv_bfloat16* d_b,
    __nv_bfloat16* d_q,
    __nv_bfloat16* d_k,
    __nv_bfloat16* d_v) {
    dim3 grid(M / TM, (N / 3) / TN, 1);
    qkv_bkn_static_tiled_split_contig_component_kernel<TM,
                                                       TN,
                                                       TK,
                                                       M,
                                                       N,
                                                       K,
                                                       0,
                                                       LoadLatency,
                                                       BLoadLatency,
                                                       StoreLatency>
        <<<grid, 1>>>(d_a, d_b, d_q);
    qkv_bkn_static_tiled_split_contig_component_kernel<TM,
                                                       TN,
                                                       TK,
                                                       M,
                                                       N,
                                                       K,
                                                       1,
                                                       LoadLatency,
                                                       BLoadLatency,
                                                       StoreLatency>
        <<<grid, 1>>>(d_a, d_b, d_k);
    qkv_bkn_static_tiled_split_contig_component_kernel<TM,
                                                       TN,
                                                       TK,
                                                       M,
                                                       N,
                                                       K,
                                                       2,
                                                       LoadLatency,
                                                       BLoadLatency,
                                                       StoreLatency>
        <<<grid, 1>>>(d_a, d_b, d_v);
}

template <int TM,
          int TN,
          int TK,
          int M,
          int N,
          int K,
          int LoadLatency = 0,
          int BLoadLatency = LoadLatency,
          int StoreLatency = LoadLatency>
void launch_static_qkv_bkn_tiled_split_contig_zcomponent_cutile(
    const __nv_bfloat16* d_a,
    const __nv_bfloat16* d_b,
    __nv_bfloat16* d_q,
    __nv_bfloat16* d_k,
    __nv_bfloat16* d_v) {
    dim3 grid(M / TM, (N / 3) / TN, 3);
    qkv_bkn_static_tiled_split_contig_zcomponent_kernel<TM,
                                                        TN,
                                                        TK,
                                                        M,
                                                        N,
                                                        K,
                                                        LoadLatency,
                                                        BLoadLatency,
                                                        StoreLatency>
        <<<grid, 1>>>(d_a, d_b, d_q, d_k, d_v);
}

template <int TM, int TN, int TK, int M, int N, int K>
void launch_static_masked_m_cutile(const __nv_bfloat16* d_a,
                                   const __nv_bfloat16* d_b,
                                   __nv_bfloat16* d_c) {
    dim3 grid(ceildiv(M, TM), N / TN, 1);
    cutile_mma_gemm_nt_bf16_static_masked_m_kernel<TM, TN, TK, M, N, K><<<grid, 1>>>(
        d_a, d_b, d_c);
}

template <int TM, int TN, int TK, int MPad, int MActual, int N, int K>
void launch_static_padded_m_cutile(const __nv_bfloat16* d_a,
                                   const __nv_bfloat16* d_b,
                                   __nv_bfloat16* d_c) {
    dim3 grid(MPad / TM, N / TN, 1);
    cutile_mma_gemm_nt_bf16_static_padded_m_kernel<TM,
                                                   TN,
                                                   TK,
                                                   MPad,
                                                   MActual,
                                                   N,
                                                   K><<<grid, 1>>>(d_a, d_b, d_c);
}

template <int TM, int TN, int TK, int M, int N, int K>
void launch_static_masked_mn_cutile(const __nv_bfloat16* d_a,
                                    const __nv_bfloat16* d_b,
                                    __nv_bfloat16* d_c) {
    dim3 grid(ceildiv(M, TM), ceildiv(N, TN), 1);
    cutile_mma_gemm_nt_bf16_static_masked_mn_kernel<TM, TN, TK, M, N, K>
        <<<grid, 1>>>(d_a, d_b, d_c);
}

template <int TM, int TN, int TK, int M, int N, int K, bool UseMatmul = false>
void run_static_variant_impl(const Shape& shape, const Options& opts, const char* variant_name) {
    int iters = opts.iters_override > 0 ? opts.iters_override : shape.iters;
    size_t a_elems = static_cast<size_t>(shape.m) * shape.k;
    size_t b_elems = static_cast<size_t>(shape.n) * shape.k;
    size_t c_elems = static_cast<size_t>(shape.m) * shape.n;
    double gib = (static_cast<double>(a_elems + b_elems + c_elems) *
                  sizeof(__nv_bfloat16)) /
                 (1024.0 * 1024.0 * 1024.0);
    dim3 grid(M / TM, N / TN, 1);

    __nv_bfloat16* d_a = nullptr;
    __nv_bfloat16* d_b = nullptr;
    __nv_bfloat16* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_b, b_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_c, c_elems * sizeof(__nv_bfloat16)));
    init_bf16(d_a, a_elems);
    init_bf16(d_b, b_elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < opts.warmup; ++i) {
        launch_static_cutile<TM, TN, TK, M, N, K, UseMatmul>(d_a, d_b, d_c);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch_static_cutile<TM, TN, TK, M, N, K, UseMatmul>(d_a, d_b, d_c);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_c, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    double flops = 2.0 * M * N * K;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "  %-10s tile=%dx%dx%d grid=(%u,%u) fullM=%d mem=%.2f GiB best=%.3f ms median=%.3f ms %.2f TF/s roof=%.1f%% checksum=%.4f\n",
        variant_name, TM, TN, TK, grid.x, grid.y, M, gib, best_ms, median_ms,
        tflops, tflops * 100.0 / kA10gDenseBf16Tflops, __bfloat162float(checksum_bf16));
}

template <int TM, int TN, int TK>
void run_static_variant(const Shape& shape, const Options& opts, const char* variant_name) {
    if (std::strcmp(shape.name, "infer_qkv") == 0) {
        run_static_variant_impl<TM, TN, TK, 78048, 1536, 256>(shape, opts, variant_name);
    } else if (std::strcmp(shape.name, "infer_ffn1") == 0) {
        run_static_variant_impl<TM, TN, TK, 78048, 1024, 256>(shape, opts, variant_name);
    } else if (std::strcmp(shape.name, "infer_ffn2") == 0) {
        run_static_variant_impl<TM, TN, TK, 78048, 256, 1024>(shape, opts, variant_name);
    } else if (std::strcmp(shape.name, "infer_attn_out") == 0) {
        run_static_variant_impl<TM, TN, TK, 78048, 256, 512>(shape, opts, variant_name);
    }
}

template <int TM, int TN, int TK, int M, int N, int K>
void launch_attn_out_residual_static_cutile(const __nv_bfloat16* d_a,
                                            const __nv_bfloat16* d_b,
                                            const __nv_bfloat16* d_residual,
                                            __nv_bfloat16* d_c) {
    dim3 grid(M / TM, N / TN, 1);
    attn_out_residual_static_kernel<TM, TN, TK, M, N, K>
        <<<grid, 1>>>(d_a, d_b, d_residual, d_c);
}

template <int TM, int TN, int TK>
void run_attn_out_residual_variant(const Shape& shape,
                                   const Options& opts,
                                   const char* variant_name) {
    if (std::strcmp(shape.name, "infer_attn_out") != 0) {
        if (opts.variant == variant_name) {
            std::printf("  %-10s skipped: attention-output residual variant\n",
                        variant_name);
        }
        return;
    }
    constexpr int M = (78060 / TM) * TM;
    constexpr int N = 256;
    constexpr int K = 512;
    int iters = opts.iters_override > 0 ? opts.iters_override : shape.iters;
    size_t a_elems = static_cast<size_t>(shape.m) * shape.k;
    size_t b_elems = static_cast<size_t>(shape.n) * shape.k;
    size_t c_elems = static_cast<size_t>(shape.m) * shape.n;
    double gib = (static_cast<double>(a_elems + b_elems + 2 * c_elems) *
                  sizeof(__nv_bfloat16)) /
                 (1024.0 * 1024.0 * 1024.0);
    dim3 grid(M / TM, N / TN, 1);

    __nv_bfloat16* d_a = nullptr;
    __nv_bfloat16* d_b = nullptr;
    __nv_bfloat16* d_residual = nullptr;
    __nv_bfloat16* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_b, b_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_residual, c_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_c, c_elems * sizeof(__nv_bfloat16)));
    init_bf16(d_a, a_elems);
    init_bf16(d_b, b_elems);
    init_bf16(d_residual, c_elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < opts.warmup; ++i) {
        launch_attn_out_residual_static_cutile<TM, TN, TK, M, N, K>(
            d_a, d_b, d_residual, d_c);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch_attn_out_residual_static_cutile<TM, TN, TK, M, N, K>(
            d_a, d_b, d_residual, d_c);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_c, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_residual));
    CUDA_CHECK(cudaFree(d_c));

    double flops = 2.0 * M * N * K;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "  %-10s tile=%dx%dx%d grid=(%u,%u) fullM=%d mem=%.2f GiB best=%.3f ms median=%.3f ms %.2f TF/s roof=%.1f%% checksum=%.4f\n",
        variant_name, TM, TN, TK, grid.x, grid.y, M, gib, best_ms, median_ms,
        tflops, tflops * 100.0 / kA10gDenseBf16Tflops,
        __bfloat162float(checksum_bf16));
}

template <int TM, int TN, int TK, bool UseMatmul = false>
void run_static_qkv_variant(const Shape& shape, const Options& opts, const char* variant_name) {
    if (std::strcmp(shape.name, "infer_qkv") != 0) {
        if (opts.variant == variant_name) {
            std::printf("  %-10s skipped: QKV-only variant\n", variant_name);
        }
        return;
    }
    constexpr int kQkvFullM = (78060 / TM) * TM;
    run_static_variant_impl<TM, TN, TK, kQkvFullM, 1536, 256, UseMatmul>(
        shape, opts, variant_name);
}

template <typename LaunchFn>
void run_qkv_custom_variant(const Shape& shape,
                            const Options& opts,
                            const char* variant_name,
                            int full_m,
                            int tile_m,
                            int tile_n,
                            int tile_k,
                            dim3 grid,
                            LaunchFn launch,
                            int flops_k = 256) {
    if (std::strcmp(shape.name, "infer_qkv") != 0) {
        if (opts.variant == variant_name) {
            std::printf("  %-10s skipped: QKV-only variant\n", variant_name);
        }
        return;
    }

    int iters = opts.iters_override > 0 ? opts.iters_override : shape.iters;
    size_t a_elems = static_cast<size_t>(shape.m) * shape.k;
    size_t b_elems = static_cast<size_t>(shape.n) * shape.k;
    size_t c_elems = static_cast<size_t>(shape.m) * shape.n;
    double gib = (static_cast<double>(a_elems + b_elems + c_elems) *
                  sizeof(__nv_bfloat16)) /
                 (1024.0 * 1024.0 * 1024.0);

    __nv_bfloat16* d_a = nullptr;
    __nv_bfloat16* d_b = nullptr;
    __nv_bfloat16* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_b, b_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_c, c_elems * sizeof(__nv_bfloat16)));
    init_bf16(d_a, a_elems);
    init_bf16(d_b, b_elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < opts.warmup; ++i) {
        launch(d_a, d_b, d_c);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch(d_a, d_b, d_c);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_c, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    double flops = 2.0 * full_m * 1536 * flops_k;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "  %-10s tile=%dx%dx%d grid=(%u,%u) fullM=%d mem=%.2f GiB best=%.3f ms median=%.3f ms %.2f TF/s roof=%.1f%% checksum=%.4f\n",
        variant_name, tile_m, tile_n, tile_k, grid.x, grid.y, full_m, gib,
        best_ms, median_ms, tflops, tflops * 100.0 / kA10gDenseBf16Tflops,
        __bfloat162float(checksum_bf16));
}

void run_static_qkv_split_n128_variant(const Shape& shape,
                                       const Options& opts,
                                       const char* variant_name) {
    constexpr int kFullM = (78060 / 32) * 32;
    dim3 grid(kFullM / 32, 1536 / 256, 1);
    run_qkv_custom_variant(
        shape, opts, variant_name, kFullM, 32, 256, 32, grid,
        [](__nv_bfloat16* d_a, __nv_bfloat16* d_b, __nv_bfloat16* d_c) {
            launch_static_qkv_split_n128_cutile<kFullM, 1536, 256>(d_a, d_b, d_c);
        });
}

void run_static_qkv_manual_variant(const Shape& shape,
                                   const Options& opts,
                                   const char* variant_name) {
    constexpr int kFullM = (78060 / 32) * 32;
    dim3 grid(kFullM / 32, 1536 / 256, 1);
    run_qkv_custom_variant(
        shape, opts, variant_name, kFullM, 32, 256, 32, grid,
        [](__nv_bfloat16* d_a, __nv_bfloat16* d_b, __nv_bfloat16* d_c) {
            launch_static_qkv_manual_cutile<kFullM, 1536, 256>(d_a, d_b, d_c);
        });
}

void run_static_qkv_bkn_variant(const Shape& shape,
                                const Options& opts,
                                const char* variant_name) {
    constexpr int kFullM = (78060 / 32) * 32;
    dim3 grid(kFullM / 32, 1536 / 256, 1);
    run_qkv_custom_variant(
        shape, opts, variant_name, kFullM, 32, 256, 32, grid,
        [](__nv_bfloat16* d_a, __nv_bfloat16* d_b, __nv_bfloat16* d_c) {
            launch_static_qkv_bkn_cutile<kFullM, 1536, 256>(d_a, d_b, d_c);
        });
}

void run_static_qkv_bkn_unroll8_variant(const Shape& shape,
                                        const Options& opts,
                                        const char* variant_name) {
    constexpr int kFullM = (78060 / 32) * 32;
    dim3 grid(kFullM / 32, 1536 / 256, 1);
    run_qkv_custom_variant(
        shape, opts, variant_name, kFullM, 32, 256, 32, grid,
        [](__nv_bfloat16* d_a, __nv_bfloat16* d_b, __nv_bfloat16* d_c) {
            launch_static_qkv_bkn_unroll8_cutile<kFullM, 1536, 256>(
                d_a, d_b, d_c);
        });
}

void run_static_qkv_bkn_pairk_variant(const Shape& shape,
                                      const Options& opts,
                                      const char* variant_name) {
    constexpr int kFullM = (78060 / 32) * 32;
    dim3 grid(kFullM / 32, 1536 / 256, 1);
    run_qkv_custom_variant(
        shape, opts, variant_name, kFullM, 32, 256, 32, grid,
        [](__nv_bfloat16* d_a, __nv_bfloat16* d_b, __nv_bfloat16* d_c) {
            launch_static_qkv_bkn_pairk_cutile<kFullM, 1536, 256>(
                d_a, d_b, d_c);
        });
}

template <int TM, int TN, int TK>
void run_static_qkv_bkn_tiled_variant(const Shape& shape,
                                      const Options& opts,
                                      const char* variant_name) {
    if (std::strcmp(shape.name, "infer_qkv") != 0) {
        if (opts.variant == variant_name) {
            std::printf("  %-10s skipped: QKV-only BKN variant\n", variant_name);
        }
        return;
    }
    constexpr int kFullM = (78060 / TM) * TM;
    dim3 grid(kFullM / TM, 1536 / TN, 1);
    run_qkv_custom_variant(
        shape, opts, variant_name, kFullM, TM, TN, TK, grid,
        [](__nv_bfloat16* d_a, __nv_bfloat16* d_b, __nv_bfloat16* d_c) {
            launch_static_qkv_bkn_tiled_cutile<TM, TN, TK, kFullM, 1536, 256>(
                d_a, d_b, d_c);
        });
}

template <int TM, int TN, int TK>
void run_static_qkv_bkn_tiled_pairk_variant(const Shape& shape,
                                            const Options& opts,
                                            const char* variant_name) {
    if (std::strcmp(shape.name, "infer_qkv") != 0) {
        if (opts.variant == variant_name) {
            std::printf("  %-10s skipped: QKV-only BKN variant\n", variant_name);
        }
        return;
    }
    constexpr int kFullM = (78060 / TM) * TM;
    dim3 grid(kFullM / TM, 1536 / TN, 1);
    run_qkv_custom_variant(
        shape, opts, variant_name, kFullM, TM, TN, TK, grid,
        [](__nv_bfloat16* d_a, __nv_bfloat16* d_b, __nv_bfloat16* d_c) {
            launch_static_qkv_bkn_tiled_pairk_cutile<TM, TN, TK, kFullM, 1536, 256>(
                d_a, d_b, d_c);
        });
}

void run_static_qkv_bkn_tiled_loadtmp_variant(const Shape& shape,
                                              const Options& opts,
                                              const char* variant_name) {
    if (std::strcmp(shape.name, "infer_qkv") != 0) {
        if (opts.variant == variant_name) {
            std::printf("  %-10s skipped: QKV-only BKN loadtmp variant\n",
                        variant_name);
        }
        return;
    }
    constexpr int kFullM = (78060 / 32) * 32;
    dim3 grid(kFullM / 32, 1536 / 256, 1);
    run_qkv_custom_variant(
        shape, opts, variant_name, kFullM, 32, 256, 16, grid,
        [](__nv_bfloat16* d_a, __nv_bfloat16* d_b, __nv_bfloat16* d_c) {
            launch_static_qkv_bkn_tiled_loadtmp_cutile<32,
                                                       256,
                                                       16,
                                                       kFullM,
                                                       1536,
                                                       256>(d_a, d_b, d_c);
        });
}

void run_static_qkv_bkn_tiled_matmul_variant(const Shape& shape,
                                             const Options& opts,
                                             const char* variant_name) {
    if (std::strcmp(shape.name, "infer_qkv") != 0) {
        if (opts.variant == variant_name) {
            std::printf("  %-10s skipped: QKV-only BKN matmul variant\n",
                        variant_name);
        }
        return;
    }
    constexpr int kFullM = (78060 / 32) * 32;
    dim3 grid(kFullM / 32, 1536 / 256, 1);
    run_qkv_custom_variant(
        shape, opts, variant_name, kFullM, 32, 256, 16, grid,
        [](__nv_bfloat16* d_a, __nv_bfloat16* d_b, __nv_bfloat16* d_c) {
            launch_static_qkv_bkn_tiled_matmul_cutile<32,
                                                      256,
                                                      16,
                                                      kFullM,
                                                      1536,
                                                      256>(d_a, d_b, d_c);
        });
}

void run_static_qkv_bkn_tiled_k128_variant(const Shape& shape,
                                           const Options& opts,
                                           const char* variant_name) {
    if (std::strcmp(shape.name, "infer_qkv") != 0) {
        if (opts.variant == variant_name) {
            std::printf("  %-10s skipped: QKV-only BKN K-prefix variant\n",
                        variant_name);
        }
        return;
    }
    constexpr int kFullM = (78060 / 32) * 32;
    dim3 grid(kFullM / 32, 1536 / 256, 1);
    run_qkv_custom_variant(
        shape, opts, variant_name, kFullM, 32, 256, 16, grid,
        [](__nv_bfloat16* d_a, __nv_bfloat16* d_b, __nv_bfloat16* d_c) {
            launch_static_qkv_bkn_tiled_kprefix_cutile<32,
                                                       256,
                                                       16,
                                                       kFullM,
                                                       1536,
                                                       256,
                                                       128>(d_a, d_b, d_c);
        },
        128);
}

template <int Occupancy>
void run_static_qkv_bkn_tiled_occ_variant(const Shape& shape,
                                          const Options& opts,
                                          const char* variant_name) {
    if (std::strcmp(shape.name, "infer_qkv") != 0) {
        if (opts.variant == variant_name) {
            std::printf("  %-10s skipped: QKV-only BKN occupancy-hint variant\n",
                        variant_name);
        }
        return;
    }
    constexpr int kFullM = (78060 / 32) * 32;
    dim3 grid(kFullM / 32, 1536 / 256, 1);
    run_qkv_custom_variant(
        shape, opts, variant_name, kFullM, 32, 256, 16, grid,
        [](__nv_bfloat16* d_a, __nv_bfloat16* d_b, __nv_bfloat16* d_c) {
            launch_static_qkv_bkn_tiled_occ_cutile<32,
                                                   256,
                                                   16,
                                                   kFullM,
                                                   1536,
                                                   256,
                                                   Occupancy>(d_a, d_b, d_c);
        });
}

template <int LoadLatency,
          int BLoadLatency = LoadLatency,
          int StoreLatency = LoadLatency,
          int TM = 32,
          int TN = 256,
          int TK = 16>
void run_static_qkv_bkn_tiled_latency_variant(const Shape& shape,
                                              const Options& opts,
                                              const char* variant_name) {
    if (std::strcmp(shape.name, "infer_qkv") != 0) {
        if (opts.variant == variant_name) {
            std::printf("  %-10s skipped: QKV-only BKN latency-hint variant\n",
                        variant_name);
        }
        return;
    }
    constexpr int kFullM = (78060 / TM) * TM;
    dim3 grid(kFullM / TM, 1536 / TN, 1);
    run_qkv_custom_variant(
        shape, opts, variant_name, kFullM, TM, TN, TK, grid,
        [](__nv_bfloat16* d_a, __nv_bfloat16* d_b, __nv_bfloat16* d_c) {
            launch_static_qkv_bkn_tiled_latency_cutile<TM,
                                                       TN,
                                                       TK,
                                                       kFullM,
                                                       1536,
                                                       256,
                                                       LoadLatency,
                                                       BLoadLatency,
                                                       StoreLatency>(d_a, d_b, d_c);
        });
}

void run_static_qkv_bkn_compound_256p128_variant(const Shape& shape,
                                                 const Options& opts,
                                                 const char* variant_name) {
    if (std::strcmp(shape.name, "infer_qkv") != 0) {
        if (opts.variant == variant_name) {
            std::printf("  %-10s skipped: QKV-only BKN compound variant\n",
                        variant_name);
        }
        return;
    }
    constexpr int kFullM = (78060 / 32) * 32;
    dim3 grid(kFullM / 32, 1536 / 384, 1);
    run_qkv_custom_variant(
        shape, opts, variant_name, kFullM, 32, 384, 16, grid,
        [](__nv_bfloat16* d_a, __nv_bfloat16* d_b, __nv_bfloat16* d_c) {
            launch_static_qkv_bkn_compound_256p128_cutile<32,
                                                          16,
                                                          kFullM,
                                                          1536,
                                                          256>(d_a, d_b, d_c);
        });
}

template <int TM, int TN, int TK>
void run_static_qkv_bkn_tiled_scatter_time_variant(const Shape& shape,
                                                   const Options& opts,
                                                   const char* variant_name) {
    if (std::strcmp(shape.name, "infer_qkv") != 0) {
        if (opts.variant == variant_name) {
            std::printf("  %-10s skipped: QKV-only BKN scatter variant\n", variant_name);
        }
        return;
    }

    constexpr int kFullM = (78060 / TM) * TM;
    constexpr int kOutElems = kTimeBatches * kQkvHeads * kTimeSeq * kQkvDim;
    int iters = opts.iters_override > 0 ? opts.iters_override : shape.iters;
    size_t a_elems = static_cast<size_t>(shape.m) * shape.k;
    size_t b_elems = static_cast<size_t>(shape.n) * shape.k;
    size_t out_elems = static_cast<size_t>(kOutElems) * 3;
    double gib = (static_cast<double>(a_elems + b_elems + out_elems) *
                  sizeof(__nv_bfloat16)) /
                 (1024.0 * 1024.0 * 1024.0);
    dim3 grid(kFullM / TM, 1536 / TN, 1);

    __nv_bfloat16* d_a = nullptr;
    __nv_bfloat16* d_b = nullptr;
    __nv_bfloat16* d_q = nullptr;
    __nv_bfloat16* d_k = nullptr;
    __nv_bfloat16* d_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_b, b_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_q, kOutElems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k, kOutElems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v, kOutElems * sizeof(__nv_bfloat16)));
    init_bf16(d_a, a_elems);
    init_bf16(d_b, b_elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto launch = [&]() {
        launch_static_qkv_bkn_tiled_scatter_time_cutile<TM,
                                                        TN,
                                                        TK,
                                                        kFullM,
                                                        1536,
                                                        256>(
            d_a, d_b, d_q, d_k, d_v);
    };

    for (int i = 0; i < opts.warmup; ++i) {
        launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_q, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));

    double flops = 2.0 * kFullM * 1536 * 256;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "  %-10s tile=%dx%dx%d grid=(%u,%u) fullM=%d mem=%.2f GiB best=%.3f ms median=%.3f ms %.2f TF/s roof=%.1f%% checksum=%.4f\n",
        variant_name, TM, TN, TK, grid.x, grid.y, kFullM, gib,
        best_ms, median_ms, tflops, tflops * 100.0 / kA10gDenseBf16Tflops,
        __bfloat162float(checksum_bf16));
}

template <int TM,
          int TN,
          int TK,
          int LoadLatency = 0,
          int BLoadLatency = LoadLatency,
          int StoreLatency = LoadLatency,
          bool BranchLocalCast = false,
          bool BranchLocalView = false>
void run_static_qkv_bkn_tiled_split_contig_variant(const Shape& shape,
                                                   const Options& opts,
                                                   const char* variant_name) {
    if (std::strcmp(shape.name, "infer_qkv") != 0) {
        if (opts.variant == variant_name) {
            std::printf("  %-10s skipped: QKV-only BKN split variant\n", variant_name);
        }
        return;
    }

    constexpr int kFullM = (78060 / TM) * TM;
    int iters = opts.iters_override > 0 ? opts.iters_override : shape.iters;
    size_t a_elems = static_cast<size_t>(shape.m) * shape.k;
    size_t b_elems = static_cast<size_t>(shape.n) * shape.k;
    size_t out_elems = static_cast<size_t>(shape.m) * (shape.n / 3) * 3;
    size_t split_elems = static_cast<size_t>(shape.m) * (shape.n / 3);
    double gib = (static_cast<double>(a_elems + b_elems + out_elems) *
                  sizeof(__nv_bfloat16)) /
                 (1024.0 * 1024.0 * 1024.0);
    dim3 grid(kFullM / TM, 1536 / TN, 1);

    __nv_bfloat16* d_a = nullptr;
    __nv_bfloat16* d_b = nullptr;
    __nv_bfloat16* d_q = nullptr;
    __nv_bfloat16* d_k = nullptr;
    __nv_bfloat16* d_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_b, b_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_q, split_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k, split_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v, split_elems * sizeof(__nv_bfloat16)));
    init_bf16(d_a, a_elems);
    init_bf16(d_b, b_elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto launch = [&]() {
        launch_static_qkv_bkn_tiled_split_contig_cutile<TM,
                                                        TN,
                                                        TK,
                                                        kFullM,
                                                        1536,
                                                        256,
                                                        LoadLatency,
                                                        BLoadLatency,
                                                        StoreLatency,
                                                        BranchLocalCast,
                                                        BranchLocalView>(
            d_a, d_b, d_q, d_k, d_v);
    };

    for (int i = 0; i < opts.warmup; ++i) {
        launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_q, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));

    double flops = 2.0 * kFullM * 1536 * 256;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "  %-10s tile=%dx%dx%d grid=(%u,%u) fullM=%d mem=%.2f GiB best=%.3f ms median=%.3f ms %.2f TF/s roof=%.1f%% checksum=%.4f\n",
        variant_name, TM, TN, TK, grid.x, grid.y, kFullM, gib,
        best_ms, median_ms, tflops, tflops * 100.0 / kA10gDenseBf16Tflops,
        __bfloat162float(checksum_bf16));
}

template <int TM,
          int TN,
          int TK,
          int LoadLatency = 0,
          int BLoadLatency = LoadLatency,
          int StoreLatency = LoadLatency>
void run_static_qkv_bkn_tiled_split_contig_direct_store_variant(
    const Shape& shape,
    const Options& opts,
    const char* variant_name) {
    if (std::strcmp(shape.name, "infer_qkv") != 0) {
        if (opts.variant == variant_name) {
            std::printf("  %-10s skipped: QKV-only BKN direct-store split variant\n",
                        variant_name);
        }
        return;
    }

    constexpr int kFullM = (78060 / TM) * TM;
    int iters = opts.iters_override > 0 ? opts.iters_override : shape.iters;
    size_t a_elems = static_cast<size_t>(shape.m) * shape.k;
    size_t b_elems = static_cast<size_t>(shape.n) * shape.k;
    size_t out_elems = static_cast<size_t>(shape.m) * (shape.n / 3) * 3;
    size_t split_elems = static_cast<size_t>(shape.m) * (shape.n / 3);
    double gib = (static_cast<double>(a_elems + b_elems + out_elems) *
                  sizeof(__nv_bfloat16)) /
                 (1024.0 * 1024.0 * 1024.0);
    dim3 grid(kFullM / TM, 1536 / TN, 1);

    __nv_bfloat16* d_a = nullptr;
    __nv_bfloat16* d_b = nullptr;
    __nv_bfloat16* d_q = nullptr;
    __nv_bfloat16* d_k = nullptr;
    __nv_bfloat16* d_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_b, b_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_q, split_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k, split_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v, split_elems * sizeof(__nv_bfloat16)));
    init_bf16(d_a, a_elems);
    init_bf16(d_b, b_elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto launch = [&]() {
        launch_static_qkv_bkn_tiled_split_contig_direct_store_cutile<TM,
                                                                     TN,
                                                                     TK,
                                                                     kFullM,
                                                                     1536,
                                                                     256,
                                                                     LoadLatency,
                                                                     BLoadLatency,
                                                                     StoreLatency>(
            d_a, d_b, d_q, d_k, d_v);
    };

    for (int i = 0; i < opts.warmup; ++i) {
        launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_q, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));

    double flops = 2.0 * kFullM * 1536 * 256;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "  %-10s tile=%dx%dx%d direct_grid=(%u,%u) fullM=%d mem=%.2f GiB best=%.3f ms median=%.3f ms %.2f TF/s roof=%.1f%% checksum=%.4f\n",
        variant_name, TM, TN, TK, grid.x, grid.y, kFullM, gib,
        best_ms, median_ms, tflops, tflops * 100.0 / kA10gDenseBf16Tflops,
        __bfloat162float(checksum_bf16));
}

template <int TM,
          int TN,
          int TK,
          int LoadLatency = 0,
          int BLoadLatency = LoadLatency,
          int StoreLatency = LoadLatency>
void run_static_qkv_bkn_tiled_split_contig_component_variant(
    const Shape& shape,
    const Options& opts,
    const char* variant_name) {
    if (std::strcmp(shape.name, "infer_qkv") != 0) {
        if (opts.variant == variant_name) {
            std::printf("  %-10s skipped: QKV-only BKN component split variant\n",
                        variant_name);
        }
        return;
    }

    constexpr int kFullM = (78060 / TM) * TM;
    int iters = opts.iters_override > 0 ? opts.iters_override : shape.iters;
    size_t a_elems = static_cast<size_t>(shape.m) * shape.k;
    size_t b_elems = static_cast<size_t>(shape.n) * shape.k;
    size_t out_elems = static_cast<size_t>(shape.m) * (shape.n / 3) * 3;
    size_t split_elems = static_cast<size_t>(shape.m) * (shape.n / 3);
    double gib = (static_cast<double>(a_elems + b_elems + out_elems) *
                  sizeof(__nv_bfloat16)) /
                 (1024.0 * 1024.0 * 1024.0);
    dim3 grid(kFullM / TM, (1536 / 3) / TN, 1);

    __nv_bfloat16* d_a = nullptr;
    __nv_bfloat16* d_b = nullptr;
    __nv_bfloat16* d_q = nullptr;
    __nv_bfloat16* d_k = nullptr;
    __nv_bfloat16* d_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_b, b_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_q, split_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k, split_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v, split_elems * sizeof(__nv_bfloat16)));
    init_bf16(d_a, a_elems);
    init_bf16(d_b, b_elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto launch = [&]() {
        launch_static_qkv_bkn_tiled_split_contig_component_cutile<TM,
                                                                  TN,
                                                                  TK,
                                                                  kFullM,
                                                                  1536,
                                                                  256,
                                                                  LoadLatency,
                                                                  BLoadLatency,
                                                                  StoreLatency>(
            d_a, d_b, d_q, d_k, d_v);
    };

    for (int i = 0; i < opts.warmup; ++i) {
        launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_q, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));

    double flops = 2.0 * kFullM * 1536 * 256;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "  %-10s tile=%dx%dx%d component_grid=(%u,%u) fullM=%d launches=3 mem=%.2f GiB best=%.3f ms median=%.3f ms %.2f TF/s roof=%.1f%% checksum=%.4f\n",
        variant_name, TM, TN, TK, grid.x, grid.y, kFullM, gib,
        best_ms, median_ms, tflops, tflops * 100.0 / kA10gDenseBf16Tflops,
        __bfloat162float(checksum_bf16));
}

template <int TM,
          int TN,
          int TK,
          int LoadLatency = 0,
          int BLoadLatency = LoadLatency,
          int StoreLatency = LoadLatency>
void run_static_qkv_bkn_tiled_split_contig_zcomponent_variant(
    const Shape& shape,
    const Options& opts,
    const char* variant_name) {
    if (std::strcmp(shape.name, "infer_qkv") != 0) {
        if (opts.variant == variant_name) {
            std::printf("  %-10s skipped: QKV-only BKN z-component split variant\n",
                        variant_name);
        }
        return;
    }

    constexpr int kFullM = (78060 / TM) * TM;
    int iters = opts.iters_override > 0 ? opts.iters_override : shape.iters;
    size_t a_elems = static_cast<size_t>(shape.m) * shape.k;
    size_t b_elems = static_cast<size_t>(shape.n) * shape.k;
    size_t out_elems = static_cast<size_t>(shape.m) * (shape.n / 3) * 3;
    size_t split_elems = static_cast<size_t>(shape.m) * (shape.n / 3);
    double gib = (static_cast<double>(a_elems + b_elems + out_elems) *
                  sizeof(__nv_bfloat16)) /
                 (1024.0 * 1024.0 * 1024.0);
    dim3 grid(kFullM / TM, (1536 / 3) / TN, 3);

    __nv_bfloat16* d_a = nullptr;
    __nv_bfloat16* d_b = nullptr;
    __nv_bfloat16* d_q = nullptr;
    __nv_bfloat16* d_k = nullptr;
    __nv_bfloat16* d_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_b, b_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_q, split_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k, split_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v, split_elems * sizeof(__nv_bfloat16)));
    init_bf16(d_a, a_elems);
    init_bf16(d_b, b_elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto launch = [&]() {
        launch_static_qkv_bkn_tiled_split_contig_zcomponent_cutile<TM,
                                                                   TN,
                                                                   TK,
                                                                   kFullM,
                                                                   1536,
                                                                   256,
                                                                   LoadLatency,
                                                                   BLoadLatency,
                                                                   StoreLatency>(
            d_a, d_b, d_q, d_k, d_v);
    };

    for (int i = 0; i < opts.warmup; ++i) {
        launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_q, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));

    double flops = 2.0 * kFullM * 1536 * 256;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "  %-10s tile=%dx%dx%d grid=(%u,%u,%u) fullM=%d mem=%.2f GiB best=%.3f ms median=%.3f ms %.2f TF/s roof=%.1f%% checksum=%.4f\n",
        variant_name, TM, TN, TK, grid.x, grid.y, grid.z, kFullM, gib,
        best_ms, median_ms, tflops, tflops * 100.0 / kA10gDenseBf16Tflops,
        __bfloat162float(checksum_bf16));
}

template <int M, int N, int K>
void run_static_bkn_variant_impl(const Shape& shape, const Options& opts, const char* variant_name) {
    int iters = opts.iters_override > 0 ? opts.iters_override : shape.iters;
    size_t a_elems = static_cast<size_t>(shape.m) * shape.k;
    size_t b_elems = static_cast<size_t>(shape.n) * shape.k;
    size_t c_elems = static_cast<size_t>(shape.m) * shape.n;
    double gib = (static_cast<double>(a_elems + b_elems + c_elems) *
                  sizeof(__nv_bfloat16)) /
                 (1024.0 * 1024.0 * 1024.0);
    dim3 grid(M / 32, N / 256, 1);

    __nv_bfloat16* d_a = nullptr;
    __nv_bfloat16* d_b = nullptr;
    __nv_bfloat16* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_b, b_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_c, c_elems * sizeof(__nv_bfloat16)));
    init_bf16(d_a, a_elems);
    init_bf16(d_b, b_elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < opts.warmup; ++i) {
        launch_static_qkv_bkn_cutile<M, N, K>(d_a, d_b, d_c);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch_static_qkv_bkn_cutile<M, N, K>(d_a, d_b, d_c);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_c, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    double flops = 2.0 * M * N * K;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "  %-10s tile=32x256x32 grid=(%u,%u) fullM=%d mem=%.2f GiB best=%.3f ms median=%.3f ms %.2f TF/s roof=%.1f%% checksum=%.4f\n",
        variant_name, grid.x, grid.y, M, gib, best_ms, median_ms, tflops,
        tflops * 100.0 / kA10gDenseBf16Tflops, __bfloat162float(checksum_bf16));
}

void run_static_bkn_variant(const Shape& shape, const Options& opts, const char* variant_name) {
    if (std::strcmp(shape.name, "infer_qkv") == 0) {
        run_static_bkn_variant_impl<78048, 1536, 256>(shape, opts, variant_name);
    } else if (std::strcmp(shape.name, "infer_ffn1") == 0) {
        run_static_bkn_variant_impl<78048, 1024, 256>(shape, opts, variant_name);
    } else if (std::strcmp(shape.name, "infer_ffn2") == 0) {
        run_static_bkn_variant_impl<78048, 256, 1024>(shape, opts, variant_name);
    } else if (std::strcmp(shape.name, "infer_attn_out") == 0) {
        run_static_bkn_variant_impl<78048, 256, 512>(shape, opts, variant_name);
    } else if (opts.variant == variant_name) {
        std::printf("  %-10s skipped: B(K,N) long-linear variant not supported\n",
                    variant_name);
    }
}

template <int TM, int TN, int TK, bool Masked = true, bool UseMatmul = false>
void run_variant(const Shape& shape, const Options& opts, const char* variant_name) {
    int m_run = Masked ? shape.m : (shape.m / TM) * TM;
    if (m_run <= 0) {
        std::printf("  %-10s skipped: no full M tile\n", variant_name);
        return;
    }
    int iters = opts.iters_override > 0 ? opts.iters_override : shape.iters;
    size_t a_elems = static_cast<size_t>(shape.m) * shape.k;
    size_t b_elems = static_cast<size_t>(shape.n) * shape.k;
    size_t c_elems = static_cast<size_t>(shape.m) * shape.n;
    double gib = (static_cast<double>(a_elems + b_elems + c_elems) *
                 sizeof(__nv_bfloat16)) /
                 (1024.0 * 1024.0 * 1024.0);
    dim3 grid(ceildiv(m_run, TM), ceildiv(shape.n, TN), 1);

    __nv_bfloat16* d_a = nullptr;
    __nv_bfloat16* d_b = nullptr;
    __nv_bfloat16* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_b, b_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_c, c_elems * sizeof(__nv_bfloat16)));
    init_bf16(d_a, a_elems);
    init_bf16(d_b, b_elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < opts.warmup; ++i) {
        launch_cutile<TM, TN, TK, Masked, UseMatmul>(shape, d_a, d_b, d_c, m_run);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch_cutile<TM, TN, TK, Masked, UseMatmul>(shape, d_a, d_b, d_c, m_run);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_c, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    double flops = 2.0 * m_run * shape.n * shape.k;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "  %-10s tile=%dx%dx%d grid=(%u,%u) mem=%.2f GiB best=%.3f ms median=%.3f ms %.2f TF/s roof=%.1f%% checksum=%.4f\n",
        variant_name, TM, TN, TK, grid.x, grid.y, gib, best_ms, median_ms, tflops,
        tflops * 100.0 / kA10gDenseBf16Tflops, __bfloat162float(checksum_bf16));
}

template <int TM, int TN, int TK, int M, int N, int K>
void run_static_masked_m_variant_impl(const Shape& shape,
                                      const Options& opts,
                                      const char* variant_name) {
    int iters = opts.iters_override > 0 ? opts.iters_override : shape.iters;
    size_t a_elems = static_cast<size_t>(shape.m) * shape.k;
    size_t b_elems = static_cast<size_t>(shape.n) * shape.k;
    size_t c_elems = static_cast<size_t>(shape.m) * shape.n;
    double gib = (static_cast<double>(a_elems + b_elems + c_elems) *
                  sizeof(__nv_bfloat16)) /
                 (1024.0 * 1024.0 * 1024.0);
    dim3 grid(ceildiv(M, TM), N / TN, 1);

    __nv_bfloat16* d_a = nullptr;
    __nv_bfloat16* d_b = nullptr;
    __nv_bfloat16* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_b, b_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_c, c_elems * sizeof(__nv_bfloat16)));
    init_bf16(d_a, a_elems);
    init_bf16(d_b, b_elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < opts.warmup; ++i) {
        launch_static_masked_m_cutile<TM, TN, TK, M, N, K>(d_a, d_b, d_c);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch_static_masked_m_cutile<TM, TN, TK, M, N, K>(d_a, d_b, d_c);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_c, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    double useful_flops = 2.0 * shape.m * shape.n * shape.k;
    double issued_flops = 2.0 * grid.x * TM * shape.n * shape.k;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double useful_tflops = useful_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double issued_tflops = issued_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "  %-10s tile=%dx%dx%d grid=(%u,%u) mem=%.2f GiB best=%.3f ms median=%.3f ms useful=%.2f TF/s issued=%.2f TF/s roof=%.1f%% checksum=%.4f\n",
        variant_name, TM, TN, TK, grid.x, grid.y, gib, best_ms, median_ms,
        useful_tflops, issued_tflops, useful_tflops * 100.0 / kA10gDenseBf16Tflops,
        __bfloat162float(checksum_bf16));
}

template <int TM, int TN, int TK, int MPad, int MActual, int N, int K>
void run_static_padded_m_variant_impl(const Shape& shape,
                                      const Options& opts,
                                      const char* variant_name) {
    int iters = opts.iters_override > 0 ? opts.iters_override : shape.iters;
    size_t a_elems = static_cast<size_t>(shape.m) * shape.k;
    size_t b_elems = static_cast<size_t>(shape.n) * shape.k;
    size_t c_elems = static_cast<size_t>(shape.m) * shape.n;
    double gib = (static_cast<double>(a_elems + b_elems + c_elems) *
                  sizeof(__nv_bfloat16)) /
                 (1024.0 * 1024.0 * 1024.0);
    dim3 grid(MPad / TM, N / TN, 1);

    __nv_bfloat16* d_a = nullptr;
    __nv_bfloat16* d_b = nullptr;
    __nv_bfloat16* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_b, b_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_c, c_elems * sizeof(__nv_bfloat16)));
    init_bf16(d_a, a_elems);
    init_bf16(d_b, b_elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < opts.warmup; ++i) {
        launch_static_padded_m_cutile<TM, TN, TK, MPad, MActual, N, K>(d_a, d_b, d_c);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch_static_padded_m_cutile<TM, TN, TK, MPad, MActual, N, K>(d_a, d_b, d_c);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_c, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    double useful_flops = 2.0 * shape.m * shape.n * shape.k;
    double issued_flops = 2.0 * MPad * shape.n * shape.k;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double useful_tflops = useful_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double issued_tflops = issued_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "  %-10s tile=%dx%dx%d grid=(%u,%u) mem=%.2f GiB best=%.3f ms median=%.3f ms useful=%.2f TF/s issued=%.2f TF/s roof=%.1f%% checksum=%.4f\n",
        variant_name, TM, TN, TK, grid.x, grid.y, gib, best_ms, median_ms,
        useful_tflops, issued_tflops, useful_tflops * 100.0 / kA10gDenseBf16Tflops,
        __bfloat162float(checksum_bf16));
}

template <int TM, int TN, int TK, int M, int N, int K>
void run_static_masked_mn_variant_impl(const Shape& shape,
                                       const Options& opts,
                                       const char* variant_name) {
    int iters = opts.iters_override > 0 ? opts.iters_override : shape.iters;
    size_t a_elems = static_cast<size_t>(shape.m) * shape.k;
    size_t b_elems = static_cast<size_t>(shape.n) * shape.k;
    size_t c_elems = static_cast<size_t>(shape.m) * shape.n;
    double gib = (static_cast<double>(a_elems + b_elems + c_elems) *
                  sizeof(__nv_bfloat16)) /
                 (1024.0 * 1024.0 * 1024.0);
    dim3 grid(ceildiv(M, TM), ceildiv(N, TN), 1);

    __nv_bfloat16* d_a = nullptr;
    __nv_bfloat16* d_b = nullptr;
    __nv_bfloat16* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_b, b_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_c, c_elems * sizeof(__nv_bfloat16)));
    init_bf16(d_a, a_elems);
    init_bf16(d_b, b_elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < opts.warmup; ++i) {
        launch_static_masked_mn_cutile<TM, TN, TK, M, N, K>(d_a, d_b, d_c);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch_static_masked_mn_cutile<TM, TN, TK, M, N, K>(d_a, d_b, d_c);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_c, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    double useful_flops = 2.0 * shape.m * shape.n * shape.k;
    double issued_flops = 2.0 * grid.x * TM * grid.y * TN * shape.k;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double useful_tflops = useful_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double issued_tflops = issued_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "  %-10s tile=%dx%dx%d grid=(%u,%u) mem=%.2f GiB best=%.3f ms median=%.3f ms useful=%.2f TF/s issued=%.2f TF/s roof=%.1f%% checksum=%.4f\n",
        variant_name, TM, TN, TK, grid.x, grid.y, gib, best_ms, median_ms,
        useful_tflops, issued_tflops, useful_tflops * 100.0 / kA10gDenseBf16Tflops,
        __bfloat162float(checksum_bf16));
}

void run_shape(const Shape& shape, const Options& opts) {
    std::printf("%-15s M=%d N=%d K=%d\n", shape.name, shape.m, shape.n, shape.k);
    auto run_static_masked_mn = [&](auto tm, auto tn, auto tk, const char* name) {
        constexpr int TM = decltype(tm)::value;
        constexpr int TN = decltype(tn)::value;
        constexpr int TK = decltype(tk)::value;
        if (opts.variant != "all" && opts.variant != name) return;
        if (shape.k % TK != 0) {
            std::printf("  %-10s skipped: K not divisible by tile\n", name);
            return;
        }
        if (std::strcmp(shape.name, "gate_n8_k256") == 0) {
            run_static_masked_mn_variant_impl<TM, TN, TK, 78060, 8, 256>(
                shape, opts, name);
        } else {
            std::printf("  %-10s skipped: static masked-MN long shape not supported\n", name);
        }
    };
    if (opts.variant == "all" || opts.variant == "t32x32x16") {
        run_variant<32, 32, 16>(shape, opts, "t32x32x16");
    }
    if (opts.variant == "all" || opts.variant == "t32x64x16") {
        run_variant<32, 64, 16>(shape, opts, "t32x64x16");
    }
    if (opts.variant == "all" || opts.variant == "t32x64x16u") {
        run_variant<32, 64, 16, false>(shape, opts, "t32x64x16u");
    }
    if (opts.variant == "all" || opts.variant == "t32x64x16m") {
        run_variant<32, 64, 16, true, true>(shape, opts, "t32x64x16m");
    }
    if (opts.variant == "all" || opts.variant == "t32x64x16s") {
        run_static_variant<32, 64, 16>(shape, opts, "t32x64x16s");
    }
    if (opts.variant == "all" || opts.variant == "t32x128x16s") {
        run_static_variant<32, 128, 16>(shape, opts, "t32x128x16s");
    }
    if (opts.variant == "all" || opts.variant == "t32x256x16s") {
        run_static_variant<32, 256, 16>(shape, opts, "t32x256x16s");
    }
    if (opts.variant == "all" || opts.variant == "t64x64x16s") {
        run_static_variant<64, 64, 16>(shape, opts, "t64x64x16s");
    }
    if (opts.variant == "all" || opts.variant == "t32x64x32s") {
        run_static_variant<32, 64, 32>(shape, opts, "t32x64x32s");
    }
    if (opts.variant == "all" || opts.variant == "t32x128x32s") {
        run_static_variant<32, 128, 32>(shape, opts, "t32x128x32s");
    }
    if (opts.variant == "all" || opts.variant == "t32x256x32s") {
        run_static_variant<32, 256, 32>(shape, opts, "t32x256x32s");
    }
    if (opts.variant == "all" || opts.variant == "t32x512x32s") {
        run_static_variant<32, 512, 32>(shape, opts, "t32x512x32s");
    }
    if (opts.variant == "all" || opts.variant == "attnres_t16x128x32") {
        run_attn_out_residual_variant<16, 128, 32>(
            shape, opts, "attnres_t16x128x32");
    }
    if (opts.variant == "all" || opts.variant == "attnres_t32x64x32") {
        run_attn_out_residual_variant<32, 64, 32>(
            shape, opts, "attnres_t32x64x32");
    }
    if (opts.variant == "all" || opts.variant == "attnres_t32x128x32") {
        run_attn_out_residual_variant<32, 128, 32>(
            shape, opts, "attnres_t32x128x32");
    }
    if (opts.variant == "all" || opts.variant == "attnres_t32x256x32") {
        run_attn_out_residual_variant<32, 256, 32>(
            shape, opts, "attnres_t32x256x32");
    }
    if (opts.variant == "all" || opts.variant == "attnres_t64x128x32") {
        run_attn_out_residual_variant<64, 128, 32>(
            shape, opts, "attnres_t64x128x32");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t16x256x32s") {
        run_static_qkv_variant<16, 256, 32>(shape, opts, "qkv_t16x256x32s");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t64x256x32s") {
        run_static_qkv_variant<64, 256, 32>(shape, opts, "qkv_t64x256x32s");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x64s") {
        run_static_qkv_variant<32, 256, 64>(shape, opts, "qkv_t32x256x64s");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x32mat") {
        run_static_qkv_variant<32, 256, 32, true>(shape, opts, "qkv_t32x256x32mat");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x32splitn128") {
        run_static_qkv_split_n128_variant(shape, opts, "qkv_t32x256x32splitn128");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x32manual") {
        run_static_qkv_manual_variant(shape, opts, "qkv_t32x256x32manual");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x32bkn") {
        run_static_qkv_bkn_variant(shape, opts, "qkv_t32x256x32bkn");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x32x32bkn") {
        run_static_qkv_bkn_tiled_variant<32, 32, 32>(
            shape, opts, "qkv_t32x32x32bkn");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x64x32bkn") {
        run_static_qkv_bkn_tiled_variant<32, 64, 32>(
            shape, opts, "qkv_t32x64x32bkn");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x128x32bkn") {
        run_static_qkv_bkn_tiled_variant<32, 128, 32>(
            shape, opts, "qkv_t32x128x32bkn");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x32bkn_tiled") {
        run_static_qkv_bkn_tiled_variant<32, 256, 32>(
            shape, opts, "qkv_t32x256x32bkn_tiled");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x32bkn_pairk") {
        run_static_qkv_bkn_pairk_variant(shape, opts, "qkv_t32x256x32bkn_pairk");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x32bkn_unroll8") {
        run_static_qkv_bkn_unroll8_variant(shape, opts, "qkv_t32x256x32bkn_unroll8");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t16x256x32bkn") {
        run_static_qkv_bkn_tiled_variant<16, 256, 32>(
            shape, opts, "qkv_t16x256x32bkn");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t64x256x32bkn") {
        run_static_qkv_bkn_tiled_variant<64, 256, 32>(
            shape, opts, "qkv_t64x256x32bkn");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x512x32bkn") {
        run_static_qkv_bkn_tiled_variant<32, 512, 32>(
            shape, opts, "qkv_t32x512x32bkn");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t16x256x16bkn") {
        run_static_qkv_bkn_tiled_variant<16, 256, 16>(
            shape, opts, "qkv_t16x256x16bkn");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x128x16bkn") {
        run_static_qkv_bkn_tiled_variant<32, 128, 16>(
            shape, opts, "qkv_t32x128x16bkn");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t64x128x16bkn") {
        run_static_qkv_bkn_tiled_variant<64, 128, 16>(
            shape, opts, "qkv_t64x128x16bkn");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x8bkn") {
        run_static_qkv_bkn_tiled_variant<32, 256, 8>(
            shape, opts, "qkv_t32x256x8bkn");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x16bkn") {
        run_static_qkv_bkn_tiled_variant<32, 256, 16>(
            shape, opts, "qkv_t32x256x16bkn");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x16bkn_pairk") {
        run_static_qkv_bkn_tiled_pairk_variant<32, 256, 16>(
            shape, opts, "qkv_t32x256x16bkn_pairk");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x16bkn_loadtmp") {
        run_static_qkv_bkn_tiled_loadtmp_variant(
            shape, opts, "qkv_t32x256x16bkn_loadtmp");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x16bkn_matmul") {
        run_static_qkv_bkn_tiled_matmul_variant(
            shape, opts, "qkv_t32x256x16bkn_matmul");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x16bkn_k128") {
        run_static_qkv_bkn_tiled_k128_variant(
            shape, opts, "qkv_t32x256x16bkn_k128");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x16bkn_occ8") {
        run_static_qkv_bkn_tiled_occ_variant<8>(shape, opts, "qkv_t32x256x16bkn_occ8");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x16bkn_occ12") {
        run_static_qkv_bkn_tiled_occ_variant<12>(shape, opts, "qkv_t32x256x16bkn_occ12");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x16bkn_occ16") {
        run_static_qkv_bkn_tiled_occ_variant<16>(shape, opts, "qkv_t32x256x16bkn_occ16");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x16bkn_occ24") {
        run_static_qkv_bkn_tiled_occ_variant<24>(shape, opts, "qkv_t32x256x16bkn_occ24");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x16bkn_lat1") {
        run_static_qkv_bkn_tiled_latency_variant<1>(shape, opts, "qkv_t32x256x16bkn_lat1");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x16bkn_lat2") {
        run_static_qkv_bkn_tiled_latency_variant<2>(shape, opts, "qkv_t32x256x16bkn_lat2");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t64x128x16bkn_lat2") {
        run_static_qkv_bkn_tiled_latency_variant<2, 2, 2, 64, 128, 16>(
            shape, opts, "qkv_t64x128x16bkn_lat2");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x16bkn_lat4") {
        run_static_qkv_bkn_tiled_latency_variant<4>(shape, opts, "qkv_t32x256x16bkn_lat4");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x16bkn_lat7") {
        run_static_qkv_bkn_tiled_latency_variant<7>(shape, opts, "qkv_t32x256x16bkn_lat7");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x16bkn_lat10") {
        run_static_qkv_bkn_tiled_latency_variant<10>(shape, opts, "qkv_t32x256x16bkn_lat10");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x16bkn_a2_b2_s0") {
        run_static_qkv_bkn_tiled_latency_variant<2, 2, 0>(
            shape, opts, "qkv_t32x256x16bkn_a2_b2_s0");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x16bkn_a2_b1_s0") {
        run_static_qkv_bkn_tiled_latency_variant<2, 1, 0>(
            shape, opts, "qkv_t32x256x16bkn_a2_b1_s0");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256p128x16bkn") {
        run_static_qkv_bkn_compound_256p128_variant(
            shape, opts, "qkv_t32x256p128x16bkn");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t64x256x16bkn") {
        run_static_qkv_bkn_tiled_variant<64, 256, 16>(
            shape, opts, "qkv_t64x256x16bkn");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x512x16bkn") {
        run_static_qkv_bkn_tiled_variant<32, 512, 16>(
            shape, opts, "qkv_t32x512x16bkn");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x16bkn_scatter_time") {
        run_static_qkv_bkn_tiled_scatter_time_variant<32, 256, 16>(
            shape, opts, "qkv_t32x256x16bkn_scatter_time");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x16bkn_split_contig") {
        run_static_qkv_bkn_tiled_split_contig_variant<32, 256, 16>(
            shape, opts, "qkv_t32x256x16bkn_split_contig");
    }
    if (opts.variant == "all" ||
        opts.variant == "qkv_t32x256x16bkn_split_contig_direct_store") {
        run_static_qkv_bkn_tiled_split_contig_direct_store_variant<32, 256, 16>(
            shape, opts, "qkv_t32x256x16bkn_split_contig_direct_store");
    }
    if (opts.variant == "all" ||
        opts.variant == "qkv_t32x256x16bkn_split_contig_latecast") {
        run_static_qkv_bkn_tiled_split_contig_variant<32, 256, 16, 0, 0, 0, true>(
            shape, opts, "qkv_t32x256x16bkn_split_contig_latecast");
    }
    if (opts.variant == "all" ||
        opts.variant == "qkv_t32x256x16bkn_split_contig_lateview") {
        run_static_qkv_bkn_tiled_split_contig_variant<32,
                                                      256,
                                                      16,
                                                      0,
                                                      0,
                                                      0,
                                                      false,
                                                      true>(
            shape, opts, "qkv_t32x256x16bkn_split_contig_lateview");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t16x256x16bkn_split_contig_lat2") {
        run_static_qkv_bkn_tiled_split_contig_variant<16, 256, 16, 2>(
            shape, opts, "qkv_t16x256x16bkn_split_contig_lat2");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x512x16bkn_split_contig") {
        run_static_qkv_bkn_tiled_split_contig_variant<32, 512, 16>(
            shape, opts, "qkv_t32x512x16bkn_split_contig");
    }
    if (opts.variant == "all" ||
        opts.variant == "qkv_t32x256x16bkn_split_contig_components") {
        run_static_qkv_bkn_tiled_split_contig_component_variant<32, 256, 16>(
            shape, opts, "qkv_t32x256x16bkn_split_contig_components");
    }
    if (opts.variant == "all" ||
        opts.variant == "qkv_t32x256x16bkn_split_contig_zcomponents") {
        run_static_qkv_bkn_tiled_split_contig_zcomponent_variant<32, 256, 16>(
            shape, opts, "qkv_t32x256x16bkn_split_contig_zcomponents");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x32x16bkn_split_contig_lat2") {
        run_static_qkv_bkn_tiled_split_contig_variant<32, 32, 16, 2>(
            shape, opts, "qkv_t32x32x16bkn_split_contig_lat2");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x64x16bkn_split_contig_lat2") {
        run_static_qkv_bkn_tiled_split_contig_variant<32, 64, 16, 2>(
            shape, opts, "qkv_t32x64x16bkn_split_contig_lat2");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x128x16bkn_split_contig_lat2") {
        run_static_qkv_bkn_tiled_split_contig_variant<32, 128, 16, 2>(
            shape, opts, "qkv_t32x128x16bkn_split_contig_lat2");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x16bkn_split_contig_lat2") {
        run_static_qkv_bkn_tiled_split_contig_variant<32, 256, 16, 2>(
            shape, opts, "qkv_t32x256x16bkn_split_contig_lat2");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t64x256x16bkn_split_contig_lat2") {
        run_static_qkv_bkn_tiled_split_contig_variant<64, 256, 16, 2>(
            shape, opts, "qkv_t64x256x16bkn_split_contig_lat2");
    }
    if (opts.variant == "all" ||
        opts.variant == "qkv_t32x256x16bkn_split_contig_direct_store_lat2") {
        run_static_qkv_bkn_tiled_split_contig_direct_store_variant<32, 256, 16, 2>(
            shape, opts, "qkv_t32x256x16bkn_split_contig_direct_store_lat2");
    }
    if (opts.variant == "all" ||
        opts.variant == "qkv_t32x256x16bkn_split_contig_latecast_lat2") {
        run_static_qkv_bkn_tiled_split_contig_variant<32, 256, 16, 2, 2, 2, true>(
            shape, opts, "qkv_t32x256x16bkn_split_contig_latecast_lat2");
    }
    if (opts.variant == "all" ||
        opts.variant == "qkv_t32x256x16bkn_split_contig_lateview_lat2") {
        run_static_qkv_bkn_tiled_split_contig_variant<32,
                                                      256,
                                                      16,
                                                      2,
                                                      2,
                                                      2,
                                                      false,
                                                      true>(
            shape, opts, "qkv_t32x256x16bkn_split_contig_lateview_lat2");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x512x16bkn_split_contig_lat2") {
        run_static_qkv_bkn_tiled_split_contig_variant<32, 512, 16, 2>(
            shape, opts, "qkv_t32x512x16bkn_split_contig_lat2");
    }
    if (opts.variant == "all" ||
        opts.variant == "qkv_t32x256x16bkn_split_contig_components_lat2") {
        run_static_qkv_bkn_tiled_split_contig_component_variant<32, 256, 16, 2>(
            shape, opts, "qkv_t32x256x16bkn_split_contig_components_lat2");
    }
    if (opts.variant == "all" ||
        opts.variant == "qkv_t32x256x16bkn_split_contig_zcomponents_lat2") {
        run_static_qkv_bkn_tiled_split_contig_zcomponent_variant<32, 256, 16, 2>(
            shape, opts, "qkv_t32x256x16bkn_split_contig_zcomponents_lat2");
    }
    if (opts.variant == "all" ||
        opts.variant == "qkv_t32x256x16bkn_split_contig_a2_b0_s0") {
        run_static_qkv_bkn_tiled_split_contig_variant<32, 256, 16, 2, 0, 0>(
            shape, opts, "qkv_t32x256x16bkn_split_contig_a2_b0_s0");
    }
    if (opts.variant == "all" ||
        opts.variant == "qkv_t32x256x16bkn_split_contig_a0_b2_s0") {
        run_static_qkv_bkn_tiled_split_contig_variant<32, 256, 16, 0, 2, 0>(
            shape, opts, "qkv_t32x256x16bkn_split_contig_a0_b2_s0");
    }
    if (opts.variant == "all" ||
        opts.variant == "qkv_t32x256x16bkn_split_contig_a0_b0_s2") {
        run_static_qkv_bkn_tiled_split_contig_variant<32, 256, 16, 0, 0, 2>(
            shape, opts, "qkv_t32x256x16bkn_split_contig_a0_b0_s2");
    }
    if (opts.variant == "all" ||
        opts.variant == "qkv_t32x256x16bkn_split_contig_a2_b1_s2") {
        run_static_qkv_bkn_tiled_split_contig_variant<32, 256, 16, 2, 1, 2>(
            shape, opts, "qkv_t32x256x16bkn_split_contig_a2_b1_s2");
    }
    if (opts.variant == "all" ||
        opts.variant == "qkv_t32x256x16bkn_split_contig_a2_b1_s0") {
        run_static_qkv_bkn_tiled_split_contig_variant<32, 256, 16, 2, 1, 0>(
            shape, opts, "qkv_t32x256x16bkn_split_contig_a2_b1_s0");
    }
    if (opts.variant == "all" ||
        opts.variant == "qkv_t32x256x16bkn_split_contig_a1_b2_s2") {
        run_static_qkv_bkn_tiled_split_contig_variant<32, 256, 16, 1, 2, 2>(
            shape, opts, "qkv_t32x256x16bkn_split_contig_a1_b2_s2");
    }
    if (opts.variant == "all" ||
        opts.variant == "qkv_t32x256x16bkn_split_contig_a2_b2_s0") {
        run_static_qkv_bkn_tiled_split_contig_variant<32, 256, 16, 2, 2, 0>(
            shape, opts, "qkv_t32x256x16bkn_split_contig_a2_b2_s0");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x64bkn") {
        run_static_qkv_bkn_tiled_variant<32, 256, 64>(
            shape, opts, "qkv_t32x256x64bkn");
    }
    if (opts.variant == "all" || opts.variant == "qkv_t32x256x128bkn") {
        run_static_qkv_bkn_tiled_variant<32, 256, 128>(
            shape, opts, "qkv_t32x256x128bkn");
    }
    if (opts.variant == "all" || opts.variant == "t32x256x32bkn") {
        run_static_bkn_variant(shape, opts, "t32x256x32bkn");
    }
    if (opts.variant == "all" || opts.variant == "t32x64x64s") {
        run_static_variant<32, 64, 64>(shape, opts, "t32x64x64s");
    }
    if (opts.variant == "all" || opts.variant == "t32x64x32") {
        run_variant<32, 64, 32>(shape, opts, "t32x64x32");
    }
    if (opts.variant == "all" || opts.variant == "t32x64x32m") {
        run_variant<32, 64, 32, true, true>(shape, opts, "t32x64x32m");
    }
    if (opts.variant == "all" || opts.variant == "t32x64x64") {
        run_variant<32, 64, 64>(shape, opts, "t32x64x64");
    }
    if (opts.variant == "all" || opts.variant == "t32x128x16") {
        run_variant<32, 128, 16>(shape, opts, "t32x128x16");
    }
    if (opts.variant == "all" || opts.variant == "t64x64x16") {
        run_variant<64, 64, 16>(shape, opts, "t64x64x16");
    }
    if (opts.variant == "all" || opts.variant == "t32x256x16") {
        run_variant<32, 256, 16>(shape, opts, "t32x256x16");
    }
    run_static_masked_mn(std::integral_constant<int, 32>{},
                         std::integral_constant<int, 8>{},
                         std::integral_constant<int, 32>{},
                         "t32x8x32smn");
    run_static_masked_mn(std::integral_constant<int, 32>{},
                         std::integral_constant<int, 16>{},
                         std::integral_constant<int, 32>{},
                         "t32x16x32smn");
}

void run_small_shape(const Shape& shape, const Options& opts) {
    std::printf("%-15s M=%d N=%d K=%d\n", shape.name, shape.m, shape.n, shape.k);
    auto run_static = [&](auto tm, auto tn, auto tk, const char* name) {
        constexpr int TM = decltype(tm)::value;
        constexpr int TN = decltype(tn)::value;
        constexpr int TK = decltype(tk)::value;
        if (opts.variant != "all" && opts.variant != name) return;
        if (shape.n % TN != 0 || shape.k % TK != 0) {
            std::printf("  %-10s skipped: N or K not divisible by tile\n", name);
            return;
        }
        if (std::strcmp(shape.name, "band_n256_k16") == 0) {
            run_static_variant_impl<TM, TN, TK, 1280, 256, 16>(shape, opts, name);
        } else if (std::strcmp(shape.name, "band_n256_k32") == 0) {
            run_static_variant_impl<TM, TN, TK, 1280, 256, 32>(shape, opts, name);
        } else if (std::strcmp(shape.name, "band_n256_k64") == 0) {
            run_static_variant_impl<TM, TN, TK, 1280, 256, 64>(shape, opts, name);
        } else if (std::strcmp(shape.name, "mask_hid_k256") == 0) {
            run_static_variant_impl<TM, TN, TK, 1280, 1024, 256>(shape, opts, name);
        } else if (std::strcmp(shape.name, "mask_hid_k1024") == 0) {
            run_static_variant_impl<TM, TN, TK, 1280, 1024, 1024>(shape, opts, name);
        } else if (std::strcmp(shape.name, "mask_out128") == 0) {
            run_static_variant_impl<TM, TN, TK, 1280, 128, 1024>(shape, opts, name);
        } else {
            std::printf("  %-10s skipped: static small shape not supported\n", name);
        }
    };
    auto run_static_masked_m = [&](auto tm, auto tn, auto tk, const char* name) {
        constexpr int TM = decltype(tm)::value;
        constexpr int TN = decltype(tn)::value;
        constexpr int TK = decltype(tk)::value;
        if (opts.variant != "all" && opts.variant != name) return;
        if (shape.n % TN != 0 || shape.k % TK != 0) {
            std::printf("  %-10s skipped: N or K not divisible by tile\n", name);
            return;
        }
        if (std::strcmp(shape.name, "band_n256_k16") == 0) {
            run_static_masked_m_variant_impl<TM, TN, TK, 1301, 256, 16>(shape, opts, name);
        } else if (std::strcmp(shape.name, "band_n256_k32") == 0) {
            run_static_masked_m_variant_impl<TM, TN, TK, 1301, 256, 32>(shape, opts, name);
        } else if (std::strcmp(shape.name, "band_n256_k64") == 0) {
            run_static_masked_m_variant_impl<TM, TN, TK, 1301, 256, 64>(shape, opts, name);
        } else if (std::strcmp(shape.name, "mask_hid_k256") == 0) {
            run_static_masked_m_variant_impl<TM, TN, TK, 1301, 1024, 256>(shape, opts, name);
        } else if (std::strcmp(shape.name, "mask_hid_k1024") == 0) {
            run_static_masked_m_variant_impl<TM, TN, TK, 1301, 1024, 1024>(shape, opts, name);
        } else if (std::strcmp(shape.name, "mask_out128") == 0) {
            run_static_masked_m_variant_impl<TM, TN, TK, 1301, 128, 1024>(shape, opts, name);
        } else {
            std::printf("  %-10s skipped: static masked-M small shape not supported\n", name);
        }
    };
    auto run_static_padded_m = [&](auto tm, auto tn, auto tk, const char* name) {
        constexpr int TM = decltype(tm)::value;
        constexpr int TN = decltype(tn)::value;
        constexpr int TK = decltype(tk)::value;
        constexpr int MPad = ((1301 + TM - 1) / TM) * TM;
        if (opts.variant != "all" && opts.variant != name) return;
        if (shape.n % TN != 0 || shape.k % TK != 0) {
            std::printf("  %-10s skipped: N or K not divisible by tile\n", name);
            return;
        }
        if (std::strcmp(shape.name, "band_n256_k16") == 0) {
            run_static_padded_m_variant_impl<TM, TN, TK, MPad, 1301, 256, 16>(
                shape, opts, name);
        } else if (std::strcmp(shape.name, "band_n256_k32") == 0) {
            run_static_padded_m_variant_impl<TM, TN, TK, MPad, 1301, 256, 32>(
                shape, opts, name);
        } else if (std::strcmp(shape.name, "band_n256_k64") == 0) {
            run_static_padded_m_variant_impl<TM, TN, TK, MPad, 1301, 256, 64>(
                shape, opts, name);
        } else if (std::strcmp(shape.name, "mask_hid_k256") == 0) {
            run_static_padded_m_variant_impl<TM, TN, TK, MPad, 1301, 1024, 256>(
                shape, opts, name);
        } else if (std::strcmp(shape.name, "mask_hid_k1024") == 0) {
            run_static_padded_m_variant_impl<TM, TN, TK, MPad, 1301, 1024, 1024>(
                shape, opts, name);
        } else if (std::strcmp(shape.name, "mask_out128") == 0) {
            run_static_padded_m_variant_impl<TM, TN, TK, MPad, 1301, 128, 1024>(
                shape, opts, name);
        } else {
            std::printf("  %-10s skipped: static padded-M small shape not supported\n", name);
        }
    };
    auto run_static_masked_mn = [&](auto tm, auto tn, auto tk, const char* name) {
        constexpr int TM = decltype(tm)::value;
        constexpr int TN = decltype(tn)::value;
        constexpr int TK = decltype(tk)::value;
        if (opts.variant != "all" && opts.variant != name) return;
        if (shape.k % TK != 0) {
            std::printf("  %-10s skipped: K not divisible by tile\n", name);
            return;
        }
        if (std::strcmp(shape.name, "mask_out48") == 0) {
            run_static_masked_mn_variant_impl<TM, TN, TK, 1301, 48, 1024>(
                shape, opts, name);
        } else if (std::strcmp(shape.name, "mask_out400") == 0) {
            run_static_masked_mn_variant_impl<TM, TN, TK, 1301, 400, 1024>(
                shape, opts, name);
        } else if (std::strcmp(shape.name, "mask_out1040") == 0) {
            run_static_masked_mn_variant_impl<TM, TN, TK, 1301, 1040, 1024>(
                shape, opts, name);
        } else {
            std::printf("  %-10s skipped: static masked-MN small shape not supported\n", name);
        }
    };
    run_static(std::integral_constant<int, 32>{}, std::integral_constant<int, 32>{},
               std::integral_constant<int, 16>{}, "t32x32x16s");
    run_static(std::integral_constant<int, 32>{}, std::integral_constant<int, 64>{},
               std::integral_constant<int, 16>{}, "t32x64x16s");
    run_static(std::integral_constant<int, 32>{}, std::integral_constant<int, 128>{},
               std::integral_constant<int, 16>{}, "t32x128x16s");
    run_static(std::integral_constant<int, 32>{}, std::integral_constant<int, 256>{},
               std::integral_constant<int, 16>{}, "t32x256x16s");
    run_static(std::integral_constant<int, 64>{}, std::integral_constant<int, 64>{},
               std::integral_constant<int, 16>{}, "t64x64x16s");
    run_static(std::integral_constant<int, 32>{}, std::integral_constant<int, 64>{},
               std::integral_constant<int, 32>{}, "t32x64x32s");
    run_static(std::integral_constant<int, 64>{}, std::integral_constant<int, 64>{},
               std::integral_constant<int, 32>{}, "t64x64x32s");
    run_static(std::integral_constant<int, 32>{}, std::integral_constant<int, 64>{},
               std::integral_constant<int, 64>{}, "t32x64x64s");
    run_static_masked_m(std::integral_constant<int, 64>{}, std::integral_constant<int, 64>{},
                        std::integral_constant<int, 16>{}, "t64x64x16sm");
    run_static_masked_m(std::integral_constant<int, 32>{}, std::integral_constant<int, 64>{},
                        std::integral_constant<int, 64>{}, "t32x64x64sm");
    run_static_padded_m(std::integral_constant<int, 64>{}, std::integral_constant<int, 64>{},
                        std::integral_constant<int, 16>{}, "t64x64x16sp");
    run_static_padded_m(std::integral_constant<int, 64>{}, std::integral_constant<int, 64>{},
                        std::integral_constant<int, 32>{}, "t64x64x32sp");
    run_static_padded_m(std::integral_constant<int, 32>{}, std::integral_constant<int, 64>{},
                        std::integral_constant<int, 64>{}, "t32x64x64sp");
    run_static_padded_m(std::integral_constant<int, 32>{}, std::integral_constant<int, 128>{},
                        std::integral_constant<int, 32>{}, "t32x128x32sp");
    run_static_padded_m(std::integral_constant<int, 32>{}, std::integral_constant<int, 128>{},
                        std::integral_constant<int, 64>{}, "t32x128x64sp");
    run_static_padded_m(std::integral_constant<int, 64>{}, std::integral_constant<int, 128>{},
                        std::integral_constant<int, 32>{}, "t64x128x32sp");
    run_static_padded_m(std::integral_constant<int, 32>{}, std::integral_constant<int, 256>{},
                        std::integral_constant<int, 32>{}, "t32x256x32sp");
    run_static_masked_mn(std::integral_constant<int, 32>{},
                         std::integral_constant<int, 64>{},
                         std::integral_constant<int, 32>{},
                         "t32x64x32smn");
    run_static_masked_mn(std::integral_constant<int, 32>{},
                         std::integral_constant<int, 64>{},
                         std::integral_constant<int, 64>{},
                         "t32x64x64smn");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opts = parse_args(argc, argv);
        if (opts.preset == "infer_linear") {
            for (const Shape& shape : infer_linear_shapes()) {
                if (opts.shape != "all" && opts.shape != shape.name) {
                    continue;
                }
                run_shape(shape, opts);
            }
        } else if (opts.preset == "infer_small_linear") {
            for (const Shape& shape : infer_small_linear_shapes()) {
                if (opts.shape != "all" && opts.shape != shape.name) {
                    continue;
                }
                run_small_shape(shape, opts);
            }
        } else {
            throw std::runtime_error("unsupported preset: " + opts.preset);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
