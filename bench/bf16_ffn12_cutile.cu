#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
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

constexpr int kM = 78048;
constexpr int kIn = 256;
constexpr int kHidden = 1024;
constexpr int kOut = 256;
constexpr int kTileM = 32;
constexpr int kTileHidden = 64;
constexpr int kTileK = 32;
constexpr int kInitTile = 256;
constexpr double kA10gDenseBf16Tflops = 70.0;
constexpr int kGeluErf = 0;
constexpr int kGeluHard = 1;
constexpr int kGeluQuick = 2;
constexpr int kGeluTanh = 3;
constexpr int kGeluErfPoly5L25 = 4;
constexpr int kGeluErfPoly7L25 = 5;
constexpr int kGeluErfPoly9L30 = 6;

using I64InitTile = ct::tile<long long, ct::shape<kInitTile>>;
using F32InitTile = ct::tile<float, ct::shape<kInitTile>>;

struct Options {
    std::string variant = "all";
    int warmup = 1;
    int iters = 4;
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
        if (std::strcmp(argv[i], "--variant") == 0) {
            opts.variant = need_value(argv[i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0) {
            opts.warmup = parse_int_arg(argv[i], need_value(argv[i]));
        } else if (std::strcmp(argv[i], "--iters") == 0) {
            opts.iters = parse_int_arg(argv[i], need_value(argv[i]));
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::printf(
                "Usage: bench_bf16_ffn12_cutile [options]\n"
                "  --variant NAME  all, separate, fused64, fused128, fused256,\n"
                "                  fused256_occ2, fused256_occ4,\n"
                "                  fused_m16_256, fused_h32_256,\n"
                "                  fused_h32_128, fused_h32_64,\n"
                "                  fused_h32_poly9, fused_h32_poly9_128,\n"
                "                  fused_h32_poly9_64,\n"
                "                  fused_h16_poly9_split2,\n"
                "                  fused_h32_poly9_split2,\n"
                "                  fused_h64_poly9_split2,\n"
                "                  fused_h32_poly9_split4,\n"
                "                  fused_m16_h32_256, fused_h16_256,\n"
                "                  fused_m16_h16_256; default all\n"
                "  --warmup N      warmup launches per variant, default 1\n"
                "  --iters N       measured launches per variant, default 4\n");
            std::exit(0);
        } else {
            throw std::runtime_error(std::string("unknown argument: ") + argv[i]);
        }
    }
    return opts;
}

float percentile(std::vector<float> values, float q) {
    std::sort(values.begin(), values.end());
    float pos = q * static_cast<float>(values.size() - 1);
    int lo = static_cast<int>(pos);
    int hi = std::min(lo + 1, static_cast<int>(values.size() - 1));
    float t = pos - static_cast<float>(lo);
    return values[lo] * (1.0f - t) + values[hi] * t;
}

template <typename T>
static __tile__ auto bf16_round(T value) {
    return ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(value));
}

template <typename TileT>
static __tile__ auto gelu_erf_approx(TileT x) {
    auto zero = x * 0.0f;
    auto one = zero + 1.0f;
    auto ax = ct::abs(x);
    auto sign = ct::select(x < zero, zero - one, one);
    auto t = one / (one + 0.3275911f * ax);
    auto poly =
        (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t -
          0.284496736f) * t + 0.254829592f) * t;
    auto erf_approx = sign * (one - poly * ct::exp(zero - ax * ax));
    return 0.5f * x * (one + erf_approx);
}

template <typename TileT>
static __tile__ auto gelu_hard_approx(TileT x) {
    auto zero = x * 0.0f;
    auto gate = ct::min(ct::max(0.5f + 0.2f * x, zero), zero + 1.0f);
    return x * gate;
}

template <typename TileT>
static __tile__ auto gelu_quick_approx(TileT x) {
    auto sigmoid = 1.0f / (1.0f + ct::exp(-1.702f * x));
    return x * sigmoid;
}

template <typename TileT>
static __tile__ auto gelu_tanh_approx(TileT x) {
    auto cubic = x * x * x;
    auto inner = 0.7978845608f * (x + 0.044715f * cubic);
    return x * (0.5f * (1.0f + tanh(inner)));
}

template <typename TileT>
static __tile__ auto gelu_erf_poly5_l25(TileT x) {
    auto zero = x * 0.0f;
    auto one = zero + 1.0f;
    auto ax = ct::abs(x);
    auto z = ax * ax;
    auto p = (((0.000677416775f * z - 0.0121774335f) * z +
               0.0889425898f) * z - 0.361254819f) * z +
             1.12684393f;
    auto erf_abs = ct::min(ct::max(ax * p, zero), one);
    auto erf_approx = ct::select(x < zero, zero - erf_abs, erf_abs);
    return 0.5f * x * (one + erf_approx);
}

template <typename TileT>
static __tile__ auto gelu_erf_poly7_l25(TileT x) {
    auto zero = x * 0.0f;
    auto one = zero + 1.0f;
    auto ax = ct::abs(x);
    auto z = ax * ax;
    auto p = ((((((0.0000119948033f * z - 0.000310497426f) * z +
                  0.00352976049f) * z - 0.0238667561f) * z +
                0.110178845f) * z - 0.37522094f) * z +
              1.12832882f);
    auto erf_abs = ct::min(ct::max(ax * p, zero), one);
    auto erf_approx = ct::select(x < zero, zero - erf_abs, erf_abs);
    return 0.5f * x * (one + erf_approx);
}

template <typename TileT>
static __tile__ auto gelu_erf_poly9_l30(TileT x) {
    auto zero = x * 0.0f;
    auto one = zero + 1.0f;
    auto ax = ct::abs(x);
    auto z = ax * ax;
    auto p = ((((((((0.00000005422539767f * z - 0.000002440964777f) * z +
                    0.00004855766724f) * z - 0.0005709642654f) * z +
                  0.004507274577f) * z - 0.02579950512f) * z +
                0.1120213868f) * z - 0.3758834075f) * z +
              1.128367753f);
    auto erf_abs = ct::min(ct::max(ax * p, zero), one);
    auto erf_approx = ct::select(x < zero, zero - erf_abs, erf_abs);
    return 0.5f * x * (one + erf_approx);
}

template <int GeluMode, typename TileT>
static __tile__ auto gelu_selected(TileT x) {
    if constexpr (GeluMode == kGeluErfPoly9L30) {
        return gelu_erf_poly9_l30(x);
    } else if constexpr (GeluMode == kGeluErfPoly7L25) {
        return gelu_erf_poly7_l25(x);
    } else if constexpr (GeluMode == kGeluErfPoly5L25) {
        return gelu_erf_poly5_l25(x);
    } else if constexpr (GeluMode == kGeluTanh) {
        return gelu_tanh_approx(x);
    } else if constexpr (GeluMode == kGeluQuick) {
        return gelu_quick_approx(x);
    } else if constexpr (GeluMode == kGeluHard) {
        return gelu_hard_approx(x);
    } else {
        static_assert(GeluMode == kGeluErf);
        return gelu_erf_approx(x);
    }
}

__tile_global__ void fill_bf16_kernel(__nv_bfloat16* __restrict__ dst, long long total) {
    dst = ct::assume_aligned(dst, 16_ic);
    I64InitTile idx = (long long)ct::bid().x * kInitTile + ct::iota<I64InitTile>();
    auto in_bounds = idx < total;
    F32InitTile values =
        0.125f + ct::element_cast<float>((idx * 17LL) & 1023LL) * 0.000244140625f;
    ct::store_masked(dst + idx, ct::element_cast<__nv_bfloat16>(values), in_bounds);
}

template <int TM, int TN, int TK, int M, int N, int K>
__tile_global__ void ffn1_gelu_bf16_kernel(const __nv_bfloat16* __restrict__ a,
                                           const __nv_bfloat16* __restrict__ w1_nt,
                                           const __nv_bfloat16* __restrict__ b1,
                                           __nv_bfloat16* __restrict__ hidden) {
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using I64Tile = ct::tile<long long, ct::shape<TM, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    w1_nt = ct::assume_aligned(w1_nt, 16_ic);
    b1 = ct::assume_aligned(b1, 16_ic);
    hidden = ct::assume_aligned(hidden, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<K, N>{}, ct::layout_left{}},
        ct::shape<TK, TN>{}
    };
    auto hidden_view = ct::partition_view{
        ct::tensor_span{hidden, ct::shape<M, N>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        acc = ct::mma(a_view.load(tile_m, kk), w1_view.load(kk, tile_n), acc);
    }

    I64Tile local = ct::iota<I64Tile>();
    auto cols = static_cast<long long>(tile_n) * TN + (local % TN);
    auto bias = ct::element_cast<float>(ct::load(b1 + cols));
    auto value = gelu_erf_approx(bf16_round(acc) + bias);
    hidden_view.store(ct::element_cast<__nv_bfloat16>(value), tile_m, tile_n);
}

template <int TM, int TN, int TK, int M, int N, int K>
__tile_global__ void ffn2_bf16_kernel(const __nv_bfloat16* __restrict__ hidden,
                                      const __nv_bfloat16* __restrict__ w2_nt,
                                      const __nv_bfloat16* __restrict__ b2,
                                      __nv_bfloat16* __restrict__ out) {
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using I64Tile = ct::tile<long long, ct::shape<TM, TN>>;

    hidden = ct::assume_aligned(hidden, 16_ic);
    w2_nt = ct::assume_aligned(w2_nt, 16_ic);
    b2 = ct::assume_aligned(b2, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto hidden_view = ct::partition_view{
        ct::tensor_span{hidden, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto w2_view = ct::partition_view{
        ct::tensor_span{w2_nt, ct::shape<K, N>{}, ct::layout_left{}},
        ct::shape<TK, TN>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out, ct::shape<M, N>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        acc = ct::mma(hidden_view.load(tile_m, kk), w2_view.load(kk, tile_n), acc);
    }

    I64Tile local = ct::iota<I64Tile>();
    auto cols = static_cast<long long>(tile_n) * TN + (local % TN);
    auto bias = ct::element_cast<float>(ct::load(b2 + cols));
    auto value = bf16_round(acc) + bias;
    out_view.store(ct::element_cast<__nv_bfloat16>(value), tile_m, tile_n);
}

template <int TM, int TNOut, int THidden, int GeluMode = kGeluErf>
static __tile__ void ffn12_fused_recompute_bf16_body(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    using HiddenAccTile = ct::tile<float, ct::shape<TM, THidden>>;
    using OutAccTile = ct::tile<float, ct::shape<TM, TNOut>>;
    using I64HiddenTile = ct::tile<long long, ct::shape<TM, THidden>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, TNOut>>;

    a = ct::assume_aligned(a, 16_ic);
    w1_nt = ct::assume_aligned(w1_nt, 16_ic);
    b1 = ct::assume_aligned(b1, 16_ic);
    w2_nt = ct::assume_aligned(w2_nt, 16_ic);
    b2 = ct::assume_aligned(b2, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<kM, kIn>{}},
        ct::shape<TM, kTileK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<kIn, kHidden>{}, ct::layout_left{}},
        ct::shape<kTileK, THidden>{}
    };
    auto w2_view = ct::partition_view{
        ct::tensor_span{w2_nt, ct::shape<kHidden, kOut>{}, ct::layout_left{}},
        ct::shape<THidden, TNOut>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out, ct::shape<kM, kOut>{}},
        ct::shape<TM, TNOut>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto out_acc = ct::full<OutAccTile>(0.0f);
    I64HiddenTile hidden_local = ct::iota<I64HiddenTile>();
    for (auto hidden_tile : ct::irange(std::size_t{0},
                                       std::size_t{kHidden / THidden})) {
        auto hidden_acc = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / kTileK})) {
            hidden_acc = ct::mma(a_view.load(tile_m, kk),
                                 w1_view.load(kk, hidden_tile),
                                 hidden_acc);
        }
        auto hidden_cols =
            static_cast<long long>(hidden_tile) * THidden +
            (hidden_local % THidden);
        auto hidden_bias = ct::element_cast<float>(ct::load(b1 + hidden_cols));
        auto hidden_value =
            ct::element_cast<__nv_bfloat16>(gelu_selected<GeluMode>(bf16_round(hidden_acc) +
                                                                    hidden_bias));
        out_acc = ct::mma(hidden_value, w2_view.load(hidden_tile, tile_n), out_acc);
    }

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_cols = static_cast<long long>(tile_n) * TNOut + (out_local % TNOut);
    auto out_bias = ct::element_cast<float>(ct::load(b2 + out_cols));
    auto value = bf16_round(out_acc) + out_bias;
    out_view.store(ct::element_cast<__nv_bfloat16>(value), tile_m, tile_n);
}

template <int TM, int TNOut, int THidden, int GeluMode = kGeluErf>
__tile_global__ void ffn12_fused_recompute_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    ffn12_fused_recompute_bf16_body<TM, TNOut, THidden, GeluMode>(
        a, w1_nt, b1, w2_nt, b2, out);
}

template <int TM, int TNOut, int THidden, int GeluMode = kGeluErf>
[[cutile::hint(860, occupancy=2)]]
__tile_global__ void ffn12_fused_recompute_bf16_occ2_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    ffn12_fused_recompute_bf16_body<TM, TNOut, THidden, GeluMode>(
        a, w1_nt, b1, w2_nt, b2, out);
}

template <int TM, int TNOut, int THidden, int GeluMode = kGeluErf>
[[cutile::hint(860, occupancy=4)]]
__tile_global__ void ffn12_fused_recompute_bf16_occ4_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    ffn12_fused_recompute_bf16_body<TM, TNOut, THidden, GeluMode>(
        a, w1_nt, b1, w2_nt, b2, out);
}

template <int TM, int THidden, int GeluMode = kGeluErf>
__tile_global__ void ffn12_fused_split2_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    constexpr int OutHalf = kOut / 2;
    using HiddenAccTile = ct::tile<float, ct::shape<TM, THidden>>;
    using OutAccTile = ct::tile<float, ct::shape<TM, OutHalf>>;
    using I64HiddenTile = ct::tile<long long, ct::shape<TM, THidden>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, OutHalf>>;

    a = ct::assume_aligned(a, 16_ic);
    w1_nt = ct::assume_aligned(w1_nt, 16_ic);
    b1 = ct::assume_aligned(b1, 16_ic);
    w2_nt = ct::assume_aligned(w2_nt, 16_ic);
    b2 = ct::assume_aligned(b2, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<kM, kIn>{}},
        ct::shape<TM, kTileK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<kIn, kHidden>{}, ct::layout_left{}},
        ct::shape<kTileK, THidden>{}
    };
    auto w2_view = ct::partition_view{
        ct::tensor_span{w2_nt, ct::shape<kHidden, kOut>{}, ct::layout_left{}},
        ct::shape<THidden, OutHalf>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out, ct::shape<kM, kOut>{}},
        ct::shape<TM, OutHalf>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_n;
    (void)tile_z;

    auto out_acc0 = ct::full<OutAccTile>(0.0f);
    auto out_acc1 = ct::full<OutAccTile>(0.0f);
    I64HiddenTile hidden_local = ct::iota<I64HiddenTile>();
    for (auto hidden_tile : ct::irange(std::size_t{0}, std::size_t{kHidden / THidden})) {
        auto hidden_acc = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / kTileK})) {
            hidden_acc = ct::mma(a_view.load(tile_m, kk),
                                 w1_view.load(kk, hidden_tile),
                                 hidden_acc);
        }
        auto hidden_cols =
            static_cast<long long>(hidden_tile) * THidden + (hidden_local % THidden);
        auto hidden_bias = ct::element_cast<float>(ct::load(b1 + hidden_cols));
        auto hidden_value =
            ct::element_cast<__nv_bfloat16>(gelu_selected<GeluMode>(bf16_round(hidden_acc) +
                                                                    hidden_bias));
        out_acc0 = ct::mma(hidden_value, w2_view.load(hidden_tile, 0), out_acc0);
        out_acc1 = ct::mma(hidden_value, w2_view.load(hidden_tile, 1), out_acc1);
    }

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_cols = out_local % OutHalf;
    auto out_bias0 = ct::element_cast<float>(ct::load(b2 + out_cols));
    auto out_bias1 = ct::element_cast<float>(ct::load(b2 + OutHalf + out_cols));
    auto value0 = bf16_round(out_acc0) + out_bias0;
    auto value1 = bf16_round(out_acc1) + out_bias1;
    out_view.store(ct::element_cast<__nv_bfloat16>(value0), tile_m, 0);
    out_view.store(ct::element_cast<__nv_bfloat16>(value1), tile_m, 1);
}

template <int TM, int THidden, int GeluMode = kGeluErf>
__tile_global__ void ffn12_fused_split4_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    constexpr int OutQuarter = kOut / 4;
    using HiddenAccTile = ct::tile<float, ct::shape<TM, THidden>>;
    using OutAccTile = ct::tile<float, ct::shape<TM, OutQuarter>>;
    using I64HiddenTile = ct::tile<long long, ct::shape<TM, THidden>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, OutQuarter>>;

    a = ct::assume_aligned(a, 16_ic);
    w1_nt = ct::assume_aligned(w1_nt, 16_ic);
    b1 = ct::assume_aligned(b1, 16_ic);
    w2_nt = ct::assume_aligned(w2_nt, 16_ic);
    b2 = ct::assume_aligned(b2, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<kM, kIn>{}},
        ct::shape<TM, kTileK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<kIn, kHidden>{}, ct::layout_left{}},
        ct::shape<kTileK, THidden>{}
    };
    auto w2_view = ct::partition_view{
        ct::tensor_span{w2_nt, ct::shape<kHidden, kOut>{}, ct::layout_left{}},
        ct::shape<THidden, OutQuarter>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out, ct::shape<kM, kOut>{}},
        ct::shape<TM, OutQuarter>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_n;
    (void)tile_z;

    auto out_acc0 = ct::full<OutAccTile>(0.0f);
    auto out_acc1 = ct::full<OutAccTile>(0.0f);
    auto out_acc2 = ct::full<OutAccTile>(0.0f);
    auto out_acc3 = ct::full<OutAccTile>(0.0f);
    I64HiddenTile hidden_local = ct::iota<I64HiddenTile>();
    for (auto hidden_tile : ct::irange(std::size_t{0}, std::size_t{kHidden / THidden})) {
        auto hidden_acc = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / kTileK})) {
            hidden_acc = ct::mma(a_view.load(tile_m, kk),
                                 w1_view.load(kk, hidden_tile),
                                 hidden_acc);
        }
        auto hidden_cols =
            static_cast<long long>(hidden_tile) * THidden + (hidden_local % THidden);
        auto hidden_bias = ct::element_cast<float>(ct::load(b1 + hidden_cols));
        auto hidden_value =
            ct::element_cast<__nv_bfloat16>(gelu_selected<GeluMode>(bf16_round(hidden_acc) +
                                                                    hidden_bias));
        out_acc0 = ct::mma(hidden_value, w2_view.load(hidden_tile, 0), out_acc0);
        out_acc1 = ct::mma(hidden_value, w2_view.load(hidden_tile, 1), out_acc1);
        out_acc2 = ct::mma(hidden_value, w2_view.load(hidden_tile, 2), out_acc2);
        out_acc3 = ct::mma(hidden_value, w2_view.load(hidden_tile, 3), out_acc3);
    }

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_cols = out_local % OutQuarter;
    auto out_bias0 = ct::element_cast<float>(ct::load(b2 + out_cols));
    auto out_bias1 = ct::element_cast<float>(ct::load(b2 + OutQuarter + out_cols));
    auto out_bias2 = ct::element_cast<float>(ct::load(b2 + 2 * OutQuarter + out_cols));
    auto out_bias3 = ct::element_cast<float>(ct::load(b2 + 3 * OutQuarter + out_cols));
    auto value0 = bf16_round(out_acc0) + out_bias0;
    auto value1 = bf16_round(out_acc1) + out_bias1;
    auto value2 = bf16_round(out_acc2) + out_bias2;
    auto value3 = bf16_round(out_acc3) + out_bias3;
    out_view.store(ct::element_cast<__nv_bfloat16>(value0), tile_m, 0);
    out_view.store(ct::element_cast<__nv_bfloat16>(value1), tile_m, 1);
    out_view.store(ct::element_cast<__nv_bfloat16>(value2), tile_m, 2);
    out_view.store(ct::element_cast<__nv_bfloat16>(value3), tile_m, 3);
}

void init_bf16(__nv_bfloat16* ptr, size_t elems) {
    fill_bf16_kernel<<<ceildiv(static_cast<int>(elems), kInitTile), 1>>>(
        ptr, static_cast<long long>(elems));
    CUDA_CHECK(cudaGetLastError());
}

void launch_separate(const __nv_bfloat16* d_a,
                     const __nv_bfloat16* d_w1,
                     const __nv_bfloat16* d_b1,
                     const __nv_bfloat16* d_w2,
                     const __nv_bfloat16* d_b2,
                     __nv_bfloat16* d_hidden,
                     __nv_bfloat16* d_out) {
    dim3 ffn1_grid(kM / kTileM, kHidden / kTileHidden, 1);
    ffn1_gelu_bf16_kernel<kTileM, kTileHidden, kTileK, kM, kHidden, kIn>
        <<<ffn1_grid, 1>>>(d_a, d_w1, d_b1, d_hidden);
    dim3 ffn2_grid(kM / kTileM, kOut / kTileHidden, 1);
    ffn2_bf16_kernel<kTileM, kTileHidden, kTileHidden, kM, kOut, kHidden>
        <<<ffn2_grid, 1>>>(d_hidden, d_w2, d_b2, d_out);
}

template <int TM,
          int TNOut,
          int THidden = kTileHidden,
          int GeluMode = kGeluErf,
          int OccupancyHint = 0>
void launch_fused(const __nv_bfloat16* d_a,
                  const __nv_bfloat16* d_w1,
                  const __nv_bfloat16* d_b1,
                  const __nv_bfloat16* d_w2,
                  const __nv_bfloat16* d_b2,
                  __nv_bfloat16* d_out) {
    dim3 grid(kM / TM, kOut / TNOut, 1);
    if constexpr (OccupancyHint == 2) {
        ffn12_fused_recompute_bf16_occ2_kernel<TM, TNOut, THidden, GeluMode>
            <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
    } else if constexpr (OccupancyHint == 4) {
        ffn12_fused_recompute_bf16_occ4_kernel<TM, TNOut, THidden, GeluMode>
            <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
    } else {
        ffn12_fused_recompute_bf16_kernel<TM, TNOut, THidden, GeluMode>
            <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
    }
}

template <int TM, int THidden = kTileHidden, int GeluMode = kGeluErf>
void launch_fused_split2(const __nv_bfloat16* d_a,
                         const __nv_bfloat16* d_w1,
                         const __nv_bfloat16* d_b1,
                         const __nv_bfloat16* d_w2,
                         const __nv_bfloat16* d_b2,
                         __nv_bfloat16* d_out) {
    dim3 grid(kM / TM, 1, 1);
    ffn12_fused_split2_bf16_kernel<TM, THidden, GeluMode>
        <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
}

template <int TM, int THidden = kTileHidden, int GeluMode = kGeluErf>
void launch_fused_split4(const __nv_bfloat16* d_a,
                         const __nv_bfloat16* d_w1,
                         const __nv_bfloat16* d_b1,
                         const __nv_bfloat16* d_w2,
                         const __nv_bfloat16* d_b2,
                         __nv_bfloat16* d_out) {
    dim3 grid(kM / TM, 1, 1);
    ffn12_fused_split4_bf16_kernel<TM, THidden, GeluMode>
        <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
}

template <typename Launch>
void run_variant(const char* name,
                 int tile_m,
                 int out_tile,
                 int recompute_factor,
                 int launches,
                 const Options& opts,
                 const __nv_bfloat16* d_a,
                 const __nv_bfloat16* d_w1,
                 const __nv_bfloat16* d_b1,
                 const __nv_bfloat16* d_w2,
                 const __nv_bfloat16* d_b2,
                 __nv_bfloat16* d_hidden,
                 __nv_bfloat16* d_out,
                 Launch launch) {
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
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
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
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    double ffn1_flops = 2.0 * kM * kHidden * kIn;
    double ffn2_flops = 2.0 * kM * kOut * kHidden;
    double useful_flops = ffn1_flops + ffn2_flops;
    double actual_flops = ffn1_flops * recompute_factor + ffn2_flops;
    double hidden_gib =
        (2.0 * static_cast<double>(kM) * kHidden * sizeof(__nv_bfloat16)) /
        (1024.0 * 1024.0 * 1024.0);
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double useful_tf = useful_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double actual_tf = actual_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "%-14s tile=%2dx%3d launches=%d recompute_ffn1=%d best=%.3f ms median=%.3f ms useful=%.2f TF/s roof=%.1f%% actual=%.2f TF/s roof=%.1f%% hidden_rw=%.3f GiB checksum=%.4f\n",
        name, tile_m, out_tile, launches, recompute_factor, best_ms, median_ms,
        useful_tf, useful_tf * 100.0 / kA10gDenseBf16Tflops,
        actual_tf, actual_tf * 100.0 / kA10gDenseBf16Tflops,
        hidden_gib, __bfloat162float(checksum_bf16));

    (void)d_a;
    (void)d_w1;
    (void)d_b1;
    (void)d_w2;
    (void)d_b2;
    (void)d_hidden;
}

bool should_run(const Options& opts, const char* name) {
    return opts.variant == "all" || opts.variant == name;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opts = parse_args(argc, argv);
        size_t a_elems = static_cast<size_t>(kM) * kIn;
        size_t w1_elems = static_cast<size_t>(kHidden) * kIn;
        size_t b1_elems = kHidden;
        size_t hidden_elems = static_cast<size_t>(kM) * kHidden;
        size_t w2_elems = static_cast<size_t>(kOut) * kHidden;
        size_t b2_elems = kOut;
        size_t out_elems = static_cast<size_t>(kM) * kOut;

        __nv_bfloat16* d_a = nullptr;
        __nv_bfloat16* d_w1 = nullptr;
        __nv_bfloat16* d_b1 = nullptr;
        __nv_bfloat16* d_hidden = nullptr;
        __nv_bfloat16* d_w2 = nullptr;
        __nv_bfloat16* d_b2 = nullptr;
        __nv_bfloat16* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_a, a_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_w1, w1_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_b1, b1_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_hidden, hidden_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_w2, w2_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_b2, b2_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_out, out_elems * sizeof(__nv_bfloat16)));

        init_bf16(d_a, a_elems);
        init_bf16(d_w1, w1_elems);
        init_bf16(d_b1, b1_elems);
        init_bf16(d_w2, w2_elems);
        init_bf16(d_b2, b2_elems);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::printf("FFN1->GELU->FFN2 CUDA Tile fusion probe\n");
        std::printf("shape: M=%d, in=%d, hidden=%d, out=%d, BF16 storage, FP32 mma accumulate\n",
                    kM, kIn, kHidden, kOut);

        if (should_run(opts, "separate")) {
            run_variant("separate", 32, 64, 1, 2, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_separate(d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out);
                        });
        }
        if (should_run(opts, "fused64")) {
            run_variant("fused64", 32, 64, 4, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused<32, 64>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused128")) {
            run_variant("fused128", 32, 128, 2, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused<32, 128>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused256")) {
            run_variant("fused256", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused<32, 256>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused256_occ2")) {
            run_variant("fused256_occ2", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused<32, 256, kTileHidden, kGeluErf, 2>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused256_occ4")) {
            run_variant("fused256_occ4", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused<32, 256, kTileHidden, kGeluErf, 4>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused256_hard")) {
            run_variant("fused256_hard", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused<32, 256, kTileHidden, kGeluHard>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused256_quick")) {
            run_variant("fused256_quick", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused<32, 256, kTileHidden, kGeluQuick>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused256_tanh")) {
            run_variant("fused256_tanh", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused<32, 256, kTileHidden, kGeluTanh>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused256_poly5")) {
            run_variant("fused256_poly5", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused<32, 256, kTileHidden, kGeluErfPoly5L25>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused256_poly7")) {
            run_variant("fused256_poly7", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused<32, 256, kTileHidden, kGeluErfPoly7L25>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused256_poly9")) {
            run_variant("fused256_poly9", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused<32, 256, kTileHidden, kGeluErfPoly9L30>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_m16_256")) {
            run_variant("fused_m16_256", 16, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused<16, 256>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_256")) {
            run_variant("fused_h32_256", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused<32, 256, 32>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_128")) {
            run_variant("fused_h32_128", 32, 128, 2, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused<32, 128, 32>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_64")) {
            run_variant("fused_h32_64", 32, 64, 4, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused<32, 64, 32>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9")) {
            run_variant("fused_h32_poly9", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused<32, 256, 32, kGeluErfPoly9L30>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h16_poly9_split2")) {
            run_variant("fused_h16_poly9_split2", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2<32, 16, kGeluErfPoly9L30>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2")) {
            run_variant("fused_h32_poly9_split2", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2<32, 32, kGeluErfPoly9L30>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h64_poly9_split2")) {
            run_variant("fused_h64_poly9_split2", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2<32, 64, kGeluErfPoly9L30>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split4")) {
            run_variant("fused_h32_poly9_split4", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split4<32, 32, kGeluErfPoly9L30>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_128")) {
            run_variant("fused_h32_poly9_128", 32, 128, 2, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused<32, 128, 32, kGeluErfPoly9L30>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_64")) {
            run_variant("fused_h32_poly9_64", 32, 64, 4, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused<32, 64, 32, kGeluErfPoly9L30>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_m16_h32_256")) {
            run_variant("fused_m16_h32_256", 16, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused<16, 256, 32>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h16_256")) {
            run_variant("fused_h16_256", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused<32, 256, 16>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_m16_h16_256")) {
            run_variant("fused_m16_h16_256", 16, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused<16, 256, 16>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_w1));
        CUDA_CHECK(cudaFree(d_b1));
        CUDA_CHECK(cudaFree(d_hidden));
        CUDA_CHECK(cudaFree(d_w2));
        CUDA_CHECK(cudaFree(d_b2));
        CUDA_CHECK(cudaFree(d_out));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
