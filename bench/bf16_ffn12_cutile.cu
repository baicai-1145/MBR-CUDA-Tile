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
constexpr int kGeluIdentity = 7;
constexpr int kGeluErfPoly9TinyBlend = 8;

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
                "                  fused_m16_h16_poly9_split2,\n"
                "                  fused_m16_h32_poly9_split2,\n"
                "                  fused_m16_h64_poly9_split2,\n"
                "                  fused_h32_poly9_split2,\n"
                "                  fused_h32_identity_split2,\n"
                "                  fused_h32_identity_nooutbias_split2,\n"
                "                  fused_h32_rawhidden_split2,\n"
                "                  fused_h32_poly9_split2_pairh32,\n"
                "                  fused_h32_poly9_split2_pairh32_tk16,\n"
                "                  fused_h32_poly9_split2_pairh32_tk64,\n"
#ifdef CUDASEP_FFN12_CANDIDATES_ONLY
                "                  fused_h32_poly9_split2_tk64,\n"
                "                  fused_h32_poly9_split2_pairh32_tk64_bkn,\n"
		                "                  fused_m8_h32_poly9_split2_pairh32_tk64,\n"
		                "                  fused_h32_poly9_split2_pairh32_tk128,\n"
		                "                  fused_h32_poly9_split2_pairh32_tk256,\n"
		                "                  fused_h64_poly9_split2_pairh64_tk64,\n"
	                "                  fused_h64_poly9_split2_pairh64_tk64_idx32,\n"
	                "                  fused_h64_poly9_split2_pairh64_tk64_outnoround,\n"
	                "                  fused_h64_poly9_split2_pairh64_tk64_idx32_outnoround,\n"
	                "                  fused_h32_poly9_split2_pairh32_tk64_halfout,\n"
	                "                  fused_h32_poly9_split2_pairh32_tk64_quarterout,\n"
	                "                  fused_h32_poly9_split2_pairh32_tk64_quarterout_w1w2lat2,\n"
	                "                  fused_h32_poly9_split2_pairh32_tk64_outseq,\n"
                "                  fused_h32_poly9_split2_pairh32_tk64_noround,\n"
                "                  fused_h32_poly9_split2_pairh32_tk64_outnoround,\n"
                "                  fused_h32_poly9_split2_pairh32_tk64_idx32,\n"
                "                  fused_h32_poly9_split2_pairh32_tk64_idx32_outnoround,\n"
	                "                  fused_h32_poly9_split2_pairh32_tk64_nooutbias,\n"
		                "                  fused_h32_poly9_split2_pairh32_tk64_nobias,\n"
		                "                  fused_h32_poly9_split2_pairh32_tk64_w1lat8,\n"
		                "                  fused_h32_poly9_split2_pairh32_tk64_w1lat2,\n"
		                "                  fused_h32_poly9_split2_pairh32_tk64_w1batched2,\n"
		                "                  fused_h32_poly9_split2_pairh32_tk64_w2batched2,\n"
		                "                  fused_h32_poly9_split2_pairh32_tk64_hsplit2_partial,\n"
		                "                  fused_h32_poly9_split2_pairh32_tk64_hsplit4_partial,\n"
		                "                  fused_h32_poly9_split2_pairh32_tk64_stagedhidden,\n"
		                "                  fused_h32_poly9_split4_pairh32_tk64,\n"
#endif
                "                  fused_h32_poly9_split2_pairh32_tk64_accgroup,\n"
                "                  fused_h32_poly9_split2_pairh32_tk64_w2lat8,\n"
                "                  fused_h32_poly9_split2_pairh32_tk64_w2lat2,\n"
                "                  fused_h32_poly9_split2_pairh32_tk64_w1w2lat2,\n"
                "                  fused_h32_poly9_split2_pairh32_tk64_w2temp,\n"
                "                  fused_h32_poly9_split2_pairh32_tk64_w2splitspan,\n"
                "                  fused_h32_poly9_split2_pairh32_tk64_w2manual,\n"
                "                  fused_h32_poly9_split2_pairh32_tk64_occ2,\n"
                "                  fused_h16_poly9_split2_pairh16_tk64,\n"
                "                  fused_h64_poly9_split2_pairh64_tk64,\n"
                "                  fused_m16_h32_poly9_split2_pairh32,\n"
                "                  fused_m16_h32_poly9_split2_pairh32_tk64,\n"
                "                  fused_h32_poly9_tinyblend_split2_pairh32,\n"
                "                  fused_h32_poly9_tinyblend_split2_pairh32_tk64,\n"
                "                  fused_h32_identity_split2_pairh32,\n"
                "                  fused_h32_identity_split2_pairh32_tk64,\n"
                "                  fused_h32_poly9_split2_pairh32_source_style,\n"
                "                  fused_h32_poly9_split4_pairh32,\n"
                "                  fused_h32_poly9_split2_quadh32,\n"
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

template <bool FullBF16, typename T>
static __tile__ auto bf16_round_if(T value) {
    if constexpr (FullBF16) {
        return bf16_round(value);
    }
    return value;
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

template <typename TileT>
static __tile__ auto gelu_erf_poly9_tinyblend_l30(TileT x) {
    auto gelu = gelu_erf_poly9_l30(x);
    return x + 0.0009765625f * (gelu - x);
}

template <int GeluMode, typename TileT>
static __tile__ auto gelu_selected(TileT x) {
    if constexpr (GeluMode == kGeluIdentity) {
        return x;
    } else if constexpr (GeluMode == kGeluErfPoly9L30) {
        return gelu_erf_poly9_l30(x);
    } else if constexpr (GeluMode == kGeluErfPoly9TinyBlend) {
        return gelu_erf_poly9_tinyblend_l30(x);
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

template <bool FullBF16, typename TileT>
static __tile__ auto gelu_erf_poly9_l30_source_style(TileT x) {
    auto zero = x * 0.0f;
    auto one = zero + 1.0f;
    auto ax = ct::select(x < zero, zero - x, x);
    auto z = ax * ax;
    z = bf16_round_if<FullBF16>(z);
    auto p = ((((((((0.00000005422539767f * z - 0.000002440964777f) * z +
                    0.00004855766724f) * z - 0.0005709642654f) * z +
                  0.004507274577f) * z - 0.02579950512f) * z +
                0.1120213868f) * z - 0.3758834075f) * z +
              1.128367753f);
    p = bf16_round_if<FullBF16>(p);
    auto erf_abs = ct::min(ct::max(ax * p, zero), one);
    erf_abs = bf16_round_if<FullBF16>(erf_abs);
    auto erf_approx = ct::select(x < zero, zero - erf_abs, erf_abs);
    auto gelu = 0.5f * x * (one + erf_approx);
    return bf16_round_if<FullBF16>(gelu);
}

template <int GeluMode, bool FullBF16, typename TileT>
static __tile__ auto gelu_selected_source_style(TileT x) {
    if constexpr (GeluMode == kGeluErfPoly9L30) {
        return gelu_erf_poly9_l30_source_style<FullBF16>(x);
    } else {
        static_assert(GeluMode == kGeluErfPoly9L30);
        return x;
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

#ifdef CUDASEP_FFN12_CANDIDATES_ONLY
__tile_global__ void nt_layout_left_to_bkn_kernel(const __nv_bfloat16* __restrict__ src_nt,
                                                  __nv_bfloat16* __restrict__ dst_bkn,
                                                  long long k,
                                                  long long n,
                                                  long long total) {
    src_nt = ct::assume_aligned(src_nt, 16_ic);
    dst_bkn = ct::assume_aligned(dst_bkn, 16_ic);
    I64InitTile idx = (long long)ct::bid().x * kInitTile + ct::iota<I64InitTile>();
    auto in_bounds = idx < total;
    auto row = idx / n;
    auto col = idx - row * n;
    ct::store_masked(dst_bkn + idx, ct::load_masked(src_nt + row + col * k, in_bounds),
                     in_bounds);
}
#endif

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

template <int TM,
          int THidden,
          int GeluMode = kGeluErf,
          bool UseHiddenBias = true,
          bool UseOutBias = true,
          int TK = kTileK>
__tile_global__ void ffn12_fused_split2_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    constexpr int OutHalf = kOut / 2;
    static_assert(kIn % TK == 0);
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
        ct::shape<TM, TK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<kIn, kHidden>{}, ct::layout_left{}},
        ct::shape<TK, THidden>{}
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
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / TK})) {
            hidden_acc = ct::mma(a_view.load(tile_m, kk),
                                 w1_view.load(kk, hidden_tile),
                                 hidden_acc);
        }
        auto hidden_cols =
            static_cast<long long>(hidden_tile) * THidden + (hidden_local % THidden);
        auto hidden_value = bf16_round(hidden_acc);
        if constexpr (UseHiddenBias) {
            auto hidden_bias = ct::element_cast<float>(ct::load(b1 + hidden_cols));
            hidden_value = hidden_value + hidden_bias;
        } else {
            (void)hidden_cols;
        }
        hidden_value = gelu_selected<GeluMode>(hidden_value);
        auto hidden_bf16 = ct::element_cast<__nv_bfloat16>(hidden_value);
        out_acc0 = ct::mma(hidden_bf16, w2_view.load(hidden_tile, 0), out_acc0);
        out_acc1 = ct::mma(hidden_bf16, w2_view.load(hidden_tile, 1), out_acc1);
    }

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_cols = out_local % OutHalf;
    auto value0 = bf16_round(out_acc0);
    auto value1 = bf16_round(out_acc1);
    if constexpr (UseOutBias) {
        auto out_bias0 = ct::element_cast<float>(ct::load(b2 + out_cols));
        auto out_bias1 = ct::element_cast<float>(ct::load(b2 + OutHalf + out_cols));
        value0 = value0 + out_bias0;
        value1 = value1 + out_bias1;
    }
    out_view.store(ct::element_cast<__nv_bfloat16>(value0), tile_m, 0);
    out_view.store(ct::element_cast<__nv_bfloat16>(value1), tile_m, 1);
}

template <int TM,
          int GeluMode = kGeluErf,
          int TK = kTileK,
          int THidden = 32,
	          bool W2LatencyHint = false,
	          bool GroupOutputOrder = false,
	          bool UseHiddenBias = true,
	          bool UseOutBias = true,
	          bool W1LatencyHint = false,
          bool RoundHiddenAcc = true,
	          bool RoundOutAcc = RoundHiddenAcc,
	          typename IndexElement = long long,
	          bool StagedHiddenEpilogue = false,
	          bool W2TempLoads = false,
	          bool W1BatchedMMA = false,
	          bool W2BatchedMMA = false,
	          int MemoryLatency = 8>
static __tile__ void ffn12_fused_split2_pairh32_bf16_body(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    constexpr int OutHalf = kOut / 2;
    static_assert(kIn % TK == 0);
    static_assert(kHidden % (2 * THidden) == 0);
    using HiddenAccTile = ct::tile<float, ct::shape<TM, THidden>>;
    using OutAccTile = ct::tile<float, ct::shape<TM, OutHalf>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using W1Tile = ct::tile<__nv_bfloat16, ct::shape<TK, THidden>>;
    using W2Tile = ct::tile<__nv_bfloat16, ct::shape<THidden, OutHalf>>;
    using HiddenPairAccTile = ct::tile<float, ct::shape<2, TM, THidden>>;
    using OutPairAccTile = ct::tile<float, ct::shape<2, TM, OutHalf>>;
    using W1PairTile = ct::tile<__nv_bfloat16, ct::shape<2, TK, THidden>>;
    using W2PairTile = ct::tile<__nv_bfloat16, ct::shape<2, THidden, OutHalf>>;
    using IndexHiddenTile = ct::tile<IndexElement, ct::shape<TM, THidden>>;
    using IndexOutTile = ct::tile<IndexElement, ct::shape<TM, OutHalf>>;
    static_assert(!W2BatchedMMA || (!W2LatencyHint && !GroupOutputOrder &&
                                    !StagedHiddenEpilogue && !W2TempLoads));

    a = ct::assume_aligned(a, 16_ic);
    w1_nt = ct::assume_aligned(w1_nt, 16_ic);
    b1 = ct::assume_aligned(b1, 16_ic);
    w2_nt = ct::assume_aligned(w2_nt, 16_ic);
    b2 = ct::assume_aligned(b2, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<kM, kIn>{}},
        ct::shape<TM, TK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<kIn, kHidden>{}, ct::layout_left{}},
        ct::shape<TK, THidden>{}
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
    auto out_pair_acc = ct::full<OutPairAccTile>(0.0f);
    IndexHiddenTile hidden_local = ct::iota<IndexHiddenTile>();
    for (auto hidden_pair : ct::irange(std::size_t{0},
                                       std::size_t{kHidden / (2 * THidden)})) {
        auto hidden_tile0 = hidden_pair * 2;
        auto hidden_tile1 = hidden_tile0 + 1;
        auto hidden_acc0 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc1 = ct::full<HiddenAccTile>(0.0f);
        if constexpr (W1BatchedMMA) {
            static_assert(!W1LatencyHint);
            auto hidden_pair_acc = ct::full<HiddenPairAccTile>(0.0f);
            for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / TK})) {
                auto a_tile = a_view.load(tile_m, kk);
                auto a_pair = ct::reshape(a_tile, ct::shape<1, TM, TK>{});
                W1PairTile w1_pair = ct::reshape(
                    ct::cat<0>(w1_view.load(kk, hidden_tile0),
                               w1_view.load(kk, hidden_tile1)),
                    ct::shape<2, TK, THidden>{});
                hidden_pair_acc = ct::mma(a_pair, w1_pair, hidden_pair_acc);
            }
            hidden_acc0 = ct::reshape(
                ct::extract(hidden_pair_acc, ct::shape<1, TM, THidden>{}, 0, 0, 0),
                ct::shape<TM, THidden>{});
            hidden_acc1 = ct::reshape(
                ct::extract(hidden_pair_acc, ct::shape<1, TM, THidden>{}, 1, 0, 0),
                ct::shape<TM, THidden>{});
        } else {
            for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / TK})) {
                if constexpr (W1LatencyHint) {
                    ATile a_tile;
                    W1Tile w1_0;
                    W1Tile w1_1;
                    [[cutile::hint(0, latency=MemoryLatency)]]
                    a_tile = a_view.load(tile_m, kk);
                    [[cutile::hint(0, latency=MemoryLatency)]]
                    w1_0 = w1_view.load(kk, hidden_tile0);
                    [[cutile::hint(0, latency=MemoryLatency)]]
                    w1_1 = w1_view.load(kk, hidden_tile1);
                    hidden_acc0 = ct::mma(a_tile, w1_0, hidden_acc0);
                    hidden_acc1 = ct::mma(a_tile, w1_1, hidden_acc1);
                } else {
                    auto a_tile = a_view.load(tile_m, kk);
                    hidden_acc0 = ct::mma(a_tile, w1_view.load(kk, hidden_tile0), hidden_acc0);
                    hidden_acc1 = ct::mma(a_tile, w1_view.load(kk, hidden_tile1), hidden_acc1);
                }
            }
        }

        auto hidden_cols0 =
            static_cast<IndexElement>(hidden_tile0) * THidden + (hidden_local % THidden);
        auto hidden_cols1 =
            static_cast<IndexElement>(hidden_tile1) * THidden + (hidden_local % THidden);
        if constexpr (StagedHiddenEpilogue) {
            static_assert(!GroupOutputOrder);
            auto hidden_value0 = bf16_round_if<RoundHiddenAcc>(hidden_acc0);
            if constexpr (UseHiddenBias) {
                auto hidden_bias0 = ct::element_cast<float>(ct::load(b1 + hidden_cols0));
                hidden_value0 = hidden_value0 + hidden_bias0;
            } else {
                (void)hidden_cols0;
            }
            hidden_value0 = gelu_selected<GeluMode>(hidden_value0);
            auto hidden_bf16_0 = ct::element_cast<__nv_bfloat16>(hidden_value0);
            if constexpr (W2LatencyHint || W2TempLoads) {
                W2Tile w2_00;
                W2Tile w2_01;
                if constexpr (W2LatencyHint) {
                    [[cutile::hint(0, latency=MemoryLatency)]]
                    w2_00 = w2_view.load(hidden_tile0, 0);
                    [[cutile::hint(0, latency=MemoryLatency)]]
                    w2_01 = w2_view.load(hidden_tile0, 1);
                } else {
                    w2_00 = w2_view.load(hidden_tile0, 0);
                    w2_01 = w2_view.load(hidden_tile0, 1);
                }
                out_acc0 = ct::mma(hidden_bf16_0, w2_00, out_acc0);
                out_acc1 = ct::mma(hidden_bf16_0, w2_01, out_acc1);
            } else {
                out_acc0 = ct::mma(hidden_bf16_0, w2_view.load(hidden_tile0, 0), out_acc0);
                out_acc1 = ct::mma(hidden_bf16_0, w2_view.load(hidden_tile0, 1), out_acc1);
            }

            auto hidden_value1 = bf16_round_if<RoundHiddenAcc>(hidden_acc1);
            if constexpr (UseHiddenBias) {
                auto hidden_bias1 = ct::element_cast<float>(ct::load(b1 + hidden_cols1));
                hidden_value1 = hidden_value1 + hidden_bias1;
            } else {
                (void)hidden_cols1;
            }
            hidden_value1 = gelu_selected<GeluMode>(hidden_value1);
            auto hidden_bf16_1 = ct::element_cast<__nv_bfloat16>(hidden_value1);
            if constexpr (W2LatencyHint || W2TempLoads) {
                W2Tile w2_10;
                W2Tile w2_11;
                if constexpr (W2LatencyHint) {
                    [[cutile::hint(0, latency=MemoryLatency)]]
                    w2_10 = w2_view.load(hidden_tile1, 0);
                    [[cutile::hint(0, latency=MemoryLatency)]]
                    w2_11 = w2_view.load(hidden_tile1, 1);
                } else {
                    w2_10 = w2_view.load(hidden_tile1, 0);
                    w2_11 = w2_view.load(hidden_tile1, 1);
                }
                out_acc0 = ct::mma(hidden_bf16_1, w2_10, out_acc0);
                out_acc1 = ct::mma(hidden_bf16_1, w2_11, out_acc1);
            } else {
                out_acc0 = ct::mma(hidden_bf16_1, w2_view.load(hidden_tile1, 0), out_acc0);
                out_acc1 = ct::mma(hidden_bf16_1, w2_view.load(hidden_tile1, 1), out_acc1);
            }
        } else {
            auto hidden_value0 = bf16_round_if<RoundHiddenAcc>(hidden_acc0);
            auto hidden_value1 = bf16_round_if<RoundHiddenAcc>(hidden_acc1);
            if constexpr (UseHiddenBias) {
                auto hidden_bias0 = ct::element_cast<float>(ct::load(b1 + hidden_cols0));
                auto hidden_bias1 = ct::element_cast<float>(ct::load(b1 + hidden_cols1));
                hidden_value0 = hidden_value0 + hidden_bias0;
                hidden_value1 = hidden_value1 + hidden_bias1;
            } else {
                (void)hidden_cols0;
                (void)hidden_cols1;
            }
            hidden_value0 = gelu_selected<GeluMode>(hidden_value0);
            hidden_value1 = gelu_selected<GeluMode>(hidden_value1);
            auto hidden_bf16_0 = ct::element_cast<__nv_bfloat16>(hidden_value0);
            auto hidden_bf16_1 = ct::element_cast<__nv_bfloat16>(hidden_value1);
            if constexpr (W2BatchedMMA) {
                auto hidden_pair0 = ct::reshape(hidden_bf16_0,
                                                ct::shape<1, TM, THidden>{});
                auto hidden_pair1 = ct::reshape(hidden_bf16_1,
                                                ct::shape<1, TM, THidden>{});
                W2PairTile w2_pair0 = ct::reshape(
                    ct::cat<0>(w2_view.load(hidden_tile0, 0),
                               w2_view.load(hidden_tile0, 1)),
                    ct::shape<2, THidden, OutHalf>{});
                W2PairTile w2_pair1 = ct::reshape(
                    ct::cat<0>(w2_view.load(hidden_tile1, 0),
                               w2_view.load(hidden_tile1, 1)),
                    ct::shape<2, THidden, OutHalf>{});
                out_pair_acc = ct::mma(hidden_pair0, w2_pair0, out_pair_acc);
                out_pair_acc = ct::mma(hidden_pair1, w2_pair1, out_pair_acc);
            } else if constexpr (W2LatencyHint || W2TempLoads) {
                W2Tile w2_00;
                W2Tile w2_01;
                W2Tile w2_10;
                W2Tile w2_11;
                if constexpr (W2LatencyHint) {
                    [[cutile::hint(0, latency=MemoryLatency)]]
                    w2_00 = w2_view.load(hidden_tile0, 0);
                    [[cutile::hint(0, latency=MemoryLatency)]]
                    w2_01 = w2_view.load(hidden_tile0, 1);
                    [[cutile::hint(0, latency=MemoryLatency)]]
                    w2_10 = w2_view.load(hidden_tile1, 0);
                    [[cutile::hint(0, latency=MemoryLatency)]]
                    w2_11 = w2_view.load(hidden_tile1, 1);
                } else {
                    w2_00 = w2_view.load(hidden_tile0, 0);
                    w2_01 = w2_view.load(hidden_tile0, 1);
                    w2_10 = w2_view.load(hidden_tile1, 0);
                    w2_11 = w2_view.load(hidden_tile1, 1);
                }
                out_acc0 = ct::mma(hidden_bf16_0, w2_00, out_acc0);
                out_acc1 = ct::mma(hidden_bf16_0, w2_01, out_acc1);
                out_acc0 = ct::mma(hidden_bf16_1, w2_10, out_acc0);
                out_acc1 = ct::mma(hidden_bf16_1, w2_11, out_acc1);
            } else if constexpr (GroupOutputOrder) {
                out_acc0 = ct::mma(hidden_bf16_0, w2_view.load(hidden_tile0, 0), out_acc0);
                out_acc0 = ct::mma(hidden_bf16_1, w2_view.load(hidden_tile1, 0), out_acc0);
                out_acc1 = ct::mma(hidden_bf16_0, w2_view.load(hidden_tile0, 1), out_acc1);
                out_acc1 = ct::mma(hidden_bf16_1, w2_view.load(hidden_tile1, 1), out_acc1);
            } else {
                out_acc0 = ct::mma(hidden_bf16_0, w2_view.load(hidden_tile0, 0), out_acc0);
                out_acc1 = ct::mma(hidden_bf16_0, w2_view.load(hidden_tile0, 1), out_acc1);
                out_acc0 = ct::mma(hidden_bf16_1, w2_view.load(hidden_tile1, 0), out_acc0);
                out_acc1 = ct::mma(hidden_bf16_1, w2_view.load(hidden_tile1, 1), out_acc1);
            }
        }
    }

    if constexpr (W2BatchedMMA) {
        out_acc0 = ct::reshape(
            ct::extract(out_pair_acc, ct::shape<1, TM, OutHalf>{}, 0, 0, 0),
            ct::shape<TM, OutHalf>{});
        out_acc1 = ct::reshape(
            ct::extract(out_pair_acc, ct::shape<1, TM, OutHalf>{}, 1, 0, 0),
            ct::shape<TM, OutHalf>{});
    }
    IndexOutTile out_local = ct::iota<IndexOutTile>();
    auto out_cols = out_local % OutHalf;
    auto value0 = bf16_round_if<RoundOutAcc>(out_acc0);
    auto value1 = bf16_round_if<RoundOutAcc>(out_acc1);
    if constexpr (UseOutBias) {
        auto out_bias0 = ct::element_cast<float>(ct::load(b2 + out_cols));
        auto out_bias1 = ct::element_cast<float>(ct::load(b2 + OutHalf + out_cols));
        value0 = value0 + out_bias0;
        value1 = value1 + out_bias1;
    } else {
        (void)out_cols;
    }
    out_view.store(ct::element_cast<__nv_bfloat16>(value0), tile_m, 0);
    out_view.store(ct::element_cast<__nv_bfloat16>(value1), tile_m, 1);
}

template <int TM,
          int GeluMode = kGeluErf,
          int TK = kTileK,
          int THidden = 32,
	          bool W2LatencyHint = false,
	          bool GroupOutputOrder = false,
	          bool UseHiddenBias = true,
	          bool UseOutBias = true,
	          bool W1LatencyHint = false,
          bool RoundHiddenAcc = true,
	          bool RoundOutAcc = RoundHiddenAcc,
	          typename IndexElement = long long,
	          bool StagedHiddenEpilogue = false,
	          bool W2TempLoads = false,
	          bool W1BatchedMMA = false,
	          bool W2BatchedMMA = false,
	          int MemoryLatency = 8>
__tile_global__ void ffn12_fused_split2_pairh32_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    ffn12_fused_split2_pairh32_bf16_body
        <TM, GeluMode, TK, THidden, W2LatencyHint, GroupOutputOrder,
	         UseHiddenBias, UseOutBias, W1LatencyHint, RoundHiddenAcc, RoundOutAcc,
	         IndexElement, StagedHiddenEpilogue, W2TempLoads, W1BatchedMMA,
	         W2BatchedMMA, MemoryLatency>(
	        a, w1_nt, b1, w2_nt, b2, out);
}

template <int TM,
          int GeluMode = kGeluErf,
          int TK = kTileK,
          int THidden = 32,
	          bool W2LatencyHint = false,
	          bool GroupOutputOrder = false,
	          bool UseHiddenBias = true,
	          bool UseOutBias = true,
	          bool W1LatencyHint = false,
          bool RoundHiddenAcc = true,
	          bool RoundOutAcc = RoundHiddenAcc,
	          typename IndexElement = long long,
	          bool StagedHiddenEpilogue = false,
	          bool W2TempLoads = false,
	          bool W1BatchedMMA = false,
	          bool W2BatchedMMA = false,
	          int MemoryLatency = 8>
[[cutile::hint(860, occupancy=2)]]
__tile_global__ void ffn12_fused_split2_pairh32_occ2_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    ffn12_fused_split2_pairh32_bf16_body
        <TM, GeluMode, TK, THidden, W2LatencyHint, GroupOutputOrder,
	         UseHiddenBias, UseOutBias, W1LatencyHint, RoundHiddenAcc, RoundOutAcc,
	         IndexElement, StagedHiddenEpilogue, W2TempLoads, W1BatchedMMA,
	         W2BatchedMMA, MemoryLatency>(
	        a, w1_nt, b1, w2_nt, b2, out);
}

#ifdef CUDASEP_FFN12_CANDIDATES_ONLY
template <int TM, int GeluMode = kGeluErfPoly9L30, int TK = 64, int THidden = 32>
__tile_global__ void ffn12_fused_split2_pairh32_bkn_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_bkn,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_bkn,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    constexpr int OutHalf = kOut / 2;
    static_assert(kIn % TK == 0);
    static_assert(kHidden % (2 * THidden) == 0);
    using HiddenAccTile = ct::tile<float, ct::shape<TM, THidden>>;
    using OutAccTile = ct::tile<float, ct::shape<TM, OutHalf>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using I64HiddenTile = ct::tile<long long, ct::shape<TM, THidden>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, OutHalf>>;

    a = ct::assume_aligned(a, 16_ic);
    w1_bkn = ct::assume_aligned(w1_bkn, 16_ic);
    b1 = ct::assume_aligned(b1, 16_ic);
    w2_bkn = ct::assume_aligned(w2_bkn, 16_ic);
    b2 = ct::assume_aligned(b2, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<kM, kIn>{}},
        ct::shape<TM, TK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_bkn, ct::shape<kIn, kHidden>{}},
        ct::shape<TK, THidden>{}
    };
    auto w2_view = ct::partition_view{
        ct::tensor_span{w2_bkn, ct::shape<kHidden, kOut>{}},
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
    for (auto hidden_pair : ct::irange(std::size_t{0},
                                       std::size_t{kHidden / (2 * THidden)})) {
        auto hidden_tile0 = hidden_pair * 2;
        auto hidden_tile1 = hidden_tile0 + 1;
        auto hidden_acc0 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc1 = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / TK})) {
            ATile a_tile = a_view.load(tile_m, kk);
            hidden_acc0 = ct::mma(a_tile, w1_view.load(kk, hidden_tile0), hidden_acc0);
            hidden_acc1 = ct::mma(a_tile, w1_view.load(kk, hidden_tile1), hidden_acc1);
        }

        auto hidden_cols0 =
            static_cast<long long>(hidden_tile0) * THidden + (hidden_local % THidden);
        auto hidden_cols1 =
            static_cast<long long>(hidden_tile1) * THidden + (hidden_local % THidden);
        auto hidden_bias0 = ct::element_cast<float>(ct::load(b1 + hidden_cols0));
        auto hidden_bias1 = ct::element_cast<float>(ct::load(b1 + hidden_cols1));
        auto hidden_value0 = gelu_selected<GeluMode>(bf16_round(hidden_acc0) + hidden_bias0);
        auto hidden_value1 = gelu_selected<GeluMode>(bf16_round(hidden_acc1) + hidden_bias1);
        auto hidden_bf16_0 = ct::element_cast<__nv_bfloat16>(hidden_value0);
        auto hidden_bf16_1 = ct::element_cast<__nv_bfloat16>(hidden_value1);
        out_acc0 = ct::mma(hidden_bf16_0, w2_view.load(hidden_tile0, 0), out_acc0);
        out_acc1 = ct::mma(hidden_bf16_0, w2_view.load(hidden_tile0, 1), out_acc1);
        out_acc0 = ct::mma(hidden_bf16_1, w2_view.load(hidden_tile1, 0), out_acc0);
        out_acc1 = ct::mma(hidden_bf16_1, w2_view.load(hidden_tile1, 1), out_acc1);
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

template <int TM, int GeluMode = kGeluErfPoly9L30, int TK = 64, int THidden = 32>
__tile_global__ void ffn12_fused_split2_pairh32_w2splitspan_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    constexpr int OutHalf = kOut / 2;
    static_assert(kIn % TK == 0);
    static_assert(kHidden % (2 * THidden) == 0);
    using HiddenAccTile = ct::tile<float, ct::shape<TM, THidden>>;
    using OutAccTile = ct::tile<float, ct::shape<TM, OutHalf>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using I64HiddenTile = ct::tile<long long, ct::shape<TM, THidden>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, OutHalf>>;

    a = ct::assume_aligned(a, 16_ic);
    w1_nt = ct::assume_aligned(w1_nt, 16_ic);
    b1 = ct::assume_aligned(b1, 16_ic);
    w2_nt = ct::assume_aligned(w2_nt, 16_ic);
    auto w2_nt_1 = ct::assume_aligned(w2_nt + static_cast<long long>(OutHalf) * kHidden,
                                      16_ic);
    b2 = ct::assume_aligned(b2, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<kM, kIn>{}},
        ct::shape<TM, TK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<kIn, kHidden>{}, ct::layout_left{}},
        ct::shape<TK, THidden>{}
    };
    auto w2_view0 = ct::partition_view{
        ct::tensor_span{w2_nt, ct::shape<kHidden, OutHalf>{}, ct::layout_left{}},
        ct::shape<THidden, OutHalf>{}
    };
    auto w2_view1 = ct::partition_view{
        ct::tensor_span{w2_nt_1, ct::shape<kHidden, OutHalf>{}, ct::layout_left{}},
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
    for (auto hidden_pair : ct::irange(std::size_t{0},
                                       std::size_t{kHidden / (2 * THidden)})) {
        auto hidden_tile0 = hidden_pair * 2;
        auto hidden_tile1 = hidden_tile0 + 1;
        auto hidden_acc0 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc1 = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / TK})) {
            ATile a_tile = a_view.load(tile_m, kk);
            hidden_acc0 = ct::mma(a_tile, w1_view.load(kk, hidden_tile0), hidden_acc0);
            hidden_acc1 = ct::mma(a_tile, w1_view.load(kk, hidden_tile1), hidden_acc1);
        }

        auto hidden_cols0 =
            static_cast<long long>(hidden_tile0) * THidden + (hidden_local % THidden);
        auto hidden_cols1 =
            static_cast<long long>(hidden_tile1) * THidden + (hidden_local % THidden);
        auto hidden_bias0 = ct::element_cast<float>(ct::load(b1 + hidden_cols0));
        auto hidden_bias1 = ct::element_cast<float>(ct::load(b1 + hidden_cols1));
        auto hidden_value0 = gelu_selected<GeluMode>(bf16_round(hidden_acc0) + hidden_bias0);
        auto hidden_value1 = gelu_selected<GeluMode>(bf16_round(hidden_acc1) + hidden_bias1);
        auto hidden_bf16_0 = ct::element_cast<__nv_bfloat16>(hidden_value0);
        auto hidden_bf16_1 = ct::element_cast<__nv_bfloat16>(hidden_value1);
        out_acc0 = ct::mma(hidden_bf16_0, w2_view0.load(hidden_tile0, 0), out_acc0);
        out_acc1 = ct::mma(hidden_bf16_0, w2_view1.load(hidden_tile0, 0), out_acc1);
        out_acc0 = ct::mma(hidden_bf16_1, w2_view0.load(hidden_tile1, 0), out_acc0);
        out_acc1 = ct::mma(hidden_bf16_1, w2_view1.load(hidden_tile1, 0), out_acc1);
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

template <int TM, int GeluMode = kGeluErfPoly9L30, int TK = 64, int THidden = 32>
__tile_global__ void ffn12_fused_split2_pairh32_w2manual_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    constexpr int OutHalf = kOut / 2;
    static_assert(kIn % TK == 0);
    static_assert(kHidden % (2 * THidden) == 0);
    using HiddenAccTile = ct::tile<float, ct::shape<TM, THidden>>;
    using OutAccTile = ct::tile<float, ct::shape<TM, OutHalf>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using W2Tile = ct::tile<__nv_bfloat16, ct::shape<THidden, OutHalf>>;
    using I64HiddenTile = ct::tile<long long, ct::shape<TM, THidden>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, OutHalf>>;
    using I64W2Tile = ct::tile<long long, ct::shape<THidden, OutHalf>>;

    a = ct::assume_aligned(a, 16_ic);
    w1_nt = ct::assume_aligned(w1_nt, 16_ic);
    b1 = ct::assume_aligned(b1, 16_ic);
    w2_nt = ct::assume_aligned(w2_nt, 16_ic);
    b2 = ct::assume_aligned(b2, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<kM, kIn>{}},
        ct::shape<TM, TK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<kIn, kHidden>{}, ct::layout_left{}},
        ct::shape<TK, THidden>{}
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
    I64W2Tile w2_local = ct::iota<I64W2Tile>();
    auto w2_row = w2_local / OutHalf;
    auto w2_col = w2_local - w2_row * OutHalf;
    for (auto hidden_pair : ct::irange(std::size_t{0},
                                       std::size_t{kHidden / (2 * THidden)})) {
        auto hidden_tile0 = hidden_pair * 2;
        auto hidden_tile1 = hidden_tile0 + 1;
        auto hidden_acc0 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc1 = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / TK})) {
            ATile a_tile = a_view.load(tile_m, kk);
            hidden_acc0 = ct::mma(a_tile, w1_view.load(kk, hidden_tile0), hidden_acc0);
            hidden_acc1 = ct::mma(a_tile, w1_view.load(kk, hidden_tile1), hidden_acc1);
        }

        auto hidden_cols0 =
            static_cast<long long>(hidden_tile0) * THidden + (hidden_local % THidden);
        auto hidden_cols1 =
            static_cast<long long>(hidden_tile1) * THidden + (hidden_local % THidden);
        auto hidden_bias0 = ct::element_cast<float>(ct::load(b1 + hidden_cols0));
        auto hidden_bias1 = ct::element_cast<float>(ct::load(b1 + hidden_cols1));
        auto hidden_value0 = gelu_selected<GeluMode>(bf16_round(hidden_acc0) + hidden_bias0);
        auto hidden_value1 = gelu_selected<GeluMode>(bf16_round(hidden_acc1) + hidden_bias1);
        auto hidden_bf16_0 = ct::element_cast<__nv_bfloat16>(hidden_value0);
        auto hidden_bf16_1 = ct::element_cast<__nv_bfloat16>(hidden_value1);

        auto hidden_base0 = static_cast<long long>(hidden_tile0) * THidden;
        auto hidden_base1 = static_cast<long long>(hidden_tile1) * THidden;
        W2Tile w2_00 = ct::load(w2_nt + hidden_base0 + w2_row + w2_col * kHidden);
        W2Tile w2_01 =
            ct::load(w2_nt + hidden_base0 + w2_row + (OutHalf + w2_col) * kHidden);
        W2Tile w2_10 = ct::load(w2_nt + hidden_base1 + w2_row + w2_col * kHidden);
        W2Tile w2_11 =
            ct::load(w2_nt + hidden_base1 + w2_row + (OutHalf + w2_col) * kHidden);
        out_acc0 = ct::mma(hidden_bf16_0, w2_00, out_acc0);
        out_acc1 = ct::mma(hidden_bf16_0, w2_01, out_acc1);
        out_acc0 = ct::mma(hidden_bf16_1, w2_10, out_acc0);
        out_acc1 = ct::mma(hidden_bf16_1, w2_11, out_acc1);
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
#endif

template <int Groups,
          int TM = 32,
          int GeluMode = kGeluErfPoly9L30,
          int TK = 64,
          int THidden = 32>
__tile_global__ void ffn12_fused_split2_pairh32_hsplit_partial_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    __nv_bfloat16* __restrict__ partial) {
    constexpr int OutHalf = kOut / 2;
    constexpr int PairsPerGroup = kHidden / (Groups * 2 * THidden);
    static_assert(kIn % TK == 0);
    static_assert(kHidden % (Groups * 2 * THidden) == 0);
    using HiddenAccTile = ct::tile<float, ct::shape<TM, THidden>>;
    using OutAccTile = ct::tile<float, ct::shape<TM, OutHalf>>;
    using I64HiddenTile = ct::tile<long long, ct::shape<TM, THidden>>;

    a = ct::assume_aligned(a, 16_ic);
    w1_nt = ct::assume_aligned(w1_nt, 16_ic);
    b1 = ct::assume_aligned(b1, 16_ic);
    w2_nt = ct::assume_aligned(w2_nt, 16_ic);
    partial = ct::assume_aligned(partial, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<kM, kIn>{}},
        ct::shape<TM, TK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<kIn, kHidden>{}, ct::layout_left{}},
        ct::shape<TK, THidden>{}
    };
    auto w2_view = ct::partition_view{
        ct::tensor_span{w2_nt, ct::shape<kHidden, kOut>{}, ct::layout_left{}},
        ct::shape<THidden, OutHalf>{}
    };

    auto [tile_m, tile_group, tile_z] = ct::bid();
    (void)tile_z;
    auto partial_base =
        partial + static_cast<long long>(tile_group) * static_cast<long long>(kM) * kOut;
    auto partial_view = ct::partition_view{
        ct::tensor_span{partial_base, ct::shape<kM, kOut>{}},
        ct::shape<TM, OutHalf>{}
    };

    auto out_acc0 = ct::full<OutAccTile>(0.0f);
    auto out_acc1 = ct::full<OutAccTile>(0.0f);
    I64HiddenTile hidden_local = ct::iota<I64HiddenTile>();
    for (auto hidden_pair : ct::irange(std::size_t{0}, std::size_t{PairsPerGroup})) {
        auto hidden_tile0 = tile_group * (PairsPerGroup * 2) + hidden_pair * 2;
        auto hidden_tile1 = hidden_tile0 + 1;
        auto hidden_acc0 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc1 = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / TK})) {
            auto a_tile = a_view.load(tile_m, kk);
            hidden_acc0 = ct::mma(a_tile, w1_view.load(kk, hidden_tile0), hidden_acc0);
            hidden_acc1 = ct::mma(a_tile, w1_view.load(kk, hidden_tile1), hidden_acc1);
        }

        auto hidden_cols0 =
            static_cast<long long>(hidden_tile0) * THidden + (hidden_local % THidden);
        auto hidden_cols1 =
            static_cast<long long>(hidden_tile1) * THidden + (hidden_local % THidden);
        auto hidden_bias0 = ct::element_cast<float>(ct::load(b1 + hidden_cols0));
        auto hidden_bias1 = ct::element_cast<float>(ct::load(b1 + hidden_cols1));
        auto hidden_value0 = gelu_selected<GeluMode>(bf16_round(hidden_acc0) + hidden_bias0);
        auto hidden_value1 = gelu_selected<GeluMode>(bf16_round(hidden_acc1) + hidden_bias1);
        auto hidden_bf16_0 = ct::element_cast<__nv_bfloat16>(hidden_value0);
        auto hidden_bf16_1 = ct::element_cast<__nv_bfloat16>(hidden_value1);
        out_acc0 = ct::mma(hidden_bf16_0, w2_view.load(hidden_tile0, 0), out_acc0);
        out_acc1 = ct::mma(hidden_bf16_0, w2_view.load(hidden_tile0, 1), out_acc1);
        out_acc0 = ct::mma(hidden_bf16_1, w2_view.load(hidden_tile1, 0), out_acc0);
        out_acc1 = ct::mma(hidden_bf16_1, w2_view.load(hidden_tile1, 1), out_acc1);
    }

    partial_view.store(ct::element_cast<__nv_bfloat16>(bf16_round(out_acc0)), tile_m, 0);
    partial_view.store(ct::element_cast<__nv_bfloat16>(bf16_round(out_acc1)), tile_m, 1);
}

template <int Groups, int TM = 32>
__tile_global__ void ffn12_hsplit_partial_reduce_bf16_kernel(
    const __nv_bfloat16* __restrict__ partial,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    constexpr int OutHalf = kOut / 2;
    using OutAccTile = ct::tile<float, ct::shape<TM, OutHalf>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, OutHalf>>;

    partial = ct::assume_aligned(partial, 16_ic);
    b2 = ct::assume_aligned(b2, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto out_view = ct::partition_view{
        ct::tensor_span{out, ct::shape<kM, kOut>{}},
        ct::shape<TM, OutHalf>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_n;
    (void)tile_z;

    auto acc0 = ct::full<OutAccTile>(0.0f);
    auto acc1 = ct::full<OutAccTile>(0.0f);
    for (auto group : ct::irange(std::size_t{0}, std::size_t{Groups})) {
        auto partial_base =
            partial + static_cast<long long>(group) * static_cast<long long>(kM) * kOut;
        auto partial_view = ct::partition_view{
            ct::tensor_span{partial_base, ct::shape<kM, kOut>{}},
            ct::shape<TM, OutHalf>{}
        };
        acc0 = acc0 + ct::element_cast<float>(partial_view.load(tile_m, 0));
        acc1 = acc1 + ct::element_cast<float>(partial_view.load(tile_m, 1));
    }

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_cols = out_local % OutHalf;
    auto out_bias0 = ct::element_cast<float>(ct::load(b2 + out_cols));
    auto out_bias1 = ct::element_cast<float>(ct::load(b2 + OutHalf + out_cols));
    auto value0 = bf16_round(acc0) + out_bias0;
    auto value1 = bf16_round(acc1) + out_bias1;
    out_view.store(ct::element_cast<__nv_bfloat16>(value0), tile_m, 0);
    out_view.store(ct::element_cast<__nv_bfloat16>(value1), tile_m, 1);
}

template <int TM,
          int GeluMode = kGeluErf,
          int TK = kTileK,
          int THidden = 32>
__tile_global__ void ffn12_fused_split2_pairh32_quarterout_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    constexpr int OutQuarter = kOut / 4;
    static_assert(kIn % TK == 0);
    static_assert(kHidden % (2 * THidden) == 0);
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
        ct::shape<TM, TK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<kIn, kHidden>{}, ct::layout_left{}},
        ct::shape<TK, THidden>{}
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
    (void)tile_z;

    auto out_acc = ct::full<OutAccTile>(0.0f);
    I64HiddenTile hidden_local = ct::iota<I64HiddenTile>();
    for (auto hidden_pair : ct::irange(std::size_t{0},
                                       std::size_t{kHidden / (2 * THidden)})) {
        auto hidden_tile0 = hidden_pair * 2;
        auto hidden_tile1 = hidden_tile0 + 1;
        auto hidden_acc0 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc1 = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / TK})) {
            auto a_tile = a_view.load(tile_m, kk);
            hidden_acc0 = ct::mma(a_tile, w1_view.load(kk, hidden_tile0), hidden_acc0);
            hidden_acc1 = ct::mma(a_tile, w1_view.load(kk, hidden_tile1), hidden_acc1);
        }

        auto hidden_cols0 =
            static_cast<long long>(hidden_tile0) * THidden + (hidden_local % THidden);
        auto hidden_cols1 =
            static_cast<long long>(hidden_tile1) * THidden + (hidden_local % THidden);
        auto hidden_bias0 = ct::element_cast<float>(ct::load(b1 + hidden_cols0));
        auto hidden_bias1 = ct::element_cast<float>(ct::load(b1 + hidden_cols1));
        auto hidden_value0 = gelu_selected<GeluMode>(bf16_round(hidden_acc0) + hidden_bias0);
        auto hidden_value1 = gelu_selected<GeluMode>(bf16_round(hidden_acc1) + hidden_bias1);
        auto hidden_bf16_0 = ct::element_cast<__nv_bfloat16>(hidden_value0);
        auto hidden_bf16_1 = ct::element_cast<__nv_bfloat16>(hidden_value1);
        out_acc = ct::mma(hidden_bf16_0, w2_view.load(hidden_tile0, tile_n), out_acc);
        out_acc = ct::mma(hidden_bf16_1, w2_view.load(hidden_tile1, tile_n), out_acc);
    }

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_cols = static_cast<long long>(tile_n) * OutQuarter + (out_local % OutQuarter);
    auto out_bias = ct::element_cast<float>(ct::load(b2 + out_cols));
    auto value = bf16_round(out_acc) + out_bias;
    out_view.store(ct::element_cast<__nv_bfloat16>(value), tile_m, tile_n);
}

template <int TM,
          int GeluMode = kGeluErf,
          int TK = kTileK,
          int THidden = 32,
          int MemoryLatency = 2>
__tile_global__ void ffn12_fused_split2_pairh32_quarterout_lat2_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    constexpr int OutQuarter = kOut / 4;
    static_assert(kIn % TK == 0);
    static_assert(kHidden % (2 * THidden) == 0);
    using HiddenAccTile = ct::tile<float, ct::shape<TM, THidden>>;
    using OutAccTile = ct::tile<float, ct::shape<TM, OutQuarter>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using W1Tile = ct::tile<__nv_bfloat16, ct::shape<TK, THidden>>;
    using W2Tile = ct::tile<__nv_bfloat16, ct::shape<THidden, OutQuarter>>;
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
        ct::shape<TM, TK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<kIn, kHidden>{}, ct::layout_left{}},
        ct::shape<TK, THidden>{}
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
    (void)tile_z;

    auto out_acc = ct::full<OutAccTile>(0.0f);
    I64HiddenTile hidden_local = ct::iota<I64HiddenTile>();
    for (auto hidden_pair : ct::irange(std::size_t{0},
                                       std::size_t{kHidden / (2 * THidden)})) {
        auto hidden_tile0 = hidden_pair * 2;
        auto hidden_tile1 = hidden_tile0 + 1;
        auto hidden_acc0 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc1 = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / TK})) {
            ATile a_tile;
            W1Tile w1_0;
            W1Tile w1_1;
            [[cutile::hint(0, latency=MemoryLatency)]]
            a_tile = a_view.load(tile_m, kk);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w1_0 = w1_view.load(kk, hidden_tile0);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w1_1 = w1_view.load(kk, hidden_tile1);
            hidden_acc0 = ct::mma(a_tile, w1_0, hidden_acc0);
            hidden_acc1 = ct::mma(a_tile, w1_1, hidden_acc1);
        }

        auto hidden_cols0 =
            static_cast<long long>(hidden_tile0) * THidden + (hidden_local % THidden);
        auto hidden_cols1 =
            static_cast<long long>(hidden_tile1) * THidden + (hidden_local % THidden);
        auto hidden_bias0 = ct::element_cast<float>(ct::load(b1 + hidden_cols0));
        auto hidden_bias1 = ct::element_cast<float>(ct::load(b1 + hidden_cols1));
        auto hidden_value0 = gelu_selected<GeluMode>(bf16_round(hidden_acc0) + hidden_bias0);
        auto hidden_value1 = gelu_selected<GeluMode>(bf16_round(hidden_acc1) + hidden_bias1);
        auto hidden_bf16_0 = ct::element_cast<__nv_bfloat16>(hidden_value0);
        auto hidden_bf16_1 = ct::element_cast<__nv_bfloat16>(hidden_value1);
        W2Tile w2_0;
        W2Tile w2_1;
        [[cutile::hint(0, latency=MemoryLatency)]]
        w2_0 = w2_view.load(hidden_tile0, tile_n);
        [[cutile::hint(0, latency=MemoryLatency)]]
        w2_1 = w2_view.load(hidden_tile1, tile_n);
        out_acc = ct::mma(hidden_bf16_0, w2_0, out_acc);
        out_acc = ct::mma(hidden_bf16_1, w2_1, out_acc);
    }

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_cols = static_cast<long long>(tile_n) * OutQuarter + (out_local % OutQuarter);
    auto out_bias = ct::element_cast<float>(ct::load(b2 + out_cols));
    auto value = bf16_round(out_acc) + out_bias;
    out_view.store(ct::element_cast<__nv_bfloat16>(value), tile_m, tile_n);
}

template <int TM,
          int GeluMode = kGeluErf,
          int TK = kTileK,
          int THidden = 32>
__tile_global__ void ffn12_fused_split2_pairh32_halfout_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    constexpr int OutHalf = kOut / 2;
    static_assert(kIn % TK == 0);
    static_assert(kHidden % (2 * THidden) == 0);
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
        ct::shape<TM, TK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<kIn, kHidden>{}, ct::layout_left{}},
        ct::shape<TK, THidden>{}
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
    (void)tile_z;

    auto out_acc = ct::full<OutAccTile>(0.0f);
    I64HiddenTile hidden_local = ct::iota<I64HiddenTile>();
    for (auto hidden_pair : ct::irange(std::size_t{0},
                                       std::size_t{kHidden / (2 * THidden)})) {
        auto hidden_tile0 = hidden_pair * 2;
        auto hidden_tile1 = hidden_tile0 + 1;
        auto hidden_acc0 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc1 = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / TK})) {
            auto a_tile = a_view.load(tile_m, kk);
            hidden_acc0 = ct::mma(a_tile, w1_view.load(kk, hidden_tile0), hidden_acc0);
            hidden_acc1 = ct::mma(a_tile, w1_view.load(kk, hidden_tile1), hidden_acc1);
        }

        auto hidden_cols0 =
            static_cast<long long>(hidden_tile0) * THidden + (hidden_local % THidden);
        auto hidden_cols1 =
            static_cast<long long>(hidden_tile1) * THidden + (hidden_local % THidden);
        auto hidden_bias0 = ct::element_cast<float>(ct::load(b1 + hidden_cols0));
        auto hidden_bias1 = ct::element_cast<float>(ct::load(b1 + hidden_cols1));
        auto hidden_value0 = gelu_selected<GeluMode>(bf16_round(hidden_acc0) + hidden_bias0);
        auto hidden_value1 = gelu_selected<GeluMode>(bf16_round(hidden_acc1) + hidden_bias1);
        auto hidden_bf16_0 = ct::element_cast<__nv_bfloat16>(hidden_value0);
        auto hidden_bf16_1 = ct::element_cast<__nv_bfloat16>(hidden_value1);
        out_acc = ct::mma(hidden_bf16_0, w2_view.load(hidden_tile0, tile_n), out_acc);
        out_acc = ct::mma(hidden_bf16_1, w2_view.load(hidden_tile1, tile_n), out_acc);
    }

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_cols = static_cast<long long>(tile_n) * OutHalf + (out_local % OutHalf);
    auto out_bias = ct::element_cast<float>(ct::load(b2 + out_cols));
    auto value = bf16_round(out_acc) + out_bias;
    out_view.store(ct::element_cast<__nv_bfloat16>(value), tile_m, tile_n);
}

#ifdef CUDASEP_FFN12_CANDIDATES_ONLY
template <int TM,
          int GeluMode = kGeluErf,
          int TK = kTileK,
          int THidden = 32>
__tile_global__ void ffn12_fused_split2_pairh32_outseq_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    constexpr int OutHalf = kOut / 2;
    static_assert(kIn % TK == 0);
    static_assert(kHidden % (2 * THidden) == 0);
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
        ct::shape<TM, TK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<kIn, kHidden>{}, ct::layout_left{}},
        ct::shape<TK, THidden>{}
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

    I64HiddenTile hidden_local = ct::iota<I64HiddenTile>();
    I64OutTile out_local = ct::iota<I64OutTile>();
    for (auto out_half : ct::irange(std::size_t{0}, std::size_t{2})) {
        auto out_acc = ct::full<OutAccTile>(0.0f);
        for (auto hidden_pair : ct::irange(std::size_t{0},
                                           std::size_t{kHidden / (2 * THidden)})) {
            auto hidden_tile0 = hidden_pair * 2;
            auto hidden_tile1 = hidden_tile0 + 1;
            auto hidden_acc0 = ct::full<HiddenAccTile>(0.0f);
            auto hidden_acc1 = ct::full<HiddenAccTile>(0.0f);
            for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / TK})) {
                auto a_tile = a_view.load(tile_m, kk);
                hidden_acc0 = ct::mma(a_tile, w1_view.load(kk, hidden_tile0), hidden_acc0);
                hidden_acc1 = ct::mma(a_tile, w1_view.load(kk, hidden_tile1), hidden_acc1);
            }

            auto hidden_cols0 =
                static_cast<long long>(hidden_tile0) * THidden + (hidden_local % THidden);
            auto hidden_cols1 =
                static_cast<long long>(hidden_tile1) * THidden + (hidden_local % THidden);
            auto hidden_bias0 = ct::element_cast<float>(ct::load(b1 + hidden_cols0));
            auto hidden_bias1 = ct::element_cast<float>(ct::load(b1 + hidden_cols1));
            auto hidden_value0 =
                gelu_selected<GeluMode>(bf16_round(hidden_acc0) + hidden_bias0);
            auto hidden_value1 =
                gelu_selected<GeluMode>(bf16_round(hidden_acc1) + hidden_bias1);
            auto hidden_bf16_0 = ct::element_cast<__nv_bfloat16>(hidden_value0);
            auto hidden_bf16_1 = ct::element_cast<__nv_bfloat16>(hidden_value1);
            out_acc = ct::mma(hidden_bf16_0, w2_view.load(hidden_tile0, out_half), out_acc);
            out_acc = ct::mma(hidden_bf16_1, w2_view.load(hidden_tile1, out_half), out_acc);
        }

        auto out_cols = static_cast<long long>(out_half) * OutHalf + (out_local % OutHalf);
        auto out_bias = ct::element_cast<float>(ct::load(b2 + out_cols));
        auto value = bf16_round(out_acc) + out_bias;
        out_view.store(ct::element_cast<__nv_bfloat16>(value), tile_m, out_half);
    }
}
#endif

template <int TM, int GeluMode = kGeluErf, bool FullBF16 = false>
__tile_global__ void ffn12_fused_split2_pairh32_source_style_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    constexpr int THidden = 32;
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
    for (auto hidden_pair : ct::irange(std::size_t{0},
                                       std::size_t{kHidden / (2 * THidden)})) {
        auto hidden_tile0 = hidden_pair * 2;
        auto hidden_tile1 = hidden_tile0 + 1;
        auto hidden_acc0 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc1 = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / kTileK})) {
            auto a_tile = a_view.load(tile_m, kk);
            hidden_acc0 = ct::mma(a_tile, w1_view.load(kk, hidden_tile0), hidden_acc0);
            hidden_acc1 = ct::mma(a_tile, w1_view.load(kk, hidden_tile1), hidden_acc1);
        }

        auto hidden_cols0 =
            static_cast<long long>(hidden_tile0) * THidden + (hidden_local % THidden);
        auto hidden_cols1 =
            static_cast<long long>(hidden_tile1) * THidden + (hidden_local % THidden);
        auto hidden_bias0 = ct::element_cast<float>(ct::load(b1 + hidden_cols0));
        auto hidden_bias1 = ct::element_cast<float>(ct::load(b1 + hidden_cols1));
        auto hidden_value0 = bf16_round(hidden_acc0) + hidden_bias0;
        auto hidden_value1 = bf16_round(hidden_acc1) + hidden_bias1;
        hidden_value0 = bf16_round_if<FullBF16>(hidden_value0);
        hidden_value1 = bf16_round_if<FullBF16>(hidden_value1);
        hidden_value0 = gelu_selected_source_style<GeluMode, FullBF16>(hidden_value0);
        hidden_value1 = gelu_selected_source_style<GeluMode, FullBF16>(hidden_value1);
        auto hidden_bf16_0 = ct::element_cast<__nv_bfloat16>(hidden_value0);
        auto hidden_bf16_1 = ct::element_cast<__nv_bfloat16>(hidden_value1);
        out_acc0 = ct::mma(hidden_bf16_0, w2_view.load(hidden_tile0, 0), out_acc0);
        out_acc1 = ct::mma(hidden_bf16_0, w2_view.load(hidden_tile0, 1), out_acc1);
        out_acc0 = ct::mma(hidden_bf16_1, w2_view.load(hidden_tile1, 0), out_acc0);
        out_acc1 = ct::mma(hidden_bf16_1, w2_view.load(hidden_tile1, 1), out_acc1);
    }

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_cols = out_local % OutHalf;
    auto out_bias0 = ct::element_cast<float>(ct::load(b2 + out_cols));
    auto out_bias1 = ct::element_cast<float>(ct::load(b2 + OutHalf + out_cols));
    auto value0 = bf16_round(out_acc0) + out_bias0;
    auto value1 = bf16_round(out_acc1) + out_bias1;
    value0 = bf16_round_if<FullBF16>(value0);
    value1 = bf16_round_if<FullBF16>(value1);
    out_view.store(ct::element_cast<__nv_bfloat16>(value0), tile_m, 0);
    out_view.store(ct::element_cast<__nv_bfloat16>(value1), tile_m, 1);
}

template <int TM, int GeluMode = kGeluErf, int TK = kTileK>
__tile_global__ void ffn12_fused_split4_pairh32_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    constexpr int THidden = 32;
    constexpr int OutQuarter = kOut / 4;
    static_assert(kIn % TK == 0);
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
        ct::shape<TM, TK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<kIn, kHidden>{}, ct::layout_left{}},
        ct::shape<TK, THidden>{}
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
    for (auto hidden_pair : ct::irange(std::size_t{0},
                                       std::size_t{kHidden / (2 * THidden)})) {
        auto hidden_tile0 = hidden_pair * 2;
        auto hidden_tile1 = hidden_tile0 + 1;
        auto hidden_acc0 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc1 = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / TK})) {
            auto a_tile = a_view.load(tile_m, kk);
            hidden_acc0 = ct::mma(a_tile, w1_view.load(kk, hidden_tile0), hidden_acc0);
            hidden_acc1 = ct::mma(a_tile, w1_view.load(kk, hidden_tile1), hidden_acc1);
        }

        auto hidden_cols0 =
            static_cast<long long>(hidden_tile0) * THidden + (hidden_local % THidden);
        auto hidden_cols1 =
            static_cast<long long>(hidden_tile1) * THidden + (hidden_local % THidden);
        auto hidden_bias0 = ct::element_cast<float>(ct::load(b1 + hidden_cols0));
        auto hidden_bias1 = ct::element_cast<float>(ct::load(b1 + hidden_cols1));
        auto hidden_value0 = gelu_selected<GeluMode>(bf16_round(hidden_acc0) + hidden_bias0);
        auto hidden_value1 = gelu_selected<GeluMode>(bf16_round(hidden_acc1) + hidden_bias1);
        auto hidden_bf16_0 = ct::element_cast<__nv_bfloat16>(hidden_value0);
        auto hidden_bf16_1 = ct::element_cast<__nv_bfloat16>(hidden_value1);
        out_acc0 = ct::mma(hidden_bf16_0, w2_view.load(hidden_tile0, 0), out_acc0);
        out_acc1 = ct::mma(hidden_bf16_0, w2_view.load(hidden_tile0, 1), out_acc1);
        out_acc2 = ct::mma(hidden_bf16_0, w2_view.load(hidden_tile0, 2), out_acc2);
        out_acc3 = ct::mma(hidden_bf16_0, w2_view.load(hidden_tile0, 3), out_acc3);
        out_acc0 = ct::mma(hidden_bf16_1, w2_view.load(hidden_tile1, 0), out_acc0);
        out_acc1 = ct::mma(hidden_bf16_1, w2_view.load(hidden_tile1, 1), out_acc1);
        out_acc2 = ct::mma(hidden_bf16_1, w2_view.load(hidden_tile1, 2), out_acc2);
        out_acc3 = ct::mma(hidden_bf16_1, w2_view.load(hidden_tile1, 3), out_acc3);
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

template <int TM, int GeluMode = kGeluErf>
__tile_global__ void ffn12_fused_split2_quadh32_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    constexpr int THidden = 32;
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
    for (auto hidden_quad : ct::irange(std::size_t{0},
                                       std::size_t{kHidden / (4 * THidden)})) {
        auto hidden_tile0 = hidden_quad * 4;
        auto hidden_tile1 = hidden_tile0 + 1;
        auto hidden_tile2 = hidden_tile0 + 2;
        auto hidden_tile3 = hidden_tile0 + 3;
        auto hidden_acc0 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc1 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc2 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc3 = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / kTileK})) {
            auto a_tile = a_view.load(tile_m, kk);
            hidden_acc0 = ct::mma(a_tile, w1_view.load(kk, hidden_tile0), hidden_acc0);
            hidden_acc1 = ct::mma(a_tile, w1_view.load(kk, hidden_tile1), hidden_acc1);
            hidden_acc2 = ct::mma(a_tile, w1_view.load(kk, hidden_tile2), hidden_acc2);
            hidden_acc3 = ct::mma(a_tile, w1_view.load(kk, hidden_tile3), hidden_acc3);
        }

        auto hidden_cols0 =
            static_cast<long long>(hidden_tile0) * THidden + (hidden_local % THidden);
        auto hidden_cols1 =
            static_cast<long long>(hidden_tile1) * THidden + (hidden_local % THidden);
        auto hidden_cols2 =
            static_cast<long long>(hidden_tile2) * THidden + (hidden_local % THidden);
        auto hidden_cols3 =
            static_cast<long long>(hidden_tile3) * THidden + (hidden_local % THidden);
        auto hidden_bias0 = ct::element_cast<float>(ct::load(b1 + hidden_cols0));
        auto hidden_bias1 = ct::element_cast<float>(ct::load(b1 + hidden_cols1));
        auto hidden_bias2 = ct::element_cast<float>(ct::load(b1 + hidden_cols2));
        auto hidden_bias3 = ct::element_cast<float>(ct::load(b1 + hidden_cols3));
        auto hidden_value0 = gelu_selected<GeluMode>(bf16_round(hidden_acc0) + hidden_bias0);
        auto hidden_value1 = gelu_selected<GeluMode>(bf16_round(hidden_acc1) + hidden_bias1);
        auto hidden_value2 = gelu_selected<GeluMode>(bf16_round(hidden_acc2) + hidden_bias2);
        auto hidden_value3 = gelu_selected<GeluMode>(bf16_round(hidden_acc3) + hidden_bias3);
        auto hidden_bf16_0 = ct::element_cast<__nv_bfloat16>(hidden_value0);
        auto hidden_bf16_1 = ct::element_cast<__nv_bfloat16>(hidden_value1);
        auto hidden_bf16_2 = ct::element_cast<__nv_bfloat16>(hidden_value2);
        auto hidden_bf16_3 = ct::element_cast<__nv_bfloat16>(hidden_value3);
        out_acc0 = ct::mma(hidden_bf16_0, w2_view.load(hidden_tile0, 0), out_acc0);
        out_acc1 = ct::mma(hidden_bf16_0, w2_view.load(hidden_tile0, 1), out_acc1);
        out_acc0 = ct::mma(hidden_bf16_1, w2_view.load(hidden_tile1, 0), out_acc0);
        out_acc1 = ct::mma(hidden_bf16_1, w2_view.load(hidden_tile1, 1), out_acc1);
        out_acc0 = ct::mma(hidden_bf16_2, w2_view.load(hidden_tile2, 0), out_acc0);
        out_acc1 = ct::mma(hidden_bf16_2, w2_view.load(hidden_tile2, 1), out_acc1);
        out_acc0 = ct::mma(hidden_bf16_3, w2_view.load(hidden_tile3, 0), out_acc0);
        out_acc1 = ct::mma(hidden_bf16_3, w2_view.load(hidden_tile3, 1), out_acc1);
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

#ifdef CUDASEP_FFN12_CANDIDATES_ONLY
void init_bkn_from_nt(const __nv_bfloat16* src_nt,
                      __nv_bfloat16* dst_bkn,
                      int k,
                      int n) {
    long long total = static_cast<long long>(k) * n;
    nt_layout_left_to_bkn_kernel<<<ceildiv(static_cast<int>(total), kInitTile), 1>>>(
        src_nt, dst_bkn, k, n, total);
    CUDA_CHECK(cudaGetLastError());
}
#endif

[[maybe_unused]] void launch_separate(const __nv_bfloat16* d_a,
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

template <int TM,
          int THidden = kTileHidden,
          int GeluMode = kGeluErf,
          bool UseHiddenBias = true,
          bool UseOutBias = true,
          int TK = kTileK>
void launch_fused_split2(const __nv_bfloat16* d_a,
                         const __nv_bfloat16* d_w1,
                         const __nv_bfloat16* d_b1,
                         const __nv_bfloat16* d_w2,
                         const __nv_bfloat16* d_b2,
                         __nv_bfloat16* d_out) {
    dim3 grid(kM / TM, 1, 1);
    ffn12_fused_split2_bf16_kernel<TM, THidden, GeluMode, UseHiddenBias, UseOutBias, TK>
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

template <int TM,
          int GeluMode = kGeluErf,
          int TK = kTileK,
          int THidden = 32,
	          bool W2LatencyHint = false,
	          bool GroupOutputOrder = false,
	          bool UseHiddenBias = true,
	          bool UseOutBias = true,
	          bool W1LatencyHint = false,
          bool RoundHiddenAcc = true,
	          bool RoundOutAcc = RoundHiddenAcc,
	          typename IndexElement = long long,
	          bool StagedHiddenEpilogue = false,
	          bool W2TempLoads = false,
	          bool W1BatchedMMA = false,
	          bool W2BatchedMMA = false,
	          int MemoryLatency = 8>
void launch_fused_split2_pairh32(const __nv_bfloat16* d_a,
                                 const __nv_bfloat16* d_w1,
                                 const __nv_bfloat16* d_b1,
                                 const __nv_bfloat16* d_w2,
                                 const __nv_bfloat16* d_b2,
                                 __nv_bfloat16* d_out) {
    dim3 grid(kM / TM, 1, 1);
    ffn12_fused_split2_pairh32_bf16_kernel
        <TM, GeluMode, TK, THidden, W2LatencyHint, GroupOutputOrder,
	         UseHiddenBias, UseOutBias, W1LatencyHint, RoundHiddenAcc, RoundOutAcc,
	         IndexElement, StagedHiddenEpilogue, W2TempLoads, W1BatchedMMA,
	         W2BatchedMMA, MemoryLatency>
	        <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
}

template <int TM,
          int GeluMode = kGeluErf,
          int TK = kTileK,
          int THidden = 32,
	          bool W2LatencyHint = false,
	          bool GroupOutputOrder = false,
	          bool UseHiddenBias = true,
	          bool UseOutBias = true,
	          bool W1LatencyHint = false,
          bool RoundHiddenAcc = true,
	          bool RoundOutAcc = RoundHiddenAcc,
	          typename IndexElement = long long,
	          bool StagedHiddenEpilogue = false,
	          bool W2TempLoads = false,
	          bool W1BatchedMMA = false,
	          bool W2BatchedMMA = false,
	          int MemoryLatency = 8>
void launch_fused_split2_pairh32_occ2(const __nv_bfloat16* d_a,
                                      const __nv_bfloat16* d_w1,
                                      const __nv_bfloat16* d_b1,
                                      const __nv_bfloat16* d_w2,
                                      const __nv_bfloat16* d_b2,
                                      __nv_bfloat16* d_out) {
    dim3 grid(kM / TM, 1, 1);
    ffn12_fused_split2_pairh32_occ2_bf16_kernel
        <TM, GeluMode, TK, THidden, W2LatencyHint, GroupOutputOrder,
	         UseHiddenBias, UseOutBias, W1LatencyHint, RoundHiddenAcc, RoundOutAcc,
	         IndexElement, StagedHiddenEpilogue, W2TempLoads, W1BatchedMMA,
	         W2BatchedMMA, MemoryLatency>
	        <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
}

#ifdef CUDASEP_FFN12_CANDIDATES_ONLY
template <int TM, int GeluMode = kGeluErfPoly9L30, int TK = 64, int THidden = 32>
void launch_fused_split2_pairh32_bkn(const __nv_bfloat16* d_a,
                                     const __nv_bfloat16* d_w1_bkn,
                                     const __nv_bfloat16* d_b1,
                                     const __nv_bfloat16* d_w2_bkn,
                                     const __nv_bfloat16* d_b2,
                                     __nv_bfloat16* d_out) {
    dim3 grid(kM / TM, 1, 1);
    ffn12_fused_split2_pairh32_bkn_bf16_kernel<TM, GeluMode, TK, THidden>
        <<<grid, 1>>>(d_a, d_w1_bkn, d_b1, d_w2_bkn, d_b2, d_out);
}

template <int TM, int GeluMode = kGeluErfPoly9L30, int TK = 64, int THidden = 32>
void launch_fused_split2_pairh32_w2splitspan(const __nv_bfloat16* d_a,
                                             const __nv_bfloat16* d_w1,
                                             const __nv_bfloat16* d_b1,
                                             const __nv_bfloat16* d_w2,
                                             const __nv_bfloat16* d_b2,
                                             __nv_bfloat16* d_out) {
    dim3 grid(kM / TM, 1, 1);
    ffn12_fused_split2_pairh32_w2splitspan_bf16_kernel<TM, GeluMode, TK, THidden>
        <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
}

template <int TM, int GeluMode = kGeluErfPoly9L30, int TK = 64, int THidden = 32>
void launch_fused_split2_pairh32_w2manual(const __nv_bfloat16* d_a,
                                          const __nv_bfloat16* d_w1,
                                          const __nv_bfloat16* d_b1,
                                          const __nv_bfloat16* d_w2,
                                          const __nv_bfloat16* d_b2,
                                          __nv_bfloat16* d_out) {
    dim3 grid(kM / TM, 1, 1);
    ffn12_fused_split2_pairh32_w2manual_bf16_kernel<TM, GeluMode, TK, THidden>
        <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
}
#endif

template <int Groups>
void launch_fused_split2_pairh32_hsplit_partial(const __nv_bfloat16* d_a,
                                                const __nv_bfloat16* d_w1,
                                                const __nv_bfloat16* d_b1,
                                                const __nv_bfloat16* d_w2,
                                                const __nv_bfloat16* d_b2,
                                                __nv_bfloat16* d_partial,
                                                __nv_bfloat16* d_out) {
    dim3 partial_grid(kM / 32, Groups, 1);
    ffn12_fused_split2_pairh32_hsplit_partial_bf16_kernel<Groups>
        <<<partial_grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_partial);
    dim3 reduce_grid(kM / 32, 1, 1);
    ffn12_hsplit_partial_reduce_bf16_kernel<Groups>
        <<<reduce_grid, 1>>>(d_partial, d_b2, d_out);
}

template <int TM,
          int GeluMode = kGeluErf,
          int TK = kTileK,
          int THidden = 32>
void launch_fused_split2_pairh32_halfout(const __nv_bfloat16* d_a,
                                         const __nv_bfloat16* d_w1,
                                         const __nv_bfloat16* d_b1,
                                         const __nv_bfloat16* d_w2,
                                         const __nv_bfloat16* d_b2,
                                         __nv_bfloat16* d_out) {
    dim3 grid(kM / TM, 2, 1);
    ffn12_fused_split2_pairh32_halfout_bf16_kernel<TM, GeluMode, TK, THidden>
        <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
}

#ifdef CUDASEP_FFN12_CANDIDATES_ONLY
template <int TM,
          int GeluMode = kGeluErf,
          int TK = kTileK,
          int THidden = 32>
void launch_fused_split2_pairh32_quarterout(const __nv_bfloat16* d_a,
                                            const __nv_bfloat16* d_w1,
                                            const __nv_bfloat16* d_b1,
                                            const __nv_bfloat16* d_w2,
                                            const __nv_bfloat16* d_b2,
                                            __nv_bfloat16* d_out) {
    dim3 grid(kM / TM, 4, 1);
    ffn12_fused_split2_pairh32_quarterout_bf16_kernel<TM, GeluMode, TK, THidden>
        <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
}

template <int TM,
          int GeluMode = kGeluErf,
          int TK = kTileK,
          int THidden = 32>
void launch_fused_split2_pairh32_quarterout_lat2(const __nv_bfloat16* d_a,
                                                 const __nv_bfloat16* d_w1,
                                                 const __nv_bfloat16* d_b1,
                                                 const __nv_bfloat16* d_w2,
                                                 const __nv_bfloat16* d_b2,
                                                 __nv_bfloat16* d_out) {
    dim3 grid(kM / TM, 4, 1);
    ffn12_fused_split2_pairh32_quarterout_lat2_bf16_kernel<TM, GeluMode, TK, THidden>
        <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
}

template <int TM,
          int GeluMode = kGeluErf,
          int TK = kTileK,
          int THidden = 32>
void launch_fused_split2_pairh32_outseq(const __nv_bfloat16* d_a,
                                        const __nv_bfloat16* d_w1,
                                        const __nv_bfloat16* d_b1,
                                        const __nv_bfloat16* d_w2,
                                        const __nv_bfloat16* d_b2,
                                        __nv_bfloat16* d_out) {
    dim3 grid(kM / TM, 1, 1);
    ffn12_fused_split2_pairh32_outseq_bf16_kernel<TM, GeluMode, TK, THidden>
        <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
}
#endif

template <int TM, int GeluMode = kGeluErf, bool FullBF16 = false>
void launch_fused_split2_pairh32_source_style(const __nv_bfloat16* d_a,
                                              const __nv_bfloat16* d_w1,
                                              const __nv_bfloat16* d_b1,
                                              const __nv_bfloat16* d_w2,
                                              const __nv_bfloat16* d_b2,
                                              __nv_bfloat16* d_out) {
    dim3 grid(kM / TM, 1, 1);
    ffn12_fused_split2_pairh32_source_style_bf16_kernel<TM, GeluMode, FullBF16>
        <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
}

template <int TM, int GeluMode = kGeluErf, int TK = kTileK>
void launch_fused_split4_pairh32(const __nv_bfloat16* d_a,
                                 const __nv_bfloat16* d_w1,
                                 const __nv_bfloat16* d_b1,
                                 const __nv_bfloat16* d_w2,
                                 const __nv_bfloat16* d_b2,
                                 __nv_bfloat16* d_out) {
    dim3 grid(kM / TM, 1, 1);
    ffn12_fused_split4_pairh32_bf16_kernel<TM, GeluMode, TK>
        <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
}

template <int TM, int GeluMode = kGeluErf>
void launch_fused_split2_quadh32(const __nv_bfloat16* d_a,
                                 const __nv_bfloat16* d_w1,
                                 const __nv_bfloat16* d_b1,
                                 const __nv_bfloat16* d_w2,
                                 const __nv_bfloat16* d_b2,
                                 __nv_bfloat16* d_out) {
    dim3 grid(kM / TM, 1, 1);
    ffn12_fused_split2_quadh32_bf16_kernel<TM, GeluMode>
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
#ifdef CUDASEP_FFN12_CANDIDATES_ONLY
        __nv_bfloat16* d_w1_bkn = nullptr;
        __nv_bfloat16* d_w2_bkn = nullptr;
#endif
        __nv_bfloat16* d_b2 = nullptr;
        __nv_bfloat16* d_out = nullptr;
        __nv_bfloat16* d_partial = nullptr;
        CUDA_CHECK(cudaMalloc(&d_a, a_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_w1, w1_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_b1, b1_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_hidden, hidden_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_w2, w2_elems * sizeof(__nv_bfloat16)));
#ifdef CUDASEP_FFN12_CANDIDATES_ONLY
        CUDA_CHECK(cudaMalloc(&d_w1_bkn, w1_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_w2_bkn, w2_elems * sizeof(__nv_bfloat16)));
#endif
        CUDA_CHECK(cudaMalloc(&d_b2, b2_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_out, out_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_partial, 4 * out_elems * sizeof(__nv_bfloat16)));

        init_bf16(d_a, a_elems);
        init_bf16(d_w1, w1_elems);
        init_bf16(d_b1, b1_elems);
        init_bf16(d_w2, w2_elems);
        init_bf16(d_b2, b2_elems);
#ifdef CUDASEP_FFN12_CANDIDATES_ONLY
        init_bkn_from_nt(d_w1, d_w1_bkn, kIn, kHidden);
        init_bkn_from_nt(d_w2, d_w2_bkn, kHidden, kOut);
#endif
        CUDA_CHECK(cudaDeviceSynchronize());

        std::printf("FFN1->GELU->FFN2 CUDA Tile fusion probe\n");
        std::printf("shape: M=%d, in=%d, hidden=%d, out=%d, BF16 storage, FP32 mma accumulate\n",
                    kM, kIn, kHidden, kOut);

#ifdef CUDASEP_FFN12_CANDIDATES_ONLY
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32<32, kGeluErfPoly9L30, 64>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_bkn")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_bkn", 32, 256, 1, 1, opts,
                        d_a, d_w1_bkn, d_b1, d_w2_bkn, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32_bkn<32, kGeluErfPoly9L30, 64>(
                                d_a, d_w1_bkn, d_b1, d_w2_bkn, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_tinyblend_split2_pairh32_tk64")) {
            run_variant("fused_h32_poly9_tinyblend_split2_pairh32_tk64",
                        32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32<32, kGeluErfPoly9TinyBlend, 64>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_identity_split2_pairh32_tk64")) {
            run_variant("fused_h32_identity_split2_pairh32_tk64",
                        32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32<32, kGeluIdentity, 64>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_tk64")) {
            run_variant("fused_h32_poly9_split2_tk64", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2<32, 32, kGeluErfPoly9L30, true, true, 64>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_m8_h32_poly9_split2_pairh32_tk64")) {
            run_variant("fused_m8_h32_poly9_split2_pairh32_tk64", 8, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32<8, kGeluErfPoly9L30, 64>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk128")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk128", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32<32, kGeluErfPoly9L30, 128>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk256")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk256", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32<32, kGeluErfPoly9L30, 256>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h64_poly9_split2_pairh64_tk64")) {
            run_variant("fused_h64_poly9_split2_pairh64_tk64", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32<32, kGeluErfPoly9L30, 64, 64>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h64_poly9_split2_pairh64_tk64_idx32")) {
            run_variant("fused_h64_poly9_split2_pairh64_tk64_idx32", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32
                                <32, kGeluErfPoly9L30, 64, 64, false, false, true, true,
                                 false, true, true, int>(
                                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h64_poly9_split2_pairh64_tk64_outnoround")) {
            run_variant("fused_h64_poly9_split2_pairh64_tk64_outnoround",
                        32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32
                                <32, kGeluErfPoly9L30, 64, 64, false, false, true, true,
                                 false, true, false>(
                                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h64_poly9_split2_pairh64_tk64_idx32_outnoround")) {
            run_variant("fused_h64_poly9_split2_pairh64_tk64_idx32_outnoround",
                        32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32
                                <32, kGeluErfPoly9L30, 64, 64, false, false, true, true,
                                 false, true, false, int>(
                                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_halfout")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_halfout", 32, 128, 2, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32_halfout<32, kGeluErfPoly9L30, 64>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_quarterout")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_quarterout",
                        32, 64, 4, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32_quarterout
                                <32, kGeluErfPoly9L30, 64>(
                                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_quarterout_w1w2lat2")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_quarterout_w1w2lat2",
                        32, 64, 4, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32_quarterout_lat2
                                <32, kGeluErfPoly9L30, 64>(
                                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_outseq")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_outseq", 32, 128, 2, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32_outseq<32, kGeluErfPoly9L30, 64>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_noround")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_noround", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32
                                <32, kGeluErfPoly9L30, 64, 32, false, false, true, true,
                                 false, false>(
                                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_outnoround")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_outnoround", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32
                                <32, kGeluErfPoly9L30, 64, 32, false, false, true, true,
                                 false, true, false>(
                                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_idx32")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_idx32", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32
                                <32, kGeluErfPoly9L30, 64, 32, false, false, true, true,
                                 false, true, true, int>(
                                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_idx32_outnoround")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_idx32_outnoround",
                        32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32
                                <32, kGeluErfPoly9L30, 64, 32, false, false, true, true,
                                 false, true, false, int>(
                                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_nooutbias")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_nooutbias", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32
                                <32, kGeluErfPoly9L30, 64, 32, false, false, true, false>(
                                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_nobias")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_nobias", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32
                                <32, kGeluErfPoly9L30, 64, 32, false, false, false, false>(
                                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_w1lat8")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_w1lat8", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32
                                <32, kGeluErfPoly9L30, 64, 32, false, false, true, true, true>(
	                                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
	                        });
	        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_w1lat2")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_w1lat2", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32
                                <32, kGeluErfPoly9L30, 64, 32, false, false, true, true,
                                 true, true, true, long long, false, false, false, false, 2>(
                                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_w1batched2")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_w1batched2",
                        32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32
                                <32, kGeluErfPoly9L30, 64, 32, false, false, true, true,
                                 false, true, true, long long, false, false, true>(
	                                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
	                        });
	        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_w2batched2")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_w2batched2",
                        32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32
                                <32, kGeluErfPoly9L30, 64, 32, false, false, true, true,
                                 false, true, true, long long, false, false, false, true>(
	                                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
	                        });
	        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_hsplit2_partial")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_hsplit2_partial",
                        32, 256, 1, 2, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32_hsplit_partial<2>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_partial, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_hsplit4_partial")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_hsplit4_partial",
                        32, 256, 1, 2, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32_hsplit_partial<4>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_partial, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_stagedhidden")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_stagedhidden", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32
                                <32, kGeluErfPoly9L30, 64, 32, false, false, true, true,
                                 false, true, true, long long, true>(
                                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split4_pairh32_tk64")) {
            run_variant("fused_h32_poly9_split4_pairh32_tk64", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split4_pairh32<32, kGeluErfPoly9L30, 64>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_accgroup")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_accgroup", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32
                                <32, kGeluErfPoly9L30, 64, 32, false, true>(
                                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_w2lat8")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_w2lat8", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32
                                <32, kGeluErfPoly9L30, 64, 32, true>(
                                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_w2lat2")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_w2lat2", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32
                                <32, kGeluErfPoly9L30, 64, 32, true, false, true, true,
                                 false, true, true, long long, false, false, false, false, 2>(
                                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_w1w2lat2")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_w1w2lat2", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32
                                <32, kGeluErfPoly9L30, 64, 32, true, false, true, true,
                                 true, true, true, long long, false, false, false, false, 2>(
                                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_w2temp")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_w2temp", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32
                                <32, kGeluErfPoly9L30, 64, 32, false, false, true, true,
                                 false, true, true, long long, false, true>(
                                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_w2splitspan")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_w2splitspan",
                        32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32_w2splitspan
                                <32, kGeluErfPoly9L30, 64>(
                                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_w2manual")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_w2manual",
                        32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32_w2manual
                                <32, kGeluErfPoly9L30, 64>(
                                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
#else
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
        if (should_run(opts, "fused_m16_h16_poly9_split2")) {
            run_variant("fused_m16_h16_poly9_split2", 16, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2<16, 16, kGeluErfPoly9L30>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_m16_h32_poly9_split2")) {
            run_variant("fused_m16_h32_poly9_split2", 16, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2<16, 32, kGeluErfPoly9L30>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_m16_h64_poly9_split2")) {
            run_variant("fused_m16_h64_poly9_split2", 16, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2<16, 64, kGeluErfPoly9L30>(
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
        if (should_run(opts, "fused_h32_identity_split2")) {
            run_variant("fused_h32_identity_split2", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2<32, 32, kGeluIdentity>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_identity_nooutbias_split2")) {
            run_variant("fused_h32_identity_nooutbias_split2", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2<32, 32, kGeluIdentity, true, false>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_rawhidden_split2")) {
            run_variant("fused_h32_rawhidden_split2", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2<32, 32, kGeluIdentity, false>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32")) {
            run_variant("fused_h32_poly9_split2_pairh32", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32<32, kGeluErfPoly9L30>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk16")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk16", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32<32, kGeluErfPoly9L30, 16>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32<32, kGeluErfPoly9L30, 64>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_accgroup")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_accgroup", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32
                                <32, kGeluErfPoly9L30, 64, 32, false, true>(
                                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_w2lat8")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_w2lat8", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32
                                <32, kGeluErfPoly9L30, 64, 32, true>(
                                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_w2temp")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_w2temp", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32
                                <32, kGeluErfPoly9L30, 64, 32, false, false, true, true,
                                 false, true, true, long long, false, true>(
                                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_tk64_occ2")) {
            run_variant("fused_h32_poly9_split2_pairh32_tk64_occ2", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32_occ2<32, kGeluErfPoly9L30, 64>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h16_poly9_split2_pairh16_tk64")) {
            run_variant("fused_h16_poly9_split2_pairh16_tk64", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32<32, kGeluErfPoly9L30, 64, 16>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h64_poly9_split2_pairh64_tk64")) {
            run_variant("fused_h64_poly9_split2_pairh64_tk64", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32<32, kGeluErfPoly9L30, 64, 64>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_m16_h32_poly9_split2_pairh32")) {
            run_variant("fused_m16_h32_poly9_split2_pairh32", 16, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32<16, kGeluErfPoly9L30>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_m16_h32_poly9_split2_pairh32_tk64")) {
            run_variant("fused_m16_h32_poly9_split2_pairh32_tk64", 16, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32<16, kGeluErfPoly9L30, 64>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_tinyblend_split2_pairh32")) {
            run_variant("fused_h32_poly9_tinyblend_split2_pairh32", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32<32, kGeluErfPoly9TinyBlend>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_tinyblend_split2_pairh32_tk64")) {
            run_variant("fused_h32_poly9_tinyblend_split2_pairh32_tk64",
                        32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32<32, kGeluErfPoly9TinyBlend, 64>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_identity_split2_pairh32")) {
            run_variant("fused_h32_identity_split2_pairh32", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32<32, kGeluIdentity>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_identity_split2_pairh32_tk64")) {
            run_variant("fused_h32_identity_split2_pairh32_tk64",
                        32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32<32, kGeluIdentity, 64>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_pairh32_source_style")) {
            run_variant("fused_h32_poly9_split2_pairh32_source_style", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_pairh32_source_style
                                <32, kGeluErfPoly9L30, false>(
                                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split4_pairh32")) {
            run_variant("fused_h32_poly9_split4_pairh32", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split4_pairh32<32, kGeluErfPoly9L30>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_out);
                        });
        }
        if (should_run(opts, "fused_h32_poly9_split2_quadh32")) {
            run_variant("fused_h32_poly9_split2_quadh32", 32, 256, 1, 1, opts,
                        d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out,
                        [&] {
                            launch_fused_split2_quadh32<32, kGeluErfPoly9L30>(
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
#endif

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_w1));
        CUDA_CHECK(cudaFree(d_b1));
        CUDA_CHECK(cudaFree(d_hidden));
        CUDA_CHECK(cudaFree(d_w2));
#ifdef CUDASEP_FFN12_CANDIDATES_ONLY
        CUDA_CHECK(cudaFree(d_w1_bkn));
        CUDA_CHECK(cudaFree(d_w2_bkn));
#endif
        CUDA_CHECK(cudaFree(d_b2));
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_partial));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
