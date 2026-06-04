#include "mbr_cuda_tile.h"

#include "cuda_tile.h"
#include <cuda_bf16.h>
#include <cstdlib>
#include <cstring>

namespace cudasep::mbr_tile {
namespace {

namespace ct = cuda::tiles;
using namespace ct::literals;

constexpr int kLinearCutileStaticM = 78048;
constexpr int kLinearCutileExpectedM = 78060;
constexpr int kLinearCutileTileM = 32;
constexpr int kLinearCutileTileK = 32;

constexpr int kGeluErf = 0;
constexpr int kGeluHard = 1;
constexpr int kGeluQuick = 2;
constexpr int kGeluTanh = 3;
constexpr int kGeluErfPoly5L25 = 4;
constexpr int kGeluErfPoly7L25 = 5;
constexpr int kGeluErfPoly9L30 = 6;
constexpr int kGeluErfPoly9TinyBlendL30 = 7;

bool env_flag_enabled(const char* name) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) return false;
    return !(raw[0] == '\0' || std::strcmp(raw, "0") == 0 ||
             std::strcmp(raw, "false") == 0 || std::strcmp(raw, "FALSE") == 0 ||
             std::strcmp(raw, "off") == 0 || std::strcmp(raw, "OFF") == 0);
}

bool ffn12_output_no_round_enabled() {
    static int disabled = env_flag_enabled("CUDASEP_DISABLE_FFN12_OUT_NO_ROUND") ? 1 : 0;
    return disabled == 0;
}

bool ffn12_pairh32_latency2_enabled() {
    static int disabled = env_flag_enabled("CUDASEP_DISABLE_FFN12_PAIRH32_LAT2") ? 1 : 0;
    return disabled == 0;
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

template <bool FullBF16, typename TileT>
static __tile__ auto gelu_erf_approx(TileT x) {
    auto zero = x * 0.0f;
    auto one = zero + 1.0f;
    auto sign = ct::select(x < zero, zero - one, one);
    auto ax = ct::select(x < zero, zero - x, x);
    auto t = one / (one + 0.3275911f * ax);
    t = bf16_round_if<FullBF16>(t);
    auto poly = (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t -
                  0.284496736f) *
                     t +
                 0.254829592f) *
                t;
    poly = bf16_round_if<FullBF16>(poly);
    auto erf_approx = sign * (one - poly * ct::exp(zero - ax * ax));
    erf_approx = bf16_round_if<FullBF16>(erf_approx);
    auto gelu = 0.5f * x * (one + erf_approx);
    return bf16_round_if<FullBF16>(gelu);
}

template <bool FullBF16, typename TileT>
static __tile__ auto gelu_hard_approx(TileT x) {
    auto zero = x * 0.0f;
    auto gate = ct::min(ct::max(0.5f + 0.2f * x, zero), zero + 1.0f);
    gate = bf16_round_if<FullBF16>(gate);
    auto gelu = x * gate;
    return bf16_round_if<FullBF16>(gelu);
}

template <bool FullBF16, typename TileT>
static __tile__ auto gelu_quick_approx(TileT x) {
    auto sigmoid = 1.0f / (1.0f + ct::exp(-1.702f * x));
    sigmoid = bf16_round_if<FullBF16>(sigmoid);
    auto gelu = x * sigmoid;
    return bf16_round_if<FullBF16>(gelu);
}

template <bool FullBF16, typename TileT>
static __tile__ auto gelu_tanh_approx(TileT x) {
    auto x2 = x * x;
    x2 = bf16_round_if<FullBF16>(x2);
    auto cubic = x2 * x;
    cubic = bf16_round_if<FullBF16>(cubic);
    auto inner = 0.7978845608f * (x + 0.044715f * cubic);
    inner = bf16_round_if<FullBF16>(inner);
    auto gate = 0.5f * (1.0f + tanh(inner));
    gate = bf16_round_if<FullBF16>(gate);
    auto gelu = x * gate;
    return bf16_round_if<FullBF16>(gelu);
}

template <bool FullBF16, typename TileT>
static __tile__ auto gelu_erf_poly5_l25(TileT x) {
    auto zero = x * 0.0f;
    auto one = zero + 1.0f;
    auto ax = ct::select(x < zero, zero - x, x);
    auto z = ax * ax;
    z = bf16_round_if<FullBF16>(z);
    auto p = (((0.000677416775f * z - 0.0121774335f) * z +
               0.0889425898f) * z - 0.361254819f) * z +
             1.12684393f;
    p = bf16_round_if<FullBF16>(p);
    auto erf_abs = ct::min(ct::max(ax * p, zero), one);
    erf_abs = bf16_round_if<FullBF16>(erf_abs);
    auto erf_approx = ct::select(x < zero, zero - erf_abs, erf_abs);
    auto gelu = 0.5f * x * (one + erf_approx);
    return bf16_round_if<FullBF16>(gelu);
}

template <bool FullBF16, typename TileT>
static __tile__ auto gelu_erf_poly7_l25(TileT x) {
    auto zero = x * 0.0f;
    auto one = zero + 1.0f;
    auto ax = ct::select(x < zero, zero - x, x);
    auto z = ax * ax;
    z = bf16_round_if<FullBF16>(z);
    auto p = ((((((0.0000119948033f * z - 0.000310497426f) * z +
                  0.00352976049f) * z - 0.0238667561f) * z +
                0.110178845f) * z - 0.37522094f) * z +
              1.12832882f);
    p = bf16_round_if<FullBF16>(p);
    auto erf_abs = ct::min(ct::max(ax * p, zero), one);
    erf_abs = bf16_round_if<FullBF16>(erf_abs);
    auto erf_approx = ct::select(x < zero, zero - erf_abs, erf_abs);
    auto gelu = 0.5f * x * (one + erf_approx);
    return bf16_round_if<FullBF16>(gelu);
}

template <bool FullBF16, typename TileT>
static __tile__ auto gelu_erf_poly9_l30(TileT x) {
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

template <typename TileT>
static __tile__ auto gelu_erf_poly9_l30_fast(TileT x) {
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

template <bool FullBF16, typename TileT>
static __tile__ auto gelu_erf_poly9_tinyblend_l30(TileT x) {
    auto gelu = gelu_erf_poly9_l30<FullBF16>(x);
    auto blended = x + 0.0009765625f * (gelu - x);
    return bf16_round_if<FullBF16>(blended);
}

template <typename TileT>
static __tile__ auto gelu_erf_poly9_tinyblend_l30_fast(TileT x) {
    auto gelu = gelu_erf_poly9_l30_fast(x);
    return x + 0.0009765625f * (gelu - x);
}

template <int GeluMode, bool FullBF16, typename TileT>
static __tile__ auto gelu_selected(TileT x) {
    if constexpr (GeluMode == kGeluErfPoly9L30) {
        return gelu_erf_poly9_l30<FullBF16>(x);
    } else if constexpr (GeluMode == kGeluErfPoly9TinyBlendL30) {
        if constexpr (FullBF16) {
            return gelu_erf_poly9_tinyblend_l30<FullBF16>(x);
        } else {
            return gelu_erf_poly9_tinyblend_l30_fast(x);
        }
    } else if constexpr (GeluMode == kGeluErfPoly7L25) {
        return gelu_erf_poly7_l25<FullBF16>(x);
    } else if constexpr (GeluMode == kGeluErfPoly5L25) {
        return gelu_erf_poly5_l25<FullBF16>(x);
    } else if constexpr (GeluMode == kGeluTanh) {
        return gelu_tanh_approx<FullBF16>(x);
    } else if constexpr (GeluMode == kGeluQuick) {
        return gelu_quick_approx<FullBF16>(x);
    } else if constexpr (GeluMode == kGeluHard) {
        return gelu_hard_approx<FullBF16>(x);
    } else {
        static_assert(GeluMode == kGeluErf);
        return gelu_erf_approx<FullBF16>(x);
    }
}

template <int GeluMode, bool FullBF16, bool AddResidual = false>
__tile_global__ void ffn12_fused256_cutile_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ out) {
    constexpr int Hidden = 1024;
    constexpr int Out = 256;
    constexpr int HiddenTile = 32;
    using HiddenAccTile = ct::tile<float, ct::shape<kLinearCutileTileM, HiddenTile>>;
    using OutAccTile = ct::tile<float, ct::shape<kLinearCutileTileM, Out>>;
    using I64HiddenTile = ct::tile<long long, ct::shape<kLinearCutileTileM, HiddenTile>>;
    using I64OutTile = ct::tile<long long, ct::shape<kLinearCutileTileM, Out>>;

    a = ct::assume_aligned(a, 16_ic);
    w1_nt = ct::assume_aligned(w1_nt, 16_ic);
    b1 = ct::assume_aligned(b1, 16_ic);
    w2_nt = ct::assume_aligned(w2_nt, 16_ic);
    b2 = ct::assume_aligned(b2, 16_ic);
    if constexpr (AddResidual) {
        residual = ct::assume_aligned(residual, 16_ic);
    }
    out = ct::assume_aligned(out, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<kLinearCutileStaticM, 256>{}},
        ct::shape<kLinearCutileTileM, kLinearCutileTileK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<256, Hidden>{}, ct::layout_left{}},
        ct::shape<kLinearCutileTileK, HiddenTile>{}
    };
    auto w2_view = ct::partition_view{
        ct::tensor_span{w2_nt, ct::shape<Hidden, Out>{}, ct::layout_left{}},
        ct::shape<HiddenTile, Out>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out, ct::shape<kLinearCutileStaticM, Out>{}},
        ct::shape<kLinearCutileTileM, Out>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_n;
    (void)tile_z;

    auto out_acc = ct::full<OutAccTile>(0.0f);
    I64HiddenTile hidden_local = ct::iota<I64HiddenTile>();
    for (auto hidden_tile : ct::irange(std::size_t{0}, std::size_t{Hidden / HiddenTile})) {
        auto hidden_acc = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{256 / kLinearCutileTileK})) {
            hidden_acc = ct::mma(a_view.load(tile_m, kk),
                                 w1_view.load(kk, hidden_tile),
                                 hidden_acc);
        }
        auto hidden_cols =
            static_cast<long long>(hidden_tile) * HiddenTile + (hidden_local % HiddenTile);
        auto hidden_bias = ct::element_cast<float>(ct::load(b1 + hidden_cols));
        auto hidden_value = bf16_round(hidden_acc) + hidden_bias;
        hidden_value = bf16_round_if<FullBF16>(hidden_value);
        hidden_value = gelu_selected<GeluMode, FullBF16>(hidden_value);
        auto hidden_bf16 = ct::element_cast<__nv_bfloat16>(hidden_value);
        out_acc = ct::mma(hidden_bf16, w2_view.load(hidden_tile, 0), out_acc);
    }

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_cols = out_local % Out;
    auto out_bias = ct::element_cast<float>(ct::load(b2 + out_cols));
    auto value = bf16_round(out_acc) + out_bias;
    value = bf16_round_if<FullBF16>(value);
    auto out_value = ct::element_cast<__nv_bfloat16>(value);
    if constexpr (AddResidual) {
        auto residual_view = ct::partition_view{
            ct::tensor_span{residual, ct::shape<kLinearCutileStaticM, Out>{}},
            ct::shape<kLinearCutileTileM, Out>{}
        };
        out_value = residual_view.load(tile_m, 0) + out_value;
    }
    out_view.store(out_value, tile_m, 0);
}

template <int GeluMode, bool FullBF16, bool AddResidual = false>
__tile_global__ void ffn12_fused256_split2_cutile_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ out) {
    constexpr int Hidden = 1024;
    constexpr int Out = 256;
    constexpr int OutHalf = Out / 2;
    constexpr int HiddenTile = 32;
    using HiddenAccTile = ct::tile<float, ct::shape<kLinearCutileTileM, HiddenTile>>;
    using OutAccTile = ct::tile<float, ct::shape<kLinearCutileTileM, OutHalf>>;
    using I64HiddenTile = ct::tile<long long, ct::shape<kLinearCutileTileM, HiddenTile>>;
    using I64OutTile = ct::tile<long long, ct::shape<kLinearCutileTileM, OutHalf>>;

    a = ct::assume_aligned(a, 16_ic);
    w1_nt = ct::assume_aligned(w1_nt, 16_ic);
    b1 = ct::assume_aligned(b1, 16_ic);
    w2_nt = ct::assume_aligned(w2_nt, 16_ic);
    b2 = ct::assume_aligned(b2, 16_ic);
    if constexpr (AddResidual) {
        residual = ct::assume_aligned(residual, 16_ic);
    }
    out = ct::assume_aligned(out, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<kLinearCutileStaticM, 256>{}},
        ct::shape<kLinearCutileTileM, kLinearCutileTileK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<256, Hidden>{}, ct::layout_left{}},
        ct::shape<kLinearCutileTileK, HiddenTile>{}
    };
    auto w2_view = ct::partition_view{
        ct::tensor_span{w2_nt, ct::shape<Hidden, Out>{}, ct::layout_left{}},
        ct::shape<HiddenTile, OutHalf>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out, ct::shape<kLinearCutileStaticM, Out>{}},
        ct::shape<kLinearCutileTileM, OutHalf>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_n;
    (void)tile_z;

    auto out_acc0 = ct::full<OutAccTile>(0.0f);
    auto out_acc1 = ct::full<OutAccTile>(0.0f);
    I64HiddenTile hidden_local = ct::iota<I64HiddenTile>();
    for (auto hidden_tile : ct::irange(std::size_t{0}, std::size_t{Hidden / HiddenTile})) {
        auto hidden_acc = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{256 / kLinearCutileTileK})) {
            hidden_acc = ct::mma(a_view.load(tile_m, kk),
                                 w1_view.load(kk, hidden_tile),
                                 hidden_acc);
        }
        auto hidden_cols =
            static_cast<long long>(hidden_tile) * HiddenTile + (hidden_local % HiddenTile);
        auto hidden_bias = ct::element_cast<float>(ct::load(b1 + hidden_cols));
        auto hidden_value = bf16_round(hidden_acc) + hidden_bias;
        hidden_value = bf16_round_if<FullBF16>(hidden_value);
        hidden_value = gelu_selected<GeluMode, FullBF16>(hidden_value);
        auto hidden_bf16 = ct::element_cast<__nv_bfloat16>(hidden_value);
        out_acc0 = ct::mma(hidden_bf16, w2_view.load(hidden_tile, 0), out_acc0);
        out_acc1 = ct::mma(hidden_bf16, w2_view.load(hidden_tile, 1), out_acc1);
    }

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_cols = out_local % OutHalf;
    auto out_bias0 = ct::element_cast<float>(ct::load(b2 + out_cols));
    auto out_bias1 = ct::element_cast<float>(ct::load(b2 + OutHalf + out_cols));
    auto value0 = bf16_round(out_acc0) + out_bias0;
    auto value1 = bf16_round(out_acc1) + out_bias1;
    value0 = bf16_round_if<FullBF16>(value0);
    value1 = bf16_round_if<FullBF16>(value1);
    auto out_value0 = ct::element_cast<__nv_bfloat16>(value0);
    auto out_value1 = ct::element_cast<__nv_bfloat16>(value1);
    if constexpr (AddResidual) {
        auto residual_view = ct::partition_view{
            ct::tensor_span{residual, ct::shape<kLinearCutileStaticM, Out>{}},
            ct::shape<kLinearCutileTileM, OutHalf>{}
        };
        out_value0 = residual_view.load(tile_m, 0) + out_value0;
        out_value1 = residual_view.load(tile_m, 1) + out_value1;
    }
    out_view.store(out_value0, tile_m, 0);
    out_view.store(out_value1, tile_m, 1);
}

template <int GeluMode,
          bool FullBF16,
          int InputTileK = kLinearCutileTileK,
          bool AddResidual = false,
          bool RoundOutputAcc = true,
          int MemoryLatency = 0>
__tile_global__ void ffn12_fused256_split2_pairh32_cutile_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ out) {
    constexpr int Hidden = 1024;
    constexpr int Out = 256;
    constexpr int OutHalf = Out / 2;
    constexpr int HiddenTile = 32;
    static_assert(256 % InputTileK == 0);
    using HiddenAccTile = ct::tile<float, ct::shape<kLinearCutileTileM, HiddenTile>>;
    using OutAccTile = ct::tile<float, ct::shape<kLinearCutileTileM, OutHalf>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<kLinearCutileTileM, InputTileK>>;
    using W1Tile = ct::tile<__nv_bfloat16, ct::shape<InputTileK, HiddenTile>>;
    using W2Tile = ct::tile<__nv_bfloat16, ct::shape<HiddenTile, OutHalf>>;
    using I64HiddenTile = ct::tile<long long, ct::shape<kLinearCutileTileM, HiddenTile>>;
    using I64OutTile = ct::tile<long long, ct::shape<kLinearCutileTileM, OutHalf>>;

    a = ct::assume_aligned(a, 16_ic);
    w1_nt = ct::assume_aligned(w1_nt, 16_ic);
    b1 = ct::assume_aligned(b1, 16_ic);
    w2_nt = ct::assume_aligned(w2_nt, 16_ic);
    b2 = ct::assume_aligned(b2, 16_ic);
    if constexpr (AddResidual) {
        residual = ct::assume_aligned(residual, 16_ic);
    }
    out = ct::assume_aligned(out, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<kLinearCutileStaticM, 256>{}},
        ct::shape<kLinearCutileTileM, InputTileK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<256, Hidden>{}, ct::layout_left{}},
        ct::shape<InputTileK, HiddenTile>{}
    };
    auto w2_view = ct::partition_view{
        ct::tensor_span{w2_nt, ct::shape<Hidden, Out>{}, ct::layout_left{}},
        ct::shape<HiddenTile, OutHalf>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out, ct::shape<kLinearCutileStaticM, Out>{}},
        ct::shape<kLinearCutileTileM, OutHalf>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_n;
    (void)tile_z;

    auto out_acc0 = ct::full<OutAccTile>(0.0f);
    auto out_acc1 = ct::full<OutAccTile>(0.0f);
    I64HiddenTile hidden_local = ct::iota<I64HiddenTile>();
    for (auto hidden_pair : ct::irange(std::size_t{0},
                                       std::size_t{Hidden / (2 * HiddenTile)})) {
        auto hidden_tile0 = hidden_pair * 2;
        auto hidden_tile1 = hidden_tile0 + 1;
        auto hidden_acc0 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc1 = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{256 / InputTileK})) {
            if constexpr (MemoryLatency > 0) {
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

        auto hidden_cols0 =
            static_cast<long long>(hidden_tile0) * HiddenTile + (hidden_local % HiddenTile);
        auto hidden_cols1 =
            static_cast<long long>(hidden_tile1) * HiddenTile + (hidden_local % HiddenTile);
        auto hidden_bias0 = ct::element_cast<float>(ct::load(b1 + hidden_cols0));
        auto hidden_bias1 = ct::element_cast<float>(ct::load(b1 + hidden_cols1));
        auto hidden_value0 = bf16_round(hidden_acc0) + hidden_bias0;
        auto hidden_value1 = bf16_round(hidden_acc1) + hidden_bias1;
        if constexpr (!FullBF16 && GeluMode == kGeluErfPoly9L30) {
            hidden_value0 = gelu_erf_poly9_l30_fast(hidden_value0);
            hidden_value1 = gelu_erf_poly9_l30_fast(hidden_value1);
        } else if constexpr (!FullBF16 && GeluMode == kGeluErfPoly9TinyBlendL30) {
            hidden_value0 = gelu_erf_poly9_tinyblend_l30_fast(hidden_value0);
            hidden_value1 = gelu_erf_poly9_tinyblend_l30_fast(hidden_value1);
        } else {
            hidden_value0 = bf16_round_if<FullBF16>(hidden_value0);
            hidden_value1 = bf16_round_if<FullBF16>(hidden_value1);
            hidden_value0 = gelu_selected<GeluMode, FullBF16>(hidden_value0);
            hidden_value1 = gelu_selected<GeluMode, FullBF16>(hidden_value1);
        }
        auto hidden_bf16_0 = ct::element_cast<__nv_bfloat16>(hidden_value0);
        auto hidden_bf16_1 = ct::element_cast<__nv_bfloat16>(hidden_value1);
        if constexpr (MemoryLatency > 0) {
            W2Tile w2_00;
            W2Tile w2_01;
            W2Tile w2_10;
            W2Tile w2_11;
            [[cutile::hint(0, latency=MemoryLatency)]]
            w2_00 = w2_view.load(hidden_tile0, 0);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w2_01 = w2_view.load(hidden_tile0, 1);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w2_10 = w2_view.load(hidden_tile1, 0);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w2_11 = w2_view.load(hidden_tile1, 1);
            out_acc0 = ct::mma(hidden_bf16_0, w2_00, out_acc0);
            out_acc1 = ct::mma(hidden_bf16_0, w2_01, out_acc1);
            out_acc0 = ct::mma(hidden_bf16_1, w2_10, out_acc0);
            out_acc1 = ct::mma(hidden_bf16_1, w2_11, out_acc1);
        } else {
            out_acc0 = ct::mma(hidden_bf16_0, w2_view.load(hidden_tile0, 0), out_acc0);
            out_acc1 = ct::mma(hidden_bf16_0, w2_view.load(hidden_tile0, 1), out_acc1);
            out_acc0 = ct::mma(hidden_bf16_1, w2_view.load(hidden_tile1, 0), out_acc0);
            out_acc1 = ct::mma(hidden_bf16_1, w2_view.load(hidden_tile1, 1), out_acc1);
        }
    }

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_cols = out_local % OutHalf;
    auto out_bias0 = ct::element_cast<float>(ct::load(b2 + out_cols));
    auto out_bias1 = ct::element_cast<float>(ct::load(b2 + OutHalf + out_cols));
    auto value0 = bf16_round_if<RoundOutputAcc>(out_acc0) + out_bias0;
    auto value1 = bf16_round_if<RoundOutputAcc>(out_acc1) + out_bias1;
    if constexpr (FullBF16) {
        value0 = bf16_round(value0);
        value1 = bf16_round(value1);
    }
    auto out_value0 = ct::element_cast<__nv_bfloat16>(value0);
    auto out_value1 = ct::element_cast<__nv_bfloat16>(value1);
    if constexpr (AddResidual) {
        auto residual_view = ct::partition_view{
            ct::tensor_span{residual, ct::shape<kLinearCutileStaticM, Out>{}},
            ct::shape<kLinearCutileTileM, OutHalf>{}
        };
        out_value0 = residual_view.load(tile_m, 0) + out_value0;
        out_value1 = residual_view.load(tile_m, 1) + out_value1;
    }
    if constexpr (MemoryLatency > 0) {
        [[cutile::hint(0, latency=MemoryLatency)]]
        out_view.store(out_value0, tile_m, 0);
        [[cutile::hint(0, latency=MemoryLatency)]]
        out_view.store(out_value1, tile_m, 1);
    } else {
        out_view.store(out_value0, tile_m, 0);
        out_view.store(out_value1, tile_m, 1);
    }
}

template <int GeluMode, bool FullBF16>
__tile_global__ void ffn12_tail_ffn1_gelu_cutile_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    __nv_bfloat16* __restrict__ hidden_tail) {
    constexpr int TailPadM = 32;
    constexpr int TailRows = kLinearCutileExpectedM - kLinearCutileStaticM;
    constexpr int Hidden = 1024;
    constexpr int HiddenTile = 64;
    using HiddenAccTile = ct::tile<float, ct::shape<TailPadM, HiddenTile>>;
    using AIndexTile = ct::tile<long long, ct::shape<TailPadM, kLinearCutileTileK>>;
    using I64HiddenTile = ct::tile<long long, ct::shape<TailPadM, HiddenTile>>;

    a = ct::assume_aligned(a, 16_ic);
    w1_nt = ct::assume_aligned(w1_nt, 16_ic);
    b1 = ct::assume_aligned(b1, 16_ic);
    hidden_tail = ct::assume_aligned(hidden_tail, 16_ic);

    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<256, Hidden>{}, ct::layout_left{}},
        ct::shape<kLinearCutileTileK, HiddenTile>{}
    };
    auto hidden_view = ct::partition_view{
        ct::tensor_span{hidden_tail, ct::shape<TailPadM, Hidden>{}},
        ct::shape<TailPadM, HiddenTile>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_m;
    (void)tile_z;

    AIndexTile a_local = ct::iota<AIndexTile>();
    auto local_rows = a_local / kLinearCutileTileK;
    auto input_rows = static_cast<long long>(kLinearCutileStaticM) + local_rows;
    auto safe_input_rows = ct::select(local_rows < TailRows, input_rows, local_rows * 0LL);
    auto a_cols_local = a_local % kLinearCutileTileK;

    auto hidden_acc = ct::full<HiddenAccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{256 / kLinearCutileTileK})) {
        auto a_cols = static_cast<long long>(kk) * kLinearCutileTileK + a_cols_local;
        auto a_tile = ct::load_masked(a + safe_input_rows * 256 + a_cols,
                                      local_rows < TailRows);
        hidden_acc = ct::mma(a_tile, w1_view.load(kk, tile_n), hidden_acc);
    }

    I64HiddenTile hidden_local = ct::iota<I64HiddenTile>();
    auto hidden_cols = static_cast<long long>(tile_n) * HiddenTile +
                       (hidden_local % HiddenTile);
    auto hidden_bias = ct::element_cast<float>(ct::load(b1 + hidden_cols));
    auto hidden_value = bf16_round(hidden_acc) + hidden_bias;
    hidden_value = bf16_round_if<FullBF16>(hidden_value);
    hidden_value = gelu_selected<GeluMode, FullBF16>(hidden_value);
    hidden_view.store(ct::element_cast<__nv_bfloat16>(hidden_value), 0, tile_n);
}

template <bool FullBF16, bool AddResidual = false>
__tile_global__ void ffn12_tail_ffn2_cutile_kernel(
    const __nv_bfloat16* __restrict__ hidden_tail,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ out) {
    constexpr int TailPadM = 32;
    constexpr int TailRows = kLinearCutileExpectedM - kLinearCutileStaticM;
    constexpr int Hidden = 1024;
    constexpr int Out = 256;
    constexpr int HiddenTile = 64;
    using OutAccTile = ct::tile<float, ct::shape<TailPadM, Out>>;
    using AIndexTile = ct::tile<long long, ct::shape<TailPadM, HiddenTile>>;
    using I64OutTile = ct::tile<long long, ct::shape<TailPadM, Out>>;

    hidden_tail = ct::assume_aligned(hidden_tail, 16_ic);
    w2_nt = ct::assume_aligned(w2_nt, 16_ic);
    b2 = ct::assume_aligned(b2, 16_ic);
    if constexpr (AddResidual) {
        residual = ct::assume_aligned(residual, 16_ic);
    }
    out = ct::assume_aligned(out, 16_ic);

    auto w2_view = ct::partition_view{
        ct::tensor_span{w2_nt, ct::shape<Hidden, Out>{}, ct::layout_left{}},
        ct::shape<HiddenTile, Out>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_m;
    (void)tile_n;
    (void)tile_z;

    AIndexTile a_local = ct::iota<AIndexTile>();
    auto hidden_local_rows = a_local / HiddenTile;
    auto a_hidden_cols = a_local % HiddenTile;

    auto out_acc = ct::full<OutAccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{Hidden / HiddenTile})) {
        auto hidden_cols = static_cast<long long>(kk) * HiddenTile + a_hidden_cols;
        auto hidden_tile = ct::load(hidden_tail + hidden_local_rows * Hidden + hidden_cols);
        out_acc = ct::mma(hidden_tile, w2_view.load(kk, 0), out_acc);
    }

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_local_rows = out_local / Out;
    auto out_cols = out_local % Out;
    auto out_bias = ct::element_cast<float>(ct::load(b2 + out_cols));
    auto value = bf16_round(out_acc) + out_bias;
    value = bf16_round_if<FullBF16>(value);
    auto out_value = ct::element_cast<__nv_bfloat16>(value);
    auto output_rows = static_cast<long long>(kLinearCutileStaticM) + out_local_rows;
    auto safe_rows = ct::select(out_local_rows < TailRows, output_rows, out_local_rows * 0LL);
    if constexpr (AddResidual) {
        auto residual_value = ct::load_masked(residual + safe_rows * Out + out_cols,
                                             out_local_rows < TailRows);
        out_value = residual_value + out_value;
    }
    ct::store_masked(out + safe_rows * Out + out_cols,
                     out_value,
                     out_local_rows < TailRows);
}

template <int GeluMode, bool FullBF16, bool AddResidual>
void launch_ffn12_fused256_cutile_typed(const Tensor& x,
                                        const Tensor& linear1_w,
                                        const Tensor& linear1_b,
                                        const Tensor& linear2_w,
                                        const Tensor& linear2_b,
                                        const Tensor* residual,
                                        Tensor& out,
                                        bool split2_output,
                                        bool split2_pairh32,
                                        bool split2_pairh32_tk64) {
    dim3 full_grid(kLinearCutileStaticM / kLinearCutileTileM, 1, 1);
    const __nv_bfloat16* residual_ptr = AddResidual ? residual->data_bf16() : nullptr;
    if (split2_output && split2_pairh32) {
        if (split2_pairh32_tk64) {
            if constexpr (!FullBF16 && !AddResidual) {
                if (ffn12_output_no_round_enabled()) {
                    if (ffn12_pairh32_latency2_enabled()) {
                        ffn12_fused256_split2_pairh32_cutile_kernel<GeluMode,
                                                                    FullBF16,
                                                                    64,
                                                                    AddResidual,
                                                                    false,
                                                                    2>
                            <<<full_grid, 1>>>(
                                x.data_bf16(),
                                linear1_w.data_bf16(),
                                linear1_b.data_bf16(),
                                linear2_w.data_bf16(),
                                linear2_b.data_bf16(),
                                residual_ptr,
                                out.data_bf16());
                    } else {
                        ffn12_fused256_split2_pairh32_cutile_kernel<GeluMode,
                                                                    FullBF16,
                                                                    64,
                                                                    AddResidual,
                                                                    false>
                            <<<full_grid, 1>>>(
                                x.data_bf16(),
                                linear1_w.data_bf16(),
                                linear1_b.data_bf16(),
                                linear2_w.data_bf16(),
                                linear2_b.data_bf16(),
                                residual_ptr,
                                out.data_bf16());
                    }
                } else {
                    if (ffn12_pairh32_latency2_enabled()) {
                        ffn12_fused256_split2_pairh32_cutile_kernel<GeluMode,
                                                                    FullBF16,
                                                                    64,
                                                                    AddResidual,
                                                                    true,
                                                                    2>
                            <<<full_grid, 1>>>(
                                x.data_bf16(),
                                linear1_w.data_bf16(),
                                linear1_b.data_bf16(),
                                linear2_w.data_bf16(),
                                linear2_b.data_bf16(),
                                residual_ptr,
                                out.data_bf16());
                    } else {
                        ffn12_fused256_split2_pairh32_cutile_kernel<GeluMode,
                                                                    FullBF16,
                                                                    64,
                                                                    AddResidual>
                            <<<full_grid, 1>>>(
                                x.data_bf16(),
                                linear1_w.data_bf16(),
                                linear1_b.data_bf16(),
                                linear2_w.data_bf16(),
                                linear2_b.data_bf16(),
                                residual_ptr,
                                out.data_bf16());
                    }
                }
            } else {
                ffn12_fused256_split2_pairh32_cutile_kernel<GeluMode, FullBF16, 64,
                                                            AddResidual>
                    <<<full_grid, 1>>>(
                        x.data_bf16(),
                        linear1_w.data_bf16(),
                        linear1_b.data_bf16(),
                        linear2_w.data_bf16(),
                        linear2_b.data_bf16(),
                        residual_ptr,
                        out.data_bf16());
            }
        } else {
            ffn12_fused256_split2_pairh32_cutile_kernel<GeluMode, FullBF16,
                                                        kLinearCutileTileK,
                                                        AddResidual>
                <<<full_grid, 1>>>(
                    x.data_bf16(),
                    linear1_w.data_bf16(),
                    linear1_b.data_bf16(),
                    linear2_w.data_bf16(),
                    linear2_b.data_bf16(),
                    residual_ptr,
                    out.data_bf16());
        }
    } else if (split2_output) {
        ffn12_fused256_split2_cutile_kernel<GeluMode, FullBF16, AddResidual>
            <<<full_grid, 1>>>(
            x.data_bf16(),
            linear1_w.data_bf16(),
            linear1_b.data_bf16(),
            linear2_w.data_bf16(),
            linear2_b.data_bf16(),
            residual_ptr,
            out.data_bf16());
    } else {
        ffn12_fused256_cutile_kernel<GeluMode, FullBF16, AddResidual><<<full_grid, 1>>>(
            x.data_bf16(),
            linear1_w.data_bf16(),
            linear1_b.data_bf16(),
            linear2_w.data_bf16(),
            linear2_b.data_bf16(),
            residual_ptr,
            out.data_bf16());
    }

    Tensor tail_hidden = Tensor::empty({32, 1024}, DType::BFloat16);
    dim3 tail_ffn1_grid(1, 1024 / 64, 1);
    ffn12_tail_ffn1_gelu_cutile_kernel<GeluMode, FullBF16><<<tail_ffn1_grid, 1>>>(
        x.data_bf16(),
        linear1_w.data_bf16(),
        linear1_b.data_bf16(),
        tail_hidden.data_bf16());
    ffn12_tail_ffn2_cutile_kernel<FullBF16, AddResidual><<<1, 1>>>(
        tail_hidden.data_bf16(),
        linear2_w.data_bf16(),
        linear2_b.data_bf16(),
        residual_ptr,
        out.data_bf16());
}

template <bool FullBF16, bool AddResidual>
void launch_ffn12_fused256_cutile_mode(int gelu_mode,
                                       const Tensor& x,
                                       const Tensor& linear1_w,
                                       const Tensor& linear1_b,
                                       const Tensor& linear2_w,
                                       const Tensor& linear2_b,
                                       const Tensor* residual,
                                       Tensor& out,
                                       bool split2_output,
                                       bool split2_pairh32,
                                       bool split2_pairh32_tk64) {
    switch (gelu_mode) {
        case kGeluErfPoly9TinyBlendL30:
            launch_ffn12_fused256_cutile_typed<kGeluErfPoly9TinyBlendL30,
                                               FullBF16, AddResidual>(
                x, linear1_w, linear1_b, linear2_w, linear2_b, residual, out,
                split2_output, split2_pairh32, split2_pairh32_tk64);
            break;
        case kGeluErfPoly9L30:
            launch_ffn12_fused256_cutile_typed<kGeluErfPoly9L30, FullBF16, AddResidual>(
                x, linear1_w, linear1_b, linear2_w, linear2_b, residual, out,
                split2_output, split2_pairh32, split2_pairh32_tk64);
            break;
        case kGeluErfPoly7L25:
            launch_ffn12_fused256_cutile_typed<kGeluErfPoly7L25, FullBF16, AddResidual>(
                x, linear1_w, linear1_b, linear2_w, linear2_b, residual, out,
                split2_output, split2_pairh32, split2_pairh32_tk64);
            break;
        case kGeluErfPoly5L25:
            launch_ffn12_fused256_cutile_typed<kGeluErfPoly5L25, FullBF16, AddResidual>(
                x, linear1_w, linear1_b, linear2_w, linear2_b, residual, out,
                split2_output, split2_pairh32, split2_pairh32_tk64);
            break;
        case kGeluTanh:
            launch_ffn12_fused256_cutile_typed<kGeluTanh, FullBF16, AddResidual>(
                x, linear1_w, linear1_b, linear2_w, linear2_b, residual, out,
                split2_output, split2_pairh32, split2_pairh32_tk64);
            break;
        case kGeluQuick:
            launch_ffn12_fused256_cutile_typed<kGeluQuick, FullBF16, AddResidual>(
                x, linear1_w, linear1_b, linear2_w, linear2_b, residual, out,
                split2_output, split2_pairh32, split2_pairh32_tk64);
            break;
        case kGeluHard:
            launch_ffn12_fused256_cutile_typed<kGeluHard, FullBF16, AddResidual>(
                x, linear1_w, linear1_b, linear2_w, linear2_b, residual, out,
                split2_output, split2_pairh32, split2_pairh32_tk64);
            break;
        default:
            launch_ffn12_fused256_cutile_typed<kGeluErf, FullBF16, AddResidual>(
                x, linear1_w, linear1_b, linear2_w, linear2_b, residual, out,
                split2_output, split2_pairh32, split2_pairh32_tk64);
            break;
    }
}

}  // namespace

void launch_ffn12_fused256_cutile(int gelu_mode,
                                  bool full_bf16,
                                  bool split2_output,
                                  bool split2_pairh32,
                                  bool split2_pairh32_tk64,
                                  const Tensor& x,
                                  const Tensor& linear1_w,
                                  const Tensor& linear1_b,
                                  const Tensor& linear2_w,
                                  const Tensor& linear2_b,
                                  Tensor& out) {
    if (full_bf16) {
        launch_ffn12_fused256_cutile_mode<true, false>(
            gelu_mode, x, linear1_w, linear1_b, linear2_w, linear2_b, nullptr, out,
            split2_output, split2_pairh32, split2_pairh32_tk64);
    } else {
        launch_ffn12_fused256_cutile_mode<false, false>(
            gelu_mode, x, linear1_w, linear1_b, linear2_w, linear2_b, nullptr, out,
            split2_output, split2_pairh32, split2_pairh32_tk64);
    }
}

void launch_ffn12_fused256_residual_cutile(int gelu_mode,
                                           bool full_bf16,
                                           bool split2_output,
                                           bool split2_pairh32,
                                           bool split2_pairh32_tk64,
                                           const Tensor& x,
                                           const Tensor& linear1_w,
                                           const Tensor& linear1_b,
                                           const Tensor& linear2_w,
                                           const Tensor& linear2_b,
                                           const Tensor& residual,
                                           Tensor& out) {
    (void)gelu_mode;
    (void)full_bf16;
    (void)split2_output;
    (void)split2_pairh32;
    (void)split2_pairh32_tk64;
    dim3 full_grid(kLinearCutileStaticM / kLinearCutileTileM, 1, 1);
    ffn12_fused256_split2_pairh32_cutile_kernel<kGeluErfPoly9L30, false, 64, true>
        <<<full_grid, 1>>>(
            x.data_bf16(),
            linear1_w.data_bf16(),
            linear1_b.data_bf16(),
            linear2_w.data_bf16(),
            linear2_b.data_bf16(),
            residual.data_bf16(),
            out.data_bf16());

    Tensor tail_hidden = Tensor::empty({32, 1024}, DType::BFloat16);
    dim3 tail_ffn1_grid(1, 1024 / 64, 1);
    ffn12_tail_ffn1_gelu_cutile_kernel<kGeluErfPoly9L30, false><<<tail_ffn1_grid, 1>>>(
        x.data_bf16(),
        linear1_w.data_bf16(),
        linear1_b.data_bf16(),
        tail_hidden.data_bf16());
    ffn12_tail_ffn2_cutile_kernel<false, true><<<1, 1>>>(
        tail_hidden.data_bf16(),
        linear2_w.data_bf16(),
        linear2_b.data_bf16(),
        residual.data_bf16(),
        out.data_bf16());
}

}  // namespace cudasep::mbr_tile
