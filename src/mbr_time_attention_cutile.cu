#include "mbr_cuda_tile.h"

#include "cuda_tile.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudasep::mbr_tile {
namespace {

namespace ct = cuda::tiles;
using namespace ct::literals;

constexpr int kTimeAttnN = 1301;
constexpr int kTimeAttnD = 64;
constexpr int kTimeAttnHeads = 8;
constexpr int kTimeAttnHeadStride = kTimeAttnHeads * kTimeAttnD;
constexpr int kTimeAttnMainN = 1280;
constexpr int kTimeAttnCutileQRows16 = 16;
constexpr int kTimeAttnCutileQRows32 = 32;
constexpr int kTimeAttnCutileQRows64 = 64;
constexpr int kTimeAttnCutileQRows128 = 128;
constexpr int kTimeAttnCutileKTile32 = 32;
constexpr int kTimeAttnCutileKTile64 = 64;
constexpr int kTimeAttnCutileKTile128 = 128;
constexpr int kTimeAttnStatsCols = 4;
constexpr float kLog2E = 1.44269504088896340736f;

static inline int64_t ceildiv(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

bool env_flag_enabled(const char* name) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) return false;
    std::string value(raw);
    return !(value.empty() || value == "0" || value == "false" || value == "FALSE" ||
             value == "off" || value == "OFF");
}

bool time_attention_approx_clamp_prob_enabled() {
    return env_flag_enabled("CUDASEP_TIME_ATTENTION_APPROX_SOFTMAX_CLAMP_PROB");
}

bool time_attention_approx_poly2_l4_enabled() {
    return env_flag_enabled("CUDASEP_TIME_ATTENTION_APPROX_SOFTMAX_POLY2_L4");
}

template <bool UseExp2, typename TileT>
static __tile__ auto softmax_exp(TileT x) {
    if constexpr (UseExp2) {
        return ct::exp2(x * kLog2E);
    }
    return ct::exp(x);
}

enum ProbMode : int {
    kProbExp = 0,
    kProbPoly3NoClamp = 8,
    kProbPoly3Clamp = 9,
    kProbPoly2NoClampL4 = 10,
};

enum AlphaMode : int {
    kAlphaExact = 0,
    kAlphaProbClamp = 1,
};

template <int Prob, typename TileT>
static __tile__ auto softmax_prob(TileT x) {
    if constexpr (Prob == kProbPoly3NoClamp) {
        auto t = x * 0.333333343f + 1.0f;
        auto t2 = t * t;
        return t2 * t;
    } else if constexpr (Prob == kProbPoly3Clamp) {
        auto zero = x * 0.0f;
        auto t = x * 0.333333343f + 1.0f;
        t = ct::select(t > zero, t, zero);
        auto t2 = t * t;
        return t2 * t;
    } else if constexpr (Prob == kProbPoly2NoClampL4) {
        auto t = x * 0.25f + 1.0f;
        return t * t;
    }
    return softmax_exp<true>(x);
}

template <int Prob, typename TileT>
static __tile__ auto softmax_alpha_approx(TileT x) {
    auto zero = x * 0.0f;
    if constexpr (Prob == kProbPoly3NoClamp || Prob == kProbPoly3Clamp) {
        auto t = x * 0.333333343f + 1.0f;
        t = ct::select(t > zero, t, zero);
        auto t2 = t * t;
        return t2 * t;
    } else if constexpr (Prob == kProbPoly2NoClampL4) {
        auto t = x * 0.25f + 1.0f;
        t = ct::select(t > zero, t, zero);
        return t * t;
    }
    return softmax_exp<true>(x);
}

template <bool UseExp2, int Prob, int Alpha, typename TileT>
static __tile__ auto softmax_alpha(TileT x) {
    if constexpr (Alpha == kAlphaProbClamp) {
        return softmax_alpha_approx<Prob>(x);
    }
    return softmax_exp<UseExp2>(x);
}

template <bool UseExp2, int Prob, typename ScoreTile, typename RowTile>
static __tile__ auto softmax_probs_from_scores(ScoreTile scores, RowTile new_m) {
    if constexpr (Prob == kProbPoly3NoClamp) {
        return softmax_prob<Prob>(scores - new_m);
    }
    return softmax_exp<UseExp2>(scores - new_m);
}

template <typename T>
static __tile__ auto bf16_round(T value) {
    return ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(value));
}

template <int QRows, typename QTile, typename TrigT>
static __tile__ auto rotate_q_tile_for_attention(QTile q_tile,
                                                 const TrigT* __restrict__ cos_f,
                                                 const TrigT* __restrict__ sin_f,
                                                 long long q_block,
                                                 bool full_bf16) {
    constexpr int kHalfDim = kTimeAttnD / 2;
    using Q4Tile = ct::tile<float, ct::shape<QRows, kHalfDim, 2>>;
    using PairTile = ct::tile<float, ct::shape<QRows, kHalfDim, 1>>;
    using I64PairTile = ct::tile<long long, ct::shape<QRows, kHalfDim, 1>>;

    Q4Tile q4 = ct::reshape(ct::element_cast<float>(q_tile),
                            ct::shape<QRows, kHalfDim, 2>{});
    PairTile even = ct::extract(q4, ct::shape<QRows, kHalfDim, 1>{}, 0, 0, 0);
    PairTile odd = ct::extract(q4, ct::shape<QRows, kHalfDim, 1>{}, 0, 0, 1);
    even = ct::select(full_bf16, bf16_round(even), even);
    odd = ct::select(full_bf16, bf16_round(odd), odd);

    I64PairTile local = ct::iota<I64PairTile>();
    auto rows = q_block * (long long)QRows + local / kHalfDim;
    auto row_valid = rows < kTimeAttnN;
    auto safe_rows = ct::select(row_valid, rows, rows * 0LL);
    auto pair = local % kHalfDim;
    auto c = ct::element_cast<float>(
        ct::load_masked(cos_f + safe_rows * kHalfDim + pair, row_valid));
    auto s = ct::element_cast<float>(
        ct::load_masked(sin_f + safe_rows * kHalfDim + pair, row_valid));
    c = ct::select(full_bf16, bf16_round(c), c);
    s = ct::select(full_bf16, bf16_round(s), s);

    auto rot_even = even * c - odd * s;
    auto rot_odd = even * s + odd * c;
    rot_even = ct::select(full_bf16, bf16_round(rot_even), rot_even);
    rot_odd = ct::select(full_bf16, bf16_round(rot_odd), rot_odd);
    auto rotated = ct::reshape(ct::cat<2>(rot_even, rot_odd),
                               ct::shape<QRows, kTimeAttnD>{});
    return ct::element_cast<__nv_bfloat16>(rotated);
}

template <int QRows, int KTile, int QBlockOffset = 0, bool UseExp2 = false>
__tile_global__ void time_attention1301_cutile_qk_softmax_av_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int KTiles = (kTimeAttnN + KTile - 1) / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kTimeAttnD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kTimeAttnD>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block_local, bh, tile_z] = ct::bid();
    (void)tile_z;
    auto q_block = q_block_local + QBlockOffset;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kTimeAttnN * kTimeAttnD;
    const __nv_bfloat16* k_batch =
        k + static_cast<std::size_t>(bh) * kTimeAttnN * kTimeAttnD;
    const __nv_bfloat16* v_batch =
        v + static_cast<std::size_t>(bh) * kTimeAttnN * kTimeAttnD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kTimeAttnN * kTimeAttnD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kTimeAttnN, kTimeAttnD>{}},
        ct::shape<QRows, kTimeAttnD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kTimeAttnD, kTimeAttnN>{}, ct::layout_left{}},
        ct::shape<kTimeAttnD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kTimeAttnN, kTimeAttnD>{}},
        ct::shape<KTile, kTimeAttnD>{}
    };

    auto q_tile = q_view.load_masked(q_block, 0);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto score_rows =
        static_cast<long long>(q_block) * QRows + score_local / KTile;
    auto score_cols_local = score_local % KTile;

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{KTiles})) {
        auto key_cols =
            static_cast<long long>(kt) * KTile + score_cols_local;
        auto valid = (score_rows < kTimeAttnN) && (key_cols < kTimeAttnN);
        ScoreTile scores = ct::mma(q_tile,
                                   k_t_view.load_masked(0, kt),
                                   ct::full<ScoreTile>(0.0f));
        auto neg_inf = scores * 0.0f - 3.402823466e38f;
        scores = ct::select(valid, scores * scale, neg_inf);

        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto probs_f32 = ct::select(valid, softmax_exp<UseExp2>(scores - new_m), scores * 0.0f);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load_masked(kt, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    out_acc = out_acc / row_l;
    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_rows = static_cast<long long>(q_block) * QRows + out_local / kTimeAttnD;
    auto out_cols = out_local % kTimeAttnD;
    auto out_valid = out_rows < kTimeAttnN;
    auto safe_rows = ct::select(out_valid, out_rows, out_rows * 0LL);
    ct::store_masked(out_batch + safe_rows * kTimeAttnD + out_cols,
                     ct::element_cast<__nv_bfloat16>(out_acc),
                     out_valid);
}

template <int QRows, int KTile, bool UseExp2 = false>
__tile_global__ void time_attention1301_main1280_cutile_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int FullKTiles = kTimeAttnMainN / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kTimeAttnD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const std::size_t batch_offset =
        static_cast<std::size_t>(bh) * kTimeAttnN * kTimeAttnD;
    const __nv_bfloat16* q_batch = q + batch_offset;
    const __nv_bfloat16* k_batch = k + batch_offset;
    const __nv_bfloat16* v_batch = v + batch_offset;
    __nv_bfloat16* out_batch = out + batch_offset;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kTimeAttnN, kTimeAttnD>{}},
        ct::shape<QRows, kTimeAttnD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kTimeAttnD, kTimeAttnN>{}, ct::layout_left{}},
        ct::shape<kTimeAttnD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kTimeAttnN, kTimeAttnD>{}},
        ct::shape<KTile, kTimeAttnD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kTimeAttnN, kTimeAttnD>{}},
        ct::shape<QRows, kTimeAttnD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto probs_f32 = softmax_exp<UseExp2>(scores - new_m);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
    auto valid = key_cols < kTimeAttnN;
    auto scores = ct::mma(q_tile,
                          k_t_view.load_masked(0, FullKTiles),
                          ct::full<ScoreTile>(0.0f));
    auto neg_inf = scores * 0.0f - 3.402823466e38f;
    scores = ct::select(valid, scores * scale, neg_inf);
    auto tile_m = ct::reduce_max<1>(scores);
    auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
    auto alpha = softmax_exp<UseExp2>(row_m - new_m);
    auto probs_f32 = ct::select(valid, softmax_exp<UseExp2>(scores - new_m), scores * 0.0f);
    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
    auto tile_l = ct::sum<1>(probs_f32);
    out_acc = out_acc * alpha +
              ct::mma(probs_bf16,
                      v_view.load_masked(FullKTiles, 0),
                      ct::full<OutTile>(0.0f));
    row_l = row_l * alpha + tile_l;

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc / row_l), q_block, 0);
}

template <int QRows,
          int KTile,
          bool UseExp2 = false,
          bool IncludeKeyTail = true,
          int Prob = kProbExp,
          int Alpha = kAlphaExact,
          bool WriteStats = false>
static __tile__ void time_attention1301_main1280_split_contig_input_body(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ stats,
    float scale,
    std::size_t q_block,
    std::size_t bh_raw) {
    constexpr int FullKTiles = kTimeAttnMainN / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kTimeAttnD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using I64RowTile = ct::tile<long long, ct::shape<QRows, 1>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;
    using NDShape = ct::shape<kTimeAttnN, kTimeAttnD>;
    using DNShape = ct::shape<kTimeAttnD, kTimeAttnN>;
    using NDStrides = ct::shape<kTimeAttnHeadStride, 1>;
    using DNStrides = ct::shape<1, kTimeAttnHeadStride>;
    using NDLayout = ct::layout_strided<NDStrides>;
    using DNLayout = ct::layout_strided<DNStrides>;
    using NDMapping = typename NDLayout::template mapping<NDShape>;
    using DNMapping = typename DNLayout::template mapping<DNShape>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    int bh = static_cast<int>(bh_raw);
    int b = bh / kTimeAttnHeads;
    int h = bh - b * kTimeAttnHeads;
    const std::size_t split_base =
        (static_cast<std::size_t>(b) * kTimeAttnN * kTimeAttnHeads + h) *
        kTimeAttnD;
    const __nv_bfloat16* q_batch = q + split_base;
    const __nv_bfloat16* k_batch = k + split_base;
    const __nv_bfloat16* v_batch = v + split_base;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kTimeAttnN * kTimeAttnD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<QRows, kTimeAttnD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, DNMapping{DNShape{}, DNStrides{}}},
        ct::shape<kTimeAttnD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<KTile, kTimeAttnD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kTimeAttnN, kTimeAttnD>{}},
        ct::shape<QRows, kTimeAttnD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_alpha<UseExp2, Prob, Alpha>(row_m - new_m);
        auto probs_f32 = softmax_probs_from_scores<UseExp2, Prob>(scores, new_m);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    if constexpr (IncludeKeyTail) {
        I64ScoreTile score_local = ct::iota<I64ScoreTile>();
        auto key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
        auto valid = key_cols < kTimeAttnN;
        auto scores = ct::mma(q_tile,
                              k_t_view.load_masked(0, FullKTiles),
                              ct::full<ScoreTile>(0.0f));
        auto neg_inf = scores * 0.0f - 3.402823466e38f;
        scores = ct::select(valid, scores * scale, neg_inf);
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_alpha<UseExp2, Prob, Alpha>(row_m - new_m);
        auto probs_f32 = ct::select(valid,
                                    softmax_probs_from_scores<UseExp2, Prob>(scores, new_m),
                                    scores * 0.0f);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);
        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load_masked(FullKTiles, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
    }

    auto out_norm = out_acc / row_l;
    if constexpr (WriteStats) {
        I64RowTile local_row = ct::iota<I64RowTile>();
        auto stat_row =
            (static_cast<long long>(bh) * (kTimeAttnMainN / QRows) +
             static_cast<long long>(q_block)) * QRows + local_row;
        auto out_l2 = ct::sum<1>(out_norm * out_norm);
        auto out_abs = ct::select(out_norm < 0.0f, out_norm * -1.0f, out_norm);
        auto out_max_abs = ct::reduce_max<1>(out_abs);
        ct::store(stats + stat_row * kTimeAttnStatsCols + 0, row_m);
        ct::store(stats + stat_row * kTimeAttnStatsCols + 1, row_l);
        ct::store(stats + stat_row * kTimeAttnStatsCols + 2, out_l2);
        ct::store(stats + stat_row * kTimeAttnStatsCols + 3, out_max_abs);
    }

    out_view.store(ct::element_cast<__nv_bfloat16>(out_norm), q_block, 0);
}

template <int QRows,
          int KTile,
          bool UseExp2 = false,
          bool IncludeKeyTail = true,
          int Prob = kProbExp,
          int Alpha = kAlphaExact>
__tile_global__ void time_attention1301_main1280_split_contig_input_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    auto [q_block, bh_raw, tile_z] = ct::bid();
    (void)tile_z;
    time_attention1301_main1280_split_contig_input_body<QRows,
                                                        KTile,
                                                        UseExp2,
                                                        IncludeKeyTail,
                                                        Prob,
                                                        Alpha,
                                                        false>(
        q, k, v, out, nullptr, scale,
        static_cast<std::size_t>(q_block),
        static_cast<std::size_t>(bh_raw));
}

template <int QRows,
          int KTile,
          bool UseExp2 = false,
          bool IncludeKeyTail = true,
          int Prob = kProbExp,
          int Alpha = kAlphaExact>
__tile_global__ void time_attention1301_main1280_split_contig_input_stats_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ stats,
    float scale) {
    auto [q_block, bh_raw, tile_z] = ct::bid();
    (void)tile_z;
    time_attention1301_main1280_split_contig_input_body<QRows,
                                                        KTile,
                                                        UseExp2,
                                                        IncludeKeyTail,
                                                        Prob,
                                                        Alpha,
                                                        true>(
        q, k, v, out, stats, scale,
        static_cast<std::size_t>(q_block),
        static_cast<std::size_t>(bh_raw));
}

template <int QRows,
          int KTile,
          bool UseExp2 = false,
          bool IncludeKeyTail = true,
          typename TrigT = float>
__tile_global__ void time_attention1301_main1280_split_contig_qrot_input_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const TrigT* __restrict__ cos_f,
    const TrigT* __restrict__ sin_f,
    __nv_bfloat16* __restrict__ out,
    float scale,
    bool full_bf16) {
    constexpr int FullKTiles = kTimeAttnMainN / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kTimeAttnD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;
    using NDShape = ct::shape<kTimeAttnN, kTimeAttnD>;
    using DNShape = ct::shape<kTimeAttnD, kTimeAttnN>;
    using NDStrides = ct::shape<kTimeAttnHeadStride, 1>;
    using DNStrides = ct::shape<1, kTimeAttnHeadStride>;
    using NDLayout = ct::layout_strided<NDStrides>;
    using DNLayout = ct::layout_strided<DNStrides>;
    using NDMapping = typename NDLayout::template mapping<NDShape>;
    using DNMapping = typename DNLayout::template mapping<DNShape>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    cos_f = ct::assume_aligned(cos_f, 16_ic);
    sin_f = ct::assume_aligned(sin_f, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh_raw, tile_z] = ct::bid();
    (void)tile_z;
    int bh = static_cast<int>(bh_raw);
    int b = bh / kTimeAttnHeads;
    int h = bh - b * kTimeAttnHeads;
    const std::size_t split_base =
        (static_cast<std::size_t>(b) * kTimeAttnN * kTimeAttnHeads + h) *
        kTimeAttnD;
    const __nv_bfloat16* q_batch = q + split_base;
    const __nv_bfloat16* k_batch = k + split_base;
    const __nv_bfloat16* v_batch = v + split_base;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kTimeAttnN * kTimeAttnD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<QRows, kTimeAttnD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, DNMapping{DNShape{}, DNStrides{}}},
        ct::shape<kTimeAttnD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<KTile, kTimeAttnD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kTimeAttnN, kTimeAttnD>{}},
        ct::shape<QRows, kTimeAttnD>{}
    };

    auto q_tile = rotate_q_tile_for_attention<QRows>(
        q_view.load(q_block, 0), cos_f, sin_f, static_cast<long long>(q_block), full_bf16);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto probs_f32 = softmax_exp<UseExp2>(scores - new_m);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    if constexpr (IncludeKeyTail) {
        I64ScoreTile score_local = ct::iota<I64ScoreTile>();
        auto key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
        auto valid = key_cols < kTimeAttnN;
        auto scores = ct::mma(q_tile,
                              k_t_view.load_masked(0, FullKTiles),
                              ct::full<ScoreTile>(0.0f));
        auto neg_inf = scores * 0.0f - 3.402823466e38f;
        scores = ct::select(valid, scores * scale, neg_inf);
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto probs_f32 = ct::select(valid, softmax_exp<UseExp2>(scores - new_m), scores * 0.0f);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);
        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load_masked(FullKTiles, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
    }

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc / row_l), q_block, 0);
}

template <int QRows, int KTile, int QBlockOffset, bool UseExp2 = false>
__tile_global__ void time_attention1301_split_contig_tail_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int KTiles = (kTimeAttnN + KTile - 1) / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kTimeAttnD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kTimeAttnD>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;
    using NDShape = ct::shape<kTimeAttnN, kTimeAttnD>;
    using DNShape = ct::shape<kTimeAttnD, kTimeAttnN>;
    using NDStrides = ct::shape<kTimeAttnHeadStride, 1>;
    using DNStrides = ct::shape<1, kTimeAttnHeadStride>;
    using NDLayout = ct::layout_strided<NDStrides>;
    using DNLayout = ct::layout_strided<DNStrides>;
    using NDMapping = typename NDLayout::template mapping<NDShape>;
    using DNMapping = typename DNLayout::template mapping<DNShape>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block_local, bh_raw, tile_z] = ct::bid();
    (void)tile_z;
    auto q_block = q_block_local + QBlockOffset;
    int bh = static_cast<int>(bh_raw);
    int b = bh / kTimeAttnHeads;
    int h = bh - b * kTimeAttnHeads;
    const std::size_t split_base =
        (static_cast<std::size_t>(b) * kTimeAttnN * kTimeAttnHeads + h) *
        kTimeAttnD;
    const __nv_bfloat16* q_batch = q + split_base;
    const __nv_bfloat16* k_batch = k + split_base;
    const __nv_bfloat16* v_batch = v + split_base;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kTimeAttnN * kTimeAttnD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<QRows, kTimeAttnD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, DNMapping{DNShape{}, DNStrides{}}},
        ct::shape<kTimeAttnD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<KTile, kTimeAttnD>{}
    };

    auto q_tile = q_view.load_masked(q_block, 0);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto score_rows =
        static_cast<long long>(q_block) * QRows + score_local / KTile;
    auto score_cols_local = score_local % KTile;

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{KTiles})) {
        auto key_cols =
            static_cast<long long>(kt) * KTile + score_cols_local;
        auto valid = (score_rows < kTimeAttnN) && (key_cols < kTimeAttnN);
        auto scores = ct::mma(q_tile,
                              k_t_view.load_masked(0, kt),
                              ct::full<ScoreTile>(0.0f));
        auto neg_inf = scores * 0.0f - 3.402823466e38f;
        scores = ct::select(valid, scores * scale, neg_inf);

        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto probs_f32 = ct::select(valid, softmax_exp<UseExp2>(scores - new_m), scores * 0.0f);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load_masked(kt, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    out_acc = out_acc / row_l;
    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_rows = static_cast<long long>(q_block) * QRows + out_local / kTimeAttnD;
    auto out_cols = out_local % kTimeAttnD;
    auto out_valid = out_rows < kTimeAttnN;
    auto safe_rows = ct::select(out_valid, out_rows, out_rows * 0LL);
    ct::store_masked(out_batch + safe_rows * kTimeAttnD + out_cols,
                     ct::element_cast<__nv_bfloat16>(out_acc),
                     out_valid);
}

template <int QRows,
          int KTile,
          int QBlockOffset,
          bool UseExp2 = false,
          typename TrigT = float>
__tile_global__ void time_attention1301_split_contig_qrot_tail_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const TrigT* __restrict__ cos_f,
    const TrigT* __restrict__ sin_f,
    __nv_bfloat16* __restrict__ out,
    float scale,
    bool full_bf16) {
    constexpr int KTiles = (kTimeAttnN + KTile - 1) / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kTimeAttnD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kTimeAttnD>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;
    using NDShape = ct::shape<kTimeAttnN, kTimeAttnD>;
    using DNShape = ct::shape<kTimeAttnD, kTimeAttnN>;
    using NDStrides = ct::shape<kTimeAttnHeadStride, 1>;
    using DNStrides = ct::shape<1, kTimeAttnHeadStride>;
    using NDLayout = ct::layout_strided<NDStrides>;
    using DNLayout = ct::layout_strided<DNStrides>;
    using NDMapping = typename NDLayout::template mapping<NDShape>;
    using DNMapping = typename DNLayout::template mapping<DNShape>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    cos_f = ct::assume_aligned(cos_f, 16_ic);
    sin_f = ct::assume_aligned(sin_f, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block_local, bh_raw, tile_z] = ct::bid();
    (void)tile_z;
    auto q_block = q_block_local + QBlockOffset;
    int bh = static_cast<int>(bh_raw);
    int b = bh / kTimeAttnHeads;
    int h = bh - b * kTimeAttnHeads;
    const std::size_t split_base =
        (static_cast<std::size_t>(b) * kTimeAttnN * kTimeAttnHeads + h) *
        kTimeAttnD;
    const __nv_bfloat16* q_batch = q + split_base;
    const __nv_bfloat16* k_batch = k + split_base;
    const __nv_bfloat16* v_batch = v + split_base;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kTimeAttnN * kTimeAttnD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<QRows, kTimeAttnD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, DNMapping{DNShape{}, DNStrides{}}},
        ct::shape<kTimeAttnD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<KTile, kTimeAttnD>{}
    };

    auto q_tile = rotate_q_tile_for_attention<QRows>(
        q_view.load_masked(q_block, 0), cos_f, sin_f, static_cast<long long>(q_block), full_bf16);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto score_rows =
        static_cast<long long>(q_block) * QRows + score_local / KTile;
    auto score_cols_local = score_local % KTile;

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{KTiles})) {
        auto key_cols =
            static_cast<long long>(kt) * KTile + score_cols_local;
        auto valid = (score_rows < kTimeAttnN) && (key_cols < kTimeAttnN);
        auto scores = ct::mma(q_tile,
                              k_t_view.load_masked(0, kt),
                              ct::full<ScoreTile>(0.0f));
        auto neg_inf = scores * 0.0f - 3.402823466e38f;
        scores = ct::select(valid, scores * scale, neg_inf);

        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto probs_f32 = ct::select(valid, softmax_exp<UseExp2>(scores - new_m), scores * 0.0f);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load_masked(kt, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    out_acc = out_acc / row_l;
    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_rows = static_cast<long long>(q_block) * QRows + out_local / kTimeAttnD;
    auto out_cols = out_local % kTimeAttnD;
    auto out_valid = out_rows < kTimeAttnN;
    auto safe_rows = ct::select(out_valid, out_rows, out_rows * 0LL);
    ct::store_masked(out_batch + safe_rows * kTimeAttnD + out_cols,
                     ct::element_cast<__nv_bfloat16>(out_acc),
                     out_valid);
}

template <typename TrigT>
void launch_qrot_main_typed(const Tensor& q,
                            const Tensor& k,
                            const Tensor& v,
                            const Tensor& cos_freqs,
                            const Tensor& sin_freqs,
                            Tensor& out,
                            float scale,
                            bool full_bf16,
                            bool use_q32,
                            bool use_exp2,
                            bool skip_keytail,
                            dim3 grid_q64,
                            dim3 grid_q32) {
    const auto* cos = static_cast<const TrigT*>(cos_freqs.data_ptr());
    const auto* sin = static_cast<const TrigT*>(sin_freqs.data_ptr());
    if (use_q32 && use_exp2 && skip_keytail) {
        time_attention1301_main1280_split_contig_qrot_input_kernel<
            kTimeAttnCutileQRows32,
            kTimeAttnCutileKTile32,
            true,
            false,
            TrigT><<<grid_q32, 1>>>(
            q.data_bf16(), k.data_bf16(), v.data_bf16(), cos, sin,
            out.data_bf16(), scale, full_bf16);
    } else if (use_q32 && skip_keytail) {
        time_attention1301_main1280_split_contig_qrot_input_kernel<
            kTimeAttnCutileQRows32,
            kTimeAttnCutileKTile32,
            false,
            false,
            TrigT><<<grid_q32, 1>>>(
            q.data_bf16(), k.data_bf16(), v.data_bf16(), cos, sin,
            out.data_bf16(), scale, full_bf16);
    } else if (use_exp2 && skip_keytail) {
        time_attention1301_main1280_split_contig_qrot_input_kernel<
            kTimeAttnCutileQRows64,
            kTimeAttnCutileKTile32,
            true,
            false,
            TrigT><<<grid_q64, 1>>>(
            q.data_bf16(), k.data_bf16(), v.data_bf16(), cos, sin,
            out.data_bf16(), scale, full_bf16);
    } else if (skip_keytail) {
        time_attention1301_main1280_split_contig_qrot_input_kernel<
            kTimeAttnCutileQRows64,
            kTimeAttnCutileKTile32,
            false,
            false,
            TrigT><<<grid_q64, 1>>>(
            q.data_bf16(), k.data_bf16(), v.data_bf16(), cos, sin,
            out.data_bf16(), scale, full_bf16);
    } else if (use_q32 && use_exp2) {
        time_attention1301_main1280_split_contig_qrot_input_kernel<
            kTimeAttnCutileQRows32,
            kTimeAttnCutileKTile32,
            true,
            true,
            TrigT><<<grid_q32, 1>>>(
            q.data_bf16(), k.data_bf16(), v.data_bf16(), cos, sin,
            out.data_bf16(), scale, full_bf16);
    } else if (use_q32) {
        time_attention1301_main1280_split_contig_qrot_input_kernel<
            kTimeAttnCutileQRows32,
            kTimeAttnCutileKTile32,
            false,
            true,
            TrigT><<<grid_q32, 1>>>(
            q.data_bf16(), k.data_bf16(), v.data_bf16(), cos, sin,
            out.data_bf16(), scale, full_bf16);
    } else if (use_exp2) {
        time_attention1301_main1280_split_contig_qrot_input_kernel<
            kTimeAttnCutileQRows64,
            kTimeAttnCutileKTile32,
            true,
            true,
            TrigT><<<grid_q64, 1>>>(
            q.data_bf16(), k.data_bf16(), v.data_bf16(), cos, sin,
            out.data_bf16(), scale, full_bf16);
    } else {
        time_attention1301_main1280_split_contig_qrot_input_kernel<
            kTimeAttnCutileQRows64,
            kTimeAttnCutileKTile32,
            false,
            true,
            TrigT><<<grid_q64, 1>>>(
            q.data_bf16(), k.data_bf16(), v.data_bf16(), cos, sin,
            out.data_bf16(), scale, full_bf16);
    }
}

template <typename TrigT>
void launch_qrot_tail_typed(const Tensor& q,
                            const Tensor& k,
                            const Tensor& v,
                            const Tensor& cos_freqs,
                            const Tensor& sin_freqs,
                            Tensor& out,
                            float scale,
                            bool full_bf16,
                            bool use_tail_q32,
                            bool use_exp2,
                            dim3 grid_tail_q64,
                            dim3 grid_tail_q32) {
    const auto* cos = static_cast<const TrigT*>(cos_freqs.data_ptr());
    const auto* sin = static_cast<const TrigT*>(sin_freqs.data_ptr());
    if (use_tail_q32 && use_exp2) {
        time_attention1301_split_contig_qrot_tail_kernel<
            kTimeAttnCutileQRows32,
            kTimeAttnCutileKTile32,
            40,
            true,
            TrigT><<<grid_tail_q32, 1>>>(
            q.data_bf16(), k.data_bf16(), v.data_bf16(), cos, sin,
            out.data_bf16(), scale, full_bf16);
    } else if (use_tail_q32) {
        time_attention1301_split_contig_qrot_tail_kernel<
            kTimeAttnCutileQRows32,
            kTimeAttnCutileKTile32,
            40,
            false,
            TrigT><<<grid_tail_q32, 1>>>(
            q.data_bf16(), k.data_bf16(), v.data_bf16(), cos, sin,
            out.data_bf16(), scale, full_bf16);
    } else if (use_exp2) {
        time_attention1301_split_contig_qrot_tail_kernel<
            kTimeAttnCutileQRows64,
            kTimeAttnCutileKTile64,
            20,
            true,
            TrigT><<<grid_tail_q64, 1>>>(
            q.data_bf16(), k.data_bf16(), v.data_bf16(), cos, sin,
            out.data_bf16(), scale, full_bf16);
    } else {
        time_attention1301_split_contig_qrot_tail_kernel<
            kTimeAttnCutileQRows64,
            kTimeAttnCutileKTile64,
            20,
            false,
            TrigT><<<grid_tail_q64, 1>>>(
            q.data_bf16(), k.data_bf16(), v.data_bf16(), cos, sin,
            out.data_bf16(), scale, full_bf16);
    }
}

}  // namespace

void launch_time_attention1301_split_tail_cutile(const Tensor& q,
                                                 const Tensor& k,
                                                 const Tensor& v,
                                                 Tensor& out,
                                                 int64_t bh,
                                                 float scale,
                                                 bool use_k32,
                                                 bool use_tail_q32,
                                                 bool use_exp2) {
    dim3 grid_main(kTimeAttnMainN / kTimeAttnCutileQRows64, static_cast<unsigned int>(bh));
    dim3 grid_tail_q64(1, static_cast<unsigned int>(bh));
    dim3 grid_tail_q32(
        static_cast<unsigned int>(ceildiv(kTimeAttnN - kTimeAttnMainN,
                                          kTimeAttnCutileQRows32)),
        static_cast<unsigned int>(bh));

    if (use_k32) {
        if (use_exp2) {
            time_attention1301_main1280_cutile_kernel<kTimeAttnCutileQRows64,
                                                      kTimeAttnCutileKTile32,
                                                      true>
                <<<grid_main, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                                   out.data_bf16(), scale);
        } else {
            time_attention1301_main1280_cutile_kernel<kTimeAttnCutileQRows64,
                                                      kTimeAttnCutileKTile32>
                <<<grid_main, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                                   out.data_bf16(), scale);
        }
    } else {
        if (use_exp2) {
            time_attention1301_main1280_cutile_kernel<kTimeAttnCutileQRows64,
                                                      kTimeAttnCutileKTile64,
                                                      true>
                <<<grid_main, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                                   out.data_bf16(), scale);
        } else {
            time_attention1301_main1280_cutile_kernel<kTimeAttnCutileQRows64,
                                                      kTimeAttnCutileKTile64>
                <<<grid_main, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                                   out.data_bf16(), scale);
        }
    }

    if (use_tail_q32) {
        if (use_exp2) {
            time_attention1301_cutile_qk_softmax_av_kernel<kTimeAttnCutileQRows32,
                                                           kTimeAttnCutileKTile32,
                                                           40,
                                                           true>
                <<<grid_tail_q32, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                                       out.data_bf16(), scale);
        } else {
            time_attention1301_cutile_qk_softmax_av_kernel<kTimeAttnCutileQRows32,
                                                           kTimeAttnCutileKTile32,
                                                           40>
                <<<grid_tail_q32, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                                       out.data_bf16(), scale);
        }
    } else if (use_exp2) {
        time_attention1301_cutile_qk_softmax_av_kernel<kTimeAttnCutileQRows64,
                                                       kTimeAttnCutileKTile64,
                                                       20,
                                                       true>
            <<<grid_tail_q64, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                                   out.data_bf16(), scale);
    } else {
        time_attention1301_cutile_qk_softmax_av_kernel<kTimeAttnCutileQRows64,
                                                       kTimeAttnCutileKTile64,
                                                       20>
            <<<grid_tail_q64, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                                   out.data_bf16(), scale);
    }
    CUDA_CHECK(cudaGetLastError());
}

void print_time_attention_stats(const Tensor& stats,
                                int64_t rows,
                                bool approx_softmax,
                                bool clamp_prob,
                                bool poly2_l4,
                                bool use_exp2,
                                bool skip_keytail) {
    std::vector<float> values = stats.to_cpu_f32();
    double sum[kTimeAttnStatsCols] = {};
    float min_value[kTimeAttnStatsCols];
    float max_value[kTimeAttnStatsCols];
    for (int c = 0; c < kTimeAttnStatsCols; ++c) {
        min_value[c] = std::numeric_limits<float>::infinity();
        max_value[c] = -std::numeric_limits<float>::infinity();
    }
    for (int64_t r = 0; r < rows; ++r) {
        for (int c = 0; c < kTimeAttnStatsCols; ++c) {
            float value = values[(std::size_t)r * kTimeAttnStatsCols + c];
            sum[c] += value;
            min_value[c] = std::min(min_value[c], value);
            max_value[c] = std::max(max_value[c], value);
        }
    }
    double inv_rows = rows > 0 ? 1.0 / static_cast<double>(rows) : 0.0;
    std::cerr
        << "[time-attn-stats] approx=" << (approx_softmax ? 1 : 0)
        << " clamp_prob=" << (clamp_prob ? 1 : 0)
        << " poly2_l4=" << (poly2_l4 ? 1 : 0)
        << " use_exp2=" << (use_exp2 ? 1 : 0)
        << " skip_keytail=" << (skip_keytail ? 1 : 0)
        << " rows=" << rows
        << " row_m_min=" << min_value[0]
        << " row_m_max=" << max_value[0]
        << " row_m_mean=" << (sum[0] * inv_rows)
        << " row_l_min=" << min_value[1]
        << " row_l_max=" << max_value[1]
        << " row_l_mean=" << (sum[1] * inv_rows)
        << " out_l2_min=" << min_value[2]
        << " out_l2_max=" << max_value[2]
        << " out_l2_mean=" << (sum[2] * inv_rows)
        << " out_max_abs_min=" << min_value[3]
        << " out_max_abs_max=" << max_value[3]
        << " out_max_abs_mean=" << (sum[3] * inv_rows)
        << '\n';
}

template <bool UseExp2, bool IncludeKeyTail, int Prob, int Alpha, bool WriteStats>
void launch_time_attention1301_split_contig_main_typed(const Tensor& q,
                                                       const Tensor& k,
                                                       const Tensor& v,
                                                       Tensor& out,
                                                       float* stats_ptr,
                                                       float scale,
                                                       dim3 grid) {
    if constexpr (WriteStats) {
        time_attention1301_main1280_split_contig_input_stats_kernel<kTimeAttnCutileQRows64,
                                                                    kTimeAttnCutileKTile32,
                                                                    UseExp2,
                                                                    IncludeKeyTail,
                                                                    Prob,
                                                                    Alpha>
            <<<grid, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                          out.data_bf16(), stats_ptr, scale);
    } else {
        (void)stats_ptr;
        time_attention1301_main1280_split_contig_input_kernel<kTimeAttnCutileQRows64,
                                                              kTimeAttnCutileKTile32,
                                                              UseExp2,
                                                              IncludeKeyTail,
                                                              Prob,
                                                              Alpha>
            <<<grid, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                          out.data_bf16(), scale);
    }
}

template <int Prob, bool WriteStats>
void launch_time_attention1301_split_contig_main_approx_prob(const Tensor& q,
                                                            const Tensor& k,
                                                            const Tensor& v,
                                                            Tensor& out,
                                                            float* stats_ptr,
                                                            float scale,
                                                            dim3 grid,
                                                            bool use_exp2,
                                                            bool skip_keytail) {
    if (skip_keytail && use_exp2) {
        launch_time_attention1301_split_contig_main_typed<true,
                                                          false,
                                                          Prob,
                                                          kAlphaProbClamp,
                                                          WriteStats>(
            q, k, v, out, stats_ptr, scale, grid);
    } else if (skip_keytail) {
        launch_time_attention1301_split_contig_main_typed<false,
                                                          false,
                                                          Prob,
                                                          kAlphaProbClamp,
                                                          WriteStats>(
            q, k, v, out, stats_ptr, scale, grid);
    } else if (use_exp2) {
        launch_time_attention1301_split_contig_main_typed<true,
                                                          true,
                                                          Prob,
                                                          kAlphaProbClamp,
                                                          WriteStats>(
            q, k, v, out, stats_ptr, scale, grid);
    } else {
        launch_time_attention1301_split_contig_main_typed<false,
                                                          true,
                                                          Prob,
                                                          kAlphaProbClamp,
                                                          WriteStats>(
            q, k, v, out, stats_ptr, scale, grid);
    }
}

template <bool WriteStats>
void launch_time_attention1301_split_contig_main_exact(const Tensor& q,
                                                       const Tensor& k,
                                                       const Tensor& v,
                                                       Tensor& out,
                                                       float* stats_ptr,
                                                       float scale,
                                                       dim3 grid,
                                                       bool use_exp2,
                                                       bool skip_keytail) {
    if (skip_keytail && use_exp2) {
        launch_time_attention1301_split_contig_main_typed<true,
                                                          false,
                                                          kProbExp,
                                                          kAlphaExact,
                                                          WriteStats>(
            q, k, v, out, stats_ptr, scale, grid);
    } else if (skip_keytail) {
        launch_time_attention1301_split_contig_main_typed<false,
                                                          false,
                                                          kProbExp,
                                                          kAlphaExact,
                                                          WriteStats>(
            q, k, v, out, stats_ptr, scale, grid);
    } else if (use_exp2) {
        launch_time_attention1301_split_contig_main_typed<true,
                                                          true,
                                                          kProbExp,
                                                          kAlphaExact,
                                                          WriteStats>(
            q, k, v, out, stats_ptr, scale, grid);
    } else {
        launch_time_attention1301_split_contig_main_typed<false,
                                                          true,
                                                          kProbExp,
                                                          kAlphaExact,
                                                          WriteStats>(
            q, k, v, out, stats_ptr, scale, grid);
    }
}

void launch_time_attention1301_split_contig_main_cutile(const Tensor& q,
                                                        const Tensor& k,
                                                        const Tensor& v,
                                                        Tensor& out,
                                                        float scale,
                                                        bool use_exp2,
                                                        bool skip_keytail,
                                                        bool approx_softmax) {
    if (q.dtype() != DType::BFloat16 || k.dtype() != DType::BFloat16 ||
        v.dtype() != DType::BFloat16 || out.dtype() != DType::BFloat16) {
        throw std::runtime_error("time split-contig CUDA Tile: expected BF16 tensors");
    }
    if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 ||
        q.size(1) != kTimeAttnN || q.size(2) != kTimeAttnHeads ||
        q.size(3) != kTimeAttnD ||
        k.shape() != q.shape() || v.shape() != q.shape()) {
        throw std::runtime_error(
            "time split-contig CUDA Tile: expected q/k/v [B,1301,8,64]");
    }
    int64_t batches = q.size(0);
    int64_t bh = batches * kTimeAttnHeads;
    if (out.ndim() != 3 || out.size(0) != bh || out.size(1) != kTimeAttnN ||
        out.size(2) != kTimeAttnD) {
        throw std::runtime_error(
            "time split-contig CUDA Tile: expected out [B*8,1301,64]");
    }
    if (batches <= 0 || bh > std::numeric_limits<unsigned int>::max()) {
        throw std::runtime_error("time split-contig CUDA Tile: invalid batch count");
    }

    dim3 grid(kTimeAttnMainN / kTimeAttnCutileQRows64,
              static_cast<unsigned int>(bh));
    bool collect_stats = time_attention_stats_enabled_for_current_context();
    int64_t stats_rows =
        bh * (kTimeAttnMainN / kTimeAttnCutileQRows64) * kTimeAttnCutileQRows64;
    Tensor stats;
    float* stats_ptr = nullptr;
    if (collect_stats) {
        stats = Tensor::empty({stats_rows, kTimeAttnStatsCols}, DType::Float32);
        stats_ptr = stats.data_f32();
    }
    bool clamp_prob = approx_softmax && time_attention_approx_clamp_prob_enabled();
    bool poly2_l4 = approx_softmax && time_attention_approx_poly2_l4_enabled();
    if (collect_stats) {
        if (poly2_l4) {
            launch_time_attention1301_split_contig_main_approx_prob<kProbPoly2NoClampL4, true>(
                q, k, v, out, stats_ptr, scale, grid, use_exp2, skip_keytail);
        } else if (clamp_prob) {
            launch_time_attention1301_split_contig_main_approx_prob<kProbPoly3Clamp, true>(
                q, k, v, out, stats_ptr, scale, grid, use_exp2, skip_keytail);
        } else if (approx_softmax) {
            launch_time_attention1301_split_contig_main_approx_prob<kProbPoly3NoClamp, true>(
                q, k, v, out, stats_ptr, scale, grid, use_exp2, skip_keytail);
        } else {
            launch_time_attention1301_split_contig_main_exact<true>(
                q, k, v, out, stats_ptr, scale, grid, use_exp2, skip_keytail);
        }
    } else {
        if (poly2_l4) {
            launch_time_attention1301_split_contig_main_approx_prob<kProbPoly2NoClampL4, false>(
                q, k, v, out, stats_ptr, scale, grid, use_exp2, skip_keytail);
        } else if (clamp_prob) {
            launch_time_attention1301_split_contig_main_approx_prob<kProbPoly3Clamp, false>(
                q, k, v, out, stats_ptr, scale, grid, use_exp2, skip_keytail);
        } else if (approx_softmax) {
            launch_time_attention1301_split_contig_main_approx_prob<kProbPoly3NoClamp, false>(
                q, k, v, out, stats_ptr, scale, grid, use_exp2, skip_keytail);
        } else {
            launch_time_attention1301_split_contig_main_exact<false>(
                q, k, v, out, stats_ptr, scale, grid, use_exp2, skip_keytail);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    if (collect_stats) {
        print_time_attention_stats(stats, stats_rows, approx_softmax, clamp_prob, poly2_l4,
                                   use_exp2, skip_keytail);
    }
}

void launch_time_attention1301_split_contig_qrot_main_cutile(const Tensor& q,
                                                             const Tensor& k,
                                                             const Tensor& v,
                                                             const Tensor& cos_freqs,
                                                             const Tensor& sin_freqs,
                                                             Tensor& out,
                                                             float scale,
                                                             bool full_bf16,
                                                             bool use_q32,
                                                             bool use_exp2,
                                                             bool skip_keytail) {
    if (q.dtype() != DType::BFloat16 || k.dtype() != DType::BFloat16 ||
        v.dtype() != DType::BFloat16 || out.dtype() != DType::BFloat16 ||
        cos_freqs.dtype() != sin_freqs.dtype() ||
        (cos_freqs.dtype() != DType::Float32 &&
         cos_freqs.dtype() != DType::BFloat16)) {
        throw std::runtime_error(
            "time split-contig qrot CUDA Tile: expected BF16 q/k/v/out and F32 or BF16 cos/sin");
    }
    if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 ||
        q.size(1) != kTimeAttnN || q.size(2) != kTimeAttnHeads ||
        q.size(3) != kTimeAttnD ||
        k.shape() != q.shape() || v.shape() != q.shape()) {
        throw std::runtime_error(
            "time split-contig qrot CUDA Tile: expected q/k/v [B,1301,8,64]");
    }
    if (cos_freqs.ndim() != 2 || sin_freqs.ndim() != 2 ||
        cos_freqs.size(0) != kTimeAttnN || sin_freqs.size(0) != kTimeAttnN ||
        cos_freqs.size(1) != kTimeAttnD / 2 || sin_freqs.size(1) != kTimeAttnD / 2) {
        throw std::runtime_error(
            "time split-contig qrot CUDA Tile: expected cos/sin [1301,32]");
    }
    int64_t batches = q.size(0);
    int64_t bh = batches * kTimeAttnHeads;
    if (out.ndim() != 3 || out.size(0) != bh || out.size(1) != kTimeAttnN ||
        out.size(2) != kTimeAttnD) {
        throw std::runtime_error(
            "time split-contig qrot CUDA Tile: expected out [B*8,1301,64]");
    }
    if (batches <= 0 || bh > std::numeric_limits<unsigned int>::max()) {
        throw std::runtime_error("time split-contig qrot CUDA Tile: invalid batch count");
    }

    dim3 grid_q64(kTimeAttnMainN / kTimeAttnCutileQRows64,
                  static_cast<unsigned int>(bh));
    dim3 grid_q32(kTimeAttnMainN / kTimeAttnCutileQRows32,
                  static_cast<unsigned int>(bh));
    if (cos_freqs.dtype() == DType::BFloat16) {
        launch_qrot_main_typed<__nv_bfloat16>(
            q, k, v, cos_freqs, sin_freqs, out, scale, full_bf16,
            use_q32, use_exp2, skip_keytail, grid_q64, grid_q32);
    } else {
        launch_qrot_main_typed<float>(
            q, k, v, cos_freqs, sin_freqs, out, scale, full_bf16,
            use_q32, use_exp2, skip_keytail, grid_q64, grid_q32);
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_time_attention1301_split_contig_tail_cutile(const Tensor& q,
                                                        const Tensor& k,
                                                        const Tensor& v,
                                                        Tensor& out,
                                                        float scale,
                                                        bool use_tail_q32,
                                                        bool use_exp2) {
    if (q.dtype() != DType::BFloat16 || k.dtype() != DType::BFloat16 ||
        v.dtype() != DType::BFloat16 || out.dtype() != DType::BFloat16) {
        throw std::runtime_error("time split-contig tail CUDA Tile: expected BF16 tensors");
    }
    if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 ||
        q.size(1) != kTimeAttnN || q.size(2) != kTimeAttnHeads ||
        q.size(3) != kTimeAttnD ||
        k.shape() != q.shape() || v.shape() != q.shape()) {
        throw std::runtime_error(
            "time split-contig tail CUDA Tile: expected q/k/v [B,1301,8,64]");
    }
    int64_t batches = q.size(0);
    int64_t bh = batches * kTimeAttnHeads;
    if (out.ndim() != 3 || out.size(0) != bh || out.size(1) != kTimeAttnN ||
        out.size(2) != kTimeAttnD) {
        throw std::runtime_error(
            "time split-contig tail CUDA Tile: expected out [B*8,1301,64]");
    }
    if (batches <= 0 || bh > std::numeric_limits<unsigned int>::max()) {
        throw std::runtime_error("time split-contig tail CUDA Tile: invalid batch count");
    }

    dim3 grid_tail_q64(1, static_cast<unsigned int>(bh));
    dim3 grid_tail_q32(
        static_cast<unsigned int>(ceildiv(kTimeAttnN - kTimeAttnMainN,
                                          kTimeAttnCutileQRows32)),
        static_cast<unsigned int>(bh));

    if (use_tail_q32) {
        if (use_exp2) {
            time_attention1301_split_contig_tail_kernel<kTimeAttnCutileQRows32,
                                                        kTimeAttnCutileKTile32,
                                                        40,
                                                        true>
                <<<grid_tail_q32, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                                       out.data_bf16(), scale);
        } else {
            time_attention1301_split_contig_tail_kernel<kTimeAttnCutileQRows32,
                                                        kTimeAttnCutileKTile32,
                                                        40>
                <<<grid_tail_q32, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                                       out.data_bf16(), scale);
        }
    } else if (use_exp2) {
        time_attention1301_split_contig_tail_kernel<kTimeAttnCutileQRows64,
                                                    kTimeAttnCutileKTile64,
                                                    20,
                                                    true>
            <<<grid_tail_q64, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                                   out.data_bf16(), scale);
    } else {
        time_attention1301_split_contig_tail_kernel<kTimeAttnCutileQRows64,
                                                    kTimeAttnCutileKTile64,
                                                    20>
            <<<grid_tail_q64, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                                   out.data_bf16(), scale);
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_time_attention1301_split_contig_qrot_tail_cutile(const Tensor& q,
                                                             const Tensor& k,
                                                             const Tensor& v,
                                                             const Tensor& cos_freqs,
                                                             const Tensor& sin_freqs,
                                                             Tensor& out,
                                                             float scale,
                                                             bool full_bf16,
                                                             bool use_tail_q32,
                                                             bool use_exp2) {
    if (q.dtype() != DType::BFloat16 || k.dtype() != DType::BFloat16 ||
        v.dtype() != DType::BFloat16 || out.dtype() != DType::BFloat16 ||
        cos_freqs.dtype() != sin_freqs.dtype() ||
        (cos_freqs.dtype() != DType::Float32 &&
         cos_freqs.dtype() != DType::BFloat16)) {
        throw std::runtime_error(
            "time split-contig qrot tail CUDA Tile: expected BF16 q/k/v/out and F32 or BF16 cos/sin");
    }
    if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 ||
        q.size(1) != kTimeAttnN || q.size(2) != kTimeAttnHeads ||
        q.size(3) != kTimeAttnD ||
        k.shape() != q.shape() || v.shape() != q.shape()) {
        throw std::runtime_error(
            "time split-contig qrot tail CUDA Tile: expected q/k/v [B,1301,8,64]");
    }
    if (cos_freqs.ndim() != 2 || sin_freqs.ndim() != 2 ||
        cos_freqs.size(0) != kTimeAttnN || sin_freqs.size(0) != kTimeAttnN ||
        cos_freqs.size(1) != kTimeAttnD / 2 || sin_freqs.size(1) != kTimeAttnD / 2) {
        throw std::runtime_error(
            "time split-contig qrot tail CUDA Tile: expected cos/sin [1301,32]");
    }
    int64_t batches = q.size(0);
    int64_t bh = batches * kTimeAttnHeads;
    if (out.ndim() != 3 || out.size(0) != bh || out.size(1) != kTimeAttnN ||
        out.size(2) != kTimeAttnD) {
        throw std::runtime_error(
            "time split-contig qrot tail CUDA Tile: expected out [B*8,1301,64]");
    }
    if (batches <= 0 || bh > std::numeric_limits<unsigned int>::max()) {
        throw std::runtime_error("time split-contig qrot tail CUDA Tile: invalid batch count");
    }

    dim3 grid_tail_q64(1, static_cast<unsigned int>(bh));
    dim3 grid_tail_q32(
        static_cast<unsigned int>(ceildiv(kTimeAttnN - kTimeAttnMainN,
                                          kTimeAttnCutileQRows32)),
        static_cast<unsigned int>(bh));

    if (cos_freqs.dtype() == DType::BFloat16) {
        launch_qrot_tail_typed<__nv_bfloat16>(
            q, k, v, cos_freqs, sin_freqs, out, scale, full_bf16,
            use_tail_q32, use_exp2, grid_tail_q64, grid_tail_q32);
    } else {
        launch_qrot_tail_typed<float>(
            q, k, v, cos_freqs, sin_freqs, out, scale, full_bf16,
            use_tail_q32, use_exp2, grid_tail_q64, grid_tail_q32);
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_time_attention1301_full_cutile(const Tensor& q,
                                           const Tensor& k,
                                           const Tensor& v,
                                           Tensor& out,
                                           int64_t bh,
                                           float scale,
                                           int qrows,
                                           int ktile) {
    if (qrows == kTimeAttnCutileQRows64 && ktile == kTimeAttnCutileKTile128) {
        dim3 grid(static_cast<unsigned int>(ceildiv(kTimeAttnN, kTimeAttnCutileQRows64)),
                  static_cast<unsigned int>(bh));
        time_attention1301_cutile_qk_softmax_av_kernel<kTimeAttnCutileQRows64,
                                                       kTimeAttnCutileKTile128>
            <<<grid, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                          out.data_bf16(), scale);
    } else if (qrows == kTimeAttnCutileQRows64 && ktile == kTimeAttnCutileKTile32) {
        dim3 grid(static_cast<unsigned int>(ceildiv(kTimeAttnN, kTimeAttnCutileQRows64)),
                  static_cast<unsigned int>(bh));
        time_attention1301_cutile_qk_softmax_av_kernel<kTimeAttnCutileQRows64,
                                                       kTimeAttnCutileKTile32>
            <<<grid, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                          out.data_bf16(), scale);
    } else if (qrows == kTimeAttnCutileQRows128 && ktile == kTimeAttnCutileKTile64) {
        dim3 grid(static_cast<unsigned int>(ceildiv(kTimeAttnN, kTimeAttnCutileQRows128)),
                  static_cast<unsigned int>(bh));
        time_attention1301_cutile_qk_softmax_av_kernel<kTimeAttnCutileQRows128,
                                                       kTimeAttnCutileKTile64>
            <<<grid, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                          out.data_bf16(), scale);
    } else if (qrows == kTimeAttnCutileQRows32 && ktile == kTimeAttnCutileKTile64) {
        dim3 grid(static_cast<unsigned int>(ceildiv(kTimeAttnN, kTimeAttnCutileQRows32)),
                  static_cast<unsigned int>(bh));
        time_attention1301_cutile_qk_softmax_av_kernel<kTimeAttnCutileQRows32,
                                                       kTimeAttnCutileKTile64>
            <<<grid, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                          out.data_bf16(), scale);
    } else if (qrows == kTimeAttnCutileQRows16 && ktile == kTimeAttnCutileKTile64) {
        dim3 grid(static_cast<unsigned int>(ceildiv(kTimeAttnN, kTimeAttnCutileQRows16)),
                  static_cast<unsigned int>(bh));
        time_attention1301_cutile_qk_softmax_av_kernel<kTimeAttnCutileQRows16,
                                                       kTimeAttnCutileKTile64>
            <<<grid, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                          out.data_bf16(), scale);
    } else if (qrows == kTimeAttnCutileQRows64 && ktile == kTimeAttnCutileKTile64) {
        dim3 grid(static_cast<unsigned int>(ceildiv(kTimeAttnN, kTimeAttnCutileQRows64)),
                  static_cast<unsigned int>(bh));
        time_attention1301_cutile_qk_softmax_av_kernel<kTimeAttnCutileQRows64,
                                                       kTimeAttnCutileKTile64>
            <<<grid, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                          out.data_bf16(), scale);
    } else {
        throw std::runtime_error("unsupported time attention CUDA Tile shape qrows=" +
                                 std::to_string(qrows) + " ktile=" + std::to_string(ktile));
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace cudasep::mbr_tile
